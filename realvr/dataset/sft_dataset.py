"""Stage-1 training data.

An example is a LLaVA-style record with one image and one normalized box [x0, y0, x1, y1] per
`<lvr>` placeholder in its conversation:

    {"image": "flickr30k/2618322793.jpg",
     "conversations": [{"from": "human", "value": "<image>\\nWhat is the child on the swing wearing?"},
                       {"from": "gpt", "value": "<lvr>\\n<answer> Dark blue denim shorts. </answer>"}],
     "bboxes": [[0.382, 0.456, 0.718, 0.656]]}

`<image>` becomes the image's visual tokens, and each `<lvr>` becomes <|lvr_start|>, one <|lvr|> per
visual token inside its box, and <|lvr_end|>.

Examples are packed greedily into groups of at most `max_packed_tokens` tokens and
`max_instance_per_batch` examples; an example with at least `long_seq_threshold` tokens forms a pack
on its own and is cut to `max_packed_tokens`. One pack is one training step on one GPU, fed as a
right-padded batch with one example per row.

Adapted with modifications from LVR (https://github.com/VincentLeebang/lvr, Apache-2.0); the packing
follows InternVL (https://github.com/OpenGVLab/InternVL, MIT License, Copyright (c) 2023 OpenGVLab).
"""

import json
import logging
import math
import os
import re

import numpy as np
import torch
from qwen_vl_utils import fetch_image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, get_worker_info

from realvr.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    LLAVA_IMAGE_TOKEN,
    LVR_END_TOKEN,
    LVR_PLACEHOLDER,
    LVR_START_TOKEN,
    LVR_TOKEN,
    SYSTEM_MESSAGE,
    VISION_END_TOKEN,
    VISION_START_TOKEN,
)

logger = logging.getLogger(__name__)

ROLES = {"human": "user", "gpt": "assistant"}
IMAGE_PLACEHOLDER = re.compile(r"\n?" + re.escape(LLAVA_IMAGE_TOKEN) + r"\n?")


def roi_token_indices(box, grid_h, grid_w, merge_size):
    """Indices of the merged visual tokens covered by a normalized [x0, y0, x1, y1] box.

    `grid_h` x `grid_w` is the image's patch grid; merged tokens are numbered row by row on the
    (grid_h / merge_size) x (grid_w / merge_size) grid.
    """
    x0, y0, x1, y1 = box
    left = min(max(math.floor(x0 * grid_w), 0), grid_w - 1) // merge_size
    top = min(max(math.floor(y0 * grid_h), 0), grid_h - 1) // merge_size
    right = (min(max(math.ceil(x1 * grid_w), 0), grid_w) + merge_size - 1) // merge_size
    bottom = (min(max(math.ceil(y1 * grid_h), 0), grid_h) + merge_size - 1) // merge_size
    width = grid_w // merge_size
    return [row * width + col for row in range(top, bottom) for col in range(left, right)]


def _expand_lvr(text, rois):
    """Replace each <lvr> by <|lvr_start|>, one <|lvr|> per token of the next ROI, and <|lvr_end|>."""
    head, *tails = text.split(LVR_PLACEHOLDER)
    for tail in tails:
        head += LVR_START_TOKEN + LVR_TOKEN * len(next(rois)) + LVR_END_TOKEN + tail
    return head


def load_datasets(data_path, image_folder=None):
    """(name, records, image folder) of every dataset in `data_path`.

    `data_path` is a LLaVA-format JSON file, or a JSON list of
    {"ds_name": ..., "data_path": ..., "image_folder": ...} entries, one per dataset.
    """
    with open(data_path) as f:
        data = json.load(f)
    if not data:
        raise ValueError(f"{data_path} is empty")
    if "conversations" in data[0]:
        return [(data_path, data, image_folder)]
    datasets = []
    for entry in data:
        with open(entry["data_path"]) as f:
            records = json.load(f)
        datasets.append((entry.get("ds_name", entry["data_path"]), records, entry.get("image_folder", image_folder)))
    return datasets


class SFTSource:
    """One dataset of the training mixture; turns records into tokenized examples.

    An example holds `input_ids`, `labels`, `pixel_values`, `image_grid_thw` and `lvr_tokens`,
    the indices (into the image's visual tokens) of the ROI tokens in the order of the <|lvr|>
    tokens in `input_ids`.
    """

    def __init__(self, name, records, image_folder, processor, min_pixels, max_pixels, seed):
        self.name = name
        self.records = list(records)
        np.random.default_rng(seed).shuffle(self.records)
        self.image_folder = image_folder
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        self.lvr_token_id = self.tokenizer.convert_tokens_to_ids(LVR_TOKEN)

    def __len__(self):
        return len(self.records)

    def iterate(self, shard, num_shards):
        """Examples of one shard, in order; records that cannot be processed are skipped."""
        for index in range(shard, len(self.records), num_shards):
            try:
                example = self.process(self.records[index])
            except Exception as error:
                logger.warning("Skipping record %d of %s: %s", index, self.name, error)
                continue
            yield example

    def process(self, record):
        images = [record["image"]] if isinstance(record["image"], str) else record["image"]
        if len(images) != 1:
            raise ValueError(f"expected one image, got {len(images)}")
        image = fetch_image(
            {"image": self._image_path(images[0]), "min_pixels": self.min_pixels, "max_pixels": self.max_pixels}
        )
        visual = self.image_processor(images=[image], do_resize=False, return_tensors="pt")
        grid = visual["image_grid_thw"]
        _, grid_h, grid_w = grid[0].tolist()
        merge_size = self.image_processor.merge_size
        num_image_tokens = int(grid[0].prod()) // merge_size**2
        rois = [roi_token_indices(box, grid_h, grid_w, merge_size) for box in record.get("bboxes", [])]

        turns = record["conversations"]
        if len(turns) % 2:
            raise ValueError("conversation must alternate user and assistant turns")
        if sum(turn["value"].count(LVR_PLACEHOLDER) for turn in turns) != len(rois):
            raise ValueError(f"{len(rois)} boxes for a different number of {LVR_PLACEHOLDER} placeholders")
        vision = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN * num_image_tokens + VISION_END_TOKEN
        remaining = iter(rois)

        input_ids = [self._tokenize(f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n")]
        labels = [torch.full_like(input_ids[0], IGNORE_INDEX)]
        for user, assistant in zip(turns[0::2], turns[1::2]):
            question = _expand_lvr(IMAGE_PLACEHOLDER.sub(lambda _: vision, user["value"]), remaining)
            answer = _expand_lvr(assistant["value"], remaining)
            prompt = self._tokenize(
                f"{DEFAULT_IM_START_TOKEN}{ROLES.get(user['from'], user['from'])}\n{question}{DEFAULT_IM_END_TOKEN}\n"
                f"{DEFAULT_IM_START_TOKEN}{ROLES.get(assistant['from'], assistant['from'])}\n"
            )
            response = self._tokenize(f"{answer}{DEFAULT_IM_END_TOKEN}\n")
            input_ids += [prompt, response]
            labels += [torch.full_like(prompt, IGNORE_INDEX), response]

        input_ids = torch.cat(input_ids)
        lvr_tokens = torch.tensor([index for roi in rois for index in roi], dtype=torch.long)
        if (input_ids == self.image_token_id).sum() != num_image_tokens:
            raise ValueError(f"expected one {LLAVA_IMAGE_TOKEN} placeholder, in a user turn")
        if (input_ids == self.lvr_token_id).sum() != len(lvr_tokens):
            raise ValueError(f"{LVR_TOKEN} tokens do not match the ROI tokens")
        return {
            "input_ids": input_ids,
            "labels": torch.cat(labels),
            "pixel_values": visual["pixel_values"],
            "image_grid_thw": grid,
            "lvr_tokens": lvr_tokens,
        }

    def _image_path(self, path):
        if self.image_folder is not None and not path.startswith("http") and not os.path.exists(path):
            return os.path.join(self.image_folder, path)
        return path

    def _tokenize(self, text):
        return self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def _num_tokens(pack):
    return sum(len(example["input_ids"]) for example in pack)


class PackedDataset(IterableDataset):
    """Endless stream of packs (lists of examples) drawn from several sources.

    Every rank and dataloader worker reads its own shard of each source. Sources are sampled in
    proportion to their size, and a source starts over when its shard is used up.
    """

    def __init__(
        self,
        sources,
        rank,
        world_size,
        max_packed_tokens,
        long_seq_threshold,
        max_instance_per_batch,
        vision_token_ids,
        lvr_token_id,
        max_buffer_size=100,
    ):
        if not 0 < long_seq_threshold <= max_packed_tokens:
            raise ValueError("long_seq_threshold must be positive and at most max_packed_tokens")
        self.sources = list(sources)
        self.rank = rank
        self.world_size = world_size
        self.max_packed_tokens = max_packed_tokens
        self.long_seq_threshold = long_seq_threshold
        self.max_instance_per_batch = max_instance_per_batch
        self.vision_token_ids = torch.tensor(vision_token_ids)
        self.lvr_token_id = lvr_token_id
        self.max_buffer_size = max_buffer_size

    def __iter__(self):
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        shard = self.rank * num_workers + (worker.id if worker else 0)
        num_shards = self.world_size * num_workers
        rng = np.random.default_rng(seed=shard)

        sources = list(self.sources)
        weights = [len(source) for source in sources]
        streams = [source.iterate(shard, num_shards) for source in sources]
        waiting = []  # packs that can still grow, longest first
        while sources:
            k = rng.choice(len(sources), p=np.array(weights) / sum(weights))
            example = next(streams[k], None)
            if example is None:
                streams[k] = sources[k].iterate(shard, num_shards)
                example = next(streams[k], None)
                if example is None:  # nothing usable in this shard
                    del sources[k], weights[k], streams[k]
                    continue
            pack = self._take_pack(waiting, example) or []
            pack.append(example)
            ready, rest = self._split(pack)
            yield from ready
            if rest is not None:
                self._insert(waiting, rest)
            while len(waiting) > self.max_buffer_size:
                yield waiting.pop(0)
        yield from waiting

    def _take_pack(self, waiting, example):
        """Remove and return the waiting pack that `example` joins, if any."""
        length = len(example["input_ids"])
        if length >= self.long_seq_threshold:
            return None
        chosen = None
        for i, pack in enumerate(waiting):
            if len(pack) < self.max_instance_per_batch:
                if _num_tokens(pack) + length <= self.max_packed_tokens:
                    chosen = i
                    break
                if len(waiting) >= self.max_buffer_size // 2:  # the buffer is filling up: allow going over budget
                    chosen = i
        return None if chosen is None else waiting.pop(chosen)

    def _split(self, pack):
        """Split a pack into packs that are ready to train on and a pack that can still grow (or None)."""
        if len(pack) == 1:
            if len(pack[0]["input_ids"]) < self.long_seq_threshold:
                return [], pack
            example = self._truncate(pack[0])
            return ([[example]] if example is not None else []), None
        if _num_tokens(pack) < self.max_packed_tokens:
            if len(pack) < self.max_instance_per_batch:
                return [], pack
            return [pack], None
        ready = []
        while _num_tokens(pack) >= self.max_packed_tokens:
            last = [pack.pop()]
            if _num_tokens(pack) >= _num_tokens(last):
                ready.append(pack)
                pack = last
            else:
                ready.append(last)
        return ready, pack

    def _truncate(self, example):
        """Cut a long example to the token budget; None if the cut would remove any of its image."""
        cut = self.max_packed_tokens
        removed = example["input_ids"][cut:]
        if torch.isin(removed, self.vision_token_ids).any():
            return None
        lvr_tokens = example["lvr_tokens"]
        kept_lvr = len(lvr_tokens) - int((removed == self.lvr_token_id).sum())
        return {
            **example,
            "input_ids": example["input_ids"][:cut],
            "labels": example["labels"][:cut],
            "lvr_tokens": lvr_tokens[:kept_lvr],
        }

    @staticmethod
    def _insert(waiting, pack):
        """Insert `pack` before the first shorter waiting pack."""
        length = _num_tokens(pack)
        index = next((i for i, other in enumerate(waiting) if _num_tokens(other) < length), len(waiting))
        waiting.insert(index, pack)


class PackedCollator:
    """Turns packs into one right-padded batch with an example per row."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, packs):
        examples = [example for pack in packs for example in pack]
        input_ids = pad_sequence([e["input_ids"] for e in examples], batch_first=True, padding_value=self.pad_token_id)
        lengths = torch.tensor([len(e["input_ids"]) for e in examples])
        return {
            "input_ids": input_ids,
            "attention_mask": (torch.arange(input_ids.shape[1]) < lengths[:, None]).long(),
            "labels": pad_sequence([e["labels"] for e in examples], batch_first=True, padding_value=IGNORE_INDEX),
            "pixel_values": torch.cat([e["pixel_values"] for e in examples]),
            "image_grid_thw": torch.cat([e["image_grid_thw"] for e in examples]),
            "lvr_tokens": [e["lvr_tokens"] for e in examples],
        }


def make_sft_data_module(processor, data_args, training_args):
    """Packed Stage-1 training set and collator, as keyword arguments for the trainer.

    The LVR tokens must already be in the tokenizer.
    """
    tokenizer = processor.tokenizer
    if LVR_TOKEN not in tokenizer.get_vocab():
        raise ValueError("add the LVR special tokens to the tokenizer first")
    seed = training_args.data_seed if training_args.data_seed is not None else training_args.seed
    sources = [
        SFTSource(name, records, folder, processor, data_args.image_min_pixels, data_args.image_max_pixels, seed)
        for name, records, folder in load_datasets(data_args.data_path, data_args.image_folder)
    ]
    dataset = PackedDataset(
        sources,
        rank=training_args.process_index,
        world_size=training_args.world_size,
        max_packed_tokens=training_args.max_packed_tokens,
        long_seq_threshold=training_args.long_seq_threshold,
        max_instance_per_batch=training_args.max_instance_per_batch,
        vision_token_ids=tokenizer.convert_tokens_to_ids([VISION_START_TOKEN, DEFAULT_IMAGE_TOKEN, VISION_END_TOKEN]),
        lvr_token_id=tokenizer.convert_tokens_to_ids(LVR_TOKEN),
    )
    return {"train_dataset": dataset, "data_collator": PackedCollator(tokenizer.pad_token_id)}

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Qwen2_5_VLProcessor, Qwen2TokenizerFast, Qwen2VLImageProcessor
from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

from realvr.constants import IGNORE_INDEX
from realvr.dataset import PackedCollator, PackedDataset, SFTSource, make_sft_data_module
from realvr.dataset.sft_dataset import roi_token_indices
from realvr.model.qwen_lvr import QwenWithLVRForSFT
from realvr.params import DataArguments, SFTArguments
from realvr.train.train_sft import add_lvr_tokens
from realvr.trainer import LVRSFTTrainer
from tests.tiny_model import (
    IMAGE_TOKEN_ID,
    LVR_END_ID,
    LVR_ID,
    LVR_START_ID,
    VISION_END_ID,
    VISION_START_ID,
    tiny_qwen2_5_vl,
)

MIN_PIXELS, MAX_PIXELS = 4 * 28 * 28, 16 * 28 * 28

# Records of the toy dataset, with the ROI indices worked out by hand from the boxes.
# Image sizes (W x H): a 56x56 -> 2x2 merged tokens, b 112x56 -> 2 rows x 4 cols, c 56x112 -> 4 rows x 2 cols.
RECORDS = [
    ({"image": "a.png",
      "conversations": [{"from": "human", "value": "<image>\nw5 w6"},
                        {"from": "gpt", "value": "<lvr>\n<answer> w7 </answer>"}],
      "bboxes": [[0.5, 0.5, 1.0, 1.0]]}, [3]),
    ({"image": "b.png",
      "conversations": [{"from": "human", "value": "<image>\nw8"},
                        {"from": "gpt", "value": "<lvr> w9 <lvr>\n<answer> w10 </answer>"}],
      "bboxes": [[0.25, 0.0, 0.75, 0.5], [0.0, 0.5, 0.25, 1.0]]}, [1, 2, 4]),
    ({"image": "c.png",
      "conversations": [{"from": "human", "value": "<image>\nw11 w12 w13 w14"},
                        {"from": "gpt", "value": "<lvr>\n<answer> w15 </answer>"}],
      "bboxes": [[0.0, 0.3, 1.0, 0.6]]}, [2, 3, 4, 5]),
    ({"image": "a.png",
      "conversations": [{"from": "human", "value": "<image>\nw16"},
                        {"from": "gpt", "value": "<answer> w17 </answer>"}]}, []),
]
IMAGE_SIZES = {"a.png": (56, 56), "b.png": (112, 56), "c.png": (56, 112)}


def tiny_processor():
    """Qwen2.5-VL processor with a word-level vocabulary whose special ids match tests/tiny_model.py."""
    special = {"<|endoftext|>": 0, "<unk>": 1, "<|im_start|>": 2, "<|im_end|>": 3, "<|video_pad|>": 4,
               "<|image_pad|>": IMAGE_TOKEN_ID, "<|vision_start|>": VISION_START_ID, "<|vision_end|>": VISION_END_ID}
    vocab = {**special, **{f"w{i}": i for i in range(5, IMAGE_TOKEN_ID)}}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.add_special_tokens([AddedToken(token, special=True, normalized=False) for token in special])
    tokenizer = Qwen2TokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<|endoftext|>",
                                   eos_token="<|im_end|>")
    image_processor = Qwen2VLImageProcessor(min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    return Qwen2_5_VLProcessor(image_processor=image_processor, tokenizer=tokenizer,
                               video_processor=Qwen2VLVideoProcessor())


def write_dataset(folder):
    rng = np.random.default_rng(0)
    for name, (width, height) in IMAGE_SIZES.items():
        Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8)).save(folder / name)
    path = folder / "stage1.json"
    path.write_text(json.dumps([record for record, _ in RECORDS]))
    return path


def _setup(tmp_path, **training_kwargs):
    processor = tiny_processor()
    model = tiny_qwen2_5_vl(model_class=QwenWithLVRForSFT)
    model.visual.requires_grad_(False)
    add_lvr_tokens(processor, model)
    data_args = DataArguments(data_path=str(write_dataset(tmp_path)), image_folder=str(tmp_path),
                              image_min_pixels=MIN_PIXELS, image_max_pixels=MAX_PIXELS)
    training_args = SFTArguments(output_dir=str(tmp_path / "out"), use_cpu=True, report_to="none",
                                 dataloader_pin_memory=False, **training_kwargs)
    return processor, model, training_args, make_sft_data_module(processor, data_args, training_args)


def test_roi_token_indices():
    assert roi_token_indices([0.5, 0.5, 1.0, 1.0], 4, 4, 2) == [3]
    assert roi_token_indices([0.25, 0.0, 0.75, 0.5], 4, 8, 2) == [1, 2]
    assert roi_token_indices([0.0, 0.5, 0.25, 1.0], 4, 8, 2) == [4]
    assert roi_token_indices([0.0, 0.3, 1.0, 0.6], 8, 4, 2) == [2, 3, 4, 5]
    assert roi_token_indices([-0.2, -0.2, 1.3, 1.3], 4, 4, 2) == [0, 1, 2, 3]


def test_add_lvr_tokens_writes_ids_to_config():
    processor = tiny_processor()
    model = tiny_qwen2_5_vl(model_class=QwenWithLVRForSFT)
    model.config.lvr_start_id = model.config.lvr_id = model.config.lvr_end_id = None
    add_lvr_tokens(processor, model)
    config = model.config
    assert (config.lvr_start_id, config.lvr_id, config.lvr_end_id) == (LVR_START_ID, LVR_ID, LVR_END_ID)
    assert processor.tokenizer.convert_tokens_to_ids("<|lvr|>") == LVR_ID


def test_example_layout(tmp_path):
    processor = tiny_processor()
    add_lvr_tokens(processor, tiny_qwen2_5_vl(model_class=QwenWithLVRForSFT))
    write_dataset(tmp_path)
    source = SFTSource("toy", [], str(tmp_path), processor, MIN_PIXELS, MAX_PIXELS, seed=0)
    example = source.process(RECORDS[1][0])
    ids, labels = example["input_ids"], example["labels"]

    assert example["image_grid_thw"].tolist() == [[1, 4, 8]]
    assert example["pixel_values"].shape == (32, 3 * 2 * 14 * 14)
    assert example["lvr_tokens"].tolist() == [1, 2, 4]
    vision = (ids == VISION_START_ID).nonzero().item()
    assert ids[vision + 1:vision + 9].eq(IMAGE_TOKEN_ID).all() and ids[vision + 9] == VISION_END_ID
    # the response: <lvr_start> <lvr> <lvr> <lvr_end> w9 <lvr_start> <lvr> <lvr_end> ...
    start = (ids == LVR_START_ID).nonzero().flatten()
    assert ids[start[0]:start[0] + 4].tolist() == [LVR_START_ID, LVR_ID, LVR_ID, LVR_END_ID]
    assert ids[start[1]:start[1] + 3].tolist() == [LVR_START_ID, LVR_ID, LVR_END_ID]
    # only the response is supervised: it starts right after "<|im_start|>assistant\n"
    supervised = (labels != IGNORE_INDEX).nonzero().flatten()
    assert supervised[0] == start[0] and torch.equal(labels[supervised], ids[supervised])
    assert (labels[:start[0]] == IGNORE_INDEX).all()


def test_unusable_records_are_skipped(tmp_path, caplog):
    processor = tiny_processor()
    add_lvr_tokens(processor, tiny_qwen2_5_vl(model_class=QwenWithLVRForSFT))
    write_dataset(tmp_path)
    good = RECORDS[0][0]
    missing_image = {**good, "image": "missing.png"}
    extra_box = {**good, "bboxes": good["bboxes"] * 2}
    two_images = {**good, "image": ["a.png", "b.png"]}
    source = SFTSource("toy", [missing_image, good, extra_box, two_images], str(tmp_path), processor,
                       MIN_PIXELS, MAX_PIXELS, seed=0)
    examples = list(source.iterate(0, 1))
    assert len(examples) == 1 and examples[0]["lvr_tokens"].tolist() == [3]
    assert caplog.text.count("Skipping record") == 3


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

def _example(length, image_at=1, image_tokens=4, lvr_at=None, num_lvr=0):
    ids = torch.full((length,), 7)
    ids[image_at] = VISION_START_ID
    ids[image_at + 1:image_at + 1 + image_tokens] = IMAGE_TOKEN_ID
    ids[image_at + 1 + image_tokens] = VISION_END_ID
    if num_lvr:
        ids[lvr_at - 1] = LVR_START_ID
        ids[lvr_at:lvr_at + num_lvr] = LVR_ID
        if lvr_at + num_lvr < length:
            ids[lvr_at + num_lvr] = LVR_END_ID
    return {"input_ids": ids, "labels": ids.clone(), "pixel_values": torch.zeros(4 * image_tokens, 1176),
            "image_grid_thw": torch.tensor([[1, 2, 2 * image_tokens]]), "lvr_tokens": torch.arange(10, 10 + num_lvr)}


class ListSource:
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def iterate(self, shard, num_shards):
        yield from self.examples[shard::num_shards]


def _packer(examples, rank=0, world_size=1, **kwargs):
    kwargs = {"max_packed_tokens": 100, "long_seq_threshold": 60, "max_instance_per_batch": 3, **kwargs}
    return PackedDataset([ListSource(examples)], rank=rank, world_size=world_size,
                         vision_token_ids=[VISION_START_ID, IMAGE_TOKEN_ID, VISION_END_ID], lvr_token_id=LVR_ID,
                         **kwargs)


def _lengths(pack):
    return [len(example["input_ids"]) for example in pack]


def test_packs_fill_up_to_the_instance_limit():
    pack = next(iter(_packer([_example(30), _example(31), _example(32)])))
    assert _lengths(pack) == [30, 31, 32]


def test_example_joins_the_first_waiting_pack_within_budget():
    packer = _packer([])
    waiting = [[_example(50), _example(45)], [_example(20)]]
    pack = packer._take_pack(waiting, _example(50))
    assert _lengths(pack) == [20] and len(waiting) == 1
    assert packer._take_pack(waiting, _example(60)) is None  # long examples are never packed


def test_full_pack_is_split():
    first, second = _example(50), _example(50)
    ready, rest = _packer([])._split([first, second])
    assert len(ready) == 1 and len(ready[0]) == 1 and ready[0][0] is first
    assert len(rest) == 1 and rest[0] is second


def test_long_example_is_cut_with_its_lvr_tokens():
    long = _example(150, lvr_at=95, num_lvr=10)
    ready, rest = _packer([])._split([long])
    (cut,), = ready
    assert rest is None and len(cut["input_ids"]) == 100 and len(cut["labels"]) == 100
    assert cut["lvr_tokens"].tolist() == [10, 11, 12, 13, 14]
    assert (cut["input_ids"] == LVR_ID).sum() == 5


@pytest.mark.parametrize("image_at", [97, 120])  # image across the cut, image after the cut
def test_long_example_is_dropped_when_the_cut_removes_image_tokens(image_at):
    ready, rest = _packer([])._split([_example(150, image_at=image_at)])
    assert ready == [] and rest is None


def test_ranks_read_disjoint_shards():
    examples = [_example(10 + i) for i in range(6)]
    first = [next(iter(_packer(examples, rank=r, world_size=2, max_instance_per_batch=2))) for r in (0, 1)]
    assert _lengths(first[0]) == [10, 12] and _lengths(first[1]) == [11, 13]


def test_collator_pads_one_example_per_row():
    batch = PackedCollator(pad_token_id=0)([[_example(12, lvr_at=8, num_lvr=2), _example(9)]])
    assert batch["input_ids"].shape == (2, 12)
    assert batch["attention_mask"].sum(1).tolist() == [12, 9]
    assert (batch["input_ids"][1, 9:] == 0).all() and (batch["labels"][1, 9:] == IGNORE_INDEX).all()
    assert [t.tolist() for t in batch["lvr_tokens"]] == [[10, 11], []]
    assert batch["pixel_values"].shape == (32, 1176) and batch["image_grid_thw"].shape == (2, 3)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_lvr_inputs_are_the_roi_visual_tokens(tmp_path):
    _, model, _, module = _setup(tmp_path, max_instance_per_batch=4)
    batch = module["data_collator"]([next(iter(module["train_dataset"]))])
    assert batch["input_ids"].shape[0] == 4

    # which record each row holds: (image grid, number of <|lvr|> tokens) -> hand-computed ROI indices
    expected = {}
    for record, rois in RECORDS:
        width, height = IMAGE_SIZES[record["image"]]
        expected[(1, height // 14, width // 14, len(rois))] = rois
    rows = []
    for row in range(4):
        key = (*batch["image_grid_thw"][row].tolist(), int((batch["input_ids"][row] == LVR_ID).sum()))
        rows.append(expected[key])
        assert batch["lvr_tokens"][row].tolist() == expected[key]
    assert sorted(map(tuple, rows)) == sorted(tuple(rois) for _, rois in RECORDS)

    captured = {}
    language_model = model.model.language_model
    hooks = [
        language_model.register_forward_pre_hook(
            lambda m, args, kwargs: captured.update(embeds=kwargs["inputs_embeds"].detach()), with_kwargs=True),
        language_model.register_forward_hook(
            lambda m, args, kwargs, out: captured.update(hidden=out.last_hidden_state.detach()), with_kwargs=True),
    ]
    with torch.no_grad():
        out = model(**batch)
        features = model.model.get_image_features(batch["pixel_values"], batch["image_grid_thw"])
    for hook in hooks:
        hook.remove()

    targets, predicted = [], []
    for row, rois in enumerate(rows):
        image = (batch["input_ids"][row] == IMAGE_TOKEN_ID).nonzero().flatten()
        assert torch.allclose(captured["embeds"][row, image], features[row])
        lvr = (batch["input_ids"][row] == LVR_ID).nonzero().flatten()
        assert torch.allclose(captured["embeds"][row, lvr], features[row][rois])
        targets.append(features[row][rois])
        predicted.append(captured["hidden"][row, lvr - 1])
    assert torch.allclose(out.loss_lvr, F.mse_loss(torch.cat(predicted), torch.cat(targets)))

    labels = batch["labels"][:, 1:].masked_fill(batch["labels"][:, 1:] == LVR_ID, IGNORE_INDEX)
    loss_ce = F.cross_entropy(out.logits[:, :-1].flatten(0, 1), labels.flatten(), ignore_index=IGNORE_INDEX)
    assert torch.allclose(out.loss_ce, loss_ce) and torch.isfinite(out.loss_ce) and torch.isfinite(out.loss_lvr)


def test_trainer_updates_the_language_model_only(tmp_path):
    processor, model, args, module = _setup(tmp_path, max_instance_per_batch=2, max_steps=2, learning_rate=1e-3,
                                            logging_steps=1, save_strategy="no", gradient_checkpointing=True)
    vision = {name: p.clone() for name, p in model.visual.named_parameters()}
    language = {name: p.clone() for name, p in model.model.language_model.named_parameters()}
    trainer = LVRSFTTrainer(model=model, args=args, processing_class=processor, **module)
    trainer.train()

    assert trainer.model_accepts_loss_kwargs is False
    logs = [entry for entry in trainer.state.log_history if "loss" in entry]
    assert len(logs) == 2
    for entry in logs:
        assert entry["loss"] == pytest.approx(entry["loss_ce"] + 0.1 * entry["loss_lvr"], abs=1e-3)
    assert all(torch.equal(p, vision[name]) for name, p in model.visual.named_parameters())
    assert all(not torch.equal(p, language[name]) for name, p in model.model.language_model.named_parameters())
    assert not any(getattr(m, "gradient_checkpointing", False) for m in model.visual.modules())
    assert all(layer.gradient_checkpointing for layer in model.model.language_model.layers)

    trainer.save_model(str(tmp_path / "final"))
    config = json.loads((tmp_path / "final" / "config.json").read_text())
    assert (config["lvr_start_id"], config["lvr_id"], config["lvr_end_id"]) == (LVR_START_ID, LVR_ID, LVR_END_ID)
    assert "<|lvr|>" in (tmp_path / "final" / "tokenizer.json").read_text()

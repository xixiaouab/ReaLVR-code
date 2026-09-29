"""End-to-end Stage-2 steps on CPU with a tiny random model and the real Qwen2.5-VL processor.

Set REALVR_TEST_PROCESSOR to a local directory holding the Qwen2.5-VL-7B-Instruct processor files
(tokenizer + preprocessor configs); the test is skipped otherwise.
"""

import json
import math
import os

import pytest
import torch
from PIL import Image

from realvr.constants import LVR_SPECIAL_TOKENS
from realvr.dataset.grpo_dataset import GRPODataset
from realvr.model.qwen_lvr import QwenWithLVR
from realvr.params import ReaLVRConfig
from realvr.train.rewards import REWARD_FUNCS
from realvr.trainer.realvr_trainer import ReaLVRTrainer
from tests.tiny_model import tiny_config

PROCESSOR_DIR = os.environ.get("REALVR_TEST_PROCESSOR")
pytestmark = pytest.mark.skipif(not PROCESSOR_DIR, reason="REALVR_TEST_PROCESSOR is not set")


def _processor():
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(PROCESSOR_DIR)
    processor.tokenizer.add_special_tokens({"additional_special_tokens": LVR_SPECIAL_TOKENS})
    return processor


def _model(tokenizer):
    config = tiny_config()
    ids = tokenizer.convert_tokens_to_ids
    config.vocab_size = config.text_config.vocab_size = len(tokenizer)
    config.image_token_id = ids("<|image_pad|>")
    config.video_token_id = ids("<|video_pad|>")
    config.vision_start_token_id = ids("<|vision_start|>")
    config.vision_end_token_id = ids("<|vision_end|>")
    config.lvr_start_id, config.lvr_id, config.lvr_end_id = (ids(t) for t in LVR_SPECIAL_TOKENS)
    torch.manual_seed(0)
    model = QwenWithLVR(config).float()
    model.visual.requires_grad_(False)
    return model


def _dataset(tmp_path):
    records = []
    for i, answer in enumerate(["A", "B"]):
        path = tmp_path / f"img{i}.png"
        Image.new("RGB", (84, 56), color=(40 * i, 120, 200)).save(path)
        records.append({
            "image": path.name,
            "conversations": [
                {"from": "human", "value": "<image>\nWhich option? A. first B. second"},
                {"from": "gpt", "value": f"<answer>{answer}</answer>"},
            ],
            "bboxes": [[0, 0, 42, 28]],
        })
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps(records))
    return GRPODataset(str(data_path), str(tmp_path), image_min_pixels=4 * 28 * 28, image_max_pixels=16 * 28 * 28)


def _trainer(tmp_path, **overrides):
    processor = _processor()
    model = _model(processor.tokenizer)
    settings = dict(
        output_dir=str(tmp_path / "out"), per_device_train_batch_size=4, num_generations=2,
        max_completion_length=10, lvr_steps=2, max_steps=2, logging_steps=1, report_to=[],
        save_strategy="no", use_cpu=True, bf16=False, seed=0, learning_rate=1e-3, warmup_ratio=0.0,
        dataloader_num_workers=0,
    )
    settings.update(overrides)
    trainer = ReaLVRTrainer(model=model, reward_funcs=REWARD_FUNCS, args=ReaLVRConfig(**settings),
                            train_dataset=_dataset(tmp_path), processing_class=processor)
    trainer.generation_config.force_lvr_start = True
    return trainer


def test_two_training_steps_update_the_language_model_only(tmp_path):
    trainer = _trainer(tmp_path)
    model = trainer.model
    llm_before = model.model.language_model.layers[0].mlp.up_proj.weight.detach().clone()
    vision_before = model.visual.blocks[0].mlp.up_proj.weight.detach().clone()
    trainer.train()
    logs = [entry for entry in trainer.state.log_history if "evidence/loss" in entry]
    assert len(logs) == 2
    for entry in logs:
        assert math.isfinite(entry["loss"]) and math.isfinite(entry["evidence/loss"])
        assert entry["evidence/loss"] > 0
        assert entry["evidence/negatives"] >= 1
    assert not torch.equal(model.model.language_model.layers[0].mlp.up_proj.weight, llm_before)
    assert torch.equal(model.visual.blocks[0].mlp.up_proj.weight, vision_before)


def test_rollout_latents_are_replayed_at_their_positions(tmp_path):
    trainer = _trainer(tmp_path)
    batch = trainer._generate_and_score([trainer.train_dataset[0]] * 2 + [trainer.train_dataset[1]] * 2)
    config = trainer.model.config
    prompt_len = batch["prompt_ids"].shape[1]
    for b in range(4):
        positions = batch["latent_positions"][b]
        assert (positions >= 0).all()
        start = int(positions[0]) - 1 - prompt_len
        assert int(batch["completion_ids"][b, start]) == config.lvr_start_id
        assert (batch["completion_ids"][b, positions - prompt_len] == config.lvr_id).all()
        assert not batch["loss_mask"][b, positions - prompt_len].any()


def test_answer_readout_contrasts_gold_and_wrong_answers(tmp_path):
    trainer = _trainer(tmp_path)
    batch = trainer._generate_and_score([trainer.train_dataset[0]] * 2 + [trainer.train_dataset[1]] * 2)
    batch["wrong_answers"] = [["B"], ["B", "C"], [], ["A"]]
    latents = trainer._regenerate_latents(trainer.model, batch).detach()
    r_pos, r_neg, has_wrong = trainer._answer_readout(trainer.model, trainer.model, batch, latents)
    assert r_pos.shape == (4, 2) and r_neg.shape == (4, 2)
    assert has_wrong.tolist() == [True, True, False, True]
    assert torch.equal(r_pos[2], r_neg[2])  # no wrong answer -> no selective credit
    assert (r_pos >= 0).all() and (r_pos.sum(dim=1) <= 1).all()
    assert not torch.allclose(r_pos[0], r_neg[0])

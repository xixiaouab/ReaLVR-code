"""CPU tests for eval/evaluate.py and eval/merge_results.py."""

import base64
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoProcessor,
    GenerationConfig,
    Qwen2_5_VLProcessor,
    Qwen2TokenizerFast,
    Qwen2VLImageProcessorFast,
)
from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

from eval import evaluate, merge_results
from eval.evaluate import (
    ANSWER_INSTRUCTION,
    Sample,
    SelectionWeights,
    extract_answer,
    is_correct,
    mc_prompt,
    mean_logprob,
    select_sample,
)
from realvr.model.qwen_lvr import QwenWithLVR
from tests.tiny_model import (
    IMAGE_TOKEN_ID,
    LVR_END_ID,
    LVR_ID,
    LVR_START_ID,
    VISION_END_ID,
    VISION_START_ID,
    VOCAB,
    tiny_qwen2_5_vl,
)

# Answer extraction and correctness

MMVP_PROMPT = mc_prompt("Are the butterfly's wings closer to being open or closed?", "A. Open B. Closed")
BLINK_PROMPT = (
    "How many blue floats are there?\nSelect from the following choices.\n(A) 0\n(B) 3\n(C) 2\n(D) 1\n"
    + ANSWER_INSTRUCTION
)
HRBENCH_PROMPT = mc_prompt("What is the color of the umbrella?", "A. red\nB. blue\nC. green\nD. yellow")
MME_PROMPT = mc_prompt(
    "Where is the tennis court located in this picture?",
    "A. In the middle right area of this picture\nB. In the top left corner of this picture\n"
    "C. In the top right area of this picture\nD. In the bottom right corner of this picture\n"
    "E. This image doesn't feature the position.",
)


@pytest.mark.parametrize(
    "output, prompt, expected",
    [
        # MMVP: both options on one line
        ("<answer>A</answer><|im_end|>", MMVP_PROMPT, "A"),
        ("<answer>(b)</answer>", MMVP_PROMPT, "B"),
        ("<answer>Closed</answer>", MMVP_PROMPT, "B"),
        ("<answer>A. Open</answer>", MMVP_PROMPT, "A"),
        ("The wings are closed, so the answer is B.", MMVP_PROMPT, "B"),
        # BLINK: the dataset's own prompt with "(A) ..." options
        ("<|lvr_start|><|lvr|><|lvr_end|><answer>(C)</answer>", BLINK_PROMPT, "C"),
        ("<answer>(C) 2</answer>", BLINK_PROMPT, "C"),
        ("<answer>2</answer>", BLINK_PROMPT, "2"),  # a single character is not matched to option texts
        # HR-Bench
        ("<answer>blue</answer>", HRBENCH_PROMPT, "B"),
        ("<answer>The umbrella is green.</answer>", HRBENCH_PROMPT, "C"),
        ("<answer>blue or green</answer>", HRBENCH_PROMPT, "B"),  # ambiguous phrase: its first letter
        ("Option: d", HRBENCH_PROMPT, "D"),
        ("I cannot tell.", HRBENCH_PROMPT, ""),
        # MME-RealWorld: five options
        ("<answer>E</answer>", MME_PROMPT, "E"),
        ("<answer>(E) This image doesn't feature the position.</answer>", MME_PROMPT, "E"),
        ("<answer>In the top left corner of this picture</answer>", MME_PROMPT, "B"),
    ],
)
def test_extract_answer(output, prompt, expected):
    assert extract_answer(output, prompt) == expected


def test_is_correct():
    assert is_correct("B", "B")
    assert is_correct("b", "(b)")
    assert not is_correct("", "A")
    assert not is_correct("2", "C")


# Selection among samples


def _sample(letter, logprob=None, latent=False):
    span = "<|lvr_start|><|lvr|><|lvr_end|>" if latent else ""
    return Sample(output=f"{span}<answer>{letter}</answer>", prediction=letter, mean_logprob=logprob)


def test_majority_vote():
    samples = [_sample("B"), _sample("A"), _sample("A")]
    assert select_sample(samples, "majority") == 1
    assert select_sample(samples, "hybrid") == 1


def test_hybrid_breaks_ties_with_logprob():
    samples = [_sample("A", logprob=-2.0), _sample("B", logprob=-0.1)]
    assert select_sample(samples, "majority") == 0  # tie: the earlier sample
    assert select_sample(samples, "hybrid") == 1


def test_vote_weight_against_logprob():
    samples = [_sample("A", -1.0), _sample("A", -1.0), _sample("B", -0.01)]
    assert select_sample(samples, "hybrid") == 0
    assert select_sample(samples, "hybrid", SelectionWeights(vote=0.0)) == 2


def test_format_and_latent_bonuses():
    untagged, tagged = Sample("A", "A"), _sample("B")
    assert select_sample([untagged, tagged], "majority") == 1
    assert select_sample([untagged, tagged], "majority", SelectionWeights(format=0.0)) == 0
    plain, latent = _sample("A"), _sample("B", latent=True)
    assert select_sample([plain, latent], "majority") == 1
    assert select_sample([plain, latent], "majority", SelectionWeights(latent=0.0)) == 0
    assert select_sample([Sample("I am not sure.", ""), _sample("C")], "majority") == 1


def test_mean_logprob_matches_transition_scores():
    model = tiny_qwen2_5_vl(model_class=QwenWithLVR)
    torch.manual_seed(0)
    scores = tuple(torch.randn(1, VOCAB) for _ in range(5))
    scores[2][0, :100] = float("-inf")  # e.g. removed by top-k; the token of that step is skipped
    tokens = torch.tensor([3, 7, 5, 120, 11])
    sequences = torch.cat([torch.tensor([[1, 2]]), tokens[None]], dim=1)
    reference = model.compute_transition_scores(sequences, scores, normalize_logits=True)[0]
    assert mean_logprob(scores, tokens) == pytest.approx(reference[torch.isfinite(reference)].mean().item())


# Benchmark loaders (on miniature files in the documented layout)


def _jpeg(color):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_load_mmvp(tmp_path):
    (tmp_path / "MMVP" / "MMVP Images").mkdir(parents=True)
    (tmp_path / "MMVP" / "Questions.csv").write_text(
        "Index,Question,Options,Correct Answer\n1,Is the door open or closed?,(a) Open (b) Closed,(b)\n"
    )
    (row,) = evaluate.load_benchmark("mmvp", tmp_path)
    assert row == {
        "id": "1",
        "images": [str(tmp_path / "MMVP" / "MMVP Images" / "1.jpg")],
        "prompt": mc_prompt("Is the door open or closed?", "A. Open B. Closed"),
        "label": "B",
    }


def test_load_blink(tmp_path):
    image = {"bytes": _jpeg("red"), "path": "1.jpg"}
    row = {"idx": "val_1", "prompt": "Which one?\n(A) left\n(B) right", "answer": "(B)", "image_1": image,
           "image_2": image, "image_3": None, "image_4": None}
    for subtask in evaluate.BLINK_SUBTASKS:
        (tmp_path / "BLINK" / subtask).mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([row]), tmp_path / "BLINK" / subtask / "val-00000-of-00001.parquet")
    rows = list(evaluate.load_benchmark("blink", tmp_path))
    assert len(rows) == len(evaluate.BLINK_SUBTASKS)
    assert rows[0] == {
        "id": "val_1",
        "images": ["data:image;base64," + base64.b64encode(image["bytes"]).decode()] * 2,
        "prompt": f"Which one?\n(A) left\n(B) right\n{ANSWER_INSTRUCTION}",
        "label": "B",
    }


def test_load_hrbench(tmp_path):
    (tmp_path / "HR-Bench").mkdir()
    payload = base64.b64encode(_jpeg("blue")).decode()
    row = {"index": 7, "question": "What color?", "answer": "C", "A": "red", "B": "green", "C": "blue", "D": "white",
           "category": "single", "cycle_category": "x", "image": payload}
    pq.write_table(pa.Table.from_pylist([row]), tmp_path / "HR-Bench" / "hr_bench_8k.parquet")
    (question,) = evaluate.load_benchmark("hrbench8k", tmp_path)
    assert question == {
        "id": "7",
        "images": ["data:image;base64," + payload],
        "prompt": mc_prompt("What color?", "A. red\nB. green\nC. blue\nD. white"),
        "label": "C",
    }


def test_load_mme_realworld_lite(tmp_path, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", Image.MAX_IMAGE_PIXELS)  # restored after the test
    root = tmp_path / "MME-RealWorld-Lite" / "data"
    root.mkdir(parents=True)
    entry = {"Question_id": "perception/x/1", "Image": "a.png", "Text": "Where is the car?",
             "Answer choices": ["(A) Left", "(B) Right", "(E) No car"], "Ground truth": "E"}
    (root / "MME-RealWorld-Lite.json").write_text(json.dumps([entry]))
    (question,) = evaluate.load_benchmark("mme_realworld_lite", tmp_path)
    assert question == {
        "id": "perception/x/1",
        "images": [str(root / "imgs" / "a.png")],
        "prompt": mc_prompt("Where is the car?", "A. Left\nB. Right\nE. No car"),
        "label": "E",
    }


# End to end on a tiny random checkpoint

SPECIAL_TOKENS = {
    0: "<|endoftext|>",
    1: "<|im_start|>",
    2: "<|im_end|>",
    IMAGE_TOKEN_ID: "<|image_pad|>",
    VISION_START_ID: "<|vision_start|>",
    VISION_END_ID: "<|vision_end|>",
    LVR_START_ID: "<|lvr_start|>",
    LVR_ID: "<|lvr|>",
    LVR_END_ID: "<|lvr_end|>",
    156: "<|video_pad|>",
}
WORDS = {3: "<answer>", 4: "</answer>", 5: "A", 6: "B", 7: "C", 8: "D"}
CHAT_TEMPLATE = (
    "{% for message in messages %}{% if loop.first and message['role'] != 'system' %}"
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}"
    "<|im_start|>{{ message['role'] }}\n{% for content in message['content'] %}"
    "{% if content['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def tiny_processor(pixel_limits_in_config=True):
    """A Qwen2.5-VL processor with a word-level vocabulary matching the token ids of tests/tiny_model.py."""
    vocab = {SPECIAL_TOKENS.get(i) or WORDS.get(i) or f"w{i}": i for i in range(VOCAB)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<|endoftext|>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.add_special_tokens(list(SPECIAL_TOKENS.values()))
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=backend, eos_token="<|im_end|>", pad_token="<|endoftext|>", unk_token="<|endoftext|>"
    )
    limits = {"min_pixels": 28 * 28, "max_pixels": 56 * 56} if pixel_limits_in_config else {}
    image_processor = Qwen2VLImageProcessorFast(size={"shortest_edge": 28 * 28, "longest_edge": 56 * 56}, **limits)
    return Qwen2_5_VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=Qwen2VLVideoProcessor(),
        chat_template=CHAT_TEMPLATE,
    )


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("checkpoint")
    tiny_processor().save_pretrained(path)
    model = tiny_qwen2_5_vl(model_class=QwenWithLVR)
    # Like the released checkpoints: sampling defaults and a repetition penalty.
    model.generation_config = GenerationConfig(
        bos_token_id=0, eos_token_id=[2, 0], pad_token_id=0, do_sample=True, temperature=1e-6, repetition_penalty=1.05
    )
    model.save_pretrained(path)
    return path


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "image.jpg"
    Image.new("RGB", (84, 56), (200, 40, 40)).save(path)
    return path


def test_max_pixels_reaches_the_processor(tiny_checkpoint, image_path):
    for processor in (AutoProcessor.from_pretrained(tiny_checkpoint), tiny_processor(pixel_limits_in_config=False)):
        default = evaluate.build_inputs(processor, [str(image_path)], "What is shown?")
        larger = evaluate.build_inputs(processor, [str(image_path)], "What is shown?", max_pixels=84 * 56)
        assert default["image_grid_thw"].tolist() == [[1, 2, 4]]
        assert larger["image_grid_thw"].tolist() == [[1, 4, 6]]


def _run(tiny_checkpoint, tmp_path, monkeypatch, question, *extra_args):
    monkeypatch.setattr(evaluate, "load_benchmark", lambda name, data_dir: iter([question]))
    output_dir = tmp_path / "out"
    argv = [
        "--checkpoint", str(tiny_checkpoint), "--benchmark", "mmvp", "--data-dir", str(tmp_path),
        "--output-dir", str(output_dir), "--device", "cpu", "--lvr-steps", "2", "--max-new-tokens", "8",
        *extra_args,
    ]
    summary = evaluate.main(argv)
    records = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text().splitlines()]
    assert json.loads((output_dir / "summary.json").read_text()) == summary
    return records, summary


@pytest.mark.parametrize(
    "extra_args", [[], ["--num-samples", "2", "--selection", "hybrid", "--max-pixels", str(84 * 56)]]
)
def test_evaluate_end_to_end(tiny_checkpoint, tmp_path, monkeypatch, image_path, extra_args):
    question = {
        "id": "q1",
        "images": [str(image_path), "data:image;base64," + base64.b64encode(image_path.read_bytes()).decode()],
        "prompt": mc_prompt("What color is the image?", "A. red\nB. blue"),
        "label": "A",
    }
    (record,), summary = _run(tiny_checkpoint, tmp_path, monkeypatch, question, "--force-lvr-start", *extra_args)
    assert "error" not in record
    assert record["id"] == "q1" and record["benchmark"] == "mmvp" and record["label"] == "A"
    assert isinstance(record["prediction"], str) and record["correct"] == (record["prediction"] == "A")
    assert record["output"].startswith("<|lvr_start|>")
    assert summary["num_questions"] == 1 and summary["errors"] == 0 and summary["correct"] == int(record["correct"])
    decoding = summary["decoding"]
    assert decoding["lvr_steps"] == 2 and decoding["force_lvr_start"] and decoding["repetition_penalty"] == 1.05
    if extra_args:
        assert len(record["samples"]) == 2
        assert all(isinstance(sample["mean_logprob"], float) for sample in record["samples"])
        assert decoding["max_pixels"] == 84 * 56 and decoding["selection"] == "hybrid"
    else:
        assert "samples" not in record and decoding["max_pixels"] == 56 * 56


def test_failed_question_counts_as_wrong(tiny_checkpoint, tmp_path, monkeypatch):
    question = {"id": "q1", "images": [str(tmp_path / "missing.jpg")], "prompt": "What is shown?", "label": "A"}
    (record,), summary = _run(tiny_checkpoint, tmp_path, monkeypatch, question)
    assert record["correct"] is False and "error" in record
    assert summary["num_questions"] == 1 and summary["errors"] == 1 and summary["correct"] == 0


# Merging shard summaries


def _shard(directory, start, num_questions, correct, lvr_steps=8):
    directory.mkdir()
    summary = {"benchmark": "hrbench4k", "checkpoint": "ckpt", "start_index": start, "max_samples": 100,
               "num_questions": num_questions, "correct": correct, "accuracy": correct / num_questions,
               "errors": 0, "decoding": {"lvr_steps": lvr_steps, "num_samples": 1}}
    (directory / "summary.json").write_text(json.dumps(summary))
    return summary


def test_merge_results(tmp_path):
    second, first = _shard(tmp_path / "b", 100, 50, 30), _shard(tmp_path / "a", 0, 100, 70)
    merged = merge_results.main([str(tmp_path / "b"), str(tmp_path / "a"), "--output", str(tmp_path / "all.json")])
    assert merged["num_questions"] == 150 and merged["correct"] == 100
    assert merged["accuracy"] == pytest.approx(100 / 150)
    assert json.loads((tmp_path / "all.json").read_text()) == merged
    with pytest.raises(ValueError, match="consecutive"):
        merge_results.merge([first, second, _shard(tmp_path / "c", 200, 10, 5)])
    with pytest.raises(ValueError, match="decoding"):
        merge_results.merge([first, _shard(tmp_path / "d", 100, 10, 5, lvr_steps=4)])

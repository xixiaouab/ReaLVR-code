"""Evaluate a Qwen2.5-VL LVR checkpoint on MMVP, BLINK, HR-Bench 4K/8K and MME-RealWorld-Lite.

By default every question gets one greedy answer: <|lvr_start|> opens 8 latent steps and decoding
stops at the first </answer>. Everything else follows the checkpoint's generation_config (e.g. its
repetition penalty), and images are capped at the processor's pixel limit unless --max-pixels is set.
With --num-samples N > 1, N answers are sampled per question and one is selected (--selection).

Run from the repository root:

    python -m eval.evaluate --checkpoint CKPT --benchmark mmvp --data-dir DATA --output-dir OUT

Expected layout of --data-dir, with the Hugging Face Hub dataset each directory is downloaded from:

    MMVP/                   MMVP/MMVP
        Questions.csv
        MMVP Images/<Index>.jpg                   (MMVP_Images/ also works)
    BLINK/                  BLINK-Benchmark/BLINK; the val split of five subtasks is used
        <subtask>/val-00000-of-00001.parquet      Counting, IQ_Test, Jigsaw, Relative_Reflectance,
                                                  Spatial_Relation
    HR-Bench/               DreamMr/HR-Bench
        hr_bench_4k.parquet
        hr_bench_8k.parquet
    MME-RealWorld-Lite/     yifanzhang114/MME-RealWorld-Lite, with data.zip extracted in place
        data/MME-RealWorld-Lite.json
        data/imgs/<Image>

OUT receives predictions.jsonl (one line per question) and summary.json. --start-index and
--max-samples select a contiguous range of questions, so a benchmark can be split across processes;
eval/merge_results.py combines the shard summaries.
"""

import argparse
import base64
import csv
import itertools
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import pyarrow.parquet as pq
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, StoppingCriteria, StoppingCriteriaList, set_seed

from realvr.constants import LVR_END_TOKEN, LVR_START_TOKEN
from realvr.model.qwen_lvr import QwenWithLVR

logger = logging.getLogger(__name__)

ANSWER_INSTRUCTION = "Answer with the option letter directly. Put the final answer in <answer>...</answer>."
BLINK_SUBTASKS = ("Counting", "IQ_Test", "Jigsaw", "Relative_Reflectance", "Spatial_Relation")


# Answer extraction and correctness


def normalize_option(text: str) -> str:
    """Option letter of `text`: a parenthesized letter, else a standalone A-D, else the first character."""
    text = str(text).strip()
    match = re.search(r"\(([a-zA-Z])\)", text)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-Da-d])\b", text)
    if match:
        return match.group(1).upper()
    return text[:1].upper()


def match_option_content(content: str, prompt: str) -> str:
    """Letter of the only A-D option in `prompt` whose text matches `content` ("" if none or several)."""
    # Options may share a line ("A. Open B. Closed"): an option's text ends at the next option marker.
    options = dict(re.findall(r"(?:^|\n|\s)\(?([A-D])[\.\)]\s*(.*?)(?=\s+\(?[A-D][\.\)]\s|\n|$)", str(prompt)))
    content = str(content).strip().lower().rstrip(".")
    if not content or not options:
        return ""
    hits = []
    for letter, option in options.items():
        option = option.strip().lower().rstrip(".")
        if content == option or content in option or option in content:
            hits.append(letter)
    return hits[0] if len(hits) == 1 else ""


def extract_answer_text(text: str) -> str:
    """Text of the <answer> tag (closed or not), else the first line."""
    text = str(text)
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"<answer>\s*([^<\n\r]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def extract_answer(text: str, prompt: str = "") -> str:
    """Predicted option letter of a generated answer ("" if none is found).

    The first <answer> tag decides; a phrase inside it is mapped to the option of `prompt` whose text
    it matches. Without a tag, "answer/option/choice: X" and then any standalone A-D letter are used.
    """
    text = str(text)
    match = re.search(r"<answer>\s*([^<\n\r]+)", text, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        is_phrase = len(re.sub(r"[^A-Za-z0-9]", "", candidate)) > 1
        if prompt and is_phrase and not re.fullmatch(r"\(?[A-Da-d]\)?[\.\s]*", candidate):
            mapped = match_option_content(extract_answer_text(text), prompt)
            if mapped:
                return mapped
        return normalize_option(candidate)
    match = re.search(r"(?:answer|option|choice)\s*[:：]?\s*\(?([A-Da-d])\)?", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-Da-d])\b", text)
    if match:
        return match.group(1).upper()
    return ""


def is_correct(prediction: str, label: str) -> bool:
    return normalize_option(prediction) == normalize_option(label)


# Benchmarks. Each loader yields questions in a fixed order as dicts with
# id, images (file paths or base64 data URIs), prompt and label (option letter).


def mc_prompt(question: str, options: str) -> str:
    return f"{question}\nOptions:\n{options}\n{ANSWER_INSTRUCTION}"


def _data_uri(base64_payload: str) -> str:
    return "data:image;base64," + base64_payload


def load_mmvp(data_dir: Path) -> Iterator[dict]:
    root = data_dir / "MMVP"
    image_dir = root / "MMVP Images"
    if not image_dir.is_dir():
        image_dir = root / "MMVP_Images"
    with open(root / "Questions.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            index = int(row["Index"])
            options = row["Options"].replace("(a)", "A.").replace("(b)", "B.")
            yield {
                "id": str(index),
                "images": [str(image_dir / f"{index}.jpg")],
                "prompt": mc_prompt(row["Question"], options),
                "label": normalize_option(row["Correct Answer"]),
            }


def load_blink(data_dir: Path) -> Iterator[dict]:
    image_keys = ("image_1", "image_2", "image_3", "image_4")
    for subtask in BLINK_SUBTASKS:
        path = data_dir / "BLINK" / subtask / "val-00000-of-00001.parquet"
        for row in pq.read_table(path, columns=["idx", "prompt", "answer", *image_keys]).to_pylist():
            images = [row[key]["bytes"] for key in image_keys if row[key] and row[key]["bytes"]]
            yield {
                "id": str(row["idx"]),
                "images": [_data_uri(base64.b64encode(image).decode("ascii")) for image in images],
                "prompt": f"{row['prompt']}\n{ANSWER_INSTRUCTION}",
                "label": normalize_option(row["answer"]),
            }


def load_hrbench(data_dir: Path, resolution: str) -> Iterator[dict]:
    parquet = pq.ParquetFile(data_dir / "HR-Bench" / f"hr_bench_{resolution}.parquet")
    columns = ["index", "question", "answer", "A", "B", "C", "D", "image"]
    for batch in parquet.iter_batches(batch_size=16, columns=columns):
        for row in batch.to_pylist():
            options = "\n".join(f"{letter}. {row[letter]}" for letter in "ABCD")
            yield {
                "id": str(int(row["index"])),
                "images": [_data_uri(str(row["image"]).split(",", 1)[-1])],
                "prompt": mc_prompt(str(row["question"]), options),
                "label": normalize_option(row["answer"]),
            }


def _mme_option(choice) -> str:
    """Rewrites "(A) text" as "A. text"."""
    choice = str(choice).strip()
    if choice.startswith("(") and ")" in choice:
        close = choice.index(")")
        return f"{choice[1:close].strip()}. {choice[close + 1:].strip()}"
    return choice


def load_mme_realworld_lite(data_dir: Path) -> Iterator[dict]:
    Image.MAX_IMAGE_PIXELS = None  # some images exceed PIL's decompression-bomb limit
    root = data_dir / "MME-RealWorld-Lite" / "data"
    with open(root / "MME-RealWorld-Lite.json", encoding="utf-8") as f:
        questions = json.load(f)
    for question in questions:
        options = "\n".join(_mme_option(choice) for choice in question.get("Answer choices", []))
        yield {
            "id": str(question.get("Question_id", "")),
            "images": [str(root / "imgs" / str(question["Image"]))],
            "prompt": mc_prompt(str(question.get("Text", "")), options),
            "label": normalize_option(str(question.get("Ground truth", ""))),
        }


BENCHMARKS = {
    "mmvp": load_mmvp,
    "blink": load_blink,
    "hrbench4k": lambda data_dir: load_hrbench(data_dir, "4k"),
    "hrbench8k": lambda data_dir: load_hrbench(data_dir, "8k"),
    "mme_realworld_lite": load_mme_realworld_lite,
}


def load_benchmark(name: str, data_dir) -> Iterator[dict]:
    """Questions of benchmark `name`, read from `data_dir` (layout in the module docstring)."""
    return BENCHMARKS[name](Path(data_dir))


# Generation


class StopAfterAnswer(StoppingCriteria):
    """Stops a sequence once its generated text contains "</answer>"."""

    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> torch.BoolTensor:
        texts = self.tokenizer.batch_decode(input_ids[:, self.prompt_length:], skip_special_tokens=False)
        return torch.tensor(["</answer>" in text.lower() for text in texts], device=input_ids.device)


def build_inputs(processor, images: Sequence[str], prompt: str, max_pixels: Optional[int] = None):
    """Model inputs for one user turn with the images followed by the prompt.

    `max_pixels` caps the image size both when the image is loaded and in the processor.
    """
    image_kwargs = {"max_pixels": max_pixels} if max_pixels else {}
    content = [{"type": "image", "image": image, **image_kwargs} for image in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    processor_kwargs = {}
    if max_pixels:
        # A lone max_pixels is ignored by processors whose saved config lacks min_pixels, so pass both.
        processor_kwargs = {"min_pixels": processor.image_processor.size["shortest_edge"], "max_pixels": max_pixels}
    return processor(text=[text], images=image_inputs, padding=True, return_tensors="pt", **processor_kwargs)


def mean_logprob(scores: Sequence[torch.Tensor], tokens: torch.LongTensor) -> Optional[float]:
    """Mean log-probability of the generated tokens under the per-step processed scores.

    Equals the mean of the finite entries of `compute_transition_scores(..., normalize_logits=True)`.
    """
    if not scores:
        return None
    values = torch.stack([torch.log_softmax(step[0].float(), dim=-1)[token] for step, token in zip(scores, tokens)])
    values = values[torch.isfinite(values)]
    return float(values.mean()) if values.numel() else None


def generate_answer(
    model,
    tokenizer,
    inputs,
    *,
    lvr_steps: int,
    max_new_tokens: int,
    force_lvr_start: bool = False,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    with_logprob: bool = False,
) -> Tuple[str, Optional[float]]:
    """One generated answer (special tokens kept) and, if requested, its mean token log-probability."""
    prompt_length = inputs["input_ids"].shape[1]
    # Passed as keyword arguments so that they override the checkpoint's generation_config.
    sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p} if do_sample else {"do_sample": False}
    out = model.generate(
        **inputs,
        **sampling,
        max_new_tokens=max_new_tokens,
        lvr_steps=lvr_steps,
        force_lvr_start=force_lvr_start and lvr_steps > 0,
        stopping_criteria=StoppingCriteriaList([StopAfterAnswer(tokenizer, prompt_length)]),
        return_dict_in_generate=True,
        output_scores=with_logprob,
    )
    tokens = out.sequences[0, prompt_length:]
    text = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    logprob = mean_logprob(out.scores, tokens) if with_logprob else None
    return text, logprob


# Selection among several sampled answers


@dataclass
class Sample:
    output: str
    prediction: str
    mean_logprob: Optional[float] = None


@dataclass(frozen=True)
class SelectionWeights:
    vote: float = 1.0
    logprob: float = 0.5
    format: float = 0.25
    latent: float = 0.15


def sample_score(sample: Sample, votes: int, selection: str, weights: SelectionWeights) -> float:
    """Selection score of one sample (higher is better).

    +1 for a non-empty prediction, +weights.format for an <answer> tag, +weights.latent for a complete
    <|lvr_start|> ... <|lvr_end|> span, +0.25 for a prediction in A-D, +weights.vote per sample with the
    same prediction (itself included) and -1e-4 per whitespace-separated word (at most 512).
    "hybrid" also adds weights.logprob times the mean token log-probability.
    """
    output = sample.output
    score = 0.0
    if sample.prediction:
        score += 1.0
    if "<answer>" in output.lower():
        score += weights.format
    if LVR_START_TOKEN in output and LVR_END_TOKEN in output:
        score += weights.latent
    if normalize_option(sample.prediction) in {"A", "B", "C", "D"}:
        score += 0.25
    score += weights.vote * votes
    if selection == "hybrid" and sample.mean_logprob is not None:
        score += weights.logprob * sample.mean_logprob
    return score - min(len(output.split()), 512) * 1e-4


def select_sample(
    samples: Sequence[Sample], selection: str = "hybrid", weights: SelectionWeights = SelectionWeights()
) -> int:
    """Index of the highest-scoring sample; ties go to the earliest one."""
    votes = Counter(normalize_option(sample.prediction) for sample in samples)
    votes.pop("", None)
    scores = [sample_score(s, votes[normalize_option(s.prediction)], selection, weights) for s in samples]
    return max(range(len(samples)), key=lambda i: (scores[i], -i))


# Evaluation loop


def answer_question(model, processor, question: dict, args, weights: SelectionWeights) -> dict:
    """Generates the answer(s) to one question, selects one and scores it."""
    inputs = build_inputs(processor, question["images"], question["prompt"], args.max_pixels).to(model.device)
    multiple = args.num_samples > 1
    samples: List[Sample] = []
    for _ in range(args.num_samples):
        output, logprob = generate_answer(
            model,
            processor.tokenizer,
            inputs,
            lvr_steps=args.lvr_steps,
            max_new_tokens=args.max_new_tokens,
            force_lvr_start=args.force_lvr_start,
            do_sample=multiple and args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            with_logprob=multiple and args.selection == "hybrid",
        )
        samples.append(Sample(output, extract_answer(output, question["prompt"]), logprob))
    chosen = samples[select_sample(samples, args.selection, weights)]
    record = {
        "id": question["id"],
        "benchmark": args.benchmark,
        "prediction": chosen.prediction,
        "label": question["label"],
        "correct": is_correct(chosen.prediction, question["label"]),
        "output": chosen.output,
    }
    if multiple:
        record["samples"] = [asdict(sample) for sample in samples]
    return record


def load_model(checkpoint: str, device: str):
    model = QwenWithLVR.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device
    )
    processor = AutoProcessor.from_pretrained(checkpoint)
    return model.eval(), processor


def decoding_settings(args, model, processor) -> dict:
    """Decoding settings in effect, as recorded in summary.json."""
    settings = {
        "lvr_steps": args.lvr_steps,
        "force_lvr_start": args.force_lvr_start,
        "max_new_tokens": args.max_new_tokens,
        "max_pixels": args.max_pixels or processor.image_processor.size["longest_edge"],
        "repetition_penalty": model.generation_config.repetition_penalty,
        "num_samples": args.num_samples,
    }
    if args.num_samples > 1:
        settings.update(
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            selection=args.selection,
            vote_weight=args.vote_weight,
            logprob_weight=args.logprob_weight,
            format_weight=args.format_weight,
            latent_weight=args.latent_weight,
        )
    return settings


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add = parser.add_argument
    add("--checkpoint", required=True, help="LVR checkpoint (directory or Hub id)")
    add("--benchmark", required=True, choices=list(BENCHMARKS))
    add("--data-dir", required=True, help="benchmark root laid out as described above")
    add("--output-dir", required=True, help="where predictions.jsonl and summary.json are written")
    add("--start-index", type=int, default=0, help="index of the first question to evaluate (default: 0)")
    add("--max-samples", type=int, default=0, help="number of questions to evaluate (default: 0, all)")
    add("--lvr-steps", type=int, default=8, help="latent steps after <|lvr_start|> (default: %(default)s)")
    add("--max-new-tokens", type=int, default=128, help="generation budget per answer (default: %(default)s)")
    add("--max-pixels", type=int, default=None, help="pixel limit per image (default: the processor's)")
    add("--force-lvr-start", action="store_true", help="make <|lvr_start|> the first generated token")
    add("--num-samples", type=int, default=1, help="answers per question, sampled if > 1 (default: %(default)s)")
    add("--temperature", type=float, default=0.7, help="sampling temperature (default: %(default)s)")
    add("--top-p", type=float, default=0.95, help="nucleus sampling threshold (default: %(default)s)")
    add(
        "--selection",
        choices=["majority", "hybrid"],
        default="hybrid",
        help="how one of several answers is chosen: votes with format/latent tie-breakers (majority), "
        "plus the mean token log-probability (hybrid) (default: %(default)s)",
    )
    add("--vote-weight", type=float, default=1.0, help="score per agreeing sample (default: %(default)s)")
    add("--logprob-weight", type=float, default=0.5, help="weight of the log-probability (default: %(default)s)")
    add("--format-weight", type=float, default=0.25, help="bonus for an <answer> tag (default: %(default)s)")
    add("--latent-weight", type=float, default=0.15, help="bonus for a latent span (default: %(default)s)")
    add("--seed", type=int, default=0, help="random seed for sampling (default: %(default)s)")
    add("--device", default="cuda", help="device the model is loaded on (default: %(default)s)")
    args = parser.parse_args(argv)
    if args.start_index < 0 or args.max_samples < 0 or args.num_samples < 1:
        parser.error("--start-index and --max-samples must be >= 0, --num-samples >= 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)
    weights = SelectionWeights(args.vote_weight, args.logprob_weight, args.format_weight, args.latent_weight)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args.checkpoint, args.device)
    stop = args.start_index + args.max_samples if args.max_samples else None
    questions = itertools.islice(load_benchmark(args.benchmark, args.data_dir), args.start_index, stop)

    num_questions = correct = errors = 0
    with open(output_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for question in tqdm(questions, desc=args.benchmark, unit="question"):
            try:
                record = answer_question(model, processor, question, args, weights)
            except Exception as exc:  # the question is kept and counted as wrong
                logger.warning("Question %s failed: %r", question["id"], exc)
                errors += 1
                record = {
                    "id": question["id"],
                    "benchmark": args.benchmark,
                    "prediction": "",
                    "label": question["label"],
                    "correct": False,
                    "output": "",
                    "error": repr(exc),
                }
            num_questions += 1
            correct += int(record["correct"])
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    summary = {
        "benchmark": args.benchmark,
        "checkpoint": args.checkpoint,
        "start_index": args.start_index,
        "max_samples": args.max_samples,
        "num_questions": num_questions,
        "correct": correct,
        "accuracy": correct / num_questions if num_questions else 0.0,
        "errors": errors,
        "decoding": decoding_settings(args, model, processor),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{args.benchmark}: {correct}/{num_questions} correct ({summary['accuracy']:.2%})")
    return summary


if __name__ == "__main__":
    main()

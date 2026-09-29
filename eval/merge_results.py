"""Combine the summaries of evaluation shards (runs of eval/evaluate.py with --start-index/--max-samples).

    python -m eval.merge_results OUT_SHARD_0 OUT_SHARD_1 ... [--output merged.json]
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence


def merge(summaries: List[dict]) -> dict:
    """Summary of the union of the shards; they must share the run settings and cover consecutive questions."""
    first = summaries[0]
    for summary in summaries[1:]:
        for key in ("benchmark", "checkpoint", "decoding"):
            if summary[key] != first[key]:
                raise ValueError(f"shards differ in {key}: {first[key]!r} vs {summary[key]!r}")
    shards = sorted((s for s in summaries if s["num_questions"] > 0), key=lambda s: s["start_index"])
    for previous, shard in zip(shards, shards[1:]):
        if shard["start_index"] != previous["start_index"] + previous["num_questions"]:
            starts = (previous["start_index"], shard["start_index"])
            raise ValueError(f"shards starting at {starts[0]} and {starts[1]} are not consecutive")
    num_questions = sum(s["num_questions"] for s in shards)
    correct = sum(s["correct"] for s in shards)
    return {
        "benchmark": first["benchmark"],
        "checkpoint": first["checkpoint"],
        "num_shards": len(summaries),
        "start_index": shards[0]["start_index"] if shards else 0,
        "num_questions": num_questions,
        "correct": correct,
        "accuracy": correct / num_questions if num_questions else 0.0,
        "errors": sum(s["errors"] for s in shards),
        "decoding": first["decoding"],
    }


def main(argv: Optional[Sequence[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shards", nargs="+", type=Path, help="output directories of the shards")
    parser.add_argument("--output", type=Path, help="also write the merged summary to this file")
    args = parser.parse_args(argv)
    merged = merge([json.loads((shard / "summary.json").read_text(encoding="utf-8")) for shard in args.shards])
    text = json.dumps(merged, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return merged


if __name__ == "__main__":
    main()

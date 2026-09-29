"""Outcome rewards for Stage 2: answer accuracy and output format."""

import re

try:
    from math_verify import parse, verify
except ImportError:  # optional dependency
    parse = verify = None

from realvr.answers import ANSWER_PATTERN, reference_answer, single_answer_block

FORMAT_PATTERN = re.compile(r"^<\|lvr_start\|>.*?<\|lvr_end\|>\s*<answer>.*?</answer>$")


def accuracy_reward(completions, assistant, **kwargs):
    """1.0 when the single <answer> block matches the reference (symbolic check, then exact match)."""
    rewards = []
    for completion, reference in zip(completions, assistant):
        content, solution = completion[0]["content"], reference["content"]
        if not single_answer_block(content):
            rewards.append(0.0)
            continue
        correct = False
        if parse is not None:
            try:
                correct = float(verify(parse(content), parse(solution))) > 0
            except Exception:
                correct = False
        if not correct:
            match = ANSWER_PATTERN.search(content)
            predicted = (match.group(1) if match else content).strip()
            correct = predicted == reference_answer(solution)
        rewards.append(1.0 if correct else 0.0)
    return rewards


def format_reward(completions, **kwargs):
    """1.0 for `<|lvr_start|> ... <|lvr_end|> <answer> ... </answer>` with exactly one answer block."""
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        rewards.append(1.0 if FORMAT_PATTERN.match(content) and single_answer_block(content) else 0.0)
    return rewards


REWARD_FUNCS = [accuracy_reward, format_reward]

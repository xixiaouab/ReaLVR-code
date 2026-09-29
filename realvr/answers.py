"""Parsing of `<answer>...</answer>` blocks, shared by the rewards and the evidence readout."""

import re
from typing import Optional

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>")


def single_answer_block(text: str) -> bool:
    """True when the text contains exactly one <answer> and one </answer> tag (case-insensitive)."""
    return len(re.findall(r"<answer>", text, re.IGNORECASE)) == 1 and len(re.findall(r"</answer>", text, re.IGNORECASE)) == 1


def parse_answer(text: str) -> Optional[str]:
    """Return the content of the single <answer>...</answer> block, or None if unparseable or empty."""
    if not single_answer_block(text):
        return None
    match = ANSWER_PATTERN.search(text)
    if match is None:
        return None
    content = match.group(1).strip()
    return content or None


def reference_answer(solution: str) -> str:
    """The gold answer inside a reference solution (the whole solution when it has no answer block)."""
    match = ANSWER_PATTERN.search(solution)
    return (match.group(1) if match else solution).strip()

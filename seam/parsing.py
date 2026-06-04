"""Split a model completion into chain-of-thought and a final answer."""
from __future__ import annotations

import re

from .config import ANSWER_TAG

_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_TAG = re.compile(re.escape(ANSWER_TAG) + r"\s*(.+)", re.IGNORECASE | re.DOTALL)


def split_response(text: str):
    """Return (cot, final_answer). Handles both <think> models and tag models.

    - <think>...</think>  -> CoT is the think block; answer is the text after it.
    - otherwise           -> CoT is everything before 'Final Answer:'; answer is
                             what follows the tag (first line), falling back to
                             the last non-empty line.
    """
    text = text or ""
    cot, rest = "", text

    m = _THINK.search(text)
    if m:
        cot = m.group(1).strip()
        rest = text[m.end():].strip()

    tag = _TAG.search(rest) or _TAG.search(text)
    if tag:
        answer = tag.group(1).strip().splitlines()[0].strip()
        if not cot:
            cot = text[:tag.start()].strip()
    else:
        lines = [ln.strip() for ln in rest.splitlines() if ln.strip()]
        answer = lines[-1] if lines else ""
        if not cot:
            cot = "\n".join(lines[:-1])
    # Trim common answer decorations.
    answer = answer.strip().strip("*").strip().rstrip(".").strip()
    return cot, answer

"""Split a model completion into chain-of-thought and a final answer.

Robust to the many ways models present a final answer: `Final Answer:`,
`**Answer:**`, `\\boxed{...}`, "The answer is ...", `<think>` reasoning blocks,
or just a last line. Over-matching here is the main cause of artificially low
clean accuracy, so we try several anchors in priority order.
"""
from __future__ import annotations

import re

_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_FINAL = re.compile(r"final\s*answer\s*[:\-=]?\s*(.+)", re.IGNORECASE)
_ANSWER = re.compile(r"\banswer\s*(?:is|:|=|-)?\s*(.+)", re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_LATEX_CMD = re.compile(r"\\[a-zA-Z]+")


def _clean(ans: str) -> str:
    ans = (ans or "").strip()
    b = _BOXED.findall(ans)
    if b:
        ans = b[-1]
    ans = ans.replace("**", "").replace("$", "").replace("\\(", "").replace("\\)", "")
    ans = ans.replace("\\[", "").replace("\\]", "").replace("\\text", "")
    ans = _LATEX_CMD.sub("", ans)
    ans = ans.strip().strip("*").strip().strip(":").strip()
    ans = ans.splitlines()[0].strip() if ans else ans      # answer is one line
    return ans.rstrip(".").strip()


def split_response(text: str):
    """Return (cot, final_answer)."""
    text = text or ""
    cot, body = "", text

    m = _THINK.search(text)
    if m:
        cot = m.group(1).strip()
        body = text[m.end():].strip()

    answer, anchor = "", None
    boxed = _BOXED.findall(body)
    for rx in (_FINAL, _ANSWER):                # explicit anchors win (last hit)
        hits = list(rx.finditer(body))
        if hits:
            answer = _clean(hits[-1].group(1))
            anchor = hits[-1].start()
            break
    if not answer and boxed:                    # \boxed{} without a tag
        answer = _clean(boxed[-1])
    if not answer:                              # fallback: last non-empty line
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        answer = _clean(lines[-1]) if lines else ""
        if not cot:
            cot = "\n".join(lines[:-1])

    if not cot:
        cot = (body[:anchor] if anchor is not None else body).strip()
    return cot, answer

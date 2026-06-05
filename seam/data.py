"""Dataset loading, JSONL I/O, and prompt iteration."""
from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List

from .config import CONDITIONS

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATASET = os.path.join(ROOT, "problems.json")


def load_dataset(path: str = DEFAULT_DATASET) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def iter_items(dataset: List[dict],
               conditions=CONDITIONS,
               categories=None,
               limit=None) -> Iterator[Dict]:
    """Yield one flat work-item per (problem, condition) with its gold info."""
    n = 0
    for rec in dataset:
        if categories and rec["category"] not in categories:
            continue
        for cond in conditions:
            v = rec["variants"][cond]
            yield {
                "id": rec["id"],
                "category": rec["category"],
                "difficulty": rec["difficulty"],
                "bias": rec["bias"],
                "condition": cond,
                "raw_prompt": v["prompt"],     # full problem text (incl. any hint)
                "answer": v["answer"],
                "answer_type": v["answer_type"],
                "answer_keywords": v.get("answer_keywords"),
                "answer_tolerance": v.get("answer_tolerance"),
                "misleading_answer": v.get("misleading_answer"),
            }
            n += 1
            if limit and n >= limit:
                return

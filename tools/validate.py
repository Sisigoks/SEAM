#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validator and summary report for the SEAM benchmark (problems.json).

Runs a battery of structural, schema, consistency and answer-parsing checks on
the committed dataset (independently of the generator), prints a per-category
summary, and exits non-zero if any check fails. Use it as a CI gate.

    python tools/validate.py            # validate ../problems.json
    python tools/validate.py path.json  # validate a specific file
"""
import json
import os
import sys
from fractions import Fraction

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VARIANTS = ("clean", "hinted", "misleading")
DIFFICULTIES = {"easy", "medium", "hard"}
ANSWER_TYPES = {"integer", "fraction", "text", "choice"}
CATEGORIES = {
    "cognitive_reflection", "probability", "rate_problems", "logic", "algebra",
    "combinatorics", "geometry", "sequences", "causal_reasoning",
    "word_problems", "number_theory",
}
SHARED_KEYS = ("answer", "answer_type", "answer_keywords", "answer_tolerance")
HINT_SEP = "\n\nHint: "
# Code points that only appear as a result of UTF-8 mojibake / double-encoding.
MOJIBAKE = ("Â", "Ã", "â", "Ï", "�", "â€")


class Report:
    def __init__(self):
        self.errors = []

    def err(self, pid, msg):
        self.errors.append(f"[{pid}] {msg}")


def parse_int(s):
    return int(str(s).strip())


def parse_number(s):
    # Accepts "8/3", "1/2", "-6.25", "240.0", "2.7", "yes, 2/3" is NOT a number.
    return float(Fraction(str(s).strip()))


def check_variant(rep, pid, name, v):
    if not isinstance(v, dict):
        rep.err(pid, f"{name} variant is not an object")
        return
    prompt = v.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        rep.err(pid, f"{name}.prompt missing or empty")
    answer = v.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        rep.err(pid, f"{name}.answer missing or empty")
    atype = v.get("answer_type")
    if atype not in ANSWER_TYPES:
        rep.err(pid, f"{name}.answer_type invalid: {atype!r}")

    kw = v.get("answer_keywords")
    if atype == "text":
        if not isinstance(kw, list) or not kw or not all(
            isinstance(k, str) and k.strip() for k in kw
        ):
            rep.err(pid, f"{name}: text answer needs a non-empty answer_keywords list")
    elif kw is not None:
        rep.err(pid, f"{name}: answer_keywords present on non-text answer_type {atype!r}")

    tol = v.get("answer_tolerance")
    if tol is not None and not isinstance(tol, (int, float)):
        rep.err(pid, f"{name}.answer_tolerance must be a number, got {tol!r}")

    # Answer must parse according to its declared type.
    try:
        if atype == "integer":
            parse_int(answer)
        elif atype == "fraction":
            parse_number(answer)
        elif atype == "choice":
            if not (len(answer.strip()) == 1 and answer.strip().isalpha()):
                rep.err(pid, f"{name}: choice answer should be a single letter, got {answer!r}")
    except Exception:
        rep.err(pid, f"{name}: answer {answer!r} does not parse as {atype}")

    for s_key, s_val in list(v.items()):
        if isinstance(s_val, str):
            for bad in MOJIBAKE:
                if bad in s_val:
                    rep.err(pid, f"{name}.{s_key}: mojibake marker {bad!r} present")


def validate(path):
    rep = Report()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        print("FATAL: top level must be a non-empty JSON array")
        return 1, {}

    ids = set()
    per_category = {}
    for rec in data:
        pid = rec.get("id", "<no id>")
        if pid in ids:
            rep.err(pid, "duplicate id")
        ids.add(pid)

        if rec.get("difficulty") not in DIFFICULTIES:
            rep.err(pid, f"bad difficulty {rec.get('difficulty')!r}")
        cat = rec.get("category")
        if cat not in CATEGORIES:
            rep.err(pid, f"unknown category {cat!r}")
        if not isinstance(rec.get("bias"), str) or not rec.get("bias"):
            rep.err(pid, "missing/empty bias")

        variants = rec.get("variants")
        if not isinstance(variants, dict) or set(variants) != set(VARIANTS):
            rep.err(pid, f"variants must be exactly {VARIANTS}")
            continue

        for name in VARIANTS:
            check_variant(rep, pid, name, variants[name])

        clean, hinted, mis = variants["clean"], variants["hinted"], variants["misleading"]

        # Hints are the clean prompt plus an appended Hint block.
        for name, v in (("hinted", hinted), ("misleading", mis)):
            cp, hp = clean.get("prompt", ""), v.get("prompt", "")
            if not hp.startswith(cp + HINT_SEP) or len(hp) <= len(cp + HINT_SEP):
                rep.err(pid, f"{name}.prompt is not clean.prompt + a non-empty hint")

        # Gold answer + answer metadata must be identical across all variants.
        for key in SHARED_KEYS:
            vals = [variants[n].get(key) for n in VARIANTS]
            if vals[0] != vals[1] or vals[1] != vals[2]:
                rep.err(pid, f"{key} differs across variants: {vals}")

        # misleading_answer rules.
        if "misleading_answer" in clean or "misleading_answer" in hinted:
            rep.err(pid, "misleading_answer must only appear on the misleading variant")
        ma = mis.get("misleading_answer")
        if not isinstance(ma, str) or not ma.strip():
            rep.err(pid, "misleading.misleading_answer missing or empty")
        elif ma.strip() == str(mis.get("answer")).strip():
            rep.err(pid, "misleading_answer equals the gold answer")

        bucket = per_category.setdefault(cat, {"n": 0, "tokens": 0, "biases": set()})
        bucket["n"] += 1
        bucket["tokens"] += len(clean.get("prompt", "").split())
        bucket["biases"].add(rec.get("bias"))

    return rep, per_category, len(data)


def main():
    here = os.path.dirname(__file__)
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(
        os.path.join(here, "..", "problems.json"))
    result = validate(path)
    if isinstance(result, tuple) and len(result) == 2:
        return result[0]
    rep, per_category, total = result

    print(f"Dataset: {path}")
    print(f"Total problems: {total}\n")
    print(f"{'category':<20}{'count':>6}{'avg_tokens':>12}{'biases':>8}")
    print("-" * 46)
    for cat in sorted(per_category):
        b = per_category[cat]
        avg = b["tokens"] / b["n"] if b["n"] else 0
        print(f"{cat:<20}{b['n']:>6}{avg:>12.1f}{len(b['biases']):>8}")
    print("-" * 46)

    if rep.errors:
        print(f"\nFAILED with {len(rep.errors)} error(s):")
        for e in rep.errors:
            print(f"  - {e}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

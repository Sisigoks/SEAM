"""Shortcut susceptibility as a function of clean-condition confidence.

The matched-condition design with a pre-recorded trap answer lets us ask a
question no existing benchmark can: does a model's confidence in its *clean*
answer predict whether it will follow a later misleading hint? We bucket clean
problems by confidence and measure the flip / shortcut rate in each bucket.

Three proxies for clean confidence (use whichever the run provides):
  * logprob   -- geometric-mean token probability (run with --logprobs)
  * consistency -- modal-answer agreement across samples (run with --samples N)
  * hedging   -- inverse density of hedge words in the clean CoT (always available)
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional

HEDGES = ("i think", "i believe", "maybe", "perhaps", "probably", "might",
          "could be", "not sure", "unsure", "i guess", "seems", "possibly",
          "i'm not certain", "hard to say")


def hedging_density(cot: str) -> float:
    t = (cot or "").lower()
    words = max(1, len(t.split()))
    return 100.0 * sum(t.count(h) for h in HEDGES) / words


def _consistency(samples) -> Optional[float]:
    if not samples:
        return None
    c = Counter(s.strip().lower() for s in samples)
    return c.most_common(1)[0][1] / sum(c.values())


def clean_confidence(row: dict, proxy: str = "auto") -> Optional[float]:
    """A scalar in [0, 1]; higher = more confident on the clean answer."""
    if proxy in ("auto", "logprob") and row.get("confidence") is not None:
        return float(row["confidence"])
    if proxy in ("auto", "consistency") and row.get("samples"):
        return _consistency(row["samples"])
    # hedging fallback: map density (0..~5) to confidence (1..0)
    return max(0.0, 1.0 - hedging_density(row.get("cot", "")) / 5.0)


def _by_id(rows):
    out = defaultdict(dict)
    for r in rows:
        out[r["id"]][r["condition"]] = r
    return out


def susceptibility(rows, proxy="auto", bins=3) -> dict:
    """Flip / shortcut rate within confidence buckets, over problems the model
    got right on the clean condition."""
    pairs = []  # (confidence, flipped, shortcut)
    for cond in _by_id(rows).values():
        if "clean" not in cond or "misleading" not in cond:
            continue
        cl, mi = cond["clean"], cond["misleading"]
        if not cl.get("correct"):
            continue
        conf = clean_confidence(cl, proxy)
        if conf is None:
            continue
        pairs.append((conf, int(not mi.get("correct")),
                      int(mi.get("label") == "shortcut")))
    if len(pairs) < bins:
        return {"n": len(pairs), "buckets": [], "point_biserial": float("nan"),
                "proxy": proxy}

    pairs.sort(key=lambda p: p[0])
    buckets, size = [], len(pairs) // bins
    labels = ["low", "medium", "high"][:bins] if bins == 3 else [f"q{i+1}" for i in range(bins)]
    for i in range(bins):
        lo = i * size
        hi = len(pairs) if i == bins - 1 else (i + 1) * size
        chunk = pairs[lo:hi]
        if not chunk:
            continue
        buckets.append({
            "bucket": labels[i],
            "n": len(chunk),
            "conf_mean": sum(c for c, _, _ in chunk) / len(chunk),
            "flip_rate": sum(f for _, f, _ in chunk) / len(chunk),
            "shortcut_rate": sum(s for _, _, s in chunk) / len(chunk),
        })

    confs = [p[0] for p in pairs]
    flips = [p[1] for p in pairs]
    return {"n": len(pairs), "buckets": buckets, "proxy": proxy,
            "point_biserial": _pearson(confs, flips)}


def _pearson(x, y) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else float("nan")


def analyze(rows_by_model: Dict[str, List[dict]], proxy="auto", bins=3) -> dict:
    return {m: susceptibility(rows, proxy, bins) for m, rows in rows_by_model.items()}


def plot(analysis: dict, path: str):
    """Grouped B&W bar chart: flip rate by confidence bucket, per model."""
    from .figstyle import style, save, style_bars, COL
    with style():
        import matplotlib.pyplot as plt
        import numpy as np
        models = [m for m, a in analysis.items() if a.get("buckets")]
        if not models:
            return None
        labels = [b["bucket"] for b in analysis[models[0]]["buckets"]]
        x = np.arange(len(labels))
        w = 0.8 / max(1, len(models))
        fig, ax = plt.subplots(figsize=(COL, 2.2))
        for i, m in enumerate(models):
            vals = [b["flip_rate"] for b in analysis[m]["buckets"]]
            bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w,
                          label=f"{m.split('-')[0]} (r={analysis[m]['point_biserial']:.2f})")
            style_bars(bars, i)
        ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in labels])
        ax.set_xlabel("Clean-answer confidence"); ax.set_ylabel("Flip rate")
        ax.set_ylim(0, 1); ax.legend(ncol=1)
        ax.set_title("Susceptibility vs. confidence")
        return save(fig, path)

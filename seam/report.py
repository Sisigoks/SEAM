"""Tables and figures from per-model metric summaries (metrics.summarize)."""
from __future__ import annotations

import csv
import os
from typing import List

from .data import ensure_dir

FAIL_TYPES = ("answer_flip", "reasoning_flip", "silent_shortcut", "confabulation")


def _fmt(x, p=3):
    try:
        return f"{x:.{p}f}"
    except (TypeError, ValueError):
        return "n/a"


def _md_table(headers, rows) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line, sep, *body])


def table_accuracy(summaries: List[dict]) -> str:
    """Table 2: accuracy by condition with deltas."""
    headers = ["model", "clean", "hinted", "misleading", "Δ hinted", "Δ misleading",
               "AFR", "shortcut", "SEAM"]
    rows = []
    for s in summaries:
        rows.append([
            s["model"], _fmt(s["accuracy_clean"]), _fmt(s["accuracy_hinted"]),
            _fmt(s["accuracy_misleading"]), _fmt(s["delta_hinted"]),
            _fmt(s["delta_misleading"]), _fmt(s["answer_flip_rate"]),
            _fmt(s["shortcut_rate"]), _fmt(s["seam_score"]),
        ])
    return _md_table(headers, rows)


def table_failures(summaries: List[dict]) -> str:
    """Table 3: failure taxonomy counts (and % of paired problems)."""
    headers = ["model", "n", *FAIL_TYPES]
    rows = []
    for s in summaries:
        tax = s.get("failure_taxonomy", {})
        n = tax.get("n_problems", 0) or 1
        cells = [s["model"], str(tax.get("n_problems", 0))]
        for ft in FAIL_TYPES:
            c = tax.get(ft, 0)
            cells.append(f"{c} ({100*c/n:.0f}%)")
        rows.append(cells)
    return _md_table(headers, rows)


def save_tables(summaries, out_dir):
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "table2_accuracy.md"), "w", encoding="utf-8") as f:
        f.write("### Table 2 — Accuracy by prompt condition\n\n" + table_accuracy(summaries) + "\n")
    with open(os.path.join(out_dir, "table3_failures.md"), "w", encoding="utf-8") as f:
        f.write("### Table 3 — Failure-type taxonomy\n\n" + table_failures(summaries) + "\n")
    with open(os.path.join(out_dir, "summary.csv"), "w", encoding="utf-8", newline="") as f:
        cols = ["model", "accuracy_clean", "accuracy_hinted", "accuracy_misleading",
                "delta_misleading", "answer_flip_rate", "shortcut_rate",
                "condition_sensitivity_kl", "ece_misleading", "mean_rcs",
                "seam_behavioral", "seam_mechanistic", "seam_score"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for s in summaries:
            w.writerow(s)


# --------------------------------------------------------------------------- #
# figures (matplotlib lazy; skipped with a note if unavailable)               #
# --------------------------------------------------------------------------- #
def save_figures(summaries, out_dir) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"[report] figures skipped (matplotlib unavailable: {e})")
        return []
    ensure_dir(out_dir)
    saved = []

    # Fig 3 — failure-type stacked bar.
    fig, ax = plt.subplots(figsize=(8, 4))
    models = [s["model"] for s in summaries]
    bottom = np.zeros(len(models))
    for ft in FAIL_TYPES:
        vals = np.array([s.get("failure_taxonomy", {}).get(ft, 0) for s in summaries], float)
        ax.bar(models, vals, bottom=bottom, label=ft)
        bottom += vals
    ax.set_ylabel("count"); ax.set_title("Fig 3 — Failure profile per model")
    ax.legend(fontsize=7); plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    p = os.path.join(out_dir, "fig3_failures.png"); fig.savefig(p, dpi=130); plt.close(fig)
    saved.append(p)

    # Fig 8 — behavioural vs mechanistic SEAM scatter (quadrant view).
    fig, ax = plt.subplots(figsize=(5, 5))
    for s in summaries:
        x, y = s.get("seam_behavioral"), s.get("seam_mechanistic")
        if x is None or y is None or (isinstance(y, float) and y != y):
            y = 0.0 if (y is None or y != y) else y
        ax.scatter(x, y); ax.annotate(s["model"], (x, y), fontsize=7)
    ax.set_xlabel("behavioural SEAM"); ax.set_ylabel("mechanistic SEAM")
    ax.set_title("Fig 8 — SEAM model ranking"); ax.axhline(.5, ls=":"); ax.axvline(.5, ls=":")
    plt.tight_layout()
    p = os.path.join(out_dir, "fig8_seam_scatter.png"); fig.savefig(p, dpi=130); plt.close(fig)
    saved.append(p)
    return saved


def generate(summaries, out_dir):
    save_tables(summaries, out_dir)
    figs = save_figures(summaries, out_dir)
    print(f"[report] wrote tables + {len(figs)} figure(s) -> {out_dir}")

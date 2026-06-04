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


def _cell(point, s, ci_key):
    ci = s.get(ci_key)
    if isinstance(ci, list) and len(ci) == 3 and ci[1] == ci[1]:
        return f"{_fmt(point)} [{_fmt(ci[1])}, {_fmt(ci[2])}]"
    return _fmt(point)


def table_accuracy(summaries: List[dict]) -> str:
    """Table 2: accuracy by condition, Shortcut Reliance Gap, deltas (with CIs)."""
    headers = ["model", "clean", "hinted", "misleading", "shortcut gap",
               "Δ misleading", "AFR", "shortcut", "SEAM"]
    rows = []
    for s in summaries:
        rows.append([
            s["model"],
            _cell(s["accuracy_clean"], s, "accuracy_clean_ci"),
            _cell(s["accuracy_hinted"], s, "accuracy_hinted_ci"),
            _cell(s["accuracy_misleading"], s, "accuracy_misleading_ci"),
            _cell(s.get("shortcut_reliance_gap"), s, "shortcut_reliance_gap_ci"),
            _fmt(s["delta_misleading"]), _fmt(s["answer_flip_rate"]),
            _fmt(s["shortcut_rate"]), _fmt(s["seam_score"]),
        ])
    return _md_table(headers, rows)


def table_detectors(detectors: dict) -> str:
    """Table 3 (RQ3): detector AUROC / AUPRC and flagged-among-correct rate."""
    names = sorted({d for res in detectors.values() for d in res.get("auroc", {})})
    headers = ["model", *[f"{n} AUROC" for n in names], *[f"{n} flag%" for n in names]]
    rows = []
    for model, res in detectors.items():
        cells = [model]
        cells += [_fmt(res.get("auroc", {}).get(n)) for n in names]
        cells += [_fmt(res.get("flagged_among_correct", {}).get(n)) for n in names]
        rows.append(cells)
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


def save_tables(summaries, out_dir, detectors=None):
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "table2_accuracy.md"), "w", encoding="utf-8") as f:
        f.write("### Table 2 — Accuracy by prompt condition\n\n" + table_accuracy(summaries) + "\n")
    with open(os.path.join(out_dir, "table_failures.md"), "w", encoding="utf-8") as f:
        f.write("### Failure-type taxonomy\n\n" + table_failures(summaries) + "\n")
    if detectors:
        with open(os.path.join(out_dir, "table3_detectors.md"), "w", encoding="utf-8") as f:
            f.write("### Table 3 — Detector comparison (RQ3)\n\n" + table_detectors(detectors) + "\n")
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
def save_figures(summaries, out_dir, detectors=None) -> List[str]:
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

    # Fig 2 — Shortcut Reliance Gap heatmap (model x domain).
    cats = sorted({c for s in summaries for c in s.get("shortcut_reliance_gap_by_category", {})})
    if cats:
        M = np.array([[ (s.get("shortcut_reliance_gap_by_category", {}).get(c) or 0.0) * 100
                        for c in cats] for s in summaries])
        fig, ax = plt.subplots(figsize=(1.1 * len(cats) + 2, 0.6 * len(summaries) + 1.5))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-max(1, np.abs(M).max()), vmax=max(1, np.abs(M).max()))
        ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(len(summaries))); ax.set_yticklabels([s["model"] for s in summaries], fontsize=7)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=7)
        ax.set_title("Fig 2 — Shortcut Reliance Gap (pp) by model x domain")
        fig.colorbar(im, ax=ax, fraction=0.025); plt.tight_layout()
        p = os.path.join(out_dir, "fig2_gap_heatmap.png"); fig.savefig(p, dpi=130); plt.close(fig)
        saved.append(p)

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

    # Fig 4 — detector comparison (held-out AUROC), pooled across models.
    if detectors:
        auroc = {}
        for res in detectors.values():
            for name, v in res.get("auroc", {}).items():
                if v == v:
                    auroc.setdefault(name, []).append(v)
        if auroc:
            names = sorted(auroc)
            means = [float(np.mean(auroc[n])) for n in names]
            fig, ax = plt.subplots(figsize=(6, 0.6 * len(names) + 1.5))
            ax.barh(names, means)
            for i, v in enumerate(means):
                ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
            ax.set_xlim(0, 1.05); ax.set_xlabel("AUROC")
            ax.set_title("Fig 4 — Shortcut detection (grouped held-out)")
            plt.tight_layout()
            p = os.path.join(out_dir, "fig4_detectors.png"); fig.savefig(p, dpi=130); plt.close(fig)
            saved.append(p)
    return saved


def generate(summaries, out_dir, detectors=None):
    save_tables(summaries, out_dir, detectors=detectors)
    figs = save_figures(summaries, out_dir, detectors=detectors)
    print(f"[report] wrote tables + {len(figs)} figure(s) -> {out_dir}")

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
    """Table 3 (RQ3): per-model detector AUROC, best probe layer, and probe gain
    (residual AUROC minus the best text detector)."""
    names = sorted({d for res in detectors.values() for d in res.get("auroc", {})})
    headers = ["model", *[f"{n.replace('_', ' ')} AUROC" for n in names],
               "best layer", "probe gain", "RWRR (residual)"]
    rows = []
    for model, res in detectors.items():
        cells = [_short(model)]
        cells += [_fmt(res.get("auroc", {}).get(n)) for n in names]
        bl = res.get("best_layer")
        cells.append(str(bl) if bl is not None else "—")
        cells.append(_fmt(res.get("probe_advantage")))
        cells.append(_fmt(res.get("flagged_among_correct", {}).get("residual_probe")))
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


def _short(name):
    return name.split("-")[0]


def table_bias(summaries, top=12) -> str:
    """Per reasoning-trap shortcut rate, pooled across models (which traps win)."""
    pooled = {}
    for s in summaries:
        for b, d in (s.get("by_bias") or {}).items():
            sr = d.get("shortcut_rate")
            if sr == sr:
                pooled.setdefault(b, []).append(sr)
    rows = sorted(((b, sum(v) / len(v), len(v)) for b, v in pooled.items()),
                  key=lambda r: -r[1])[:top]
    return _md_table(["bias / trap", "mean shortcut rate", "#models"],
                     [[b, _fmt(sr), str(n)] for b, sr, n in rows])


# --------------------------------------------------------------------------- #
# figures — compact, black-and-white, ACL style (figstyle)                    #
# --------------------------------------------------------------------------- #
def save_figures(summaries, out_dir, detectors=None, confidence=None) -> List[str]:
    try:
        from .figstyle import style, save, style_bars, COL, WIDE, HATCHES, MARKERS
        with style():
            import matplotlib.pyplot as plt  # noqa: F401  (ensures backend usable)
            import numpy as np
    except Exception as e:
        print(f"[report] figures skipped (matplotlib unavailable: {e})")
        return []
    ensure_dir(out_dir)
    saved = []
    from .figstyle import style, save, style_bars, COL, WIDE, HATCHES, MARKERS
    import numpy as np
    models = [s["model"] for s in summaries]

    # Fig 1 — accuracy by condition (grouped bars, bootstrap-CI error bars).
    conds = [("clean", "accuracy_clean", "accuracy_clean_ci"),
             ("hinted", "accuracy_hinted", "accuracy_hinted_ci"),
             ("misleading", "accuracy_misleading", "accuracy_misleading_ci")]
    if any("accuracy_counterfactual" in s for s in summaries):
        conds.append(("counterfactual", "accuracy_counterfactual", None))
    with style():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(WIDE if len(models) > 3 else COL, 2.4))
        x = np.arange(len(models)); w = 0.8 / len(conds)
        for k, (label, key, cikey) in enumerate(conds):
            vals = [s.get(key, float("nan")) for s in summaries]
            yerr = None
            if cikey and all(isinstance(s.get(cikey), list) for s in summaries):
                lo = [max(0.0, s[key] - s[cikey][1]) for s in summaries]
                hi = [max(0.0, s[cikey][2] - s[key]) for s in summaries]
                yerr = [lo, hi]
            bars = ax.bar(x + k * w - 0.4 + w / 2, vals, w, yerr=yerr, capsize=2,
                          error_kw={"lw": 0.6}, label=label)
            style_bars(bars, k)
        ax.set_xticks(x); ax.set_xticklabels([_short(m) for m in models], rotation=20, ha="right")
        ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.08); ax.margins(x=0.02)
        ax.legend(ncol=len(conds), loc="upper center", bbox_to_anchor=(0.5, 1.20))
        ax.set_title("Accuracy by condition")
        saved.append(save(fig, os.path.join(out_dir, "fig1_accuracy.png")))

    # Fig 2 — Shortcut Reliance Gap heatmap (model x domain), grayscale.
    cats = sorted({c for s in summaries for c in s.get("shortcut_reliance_gap_by_category", {})})
    if cats:
        M = np.array([[(s.get("shortcut_reliance_gap_by_category", {}).get(c) or 0.0) * 100
                       for c in cats] for s in summaries])
        with style():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(min(WIDE, 0.5 * len(cats) + 1.6), 0.42 * len(models) + 1.1))
            vmax = max(1.0, float(np.abs(M).max()))
            ax.imshow(np.abs(M), cmap="Greys", vmin=0, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(cats))); ax.set_xticklabels([c[:10] for c in cats], rotation=40, ha="right")
            ax.set_yticks(range(len(models))); ax.set_yticklabels([_short(m) for m in models])
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    v = M[i, j]
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            color="white" if abs(v) > 0.6 * vmax else "black", fontsize=6.5)
            ax.set_title("Shortcut Reliance Gap (pp)")
            saved.append(save(fig, os.path.join(out_dir, "fig2_gap_heatmap.png")))

    # Fig 3 — failure-type stacked bar (grayscale + hatches).
    with style():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(COL, 2.3))
        bottom = np.zeros(len(models))
        for k, ft in enumerate(FAIL_TYPES):
            vals = np.array([s.get("failure_taxonomy", {}).get(ft, 0) for s in summaries], float)
            bars = ax.bar(range(len(models)), vals, bottom=bottom, label=ft.replace("_", " "))
            style_bars(bars, k); bottom += vals
        top = float(bottom.max()) if len(bottom) and bottom.max() > 0 else 1.0
        ax.set_ylim(0, top * 1.22); ax.margins(x=0.06)
        ax.set_xticks(range(len(models))); ax.set_xticklabels([_short(m) for m in models], rotation=25, ha="right")
        ax.set_ylabel("count"); ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.30))
        ax.set_title("Failure profile")
        saved.append(save(fig, os.path.join(out_dir, "fig3_failures.png")))

    # Fig 4 — detector comparison (held-out AUROC), pooled across models.
    if detectors:
        auroc = {}
        for res in detectors.values():
            for name, v in res.get("auroc", {}).items():
                if v == v:
                    auroc.setdefault(name, []).append(v)
        if auroc:
            names = sorted(auroc, key=lambda n: np.mean(auroc[n]))
            means = [float(np.mean(auroc[n])) for n in names]
            with style():
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(COL, 0.45 * len(names) + 1.0))
                bars = ax.barh(range(len(names)), means)
                style_bars(bars, 0)
                ax.set_yticks(range(len(names))); ax.set_yticklabels([n.replace("_", " ") for n in names])
                for i, v in enumerate(means):
                    ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=7)
                ax.set_xlim(0, 1.16); ax.set_ylim(-0.7, len(names) - 0.3)
                ax.set_xlabel("held-out AUROC"); ax.set_title("Shortcut detection")
                saved.append(save(fig, os.path.join(out_dir, "fig4_detectors.png")))

    # Fig 5 — residual-probe AUROC by layer, OVERLAID for every model (uniform).
    layer_models = {m: {int(k): v for k, v in res["layer_auroc"].items()}
                    for m, res in (detectors or {}).items() if res.get("layer_auroc")}
    if layer_models:
        from .figstyle import LINESTYLES
        with style():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(COL, 2.2))
            for i, (m, la) in enumerate(layer_models.items()):
                xs = sorted(la)
                ax.plot([x / max(xs) for x in xs], [la[x] for x in xs], color="black",
                        linestyle=LINESTYLES[i % len(LINESTYLES)], marker=MARKERS[i % len(MARKERS)],
                        markersize=2.6, linewidth=1.0, label=_short(m))
            txt = [r["auroc"].get("cot_tfidf") for r in detectors.values()
                   if r["auroc"].get("cot_tfidf") == r["auroc"].get("cot_tfidf")]
            if txt:
                ax.axhline(float(np.mean(txt)), color="0.55", linestyle=":", linewidth=0.9,
                           label="CoT TF-IDF")
            ax.set_xlabel("relative depth (layer / final)"); ax.set_ylabel("held-out AUROC")
            ax.set_ylim(0.45, 1.03); ax.margins(x=0.04); ax.legend(ncol=1, loc="lower right")
            ax.set_title("Shortcut detectability by layer")
            saved.append(save(fig, os.path.join(out_dir, "fig5_layer_sweep.png")))

    # Fig 9 — residual-probe advantage per model (probe AUROC − best text AUROC).
    adv = [(m, res.get("probe_advantage")) for m, res in (detectors or {}).items()
           if res.get("probe_advantage") is not None and res.get("probe_advantage") == res.get("probe_advantage")]
    if adv:
        with style():
            import matplotlib.pyplot as plt
            ms, vs = [_short(m) for m, _ in adv], [v for _, v in adv]
            fig, ax = plt.subplots(figsize=(COL, 0.45 * len(adv) + 1.0))
            bars = ax.barh(range(len(ms)), vs); style_bars(bars, 2)
            ax.axvline(0, color="0.35", lw=0.7)
            ax.set_yticks(range(len(ms))); ax.set_yticklabels(ms)
            for i, v in enumerate(vs):
                ax.text(v + (0.004 if v >= 0 else -0.004), i, f"{v:+.3f}", va="center",
                        ha="left" if v >= 0 else "right", fontsize=7)
            ax.set_xlabel("probe AUROC − best text detector"); ax.margins(x=0.18, y=0.08)
            ax.set_title("Residual-probe advantage")
            saved.append(save(fig, os.path.join(out_dir, "fig9_probe_advantage.png")))

    # Fig 6 — susceptibility vs. confidence (the confidence analysis).
    if confidence:
        from . import confidence as conf
        p = conf.plot(confidence, os.path.join(out_dir, "fig6_confidence.png"))
        if p:
            saved.append(p)

    # Fig 7 — top reasoning-trap shortcut rates (per-bias finding), grayscale.
    pooled = {}
    for s in summaries:
        for b, d in (s.get("by_bias") or {}).items():
            if d.get("shortcut_rate") == d.get("shortcut_rate"):
                pooled.setdefault(b, []).append(d["shortcut_rate"])
    if pooled:
        items = sorted(((b, sum(v) / len(v)) for b, v in pooled.items()), key=lambda r: r[1])[-10:]
        with style():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(COL, 0.32 * len(items) + 0.9))
            bars = ax.barh([b.replace("_", " ") for b, _ in items], [v for _, v in items])
            style_bars(bars, 1)
            ax.set_xlabel("mean shortcut rate"); ax.set_xlim(0, 1.05)
            ax.set_ylim(-0.7, len(items) - 0.3)
            ax.set_title("Most-effective reasoning traps")
            saved.append(save(fig, os.path.join(out_dir, "fig7_bias.png")))

    # Fig 8 — behavioural vs mechanistic SEAM scatter (quadrant view).
    with style():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(COL, 2.6))
        for i, s in enumerate(summaries):
            x, y = s.get("seam_behavioral"), s.get("seam_mechanistic")
            y = 0.0 if (y is None or (isinstance(y, float) and y != y)) else y
            x = 0.0 if (x is None or (isinstance(x, float) and x != x)) else x
            ax.scatter(x, y, marker=MARKERS[i % len(MARKERS)], color="black", s=18)
            ax.annotate(_short(s["model"]), (x, y), fontsize=6.5, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_xlabel("behavioural SEAM"); ax.set_ylabel("mechanistic SEAM")
        ax.axhline(.5, ls=":", color="0.6", lw=0.7); ax.axvline(.5, ls=":", color="0.6", lw=0.7)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.set_title("SEAM ranking")
        saved.append(save(fig, os.path.join(out_dir, "fig8_seam_scatter.png")))

    return [p for p in saved if p]


def generate(summaries, out_dir, detectors=None, confidence=None):
    save_tables(summaries, out_dir, detectors=detectors)
    if any(s.get("by_bias") for s in summaries):
        with open(os.path.join(out_dir, "table4_bias.md"), "w", encoding="utf-8") as f:
            f.write("### Table 4 — Most-effective reasoning traps\n\n" + table_bias(summaries) + "\n")
    figs = save_figures(summaries, out_dir, detectors=detectors, confidence=confidence)
    print(f"[report] wrote tables + {len(figs)} figure(s) -> {out_dir}")

"""Shortcut-reliance detectors and their comparison (paper RQ3, Fig 3 / Fig 4).

Detection task (honest, reproducible): on the *misleading* items, predict the
observable shortcut label `followed_shortcut` (the model emitted the trap answer)
from text or activations, evaluated held-out and **grouped by base problem** so
paraphrases of one item never straddle the split.

Detectors:
  * cot_lexical    -- overlap between the misleading hint and the generated CoT
  * cot_tfidf      -- TF-IDF of the CoT + logistic regression (grouped CV)
  * residual_probe -- logistic probe on supplied activations (grouped CV)

`flagged_among_correct` (Fig 3) applies a trained detector to the *correct*
answers and reports the flagged fraction = the observable right-answer/
wrong-reason (RWRR) rate.

The final-answer column from the paper is intentionally omitted here: against the
observable label it is the label itself (AUROC 1.0), so it carries no signal for
this target. All sklearn deps are imported lazily.
"""
from __future__ import annotations

import re
from typing import Dict, List


def _hint(raw_prompt: str) -> str:
    return raw_prompt.split("Hint:", 1)[1].strip() if "Hint:" in (raw_prompt or "") else ""


def _toks(s: str):
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def misleading_examples(rows):
    """(rows, y, groups) for misleading items with a shortcut label."""
    mis = [r for r in rows if r.get("condition") == "misleading"]
    y = [int(bool(r.get("followed_shortcut"))) for r in mis]
    groups = [r["id"] for r in mis]
    return mis, y, groups


def lexical_scores(rows) -> List[float]:
    out = []
    for r in rows:
        h, c = _toks(_hint(r.get("raw_prompt", ""))), _toks(r.get("cot", ""))
        out.append(len(h & c) / len(h) if h else 0.0)
    return out


def auroc_auprc(y, scores) -> Dict[str, float]:
    if len(set(y)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    from sklearn.metrics import average_precision_score, roc_auc_score
    return {"auroc": float(roc_auc_score(y, scores)),
            "auprc": float(average_precision_score(y, scores))}


def _grouped_oof(make_model, X, y, groups):
    """Out-of-fold probabilities via GroupKFold (in-sample if too few groups)."""
    import numpy as np
    from sklearn.model_selection import GroupKFold
    y = np.asarray(y)
    n_groups = len(set(groups))
    if n_groups < 2 or len(set(y)) < 2:
        m = make_model().fit(X, y)
        return _proba(m, X)
    oof = np.zeros(len(y), float)
    k = min(5, n_groups)
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        m = make_model().fit(_take(X, tr), y[tr])
        oof[te] = _proba(m, _take(X, te))
    return oof


def _take(X, idx):
    try:
        return X[idx]                     # numpy array
    except TypeError:
        return [X[i] for i in idx]        # list (e.g. raw CoT strings)


def _proba(model, X):
    return model.predict_proba(X)[:, 1]


def _tfidf_model():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    return make_pipeline(TfidfVectorizer(min_df=1, ngram_range=(1, 2)),
                         LogisticRegression(max_iter=1000, class_weight="balanced"))


def _probe_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000, class_weight="balanced"))


def compare(rows, activations=None, threshold: float = 0.5) -> dict:
    """Run all detectors and report grouped-held-out AUROC/AUPRC + Fig-3 rate.

    `activations`: optional (n_misleading, d) array aligned to the misleading
    rows (in dataset order) to enable the residual probe.
    """
    mis, y, groups = misleading_examples(rows)
    result = {"n_misleading": len(mis), "n_shortcut": int(sum(y)), "auroc": {},
              "auprc": {}, "flagged_among_correct": {}}
    if not mis:
        return result

    cots = [r.get("cot", "") for r in mis]
    correct_idx = [i for i, r in enumerate(mis) if r.get("correct")]

    detectors = {"cot_lexical": (lexical_scores(mis), None),
                 "cot_tfidf": (_grouped_oof(_tfidf_model, cots, y, groups), _tfidf_model)}
    if activations is not None:
        import numpy as np
        X = np.asarray(activations, dtype=float)
        detectors["residual_probe"] = (_grouped_oof(_probe_model, X, y, groups), _probe_model)

    print(f"  detectors: scoring {', '.join(detectors)} "
          f"({len(mis)} misleading, {int(sum(y))} shortcut)...", flush=True)
    for name, (scores, maker) in detectors.items():
        m = auroc_auprc(y, scores)
        result["auroc"][name] = m["auroc"]
        result["auprc"][name] = m["auprc"]
        # Fig 3: flagged fraction among correct answers (observable RWRR).
        if correct_idx:
            result["flagged_among_correct"][name] = _flag_correct(
                name, scores, correct_idx, maker, cots, y, groups, activations, threshold)
    return result


def _flag_correct(name, scores, correct_idx, maker, cots, y, groups, activations, thr):
    import numpy as np
    if name == "cot_lexical":
        s = np.asarray(scores)[correct_idx]
    else:                                  # refit on all, score the correct subset
        if name == "cot_tfidf":
            X_all, X_cor = cots, [cots[i] for i in correct_idx]
        else:
            X_all = np.asarray(activations, float)
            X_cor = X_all[correct_idx]
        if len(set(y)) < 2:
            return float("nan")
        model = maker().fit(X_all, np.asarray(y))
        s = _proba(model, X_cor)
    return float((np.asarray(s) >= thr).mean())


def selftest() -> dict:
    """Synthetic sanity check: CoT carries a signal, detectors separate it."""
    import random
    rng = random.Random(0)
    rows = []
    for i in range(120):
        pid = f"x_{i//3:03d}"
        followed = i % 3 == 0
        cot = ("just apply the hint directly and take the shortcut value"
               if followed else "set up the equation and solve carefully step by step")
        rows.append(dict(id=pid, condition="misleading", followed_shortcut=followed,
                         correct=not followed and rng.random() < 0.9,
                         raw_prompt="Q\n\nHint: take the shortcut value directly",
                         cot=cot, answer_type="integer"))
    out = compare(rows)
    return {"auroc": {k: round(v, 3) for k, v in out["auroc"].items()},
            "flagged_among_correct": {k: round(v, 3)
                                      for k, v in out["flagged_among_correct"].items()}}

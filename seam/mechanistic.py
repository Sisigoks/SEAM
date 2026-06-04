"""Mechanistic metrics: activation separability, SAE feature deltas, activation
patching effect sizes, and causal localization.

These operate on plain numpy arrays that you extract from your models (e.g. via
an HF forward pass with hooks, or llama.cpp embeddings), so they are backend
agnostic. `selftest()` exercises every function on synthetic data.
"""
from __future__ import annotations

from typing import Dict, Sequence


def activation_silhouette(activations, labels, n_pca: int = 50) -> float:
    """Silhouette score of clean-vs-misleading clusters after PCA.

    activations: (n_samples, d). labels: 0/1 per sample. Higher (-> 1) means the
    two conditions occupy separable regions of activation space.
    """
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    X = np.asarray(activations, dtype=float)
    y = np.asarray(labels)
    if len(set(y.tolist())) < 2 or X.shape[0] < 3:
        return float("nan")
    k = min(n_pca, X.shape[1], X.shape[0] - 1)
    Xp = PCA(n_components=k).fit_transform(X) if k >= 2 else X
    return float(silhouette_score(Xp, y))


def sae_feature_delta(feats_a, feats_b, top_n: int = 20) -> Dict:
    """Mean absolute change in sparse-autoencoder feature activations.

    Returns the overall mean |delta| plus the most differentially-active feature
    indices (Table 5 / Fig 5).
    """
    import numpy as np
    A = np.asarray(feats_a, dtype=float).mean(axis=0)
    B = np.asarray(feats_b, dtype=float).mean(axis=0)
    delta = np.abs(B - A)
    order = np.argsort(delta)[::-1][:top_n]
    return {
        "mean_abs_delta": float(delta.mean()),
        "top_features": [int(i) for i in order],
        "top_deltas": [float(delta[i]) for i in order],
    }


def patching_logit_diff(logits_clean, logits_patched, target_index: int) -> Dict:
    """Effect of a patch on the target answer logit (the 'smoking gun', Table 6).

    logits_* are (n_samples, vocab) arrays of the final-token logits with and
    without the patched component. Returns the mean logit shift toward the
    target and a crude two-sided p-value via a paired t-test.
    """
    import numpy as np
    C = np.asarray(logits_clean, dtype=float)[:, target_index]
    P = np.asarray(logits_patched, dtype=float)[:, target_index]
    diff = P - C
    mean = float(diff.mean())
    n = len(diff)
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    if sd > 0 and n > 1:
        t = mean / (sd / n ** 0.5)
        # Normal approximation to the t-distribution tail (good for n >~ 20).
        p = math_erfc(abs(t) / 2 ** 0.5)
    else:
        t, p = float("nan"), float("nan")
    return {"mean_logit_diff": mean, "t_stat": t, "p_value": p, "n": n}


def causal_localization(component_effects: Sequence[float], k: int = 5) -> float:
    """Fraction of total absolute patching effect explained by the top-k components."""
    import numpy as np
    e = np.abs(np.asarray(component_effects, dtype=float))
    total = e.sum()
    if total == 0:
        return float("nan")
    topk = np.sort(e)[::-1][:k].sum()
    return float(topk / total)


def math_erfc(x: float) -> float:
    import math
    return math.erfc(x)


def selftest() -> Dict:
    """Run every metric on synthetic data so the module is self-checking."""
    import numpy as np
    rng = np.random.default_rng(0)
    n, d, F = 80, 128, 256
    clean = rng.normal(0, 1, (n, d))
    mislead = rng.normal(0.6, 1, (n, d))            # shifted -> separable
    X = np.vstack([clean, mislead])
    y = [0] * n + [1] * n

    fa = rng.random((n, F)) * 0.1
    fb = fa.copy()
    fb[:, [3, 17, 42]] += 0.9                        # a few features light up

    lc = rng.normal(0, 1, (n, 50))
    lp = lc.copy()
    lp[:, 7] += rng.normal(1.5, 0.5, n)              # patch pushes target logit
    effects = np.abs(rng.normal(0, 1, 30))
    effects[:5] += 6                                 # effect concentrated in 5

    return {
        "silhouette": round(activation_silhouette(X, y), 3),
        "sae": {k: v for k, v in sae_feature_delta(fa, fb, top_n=3).items()},
        "patching": patching_logit_diff(lc, lp, target_index=7),
        "causal_localization_top5": round(causal_localization(effects), 3),
    }

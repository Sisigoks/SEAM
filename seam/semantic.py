"""Reasoning-Consistency Score (RCS) and related CoT-similarity metrics.

RCS = cosine similarity between the chain-of-thought under the clean condition
and under the misleading condition, using a sentence-transformer. Also provides
optional BERTScore / NLI-entailment / coverage signals and a routine to
*fine-tune* the sentence-transformer so similarity tracks reasoning consistency.
All heavy deps are imported lazily.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .config import RCS_MODEL, RCS_MODEL_SMALL

_EMBEDDER = {}


def get_embedder(model_name: str = RCS_MODEL):
    if model_name not in _EMBEDDER:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER[model_name] = SentenceTransformer(model_name)
    return _EMBEDDER[model_name]


def _cos(u, v) -> float:
    import numpy as np
    u, v = np.asarray(u), np.asarray(v)
    d = (np.linalg.norm(u) * np.linalg.norm(v)) or 1.0
    return float(np.dot(u, v) / d)


def _pair_cots(rows, a="clean", b="misleading"):
    by_id = defaultdict(dict)
    for r in rows:
        by_id[r["id"]][r["condition"]] = r.get("cot", "") or ""
    return {pid: (c[a], c[b]) for pid, c in by_id.items() if a in c and b in c}


def rcs_scores(rows, model_name: str = RCS_MODEL,
               a="clean", b="misleading") -> Dict[str, float]:
    """Per-problem cosine similarity between CoT under conditions a and b."""
    pairs = _pair_cots(rows, a, b)
    if not pairs:
        return {}
    model = get_embedder(model_name)
    ids = list(pairs)
    print(f"RCS: encoding {2 * len(ids)} chains-of-thought...", flush=True)
    left = model.encode([pairs[i][0] for i in ids], normalize_embeddings=True,
                        show_progress_bar=True)
    right = model.encode([pairs[i][1] for i in ids], normalize_embeddings=True,
                         show_progress_bar=True)
    return {pid: _cos(left[i], right[i]) for i, pid in enumerate(ids)}


def bertscore_f1(candidates: List[str], references: List[str]) -> List[float]:
    from bert_score import score              # lazy optional dep
    _, _, f1 = score(candidates, references, lang="en", rescale_with_baseline=True)
    return [float(x) for x in f1]


def nli_entailment(cots: List[str], answers: List[str],
                   model_name="cross-encoder/nli-deberta-v3-base") -> List[float]:
    """P(CoT entails the stated answer) for each pair."""
    from sentence_transformers import CrossEncoder
    import numpy as np
    ce = CrossEncoder(model_name)
    logits = ce.predict([(c, f"The answer is {a}.") for c, a in zip(cots, answers)])
    logits = np.asarray(logits)
    ent = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return [float(p[1]) for p in ent]         # index 1 = entailment for this model


def coverage_score(cot: str, key_terms: List[str]) -> float:
    text = cot.lower()
    terms = [t.lower() for t in key_terms if len(t) > 2]
    return (sum(t in text for t in terms) / len(terms)) if terms else float("nan")


# --------------------------------------------------------------------------- #
# fine-tuning the sentence-transformer for RCS validation                     #
# --------------------------------------------------------------------------- #
def build_pairs(rows):
    """(cot_a, cot_b, label) pairs: consistent reasoning -> 1.0, flipped -> 0.0."""
    by_id = defaultdict(dict)
    for r in rows:
        by_id[(r.get("model"), r["id"])][r["condition"]] = r
    pairs = []
    for cond in by_id.values():
        if {"clean", "misleading"} <= set(cond):
            cl, mi = cond["clean"], cond["misleading"]
            consistent = cl["correct"] and mi["correct"] and not mi.get("followed_shortcut")
            pairs.append((cl.get("cot", ""), mi.get("cot", ""), 1.0 if consistent else 0.0))
        if {"clean", "hinted"} <= set(cond):                # positive anchor
            cl, hi = cond["clean"], cond["hinted"]
            if cl["correct"] and hi["correct"]:
                pairs.append((cl.get("cot", ""), hi.get("cot", ""), 1.0))
    return pairs


def _auc(scores, labels) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except Exception:
        # rank-based fallback (Mann-Whitney) if sklearn missing or one class.
        pos = [s for s, l in zip(scores, labels) if l >= 0.5]
        neg = [s for s, l in zip(scores, labels) if l < 0.5]
        if not pos or not neg:
            return float("nan")
        wins = sum(p > n for p in pos for n in neg) + 0.5 * sum(p == n for p in pos for n in neg)
        return wins / (len(pos) * len(neg))


def finetune(rows, base_model: str = RCS_MODEL_SMALL, out_dir: str = "models/rcs-ft",
             epochs: int = 1, batch_size: int = 16, test_frac: float = 0.25) -> dict:
    """Fine-tune a sentence-transformer so CoT similarity tracks consistency.

    Trains with CosineSimilarityLoss on (clean/hinted = consistent -> 1.0,
    clean/misleading-when-flipped -> 0.0) pairs, then reports test ROC-AUC of
    pre- vs post-fine-tuning similarity at separating consistent from flipped
    reasoning. Returns the saved model path and the validation numbers.
    """
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    pairs = build_pairs(rows)
    if len(pairs) < 8 or len({l for *_, l in pairs}) < 2:
        raise ValueError("Not enough labelled, two-class CoT pairs to fine-tune.")
    split = int(len(pairs) * (1 - test_frac))
    train, test = pairs[:split], pairs[split:]

    model = SentenceTransformer(base_model)
    before = _eval_pairs(model, test)

    examples = [InputExample(texts=[a, b], label=l) for a, b, l in train]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.CosineSimilarityLoss(model)
    print(f"Fine-tuning {base_model} on {len(train)} CoT pairs, {epochs} epoch(s)...",
          flush=True)
    model.fit(train_objectives=[(loader, loss)], epochs=epochs,
              warmup_steps=max(1, len(loader) // 10), show_progress_bar=True)
    model.save(out_dir)
    after = _eval_pairs(model, test)

    return {"path": out_dir, "base_model": base_model, "n_train": len(train),
            "n_test": len(test), "auc_before": before, "auc_after": after}


def _eval_pairs(model, test) -> float:
    if not test:
        return float("nan")
    a = model.encode([p[0] for p in test], normalize_embeddings=True)
    b = model.encode([p[1] for p in test], normalize_embeddings=True)
    sims = [_cos(a[i], b[i]) for i in range(len(test))]
    return _auc(sims, [p[2] for p in test])

"""Collect model responses (with chain-of-thought) via llama.cpp.

Loads a local GGUF with llama-cpp-python (Q4_K_M, T4-runnable models), prompts
every problem variant for step-by-step reasoning ending in `Final Answer:`, and
stores the response, parsed CoT, final answer, and a confidence.

Generation params are grouped so only ones llama.cpp reliably accepts are
forwarded. In particular `logprobs` is *blocked* unless the model is loaded with
`logits_all=True`; we therefore gate it behind `want_logprobs` (which flips
`logits_all` on at load) and still fall back gracefully if the build rejects it.
"""
from __future__ import annotations

import collections
import math
import os
from typing import List, Optional

from . import data
from .config import GEN, MODELS, SAMPLE_TEMPERATURE
from .parsing import split_response
from .progress import track

# Sampling params that llama-cpp-python's create_completion accepts on every
# recent build. Anything outside this set is dropped before the call.
SUPPORTED_GEN = ("max_tokens", "temperature", "top_p", "min_p", "top_k", "seed",
                 "stop", "repeat_penalty", "presence_penalty", "frequency_penalty")


def load_llm(model_key: str, gguf_path: Optional[str] = None,
             models_dir: Optional[str] = None, logits_all: bool = False, **overrides):
    """Load a GGUF model with llama-cpp-python (lazy import).

    `logits_all=True` is required for per-token logprobs (confidence); it raises
    memory use, so it is off unless logprobs are requested.
    """
    from llama_cpp import Llama  # noqa: lazy heavy dep

    spec = MODELS.get(model_key, {})
    if gguf_path is None:
        fname = spec.get("file", f"{model_key}.gguf")
        gguf_path = os.path.join(models_dir or ".", fname)
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(
            f"GGUF not found: {gguf_path}. Download {spec.get('repo','?')} "
            f"(Q4_K_M) or pass --gguf <path>.")
    params = dict(n_ctx=spec.get("n_ctx", 8192), n_gpu_layers=-1,
                  logits_all=logits_all, verbose=False)
    params.update(overrides)
    return Llama(model_path=gguf_path, **params)


def generate(llm, prompt: str, gen: dict, want_logprobs: bool = False) -> dict:
    """One completion. Forwards only SUPPORTED_GEN params; logprobs are opt-in
    and degrade to no-confidence if the build/model rejects them."""
    kwargs = {k: gen[k] for k in SUPPORTED_GEN if k in gen}
    if want_logprobs:
        kwargs["logprobs"] = gen.get("logprobs", 1)
    try:
        out = llm.create_completion(prompt=prompt, **kwargs)
    except (ValueError, TypeError, KeyError):
        # e.g. logprobs blocked (logits_all=False) or an unsupported kwarg.
        kwargs.pop("logprobs", None)
        out = llm.create_completion(prompt=prompt, **kwargs)

    choice = out["choices"][0]
    conf = None
    lp = [x for x in ((choice.get("logprobs") or {}).get("token_logprobs") or [])
          if x is not None]
    if lp:
        conf = math.exp(sum(lp) / len(lp))          # geometric-mean token prob
    return {"text": choice.get("text", ""), "confidence": conf}


def _consistency_confidence(answers: List[str]) -> Optional[float]:
    if not answers:
        return None
    c = collections.Counter(a.strip().lower() for a in answers)
    return c.most_common(1)[0][1] / sum(c.values())


def run(dataset: List[dict], model_key: str, out_path: str, *,
        gguf_path=None, models_dir=None, samples=1, want_logprobs=False,
        limit=None, categories=None, progress=True) -> str:
    """Run one model over the dataset and write a responses JSONL.

    Confidence comes from token logprobs when `want_logprobs` is set; otherwise,
    with `samples>1`, it falls back to the self-consistency agreement rate so
    calibration (ECE) is still computable.
    """
    llm = load_llm(model_key, gguf_path=gguf_path, models_dir=models_dir,
                   logits_all=want_logprobs)

    rows = []
    items = list(data.iter_items(dataset, categories=categories, limit=limit))
    from .config import CONDITIONS
    print(f"Running {model_key} over {len(items)} prompts "
          f"(~{len(items) // len(CONDITIONS)} problems x {len(CONDITIONS)} variants)...", flush=True)
    iterator = track(items, desc=f"run:{model_key}") if progress else items
    for item in iterator:
        gens = []
        for s in range(max(1, samples)):
            g = dict(GEN) if s == 0 else dict(GEN, temperature=SAMPLE_TEMPERATURE, seed=s)
            gens.append(generate(llm, item["prompt"], g, want_logprobs=want_logprobs))

        cot, final = split_response(gens[0]["text"])
        sample_answers = [split_response(g["text"])[1] for g in gens]
        conf = gens[0].get("confidence")
        if conf is None and samples > 1:
            conf = _consistency_confidence(sample_answers)

        row = dict(item)
        row.pop("prompt", None)                          # keep rows compact
        row.update(model=model_key, response=gens[0]["text"], cot=cot,
                   final_answer=final, confidence=conf,
                   samples=sample_answers if samples > 1 else None)
        rows.append(row)

    data.write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} responses -> {out_path}")
    return out_path

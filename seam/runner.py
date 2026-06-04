"""Collect model responses (with chain-of-thought) via llama.cpp.

Loads a local GGUF with llama-cpp-python (Q4_K_M recommended), prompts every
problem variant for step-by-step reasoning ending in `Final Answer:`, and stores
the response, parsed CoT, final answer, and a token-logprob confidence.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional

from . import data
from .config import GEN, MODELS, SAMPLE_TEMPERATURE
from .parsing import split_response


def load_llm(model_key: str, gguf_path: Optional[str] = None,
             models_dir: Optional[str] = None, **overrides):
    """Load a GGUF model with llama-cpp-python (lazy import)."""
    from llama_cpp import Llama  # noqa: lazy heavy dep

    spec = MODELS.get(model_key, {})
    if gguf_path is None:
        fname = spec.get("file", f"{model_key}.gguf")
        gguf_path = os.path.join(models_dir or ".", fname)
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(
            f"GGUF not found: {gguf_path}. Download {spec.get('repo','?')} "
            f"(Q4_K_M) or pass --gguf <path>.")
    params = dict(n_ctx=spec.get("n_ctx", 8192), n_gpu_layers=-1, logits_all=False,
                  verbose=False)
    params.update(overrides)
    return Llama(model_path=gguf_path, **params)


def generate(llm, prompt: str, gen: dict) -> dict:
    """One completion; confidence = geometric-mean token probability."""
    out = llm.create_completion(
        prompt=prompt, max_tokens=gen["max_tokens"], temperature=gen["temperature"],
        top_p=gen.get("top_p", 0.95), seed=gen.get("seed", 0), logprobs=1)
    choice = out["choices"][0]
    conf = None
    lp = [x for x in ((choice.get("logprobs") or {}).get("token_logprobs") or [])
          if x is not None]
    if lp:
        conf = math.exp(sum(lp) / len(lp))
    return {"text": choice.get("text", ""), "confidence": conf}


def run(dataset: List[dict], model_key: str, out_path: str, *,
        gguf_path=None, models_dir=None, samples=1, limit=None, categories=None,
        progress=True) -> str:
    """Run one model over the dataset and write a responses JSONL."""
    llm = load_llm(model_key, gguf_path=gguf_path, models_dir=models_dir)

    rows, items = [], list(data.iter_items(dataset, categories=categories, limit=limit))
    for i, item in enumerate(items):
        gens = []
        for s in range(max(1, samples)):
            g = dict(GEN) if s == 0 else dict(GEN, temperature=SAMPLE_TEMPERATURE, seed=s)
            gens.append(generate(llm, item["prompt"], g))

        cot, final = split_response(gens[0]["text"])
        sample_answers = [split_response(g["text"])[1] for g in gens]
        row = dict(item)
        row.pop("prompt", None)                          # keep rows compact
        row.update(model=model_key, response=gens[0]["text"], cot=cot,
                   final_answer=final, confidence=gens[0].get("confidence"),
                   samples=sample_answers if samples > 1 else None)
        rows.append(row)
        if progress and (i + 1) % 100 == 0:
            print(f"  {model_key}: {i + 1}/{len(items)}")

    data.write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} responses -> {out_path}")
    return out_path

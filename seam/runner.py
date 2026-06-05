"""Collect model responses (with chain-of-thought) via llama.cpp.

Uses the **chat completion** API so each model's own chat template is applied and
generation stops at the model's EOS token. Sending a raw prompt to
`create_completion` instead makes instruct models ramble to `max_tokens` (e.g.
~2048 tokens every call), which is the usual cause of multi-minute-per-prompt
runs. The loader also warns loudly when llama-cpp-python is a CPU-only build, the
other usual cause of T4-idle slowness.
"""
from __future__ import annotations

import collections
import math
import os
from typing import List, Optional

from . import data
from .config import GEN, MODELS, SAMPLE_TEMPERATURE, build_messages
from .parsing import split_response
from .progress import track

# Sampling params llama-cpp-python's create_chat_completion accepts on recent
# builds. `logprobs`/`top_logprobs` are gated separately (need logits_all=True).
SUPPORTED_GEN = ("max_tokens", "temperature", "top_p", "min_p", "top_k", "seed",
                 "stop", "repeat_penalty", "presence_penalty", "frequency_penalty")


def gpu_offload_supported() -> Optional[bool]:
    """True/False if the installed llama-cpp-python reports GPU offload, else None."""
    try:
        import llama_cpp
        fn = getattr(llama_cpp, "llama_supports_gpu_offload", None)
        return bool(fn()) if fn else None
    except Exception:
        return None


def load_llm(model_key: str, gguf_path: Optional[str] = None,
             models_dir: Optional[str] = None, logits_all: bool = False,
             n_gpu_layers: int = -1, **overrides):
    """Load a GGUF model with llama-cpp-python (lazy import).

    `n_gpu_layers=-1` offloads all layers to the GPU — but only if the wheel was
    built with CUDA. We detect a CPU-only build and warn, since that is the most
    common reason a T4 run crawls.
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

    supports_gpu = gpu_offload_supported()
    if supports_gpu is False and n_gpu_layers != 0:
        print(
            "WARNING: llama-cpp-python has NO GPU support (CPU-only build); the "
            "T4 will be idle and generation will be very slow.\n"
            "  Reinstall the CUDA build, e.g. on Colab/T4:\n"
            '    CMAKE_ARGS="-DGGML_CUDA=on" pip install --upgrade --force-reinstall '
            "--no-cache-dir llama-cpp-python\n"
            "  (or `pip install llama-cpp-python --extra-index-url "
            "https://abetlen.github.io/llama-cpp-python/whl/cu121`).", flush=True)

    params = dict(n_ctx=spec.get("n_ctx", 8192), n_gpu_layers=n_gpu_layers,
                  logits_all=logits_all, verbose=False)
    params.update(overrides)
    llm = Llama(model_path=gguf_path, **params)
    print(f"Loaded {os.path.basename(gguf_path)} | n_ctx={params['n_ctx']} | "
          f"n_gpu_layers={n_gpu_layers} | gpu_offload={supports_gpu}", flush=True)
    return llm


def _chat_confidence(choice) -> Optional[float]:
    content = (choice.get("logprobs") or {}).get("content")
    if not content:
        return None
    vals = [t.get("logprob") for t in content if t.get("logprob") is not None]
    return math.exp(sum(vals) / len(vals)) if vals else None


def generate(llm, messages, gen: dict, want_logprobs: bool = False) -> dict:
    """One chat completion. Only SUPPORTED_GEN params are forwarded; logprobs are
    opt-in and degrade to no-confidence if the build/model rejects them."""
    kwargs = {k: gen[k] for k in SUPPORTED_GEN if k in gen}
    if want_logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 1
    try:
        out = llm.create_chat_completion(messages=messages, **kwargs)
    except (ValueError, TypeError, KeyError):
        kwargs.pop("logprobs", None)
        kwargs.pop("top_logprobs", None)
        out = llm.create_chat_completion(messages=messages, **kwargs)

    choice = out["choices"][0]
    text = (choice.get("message") or {}).get("content") or ""
    return {"text": text, "confidence": _chat_confidence(choice)}


def _consistency_confidence(answers: List[str]) -> Optional[float]:
    if not answers:
        return None
    c = collections.Counter(a.strip().lower() for a in answers)
    return c.most_common(1)[0][1] / sum(c.values())


def run(dataset: List[dict], model_key: str, out_path: str, *,
        gguf_path=None, models_dir=None, samples=1, want_logprobs=False,
        n_gpu_layers=-1, limit=None, categories=None, progress=True) -> str:
    """Run one model over the dataset and write a responses JSONL.

    Confidence comes from token logprobs when `want_logprobs` is set; otherwise,
    with `samples>1`, it falls back to the self-consistency agreement rate.
    """
    llm = load_llm(model_key, gguf_path=gguf_path, models_dir=models_dir,
                   logits_all=want_logprobs, n_gpu_layers=n_gpu_layers)

    rows = []
    items = list(data.iter_items(dataset, categories=categories, limit=limit))
    from .config import CONDITIONS
    print(f"Running {model_key} over {len(items)} prompts "
          f"(~{len(items) // len(CONDITIONS)} problems x {len(CONDITIONS)} variants)...", flush=True)
    iterator = track(items, desc=f"run:{model_key}") if progress else items
    for item in iterator:
        messages = build_messages(item["raw_prompt"])
        gens = []
        for s in range(max(1, samples)):
            g = dict(GEN) if s == 0 else dict(GEN, temperature=SAMPLE_TEMPERATURE, seed=s)
            gens.append(generate(llm, messages, g, want_logprobs=want_logprobs))

        cot, final = split_response(gens[0]["text"])
        sample_answers = [split_response(g["text"])[1] for g in gens]
        conf = gens[0].get("confidence")
        if conf is None and samples > 1:
            conf = _consistency_confidence(sample_answers)

        row = dict(item)
        row.update(model=model_key, response=gens[0]["text"], cot=cot,
                   final_answer=final, confidence=conf,
                   samples=sample_answers if samples > 1 else None)
        rows.append(row)

    data.write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} responses -> {out_path}")
    return out_path

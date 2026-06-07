"""Residual-stream activation extraction via HF transformers.

Runs the HF (non-GGUF) checkpoint, captures the residual stream at the final
token of every layer, and saves an array aligned to the rows so the residual
probe in `detectors` can be trained/evaluated. This is the mechanistic half of
SEAM ("Activation Mapping") — the part a behavioural-only run never produces.

Output `.npz`: X (n_rows, n_layers, hidden), ids (n_rows,), layers (n_layers,).
On an L40S (48 GB) a 7B in fp16 needs no quantization; pass `load_4bit=True`
for larger checkpoints.
"""
from __future__ import annotations

import os
from typing import List, Optional

from .config import build_messages
from .progress import track


def load_hf(hf_id: str, dtype: str = "float16", load_4bit: bool = False,
            device: Optional[str] = None, output_hidden_states: bool = False):
    """Load an HF causal-LM checkpoint and tokenizer. Shared by `extract` and the
    `run --backend hf` path so behaviour and activations come from one model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    load_kw = dict(output_hidden_states=output_hidden_states)
    if load_4bit:
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        load_kw["device_map"] = "auto"
    else:
        load_kw["torch_dtype"] = getattr(torch, dtype)
        load_kw["device_map"] = device
    model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kw).eval()
    return model, tok, device


def extract_activations(rows: List[dict], hf_id: str, layers: Optional[List[int]] = None,
                        dtype: str = "float16", load_4bit: bool = False,
                        device: Optional[str] = None):
    """Return (X, ids, layer_indices) where X is (n_rows, n_layers, hidden)."""
    import numpy as np
    import torch

    model, tok, device = load_hf(hf_id, dtype, load_4bit, device, output_hidden_states=True)
    print(f"Loaded {hf_id} on {device} (4bit={load_4bit}); extracting "
          f"residual stream for {len(rows)} rows...", flush=True)

    feats, ids = [], []
    with torch.no_grad():
        for r in track(rows, desc=f"acts:{hf_id.split('/')[-1]}"):
            
            inputs = tok.apply_chat_template(
                build_messages(r["raw_prompt"]),
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            
            hs = outputs.hidden_states         # tuple len L+1, each (1,seq,H)
            vec = torch.stack([h[0, -1, :] for h in hs], 0)   # (L+1, H) last token
            feats.append(vec.float().cpu().numpy())
            ids.append(r["id"])

    X = np.stack(feats, 0)                                # (n, L+1, H)
    layer_idx = list(layers) if layers is not None else list(range(X.shape[1]))
    if layers is not None:
        X = X[:, layers, :]
    return X, ids, layer_idx


def save(path: str, X, ids, layers) -> str:
    import numpy as np
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, X=X, ids=np.asarray(ids), layers=np.asarray(layers))
    print(f"Saved activations {tuple(X.shape)} -> {path}", flush=True)
    return path


def load(path: str):
    """Return (X, ids, layers). Accepts .npz (this module) or a bare 2-D .npy."""
    import numpy as np
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        return d["X"], list(d["ids"]), list(d["layers"])
    X = np.load(path)
    if X.ndim == 2:
        X = X[:, None, :]                                # single "layer"
    return X, None, list(range(X.shape[1]))

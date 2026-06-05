"""Lightweight real-time progress reporting.

Uses tqdm (via tqdm.auto, which renders nicely in Colab/Jupyter) when available,
and otherwise falls back to periodic flushed prints so long runs always show how
far they have got. Import is cheap and never hard-fails.
"""
from __future__ import annotations

import time


def track(iterable, desc="work", total=None, every=10):
    """Yield items from `iterable` while reporting progress.

    desc   : label shown in the bar / line.
    total  : item count (inferred from len() when possible).
    every  : fallback prints a line every `every` items (and on first/last).
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    try:
        from tqdm.auto import tqdm
        yield from tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)
        return
    except Exception:
        pass

    start = time.time()
    for i, item in enumerate(iterable, 1):
        yield item
        if i == 1 or i % every == 0 or i == total:
            elapsed = time.time() - start
            rate = i / elapsed if elapsed else 0.0
            msg = f"{desc}: {i}/{total if total else '?'}"
            if total:
                msg += f" ({100 * i / total:.0f}%)"
            msg += f"  {rate:.2f} it/s"
            if total and rate:
                msg += f"  eta {(total - i) / rate:.0f}s"
            print(msg, flush=True)

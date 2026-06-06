"""Compact, black-and-white, ACL-style matplotlib styling.

Figures are grayscale (print-safe), serif, small, and use hatches / linestyles /
markers to distinguish series without colour. Use as a context manager:

    from .figstyle import style, COL, WIDE, HATCHES, GRAYS
    with style():
        fig, ax = plt.subplots(figsize=(COL, 2.2))
        ...
        save(fig, "fig.png")
"""
from __future__ import annotations

import contextlib

# Series distinguishers for B&W output (cycle as needed).
HATCHES = ["", "////", "\\\\\\\\", "xxxx", "....", "++++"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P"]
GRAYS = ["0.20", "0.45", "0.65", "0.82", "0.35", "0.55"]   # bar fills

COL = 3.3      # single ACL column width (inches)
WIDE = 6.9     # full text width

_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.linewidth": 0.4,
    "grid.color": "0.88",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.5,
    "image.cmap": "Greys",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


@contextlib.contextmanager
def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context(_RC):
        yield


def style_bars(bars, idx=0):
    """Apply a grayscale fill + hatch to a bar container (series `idx`)."""
    for b in bars:
        b.set_facecolor(GRAYS[idx % len(GRAYS)])
        b.set_edgecolor("black")
        b.set_linewidth(0.6)
        h = HATCHES[idx % len(HATCHES)]
        if h:
            b.set_hatch(h)


def save(fig, path):
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path

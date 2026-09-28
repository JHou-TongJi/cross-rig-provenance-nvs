# -*- coding: utf-8 -*-
"""Rebuild the A-ONLY process trace with English labels.

The delivered composite (figures/aonly_process_trace_zh.png, 5600 x 2500) has
Chinese panel titles and prints an internal /media/2T_HD/... path under every
panel, neither of which belongs in the article. The ten panel images are real
inference artefacts, so they are kept: this script crops them out of the
composite on its regular grid and relabels them in English.

The English names are taken verbatim from the sentence in Section 6.8 that
enumerates the trace, so figure and prose cannot drift apart.

Grid, measured from the composite rather than assumed: panels are 990 x 557 px
at x = 130 + 1115k for k = 0..4, rows at y = 474 and y = 1579.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SRC = HERE / "figures" / "aonly_process_trace_zh.png"
OUT = HERE / "figures" / "aonly_process_trace.png"

X0, DX, W = 130, 1115, 990
ROWS = (474, 1579)
H = 557

PANELS = [
    "Rectified source RGB",
    "Dense depth candidate",
    "Semantic feature projection",
    "Structure-constraint audit",
    "Target surface depth",
    "Visibility and provenance",
    "Gate output",
    "Final reconstruction",
    "Absolute difference",
    "Redistorted delivery",
]

ACC = "#1f4e79"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9.75, "savefig.dpi": 400, "savefig.bbox": "tight",
})


def main():
    if not SRC.exists():
        raise SystemExit("missing %s -- keep the delivered composite alongside "
                         "the rebuilt one" % SRC)
    im = np.asarray(Image.open(SRC).convert("RGB"))

    fig, axes = plt.subplots(2, 5, figsize=(6.6, 2.30))
    for k, (ax, name) in enumerate(zip(axes.ravel(), PANELS)):
        r, c = divmod(k, 5)
        y, x = ROWS[r], X0 + DX * c
        ax.imshow(im[y:y + H, x:x + W])
        ax.set_title("%02d  %s" % (k + 1, name), fontsize=7.19, pad=2.2,
                     color=ACC)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5)
            s.set_edgecolor("#b0b0b0")
    fig.tight_layout(h_pad=1.1, w_pad=0.25)
    fig.savefig(OUT)
    plt.close(fig)
    w, h = Image.open(OUT).size
    print("wrote %s  %d x %d px (%.2f in at 400 dpi)" % (OUT.name, w, h, w / 400.0))


if __name__ == "__main__":
    main()

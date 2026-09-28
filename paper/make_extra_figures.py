# -*- coding: utf-8 -*-
"""Four figures that show what the tables cannot.

The six figures dropped in Round 16 were each a second rendering of a table
the article already prints in full, and restoring them would put a plot back
beside the numbers it plots. These four are chosen on the opposite criterion:
each shows a structure or a claim that no table in the article carries.

  focal_vs_degradation  the paper's central mechanism claim -- that a long
                        focal length magnifies image-space displacement -- is
                        argued in prose and never plotted
  camera_paths          per-camera trajectories across the three bands; the
                        tables give endpoints, not the paths between them
  band_spread           the within-band gate-fill spread of 50.0 to 61.2
                        points is quoted in prose and never shown
  contract_compliance   zero locked-pixel modification across all 21 cases and
                        budget closure to six decimals, which is the contract's
                        central guarantee and appears only as a sentence

Authoring font is 9.75 pt so that a figure placed at 0.820 of its authored
width renders text at 8.0 pt, matching every other figure in the article.
"""
import csv
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = HERE.parents[1] / "05_评价结果"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9.75,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
})

ACC, WARN, MID = "#1f4e79", "#c0392b", "#2e8b57"

# rectified focal lengths, from the per-frame calibration; the same values
# audit_arithmetic.py re-derives its focal-ratio claims from
FX = {"CAM_FRONT_WIDE": 1921.750887517771, "CAM_FRONT_NARROW": 7340.01536928612,
      "CAM_FRONT_LEFT": 1085.111126604624, "CAM_FRONT_RIGHT": 1113.461101621795,
      "CAM_BACK_LEFT": 1081.731828390162, "CAM_BACK_RIGHT": 1087.331399652501,
      "CAM_BACK": 1903.325095306623}
# Native raster width. The two front cameras are 3840 px wide and the surrounds
# 1920, so plotting raw f_x puts cameras with different rasters on one axis and
# overstates the spread by the width ratio -- the same confound Section 7.2
# corrects in the prose. Everything is scaled to a common 1920 px raster.
NATIVE_W = {"CAM_FRONT_WIDE": 3840, "CAM_FRONT_NARROW": 3840, "CAM_FRONT_LEFT": 1920,
            "CAM_FRONT_RIGHT": 1920, "CAM_BACK_LEFT": 1920, "CAM_BACK_RIGHT": 1920,
            "CAM_BACK": 1920}
FX = {c: f * 1920.0 / NATIVE_W[c] for c, f in FX.items()}

SHORT = {"CAM_FRONT_WIDE": "F-Wide", "CAM_FRONT_NARROW": "F-Narrow",
         "CAM_FRONT_LEFT": "F-Left", "CAM_FRONT_RIGHT": "F-Right",
         "CAM_BACK": "Back", "CAM_BACK_LEFT": "B-Left", "CAM_BACK_RIGHT": "B-Right"}
BANDS = ["mixed_5_10cm", "mixed_10_20cm", "mixed_20_50cm"]
BANDLAB = ["5-10 cm", "10-20 cm", "20-50 cm"]


def load():
    rows = list(csv.DictReader(io.open(EVAL / "gcr-nvs" / "quality_metrics.csv",
                                       encoding="utf-8-sig")))
    for r in rows:
        for k in ("observed_coverage", "gate_fill_of_hole",
                  "generated_fraction_all", "vehicle_risk_fraction_of_hole"):
            r[k] = float(r[k])
        r["locked"] = int(r["locked_pixels_modified"])
    return rows


def get(rows, cam, band, key):
    return next(r[key] for r in rows if r["camera"] == cam and r["pose"] == band)


# --------------------------------------------------------------------------- #
def fig_focal(_unused):
    """Focal length against geometry coverage, with and without the telephoto.

    The article explains the seven-camera coverage spread by focal length: a
    long focal length magnifies the image-space displacement a given metric
    translation produces. Measured on the two-rig geometry audit the relation
    is there -- r = -0.51 under R1 and -0.84 under R2 -- but removing the one
    telephoto collapses it to +0.05 and -0.06. It is a single-point effect,
    exactly like the LiDAR-support relation the article already reports as a
    structural outlier rather than a trend, and this figure holds the two
    claims to the same standard.
    """
    allq = list(csv.DictReader(io.open(EVAL / "quality_metrics.csv",
                                       encoding="utf-8-sig")))
    ref = [r for r in allq if r["metric_scope"] == "no_gt_proxy_and_topology_audit"]
    cur = list(csv.DictReader(io.open(EVAL / "current_rig_quality_metrics.csv",
                                      encoding="utf-8-sig")))
    cams = sorted(FX)
    lf = np.log(np.array([FX[c] for c in cams]))

    def cov(rows, c):
        return 100 * float(next(r["geometry_coverage"] for r in rows
                                if r["camera"] == c))

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    out = {}
    for a, rows, tag in ((ax[0], ref, "R1"), (ax[1], cur, "R2")):
        y = np.array([cov(rows, c) for c in cams])
        keep = [i for i, c in enumerate(cams) if c != "CAM_FRONT_NARROW"]
        r_all = np.corrcoef(lf, y)[0, 1]
        r_wo = np.corrcoef(lf[keep], y[keep])[0, 1]
        out[tag] = {"r_all": round(float(r_all), 3), "r_without": round(float(r_wo), 3)}

        a.scatter(np.exp(lf[keep]), y[keep], s=34, color=ACC, zorder=3)
        i = cams.index("CAM_FRONT_NARROW")
        a.scatter([FX["CAM_FRONT_NARROW"]], [y[i]], s=62, color=WARN, marker="D",
                  zorder=4)
        # only the extremes are labelled: the six short-focal cameras overlap
        # within 40 px of one another and labelling them all is unreadable
        # labels sit below their points; above collides with the title
        # F-Narrow labels to the left below it, F-Wide to the right, because the
        # six short-focal cameras occupy everything left of F-Wide
        for c, off, ha in (("CAM_FRONT_NARROW", (-10, -12), "right"),
                           ("CAM_FRONT_WIDE", (10, -3), "left")):
            a.annotate(SHORT[c], (FX[c], y[cams.index(c)]),
                       textcoords="offset points", xytext=off,
                       ha=ha, fontsize=9.0, color="0.25")
        a.set_xscale("log")
        a.set_xlim(8e2, 2.2e4)
        lo, hi = y.min(), y.max()
        a.set_ylim(lo - 0.14 * (hi - lo), hi + 0.10 * (hi - lo))
        a.set_xlabel("Focal length at a common 1920 px raster (px, log scale)")
        a.set_ylabel("Geometry coverage (%%), rig %s" % tag)
        a.set_title("(%s) rig %s:  r = %+.2f over seven, %+.2f without the telephoto"
                    % ("ab"[0 if tag == "R1" else 1], tag, r_all, r_wo),
                    fontsize=8.8)
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "focal_vs_coverage.png")
    plt.close(fig)
    return out


def fig_paths(rows):
    """Each camera's trajectory across the three perturbation bands."""
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(sorted(FX)):
        o = [100 * get(rows, c, b, "observed_coverage") for b in BANDS]
        g = [100 * get(rows, c, b, "generated_fraction_all") for b in BANDS]
        k = [100 * get(rows, c, b, "gate_fill_of_hole") for b in BANDS]
        col = cmap(i % 10)
        ax[0].plot(o, g, "-o", color=col, lw=1.2, ms=3.2, label=SHORT[c], zorder=3)
        ax[0].annotate(SHORT[c], (o[-1], g[-1]), textcoords="offset points",
                       xytext=(4, 2), fontsize=8.6, color=col)
        ax[1].plot(range(3), k, "-o", color=col, lw=1.2, ms=3.2, zorder=3)
    ax[0].set_xlabel("Observed coverage (%)")
    ax[0].set_ylabel("Generated fraction (%)")
    ax[0].set_title("(a) Each camera's path across the three bands", fontsize=9.75)
    ax[0].invert_xaxis()
    ax[1].set_xticks(range(3))
    ax[1].set_xticklabels(BANDLAB)
    ax[1].set_xlabel("Perturbation band")
    ax[1].set_ylabel("Gate fill of hole (%)")
    ax[1].set_title("(b) Gate acceptance, per camera", fontsize=9.75)
    for a in ax:
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "camera_paths.png")
    plt.close(fig)
    return {"cameras": len(FX)}


def fig_spread(rows):
    """The within-band spread that the prose quotes and no table shows."""
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4))
    vals = [[100 * get(rows, c, b, "gate_fill_of_hole") for c in sorted(FX)]
            for b in BANDS]
    means = [np.mean(v) for v in vals]
    spreads = [max(v) - min(v) for v in vals]

    a = ax[0]
    for i, v in enumerate(vals):
        a.scatter(np.full(len(v), i) + np.linspace(-.09, .09, len(v)), v,
                  s=26, color=ACC, zorder=3)
        a.plot([i - .22, i + .22], [means[i]] * 2, color=WARN, lw=1.6, zorder=4)
    a.set_xticks(range(3))
    a.set_xticklabels(BANDLAB)
    a.set_xlabel("Perturbation band")
    a.set_ylabel("Gate fill of hole (%)")
    a.set_title("(a) Cameras within a band, bar at the mean", fontsize=9.75)

    a = ax[1]
    a.bar(range(3), spreads, 0.5, color=ACC, alpha=0.85, zorder=2,
          label="Spread across cameras, within band")
    a.axhline(max(means) - min(means), color=WARN, ls="--", lw=1.3, zorder=3,
              label="Spread of the band means")
    a.set_xticks(range(3))
    a.set_xticklabels(BANDLAB)
    a.set_xlabel("Perturbation band")
    a.set_ylabel("Spread (percentage points)")
    a.set_ylim(0, max(spreads) * 1.42)          # headroom so the legend clears the bars
    a.set_title("(b) Within-band spread dwarfs the band effect", fontsize=9.75)
    a.legend(frameon=False, fontsize=8.8, loc="upper center")
    for a in ax:
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "band_spread.png")
    plt.close(fig)
    return {"spreads": [round(s, 2) for s in spreads],
            "band_mean_spread": round(max(means) - min(means), 2)}


def fig_compliance(rows):
    """The contract's two guarantees, over all 21 cases."""
    rows_s = sorted(rows, key=lambda r: (BANDS.index(r["pose"]), r["camera"]))
    n = len(rows_s)
    obs = np.array([100 * r["observed_coverage"] for r in rows_s])
    gate = np.array([100 * r["gate_fill_of_hole"] * (1 - r["observed_coverage"])
                     for r in rows_s])
    gen = np.array([100 * r["generated_fraction_all"] for r in rows_s])
    resid = np.abs(obs + gate + gen - 100.0)
    locked = np.array([r["locked"] for r in rows_s])

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    a = ax[0]
    x = np.arange(n)
    a.bar(x, obs, 0.8, color=ACC, label="Observed", zorder=2)
    a.bar(x, gate, 0.8, bottom=obs, color=MID, label="Gate-recovered", zorder=2)
    a.bar(x, gen, 0.8, bottom=obs + gate, color=WARN, label="Generated", zorder=2)
    a.set_xlabel("Shifted-pose case (7 cameras x 3 bands)")
    a.set_ylabel("Share of the frame (%)")
    a.set_ylim(90, 100.6)
    a.set_title("(a) Every case accounts for the whole frame", fontsize=9.75)
    # colours are named in the caption; an in-plot legend here sits on the bars

    a = ax[1]
    # No floor is applied. An earlier version clamped the residual at 1e-7,
    # which put every case on the clamp and made closure look eight orders of
    # magnitude worse than it is.
    a.semilogy(x, np.maximum(resid, 1e-17), "o", color=ACC, ms=3.4, zorder=3,
               label="|budget - 100%|")
    a.axhline(5e-6, color=WARN, ls="--", lw=1.2, zorder=2, label="5e-6 tolerance")
    a.set_ylim(1e-17, 1e-4)
    a.set_xlabel("Shifted-pose case")
    a.set_ylabel("Budget residual (points, log)")
    a.set_title("(b) Closure holds; locked pixels modified: %d in all %d cases"
                % (int(locked.sum()), n), fontsize=9.75)
    a.legend(frameon=False, fontsize=8.8, loc="center right")
    for a in ax:
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "contract_compliance.png")
    plt.close(fig)
    return {"cases": n, "locked_modified_total": int(locked.sum()),
            "max_residual": float(resid.max())}


def main():
    rows = load()
    stats = {"focal": fig_focal(rows), "paths": fig_paths(rows),
             "spread": fig_spread(rows), "compliance": fig_compliance(rows)}
    io.open(HERE / "extra_figure_stats.json", "w", encoding="utf-8").write(
        json.dumps(stats, indent=2))
    for k, v in stats.items():
        print("%-12s %s" % (k, v))
    print("\nfour figures -> %s" % OUT)


if __name__ == "__main__":
    main()

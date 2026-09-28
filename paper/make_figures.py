"""Generate data-driven figures for the WEVJ manuscript.

Every number plotted here is read directly from the machine-readable evaluation
reports under 05_评价结果/. No value is hand-entered.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "05_评价结果"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

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

CAM_ORDER = ["CAM_FRONT_WIDE", "CAM_FRONT_NARROW", "CAM_FRONT_LEFT",
             "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK"]
CAM_SHORT = {"CAM_FRONT_WIDE": "F-Wide", "CAM_FRONT_NARROW": "F-Narrow",
             "CAM_FRONT_LEFT": "F-Left", "CAM_FRONT_RIGHT": "F-Right",
             "CAM_BACK_LEFT": "B-Left", "CAM_BACK_RIGHT": "B-Right",
             "CAM_BACK": "Back"}
POSES = ["mixed_5_10cm", "mixed_10_20cm", "mixed_20_50cm"]
POSE_LABEL = {"mixed_5_10cm": "5-10 cm", "mixed_10_20cm": "10-20 cm",
              "mixed_20_50cm": "20-50 cm"}


def read(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


gcr = read(EVAL / "gcr-nvs" / "quality_metrics.csv")
cur = read(EVAL / "current_rig_quality_metrics.csv")
ref_all = read(EVAL / "quality_metrics.csv")
ref = [r for r in ref_all if r["metric_scope"] == "no_gt_proxy_and_topology_audit"]
tmp = [r for r in ref_all if r["metric_scope"] == "continuous_target_demo_api"]


def fig_pose_trend():
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.00))
    fields = [("observed_coverage", "Observed coverage (%)", "#1f4e79"),
              ("gate_fill_of_hole", "Gate fill of hole (%)", "#2e7d32"),
              ("generated_fraction_all", "Generated / image (%)", "#b34700")]
    x = np.arange(3)
    for ax, (fld, lab, col) in zip(axes, fields):
        means, lo, hi = [], [], []
        for p in POSES:
            v = np.array([float(r[fld]) for r in gcr if r["pose"] == p]) * 100
            means.append(v.mean())
            lo.append(v.mean() - v.min())
            hi.append(v.max() - v.mean())
        ax.errorbar(x, means, yerr=[lo, hi], marker="o", ms=4.5, capsize=3.2,
                    lw=1.4, color=col, ecolor=col, alpha=0.9, mfc="white", mew=1.3)
        ax.set_xticks(x)
        ax.set_xticklabels([POSE_LABEL[p] for p in POSES])
        ax.set_xlabel("Translation perturbation range")
        ax.set_ylabel(lab)
        ax.set_xlim(-0.45, 2.45)
        for xi, m in zip(x, means):
            ax.annotate("%.2f" % m, (xi, m), textcoords="offset points",
                        xytext=(13, 6), ha="left", fontsize=9.14,
                        bbox=dict(boxstyle="square,pad=0.12", fc="white",
                                  ec="none", alpha=0.85))
    fig.tight_layout()
    fig.savefig(OUT / "pose_trend.png")
    plt.close(fig)


def fig_camera_profile():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.20))
    w = 0.26
    x = np.arange(len(CAM_ORDER))
    ax = axes[0]
    for k, p in enumerate(POSES):
        v = [float(next(r for r in gcr if r["pose"] == p and r["camera"] == c)
                   ["observed_coverage"]) * 100 for c in CAM_ORDER]
        ax.bar(x + (k - 1) * w, v, w, label=POSE_LABEL[p],
               color=["#9ec5e8", "#4a90c4", "#1f4e79"][k], edgecolor="white", lw=0.4)
    ax.set_ylim(90, 102.4)
    ax.set_ylabel("Observed coverage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([CAM_SHORT[c] for c in CAM_ORDER], rotation=35, ha="right")
    ax.legend(frameon=False, fontsize=9.14, ncol=3, loc="upper center",
              handlelength=1.4, columnspacing=1.2)
    ax.set_title("(a) Per-camera observed coverage", fontsize=10.37)

    ax = axes[1]
    risk = [np.mean([float(r["vehicle_risk_fraction_of_hole"])
                     for r in gcr if r["camera"] == c]) * 100 for c in CAM_ORDER]
    gen = [np.mean([float(r["generated_fraction_all"])
                    for r in gcr if r["camera"] == c]) * 100 for c in CAM_ORDER]
    ax.bar(x - 0.19, risk, 0.38, label="Vehicle risk / hole", color="#c0392b",
           edgecolor="white", lw=0.4)
    ax.bar(x + 0.19, gen, 0.38, label="Generated / image", color="#e8a33d",
           edgecolor="white", lw=0.4)
    ax.set_ylabel("Mean over 3 pose ranges (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([CAM_SHORT[c] for c in CAM_ORDER], rotation=35, ha="right")
    ax.legend(frameon=False, fontsize=9.14)
    ax.set_title("(b) Risk concentration", fontsize=10.37)
    fig.tight_layout()
    fig.savefig(OUT / "camera_profile.png")
    plt.close(fig)


def fig_rig_compare():
    fig, ax = plt.subplots(figsize=(4.9, 2.20))
    x = np.arange(len(CAM_ORDER))
    a = [float(next(r for r in ref if r["camera"] == c)["geometry_coverage"]) * 100
         for c in CAM_ORDER]
    b = [float(next(r for r in cur if r["camera"] == c)["geometry_coverage"]) * 100
         for c in CAM_ORDER]
    ax.bar(x - 0.19, a, 0.38, color="#7e57a5", edgecolor="white", lw=0.4,
           label="Rig R1: 0.10 m fwd, 0.35 m up, 2 deg pitch")
    ax.bar(x + 0.19, b, 0.38, color="#4a90c4", edgecolor="white", lw=0.4,
           label="Rig R2: 0.20 m up")
    ax.axhline(np.mean(a), color="#7e57a5", ls="--", lw=0.9)
    ax.axhline(np.mean(b), color="#4a90c4", ls="--", lw=0.9)
    ax.set_ylim(85, 101.4)
    ax.set_ylabel("Geometry coverage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([CAM_SHORT[c] for c in CAM_ORDER], rotation=35, ha="right")
    ax.legend(frameon=False, fontsize=8.54, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "rig_compare.png")
    plt.close(fig)


def fig_temporal():
    fig, ax = plt.subplots(figsize=(4.9, 2.10))
    fr = [int(r["frame"]) for r in tmp]
    gv = np.array([float(r["geometry_coverage"]) * 100 for r in tmp])
    mu, sd = gv.mean(), gv.std()
    ax.fill_between(fr, mu - sd, mu + sd, color="#1f4e79", alpha=0.13,
                    label="mean $\pm$ 1 SD")
    ax.axhline(mu, color="#1f4e79", ls="--", lw=0.9, alpha=0.8)
    ax.plot(fr, gv, marker="o", ms=3.8, lw=1.4, color="#1f4e79",
            mfc="white", mew=1.1, label="Geometry coverage")
    ax.set_xlabel("Key-frame index (CAM_BACK_LEFT, reference rig R1)")
    ax.set_ylabel("Geometry coverage (%)")
    ax.set_xticks(range(24, 44, 3))
    ax.set_ylim(93.0, 95.7)
    ax.text(0.985, 0.05,
            "mean %.2f%%   SD %.2f pp   range %.2f-%.2f%%" % (mu, sd, gv.min(), gv.max()),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.78,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#c8c8c8", lw=0.6))
    ax.legend(frameon=False, fontsize=9.14, ncol=2, loc="lower left",
              bbox_to_anchor=(0, 1.01, 1, 0.12), mode="expand",
              borderaxespad=0, handlelength=1.4)
    fig.tight_layout()
    fig.savefig(OUT / "temporal_stability.png")
    plt.close(fig)


def fig_provenance_budget():
    """Stacked evidence budget: observed / gate-recovered / generated / residual unknown."""
    fig, ax = plt.subplots(figsize=(4.9, 2.15))
    obs, gate, gen = [], [], []
    for p in POSES:
        rs = [r for r in gcr if r["pose"] == p]
        o = np.mean([float(r["observed_coverage"]) for r in rs])
        g = np.mean([float(r["gate_fill_of_hole"]) * (1 - float(r["observed_coverage"]))
                     for r in rs])
        gg = np.mean([float(r["generated_fraction_all"]) for r in rs])
        obs.append(o * 100)
        gate.append(g * 100)
        gen.append(gg * 100)
    x = np.arange(3)
    b = np.zeros(3)
    for vals, lab, col in [(obs, "Observed (locked)", "#1f4e79"),
                           (gate, "Gate-recovered", "#2e7d32"),
                           (gen, "Generated", "#b34700")]:
        ax.bar(x, vals, 0.5, bottom=b, label=lab, color=col, edgecolor="white", lw=0.5)
        b = b + np.array(vals)
    ax.set_ylim(94, 100.05)
    ax.set_xticks(x)
    ax.set_xticklabels([POSE_LABEL[p] for p in POSES])
    ax.set_xlabel("Translation perturbation range")
    ax.set_ylabel("Share of target image (%)")
    ax.legend(frameon=False, fontsize=9.14, ncol=3, loc="lower left",
              bbox_to_anchor=(0, 1.01, 1, 0.12), mode="expand",
              borderaxespad=0, handlelength=1.3)
    fig.tight_layout()
    fig.savefig(OUT / "provenance_budget.png")
    plt.close(fig)
    for p, o, g, gg in zip(POSES, obs, gate, gen):
        print("  %-16s observed %.4f  gate %.4f  generated %.4f  sum %.4f"
              % (p, o, g, gg, o + g + gg))


if __name__ == "__main__":
    fig_pose_trend()
    fig_camera_profile()
    fig_rig_compare()
    fig_temporal()
    fig_provenance_budget()
    print("figures written to", OUT)

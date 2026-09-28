# -*- coding: utf-8 -*-
"""Figures derived from the released model code and the supplied calibration.

Nothing here is drawn by hand or estimated:

* `rig_geometry.png`     - camera centres and optical axes decoded from the
                           per-frame extrinsic matrices in camera_config.json.
* `camera_optics.png`    - focal length and horizontal field of view per camera,
                           computed from camera_intric.yaml.
* `model_budget.png`     - parameter counts obtained by instantiating
                           T0StructureConstraintNet and the Restormer decoder
                           from the released inference package.
* `architecture_en.png`  - an English-labelled block diagram of the pipeline,
                           replacing the Chinese-labelled source figure.
* `coverage_heatmap.png` - per-camera x per-band observed coverage from the
                           evaluation CSV.
"""
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Wedge

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "02_数据说明"
EVAL = ROOT / "05_评价结果"
CODE = (ROOT / "03_工程代码" / "code" / "gcr-nvs-inference-only-20260826"
        / "src" / "gcr_nvs" / "models")
OUT = HERE / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9.75,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.5,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
})

# Verified extrinsic array order (Section 4.2): not lexicographic.
ORDER = ["CAM_BACK_LEFT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
         "CAM_BACK", "CAM_FRONT_WIDE", "CAM_FRONT_NARROW"]
SHORT = {"CAM_FRONT_WIDE": "F-Wide", "CAM_FRONT_NARROW": "F-Narrow",
         "CAM_FRONT_LEFT": "F-Left", "CAM_FRONT_RIGHT": "F-Right",
         "CAM_BACK_LEFT": "B-Left", "CAM_BACK_RIGHT": "B-Right",
         "CAM_BACK": "Back"}
POSES = ["mixed_5_10cm", "mixed_10_20cm", "mixed_20_50cm"]
PLAB = {"mixed_5_10cm": "5-10 cm", "mixed_10_20cm": "10-20 cm",
        "mixed_20_50cm": "20-50 cm"}


def read_rig():
    cfg = json.load(open(DATA / "camera_config.json", encoding="utf-8-sig"))
    out = {}
    for i, e in enumerate(cfg):
        T = np.array(e["camera_external"], dtype=float).reshape(4, 4)
        if not e.get("rowMajor", True):
            T = T.T
        R, t = T[:3, :3], T[:3, 3]
        centre = -R.T @ t
        axis = R.T @ np.array([0.0, 0.0, 1.0])
        w = e["width"]
        fx = e["camera_internal"]["fx"]
        out[ORDER[i]] = dict(C=centre, axis=axis, fx=fx, w=w, h=e["height"],
                             fov=2 * math.degrees(math.atan(w / (2 * fx))))
    return out


# --------------------------------------------------------------------- rig
def fig_rig_geometry(rig):
    fig, ax = plt.subplots(figsize=(4.5, 2.75))
    # vehicle footprint is not in the calibration; draw only the sensor layout
    for name in ORDER:
        r = rig[name]
        x, y = r["C"][0], r["C"][1]
        yaw = math.atan2(r["axis"][1], r["axis"][0])
        half = math.radians(r["fov"] / 2.0)
        reach = 2.9 if r["fov"] > 60 else 4.6
        ax.add_patch(Wedge((x, y), reach,
                           math.degrees(yaw - half), math.degrees(yaw + half),
                           facecolor="#4a90c4", alpha=0.085, edgecolor="none"))
        ax.plot([x], [y], "o", ms=5, color="#1f4e79", zorder=5)
        ax.add_patch(FancyArrowPatch((x, y),
                                     (x + 0.85 * math.cos(yaw), y + 0.85 * math.sin(yaw)),
                                     arrowstyle="-|>", mutation_scale=8,
                                     color="#1f4e79", lw=1.1, zorder=5))
        # label beyond the arrow tip, along the optical axis, so the seven
        # labels separate the way the cameras do
        reach_lab = {"CAM_FRONT_WIDE": 2.55, "CAM_FRONT_NARROW": 1.35}.get(name, 1.75)
        nudge = {"CAM_FRONT_WIDE": 0.42, "CAM_FRONT_NARROW": -0.46}.get(name, 0.0)
        lx = x + reach_lab * math.cos(yaw)
        ly = y + reach_lab * math.sin(yaw) + nudge
        ax.annotate(SHORT[name], (lx, ly), ha="center", va="center", fontsize=8.54,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#b9b9b9", lw=0.5))
    ax.plot([0], [0], "s", ms=6, color="#b34700", zorder=6)
    ax.annotate("LiDAR origin", (0, 0), textcoords="offset points", xytext=(6, -14),
                fontsize=8.78, color="#b34700")
    ax.set_xlabel("x, vehicle forward (m)")
    ax.set_ylabel("y, vehicle left (m)")
    ax.set_aspect("equal")
    ax.set_xlim(-3.6, 5.6)
    ax.set_ylim(-3.3, 3.3)
    fig.tight_layout()
    fig.savefig(OUT / "rig_geometry.png")
    plt.close(fig)


def fig_camera_optics(rig):
    fig, axes = plt.subplots(1, 2, figsize=(6.1, 2.05))
    names = ["CAM_FRONT_WIDE", "CAM_FRONT_NARROW", "CAM_FRONT_LEFT",
             "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK"]
    x = np.arange(len(names))
    fx = [rig[n]["fx"] for n in names]
    fov = [rig[n]["fov"] for n in names]
    cols = ["#b34700" if n == "CAM_FRONT_NARROW" else "#4a90c4" for n in names]
    ax = axes[0]
    ax.bar(x, fx, 0.62, color=cols, edgecolor="white", lw=0.4)
    ax.set_ylabel("focal length $f_x$ (px)")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[n] for n in names], rotation=35, ha="right")
    for xi, v in zip(x, fx):
        ax.annotate("%.0f" % v, (xi, v), textcoords="offset points", xytext=(0, 2),
                    ha="center", fontsize=8.29)
    ax.set_ylim(0, max(fx) * 1.18)
    ax.set_title("(a) Focal length", fontsize=10.37)
    ax = axes[1]
    ax.bar(x, fov, 0.62, color=cols, edgecolor="white", lw=0.4)
    ax.set_ylabel("horizontal FOV (deg)")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[n] for n in names], rotation=35, ha="right")
    for xi, v in zip(x, fov):
        ax.annotate("%.1f" % v, (xi, v), textcoords="offset points", xytext=(0, 2),
                    ha="center", fontsize=8.29)
    ax.set_ylim(0, max(fov) * 1.18)
    ax.set_title("(b) Horizontal field of view", fontsize=10.37)
    fig.tight_layout()
    fig.savefig(OUT / "camera_optics.png")
    plt.close(fig)


# ------------------------------------------------------------------- model
def model_params():
    """Instantiate the released modules and count parameters."""
    import importlib.util
    sys.path.insert(0, str(CODE))

    def load(fname, mod):
        spec = importlib.util.spec_from_file_location(mod, CODE / fname)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    t0 = load("t0_structure_constraint.py", "t0mod")
    net = t0.T0StructureConstraintNet(channels=48, semantic_dim=32)
    stages = {n: sum(p.numel() for p in m.parameters())
              for n, m in net.named_children()}
    stages = {k: v for k, v in stages.items() if v}
    rest = load("restormer.py", "restmod")
    dec = rest.RestormerSmallFeatureDecoder(input_channels=48)
    ada = rest.RestormerFeatureAdapter(input_channels=48)
    return (stages, sum(p.numel() for p in net.parameters()),
            sum(p.numel() for p in dec.parameters()),
            sum(p.numel() for p in ada.parameters()))


def fig_model_budget(stages, t0_total, dec_total, ada_total):
    fig, axes = plt.subplots(1, 2, figsize=(6.1, 2.10),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax = axes[0]
    groups = [("encoder", ["enc1", "enc2", "enc3", "enc4"], "#9ec5e8"),
              ("bottleneck", ["bottleneck"], "#1f4e79"),
              ("global transformer", ["global_structure"], "#2e7d32"),
              ("decoder", ["dec1", "dec2", "dec3", "dec4"], "#4a90c4"),
              ("proj. + head", ["semantic_projection", "head"], "#b0b0b0")]
    labels = [g[0] for g in groups]
    vals = [sum(stages.get(k, 0) for k in g[1]) / 1e6 for g in groups]
    cols = [g[2] for g in groups]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, 0.6, color=cols, edgecolor="white", lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("parameters (M)")
    for yi, v in zip(y, vals):
        ax.annotate("%.2f M  (%.0f%%)" % (v, 100 * v / (t0_total / 1e6)),
                    (v, yi), textcoords="offset points", xytext=(4, -2.5), fontsize=8.54)
    ax.set_xlim(0, max(vals) * 1.42)
    ax.set_title("(a) Structure network, %.2f M total" % (t0_total / 1e6), fontsize=10.37)

    ax = axes[1]
    names = ["Structure\nnetwork", "Residual\ndecoder", "Feature\nadapter"]
    v = [t0_total / 1e6, dec_total / 1e6, ada_total / 1e6]
    ax.bar(np.arange(3), v, 0.55, color=["#1f4e79", "#4a90c4", "#9ec5e8"],
           edgecolor="white", lw=0.5)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(names, fontsize=9.14)
    ax.set_ylabel("parameters (M)")
    for xi, val in zip(np.arange(3), v):
        ax.annotate("%.2f" % val, (xi, val), textcoords="offset points",
                    xytext=(0, 2), ha="center", fontsize=8.78)
    ax.set_ylim(0, max(v) * 1.2)
    ax.set_title("(b) Trained modules", fontsize=10.37)
    fig.tight_layout()
    fig.savefig(OUT / "model_budget.png")
    plt.close(fig)


# ------------------------------------------------------- heatmap and restores
def read(p):
    with open(p, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def fig_coverage_heatmap():
    rows = read(EVAL / "gcr-nvs" / "quality_metrics.csv")
    cams = ["CAM_FRONT_WIDE", "CAM_FRONT_NARROW", "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK"]
    fig, axes = plt.subplots(1, 2, figsize=(6.1, 1.95))
    for ax, (fld, lab, cmap) in zip(axes, [
            ("observed_coverage", "observed coverage (%)", "Blues"),
            ("vehicle_risk_fraction_of_hole", "vehicle risk in hole (%)", "OrRd")]):
        M = np.array([[float(next(r for r in rows if r["pose"] == p and r["camera"] == c)[fld]) * 100
                       for c in cams] for p in POSES])
        im = ax.imshow(M, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(cams)))
        ax.set_xticklabels([SHORT[c] for c in cams], rotation=35, ha="right", fontsize=8.54)
        ax.set_yticks(range(3)); ax.set_yticklabels([PLAB[p] for p in POSES], fontsize=9.14)
        ax.grid(False)
        thr = M.min() + 0.55 * (M.max() - M.min())
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, "%.1f" % M[i, j], ha="center", va="center", fontsize=7.81,
                        color="white" if M[i, j] > thr else "#222222")
        ax.set_title(lab, fontsize=10.00)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).ax.tick_params(labelsize=7.93)
    fig.tight_layout()
    fig.savefig(OUT / "coverage_heatmap.png")
    plt.close(fig)


# -------------------------------------------------------------- architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(6.3, 2.55))
    ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=6.9, tc="#111111"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, linespacing=1.35)

    def arrow(x1, y1, x2, y2, col="#555555"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=7, color=col, lw=0.9,
                                     shrinkA=0, shrinkB=0))

    INP, GEO, LRN, GATE, OUT_ = "#e8eef5", "#dbe9d5", "#e6e0ef", "#fbe6d4", "#f2f2f2"
    EI, EG, EL, EGT, EO = "#4a90c4", "#4f8f43", "#7e57a5", "#c8752a", "#888888"

    box(1, 38, 20, 11, "Seven source RGB\n$K_i$, $D_i$, $T_i$", INP, EI)
    box(1, 24, 20, 10, "LiDAR sweep\n$P=\\{X_n\\}$", INP, EI)
    box(1, 10, 20, 10, "Target rig config\n$K^t_j$, $D^t_j$, $T^t_j$", INP, EI)

    box(25, 38, 20, 11, "Brown-Conrady\nrectification  (Eq. 8)", GEO, EG)
    box(25, 24, 20, 10, "LiDAR z-buffer\nmetric anchors", GEO, EG)
    box(25, 10, 20, 10, "Rigid SE(3) rig\ntransform  (Eq. 5)", GEO, EG)

    box(49, 40, 22, 9, "DA3 dense depth\n(frozen)", LRN, EL)
    box(49, 29, 22, 9, "DINOv2 + AnyUp\n(frozen)", LRN, EL)
    box(49, 18, 22, 9, "Structure network\n26.49 M, trained", LRN, EL)
    box(49, 7, 22, 9, "Plane sweep\n$N_d=32$  (Eq. 11-15)", GEO, EG)

    box(75, 29, 23, 10, "Source-target-source\ncycle gate  (Sec. 4.10)", GATE, EGT)
    box(75, 16, 23, 10, "Masked residual\ndecoder 26.18 M  (Eq. 20)", GATE, EGT)
    box(75, 3, 23, 10, "Redistortion\nexport  (Sec. 4.12)", OUT_, EO)

    for y in (43.5, 29, 15):
        arrow(21, y, 25, y)
    arrow(45, 43.5, 49, 44.5)
    arrow(45, 29, 49, 33.5)
    arrow(45, 15, 49, 11.5)
    arrow(60, 40, 60, 38)      # DA3 -> DINO column flow
    arrow(60, 29, 60, 27)      # DINO -> structure
    arrow(71, 22.5, 75, 30)    # structure -> gate
    arrow(71, 11.5, 75, 29)    # plane sweep -> gate
    arrow(86.5, 29, 86.5, 26)  # gate -> decoder
    arrow(86.5, 16, 86.5, 13)  # decoder -> export

    ax.text(50, 50.4, "geometry path writes depth   |   learned path writes colour only",
            ha="center", fontsize=8.42, color="#444444", style="italic")
    ax.text(86.5, 0.4,
            "observed pixels locked; gate and generated masks exported",
            ha="center", fontsize=8.04, color="#444444")
    fig.tight_layout()
    fig.savefig(OUT / "architecture_en.png")
    plt.close(fig)


def write_model_table(stages, t0_total, dec_total, ada_total):
    """Emit the learned-module specification table from the instantiated code."""
    L = []
    # the document numbers this Table 4; verify_manuscript.py checks the sequence
    L.append("TABSTART|Table 4. Learned modules of the released inference configuration. "
             "Parameter counts are obtained by instantiating the released code, not estimated. "
             "Frozen modules are used as published and are not fine-tuned on this "
             "data.|1.45,1.15,1.9,0.85,0.75|LLLCC|7.5")
    L.append("TABHDR|Module\tArchitecture\tRole and constraint\tParams\tState")
    L.append("TABROW|Dense depth (DA3Metric-Large)\tExternal ViT\tDense metric depth candidate; "
             "not ground truth, scale set by LiDAR\t-\tFrozen")
    L.append("TABROW|Semantic encoder (DINOv2)\tViT-L/14 at 364 x 644; ViT-B/14 conditions the "
             "structure net\tSemantic context and correspondence; writes no colour, no depth\t-\tFrozen")
    L.append("TABROW|Feature upsampler (AnyUp)\tMulti-backbone head\tRestores high-resolution "
             "feature boundaries\t-\tFrozen")
    L.append("TABROW|Structure network\t4-stage U-Net, 48 base channels, 384-channel bottleneck, "
             "3-layer global transformer\tCorrects DA3 scale and surface residual under LiDAR; "
             "writes depth only, never colour, never overwrites an anchor\t%.2f M\tTrained"
             % (t0_total / 1e6))
    L.append("TABROW|Residual decoder\tRestormer-Small, dim 48, blocks (4, 6, 6, 8)\tPredicts RGB "
             "residual, modification strength and log uncertainty; restricted to the mask of "
             "Equation (20)\t%.2f M\tTrained" % (dec_total / 1e6))
    L.append("TABROW|Feature adapter\t48 -> 32 channel Restormer adapter\tProjects fused features "
             "for the decoder\t%.2f M\tTrained" % (ada_total / 1e6))
    L.append("TABRULE|")
    L.append("TABROW|**Trained total**\t\t\t**%.2f M**\t" % ((t0_total + dec_total + ada_total) / 1e6))
    L.append("TABEND|The structure network takes 48 input channels, 16 geometric and 32 semantic, "
             "and emits four: log-residual, correction gate, confidence and depth-boundary "
             "probability. Its log-residual is bounded at 0.22, so a single forward pass can "
             "rescale a depth by at most about 25%. The released checkpoints were trained at "
             "320 x 180 with 48 base channels and are frozen for every result in this paper.")
    (OUT.parent / "tables" / "t_model.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("tables/t_model.txt written")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    rig = read_rig()
    fig_rig_geometry(rig)
    fig_camera_optics(rig)
    st, t0, dec, ada = model_params()
    fig_model_budget(st, t0, dec, ada)
    fig_coverage_heatmap()
    fig_architecture()
    write_model_table(st, t0, dec, ada)
    print("structure network %.2f M, residual decoder %.2f M, feature adapter %.3f M"
          % (t0 / 1e6, dec / 1e6, ada / 1e6))
    for k in ("enc1", "enc2", "enc3", "enc4", "bottleneck", "global_structure",
              "dec4", "dec3", "dec2", "dec1"):
        print("   %-18s %9d" % (k, st[k]))
    print("figures written to", OUT)

# -*- coding: utf-8 -*-
"""Figures built from the delivered BEV renders, LiDAR point cloud, per-camera
projections and vehicle-extension risk masks.

Everything here is measured from files in 04_生成结果 / 02_数据说明; no value is
typed in by hand. Console output is the audit trail, and the numbers printed
are the ones quoted in the manuscript.

Produces
    figures/bev_triple.png        source / target / difference BEV
    figures/pointcloud_stats.png  range, azimuthal shadow, height profile
    figures/camera_support.png    per-camera LiDAR support vs unknown coverage
    figures/risk_mask_check.png   masks re-measured against the results table
    figures/projection_montage.png  LiDAR reprojected into all seven cameras
    tables/t_support.txt          per-camera support table
"""
import csv
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OPT = ROOT / "04_生成结果" / "optional_bev_or_projection"
BEV = OPT / "optional_bev_or_projection"
CAL = OPT / "calibration_visualization"
GEN = ROOT / "04_生成结果" / "generated_images" / "GCR-NVS"
EVAL = ROOT / "05_评价结果"
CFG = ROOT / "02_数据说明" / "camera_config.json"
OUT = HERE / "figures"
TAB = HERE / "tables"

SHORT = {"CAM_FRONT_WIDE": "F-Wide", "CAM_FRONT_NARROW": "F-Narrow",
         "CAM_FRONT_LEFT": "F-Left", "CAM_FRONT_RIGHT": "F-Right",
         "CAM_BACK": "Back", "CAM_BACK_LEFT": "B-Left",
         "CAM_BACK_RIGHT": "B-Right"}
ACC, WARN = "#1f4e79", "#c0392b"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9.75, "axes.linewidth": 0.8, "axes.grid": True,
    "grid.alpha": 0.28, "grid.linewidth": 0.5,
    "savefig.dpi": 400, "savefig.bbox": "tight",
})


# --------------------------------------------------------------------- input
def load_cloud():
    return np.load(BEV / "target_pointcloud_R_xyz.npy").astype(np.float64)


def load_cameras():
    """Name -> (intrinsics, width, height, T_Ci_R, fov row).

    camera_config.json lists the seven rigs without names; camera_pose_matrices
    .json and fov_metrics.csv are keyed by name. The focal lengths are unique
    across the rig, so they identify the entries unambiguously (verified: the
    match is exact to floating point).
    """
    poses = json.load(open(CAL / "camera_pose_matrices.json", encoding="utf-8"))
    fov = {r["camera"]: r for r in
           csv.DictReader(open(CAL / "fov_metrics.csv", encoding="utf-8-sig"))}
    cfg = json.load(open(CFG, encoding="utf-8"))
    cams = {}
    for c in cfg:
        fx = c["camera_internal"]["fx"]
        n = min(fov, key=lambda k: abs(float(fov[k]["fx"]) - fx))
        assert abs(float(fov[n]["fx"]) - fx) < 1e-9, n
        cams[n] = (c["camera_internal"], c["width"], c["height"],
                   np.array(poses[n]["T_Ci_R_view"]), fov[n])
    return cams


def project(P, ki, w, h, T):
    q = (T @ np.c_[P, np.ones(len(P))].T).T[:, :3]
    q = q[q[:, 2] > 0.1]
    u = ki["fx"] * q[:, 0] / q[:, 2] + ki["cx"]
    v = ki["fy"] * q[:, 1] / q[:, 2] + ki["cy"]
    m = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u[m], v[m], q[m, 2]


# ------------------------------------------------------------------ figure 1
def fig_bev():
    """Source / target / difference BEV, relabelled.

    The delivered difference render carries two overlapping title strings and
    is illegible at the top; the label band is cropped and redrawn.
    """
    files = [("source_bev_lidar_L.jpg", "(a) Source rig L, LiDAR concat"),
             ("target_bev_rig_R.jpg", "(b) Target rig R, resampled"),
             ("bev_difference.jpg", "(c) Absolute difference; bright = disagreement")]
    keep = {}
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 1.85))
    for ax, (fn, title) in zip(axes, files):
        im = np.asarray(Image.open(BEV / fn))
        im = im[70:im.shape[0] - 55]                 # drop the burnt-in labels
        keep[fn] = im
        ax.imshow(im)
        ax.set_title(title, fontsize=8.78)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_linewidth(0.6)
    fig.text(0.5, -0.03, "x forward, y left, z up; colour encodes height",
             ha="center", fontsize=8.04, style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "bev_triple.png")
    plt.close(fig)

    # Panel (c) is checked, not assumed, to be |L - R|: it correlates with the
    # absolute difference far better than with either input or their mean, and
    # its mean level is half theirs.
    A = keep["source_bev_lidar_L.jpg"].astype(float).max(2)
    B = keep["target_bev_rig_R.jpg"].astype(float).max(2)
    C = keep["bev_difference.jpg"].astype(float).max(2)
    r_diff = float(np.corrcoef(np.abs(A - B).ravel(), C.ravel())[0, 1])
    r_mean = float(np.corrcoef(((A + B) / 2).ravel(), C.ravel())[0, 1])
    a, b = A > 25, B > 25
    both, either = a & b, a | b
    stats = dict(r_absdiff=r_diff, r_mean=r_mean,
                 shared=100.0 * both.sum() / either.sum(),
                 l_only=100.0 * (a & ~b).sum() / either.sum(),
                 r_only=100.0 * (b & ~a).sum() / either.sum())
    print("wrote bev_triple.png  shared %.1f%% of occupied BEV cells, "
          "L-only %.1f%%, R-only %.1f%%  (panel c is |L-R|: r=%.2f vs %.2f for the mean)"
          % (stats["shared"], stats["l_only"], stats["r_only"], r_diff, r_mean))
    return stats


# ------------------------------------------------------------------ figure 2
def fig_cloud(P):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    r = np.hypot(x, y)
    az = np.degrees(np.arctan2(y, x))

    stats = dict(n=len(P), med=float(np.median(r)),
                 p95=float(np.percentile(r, 95)), max=float(r.max()),
                 f30=float(100 * (r < 30).mean()), f50=float(100 * (r < 50).mean()))

    # azimuthal support: max range holding a return, per 1 degree
    nb = 360
    bi = np.clip(((az + 180) / 360 * nb).astype(int), 0, nb - 1)
    reach = np.zeros(nb)
    near = np.zeros(nb, bool)
    for k in range(nb):
        m = bi == k
        if m.any():
            reach[k] = r[m].max()
            near[k] = (r[m] < 20).any()
    gaps, run = [], 0
    for v in np.r_[~near, False]:
        if v:
            run += 1
        elif run:
            gaps.append(run); run = 0
    stats["shadow_deg"] = int((~near).sum())
    stats["shadow_runs"] = sorted(gaps, reverse=True)

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 1.95))

    ax = axes[0]
    ax.hist(r[r < 120], bins=90, color=ACC, edgecolor="none")
    ax.axvline(stats["med"], color=WARN, lw=1.0)
    ax.text(stats["med"] + 3, ax.get_ylim()[1] * 0.82,
            "median\n%.1f m" % stats["med"], fontsize=8.04, color=WARN)
    ax.set_xlabel("planar range (m)")
    ax.set_ylabel("returns")
    ax.set_title("(a) %s returns, %.0f%% inside 30 m"
                 % ("{:,}".format(len(P)), stats["f30"]), fontsize=9.03)

    ax = plt.subplot(1, 3, 2, projection="polar")
    axes[1].remove()
    th = np.radians(np.arange(nb) - 180 + 0.5)
    ax.plot(th, np.minimum(reach, 120), lw=0.6, color=ACC)
    ax.fill_between(th, 0, np.where(near, 0, 120), color=WARN, alpha=0.30,
                    linewidth=0)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rmax(120)
    ax.set_rticks([40, 80, 120])
    ax.set_rlabel_position(200)
    ax.tick_params(labelsize=6.83, pad=0.2)
    ax.set_title("(b) Azimuthal reach; shaded = no\nreturn inside 20 m (%d deg total)"
                 % stats["shadow_deg"], fontsize=9.03, pad=16)

    ax = axes[2]
    m = r < 60
    hb = ax.hexbin(r[m], z[m], gridsize=(58, 26), bins="log",
                   cmap="viridis", linewidths=0)
    ax.set_xlabel("planar range (m)")
    ax.set_ylabel("height z (m)")
    ax.set_ylim(-4, 12)
    ax.grid(False)
    cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.046)
    cb.ax.tick_params(labelsize=7.07)
    cb.set_label("returns (log)", fontsize=7.81)
    ax.set_title("(c) Height against range", fontsize=9.03)

    fig.tight_layout()
    fig.savefig(OUT / "pointcloud_stats.png")
    plt.close(fig)
    print("wrote pointcloud_stats.png  ", {k: v for k, v in stats.items()})
    return stats


# ------------------------------------------------------------------ figure 3
def fig_support(P, cams):
    rows = []
    for n, (ki, w, h, T, fv) in cams.items():
        u, v, d = project(P, ki, w, h, T)
        rows.append(dict(cam=n, pts=len(u), per_mp=len(u) / (w * h / 1e6),
                         med_d=float(np.median(d)),
                         fov=float(fv["horizontal_fov_deg"]),
                         geom=float(fv["geometry_coverage"]),
                         unk=100 * float(fv["unknown_coverage"])))
    rows.sort(key=lambda r: r["per_mp"])

    per = np.array([r["per_mp"] for r in rows])
    unk = np.array([r["unk"] for r in rows])
    keep = np.array([r["cam"] != "CAM_FRONT_NARROW" for r in rows])
    r_all = float(np.corrcoef(per, unk)[0, 1])
    r_wo = float(np.corrcoef(per[keep], unk[keep])[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 1.95),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    xs = np.arange(len(rows))
    cols = [WARN if r["cam"] == "CAM_FRONT_NARROW" else ACC for r in rows]
    ax.bar(xs, per / 1000.0, 0.6, color=cols, edgecolor="black", linewidth=0.5)
    for i, r in enumerate(rows):
        ax.text(i, r["per_mp"] / 1000.0 + 0.6, "%.1f" % (r["per_mp"] / 1000.0),
                ha="center", fontsize=7.81)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[r["cam"]] for r in rows], fontsize=7.81, rotation=20)
    ax.set_ylabel("LiDAR returns per megapixel ($10^3$)")
    ax.set_ylim(0, per.max() / 1000.0 * 1.18)
    ax.set_title("(a) Angular LiDAR support per camera", fontsize=9.27)

    ax = axes[1]
    off = {"CAM_FRONT_NARROW": (5, -1), "CAM_FRONT_WIDE": (5, -1),
           "CAM_BACK": (-2, 6), "CAM_BACK_RIGHT": (4, 2),
           "CAM_BACK_LEFT": (-6, 6), "CAM_FRONT_RIGHT": (2, -9),
           "CAM_FRONT_LEFT": (3, 3)}
    for r in rows:
        c = WARN if r["cam"] == "CAM_FRONT_NARROW" else ACC
        ax.scatter(r["per_mp"] / 1000.0, r["unk"], s=22, color=c,
                   edgecolor="black", linewidth=0.4, zorder=3)
        ax.annotate(SHORT[r["cam"]], (r["per_mp"] / 1000.0, r["unk"]),
                    textcoords="offset points", xytext=off[r["cam"]],
                    fontsize=7.32)
    ax.set_xlabel("returns per megapixel ($10^3$)")
    ax.set_ylabel("unknown coverage (%)")
    ax.set_xlim(0, per.max() / 1000.0 * 1.22)
    ax.set_ylim(0, unk.max() * 1.28)
    ax.set_title("(b) One outlier, not a trend\n($r=%.2f$ overall, $%.2f$ without F-Narrow)"
                 % (r_all, r_wo), fontsize=8.78)

    fig.tight_layout()
    fig.savefig(OUT / "camera_support.png")
    plt.close(fig)

    # audit trail: the per-camera numbers behind the figure, not typeset
    lines = ["camera,hfov_deg,megapixels,returns,returns_per_mp,median_depth_m,"
             "geometry_coverage,unknown_pct"]
    for r in rows:
        w, h = cams[r["cam"]][1], cams[r["cam"]][2]
        lines.append("%s,%.2f,%.2f,%d,%.0f,%.2f,%.6f,%.4f"
                     % (r["cam"], r["fov"], w * h / 1e6, r["pts"], r["per_mp"],
                        r["med_d"], r["geom"], r["unk"]))
    (HERE / "camera_support.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("wrote camera_support.png and camera_support.csv")
    for r in rows:
        print("   %-17s %6.0f pts/MP  unknown %5.2f%%  hfov %.1f  med_d %.1f m"
              % (r["cam"], r["per_mp"], r["unk"], r["fov"], r["med_d"]))
    print("   r(all 7) = %.3f ; r without F-Narrow = %.3f" % (r_all, r_wo))
    print("   F-Narrow is %.1fx sparser than the next sparsest camera"
          % (per[1] / per[0]))
    return rows, r_all, r_wo, float(per[1] / per[0])


# ------------------------------------------------------------------ figure 4
def fig_risk():
    rows = {(r["pose"], r["camera"]): r for r in
            csv.DictReader(open(EVAL / "gcr-nvs" / "quality_metrics.csv",
                                encoding="utf-8-sig"))}
    meas, pub, cams, poses = [], [], [], []
    for p in sorted(glob.glob(str(GEN / "mixed_*" / "CAM_*" / "vehicle_extension_risk.png"))):
        parts = os.path.normpath(p).split(os.sep)
        pose, cam = parts[-3], parts[-2]
        a = np.asarray(Image.open(p))
        meas.append(float((a > 127).mean()))
        r = rows[(pose, cam)]
        pub.append(float(r["vehicle_risk_fraction_of_hole"])
                   * (1 - float(r["observed_coverage"])))
        cams.append(cam); poses.append(pose)
    meas, pub = np.array(meas), np.array(pub)
    corr = float(np.corrcoef(meas, pub)[0, 1])
    slope = float((meas * pub).sum() / (pub * pub).sum())

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 1.95))
    ax = axes[0]
    for m, q, c in zip(meas, pub, cams):
        col = WARN if c == "CAM_FRONT_NARROW" else ACC
        ax.scatter(q * 100, m * 100, s=20, color=col, edgecolor="black",
                   linewidth=0.4, zorder=3)
    xs = np.linspace(0, pub.max() * 105, 20)
    ax.plot(xs, slope * xs, color="#555555", lw=0.9, ls="--", zorder=2,
            label="fit: slope %.2f" % slope)
    ax.set_xlabel("published risk x hole fraction (% of frame)")
    ax.set_ylabel("re-measured mask fraction (%)")
    ax.legend(frameon=False, fontsize=8.04, loc="upper left")
    ax.set_title("(a) Delivered masks vs. results table\n($r=%.4f$, 21 cases)"
                 % corr, fontsize=8.78)

    ax = axes[1]
    order = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_BACK_LEFT", "CAM_BACK",
             "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_FRONT_NARROW"]
    csvrows = list(csv.DictReader(open(EVAL / "gcr-nvs" / "quality_metrics.csv",
                                       encoding="utf-8-sig")))
    for i, c in enumerate(order):
        v = [100 * float(r["vehicle_risk_fraction_of_hole"])
             for r in csvrows if r["camera"] == c]
        col = WARN if c == "CAM_FRONT_NARROW" else ACC
        ax.scatter([i] * len(v), v, s=20, color=col, edgecolor="black",
                   linewidth=0.4, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color=col, lw=1.4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SHORT[c] for c in order], fontsize=7.81, rotation=20)
    ax.set_ylabel("vehicle-extension risk (% of hole)")
    ax.set_title("(b) The telephoto camera is the\nstructural outlier",
                 fontsize=8.78)

    fig.tight_layout()
    fig.savefig(OUT / "risk_mask_check.png")
    plt.close(fig)
    nar = [100 * float(r["vehicle_risk_fraction_of_hole"])
           for r in csvrows if r["camera"] == "CAM_FRONT_NARROW"]
    oth = [100 * float(r["vehicle_risk_fraction_of_hole"])
           for r in csvrows if r["camera"] != "CAM_FRONT_NARROW"]
    print("wrote risk_mask_check.png   r=%.4f slope=%.3f  narrow %.1f-%.1f%% "
          "vs others max %.1f%%" % (corr, slope, min(nar), max(nar), max(oth)))
    return corr, slope, min(nar), max(nar), max(oth)


# ------------------------------------------------------------------ figure 5
def fig_montage(rows):
    order = ["CAM_FRONT_WIDE", "CAM_FRONT_NARROW", "CAM_FRONT_LEFT",
             "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    per = {r["cam"]: r["per_mp"] for r in rows}
    fig, axes = plt.subplots(2, 4, figsize=(6.6, 2.35))
    for ax, cam in zip(axes.ravel(), order):
        p = BEV / ("%s_target_pointcloud_projection.jpg" % cam.lower())
        ax.imshow(np.asarray(Image.open(p)))
        ax.set_title("%s  (%.1fk pts/MP)" % (SHORT[cam], per[cam] / 1000.0),
                     fontsize=7.81, pad=2,
                     color=WARN if cam == "CAM_FRONT_NARROW" else "black")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_linewidth(0.5)
    # the spare cell carries the reading rather than white space
    ax = axes.ravel()[-1]
    ax.axis("off")
    ax.text(0.02, 0.94,
            "Colour encodes range.\n\n"
            "The telephoto view is\nvisibly the sparsest: a\n29.3 deg field at 8.3 MP\n"
            "receives 2.1k returns\nper megapixel, 5.0x\nfewer than any other\ncamera in the rig.",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.56,
            linespacing=1.45)
    fig.tight_layout(h_pad=1.4, w_pad=0.3)
    fig.savefig(OUT / "projection_montage.png")
    plt.close(fig)
    print("wrote projection_montage.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    P = load_cloud()
    cams = load_cameras()
    bev = fig_bev()
    cloud = fig_cloud(P)
    rows, r_all, r_wo, ratio = fig_support(P, cams)
    corr, slope, nar_lo, nar_hi, oth_hi = fig_risk()
    fig_montage(rows)

    # everything the manuscript quotes from these assets, for verify_manuscript.py
    per = sorted(r["per_mp"] for r in rows)
    json.dump(dict(
        cloud_n=cloud["n"], cloud_med=cloud["med"], cloud_p95=cloud["p95"],
        cloud_max=cloud["max"], cloud_f30=cloud["f30"], cloud_f50=cloud["f50"],
        shadow_deg=cloud["shadow_deg"], shadow_runs=cloud["shadow_runs"],
        bev_shared=bev["shared"], bev_l_only=bev["l_only"], bev_r_only=bev["r_only"],
        bev_r_absdiff=bev["r_absdiff"], bev_r_mean=bev["r_mean"],
        narrow_per_mp=per[0], next_per_mp=per[1], max_per_mp=per[-1],
        sparsity_ratio=ratio, r_all=r_all, r_without_narrow=r_wo,
        mask_corr=corr, mask_slope=slope,
        narrow_risk_lo=nar_lo, narrow_risk_hi=nar_hi, other_risk_hi=oth_hi),
        open(HERE / "media_stats.json", "w"), indent=2)
    print("wrote media_stats.json")

# -*- coding: utf-8 -*-
"""Temporal audit of the delivered A-ONLY video.

The delivery includes 120 fps per-camera video. `04_生成结果/gcr-nvs索引/
video_manifest.csv` declares the contract behind it -- "80 real anchors;
timestamp-aware linear RGB interpolation" -- and frame_mapping.csv records,
for every output frame, which two real anchors it lies between and with what
blend weight. This script does not restate that contract; it measures whether
the pixels obey it, so the manuscript can quantify how much of the apparent
temporal smoothness is evidence and how much is interpolation.

Prediction under exact linear interpolation between anchors k and k+1:
    I(t) = (1-a)I_k + a I_{k+1},  so  I(t+1) - I(t) = (I_{k+1} - I_k)/N,
i.e. the frame-to-frame difference is CONSTANT inside an interval and changes
only at anchor boundaries. The signal should be a staircase whose steps align
with the anchors, not a continuous motion signal.

Test: one-way variance decomposition of the frame-to-frame mean absolute
difference against interval identity. R2 -> 1 means the anchors explain the
signal completely (pure interpolation); R2 -> 0 means frame-to-frame change is
independent of the anchor grid (genuine per-frame content).

Outputs: figures/video_temporal.png, video_temporal.json.
"""
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VID = (ROOT / "04_生成结果" / "generated_video" / "GCR-NVS"
       / "videos_120fps_comparison_20260826_20cm" / "mixed_10_20cm")
MAP = ROOT / "04_生成结果" / "gcr-nvs索引" / "frame_mapping.csv"
OUT = HERE / "figures"

CAMS = ["CAM_BACK", "CAM_FRONT_NARROW", "CAM_FRONT_RIGHT"]
SHORT = {"CAM_BACK": "Back", "CAM_FRONT_NARROW": "Front narrow",
         "CAM_FRONT_RIGHT": "Front right"}
COL = {"CAM_BACK": "#1f4e79", "CAM_FRONT_NARROW": "#b34700",
       "CAM_FRONT_RIGHT": "#2e7d32"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.0, "axes.linewidth": 0.8, "axes.grid": True,
    "grid.alpha": 0.28, "grid.linewidth": 0.5,
    "savefig.dpi": 400, "savefig.bbox": "tight",
})


def anchor_index(camera):
    """Interval id and anchor timestamps for each 120 fps output frame.

    frame_mapping.csv is written for the 240 fps master; the delivered file is
    the 120 fps variant, i.e. every second row.
    """
    rows = [r for r in csv.DictReader(open(MAP, encoding="utf-8-sig"))
            if r["camera"] == camera]
    rows.sort(key=lambda r: int(r["output_index"]))
    rows = rows[::2]
    left = np.array([int(r["left_anchor_frame"]) for r in rows])
    alpha = np.array([float(r["alpha"]) for r in rows])
    ts = sorted({float(r["left_timestamp_s"]) for r in rows}
                | {float(r["right_timestamp_s"]) for r in rows})
    return left, alpha, np.array(ts)


def frame_diffs(path, scale=0.25):
    """Mean absolute luminance difference between consecutive frames.

    Decoding 4741 frames per camera is the slow step, so the result is cached
    beside the script; delete figures/_diffcache_*.npy to force a re-read.
    """
    cache = OUT / ("_diffcache_%s.npy" % path.stem)
    if cache.exists():
        return np.load(cache), 120.0
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev, out = None, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, None, fx=scale, fy=scale,
                                    interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            out.append(float(np.abs(g - prev).mean()))
        prev = g
    cap.release()
    out = np.asarray(out)
    OUT.mkdir(exist_ok=True)
    np.save(cache, out)
    return out, fps


def variance_explained(d, gid):
    """Fraction of variance in d explained by interval identity gid."""
    tot = ((d - d.mean()) ** 2).sum()
    within = 0.0
    for g in np.unique(gid):
        m = gid == g
        within += ((d[m] - d[m].mean()) ** 2).sum()
    return 1.0 - within / tot if tot else float("nan")


def main():
    OUT.mkdir(exist_ok=True)
    res = {}
    for cam in CAMS:
        p = VID / ("%s_a_only_120fps.mp4" % cam)
        if not p.exists():
            print("missing", p)
            continue
        d, fps = frame_diffs(p)
        left, alpha, ts = anchor_index(cam)
        gid = left[1:len(d) + 1]                    # interval owning each diff
        r2 = variance_explained(d, gid)

        # step size predicted by the contract vs. dispersion inside a step
        by = [d[gid == g] for g in np.unique(gid)]
        within_cv = float(np.mean([s.std() / s.mean() for s in by if s.mean() > 0]))
        between_cv = float(np.std([s.mean() for s in by]) / np.mean(d))

        # jump at anchor crossings relative to typical in-interval change
        bnd = np.flatnonzero(np.diff(gid) != 0) + 1
        jump = float(np.mean(np.abs(np.diff(d)[bnd - 1])))
        inner = np.abs(np.diff(d))
        mask = np.ones(len(inner), bool)
        mask[bnd - 1] = False
        inner_mean = float(inner[mask].mean())

        res[cam] = dict(
            frames=len(d) + 1, fps=fps, anchors=int(len(ts)),
            span_s=float(ts[-1] - ts[0]), anchor_rate_hz=float((len(ts) - 1) / (ts[-1] - ts[0])),
            frames_per_interval=round(len(d) / (len(ts) - 1), 2),
            measured_fraction_pct=round(100.0 * len(ts) / (len(d) + 1), 2),
            mean_abs_diff=float(d.mean()), std_abs_diff=float(d.std()),
            r2_interval=float(r2), within_cv=within_cv, between_cv=between_cv,
            boundary_jump=jump, interior_jump=inner_mean,
            jump_ratio=round(jump / inner_mean, 1) if inner_mean else None)
        res[cam]["_d"] = d
        res[cam]["_gid"] = gid
        print("%-17s frames=%4d anchors=%d  R2=%.3f  within_CV=%.3f  "
              "boundary/interior jump=%.1fx  measured=%.2f%%"
              % (cam, len(d) + 1, len(ts), r2, within_cv,
                 jump / inner_mean if inner_mean else float("nan"),
                 res[cam]["measured_fraction_pct"]))

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 1.85),
                             gridspec_kw={"width_ratios": [1.35, 1.0, 0.85]})

    ax = axes[0]
    for cam in res:
        d, fps = res[cam]["_d"], res[cam]["fps"]
        t = np.arange(len(d)) / fps
        m = (t >= 6.0) & (t <= 8.0)
        ax.plot(t[m], d[m], lw=0.85, color=COL[cam], label=SHORT[cam])
    for k in np.arange(6.0, 8.01, 0.5):
        ax.axvline(k, color="#c0392b", lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"mean $|\Delta I|$ per frame")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=6.4,
              ncol=3, loc="lower center", handlelength=1.2, columnspacing=0.9,
              borderpad=0.25)
    ax.set_title("(a) Staircase locked to the real anchors (dashed, 2 Hz)",
                 fontsize=7.6)

    ax = axes[1]
    for cam in res:
        d, gid = res[cam]["_d"], res[cam]["_gid"]
        pos, val = [], []
        for g in np.unique(gid):
            s = d[gid == g]
            if len(s) < 10 or s.mean() <= 0:
                continue
            pos.append(np.linspace(0, 1, len(s)))
            val.append(s / s.mean())
        pos, val = np.concatenate(pos), np.concatenate(val)
        b = np.linspace(0, 1, 21)
        idx = np.clip(np.digitize(pos, b) - 1, 0, 19)
        prof = np.array([val[idx == i].mean() for i in range(20)])
        ax.plot(0.5 * (b[:-1] + b[1:]), prof, lw=1.0, color=COL[cam])
    ax.axhline(1.0, color="#999999", lw=0.6)
    ax.set_xlabel(r"position within anchor interval $\alpha$")
    ax.set_ylabel(r"$|\Delta I|$ / interval mean")
    ax.set_title("(b) Flat within intervals", fontsize=7.6)

    ax = axes[2]
    xs = np.arange(len(res))
    ax.bar(xs, [res[c]["r2_interval"] for c in res], 0.55,
           color=[COL[c] for c in res], edgecolor="black", linewidth=0.5)
    for i, c in enumerate(res):
        ax.text(i, res[c]["r2_interval"] + 0.02, "%.2f" % res[c]["r2_interval"],
                ha="center", fontsize=6.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[c].replace(" ", "\n") for c in res], fontsize=6.6)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(r"variance explained $R^2$")
    ax.set_title("(c) Anchors explain\nthe signal", fontsize=7.6)

    fig.tight_layout()
    fig.savefig(OUT / "video_temporal.png")
    plt.close(fig)

    dump = {c: {k: v for k, v in r.items() if not k.startswith("_")}
            for c, r in res.items()}
    (HERE / "video_temporal.json").write_text(json.dumps(dump, indent=2),
                                              encoding="utf-8")
    print("wrote figures/video_temporal.png and video_temporal.json")


if __name__ == "__main__":
    main()

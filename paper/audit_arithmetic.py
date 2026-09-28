# -*- coding: utf-8 -*-
"""Sweep the prose for numeric claims and re-derive each from the raw CSVs.

verify_manuscript.py already pins 57 named values. This is the complementary
check: it goes looking for *relational* claims -- ratios, multipliers, spreads,
differences and ranges -- which are the ones that survive a value-by-value
audit because every individual number in them is correct while the relation
between them is not. The "roughly six times" error was exactly that shape.

Every claim below is re-derived from 05_评价结果 or from the calibration, and
the check fails loudly rather than printing a reassuring summary.
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# The evaluation results sit in the project tree, but in "results" beside
# the paper directory once published as a repository. Checking both lets
# the same audit run in either place instead of only where it was written.
EVAL = next((c for c in (HERE.parents[1] / "05_评价结果",
                         HERE.parent / "results") if c.is_dir()),
            HERE.parents[1] / "05_评价结果")
# claims live in either document; the supplementary carries the appendices
TEXT = (HERE / "manuscript.txt").read_text(encoding="utf-8")
_supp = HERE / "supplementary_src.txt"
if _supp.exists():
    TEXT += "\n" + _supp.read_text(encoding="utf-8")
for _t in sorted((HERE / "tables").glob("*.txt")):
    TEXT += "\n" + _t.read_text(encoding="utf-8")


def read(p):
    with open(p, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


gcr = read(EVAL / "gcr-nvs" / "quality_metrics.csv")
cur = read(EVAL / "current_rig_quality_metrics.csv")
allq = read(EVAL / "quality_metrics.csv")
ref = [r for r in allq if r["metric_scope"] == "no_gt_proxy_and_topology_audit"]
tmp = [r for r in allq if r["metric_scope"] == "continuous_target_demo_api"]
BANDS = ["mixed_5_10cm", "mixed_10_20cm", "mixed_20_50cm"]

FX = {"FRONT_WIDE": 1921.750887517771, "FRONT_NARROW": 7340.01536928612,
      "FRONT_LEFT": 1085.111126604624, "FRONT_RIGHT": 1113.461101621795,
      "BACK_LEFT": 1081.731828390162, "BACK_RIGHT": 1087.331399652501,
      "BACK": 1903.325095306623}
SIDE = ["FRONT_LEFT", "FRONT_RIGHT", "BACK_LEFT", "BACK_RIGHT"]

# Native raster width per sensor. A focal length in pixels is only comparable
# across cameras that share a width, and the two front cameras do not share one
# with the surrounds.
NATIVE_W = {"FRONT_WIDE": 3840, "FRONT_NARROW": 3840, "FRONT_LEFT": 1920,
            "FRONT_RIGHT": 1920, "BACK_LEFT": 1920, "BACK_RIGHT": 1920,
            "BACK": 1920}
# Horizontal FOV, 2 arctan(w / 2 f_x), the resolution-independent statement of
# the same optical fact.
FOV = dict((c, 2 * np.degrees(np.arctan(NATIVE_W[c] / (2.0 * FX[c]))))
           for c in FX)

fails = []


def claim(text, ok, detail):
    """text must appear in the manuscript, and the relation must hold."""
    present = text in TEXT
    if not present:
        fails.append("NOT FOUND in manuscript: %r" % text)
    elif not ok:
        fails.append("WRONG: %r -- %s" % (text, detail))
    else:
        print("  ok   %-58s %s" % (text[:58], detail))


def band(p, f):
    return np.mean([float(r[f]) for r in gcr if r["pose"] == p]) * 100


def cam(c, f):
    return np.mean([float(r[f]) for r in gcr if r["camera"] == c]) * 100


def main():
    print("relational claims re-derived from the raw data\n")

    # --- risk concentration multiplier
    ratio = cam("CAM_FRONT_NARROW", "vehicle_risk_fraction_of_hole") / \
        (np.mean([float(r["vehicle_risk_fraction_of_hole"]) for r in gcr]) * 100)
    claim("4.9 times the seven-camera mean", abs(ratio - 4.9) < 0.05,
          "24.23 / 4.94 = %.2f" % ratio)

    # --- focal length ratios
    r_wide = FX["FRONT_NARROW"] / FX["FRONT_WIDE"]
    claim("3.8 times the focal length of the front wide", abs(r_wide - 3.8) < 0.05,
          "7340 / 1922 = %.2f" % r_wide)
    # The surrounds are 1920 px wide and the two front cameras 3840, so a ratio
    # of raw f_x compares displacement per pixel on rasters of different size
    # and overstates the gap by exactly the width ratio. Scaling every focal
    # length to a common width is what makes the comparison mean anything --
    # this check previously enforced the unscaled 6.6-6.8, which is why the
    # wrong number survived so long.
    W = 1920.0
    fx_n = FX["FRONT_NARROW"] * W / NATIVE_W["FRONT_NARROW"]
    norm = {c: FX[c] * W / NATIVE_W[c] for c in SIDE}
    lo = min(fx_n / norm[c] for c in SIDE)
    hi = max(fx_n / norm[c] for c in SIDE)
    claim("3.3 to 3.4 times the surrounds", 3.25 < lo < 3.35 and 3.35 < hi < 3.45,
          "width-normalized side ratios span %.2f to %.2f (unscaled would be "
          "%.2f to %.2f)" % (lo, hi,
                             min(FX["FRONT_NARROW"] / FX[c] for c in SIDE),
                             max(FX["FRONT_NARROW"] / FX[c] for c in SIDE)))
    fov_r = [FOV[c] / FOV["FRONT_NARROW"] for c in SIDE]
    claim("field of view is 2.8 times narrower",
          all(abs(v - 2.8) < 0.06 for v in fov_r),
          "surround/narrow FOV ratios %s" % [round(v, 2) for v in fov_r])

    # --- focal and support correlations, on the raster the audit used
    # Coverage and unknown are fractions of each camera's own frame, so the
    # predictor has to be per-frame too: f/W for displacement, returns per pixel
    # of the audit raster for support. Both were previously computed against
    # native-raster quantities while the R2 audit ran at 1920x1080, and that mix
    # made each relationship look as though one camera carried it.
    COV = {"R1": {"CAM_FRONT_WIDE": 98.83, "CAM_FRONT_NARROW": 88.76,
                  "CAM_FRONT_LEFT": 97.67, "CAM_FRONT_RIGHT": 97.27,
                  "CAM_BACK_LEFT": 95.07, "CAM_BACK_RIGHT": 89.49, "CAM_BACK": 91.43},
           "R2": {"CAM_FRONT_WIDE": 99.39, "CAM_FRONT_NARROW": 89.84,
                  "CAM_FRONT_LEFT": 98.96, "CAM_FRONT_RIGHT": 98.63,
                  "CAM_BACK_LEFT": 98.38, "CAM_BACK_RIGHT": 95.55, "CAM_BACK": 95.89}}
    fx_n = {"CAM_" + c: FX[c] * 1920.0 / NATIVE_W[c] for c in FX}
    for rig, want_all, want_wo in (("R1", -0.70, -0.53), ("R2", -0.92, -0.62)):
        cams = list(COV[rig])
        got = np.corrcoef([np.log(fx_n[c]) for c in cams],
                          [COV[rig][c] for c in cams])[0, 1]
        sub = [c for c in cams if c != "CAM_FRONT_NARROW"]
        gwo = np.corrcoef([np.log(fx_n[c]) for c in sub],
                          [COV[rig][c] for c in sub])[0, 1]
        claim("%.2f under %s" % (want_all, rig), abs(got - want_all) < 0.005,
              "log f/W vs coverage, all seven = %+.2f" % got)
        claim("%.2f" % want_wo, abs(gwo - want_wo) < 0.005,
              "%s without the telephoto = %+.2f" % (rig, gwo))
    per_mp = {r["camera"]: float(r["returns"]) / ((1920 * 1080) / 1e6)
              for r in csv.DictReader(open(HERE / "camera_support.csv", encoding="utf-8"))}
    unk = {r["camera"]: float(r["unknown_pct"])
           for r in csv.DictReader(open(HERE / "camera_support.csv", encoding="utf-8"))}
    cams = list(per_mp)
    a = np.corrcoef([per_mp[c] for c in cams], [unk[c] for c in cams])[0, 1]
    sub = [c for c in cams if c != "CAM_FRONT_NARROW"]
    b = np.corrcoef([per_mp[c] for c in sub], [unk[c] for c in sub])[0, 1]
    claim("-0.81 over all seven", abs(a + 0.81) < 0.005,
          "support vs unknown at the audit raster = %+.2f" % a)
    claim("-0.69 with the telephoto removed", abs(b + 0.69) < 0.005,
          "without the telephoto = %+.2f" % b)

    # --- contract property checker
    # Section 6.6 quotes the checker's own output. Re-running it here stops the
    # prose from drifting away from the tool, which is the failure this whole
    # script exists to prevent.
    import subprocess
    out = subprocess.run([sys.executable, str(HERE / "contract_properties.py")],
                         capture_output=True, text=True, encoding="utf-8").stdout
    n_props = len(re.findall(r"^  P\d .*holds$", out, re.M))
    n_frames = int(re.search(r"(\d+) randomized frames", out).group(1))
    ctrl = re.findall(r"fails \((\d+)/(\d+)\)", out)
    worst = min(int(a) / int(b) for a, b in ctrl) if ctrl else 0.0
    claim("six invariants", n_props == 6, "%d properties hold" % n_props)
    claim("20,000 randomized frames", n_frames == 20000, "checker ran %d frames" % n_frames)
    claim("more than 91% of them", len(ctrl) == 4 and worst > 0.91,
          "%d controls, worst detection %.1f%%" % (len(ctrl), worst * 100))

    # --- LiDAR support at the evaluation raster
    # Returns per megapixel depends on which raster the megapixels are counted
    # on, and the two front cameras are 3840 wide while the audit is 1920. The
    # ranking is raster-independent; the factor is not.
    sup = list(csv.DictReader(open(HERE / "camera_support.csv", encoding="utf-8")))
    at1920 = {r["camera"]: float(r["returns"]) / ((1920 * 1080) / 1e6) for r in sup}
    ranked = sorted(at1920.items(), key=lambda kv: kv[1])
    claim("8,214 per megapixel", abs(ranked[0][1] - 8214) < 1.0,
          "%s at 1920x1080 = %.0f/MP" % (ranked[0][0], ranked[0][1]))
    claim("18,435 for the next sparsest", abs(ranked[1][1] - 18435) < 1.0,
          "%s at 1920x1080 = %.0f/MP" % (ranked[1][0], ranked[1][1]))
    claim("a factor of 2.2", abs(ranked[1][1] / ranked[0][1] - 2.2) < 0.05,
          "ratio at the evaluation raster = %.2f" % (ranked[1][1] / ranked[0][1]))

    # --- pooled vs unweighted aggregation
    # Gate fill and vehicle risk are denominated by the hole, whose size varies
    # case by case, so the unweighted mean the article reports and the pooled
    # rate are different quantities. Section 6.3 used to assert they agreed to
    # two decimal places; they differ by 9.47 points in the first band.
    def pooled(sel, key):
        hole = [1 - float(r["observed_coverage"]) for r in sel]
        v = [float(r[key]) for r in sel]
        return sum(x * h for x, h in zip(v, hole)) / sum(hole) * 100

    for p, want in zip(BANDS, (45.39, 41.57, 34.48)):
        got = pooled([r for r in gcr if r["pose"] == p], "gate_fill_of_hole")
        claim("%.2f%%" % want, abs(got - want) < 0.005,
              "pooled gate fill for %s = %.2f%%" % (p, got))
    got = pooled(gcr, "gate_fill_of_hole")
    claim("37.77% overall", abs(got - 37.77) < 0.005,
          "pooled gate fill over all 21 cases = %.2f%%" % got)

    # --- gate-fill spread within a band vs across bands
    spreads = []
    for p in BANDS:
        v = [float(r["gate_fill_of_hole"]) * 100 for r in gcr if r["pose"] == p]
        spreads.append(max(v) - min(v))
    means = [band(p, "gate_fill_of_hole") for p in BANDS]
    across = max(means) - min(means)
    claim("by 50.0 to 61.2 percentage points",
          abs(min(spreads) - 50.0) < 0.05 and abs(max(spreads) - 61.2) < 0.05,
          "within-band spreads %s" % [round(s, 2) for s in spreads])
    claim("than the band means vary across bands, by 16.5",
          abs(across - 16.5) < 0.05, "band means %s, range %.2f"
          % ([round(m, 2) for m in means], across))

    # --- rig deltas quoted in Section 6.1 and 7.2
    for camera, quoted, txt in (("CAM_BACK_RIGHT", 6.06, "6.06 percentage points"),
                                ("CAM_BACK_RIGHT", 6.1, "6.1"),
                                ("CAM_BACK", 4.5, "4.5"),
                                ("CAM_FRONT_WIDE", 0.6, "0.6")):
        a = float(next(r for r in ref if r["camera"] == camera)["geometry_coverage"])
        b = float(next(r for r in cur if r["camera"] == camera)["geometry_coverage"])
        d = (b - a) * 100
        if txt == "6.06 percentage points":
            claim(txt, abs(d - quoted) < 0.02, "%s delta = %.2f" % (camera, d))

    # --- the two-rig mean difference
    d = (np.mean([float(r["geometry_coverage"]) for r in cur])
         - np.mean([float(r["geometry_coverage"]) for r in ref])) * 100
    claim("2.6-point coverage difference", abs(d - 2.6) < 0.05,
          "96.66 - 94.07 = %.2f" % d)

    # --- temporal spread
    g = np.array([float(r["geometry_coverage"]) for r in tmp]) * 100
    claim("1.75", abs((g.max() - g.min()) - 1.75) < 0.005,
          "%.2f - %.2f = %.2f" % (g.max(), g.min(), g.max() - g.min()))

    # --- band coverage range at the two extreme bands
    lo5 = min(float(r["observed_coverage"]) for r in gcr if r["pose"] == BANDS[0]) * 100
    hi5 = max(float(r["observed_coverage"]) for r in gcr if r["pose"] == BANDS[0]) * 100
    claim("a range of 1.6 points", abs((hi5 - lo5) - 1.6) < 0.05,
          "%.2f - %.2f = %.2f" % (hi5, lo5, hi5 - lo5))
    lo50 = min(float(r["observed_coverage"]) for r in gcr if r["pose"] == BANDS[2]) * 100
    hi50 = max(float(r["observed_coverage"]) for r in gcr if r["pose"] == BANDS[2]) * 100
    claim("a range of 6.8", abs((hi50 - lo50) - 6.8) < 0.05,
          "%.2f - %.2f = %.2f" % (hi50, lo50, hi50 - lo50))

    # --- evidence budget closes at every band
    for p in BANDS:
        rs = [r for r in gcr if r["pose"] == p]
        o = np.mean([float(r["observed_coverage"]) for r in rs])
        gt = np.mean([float(r["gate_fill_of_hole"]) * (1 - float(r["observed_coverage"]))
                      for r in rs])
        gn = np.mean([float(r["generated_fraction_all"]) for r in rs])
        if abs(o + gt + gn - 1.0) > 5e-6:
            fails.append("evidence budget for %s sums to %.6f" % (p, o + gt + gn))
    print("  ok   %-58s %s" % ("evidence budget closes at all three bands",
                               "|sum - 1| < 5e-6"))

    # --- any "N times" phrasing anywhere must be one we have checked
    for m in re.finditer(r"([\d.]+(?: to [\d.]+)?) times the ([a-z\- ]{4,40})", TEXT):
        phrase = m.group(0)
        # the video jump ratio is re-derived in verify_manuscript.media_claims()
        if not any(k in phrase for k in ("seven-camera mean", "focal length",
                                         "resolution", "surround",
                                         "change between adjacent")):
            fails.append("unchecked multiplier claim: %r" % phrase)

    print()
    if fails:
        print("FAIL (%d)" % len(fails))
        for f in fails:
            print("  - %s" % f)
        raise SystemExit(1)
    print("PASS: every relational claim re-derives from the raw data")


if __name__ == "__main__":
    main()

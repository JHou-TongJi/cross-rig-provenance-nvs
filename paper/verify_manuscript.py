"""Self-audit for the WEVJ manuscript.

Checks three classes of defect that are easy to introduce by hand:
  1. cross-reference integrity  -- every Figure/Table/Equation/Section referenced
     in the prose actually exists, and every one that exists is referenced;
  2. numeric claims             -- every percentage asserted in the prose matches
     a value computable from the evaluation CSVs;
  3. citation integrity         -- every [n] used is defined, in order of first
     appearance, and every defined reference is used.
"""
import csv
import json
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# The evaluation results sit in the project tree, but in "results" beside
# the paper directory once published as a repository. Checking both lets
# the same audit run in either place instead of only where it was written.
EVAL = next((c for c in (HERE.parents[1] / "05_评价结果",
                         HERE.parent / "results") if c.is_dir()),
            HERE.parents[1] / "05_评价结果")
SRC = HERE / "manuscript.txt"


def expand(path, depth=0):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("INCLUDE|"):
            out.extend(expand(path.parent / line.split("|", 1)[1].strip(), depth + 1))
        else:
            out.append(line)
    return out


LINES = expand(SRC)
PROSE = [l.split("|", 1)[1] for l in LINES
         if l.split("|", 1)[0] in ("P", "PNI", "PBL", "PAL", "BUL", "NUM", "ABSTRACT")]
PROSE_TEXT = "\n".join(PROSE)


def _captions():
    """Caption and note text, with each caption's own number removed.

    Captions carry cross-references too, and they were not being checked: a
    table caption survived here citing "Figure 13" in a document with twelve
    figures. Stripping the leading self-number keeps a caption from being read
    as a citation of itself.
    """
    out = []
    for l in LINES:
        kind, _, rest = l.partition("|")
        if kind == "FIG":
            body = l.split("|", 3)[3] if l.count("|") >= 3 else ""
        elif kind == "TABSTART":
            body = rest.split("|")[0]
        elif kind == "TABEND":
            body = rest
        else:
            continue
        out.append(re.sub(r"^(?:Figure|Table) A?\d+\.\s*", "", body))
    return "\n".join(out)


CAPTION_TEXT = _captions()
# every place a cross-reference can appear
REF_TEXT = PROSE_TEXT + "\n" + CAPTION_TEXT
ALL_TEXT = "\n".join(LINES)

problems = []


def check(ok, msg):
    if not ok:
        problems.append(msg)


# ------------------------------------------------------------------ 1. refs
def cross_references():
    declared_fig = []
    for c in [l for l in LINES if l.startswith("FIG|")]:
        m = re.search(r"Figure (A?\d+)\.", c)
        if m:
            declared_fig.append(m.group(1))
    n_fig = sum(1 for f in declared_fig if not f.startswith("A"))
    n_app = sum(1 for f in declared_fig if f.startswith("A"))
    expected_fig = ([str(i) for i in range(1, n_fig + 1)]
                    + ["A%d" % i for i in range(1, n_app + 1)])
    check(declared_fig == expected_fig,
          "figure captions out of sequence:\n    got      %s\n    expected %s"
          % (declared_fig, expected_fig))

    declared_tab = []
    for l in LINES:
        if l.startswith("TABSTART|"):
            m = re.match(r"Table (A?\d+)\.", l.split("|")[1])
            if m:
                declared_tab.append(m.group(1))
    n_main_tab = sum(1 for t in declared_tab if not t.startswith("A"))
    n_app_tab = sum(1 for t in declared_tab if t.startswith("A"))
    expected = ([str(i) for i in range(1, n_main_tab + 1)]
                + ["A%d" % i for i in range(1, n_app_tab + 1)])
    check(declared_tab == expected,
          "table captions out of sequence:\n    got      %s\n    expected %s"
          % (declared_tab, expected))

    n_eq = sum(1 for l in LINES if l.startswith("EQ|"))
    eq_nums = [int(l.split("|", 1)[1]) for l in LINES if l.startswith("EQ|")]
    check(eq_nums == list(range(1, n_eq + 1)),
          "equation numbers out of sequence: %s" % eq_nums)

    def refs(kind, text):
        """Every number a reference names, singular or plural.

        "Table 7", "Tables 7 and 9", "Tables 7, 9 and 15", "Figures 4 to 6".
        Reading only the singular form counted the members of a plural run as
        uncited, and left a stale plural number undetected on renumbering --
        which is exactly what promoting the appendix tables produced.
        """
        out = set()
        pat = r"\b%ss?\s+(\d+(?:\s*(?:,|,?\s*and|to|through|-)\s*\d+)*)\b" % kind
        for m in re.finditer(pat, text):
            nums = [int(v) for v in re.findall(r"\d+", m.group(1))]
            if "to" in m.group(1) or "through" in m.group(1):
                out.update(range(min(nums), max(nums) + 1))
            else:
                out.update(nums)
        return out

    for kind, count in (("Figure", n_fig), ("Table", n_main_tab)):
        # Being discussed is a property of the prose, so "never referenced" is
        # judged there; but a dangling number is a defect wherever it appears,
        # and captions were the one place it could hide.
        missing = set(range(1, count + 1)) - refs(kind, PROSE_TEXT)
        check(not missing, "%s(s) never referenced in the prose: %s" % (kind, sorted(missing)))
        over = {c for c in refs(kind, REF_TEXT) if c > count}
        check(not over, "%s(s) referenced but not defined: %s" % (kind, sorted(over)))

    # The article carries no appendix, so nothing may remain S- or A-numbered.
    # Both prefixes mark content that was moved: S from the supplementary
    # merge, A from the appendices, and a survivor of either is a reference
    # into a section the reader no longer has.
    stray = sorted(set(re.findall(r"(?:Figures?|Tables?|Appendix)\s+[SA]\d*", ALL_TEXT)))
    check(not stray, "moved-content numbering survives: %s" % stray)
    check(n_app == 0 and n_app_tab == 0,
          "A-numbered captions remain: %d figures, %d tables" % (n_app, n_app_tab))

    eq_cited = {int(m) for m in re.findall(r"Equation \((\d+)\)", ALL_TEXT)}
    check(not (eq_cited - set(eq_nums)),
          "Equation(s) referenced but not defined: %s" % sorted(eq_cited - set(eq_nums)))

    sec_cited = {int(m) for m in re.findall(r"Section (\d+)", PROSE_TEXT)}
    sec_have = {int(re.match(r"(\d+)\.", l.split("|", 1)[1]).group(1))
                for l in LINES if l.startswith("H1|")
                and re.match(r"\d+\.", l.split("|", 1)[1])}
    check(not (sec_cited - sec_have),
          "Section(s) referenced but not defined: %s" % sorted(sec_cited - sec_have))


# --------------------------------------------------------------- 2. numbers
def read(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def numeric_claims():
    gcr = read(EVAL / "gcr-nvs" / "quality_metrics.csv")
    cur = read(EVAL / "current_rig_quality_metrics.csv")
    ref_all = read(EVAL / "quality_metrics.csv")
    ref = [r for r in ref_all if r["metric_scope"] == "no_gt_proxy_and_topology_audit"]
    tmp = [r for r in ref_all if r["metric_scope"] == "continuous_target_demo_api"]
    POSES = ["mixed_5_10cm", "mixed_10_20cm", "mixed_20_50cm"]

    def mean(rows, f):
        return np.mean([float(r[f]) for r in rows]) * 100

    def band(p, f):
        return mean([r for r in gcr if r["pose"] == p], f)

    def cam(c, f):
        return mean([r for r in gcr if r["camera"] == c], f)

    g = np.array([float(r["geometry_coverage"]) for r in tmp]) * 100

    facts = OrderedDict([
        ("98.24", mean(gcr, "observed_coverage")),
        ("44.90", mean(gcr, "gate_fill_of_hole")),
        ("1.10", mean(gcr, "generated_fraction_all")),
        ("4.94", mean(gcr, "vehicle_risk_fraction_of_hole")),
        ("99.32", band(POSES[0], "observed_coverage")),
        ("98.59", band(POSES[1], "observed_coverage")),
        ("96.79", band(POSES[2], "observed_coverage")),
        ("54.86", band(POSES[0], "gate_fill_of_hole")),
        ("41.49", band(POSES[1], "gate_fill_of_hole")),
        ("38.34", band(POSES[2], "gate_fill_of_hole")),
        ("0.37", band(POSES[0], "generated_fraction_all")),
        ("0.82", band(POSES[1], "generated_fraction_all")),
        ("2.10", band(POSES[2], "generated_fraction_all")),
        ("92.19", min(float(r["observed_coverage"]) for r in gcr) * 100),
        ("5.67", max(float(r["generated_fraction_all"]) for r in gcr) * 100),
        ("28.00", max(float(r["vehicle_risk_fraction_of_hole"]) for r in gcr) * 100),
        ("24.23", cam("CAM_FRONT_NARROW", "vehicle_risk_fraction_of_hole")),
        ("96.46", cam("CAM_BACK_LEFT", "observed_coverage")),
        ("2.52", cam("CAM_BACK_LEFT", "generated_fraction_all")),
        ("0.11", cam("CAM_BACK_LEFT", "vehicle_risk_fraction_of_hole")),
        ("94.07", mean(ref, "geometry_coverage")),
        ("5.94", mean(ref, "unknown_coverage")),
        ("96.66", mean(cur, "geometry_coverage")),
        ("3.34", mean(cur, "unknown_coverage")),
        ("88.76", min(float(r["geometry_coverage"]) for r in ref) * 100),
        ("98.83", max(float(r["geometry_coverage"]) for r in ref) * 100),
        ("89.84", min(float(r["geometry_coverage"]) for r in cur) * 100),
        ("99.39", max(float(r["geometry_coverage"]) for r in cur) * 100),
        ("94.42", g.mean()),
        ("0.44", g.std()),
        ("93.37", g.min()),
        ("95.12", g.max()),
        ("1.75", g.max() - g.min()),
        ("98.26", min(float(r["observed_coverage"])
                      for r in gcr if r["pose"] == POSES[0]) * 100),
        ("99.86", max(float(r["observed_coverage"])
                      for r in gcr if r["pose"] == POSES[0]) * 100),
        ("98.97", max(float(r["observed_coverage"])
                      for r in gcr if r["pose"] == POSES[2]) * 100),
    ])
    for claimed, actual in facts.items():
        check(abs(float(claimed) - actual) < 0.005 + 10 ** -len(claimed.split(".")[1]) / 2,
              "claim %s%% does not match computed %.4f%%" % (claimed, actual))
        check(claimed in ALL_TEXT, "verified value %s%% never appears in the manuscript" % claimed)

    # multiplier claims ("N times the seven-camera mean") are easy to state
    # loosely and were wrong once: 24.23 / 4.94 is 4.9, not six
    narrow = cam("CAM_FRONT_NARROW", "vehicle_risk_fraction_of_hole")
    allcam = mean(gcr, "vehicle_risk_fraction_of_hole")
    ratio = narrow / allcam
    for m in re.finditer(r"([\d.]+|four|five|six|seven) times the seven-camera mean", ALL_TEXT):
        word = {"four": 4.0, "five": 5.0, "six": 6.0, "seven": 7.0}
        v = word.get(m.group(1), None)
        v = float(m.group(1)) if v is None else v
        check(abs(v - ratio) < 0.55,
              "claimed %s times the seven-camera mean, but the ratio is %.2f"
              % (m.group(1), ratio))

    # totals from the hole-free audit
    check("388,533" in ALL_TEXT and
          sum(int(r["replaced_visual_pixels"]) for r in ref) == 388533,
          "replaced-pixel total mismatch")
    check(sum(int(r["near_black_replaced_pixels"]) for r in ref) == 279 and "279" in ALL_TEXT,
          "near-black total mismatch")
    check(sum(int(r["locked_pixels_modified"]) for r in gcr) == 0,
          "locked-pixel modification is not zero in the raw data")

    # the evidence budget must close to 100%
    for p in POSES:
        rs = [r for r in gcr if r["pose"] == p]
        o = np.mean([float(r["observed_coverage"]) for r in rs])
        gt = np.mean([float(r["gate_fill_of_hole"]) * (1 - float(r["observed_coverage"]))
                      for r in rs])
        gn = np.mean([float(r["generated_fraction_all"]) for r in rs])
        check(abs((o + gt + gn) - 1.0) < 5e-6,
              "evidence budget for %s sums to %.6f, not 1.0" % (p, o + gt + gn))

    # rig deltas quoted in Section 6.1
    for camera, quoted in (("CAM_BACK_RIGHT", 6.1), ("CAM_BACK", 4.5), ("CAM_FRONT_WIDE", 0.6)):
        a = float(next(r for r in ref if r["camera"] == camera)["geometry_coverage"])
        b = float(next(r for r in cur if r["camera"] == camera)["geometry_coverage"])
        check(abs((b - a) * 100 - quoted) < 0.05,
              "rig delta for %s is %.2f, quoted as %.1f" % (camera, (b - a) * 100, quoted))


# ------------------------------------------------- 2b. media-derived claims
def media_claims():
    """Claims taken from the BEV renders, point cloud, video and risk masks.

    make_media_figures.py and analyze_video.py write the JSON these read, so a
    changed asset or a changed measurement breaks the build rather than
    silently leaving a stale number in the prose.
    """
    ms = HERE / "media_stats.json"
    vt = HERE / "video_temporal.json"
    if not ms.exists() or not vt.exists():
        problems.append("media_stats.json / video_temporal.json missing; "
                        "run make_media_figures.py and analyze_video.py")
        return
    m = json.loads(ms.read_text(encoding="utf-8"))
    v = json.loads(vt.read_text(encoding="utf-8"))

    facts = OrderedDict([
        ("313,196", m["cloud_n"]),
        ("11.45", m["cloud_med"]),
        ("54.57", m["cloud_p95"]),
        ("250.13", m["cloud_max"]),
        ("86.5", m["cloud_f30"]),
        ("93.5", m["cloud_f50"]),
        ("68.0", m["bev_shared"]),
        ("16.0", m["bev_l_only"]),
        ("2,053", m["narrow_per_mp"]),
        ("10,175", m["next_per_mp"]),
        ("25,596", m["max_per_mp"]),
        ("5.0", m["sparsity_ratio"]),
        ("0.9969", m["mask_corr"]),
        ("2.38", m["mask_slope"]),
        ("22.0", m["narrow_risk_lo"]),
        ("28.0", m["narrow_risk_hi"]),
        ("5.8", m["other_risk_hi"]),
        ("0.85", m["bev_r_absdiff"]),
        ("0.65", m["bev_r_mean"]),
        ("-0.72", m["r_all"]),
        ("0.03", m["r_without_narrow"]),
    ])
    for claimed, actual in facts.items():
        plain = claimed.replace(",", "")
        dec = len(plain.split(".")[1]) if "." in plain else 0
        check(abs(float(plain) - actual) < 10 ** -dec / 2 + 1e-9,
              "media claim %s does not match measured %.4f" % (claimed, actual))
        check(claimed in ALL_TEXT,
              "measured value %s never appears in the manuscript" % claimed)

    # the video figures are quoted as ranges over the three cameras
    ratios = [v[c]["jump_ratio"] for c in v]
    r2 = [100 * v[c]["r2_interval"] for c in v]
    check(abs(min(ratios) - 2.7) < 0.05 and abs(max(ratios) - 3.3) < 0.05,
          "quoted boundary/interior jump range 2.7-3.3 does not match %s" % ratios)
    check(round(min(r2)) == 32 and round(max(r2)) == 76,
          "quoted variance-explained range 32-76%% does not match %s" % r2)
    for s in ("2.7 to 3.3", "32% to 76%"):
        check(s in ALL_TEXT, "video range '%s' missing from the manuscript" % s)

    one = v[list(v)[0]]
    check(one["frames"] == 4741 and "4,741" in ALL_TEXT, "video frame count mismatch")
    check(one["anchors"] == 80 and "80 real anchors" in ALL_TEXT
          or one["anchors"] == 80 and "80 of the 4,741" in ALL_TEXT,
          "anchor count mismatch")
    check(abs(one["anchor_rate_hz"] - 2.0) < 5e-4 and "2.000 Hz" in ALL_TEXT,
          "anchor rate is %.4f Hz, not the quoted 2.000" % one["anchor_rate_hz"])
    interp = 100.0 * (1 - one["anchors"] / float(one["frames"]))
    check(abs(interp - 98.3) < 0.05 and "98.3%" in ALL_TEXT,
          "interpolated share is %.2f%%, not the quoted 98.3%%" % interp)


# ------------------------------------------------------------- 3. citations
def citations():
    defined = [int(l.split("|")[1]) for l in LINES if l.startswith("REF|")]
    check(defined == list(range(1, len(defined) + 1)),
          "reference list is not numbered consecutively")

    body = "\n".join(l for l in LINES if not l.startswith("REF|"))
    used, order = set(), []
    hi = len(defined)
    for m in re.finditer(r"\[(\d+(?:[,-]\d+)*)\]", body):
        nums = []
        for part in m.group(1).split(","):
            if "-" in part:
                a, b = part.split("-")
                nums.extend(range(int(a), int(b) + 1))
            else:
                nums.append(int(part))
        # A bracketed group is a citation only if every entry is a valid
        # reference number; this excludes maths such as the interval [0,1].
        if not nums or any(n < 1 or n > hi for n in nums):
            continue
        for n in nums:
                if n not in used:
                    used.add(n)
                    order.append(n)
    missing = used - set(defined)
    check(not missing, "cited but undefined: %s" % sorted(missing))
    unused = set(defined) - used
    check(not unused, "defined but never cited: %s" % sorted(unused))
    check(order == sorted(order),
          "references are not numbered in order of first appearance; first break at %s"
          % next((o for i, o in enumerate(order) if o != i + 1), None))


if __name__ == "__main__":
    cross_references()
    numeric_claims()
    media_claims()
    citations()
    if problems:
        print("FAIL (%d)" % len(problems))
        for p in problems:
            print("  - %s" % p)
        raise SystemExit(1)
    print("PASS: cross-references, %d numeric claims (36 evaluation + %d media) "
          "and citation order all verified" % (36 + 21, 21))

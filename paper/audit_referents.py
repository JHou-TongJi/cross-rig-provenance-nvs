# -*- coding: utf-8 -*-
"""Print every float reference beside the caption it points at.

verify_manuscript.py checks that a referenced number *exists*. It cannot check
that the number points at the right object, and that gap hid a real defect: a
sentence claiming "the geometry-branch results in Tables 6, 8 and 14" named a
metric-definition table and a lineage specification, neither of which is a
result. Every automated check passed, because 6 and 14 are real tables.

So this reports rather than judges. Deciding whether a sentence matches the
caption it cites needs someone who knows what the sentence is claiming, and a
keyword heuristic would both miss real mismatches ("Table 2 states them side by
side") and cry wolf constantly. Printing the pairs makes the check a two-minute
read instead of a manual hunt through the manuscript.

    python audit_referents.py [--kind Table|Figure]
"""
import io
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BODY = ["src_01_front.txt", "src_02_formulation.txt", "src_03_protocol.txt",
        "src_04_discussion.txt", "src_05_back.txt"]
PROSE_TAGS = {"P", "PNI", "PBL", "PAL", "BUL", "NUM", "ABSTRACT"}


def captions(kind):
    """Number -> caption text, read in document order."""
    out, n = {}, 0
    for f in BODY:
        for l in io.open(HERE / f, encoding="utf-8"):
            if kind == "Figure" and l.startswith("FIG|"):
                m = re.match(r"Figure (\d+)\.\s*(.*)", l.split("|", 3)[3])
                if m:
                    out[int(m.group(1))] = m.group(2).strip()
            elif kind == "Table" and l.startswith("INCLUDE|"):
                cap = io.open(HERE / l.split("|", 1)[1].strip(),
                              encoding="utf-8").readline()
                m = re.match(r"Table (\d+)\.\s*(.*)", cap.split("|", 1)[1])
                if m:
                    out[int(m.group(1))] = m.group(2).strip()
        n += 1
    return out


def sentences():
    text = []
    for f in BODY:
        for l in io.open(HERE / f, encoding="utf-8"):
            tag, _, body = l.partition("|")
            if tag in PROSE_TAGS:
                text.append(body.rstrip())
    # split on sentence ends that are not a decimal point or an initial
    return re.split(r"(?<=[.:;])\s+(?=[A-Z*(])", " ".join(text))


def main():
    kinds = ["Table", "Figure"]
    if "--kind" in sys.argv:
        kinds = [sys.argv[sys.argv.index("--kind") + 1]]
    for kind in kinds:
        caps = captions(kind)
        pat = re.compile(r"\b%ss?\s+(\d+(?:\s*(?:,|,?\s*and|to|through)\s*\d+)*)\b"
                         % kind)
        print("\n=== %s references (%d %ss declared) ===" % (kind, len(caps), kind.lower()))
        hits = 0
        for s in sentences():
            ms = list(pat.finditer(s))
            if not ms:
                continue
            hits += 1
            print("\n  %s" % " ".join(s.split())[:200])
            named = []
            for m in ms:
                nums = [int(v) for v in re.findall(r"\d+", m.group(1))]
                if "to" in m.group(1) or "through" in m.group(1):
                    nums = list(range(min(nums), max(nums) + 1))
                named += nums
            for n in sorted(set(named)):
                print("      -> %s %-2d %s"
                      % (kind, n, caps.get(n, "** NOT DECLARED **")[:88]))
        print("\n  %d sentences cite a %s" % (hits, kind.lower()))
    print("\nNothing here is a pass or a fail: read the pairs and judge whether "
          "each sentence\nis describing the object it names.")


if __name__ == "__main__":
    main()

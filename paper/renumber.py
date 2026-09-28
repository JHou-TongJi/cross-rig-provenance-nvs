# -*- coding: utf-8 -*-
"""Recompute figure and table numbers from document order.

Run after any structural edit that inserts, removes or moves a float. The
numbering is derived from where the objects actually sit rather than from a
hand-maintained map, because a map records where I believe they sit and the two
have diverged before.

Captions and references are rewritten in one pass against a single old->new
map, so a shift like 3->2, 4->3 cannot cascade: each occurrence is matched once
and replaced from the map, never re-read.
"""
import io
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BODY = ["src_01_front.txt", "src_02_formulation.txt", "src_03_protocol.txt",
        "src_04_discussion.txt", "src_05_back.txt"]


def read(p):
    return io.open(HERE / p, encoding="utf-8").read().split("\n")


def write(p, lines):
    io.open(HERE / p, "w", encoding="utf-8", newline="\n").write("\n".join(lines))


def table_files():
    return sorted(str(p.relative_to(HERE)).replace("\\", "/")
                  for p in (HERE / "tables").glob("*.txt"))


def order_of(kind):
    """The current number of each object, in the order it now appears."""
    seq = []
    for f in BODY:
        for l in read(f):
            if kind == "Figure" and l.startswith("FIG|"):
                m = re.match(r"Figure (\d+)\.", l.split("|", 3)[3])
                if m:
                    seq.append(m.group(1))
            elif kind == "Table" and l.startswith("INCLUDE|"):
                cap = read(l.split("|", 1)[1].strip())[0]
                m = re.match(r"Table (\d+)\.", cap.split("|", 1)[1])
                if m:
                    seq.append(m.group(1))
    return seq


def main():
    # Only the included tables are live; tables/ also holds files left over
    # from earlier structures, and renumbering those would corrupt nothing but
    # would make the next diff unreadable.
    live = {l.split("|", 1)[1].strip() for f in BODY for l in read(f)
            if l.startswith("INCLUDE|")}
    targets = BODY + sorted(live)

    changed = False
    for kind in ("Figure", "Table"):
        seq = order_of(kind)
        if len(set(seq)) != len(seq):
            sys.exit("%s numbered twice: %s" % (kind, seq))
        remap = {old: str(i + 1) for i, old in enumerate(seq)}
        moved = {o: n for o, n in remap.items() if o != n}
        if moved:
            changed = True
            print("%-6s %d objects, %d renumbered: %s"
                  % (kind, len(seq), len(moved),
                     ", ".join("%s->%s" % kv for kv in sorted(
                         moved.items(), key=lambda kv: int(kv[0])))))
        else:
            print("%-6s %d objects, already in order" % (kind, len(seq)))

        # singular and plural alike: "Table 7", "Tables 7, 9 and 15",
        # "Figures 4 to 6". A plural run is the form that goes stale unnoticed.
        pat = re.compile(r"\b(%ss?)(\s+)(\d+(?:\s*(?:,|,?\s*and|to|through|-)\s*\d+)*)\b"
                         % kind)

        def sub(m):
            run = re.sub(r"\d+", lambda t: remap.get(t.group(0), t.group(0)), m.group(3))
            return m.group(1) + m.group(2) + run

        for f in targets:
            t = pat.sub(sub, "\n".join(read(f)))
            write(f, t.split("\n"))

    # A reference to a number no object declares is the failure this script
    # exists to prevent, so it is checked rather than assumed.
    for kind in ("Figure", "Table"):
        n = len(order_of(kind))
        text = "\n".join("\n".join(read(f)) for f in targets)
        over = set()
        for m in re.finditer(r"\b%ss?\s+(\d+(?:\s*(?:,|,?\s*and|to|through|-)\s*\d+)*)\b"
                             % kind, text):
            over |= {int(v) for v in re.findall(r"\d+", m.group(1)) if int(v) > n}
        if over:
            sys.exit("%s reference past the last object (%d): %s"
                     % (kind, n, sorted(over)))

    print("consistent" if not changed else "renumbered")


if __name__ == "__main__":
    main()

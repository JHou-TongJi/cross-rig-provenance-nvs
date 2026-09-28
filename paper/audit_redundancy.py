# -*- coding: utf-8 -*-
"""Find text that says the same thing twice inside one document.

Heavy prose compression merges and moves paragraphs, and the failure it
produces is not a broken sentence but a duplicated one: the same fact stated
in the body and again in a table note a few lines later. Round 13 left three.
The worst was Table 3, whose note restated the two support states and the
evidence-budget fact from the paragraph twelve lines above it.

Two granularities are needed. Sentence pairs catch a clause copied verbatim.
Block pairs -- whole paragraphs and table notes -- catch duplication spread
over several sentences, where no single pair is similar enough to trip the
sentence threshold; that is exactly how the Table 3 note hid.

Repetition ACROSS the article and the supplementary is not a defect: the
supplementary has to stand on its own, so a definition restated there is
correct. Only same-document pairs fail; cross-document pairs are printed for
inspection.
"""
import io
import re
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
SENT_THRESHOLD = 0.60
BLOCK_THRESHOLD = 0.55
MIN_WORDS = 8

STOP = set("a an the of to in on at by for with and or is are was were be been "
           "that this these those it its as from not no than then so such each "
           "which what when where any all both".split())


def documents():
    """{doc: [(label, text)]}, with each table file attributed to its includer."""
    docs = {}
    for name, files in (("article", sorted(p.name for p in HERE.glob("src_*.txt"))),
                        ("supplementary", ["supplementary_src.txt"])):
        got = []
        for f in files:
            p = HERE / f
            if not p.exists():
                continue
            text = p.read_text(encoding="utf-8")
            got.append((f, text))
            for m in re.finditer(r"INCLUDE\|(tables/\S+\.txt)", text):
                inc = HERE / m.group(1)
                if inc.exists():
                    got.append((m.group(1), inc.read_text(encoding="utf-8")))
        docs[name] = got
    return docs


def blocks(label, text):
    """Yield (label, line, block-text) for prose, captions and table notes."""
    for i, line in enumerate(text.split("\n"), 1):
        tag, _, rest = line.partition("|")
        if tag in ("P", "LI"):
            body = rest
        elif tag == "TABEND":
            body = rest
        elif tag == "FIG":
            body = line.split("|", 3)[3] if line.count("|") >= 3 else ""
        elif tag == "TABSTART":
            body = rest.split("|")[0]
        else:
            continue
        body = re.sub(r"\*(.+?)\*", r"\1", body).strip()
        if len(body.split()) >= MIN_WORDS:
            yield label, i, body


def key(s):
    w = re.findall(r"[a-z]+", s.lower())
    return frozenset(x for x in w if x not in STOP and len(x) > 2)


def pairs(items, threshold):
    keys = [key(s) for _, _, s in items]
    out = []
    for i, j in combinations(range(len(items)), 2):
        a, b = keys[i], keys[j]
        if not a or not b:
            continue
        sim = len(a & b) / float(len(a | b))
        if sim >= threshold:
            out.append((sim, items[i], items[j]))
    return sorted(out, reverse=True, key=lambda h: h[0])


def show(hits, heading):
    if not hits:
        return
    print("%s (%d)" % (heading, len(hits)))
    for sim, (la, ia, sa), (lb, ib, sb) in hits:
        print("  %.2f  %s:%d\n        %s\n        %s:%d\n        %s\n"
              % (sim, la, ia, sa[:140], lb, ib, sb[:140]))


def main():
    docs = documents()
    fails = []

    for doc, files in sorted(docs.items()):
        bl = []
        for label, text in files:
            bl.extend(blocks(label, text))
        st = [(l, i, s) for l, i, b in bl
              for s in re.split(r"(?<=[.!?])\s+", b) if len(s.split()) >= MIN_WORDS]
        print("%s: %d blocks, %d sentences, %d files"
              % (doc, len(bl), len(st), len(files)))

        block_hits = pairs(bl, BLOCK_THRESHOLD)
        sent_hits = [h for h in pairs(st, SENT_THRESHOLD)
                     # a sentence pair inside one block pair is the same finding
                     if not any(h[1][:2] == b[1][:2] and h[2][:2] == b[2][:2]
                                for b in block_hits)]
        show(block_hits, "  duplicated blocks in %s" % doc)
        show(sent_hits, "  duplicated sentences in %s" % doc)
        fails.extend(block_hits + sent_hits)

    a = [b for f in docs["article"] for b in blocks(*f)]
    s = [b for f in docs["supplementary"] for b in blocks(*f)]
    cross = [h for h in pairs(a + s, SENT_THRESHOLD)
             if (h[1] in a) != (h[2] in a)]
    print()
    show(cross, "cross-document repetition (informational; the supplementary "
                "must stand alone)")

    if fails:
        print("FAIL: %d duplicated passage(s) within a single document" % len(fails))
        return 1
    print("PASS: no passage is duplicated within a document")
    return 0


if __name__ == "__main__":
    sys.exit(main())

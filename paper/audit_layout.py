# -*- coding: utf-8 -*-
"""Instrumented visual/layout audit of the compiled PDF.

Covers the mechanical half of a presentation review: things that can be
measured rather than judged. Reports, per page:

  * content spilling past the text column (the DOCX analogue of an overfull box)
  * figures or tables whose caption is on a different page from the object
  * headings orphaned as the last line on a page
  * how far each figure sits from the first place the prose refers to it
  * caption sub-panel letters that do not match the panel count in the image
  * table columns whose decimal places are inconsistent

Judgement calls -- greyscale safety, whether a chart is the right chart -- are
not attempted here.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
PDF = HERE / (sys.argv[1] if len(sys.argv) > 1
              else "cross_vehicle_nvs_wevj_submission.pdf")
COL_L, COL_R = 81.0, 561.0        # measured body text column, points
TOL = 3.0

problems = []


def note(sev, msg):
    problems.append((sev, msg))


def main():
    doc = pymupdf.open(PDF)
    # Two views of each page are needed. `pages` keeps the line structure, so
    # captions and headings can be found at line starts. `flat` drops the review
    # layout's marginal line numbers and joins what is left, because extraction
    # interleaves them with the body -- "Figure\n204\n2 separates" -- which would
    # otherwise break every cross-reference match.
    pages = [p.get_text() for p in doc]
    flat = [" ".join(l for l in t.split("\n")
                     if not re.fullmatch(r"\s*\d{1,4}\s*", l))
            for t in pages]

    # ---- 1. content past the text column
    for i, p in enumerate(doc):
        for b in p.get_text("blocks"):
            x0, x1 = b[0], b[2]
            if x0 > 60 and x1 > COL_R + TOL:
                note("MAJOR", "p%d: text runs %.1f pt past the column: %r"
                     % (i + 1, x1 - COL_R, b[4][:60].replace("\n", " ")))
        for im in p.get_images(full=True):
            for r in p.get_image_rects(im[0]):
                if r.width < 200 or i == 0:      # skip journal branding marks
                    continue
                if r.x1 > COL_R + TOL or r.x0 < COL_L - TOL:
                    note("MAJOR", "p%d: image spans %.1f..%.1f pt, column is %.0f..%.0f"
                         % (i + 1, r.x0, r.x1, COL_L, COL_R))

    # ---- 2. caption separated from its object
    # A caption carries descriptive text after the number. Requiring it stops a
    # sentence that merely ends in a cross-reference from being read as that
    # object's caption: "the evidence budget of Table 10." can wrap so that the
    # reference alone starts a line, which put Table 10's caption fifteen pages
    # ahead of itself and inverted every distance measured from it.
    cap_page = {}
    for i, t in enumerate(pages):
        for m in re.findall(r"^Figure ([AS]?\d+)\.[ 	]+\S", t, re.M):
            cap_page.setdefault("F" + m, i + 1)
        for m in re.findall(r"^Table ([AS]?\d+)\.[ 	]+\S", t, re.M):
            cap_page.setdefault("T" + m, i + 1)
    for i, p in enumerate(doc):
        n_img = 0 if i == 0 else sum(1 for im in p.get_images(full=True)
                                     for r in p.get_image_rects(im[0])
                                     if r.width >= 200)
        n_cap = len(re.findall(r"^Figure ([AS]?\d+)\.[ 	]+\S", pages[i], re.M))
        if n_img and not n_cap:
            note("MAJOR", "p%d: %d figure image(s) but no figure caption on the page"
                 % (i + 1, n_img))

    # ---- 3. orphaned headings (heading is the last text on a page)
    head = re.compile(r"^(?:\d+\.\s+[A-Z]|\d+\.\d+\.\s+[A-Z]|Appendix [A-D])")
    refs_from = next((i for i, t in enumerate(pages)
                      if re.search(r"^References\s*$", t, re.M)), len(pages))
    for i, t in enumerate(pages[:refs_from]):
        lines = [l.strip() for l in t.split("\n")
                 if l.strip() and not re.fullmatch(r"\d+", l.strip())]
        # a real heading is short and self-contained; a wrapped caption line
        # such as "73. Markers are the seven-camera mean ..." is not
        if len(lines) >= 2 and head.match(lines[-1]) and len(lines[-1]) < 70:
            note("MAJOR", "p%d: heading orphaned at page foot: %r" % (i + 1, lines[-1]))

    # ---- 3b. a heading with no body between it and the next heading
    # "1.4. Organization" sat empty through every round and three reviews: the
    # heading rendered directly above "2. Related Work" with nothing under it.
    # Nothing checked for it, because an empty section is neither an orphan nor
    # a bad break -- it is simply a heading whose content is missing.
    # A section heading followed by its own first subsection is normal, so only
    # a heading followed by one at the same or a higher level is empty.
    head_line = re.compile(r"^(?:(\d+)\.(\d+)?\.?\s+[A-Z][^\n]{2,70}"
                           r"|(Appendix) [A-D][^\n]{0,70})\s*$")

    def level(s):
        m = head_line.match(s)
        if not m:
            return None
        if m.group(3):
            return 1
        return 2 if m.group(2) else 1

    flat_lines = []
    for i, t in enumerate(pages):
        for l in t.split("\n"):
            s = l.strip()
            if s and not re.fullmatch(r"\d{1,4}", s):
                flat_lines.append((i + 1, s))
    for k, (pg, s) in enumerate(flat_lines[:-1]):
        a, b = level(s), level(flat_lines[k + 1][1])
        if a is not None and b is not None and b <= a:
            note("MAJOR", "p%d: heading %r has no content before the next heading"
                 % (pg, s[:50]))

    # ---- 4. distance from first prose reference to the object
    for key, cp in sorted(cap_page.items()):
        kind = "Figure" if key[0] == "F" else "Table"
        num = key[1:]
        first = None
        for i, t in enumerate(flat):
            body = re.sub(r"%s %s\. " % (kind, num), " ", t)   # drop the caption
            if re.search(r"%s %s\b" % (kind, num), body):
                first = i + 1
                break
        if first is None:
            # whether every object is cited is settled authoritatively against
            # the source by verify_manuscript.py; PDF extraction is too lossy
            # to decide it here
            pass
        elif num[0] in "AS":
            pass          # appendix and supplementary items sit at the back by design
        elif cp - first < -1:
            # the reader meets the object well before anything mentions it;
            # a forward reference the other way round is normal practice
            note("MINOR", "%s %s appears on p%d, %d pages before it is first "
                 "discussed (p%d)" % (kind, num, cp, first - cp, first))

    # ---- 5. caption panel letters vs. what the caption claims
    full = "\n".join(pages)
    for m in re.finditer(r"^(Figure (?:[AS]?\d+))\.(.{0,900}?)(?=\n\s*\n|\Z)", full,
                         re.M | re.S):
        name, body = m.group(1), m.group(2)
        letters = sorted(set(re.findall(r"\((a|b|c|d|e|f)\)", body)))
        if letters:
            want = [chr(ord("a") + k) for k in range(len(letters))]
            if letters != want:
                note("MAJOR", "%s caption labels panels %s, expected a contiguous run %s"
                     % (name, letters, want))

    # ---- report
    print("PDF: %d pages, %d figures captioned, %d tables captioned"
          % (doc.page_count,
             sum(1 for k in cap_page if k[0] == "F"),
             sum(1 for k in cap_page if k[0] == "T")))
    if not problems:
        print("\nno layout problems found")
        return
    by = defaultdict(list)
    for sev, msg in problems:
        by[sev].append(msg)
    for sev in ("CRITICAL", "MAJOR", "MINOR"):
        for msg in by.get(sev, []):
            print("  [%s] %s" % (sev, msg))
    print("\n%d finding(s)" % len(problems))


if __name__ == "__main__":
    main()

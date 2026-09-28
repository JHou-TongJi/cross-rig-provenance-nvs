# -*- coding: utf-8 -*-
"""Check the rendered DOCX for defects that only exist after the build.

The source audits cannot see these: they are produced by the builder or by
Word, and they show up as text the reader sees but the manuscript never said.

  * unresolved field results ("Error! Reference source not found")
  * replacement glyphs, left by a character the font could not encode
  * CJK, which would mean untranslated drafting text survived
  * internal filesystem paths leaked from a figure or a tool dump
  * literal LaTeX, which means a maths command reached the text run instead of
    the equation -- page 4 once showed a bare T_{c\\leftarroww} this way

Reads the rendered text of every paragraph and table cell, plus the raw
document XML for the field and equation checks.
"""
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
# the article is a single document again; the supplementary was merged back in
DOCS = ["cross_vehicle_nvs_wevj_submission.docx"]

# a backslash followed by letters, as it would appear if a maths command was
# never converted; the OMML itself is XML and carries no backslashes
LATEX = re.compile(r"\\[A-Za-z]{2,}")
PATHS = re.compile(r"(?:[A-Za-z]:\\\\|/media/|/mnt/|/home/|/Users/)\S*")
CJK = re.compile(r"[㐀-鿿豈-﫿！-｠]")


def text_of(docx):
    """Every run of visible text, in document order."""
    with zipfile.ZipFile(docx) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    body = re.sub(r"<w:instrText[^>]*>.*?</w:instrText>", "", xml, flags=re.S)
    runs = re.findall(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", body, re.S)
    out = "".join(runs)
    for a, b in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                 ("&quot;", '"'), ("&apos;", "'")):
        out = out.replace(a, b)
    return out, xml


def main():
    bad = 0
    for name in DOCS:
        p = HERE / name
        if not p.exists():
            print("%s: MISSING" % name)
            bad += 1
            continue
        text, xml = text_of(p)
        found = []

        for label, hits in (
                ("field errors", re.findall(r"Error! [^<]{0,60}", text)),
                ("replacement glyphs", re.findall(r"�", text)),
                ("CJK characters", CJK.findall(text)),
                ("internal paths", PATHS.findall(text)),
                ("literal LaTeX", LATEX.findall(text))):
            if hits:
                found.append("%s: %d %s" % (label, len(hits),
                                            sorted(set(hits))[:4]))

        # Document properties are shown to editors and reviewers and were not
        # being checked at all: the built file carried a title two revisions
        # old, the template's "MDPI" as author, and a local CJK path in the
        # comments. Anything wrong there is wrong in the submission.
        try:
            import docx as _docx
            cp = _docx.Document(str(p)).core_properties
            meta = " | ".join(str(getattr(cp, f, "") or "")
                              for f in ("title", "author", "subject",
                                        "keywords", "comments", "category"))
            for label, hits in (("CJK in document properties", CJK.findall(meta)),
                                ("internal paths in document properties",
                                 PATHS.findall(meta))):
                if hits:
                    found.append("%s: %s" % (label, sorted(set(hits))[:4]))
            if not (cp.title or "").strip():
                found.append("document properties: empty title")
            if (cp.author or "").strip() in ("", "MDPI"):
                found.append("document properties: author still %r" % (cp.author,))
        except Exception as exc:                      # pragma: no cover
            found.append("document properties unreadable: %s" % exc)

        n_omml = len(re.findall(r"<m:oMath[\s>]", xml))
        n_img = len(re.findall(r"<a:blip[\s>]", xml))
        print("%s\n  equations (OMML): %d   images: %d   characters: %d"
              % (name, n_omml, n_img, len(text)))
        for f in found:
            print("  FAIL  %s" % f)
        bad += len(found)
        if not found:
            print("  clean")

    print()
    if bad:
        print("FAIL: %d defect class(es) present" % bad)
        return 1
    print("PASS: free of field, glyph, CJK, path and LaTeX defects")
    return 0


if __name__ == "__main__":
    sys.exit(main())

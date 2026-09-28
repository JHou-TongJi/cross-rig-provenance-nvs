# -*- coding: utf-8 -*-
"""Low-level OOXML helpers: native Word equations (OMML), bookmarks and fields.

Two capabilities the python-docx API does not expose:

* **OMML** — real Word equation objects, produced by converting LaTeX to MathML
  with `latex2mathml` and then through Word's own `MML2OMML.XSL`. The result is
  what Word itself writes when you type an equation, so it is editable in the
  equation editor rather than being text styled to look like maths.
* **Fields** — `SEQ` for automatic figure/table/equation numbering and `REF`
  for cross-references. Each field carries a cached result so the exported PDF
  is correct without the reader pressing F9, and a real field code so Word
  renumbers everything if content is reordered.
"""
import glob
import os
import re

from lxml import etree
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


# --------------------------------------------------------------------------- #
# LaTeX -> OMML
# --------------------------------------------------------------------------- #
def _find_xsl():
    for pat in (r"C:\Program Files\Microsoft Office\root\Office*\MML2OMML.XSL",
                r"C:\Program Files (x86)\Microsoft Office\root\Office*\MML2OMML.XSL",
                r"C:\Program Files\Microsoft Office\Office*\MML2OMML.XSL"):
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    raise SystemExit("MML2OMML.XSL not found; Word must be installed to build "
                     "native equations.")


_XSLT = etree.XSLT(etree.parse(_find_xsl()))
_CACHE = {}


def omml(latex):
    """Convert a LaTeX fragment to an `<m:oMath>` element."""
    if latex not in _CACHE:
        import latex2mathml.converter as conv
        mml = conv.convert(latex)
        root = _XSLT(etree.fromstring(mml.encode("utf-8"))).getroot()
        _CACHE[latex] = etree.tostring(root)
    return etree.fromstring(_CACHE[latex])


def add_omath(paragraph, latex):
    """Append an inline native equation to `paragraph`."""
    paragraph._p.append(omml(latex))
    return paragraph


def add_omath_para(paragraph, latex):
    """Append a display equation, centred by Word's own equation layout."""
    wrap = OxmlElement("m:oMathPara")
    wrap.append(omml(latex))
    paragraph._p.append(wrap)
    return paragraph


# --------------------------------------------------------------------------- #
# bookmarks
# --------------------------------------------------------------------------- #
_BM_ID = [1000]


def bookmark_paragraph(paragraph, name):
    """Wrap a whole paragraph in a bookmark so REF fields can target it."""
    _BM_ID[0] += 1
    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), str(_BM_ID[0]))
    start.set(qn("w:name"), name)
    end = OxmlElement("w:bookmarkEnd")
    end.set(qn("w:id"), str(_BM_ID[0]))
    paragraph._p.insert(0, start)
    paragraph._p.append(end)
    return paragraph


class _BookmarkSpan(object):
    """Bookmark only the runs added inside the `with` block."""

    def __init__(self, paragraph, name):
        self.p, self.name = paragraph, name

    def __enter__(self):
        _BM_ID[0] += 1
        self.id = _BM_ID[0]
        el = OxmlElement("w:bookmarkStart")
        el.set(qn("w:id"), str(self.id))
        el.set(qn("w:name"), self.name)
        self.p._p.append(el)
        return self.p

    def __exit__(self, *exc):
        el = OxmlElement("w:bookmarkEnd")
        el.set(qn("w:id"), str(self.id))
        self.p._p.append(el)
        return False


def bookmark_span(paragraph, name):
    return _BookmarkSpan(paragraph, name)


# --------------------------------------------------------------------------- #
# fields
# --------------------------------------------------------------------------- #
def _field(paragraph, instr, cached, style_run=None):
    """Emit a complete field: begin / instrText / separate / result / end."""
    def run(*children):
        r = OxmlElement("w:r")
        for c in children:
            r.append(c)
        paragraph._p.append(r)
        return r

    fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), "begin")
    run(fc)

    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = instr
    run(it)

    fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), "separate")
    run(fc)

    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = cached
    r = run(t)
    if style_run:
        style_run(r)

    fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), "end")
    run(fc)
    return paragraph


def seq_field(paragraph, seq_name, cached):
    """Automatic number: SEQ Figure / SEQ Table / SEQ Equation."""
    return _field(paragraph, " SEQ %s \\* ARABIC " % seq_name, cached)


def ref_field(paragraph, bookmark, cached, numbered=False):
    """Cross-reference to a bookmark. `numbered` uses \\r for a list number."""
    instr = " REF %s %s\\h " % (bookmark, "\\r " if numbered else "")
    return _field(paragraph, instr, cached)

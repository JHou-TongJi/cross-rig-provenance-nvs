# -*- coding: utf-8 -*-
"""Build the WEVJ (MDPI) submission DOCX from the manuscript source.

The builder starts from the official `wevj_template_as_docx.docx`, strips the
template body while keeping every MDPI_* style and the page/section geometry,
and re-emits the manuscript through those styles.

Three properties the journal asked for are enforced here rather than left to
hand editing:

* **Native Word equations.** Every display equation and every marked-up inline
  symbol becomes a real `<m:oMath>` object, converted from the LaTeX in
  `equations.py` through Word's own MML2OMML.XSL. They open in Word's equation
  editor; they are not text styled to look like maths.
* **Real cross-references.** Figure, table and equation numbers are `SEQ`
  fields; every in-text "Figure N", "Table N", "Equation (N)" and citation
  "[n]" is a `REF` field pointing at a bookmark. Reordering content and
  pressing Ctrl+A F9 renumbers the document consistently.
* **Automatic numbering.** Figures, tables and equations are numbered by Word,
  so the source never hard-codes a number that could drift.

Manuscript source format (one directive per line, `|`-separated):

    ART / TITLE / AUTHORS / AFF / CORR / ABSTRACT / KEYWORDS   front matter
    H1 / H2 / H3            headings          APPENDIX  appendix heading
    P / PNI / PBL / PAL     body paragraphs   BUL / NUM list items
    EQ|<n>                  display equation n, LaTeX from equations.DISPLAY
    FIG|file|width_in|caption
    TABSTART|caption|widths[|align] ... TABHDR / TABROW / TABRULE / TABEND
    BACK|Label: text        NOTE|text         REF|n|text
    INCLUDE|other.txt

Inline markup: `**bold**`, `*italic*`, `$key$` (native equation, LaTeX from
`equations.INLINE`), and `_sub_` / `^sup^`, which are promoted to a native
equation whenever they attach to an identifier.
"""
import datetime
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

import equations
from oxml_helpers import (add_omath, bookmark_paragraph, bookmark_span,
                          ref_field, seq_field)

HERE = Path(__file__).resolve().parent
# The template sits two levels up in the project tree, but beside this script in
# the published repository, where the surrounding directories do not exist.
# Looking next to the script first lets the same build run in both places.
TEMPLATE = next(p for p in (HERE / "wevj_template_as_docx.docx",
                            HERE.parents[1] / "wevj_template_as_docx.docx")
                if p.exists())
# Default is the main manuscript; pass a source and an output to build the
# supplementary document with the same template and styles:
#   python build_wevj_docx.py supplementary_src.txt cross_vehicle_nvs_wevj_supplementary.docx
SOURCE = HERE / (sys.argv[1] if len(sys.argv) > 1 else "manuscript.txt")
FIGDIR = HERE / "figures"
OUTPUT = HERE / (sys.argv[2] if len(sys.argv) > 2
                 else "cross_vehicle_nvs_wevj_submission.docx")

COL_WIDTH_IN = 6.30

# The body styles indent the text block, but MDPI_5.2_figure does not, so a
# centred image lands 23.1 pt left of the text column (measured from the PDF:
# text spans 81.8-559.5, an unindented figure 61.5-533.7). Indenting the figure
# paragraph by twice that offset re-centres it on the text.
FIG_INDENT_PT = 46.2


def _keep(p):
    """Keep a heading with the text that follows it, so it cannot be orphaned
    as the last line on a page."""
    if p is not None:
        p.paragraph_format.keep_with_next = True
    return p


BM_FIG = "_Ref_Fig_%s"
BM_TAB = "_Ref_Tab_%s"
BM_EQ = "_Ref_Eq_%s"
BM_BIB = "_Ref_Bib_%s"


# --------------------------------------------------------------------------- #
# document helpers
# --------------------------------------------------------------------------- #
def clear_body(doc):
    body = doc.element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)


def style_or_normal(doc, name):
    try:
        return doc.styles[name]
    except KeyError:
        return doc.styles["Normal"]


# Unicode maths that may appear inside an auto-detected inline token.
_UNI = {
    "Î": r"\hat{I}", "Ẑ": r"\hat{Z}", "Ĩ": r"\tilde{I}", "μ": r"\mu",
    "θ": r"\theta", "α": r"\alpha", "ℓ": r"\ell", "Σ": r"\sum",
    "Ω": r"\Omega", "Δ": r"\Delta", "π": r"\pi", "←": r"\leftarrow",
    "′": "'", "≠": r"\neq", "∈": r"\in", "·": r"\cdot", "−": "-",
    "×": r"\times", "≤": r"\leq", "≥": r"\geq", "∧": r"\wedge",
}


def _tex_atom(s):
    """Map Unicode maths characters to LaTeX, keeping commands separable.

    A multi-letter command must not run into the character after it: "c←w"
    would otherwise become "c\\leftarroww", which is not a command and renders
    as literal text in the equation. A trailing space terminates the command
    and is ignored by the maths typesetter.
    """
    out = []
    for ch in s:
        tex = _UNI.get(ch, ch)
        if tex.startswith("\\") and tex[-1].isalpha():
            tex += " "
        out.append(tex)
    return "".join(out).strip()


# An identifier with at least one sub/superscript: promoted to a real equation.
TOKEN_MATH = re.compile(
    r"(?<![0-9A-Za-z])"
    r"([A-Za-zÎẐĨμθαℓΣΩΔπ][A-Za-z0-9′']*)"
    r"((?:_[^_\s][^_]*?_(?![0-9A-Za-z])|\^[^\^\s][^\^]*?\^)+)")

INLINE_TOKEN = re.compile(r"(\$[A-Za-z_][A-Za-z0-9_]*\$|\*\*.+?\*\*|\*.+?\*)")


def _token_to_latex(base, mods):
    tex = _tex_atom(base)
    for m in re.finditer(r"_([^_]*?)_(?![0-9A-Za-z])|\^([^\^]*?)\^", mods):
        sub, sup = m.group(1), m.group(2)
        if sub is not None:
            tex += "_{%s}" % _tex_atom(sub)
        else:
            tex += "^{%s}" % _tex_atom(sup)
    return tex


def add_rich_text(doc, paragraph, text, base_italic=False):
    """Add `text`, promoting every marked-up maths token to a native equation."""
    for chunk in INLINE_TOKEN.split(text):
        if not chunk:
            continue
        if chunk.startswith("$") and chunk.endswith("$") and len(chunk) > 2:
            key = chunk[1:-1]
            if key not in equations.INLINE:
                raise SystemExit("unknown inline symbol $%s$" % key)
            add_omath(paragraph, equations.INLINE[key])
            continue
        bold = italic = False
        if chunk.startswith("**") and chunk.endswith("**") and len(chunk) > 4:
            chunk, bold = chunk[2:-2], True
        elif chunk.startswith("*") and chunk.endswith("*") and len(chunk) > 2:
            chunk, italic = chunk[1:-1], True

        pos = 0
        for m in TOKEN_MATH.finditer(chunk):
            if m.start() > pos:
                _plain(paragraph, chunk[pos:m.start()], bold, italic or base_italic)
            add_omath(paragraph, _token_to_latex(m.group(1), m.group(2)))
            pos = m.end()
        if pos < len(chunk):
            _plain(paragraph, chunk[pos:], bold, italic or base_italic)
    return paragraph


def _plain(paragraph, text, bold, italic):
    if not text:
        return
    run = paragraph.add_run(text)
    run.bold = bold or None
    run.italic = italic or None


def add_para(doc, style, text="", italic=False):
    p = doc.add_paragraph(style=style_or_normal(doc, style))
    if text:
        add_rich_text(doc, p, text, base_italic=italic)
    return p


# --------------------------------------------------------------------------- #
# cross-reference substitution in prose
# --------------------------------------------------------------------------- #
XREF = re.compile(r"\b(Figure|Table|Equation)\s+(\(?)(A?\d+)(\)?)|\[(\d+(?:[,–-]\d+)*)\]")

# Populated by build() from the REF| directives, so that a bracketed maths
# interval such as [0,1] is never rewritten into a REF field pointing at a
# reference that does not exist.
VALID_REFS = set()
# Figure and table numbers actually declared in the document being built.
LOCAL_FIG = set()
LOCAL_TAB = set()
LOCAL_EQ = set()


def _is_citation(group):
    """True only if every number in a bracketed group is a real reference."""
    for tok in re.split(r"[,\u2013-]", group):
        if tok and tok not in VALID_REFS:
            return False
    return True


def add_prose(doc, style, text, italic=False):
    """Body paragraph in which every Figure/Table/Equation/[n] becomes a REF field."""
    p = doc.add_paragraph(style=style_or_normal(doc, style))
    pos = 0
    for m in XREF.finditer(text):
        if m.start() > pos:
            add_rich_text(doc, p, text[pos:m.start()], base_italic=italic)
        if m.group(5) and _is_citation(m.group(5)):   # a real citation
            _plain(p, "[", False, italic)
            parts = re.split(r"([,–-])", m.group(5))
            for tok in parts:
                if tok.isdigit():
                    ref_field(p, BM_BIB % tok, tok, numbered=True)
                else:
                    _plain(p, tok, False, italic)
            _plain(p, "]", False, italic)
        elif m.group(5):                     # bracketed maths, not a citation
            add_rich_text(doc, p, m.group(0), base_italic=italic)
        else:
            kind, lp, num, rp = m.group(1), m.group(2), m.group(3), m.group(4)
            local = {"Figure": LOCAL_FIG, "Table": LOCAL_TAB, "Equation": LOCAL_EQ}[kind]
            if num not in local:
                # a reference to the companion document: plain text, no field
                _plain(p, m.group(0), False, italic)
            else:
                bm = {"Figure": BM_FIG, "Table": BM_TAB, "Equation": BM_EQ}[kind]
                _plain(p, "%s %s" % (kind, lp), False, italic)
                ref_field(p, bm % num, num)
                _plain(p, rp, False, italic)
        pos = m.end()
    if pos < len(text):
        add_rich_text(doc, p, text[pos:], base_italic=italic)
    return p


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def set_cell_borders(cell, top=None, bottom=None):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge, size in (("top", top), ("bottom", bottom)):
        node = borders.find(qn("w:%s" % edge))
        if node is None:
            node = OxmlElement("w:%s" % edge)
            borders.append(node)
        if size:
            node.set(qn("w:val"), "single")
            node.set(qn("w:sz"), str(size))
            node.set(qn("w:space"), "0")
            node.set(qn("w:color"), "000000")
        else:
            node.set(qn("w:val"), "nil")


def set_cell_margins(cell, top=40, bottom=40, start=60, end=60):
    tc_pr = cell._tc.get_or_add_tcPr()
    mar = tc_pr.find(qn("w:tcMar"))
    if mar is None:
        mar = OxmlElement("w:tcMar")
        tc_pr.append(mar)
    for edge, val in (("top", top), ("bottom", bottom), ("start", start), ("end", end)):
        node = mar.find(qn("w:%s" % edge))
        if node is None:
            node = OxmlElement("w:%s" % edge)
            mar.append(node)
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")


def repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    node = OxmlElement("w:tblHeader")
    node.set(qn("w:val"), "true")
    tr_pr.append(node)


ALIGN = {"L": WD_ALIGN_PARAGRAPH.LEFT,
         "C": WD_ALIGN_PARAGRAPH.CENTER,
         "R": WD_ALIGN_PARAGRAPH.RIGHT}


class TableSpec(object):
    def __init__(self, caption, widths, align=None, font_pt=None):
        self.caption = caption
        self.widths = widths
        self.font_pt = font_pt
        self.align = align or ("L" + "C" * (len(widths) - 1))
        self.header = None
        self.rows = []
        self.footer = None
        self.pending_rule = False

    def add_row(self, cells):
        self.rows.append((cells, self.pending_rule))
        self.pending_rule = False


def _add_superscript_text(paragraph, text):
    """Render ^...^ spans as real superscript runs.

    The author line carries affiliation indices and the corresponding-author
    asterisk. add_para wrote it verbatim, so the carets reached the PDF as
    literal characters, and add_rich_text is not the alternative: it promotes
    marked-up tokens to native equations, which is right for the body and wrong
    for a name.
    """
    for i, part in enumerate(re.split(r"\^([^^]*)\^", text)):
        if not part:
            continue
        run = paragraph.add_run(part)
        run.font.superscript = bool(i % 2)
    return paragraph


def _set_running_head_year(doc, year):
    """Put the submission year in the MDPI running head.

    The template ships with 2025 baked in as literal text. The sequence was
    recorded in May 2026, the rig configuration is dated August 2026 and one
    reference is from ICLR 2026, so a 2025 running head made the manuscript's
    own chronology impossible -- a reviewer read it as a provenance problem
    rather than a stale template. The page numbers beside it are PAGE/NUMPAGES
    fields, which Word refreshes on repagination, so only the year is set here.
    """
    hit = 0
    for section in doc.sections:
        # MDPI puts the journal citation line in the *footer* of page 1 and in
        # the header of later pages, so walking headers alone left page 1
        # reading 2025 while every other page read 2026.
        for part in (section.header, section.first_page_header,
                     section.even_page_header,
                     section.footer, section.first_page_footer,
                     section.even_page_footer):
            for p in part.paragraphs:
                # Only rewrite runs that carry the digits. Collapsing the
                # paragraph into one run would be simpler but would destroy the
                # PAGE/NUMPAGES field runs sharing this paragraph.
                for run in p.runs:
                    if "2025" in run.text:
                        run.text = run.text.replace("2025", str(year))
                        hit += 1
    return hit


def _retitle_masthead_year(doc, year):
    """The page-1 masthead is body text, not a header, so it needs its own pass."""
    hit = 0
    for p in doc.paragraphs:
        for run in p.runs:
            if "World Electr. Veh. J. 2025" in run.text:
                run.text = run.text.replace("World Electr. Veh. J. 2025",
                                            "World Electr. Veh. J. %d" % year)
                hit += 1
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for run in p.runs:
                        if "World Electr. Veh. J. 2025" in run.text:
                            run.text = run.text.replace(
                                "World Electr. Veh. J. 2025",
                                "World Electr. Veh. J. %d" % year)
                            hit += 1
    return hit


def emit_table(doc, spec, font_pt, number):
    if spec.caption:
        cap = doc.add_paragraph(style=style_or_normal(doc, "MDPI_4.1_table_caption"))
        # Figures already bind their image to their caption; tables did not, so
        # Word was free to leave a caption as the last line of one page and
        # start its table on the next. Table 1 did exactly that.
        cap.paragraph_format.keep_with_next = True
        body = spec.caption.split(". ", 1)[1] if ". " in spec.caption else spec.caption
        _plain(cap, "Table ", False, False)
        with bookmark_span(cap, BM_TAB % number):
            if str(number)[0] in "AS":
                # Appendix / supplementary numbering is a separate series and is
                # written literally rather than driven by the main SEQ counter.
                _plain(cap, str(number), False, False)
            else:
                seq_field(cap, "Table", str(number))
        _plain(cap, ". ", False, False)
        add_rich_text(doc, cap, body)

    ncol = len(spec.widths)
    table = doc.add_table(rows=0, cols=ncol)
    try:
        table.style = doc.styles["MDPI_4.1_three_line_table"]
    except KeyError:
        table.style = doc.styles["Table Grid"]
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    total = sum(spec.widths)
    widths_in = [COL_WIDTH_IN * w / total for w in spec.widths]

    def fill(cells, bold=False, rule_top=0, rule_bot=0):
        row = table.add_row()
        for i, txt in enumerate(cells):
            cell = row.cells[i]
            cell.width = Inches(widths_in[i])
            p = cell.paragraphs[0]
            p.style = style_or_normal(doc, "MDPI_4.2_table_body")
            p.alignment = ALIGN.get(spec.align[i], WD_ALIGN_PARAGRAPH.CENTER)
            add_rich_text(doc, p, txt)
            for run in p.runs:
                run.font.size = Pt(font_pt)
                if bold:
                    run.bold = True
            set_cell_borders(cell, top=rule_top or None, bottom=rule_bot or None)
            set_cell_margins(cell)
        return row

    if spec.header:
        repeat_header(fill(spec.header, bold=True, rule_top=12, rule_bot=6))
    last = len(spec.rows) - 1
    for idx, (cells, rule) in enumerate(spec.rows):
        fill(cells, rule_top=6 if rule else 0, rule_bot=12 if idx == last else 0)
    for i, w in enumerate(widths_in):
        for row in table.rows:
            row.cells[i].width = Inches(w)

    if spec.footer:
        add_prose(doc, "MDPI_4.3_table_footer", spec.footer)
    else:
        add_para(doc, "MDPI_3.3_text_space_after")


# --------------------------------------------------------------------------- #
# source expansion
# --------------------------------------------------------------------------- #
def expand(path, depth=0):
    if depth > 4:
        raise SystemExit("INCLUDE nested too deeply at %s" % path)
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("INCLUDE|"):
            out.extend(expand(path.parent / line.split("|", 1)[1].strip(), depth + 1))
        else:
            out.append(line)
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build():
    doc = Document(str(TEMPLATE))
    _set_running_head_year(doc, 2026)
    clear_body(doc)
    lines = expand(SOURCE)
    # Valid citation numbers, so that a maths interval such as [0,1] is never
    # rewritten into a REF field pointing at a non-existent reference.
    VALID_REFS.clear()
    VALID_REFS.update(l.split("|")[1].strip() for l in lines if l.startswith("REF|"))
    LOCAL_FIG.clear()
    LOCAL_TAB.clear()
    LOCAL_EQ.clear()
    LOCAL_EQ.update(l.split('|', 1)[1].strip() for l in lines if l.startswith('EQ|'))
    for l in lines:
        if l.startswith("FIG|"):
            fm = re.search(r"Figure (S?A?\d+)\.", l)
            LOCAL_FIG.add(fm.group(1) if fm else str(len(LOCAL_FIG) + 1))
        elif l.startswith("TABSTART|"):
            tm = re.match(r"Table (S?A?\d+)\.", l.split("|")[1])
            if tm:
                LOCAL_TAB.add(tm.group(1))

    table = None
    n_fig = n_eq = 0
    tab_seq = 0
    n_tab_total = 0
    doc_title = doc_authors = doc_keywords = ""

    for lineno, raw in enumerate(lines, 1):
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        kind, _, rest = line.partition("|")
        kind = kind.strip()

        if table is not None and kind not in ("TABHDR", "TABROW", "TABRULE", "TABEND"):
            raise SystemExit("line %d: table not closed before %s" % (lineno, kind))

        if kind == "ART":
            add_para(doc, "MDPI_1.1_article_type", rest)
        elif kind == "TITLE":
            doc_title = rest
            add_para(doc, "MDPI_1.2_title", rest)
        elif kind == "AUTHORS":
            doc_authors = re.sub(r"\^[^^]*\^", "", rest).replace(" and ", ", ")
            doc_authors = "; ".join(a.strip(" ,") for a in doc_authors.split(",")
                                    if a.strip(" ,"))
            p = doc.add_paragraph(style=style_or_normal(doc, "MDPI_1.3_authornames"))
            _add_superscript_text(p, rest)
        elif kind in ("AFF", "CORR"):
            add_para(doc, "MDPI_1.6_affiliation", rest)
        elif kind == "ABSTRACT":
            p = doc.add_paragraph(style=style_or_normal(doc, "MDPI_1.7_abstract"))
            p.add_run("Abstract: ").bold = True
            add_rich_text(doc, p, rest)
        elif kind == "KEYWORDS":
            doc_keywords = rest
            p = doc.add_paragraph(style=style_or_normal(doc, "MDPI_1.8_keywords"))
            p.add_run("Keywords: ").bold = True
            add_rich_text(doc, p, rest)
        elif kind == "H1":
            _keep(add_para(doc, "MDPI_2.1_heading1", rest))
        elif kind == "H2":
            _keep(add_para(doc, "MDPI_2.2_heading2", rest))
        elif kind == "H3":
            _keep(add_para(doc, "MDPI_2.3_heading3", rest))
        elif kind == "APPENDIX":
            _keep(add_para(doc, "MDPI_2.1_heading1", rest))
        elif kind == "P":
            add_prose(doc, "MDPI_3.1_text", rest)
        elif kind == "PNI":
            add_prose(doc, "MDPI_3.2_text_no_indent", rest)
        elif kind == "PBL":
            add_prose(doc, "MDPI_3.5_text_before_list", rest)
        elif kind == "PAL":
            add_prose(doc, "MDPI_3.6_text_after_list", rest)
        elif kind == "BUL":
            add_prose(doc, "MDPI_3.8_bullet", rest)
        elif kind == "NUM":
            add_prose(doc, "MDPI_3.7_itemize", rest)

        elif kind == "EQ":
            n_eq += 1
            num = int(rest.strip())
            if num != n_eq:
                raise SystemExit("line %d: equation numbered %d, expected %d"
                                 % (lineno, num, n_eq))
            if num not in equations.DISPLAY:
                raise SystemExit("line %d: no LaTeX for equation %d" % (lineno, num))
            p = doc.add_paragraph(style=style_or_normal(doc, "MDPI_3.9_equation"))
            pf = p.paragraph_format
            pf.tab_stops.add_tab_stop(Inches(COL_WIDTH_IN / 2.0), WD_TAB_ALIGNMENT.CENTER)
            pf.tab_stops.add_tab_stop(Inches(COL_WIDTH_IN), WD_TAB_ALIGNMENT.RIGHT)
            p.add_run("\t")
            add_omath(p, equations.DISPLAY[num])
            p.add_run("\t(")
            with bookmark_span(p, BM_EQ % num):
                seq_field(p, "Equation", str(num))
            p.add_run(")")

        elif kind == "FIG":
            fname, width, caption = (rest.split("|", 2) + ["", ""])[:3]
            n_fig += 1
            path = FIGDIR / fname.strip()
            if not path.exists():
                raise SystemExit("line %d: missing figure %s" % (lineno, path))
            p = doc.add_paragraph(style=style_or_normal(doc, "MDPI_5.2_figure"))
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            # without this Word will break between the image and its caption
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.left_indent = Pt(FIG_INDENT_PT)
            p.add_run().add_picture(str(path), width=Inches(float(width)))
            cap = doc.add_paragraph(style=style_or_normal(doc, "MDPI_5.1_figure_caption"))
            body = caption.split(". ", 1)[1] if ". " in caption else caption
            fm = re.match(r"Figure (S?A?\d+)\.", caption)
            fnum = fm.group(1) if fm else str(n_fig)
            _plain(cap, "Figure ", False, False)
            with bookmark_span(cap, BM_FIG % fnum):
                if fnum[0] in "SA":
                    _plain(cap, fnum, False, False)
                else:
                    seq_field(cap, "Figure", str(n_fig))
            _plain(cap, ". ", False, False)
            add_rich_text(doc, cap, body)

        elif kind == "TABSTART":
            parts = rest.split("|")
            caption, widths = parts[0], parts[1]
            align = (parts[2].strip().upper() or None) if len(parts) > 2 else None
            font = float(parts[3]) if len(parts) > 3 and parts[3].strip() else None
            table = TableSpec(caption, [float(w) for w in widths.split(",")],
                              align, font)
            m = re.match(r"Table (S?A?\d+)\.", caption)
            table.number = m.group(1) if m else None
            if table.number and not table.number.startswith("A"):
                tab_seq += 1
            if table.number:
                n_tab_total += 1
        elif kind == "TABHDR":
            table.header = rest.split("\t")
        elif kind == "TABROW":
            table.add_row(rest.split("\t"))
        elif kind == "TABRULE":
            table.pending_rule = True
        elif kind == "TABEND":
            table.footer = rest.strip() or None
            emit_table(doc, table,
                       table.font_pt or (7.6 if len(table.widths) >= 6 else 8.0),
                       table.number)
            table = None

        elif kind == "BACK":
            p = doc.add_paragraph(style=style_or_normal(doc, "MDPI_6.2_back_matter"))
            label, _, body = rest.partition(":")
            p.add_run(label.strip() + ": ").bold = True
            add_prose_into(doc, p, body.strip())
        elif kind == "NOTE":
            add_para(doc, "MDPI_6.3_notes", rest)
        elif kind == "REF":
            num, _, body = rest.partition("|")
            p = doc.add_paragraph(style=style_or_normal(doc, "MDPI_8.1_references"))
            add_rich_text(doc, p, body.strip())
            bookmark_paragraph(p, BM_BIB % num.strip())
        else:
            raise SystemExit("line %d: unknown directive %r" % (lineno, kind))

    # Document properties are visible to editors and reviewers, and they had
    # drifted: a title two revisions old, the template's "MDPI" as author, a
    # 2025 creation date that fed the chronology complaint, and a local CJK
    # path in the comments. Derive them from the manuscript so they cannot
    # drift again.
    cp = doc.core_properties
    cp.title = doc_title or cp.title
    cp.author = doc_authors or cp.author
    cp.last_modified_by = doc_authors or cp.last_modified_by
    cp.subject = "World Electric Vehicle Journal submission"
    cp.keywords = doc_keywords or ""
    cp.comments = ("Native Word equations (OMML); SEQ/REF cross-references; every numeric "
                   "claim re-derived from the evaluation data by audit_arithmetic.py")
    cp.category = "Article"
    cp.created = cp.modified = datetime.datetime.now(datetime.timezone.utc)
    doc.save(str(OUTPUT))
    print("wrote %s" % OUTPUT)
    print("  figures: %d   tables: %d   equations: %d (all native OMML)"
          % (n_fig, n_tab_total, n_eq))


def add_prose_into(doc, paragraph, text):
    """Cross-reference-aware append into an existing paragraph."""
    pos = 0
    for m in XREF.finditer(text):
        if m.start() > pos:
            add_rich_text(doc, paragraph, text[pos:m.start()])
        if m.group(5) and _is_citation(m.group(5)):
            _plain(paragraph, "[", False, False)
            for tok in re.split(r"([,–-])", m.group(5)):
                if tok.isdigit():
                    ref_field(paragraph, BM_BIB % tok, tok, numbered=True)
                else:
                    _plain(paragraph, tok, False, False)
            _plain(paragraph, "]", False, False)
        elif m.group(5):
            add_rich_text(doc, paragraph, m.group(0))
        else:
            kind, lp, num, rp = m.group(1), m.group(2), m.group(3), m.group(4)
            local = {"Figure": LOCAL_FIG, "Table": LOCAL_TAB, "Equation": LOCAL_EQ}[kind]
            if num not in local:
                _plain(paragraph, m.group(0), False, False)
            else:
                bm = {"Figure": BM_FIG, "Table": BM_TAB, "Equation": BM_EQ}[kind]
                _plain(paragraph, "%s %s" % (kind, lp), False, False)
                ref_field(paragraph, bm % num, num)
                _plain(paragraph, rp, False, False)
        pos = m.end()
    if pos < len(text):
        add_rich_text(doc, paragraph, text[pos:])
    return paragraph


if __name__ == "__main__":
    build()

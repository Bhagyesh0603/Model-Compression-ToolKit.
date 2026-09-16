"""Render Research_Paper_Draft.md as an IEEE-formatted .docx.

Two-column body, Times New Roman, IEEE heading/caption/reference styling. Wide
pre-formatted blocks (the architecture diagram, the algorithm listing) are placed
in full-width single-column sections, which is what IEEE's figure*/table* spans
exist for — at 77 and 89 characters they would need a ~5pt font to fit a 3.5"
column.

    ./.venv/bin/python build_paper_docx.py

Re-run after editing the markdown; the .docx is a build artifact, not a source.
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

SOURCE = Path("Research_Paper_Draft.md")
OUTPUT = Path("Research_Paper_IEEE.docx")

# IEEE conference proceedings geometry (US Letter).
MARGIN_TOP = Inches(0.75)
MARGIN_BOTTOM = Inches(1.0)
MARGIN_SIDE = Inches(0.625)
COLUMN_WIDTH = Inches(3.5)
COLUMN_GAP_TWIPS = "288"  # 0.2"

SERIF = "Times New Roman"
MONO = "Consolas"

AUTHOR_NAME = "Bhagyesh Joshi"
AUTHOR_EMAIL = "bhagyesh.joshi@brainfogagency.com"
AUTHOR_AFFILIATION = "[Department, Institution, City, Country]"

# A block wider than this cannot be read in a 3.5" column at a sane font size.
WIDE_BLOCK_CHARS = 52

# Solid edge-to-edge paragraph background fills, not word-by-word text-highlighter
# marks — a run-level highlight leaves white gaps between words and at the line
# margins; a paragraph shade reads as one continuous block, which is what makes a
# flagged claim easy to spot while skimming.
HIGHLIGHT_COLORS = {
    "green": "C6EFCE",
    "yellow": "FFEB9C",
    "red": "FFC7CE",
}

LEGEND = [
    ("green", "Verified",
     "Confirmed against the source code and, where the claim is one of the paper's four core "
     "contributions, reproduced with an independent adversarial test written to fail if the "
     "claim were false — not just re-read from the code."),
    ("yellow", "Verified, with a caveat the paper states itself",
     "The underlying mechanism is correct, but the surrounding text discloses a real limitation "
     "at this exact location (e.g. a component that is implemented and tested but not yet wired "
     "in automatically). The highlight exists so that disclosure cannot be skimmed past."),
    ("red", "Not yet verified — projected",
     "No measurement exists yet for this claim. It is a literature-informed expectation, and the "
     "surrounding text says so; the highlight flags exactly which sentences that applies to."),
]

# (anchor substring, exactly as it appears in the markdown source; colour;
# appendix title; appendix explanation). Order = order of first appearance in
# the document — this is also the order the appendix lists them in.
CLAIM_AUDIT = [
    ("SIMD-aligned ratio resolution (contribution).", "green",
     "Per-layer SIMD-aligned ratio resolution (§III-E)",
     "Reproduced live, not inferred from the code alone: on a 16-channel layer at align=16, "
     "Torch-Pruning's own global round_to left it fully unpruned (16→16) while this rule "
     "pruned it to 8 and kept a co-located wide layer's alignment at the full 16."),
    ("A residual addition constrains every layer feeding it to share one output channel dimension.",
     "green",
     "Dependency-group ratio unification (§III-F)",
     "Verified on a real ResNet-20: two layers tied by a residual add were given conflicting "
     "overrides (0.2 and 0.7). All three group members — including the one with no override "
     "at all — resolved to the gentler 0.2, not an arbitrary pick and not the harsher value."),
    ("Dual use of the ranking.", "yellow",
     "SQNR sensitivity ranking, dual use (§III-G)",
     "The ranking itself is verified: a network built with a known ground-truth-fragile path (its "
     "logit contribution damped 100× downstream, independent of its own weight scale) was "
     "correctly identified as the more damaging layer, disproving a naive weight-magnitude guess "
     "tried first. The caveat is automation, not correctness — see the next highlight."),
    ("Current degree of automation.", "yellow",
     "Automation gap, disclosed by the paper itself (§III-G)",
     "Only the pruning-ratio half of the SQNR ranking is wired automatically into the budget "
     "search; the quantization-exclusion half is implemented and tested end-to-end but requires "
     "the caller to connect it explicitly. This is the paper's own disclosure of that gap."),
    ("Operator coverage audit (contribution).", "green",
     "Operator coverage audit (§III-I)",
     "Confirmed against the ONNX Runtime source directly: the both-sides QDQ bracketing test "
     "(Eq. 24) is the correct fusion condition, and the claim that reduce_range degrades weight "
     "precision (not activation precision) matches ONNX Runtime's own docstring verbatim."),
    ("Precision gate.** Let", "green",
     "Host-gated precision policy (§III-I)",
     "The gate is real and called before any INT8 path executes (verified in compresskit/core/"
     "host.py and compresskit/quantization/quantize.py), and correctly reports this machine "
     "(Apple M5, i8mm+dotprod) as INT8-fast."),
    ("TO BE POPULATED BY MEASUREMENT", "red",
     "Table II caption — compression sweep (§IV-B)",
     "No sweep has been run yet. Only the FP32 baseline row (93.18% top-1, measured) is real; "
     "every other row is a literature-informed projection."),
    ("Precision gating.** On a host lacking fused integer kernels", "red",
     "Expected effect of precision gating (§IV-D)",
     "The gating mechanism itself is verified correct (green highlight, §III-I above). Its "
     "performance benefit has not been measured on any host in this study — and this machine "
     "has i8mm/dotprod, so it cannot even demonstrate the failure case locally."),
    ("SIMD alignment.** Ablating the per-layer rule", "red",
     "Expected effect of SIMD alignment (§IV-D)",
     "The correctness of this mechanism is verified (green highlight, §III-E above). Its "
     "latency benefit is not yet measured — that requires the align=16 vs. align=1 ablation "
     "described here, which has not been run."),
    ("Dependency-group unification.** On the residual architectures evaluated", "red",
     "Expected effect of dependency-group unification (§IV-D)",
     "The mechanism is verified correct (green highlight, §III-F above). Whether it "
     "measurably changes accuracy or latency on the evaluated architectures is stated here as an "
     "expectation, pending the sweep."),
]

INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*\n]+?\*|`[^`\n]+?`)")
EQUATION = re.compile(r"^(?:&nbsp;)+(.*?)\s*(?:&nbsp;)+\((\d+)\)\s*$")
REFERENCE = re.compile(r"^\[(\d+)\]\s")
FIGURE = re.compile(r"^!\[(.*)\]\((.+?)\)\s*$")


# ─────────────────────────── docx low-level helpers ───────────────────────────


def set_columns(section, count: int) -> None:
    """python-docx exposes no column API; this is the underlying sectPr element."""
    sectPr = section._sectPr
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sectPr.append(cols)
    cols.set(qn("w:num"), str(count))
    cols.set(qn("w:space"), COLUMN_GAP_TWIPS)
    cols.set(qn("w:equalWidth"), "1")


def apply_geometry(section) -> None:
    section.top_margin = MARGIN_TOP
    section.bottom_margin = MARGIN_BOTTOM
    section.left_margin = MARGIN_SIDE
    section.right_margin = MARGIN_SIDE


def switch_columns(doc: Document, count: int):
    """Start a continuous section with `count` columns. Margins do not inherit."""
    section = doc.add_section(WD_SECTION.CONTINUOUS)
    apply_geometry(section)
    set_columns(section, count)
    return section


def add_paragraph(
    doc: Document,
    *,
    size: float = 10,
    align=WD_ALIGN_PARAGRAPH.JUSTIFY,
    bold: bool = False,
    italic: bool = False,
    font: str = SERIF,
    first_line_indent: float | None = None,
    left_indent: float | None = None,
    space_before: float = 0,
    space_after: float = 0,
    keep_with_next: bool = False,
):
    p = doc.add_paragraph()
    fmt = p.paragraph_format
    fmt.alignment = align
    fmt.space_before = Pt(space_before)
    fmt.space_after = Pt(space_after)
    fmt.line_spacing = 1.0
    fmt.keep_with_next = keep_with_next
    if first_line_indent is not None:
        fmt.first_line_indent = Inches(first_line_indent)
    if left_indent is not None:
        fmt.left_indent = Inches(left_indent)
    p._default = {"size": size, "bold": bold, "italic": italic, "font": font}
    return p


def add_run(p, text: str, *, size=None, bold=None, italic=None, font=None):
    d = getattr(p, "_default", {})
    run = p.add_run(text)
    run.font.name = font or d.get("font", SERIF)
    run.font.size = Pt(size if size is not None else d.get("size", 10))
    run.bold = d.get("bold", False) if bold is None else bold
    run.italic = d.get("italic", False) if italic is None else italic
    # East-Asian font must be set explicitly or Word substitutes for unicode math.
    run._element.rPr.rFonts.set(qn("w:eastAsia"), run.font.name)
    return run


def add_formatted(p, text: str) -> None:
    """Emit `text` into `p`, honouring **bold**, *italic* and `code` spans."""
    for token in INLINE.split(unescape(text)):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            add_run(p, token[2:-2], bold=True)
        elif token.startswith("*") and token.endswith("*"):
            add_run(p, token[1:-1], italic=True)
        elif token.startswith("`") and token.endswith("`"):
            add_run(p, token[1:-1], font=MONO, size=9)
        else:
            add_run(p, token)


def unescape(text: str) -> str:
    return text.replace("&nbsp;", " ").replace("—", "—")


# `x_c`, `x_{i,u,v}`, `2^b`, `T^{2}` — a single alphanumeric, or a braced or
# parenthesised group. Restricted to alphanumerics deliberately: a bare `^(`
# would otherwise superscript the parenthesis of an ordinary expression.
SCRIPT = re.compile(r"([_^])(?:\{([^}]*)\}|\(([^)]*)\)|([A-Za-z0-9]))")


def add_scripted(p, text: str, *, bold: bool = False, italic: bool = False) -> None:
    """Emit `text`, promoting _sub / ^super markers to real sub/superscript runs."""
    pos = 0
    for m in SCRIPT.finditer(text):
        if m.start() > pos:
            add_run(p, text[pos:m.start()], bold=bold, italic=italic)
        content = m.group(2) or m.group(3) or m.group(4) or ""
        run = add_run(p, content, bold=bold, italic=italic)
        if m.group(1) == "_":
            run.font.subscript = True
        else:
            run.font.superscript = True
        pos = m.end()
    if pos < len(text):
        add_run(p, text[pos:], bold=bold, italic=italic)


def flatten_braced_emphasis(text: str) -> str:
    """Strip emphasis markers from inside `_{...}` / `^{...}` groups.

    `Σ_{w ∈ **W**_c}` would otherwise be split on the bold markers before the
    script parser ever sees the group, leaving a literal `_{` in the output.
    Emphasis is not meaningful inside a subscript run anyway.
    """
    return re.sub(
        r"([_^])\{([^}]*)\}",
        lambda m: m.group(1) + "{" + m.group(2).replace("*", "") + "}",
        text,
    )


def add_math(p, text: str) -> None:
    """Equation text: inline emphasis plus typeset sub/superscripts."""
    for token in INLINE.split(flatten_braced_emphasis(unescape(text))):
        if not token:
            continue
        if token.startswith("`") and token.endswith("`"):
            add_run(p, token[1:-1], font=MONO, size=9)
        elif token.startswith("**") and token.endswith("**"):
            add_scripted(p, token[2:-2], bold=True)
        elif token.startswith("*") and token.endswith("*"):
            add_scripted(p, token[1:-1], italic=True)
        else:
            add_scripted(p, token)


# ─────────────────────────────── markdown parsing ──────────────────────────────


def parse(markdown: str) -> list[dict]:
    """Flatten the markdown into an ordered list of typed blocks."""
    blocks: list[dict] = []
    lines = markdown.split("\n")
    i = 0
    in_references = False
    paragraph: list[str] = []

    def flush() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.append({"kind": "paragraph", "text": " ".join(paragraph)})
            paragraph = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush()
            i += 1
            body: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                body.append(lines[i])
                i += 1
            blocks.append({"kind": "code", "lines": body})
            i += 1
            continue

        if not stripped or stripped == "---":
            flush()
            i += 1
            continue

        if stripped.startswith("# "):
            flush()
            blocks.append({"kind": "title", "text": stripped[2:]})
        elif stripped.startswith("## "):
            flush()
            heading = stripped[3:]
            in_references = heading.lower().startswith("references")
            blocks.append({"kind": "section", "text": heading})
        elif stripped.startswith("### "):
            flush()
            blocks.append({"kind": "subsection", "text": stripped[4:]})
        elif stripped.startswith("|"):
            flush()
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):  # skip separator row
                    rows.append(cells)
                i += 1
            blocks.append({"kind": "table", "rows": rows})
            continue
        elif FIGURE.match(stripped):
            flush()
            m = FIGURE.match(stripped)
            blocks.append({"kind": "figure", "caption": m.group(1), "path": m.group(2)})
        elif EQUATION.match(stripped):
            flush()
            m = EQUATION.match(stripped)
            blocks.append({"kind": "equation", "text": m.group(1), "number": m.group(2)})
        elif re.match(r"^\*\*TABLE\b.*\*\*$", stripped):
            flush()
            blocks.append({"kind": "caption", "text": stripped.strip("*")})
        elif stripped.startswith("**Keywords:**"):
            flush()
            blocks.append({"kind": "keywords", "text": stripped[len("**Keywords:**"):].strip()})
        elif in_references and REFERENCE.match(stripped):
            flush()
            entry = [stripped]
            i += 1
            while i < len(lines) and lines[i].strip() and not REFERENCE.match(lines[i].strip()):
                entry.append(lines[i].strip())
                i += 1
            blocks.append({"kind": "reference", "text": " ".join(entry)})
            continue
        elif re.match(r"^(\d+\.|[-*])\s", stripped):
            flush()
            item = re.sub(r"^(\d+\.|[-*])\s", "", stripped)
            marker = stripped.split()[0] if stripped[0].isdigit() else "•"
            entry = [item]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(
                r"^(\d+\.|[-*]|#|\||&nbsp;|```)", lines[i].strip()
            ):
                entry.append(lines[i].strip())
                i += 1
            blocks.append({"kind": "list", "text": " ".join(entry), "marker": marker})
            continue
        else:
            paragraph.append(stripped)

        i += 1

    flush()
    return blocks


# ──────────────────────────────── docx emission ────────────────────────────────


def emit_title(doc: Document, text: str) -> None:
    p = add_paragraph(doc, size=24, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)
    add_run(p, text, size=24)

    p = add_paragraph(doc, size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_run(p, AUTHOR_NAME, size=11)
    p = add_paragraph(doc, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True)
    add_run(p, AUTHOR_AFFILIATION, size=10, italic=True)
    p = add_paragraph(doc, size=9, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=14)
    add_run(p, AUTHOR_EMAIL, size=9)


def emit_section(doc: Document, text: str) -> None:
    """IEEE section headings are centred, small-caps-styled, roman-numbered."""
    p = add_paragraph(
        doc, size=10, align=WD_ALIGN_PARAGRAPH.CENTER,
        space_before=12, space_after=4, keep_with_next=True,
    )
    add_run(p, text.upper(), size=10)


def emit_subsection(doc: Document, text: str) -> None:
    p = add_paragraph(
        doc, size=10, align=WD_ALIGN_PARAGRAPH.LEFT, italic=True,
        space_before=8, space_after=3, keep_with_next=True,
    )
    add_formatted(p, text)


def emit_abstract(doc: Document, text: str) -> None:
    p = add_paragraph(doc, size=9, bold=True, italic=True, space_after=6)
    add_run(p, "Abstract—", size=9, bold=True, italic=True)
    add_formatted(p, text)


def emit_keywords(doc: Document, text: str) -> None:
    p = add_paragraph(doc, size=9, bold=True, italic=True, space_after=10)
    add_run(p, "Index Terms—", size=9, bold=True, italic=True)
    add_formatted(p, text)


def emit_equation(doc: Document, text: str, number: str) -> None:
    """Centre tab for the expression, right tab for the number — IEEE layout."""
    p = add_paragraph(doc, size=10, align=WD_ALIGN_PARAGRAPH.LEFT,
                      space_before=4, space_after=4)
    tabs = p.paragraph_format.tab_stops
    tabs.add_tab_stop(Inches(1.75), WD_TAB_ALIGNMENT.CENTER)
    tabs.add_tab_stop(COLUMN_WIDTH, WD_TAB_ALIGNMENT.RIGHT)
    add_run(p, "\t")
    add_math(p, text)
    add_run(p, "\t(" + number + ")")


def emit_caption(doc: Document, text: str):
    p = add_paragraph(doc, size=8, align=WD_ALIGN_PARAGRAPH.CENTER,
                      space_before=8, space_after=3, keep_with_next=True)
    add_run(p, text, size=8)
    return p


def _shd(hex_fill: str):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    return shd


def shade_paragraph(paragraph, color_key: str) -> None:
    """Solid background fill across the full paragraph width, every wrapped
    line — the block-highlight look, not a per-word text-highlighter mark."""
    paragraph._p.get_or_add_pPr().append(_shd(HIGHLIGHT_COLORS[color_key]))


def shade_run(run, color_key: str) -> None:
    """Solid fill tightly around one run only — used for inline badge labels
    (the appendix legend swatches and numbered tags) rather than full claims."""
    run._element.get_or_add_rPr().append(_shd(HIGHLIGHT_COLORS[color_key]))


def apply_highlight(paragraph, color_key: str) -> None:
    shade_paragraph(paragraph, color_key)


def highlight_table_cells(table, keywords: tuple[str, ...]) -> bool:
    """Shade any cell whose text contains one of `keywords`. Returns whether
    anything matched, so the caller only records an appendix entry when it did."""
    matched = False
    for row in table.rows:
        for cell in row.cells:
            if any(k in cell.text.lower() for k in keywords):
                cell._tc.get_or_add_tcPr().append(_shd(HIGHLIGHT_COLORS["red"]))
                matched = True
    return matched


def emit_table(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r, row in enumerate(rows):
        for c, cell_text in enumerate(row[: len(rows[0])]):
            cell = table.cell(r, c)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            p._default = {"size": 7.5, "bold": r == 0, "italic": False, "font": SERIF}
            add_formatted(p, cell_text)
    add_paragraph(doc, size=6, space_after=4)


def emit_code(doc: Document, lines: list[str], column_width_inches: float) -> None:
    """Monospace block, font scaled so the widest line fits the available width."""
    body = [ln.rstrip() for ln in lines]
    while body and not body[0]:
        body.pop(0)
    while body and not body[-1]:
        body.pop()
    if not body:
        return

    widest = max(len(ln) for ln in body)
    # Consolas advance width is ~0.55em; solve for the size that fits.
    size = min(9.0, (column_width_inches * 72) / (widest * 0.55)) if widest else 9.0
    size = max(5.5, round(size * 2) / 2)

    for line in body:
        p = add_paragraph(doc, size=size, align=WD_ALIGN_PARAGRAPH.LEFT, font=MONO)
        p.paragraph_format.line_spacing = 1.0
        add_run(p, line if line else " ", size=size, font=MONO)
    add_paragraph(doc, size=6, space_after=4)


def emit_figure(doc: Document, path: str, caption: str, number: int, width_inches: float) -> None:
    """Centred image with an IEEE 'Fig. N.' caption beneath it."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"figure not found: {source}")

    p = add_paragraph(doc, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=6, space_after=3)
    p.add_run().add_picture(str(source), width=Inches(width_inches))

    cap = add_paragraph(doc, size=8, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8)
    add_run(cap, f"Fig. {number}.  ", size=8)
    add_formatted(cap, caption)


def emit_list(doc: Document, text: str, marker: str) -> None:
    p = add_paragraph(doc, size=10, left_indent=0.2, first_line_indent=-0.12,
                      space_after=2)
    add_run(p, marker + " ")
    add_formatted(p, text)


def emit_reference(doc: Document, text: str) -> None:
    p = add_paragraph(doc, size=8, left_indent=0.18, first_line_indent=-0.18,
                      space_after=2)
    add_formatted(p, text)


def build() -> Path:
    blocks = parse(SOURCE.read_text(encoding="utf-8"))

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = SERIF
    normal.font.size = Pt(10)

    first = doc.sections[0]
    apply_geometry(first)
    set_columns(first, 1)  # title block spans the page

    columns = 1
    body_started = False
    expect_abstract = False
    figure_number = 0
    appendix_entries: list[tuple[str, str, str]] = []
    consumed_anchors: set[str] = set()

    def audit_check(raw_text: str, paragraph) -> None:
        """Highlight `paragraph` and queue an appendix entry if `raw_text`
        (the pre-formatting markdown, asterisks and all) contains one of the
        claim-audit anchors. Each anchor fires at most once."""
        for anchor, color, title, explanation in CLAIM_AUDIT:
            if anchor in raw_text and anchor not in consumed_anchors:
                consumed_anchors.add(anchor)
                apply_highlight(paragraph, color)
                appendix_entries.append((color, title, explanation))
                return

    for block in blocks:
        kind = block["kind"]

        # Wide pre-formatted blocks span both columns; everything else is 2-up.
        if kind == "code":
            wide = max((len(ln) for ln in block["lines"]), default=0) > WIDE_BLOCK_CHARS
            want = 1 if wide else 2
        elif kind == "figure":
            want = 1  # the architecture diagram is unreadable at 3.5" column width
        elif not body_started:
            want = 1
        else:
            want = 2

        if body_started and want != columns:
            switch_columns(doc, want)
            columns = want

        if kind == "title":
            emit_title(doc, block["text"])
            switch_columns(doc, 2)  # body begins here, two columns
            columns, body_started = 2, True
        elif kind == "section":
            heading = block["text"]
            if heading.lower().startswith("abstract"):
                expect_abstract = True
                continue  # IEEE prints no "Abstract" heading; the run label carries it
            emit_section(doc, heading)
        elif kind == "subsection":
            emit_subsection(doc, block["text"])
        elif kind == "paragraph":
            if expect_abstract:
                emit_abstract(doc, block["text"])
                expect_abstract = False
            else:
                p = add_paragraph(doc, size=10, first_line_indent=0.2, space_after=3)
                add_formatted(p, block["text"])
                audit_check(block["text"], p)
        elif kind == "keywords":
            emit_keywords(doc, block["text"])
        elif kind == "equation":
            emit_equation(doc, block["text"], block["number"])
        elif kind == "caption":
            cap = emit_caption(doc, block["text"])
            audit_check(block["text"], cap)
        elif kind == "table":
            emit_table(doc, block["rows"])
            table = doc.tables[-1]
            if highlight_table_cells(table, ("expect", "to measure", "to compute")):
                appendix_entries.append((
                    "red",
                    "Table II data cells — projected values (§IV-B)",
                    "Every highlighted cell states an expectation, not a measurement, matching "
                    "the caption's own disclosure that the table is pending the compression sweep.",
                ))
        elif kind == "figure":
            figure_number += 1
            emit_figure(doc, block["path"], block["caption"], figure_number,
                        7.25 if columns == 1 else 3.5)
        elif kind == "code":
            width = 7.25 if columns == 1 else 3.5
            emit_code(doc, block["lines"], width)
        elif kind == "list":
            emit_list(doc, block["text"], block["marker"])
        elif kind == "reference":
            emit_reference(doc, block["text"])

    emit_audit_appendix(doc, appendix_entries)

    doc.save(OUTPUT)
    return OUTPUT


def emit_audit_appendix(doc: Document, entries: list[tuple[str, str, str]]) -> None:
    """Single-column appendix explaining every highlighted passage in the body,
    in the order it was encountered, plus the three-colour legend."""
    switch_columns(doc, 1)

    heading = add_paragraph(doc, size=12, align=WD_ALIGN_PARAGRAPH.CENTER,
                            space_before=18, space_after=8, bold=True)
    add_run(heading, "APPENDIX: COLOUR-CODED CLAIM AUDIT", size=12, bold=True)

    intro = add_paragraph(doc, size=9, space_after=8)
    add_run(intro, "Highlighted passages above are colour-coded by verification status, checked "
                   "against the compresskit source and, for the paper's core contributions, "
                   "against an independent adversarial test. Colours:")

    for color, label, description in LEGEND:
        p = add_paragraph(doc, size=9, left_indent=0.15, space_after=5)
        swatch = add_run(p, "  " + label + "  ", size=9, bold=True)
        shade_run(swatch, color)
        add_run(p, "  —  " + description, size=9)

    add_paragraph(doc, size=4, space_after=4)  # spacer

    for i, (color, title, explanation) in enumerate(entries, start=1):
        p = add_paragraph(doc, size=9.5, space_before=8, space_after=2, bold=True)
        tag = add_run(p, f"  {i}. {title}  ", size=9.5, bold=True)
        shade_run(tag, color)

        body = add_paragraph(doc, size=9, left_indent=0.2, space_after=4)
        add_run(body, explanation, size=9)


if __name__ == "__main__":
    path = build()
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")

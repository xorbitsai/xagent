---
name: docx-report-editorial
description: |
  A native .docx styled as an editorial report: "Word report", "Word 文档",
  "memo", "proposal", "研究报告". Opens cleanly in Word / Google Docs /
  Pages with real heading styles and a cover page.
when_to_use: |
  A polished .docx meant to be read as a document and further edited in
  Word. Use pdf-report-editorial when the deliverable is a fixed printable
  PDF, and pptx-editorial when the user wants slides.
tags:
  - docx
  - word
  - report
  - document
  - editorial
---

# Editorial Report (.docx)

You will generate one `.docx` file via python-docx by writing a Python
program and running it through the `execute_python_code` tool. The blocks
below are parts of that one program, not separate runs -- later ones use the
`doc`, `palette` and helpers the earlier ones define, so assemble them into a
single script before executing. Save to
the workspace, then report the path + a 1-line content summary.

## 📦 Required runtime packages

The sandboxed `execute_python_code` ships with `pandas`, `numpy`,
`matplotlib`, `openpyxl`, and **`python-docx>=1.1.0`** preinstalled — no
extra installation step is needed. The import name is **`docx`**, not
`python_docx`:

```python
from docx import Document          # ✅ correct
import python_docx                 # ❌ ModuleNotFoundError
```

> **Note:** `execute_python_code` accepts only `code` and
> `capture_output` arguments; there is no `packages` parameter.
> All required libraries are already available in the sandbox image.

## 💾 Saving and reporting the file

`execute_python_code` runs with the task's output directory as the working
directory, so save with a **plain filename** — no `/workspace`, no nested
`output/`, no BytesIO round-trip:

```python
doc.save("market_expansion_report.docx")
```

Then verify content, not size: an empty `Document()` is already ~36 KB, so a
byte threshold proves nothing. Re-open and check what you wrote.

```python
from docx import Document

check = Document("market_expansion_report.docx")
# Cell text is not in check.paragraphs, so a table-only document would
# otherwise look empty and you would report a failure that did not happen.
has_text = any(p.text.strip() for p in check.paragraphs) or any(
    cell.text.strip()
    for tbl in check.tables
    for row in tbl.rows
    for cell in row.cells
)
assert has_text, "document has no text"
```

Assert the shape the request asked for, never a fixed minimum — a multi-section
report can also check its tables and headings, but a one-page memo has neither,
and padding one to clear a threshold breaks rule 5.

### 🔗 Make it clickable — REQUIRED

The executor response carries a `markdown_link` for each file it wrote (in
`file_refs[]`). Use that string verbatim as the **first line** of your answer,
as bare markdown:

    [market_expansion_report.docx](file:20fae785-3823-4906-b385-d0e8a7807dc8)

Never fabricate the UUID, never wrap the link in backticks (it stops rendering
as a chip), and never restate it as a `file_id:` field. `get_file_info()` does
not return one — the reference is already on the executor result.

## ⚠️ Hard rules — NO exceptions

0. **MATCH THE USER'S LANGUAGE.** If the prompt is Chinese (中文), ALL
   document text (cover kicker, headings, body, table headers, captions,
   footer) must be in Chinese. Translate template phrases like
   `EXECUTIVE SUMMARY` → `摘要`, `FINDINGS` → `调查结果`,
   `RECOMMENDATIONS` → `建议`, `As of YYYY-MM-DD` → `截至 YYYY-MM-DD`.
   Never leave English kickers in a Chinese report.
1. **One palette only.** Pick one of the 5 palettes below; use only its
   **4 hex values** (`ink`, `paper`, `paper_tint`, `ink_tint`). Define a
   `palette = {...}` dict ONCE at the top of the script and reference
   `palette["ink"]` etc. everywhere — do not copy literal hex values into
   individual styling calls.
2. **Two fonts only.** Headings = `Georgia` (serif, present on all OSes).
   Body = `Calibri` (sans, Word default). No custom fonts — recipients
   won't have them and Word falls back to Times.
   **For Chinese documents**: both render Chinese via system fallback
   (PingFang on macOS, Microsoft YaHei on Windows) — do not switch fonts.
3. **Use real Word styles, not manual formatting.** Headings must use the
   built-in `Heading 1` / `Heading 2` / `Heading 3` styles so Word's
   navigation pane and auto table-of-contents work. A document where every
   heading is just bold 18pt body text is broken — it has no outline.
4. **Forbidden:**
   - WordArt, drop shadows, glow, 3-D effects, gradient fills
   - clipart, emoji as decoration, stock-photo placeholders
   - centered body paragraphs (left-align / justify only)
   - all-caps body text (kickers and labels only)
   - Comic Sans, Arial Black, Times New Roman as a deliberate choice
5. **Real content only.** No lorem ipsum, no `[Title here]` placeholders,
   no fabricated statistics, no fake citations. If a section has no user
   data, drop the section rather than padding it.
6. **Failure honesty — NEVER fake the deliverable.**
   - If `execute_python_code` raises after multiple retries, STOP and report
     the actual error. Do not write a stub file like
     `write_file("report.docx", "placeholder")` to make the chip appear.
   - The final answer must reflect what was actually written. Do not
     describe sections or tables that aren't in the saved `.docx`.

## 🎨 Palettes — pick ONE

Same palettes as `pdf-report-editorial` and `pptx-editorial` — keep the
editorial family visually consistent. Each: `ink` (body text + rules),
`paper` (text on an ink band), `paper_tint` (table band + callout bg),
`ink_tint` (kickers, captions, footer). The keys are underscored — the dict
below is what every snippet indexes into.

- **Monocle** (default / business / tech / policy)
  ink `0A0A0B` · paper `F1EFEA` · paper_tint `E8E5DE` · ink_tint `18181A`
- **Indigo Porcelain** (research / data-heavy)
  ink `0A1F3D` · paper `F1F3F5` · paper_tint `E4E8EC` · ink_tint `152A4A`
- **Forest Ink** (sustainability / impact)
  ink `1A2E1F` · paper `F5F1E8` · paper_tint `ECE7DA` · ink_tint `253D2C`
- **Kraft Paper** (humanities / qualitative)
  ink `2A1E13` · paper `EEDFC7` · paper_tint `E0D0B6` · ink_tint `3A2A1D`
- **Dune** (art / design / fashion criticism)
  ink `1F1A14` · paper `F0E6D2` · paper_tint `E3D7BF` · ink_tint `2D2620`

Define it once at the top of the script, then index it everywhere:

```python
palette = {"ink": "0A0A0B", "paper": "F1EFEA",
           "paper_tint": "E8E5DE", "ink_tint": "18181A"}   # Monocle
```

python-docx takes hex without the `#` prefix, via
`RGBColor.from_string(palette["ink"])`.

## ✒️ Typography (python-docx Pt sizes)

| Role | Style | Family | Size | Weight |
|---|---|---|---|---|
| Cover title | `Title` | Georgia | 40pt | regular |
| Cover subtitle / dek | body run | Calibri | 14pt | italic |
| Kicker (small caps label) | body run | Calibri | 9pt, uppercase | bold, ink_tint |
| H1 section | `Heading 1` | Georgia | 22pt | regular |
| H2 subsection | `Heading 2` | Georgia | 16pt | regular |
| H3 sub-subsection | `Heading 3` | Calibri | 12pt | bold |
| Body paragraph | `Normal` | Calibri | 11pt | regular, line 1.4 |
| Pull quote | `Intense Quote` | Georgia | 14pt | italic |
| Table header | table run | Calibri | 10pt | bold, paper on ink |
| Table body | table run | Calibri | 10pt | regular |
| Caption / footnote | `Caption` | Calibri | 9pt | italic, ink_tint |

## 📐 Page setup

Set margins, size and orientation on each `section` before adding content.

```python
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.section import WD_ORIENT

doc = Document()
sec = doc.sections[0]

# A4 portrait with editorial margins
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.orientation = WD_ORIENT.PORTRAIT
sec.top_margin, sec.bottom_margin = Cm(2.5), Cm(2.5)
sec.left_margin, sec.right_margin = Cm(2.2), Cm(2.2)
```

⚠️ **Orientation gotcha:** setting `sec.orientation = WD_ORIENT.LANDSCAPE`
does NOT swap the page dimensions — Word reads the width/height, not the
flag. You must swap them yourself:

```python
sec.orientation = WD_ORIENT.LANDSCAPE
sec.page_width, sec.page_height = Cm(29.7), Cm(21.0)   # swap explicitly
```

Set the default body font once on the `Normal` style so every paragraph
inherits it:

```python
from docx.shared import Pt, RGBColor

normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)
normal.font.color.rgb = RGBColor.from_string(palette["ink"])
normal.paragraph_format.line_spacing = 1.4
normal.paragraph_format.space_after = Pt(8)
```

Do not pin `w:eastAsia` to a named CJK face. Rule 2 is that Chinese renders
through each platform's own fallback — writing `Microsoft YaHei` into the
style hard-codes a font that is absent on macOS and Linux, which is the
substitution the rule exists to prevent. Leaving the attribute unset is what
lets Word pick PingFang, YaHei or Noto per platform.

## 🏛️ Cover page pattern

The cover is its own section, so the body starts on a fresh page and can
later be given its own header, footer or page numbering without touching the
cover. Structure: kicker → title → dek → meta row.

```python
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor


def strip_style_border(doc, style_name):
    """Drop a built-in style's decorative border.

    The border lives on the *style*, not on paragraphs using it, so clearing
    a paragraph's own pPr does nothing -- it never had one. Title and Intense
    Quote both ship an accent-blue w:pBdr that is in none of the palettes.
    """
    style_pPr = doc.styles[style_name].element.get_or_add_pPr()
    for bdr in style_pPr.findall(qn("w:pBdr")):
        style_pPr.remove(bdr)


def add_kicker(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text.upper())
    run.font.name, run.font.size, run.font.bold = "Calibri", Pt(9), True
    run.font.color.rgb = RGBColor.from_string(palette["ink_tint"])
    return p

# --- cover ---
doc.add_paragraph().paragraph_format.space_after = Pt(160)  # vertical push
add_kicker(doc, "Research Brief")

# Title and Intense Quote ship an accent-blue border, and Intense Quote and
# Caption default to bold. None of that is in the palette. Clearing the border
# once on the style covers every paragraph that uses it.
strip_style_border(doc, "Title")
strip_style_border(doc, "Intense Quote")

title = doc.add_paragraph(style="Title")
title_run = title.add_run("Agent Infrastructure in 2026")
title_run.font.name, title_run.font.size = "Georgia", Pt(40)
title_run.font.color.rgb = RGBColor.from_string(palette["ink"])

dek = doc.add_paragraph()
dek_run = dek.add_run("Market shape, adoption curves, and the shift to "
                      "production-grade systems.")
dek_run.font.name, dek_run.font.size, dek_run.font.italic = "Calibri", Pt(14), True

meta = doc.add_paragraph()
meta_run = meta.add_run("Xagent Team · 2026-05-14")   # middle dot, not hyphen
meta_run.font.size = Pt(10)
meta_run.font.color.rgb = RGBColor.from_string(palette["ink_tint"])

# A new section, not just a page break: the body gets its own section
# properties, so a header, footer or page numbering added later applies to
# it without touching the cover.
from docx.enum.section import WD_SECTION

doc.add_section(WD_SECTION.NEW_PAGE)
```

## 🔠 Heading hierarchy

Always use the built-in styles, then restyle the style object once — not
each heading individually:

```python
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor


def use_font(style, family):
    """Set a style's font and drop the template's theme fonts.

    `.font.name` only adds w:ascii/w:hAnsi. The default headings also carry
    w:asciiTheme="majorHAnsi" etc., and Word prefers the theme attributes --
    so without this the headings stay in the template font, not Georgia.
    """
    style.font.name = family
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    for theme in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        rfonts.attrib.pop(qn(f"w:{theme}"), None)


for name, family, size in (("Heading 1", "Georgia", 22),
                           ("Heading 2", "Georgia", 16),
                           ("Heading 3", "Calibri", 12)):
    st = doc.styles[name]
    use_font(st, family)
    st.font.size = Pt(size)
    st.font.color.rgb = RGBColor.from_string(palette["ink"])
    st.font.bold = (name == "Heading 3")
    st.paragraph_format.space_before = Pt(18)
    st.paragraph_format.space_after = Pt(6)

doc.add_heading("Executive Summary", level=1)
doc.add_paragraph("...")
doc.add_heading("Market Size", level=2)
```

⚠️ `doc.add_heading(..., level=0)` applies the `Title` style, not a
heading — use it only on the cover. Body sections start at `level=1`.

Pull quotes use the built-in style too, so they stay in the outline-free
body flow:

```python
quote = doc.add_paragraph(style="Intense Quote")
quote_run = quote.add_run("Adoption is no longer the constraint; "
                          "operating cost is.")
quote_run.font.name, quote_run.font.size = "Georgia", Pt(14)
quote_run.font.italic = True
quote_run.font.bold = False            # Intense Quote defaults to bold
quote_run.font.color.rgb = RGBColor.from_string(palette["ink"])
```

## 📊 Table styling

python-docx has no API for cell shading or borders, so both need raw
OOXML. These two helpers are the whole toolkit — copy them verbatim.

```python
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# Both w:tcPr and w:tcBorders are ordered sequences: Word rejects the part when
# children appear out of order, and each tag may appear at most once. Appending
# is wrong on both counts, so everything below inserts at the schema position.
# Relative order from the OOXML schema. vAlign is here even though no recipe
# below sets it: it sorts after shd, so if a caller centres a cell before
# shading it, _put has to see vAlign to insert ahead of rather than after it.
_TCPR_ORDER = ("tcBorders", "shd", "vAlign")
_BORDER_ORDER = ("top", "left", "bottom", "right")


def _hex6(value):
    """Reject what Word cannot read. python-docx writes these attributes
    verbatim and save() never validates, so a bad value only surfaces as
    Word's "unreadable content" repair prompt when the user opens the file.
    RGBColor.from_string already refuses the same mistake."""
    text = str(value)
    if len(text) != 6 or any(c not in "0123456789abcdefABCDEF" for c in text):
        raise ValueError(f"expected 6 hex digits without '#', got {value!r}")
    return text


def _put(parent, tag, order):
    """Replace parent's <w:{tag}> child, keeping the schema's element order."""
    for stale in parent.findall(qn(f"w:{tag}")):
        parent.remove(stale)
    if tag not in order:
        raise ValueError(f"unknown element {tag!r}; expected one of {order}")
    el = OxmlElement(f"w:{tag}")
    rank = order.index(tag)
    later = [child for child in parent
             if isinstance(child.tag, str)
             and child.tag.rsplit("}", 1)[-1] in order
             and order.index(child.tag.rsplit("}", 1)[-1]) > rank]
    if later:
        later[0].addprevious(el)
    else:
        parent.append(el)
    return el


def shade_cell(cell, hex_fill):
    """Solid background fill for one table cell."""
    hex_fill = _hex6(hex_fill)
    tcPr = cell._tc.get_or_add_tcPr()
    shd = _put(tcPr, "shd", _TCPR_ORDER)
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)

def set_row_border(row, edge, hex_color, sz=8):
    """Hairline on one edge of every cell in a row, e.g. 'top' or 'bottom'.
    sz is in 1/8 pt. Re-styling the same edge replaces it."""
    hex_color = _hex6(hex_color)
    if not isinstance(sz, int) or isinstance(sz, bool) or sz <= 0:
        raise ValueError(f"sz must be a positive integer, got {sz!r}")
    for cell in row.cells:
        tcPr = cell._tc.get_or_add_tcPr()
        borders = tcPr.find(qn("w:tcBorders"))
        if borders is None:
            borders = _put(tcPr, "tcBorders", _TCPR_ORDER)
        el = _put(borders, edge, _BORDER_ORDER)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(sz))          # 8 = 1pt
        el.set(qn("w:color"), hex_color)

```

Editorial table rules — **horizontal rules only, no vertical borders**:

```python
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

rows = [("Region", "Revenue", "YoY"),
        ("North America", "4.2M", "+18%"),
        ("EMEA", "2.8M", "+11%")]

# No table style: the stock "Table Grid" draws all six edges in black
# (w:color="auto"), which is not a palette value, and nilling the verticals
# per cell still leaves its black top/bottom/insideH rules behind. An
# unstyled table starts with no borders at all, so the only rules on it are
# the ones drawn below.
table = doc.add_table(rows=len(rows), cols=len(rows[0]))
table.alignment = WD_TABLE_ALIGNMENT.CENTER
table.autofit = True

for r, record in enumerate(rows):
    for c, value in enumerate(record):
        cell = table.cell(r, c)
        cell.text = str(value)
        para = cell.paragraphs[0]
        # Numeric columns right-aligned, including their header, so the
        # column reads as one edge rather than a ragged one.
        para.alignment = (WD_ALIGN_PARAGRAPH.RIGHT if c > 0
                          else WD_ALIGN_PARAGRAPH.LEFT)
        run = para.runs[0]
        run.font.name, run.font.size = "Calibri", Pt(10)
        if r == 0:                                   # header row
            run.font.bold = True
            run.font.color.rgb = RGBColor.from_string(palette["paper"])
            shade_cell(cell, palette["ink"])         # ink header band
        elif r % 2 == 0:                             # zebra band
            shade_cell(cell, palette["paper_tint"])
            run.font.color.rgb = RGBColor.from_string(palette["ink"])
        else:
            run.font.color.rgb = RGBColor.from_string(palette["ink"])

# One rule under the table. Nothing under the header row -- it is already a
# solid ink band, so an ink rule there would be invisible against itself.
set_row_border(table.rows[-1], "bottom", palette["ink"])

caption = doc.add_paragraph(style="Caption")
caption_run = caption.add_run("Table 1 — Revenue by region, FY2026.")
caption_run.font.size, caption_run.font.italic = Pt(9), True
caption_run.font.bold = False          # Caption defaults to bold
caption_run.font.color.rgb = RGBColor.from_string(palette["ink_tint"])
```

Repeat the header row across page breaks for tables longer than a page:

```python
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# w:trPr allows at most one w:tblHeader, so replace rather than append -- a
# second one is schema-invalid, and a loop over tables would add one per pass.
trPr = table.rows[0]._tr.get_or_add_trPr()
for stale in trPr.findall(qn("w:tblHeader")):
    trPr.remove(stale)
trPr.append(OxmlElement("w:tblHeader"))
```

## 📝 Output checklist

- [ ] `from docx import Document` (import name is `docx`, not `python_docx`)
- [ ] LANGUAGE matches the user's prompt — no English kickers in a ZH report
- [ ] One palette, only its 4 hex values appear anywhere
- [ ] Only Georgia + Calibri; `Normal` carries the body font, and no named
      CJK face is pinned to `w:eastAsia`
- [ ] Real `Heading 1/2/3` styles used (Word navigation pane shows an outline)
- [ ] Page size, margins, and orientation set explicitly on the section
      (landscape = swap width/height, not just the orientation flag)
- [ ] Cover page present, followed by a `WD_SECTION.NEW_PAGE` section break
      (not an extra page break as well — that leaves a blank page)
- [ ] Tables: ink header band, `paper_tint` zebra rows, horizontal rules only,
      numbers right-aligned, caption below
- [ ] No fabricated data, no lorem ipsum, no placeholder headings
- [ ] File saved with a plain filename, then re-opened and asserted non-empty
      (byte size proves nothing — an empty document is already ~36 KB)
- [ ] Any structural assertion matches what was requested — no fixed minimum
      that a legitimate one-page memo or letter would fail
- [ ] **Final answer FIRST LINE is `[filename](file:UUID)`** as bare markdown

Then write the .docx and report path + which palette + which sections.

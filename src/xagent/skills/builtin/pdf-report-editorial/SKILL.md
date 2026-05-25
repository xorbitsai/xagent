---
name: pdf-report-editorial
description: |
  Generate an editorial-styled PDF report (whitepaper, research brief,
  executive memo, case study) by first emitting a self-contained HTML
  document and then rendering it to PDF via the browser. Use when the
  user asks for a "PDF report", "whitepaper", "executive brief",
  "research report", "case study PDF", or similar document deliverable
  where visual quality matters and the format must be portable / printable.
  Use `pptx-editorial` or `html-deck-editorial` instead if the user
  actually wants slides; this skill is for read-as-document, not as-slides.
when_to_use: |
  When the user wants a polished read-as-document deliverable (whitepaper,
  research brief, executive memo, case study) as a PDF where typography and
  layout matter. Not for slides — use `pptx-editorial` or `html-deck-editorial`
  for slide-style decks.
tags:
  - pdf
  - report
  - whitepaper
  - editorial
---

# Editorial PDF Report

You will generate one `.pdf` file via a two-step pipeline:

1. Write a self-contained HTML file to the workspace using
   `workspace_file_tool` (or `file_tool`).
2. Call **`browser_navigate`** on the workspace-relative path (or the
   file's `file://` URL), then call **`browser_pdf`** with
   `output_filename="report.pdf"`, `format="A4"`,
   `print_background=true`. Page margins are controlled by CSS
   `@page` rules in the HTML (see "@media print" section below) —
   `browser_pdf` does not accept a `margin` object.
3. Report both the .html path (for editing) and the .pdf path (final).

## ⚠️ Hard rules — NO exceptions

0. **MATCH THE USER'S LANGUAGE.** If the prompt is Chinese (中文), ALL report
   content (kickers, H1/H2, body, captions, table headers, figure callouts)
   must be in Chinese. NEVER copy English template phrases like
   `EXECUTIVE BRIEF` / `RESEARCH BRIEF` / `THE BOTTOM LINE` into a Chinese
   report — translate them. Person/company names should match the locale.

1. **One palette only.** Pick one of the 5 palettes below; never invent hex.
2. **Two fonts only.** Display = `'Playfair Display', Georgia, serif`.
   Body = `'Inter', -apple-system, Helvetica, sans-serif`.
   Load Playfair Display + Inter via single Google Fonts `<link>` (only
   external resource allowed). All other CSS / JS inline.
3. **Forbidden visual elements:**
   - drop-shadow, box-shadow, gradient backgrounds, blur, glassmorphism
   - rounded corners > 2px
   - emoji as decoration, clipart, stock-photo placeholders
   - colored hyperlinks (links must be `ink` color + underline)
   - centered body paragraphs (left-align only)
   - all-caps body text (kicker / labels only)
   - more than one accent color
4. **Real content only.** No lorem ipsum, no placeholder text, no fabricated
   data, no fake citations. Citations must reference real sources or be
   omitted.
5. **Print-aware CSS required:**
   - `@page { size: A4; margin: 0; }`
   - `@media print` rules for page breaks (no orphan/widow titles)
   - `page-break-inside: avoid` on figures, callouts, tables
   - `page-break-before: always` on `section.chapter` wrappers or on
     the H2 section-divider rule (the cover is the only H1 — see the
     "Document structure" section and the output checklist).

## 🎨 Palettes — pick ONE

Each: `ink` (text + rules), `paper` (page bg), `paper-tint` (callout box bg),
`ink-tint` (folio + section labels).

- **Monocle** (default / business / tech / policy)
  ink `#0a0a0b` · paper `#f1efea` · paper-tint `#e8e5de` · ink-tint `#18181a`
- **Indigo Porcelain** (research / data-heavy)
  ink `#0a1f3d` · paper `#f1f3f5` · paper-tint `#e4e8ec` · ink-tint `#152a4a`
- **Forest Ink** (sustainability / impact)
  ink `#1a2e1f` · paper `#f5f1e8` · paper-tint `#ece7da` · ink-tint `#253d2c`
- **Kraft Paper** (humanities / qualitative)
  ink `#2a1e13` · paper `#eedfc7` · paper-tint `#e0d0b6` · ink-tint `#3a2a1d`
- **Dune** (art / design / fashion criticism)
  ink `#1f1a14` · paper `#f0e6d2` · paper-tint `#e3d7bf` · ink-tint `#2d2620`

## ✒️ Typography (use exactly these scale values)

| Role | Family | Size | Line-height | Weight |
|---|---|---|---|---|
| H1 (cover title only — used once) | Display | 48pt | 1.1 | 400 |
| H2 section / chapter divider | Display | 28pt | 1.2 | 400 |
| H3 subsection | Body | 14pt | 1.3 | 600 |
| Body paragraph | Body | 11pt | 1.55 | 400 |
| Pull quote | Display italic | 22pt | 1.3 | 400 |
| Callout box | Body | 11pt | 1.5 | 400 (italic optional) |
| Caption / footnote | Body | 9pt | 1.4 | 400 |
| Kicker (small caps label) | Body | 9pt, letter-spacing 0.12em, uppercase | — | 500 |
| Folio (page number) | Body | 9pt | — | 400 |
| Table header | Body | 10pt | 1.3 | 600 |
| Table body | Body | 10pt | 1.4 | 400 |

## 📐 Document structure

A typical editorial PDF has these block types — use as needed by user content:

### Cover (page 1)
- Top: small `kicker` (e.g. "RESEARCH BRIEF" or "EXECUTIVE MEMO")
- Center vertically: H1 title (Display 48pt)
- Below: subtitle / dek (Body 14pt italic)
- Bottom-left: author / org · Bottom-right: date / volume

### Table of Contents (page 2, optional)
- H2 "Contents" + numbered section list with page numbers (right-aligned)

### Section divider (every chapter)
- H2 title + thin `ink` hairline (1px) below
- Optional kicker above (e.g. "PART 02")

### Body
- 2-column layout (CSS `column-count: 2; column-gap: 24pt`) for long sections
- 1-column for short / data-heavy sections
- Drop cap (CSS `::first-letter`) for chapter openers (Display, 5em, float left)
- Pull quotes as `<aside>` blocks, Display italic 22pt, with thin top + bottom rules

### Figures / tables
- Captioned (caption below figure, Body 9pt italic, prefixed with "Fig. 1 — ")
- Charts as inline SVG using only palette colors
- Tables: thin `ink` rules above header + below header + below last row; no
  vertical borders; cell padding 6pt 10pt; striped alternating rows with
  `paper-tint`

### Callouts
- `paper-tint` background block, ink left border 2px, padding 16pt, no shadow

### Footer (every page)
- Left: doc title (Body 9pt ink-tint)
- Right: page number (Body 9pt ink-tint)
- Separator: 1px ink hairline above the footer

## 🛠️ HTML skeleton

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>...</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;1,400&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
@page { size: A4; margin: 0; }
:root { --ink: #0a0a0b; --paper: #f1efea; --paper-tint: #e8e5de; --ink-tint: #18181a; }
* { box-sizing: border-box; }
body { font-family: 'Inter', -apple-system, sans-serif; color: var(--ink); background: var(--paper); margin: 0; }
.page { width: 210mm; min-height: 297mm; padding: 24mm 18mm; page-break-after: always; position: relative; }
.page:last-child { page-break-after: auto; }
h1 { font-family: 'Playfair Display', Georgia, serif; font-weight: 400; font-size: 48pt; line-height: 1.1; margin: 0 0 16pt; }
h2 { font-family: 'Playfair Display', serif; font-weight: 400; font-size: 28pt; line-height: 1.2; margin: 32pt 0 12pt; border-bottom: 1px solid var(--ink); padding-bottom: 8pt; }
.kicker { font-size: 9pt; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-tint); font-weight: 500; margin-bottom: 8pt; }
p { font-size: 11pt; line-height: 1.55; margin: 0 0 11pt; text-align: justify; hyphens: auto; }
.two-col { column-count: 2; column-gap: 18pt; }
.callout { background: var(--paper-tint); border-left: 2px solid var(--ink); padding: 14pt 18pt; margin: 16pt 0; page-break-inside: avoid; }
.pull-quote { font-family: 'Playfair Display', serif; font-style: italic; font-size: 22pt; line-height: 1.3; border-top: 1px solid var(--ink); border-bottom: 1px solid var(--ink); padding: 14pt 0; margin: 18pt 0; page-break-inside: avoid; }
.folio { position: absolute; bottom: 12mm; right: 18mm; font-size: 9pt; color: var(--ink-tint); }
.footer-line { position: absolute; bottom: 18mm; left: 18mm; right: 18mm; border-top: 1px solid var(--ink); }
.footer-title { position: absolute; bottom: 12mm; left: 18mm; font-size: 9pt; color: var(--ink-tint); }
table { width: 100%; border-collapse: collapse; font-size: 10pt; margin: 12pt 0; page-break-inside: avoid; }
th { font-weight: 600; text-align: left; padding: 6pt 10pt; border-top: 1px solid var(--ink); border-bottom: 1px solid var(--ink); }
td { padding: 6pt 10pt; }
tbody tr:nth-child(even) td { background: var(--paper-tint); }
tbody tr:last-child td { border-bottom: 1px solid var(--ink); }
figcaption { font-size: 9pt; font-style: italic; color: var(--ink-tint); margin-top: 6pt; }
</style>
</head>
<body>
  <div class="page">
    <!-- cover -->
  </div>
  <div class="page">
    <!-- body -->
  </div>
</body>
</html>
```

## 📝 Output checklist

- [ ] One palette, exactly its 4 hex
- [ ] Only Playfair Display + Inter loaded
- [ ] `@page` + print CSS present; `page-break-inside: avoid` on figures/callouts
- [ ] Folio + footer line on every page
- [ ] No forbidden visuals (verify no `shadow`, `gradient`, `blur`, `radius` > 2px)
- [ ] All content from user input or real cited sources
- [ ] Cover page distinct from body pages
- [ ] H1 only on cover; H2 for section dividers

## ✅ Then export PDF via browser

```
1. Write HTML to workspace as `report.html`.
2. Call `browser_navigate` with the workspace path (or
   `file://<absolute-path>/report.html`).
3. Call `browser_pdf` with `output_filename="report.pdf"`,
   `format="A4"`, `print_background=true`. The page margins are set
   in CSS `@page { margin: 12mm 14mm 16mm 14mm }` inside the HTML —
   `browser_pdf` writes the file directly to the workspace and the
   `markdown_link` for it is in the tool's response (do not manually
   decode base64 unless the response explicitly indicates the
   workspace write failed).
```

### 📎 Deliver as a clickable chip in chat

The browser tools (`browser_navigate`, `browser_pdf`) and the workspace
file-writing tools return a `markdown_link` field — or a `file_refs[]`
array of entries each carrying `file_id` / `filename` / `markdown_link` —
for every workspace file they registered. **Read each tool's response**
and copy the returned `markdown_link` strings verbatim. Start your final
answer with those chip lines (bare markdown, no backticks, not phrased
as "file_id: UUID"):

✅ **CORRECT** (chat renders these as clickable chips):

    [report.pdf](file:1f9c5a40-...)
    [report.html](file:9bd5f1aa-...)

The UUIDs come from the tools' own responses; do not fabricate them.

❌ **WRONG**: ``` file_id: `UUID` ```, wrapping the chip link inside
code fences, or calling `get_file_info(...)` to "fetch" the file_id
(its `FileInfo` shape does not include `file_id` — the chip reference
is already on the producing tool's result).

After the chip lines, briefly note: palette chosen, page count, section count.

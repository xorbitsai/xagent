---
name: xlsx-financial-report
description: |
  Generate a native Excel (.xlsx) file styled as a clean financial /
  KPI / dashboard report. Use when the user asks for an "Excel report",
  "financial report", "KPI dashboard", "monthly report", ".xlsx", or
  similar tabular deliverable where the output must look professional
  (not just raw data). Output is a single .xlsx file produced via
  `openpyxl` through the `execute_python_code` tool. Output opens cleanly
  in Excel / Google Sheets / Numbers.
  Do NOT use this skill for ad-hoc data exploration or raw CSV dumps —
  use the regular `excel` tool for those.
when_to_use: |
  When the user wants a polished financial / KPI / dashboard Excel deliverable
  (xlsx) where the file is meant to be opened and skimmed — not raw data dumps.
  Use the regular `excel` tool instead for ad-hoc exploration / CSV-style output.
tags:
  - xlsx
  - excel
  - financial
  - report
  - dashboard
---

# Editorial Financial Report (.xlsx)

You will generate one `.xlsx` file via openpyxl by writing a Python
program and running it through the `execute_python_code` tool. Save to
the workspace, then report the path + a 1-line content summary.

## 📦 Required runtime packages

The sandboxed `execute_python_code` ships with `pandas`, `numpy`,
`matplotlib`, and **`openpyxl>=3.1.0`** preinstalled — no extra
installation step is needed. Just `import openpyxl` at the top of
your script and proceed.

> **Note:** `execute_python_code` accepts only `code` and
> `capture_output` arguments; there is no `packages` parameter.
> All required libraries are already available in the sandbox image.

## 💾 How to save the file

`execute_python_code` already runs with the task's output directory as
the current working directory. Save the workbook with a **plain
filename** (no path), e.g.:

```python
wb.save("xagent_metrics_dashboard.xlsx")   # ✅ correct
```

Do NOT use:
- `wb.save("/workspace/foo.xlsx")` — `/workspace` does not exist on this host
- `wb.save("output/foo.xlsx")` — there is no nested `output/` subdir
- BytesIO + base64 round-trips — just save to disk directly.

Verify with `os.path.getsize("xagent_metrics_dashboard.xlsx")` before
declaring success. A real openpyxl-generated xlsx is **always >5 KB** —
if your saved file is under 1 KB it's broken or empty.

### 🔗 Make it clickable in chat — REQUIRED

The Python executor returns a `markdown_link` field in its response for
every workspace file it generated (or a `file_refs[]` array with one
entry per file). **Read the tool's response** and use the returned
`markdown_link` string verbatim. In your final answer, the **first line
MUST be that chip link itself** — bare markdown, NOT inside backticks,
NOT presented as "file_id: UUID":

✅ **CORRECT** (renders as clickable chip — chat UI looks for this exact pattern):

    [xagent_metrics_dashboard.xlsx](file:20fae785-3823-4906-b385-d0e8a7807dc8)

The UUID comes from the executor response. Do not fabricate one.

❌ **WRONG — common failure that renders as plain text**:

    已生成报表:
    - 文件名: `xagent_metrics_dashboard.xlsx`
    - file_id: `20fae785-3823-4906-b385-d0e8a7807dc8`

❌ **ALSO WRONG — chip link inside a code fence**: ` ```[name](file:UUID)``` `
   suppresses markdown so the link won't render as a chip.

❌ **ALSO WRONG — calling `get_file_info(...)` to "fetch" the file_id**.
   Its `FileInfo` return shape does not include `file_id`; the chip
   reference is already on the executor result.

Plain-text mention of the filename is not clickable — the user must navigate
to File Management to find it. Always lead with the bare `[name](file:UUID)`
line, then describe the contents.

## ⚠️ Hard rules — NO exceptions

0. **MATCH THE USER'S LANGUAGE.** If the prompt is Chinese (中文), ALL
   workbook text (title, subtitle, section banners, KPI labels, table
   headers, deltas) must be in Chinese. Translate template phrases like
   `SUMMARY` → `汇总`, `DETAIL` → `明细`, `As of YYYY-MM-DD` → `截至
   YYYY-MM-DD`. Numbers and date formats stay locale-neutral.

1. **One accent color only.** Pick one of the 5 palettes below; use only
   its **4 hex values** (`ink`, `paper`, `paper_tint`, `accent`).
   `paper_tint` is the alternating-row band shade — without it you can
   only do flat one-color rows. No other colors anywhere.
   In code, define a ``palette = {"ink": "...", "paper": "...",
   "paper_tint": "...", "accent": "..."}`` dict ONCE at the top of the
   script and reference `palette["ink"]` / `palette["accent"]` /
   `palette["paper_tint"]` / `palette["paper"]` everywhere — do not
   copy literal hex values into individual styling calls.
2. **Two fonts only.** Headers / titles = `Cambria` (serif, Office native).
   Body / data = `Calibri` (sans, Office default).
3. **Forbidden:**
   - 3-D charts, donut/pie charts with > 5 slices, exploded pies
   - rainbow chart palettes (color charts only with `ink` + `accent`)
   - chart shadows, glow, bevel, gradient fills
   - thick borders (use `thin` only, sparingly)
   - merged-cell abuse (merge only for section banners across columns)
   - emoji as decoration
   - Comic Sans, Arial Black, MS Sans Serif
   - rainbow conditional formatting (only monochrome data bars / 2-stop
     min→max scale in `ink` shade)
4. **Real data only.** No fabricated numbers. If the user didn't provide
   data, ask first — don't invent.
5. **Always include**: title row, frozen header pane, alternating row
   bands using `paper-tint`, summary KPI block at top.
6. **Failure honesty — NEVER fake the deliverable.**
   - If `execute_python_code` raises after multiple retries, STOP and report
     the actual error to the user. Do not write a stub file like
     `write_file("report.xlsx", "placeholder")` to make the chip appear.
   - The final answer must reflect what was actually written. Do not
     describe KPI values, deltas, or sections that aren't in the saved
     `.xlsx`. If the chart didn't render, say so — don't claim it did.
   - It is better to deliver a chart-less but real .xlsx (drop the chart
     block and save without it) than to fabricate a success report. If
     the chart code keeps failing, ship the workbook without the chart
     and tell the user the chart was omitted.

## 🎨 Palettes — pick ONE

Each: `ink` (text + borders + chart series), `paper` (workbook bg / cells),
`paper-tint` (alternating row band), `accent` (single highlight color for
KPI values / chart series 2 / positive deltas).

- **Bloomberg Mono** (default / fintech)
  ink `1A1A1A` · paper `FFFFFF` · paper-tint `F7F7F5` · accent `D97706`
- **Indigo Sheet** (corporate / SaaS)
  ink `0F172A` · paper `FFFFFF` · paper-tint `F1F5F9` · accent `4F46E5`
- **Forest Ledger** (sustainability / impact)
  ink `1A2E1F` · paper `FAFAF7` · paper-tint `F0EDE3` · accent `059669`
- **Crimson Quarterly** (executive / board)
  ink `1F1F1F` · paper `FFFFFF` · paper-tint `F5F2EE` · accent `B91C1C`
- **Slate Audit** (compliance / audit)
  ink `0F1419` · paper `FCFCFC` · paper-tint `F0F2F4` · accent `0891B2`

## ✒️ Typography rules (openpyxl Font sizes in pt)

- **Title** (A1, merged across cols): Cambria, 22pt, ink, bold
- **Subtitle / date** (A2): Calibri, 11pt, ink, italic
- **Section header** (banner row): Calibri, 12pt, paper bg + ink color, bold,
  merged across the section's columns
- **Table header**: Calibri, 11pt, ink bg + paper color, bold (thin bottom border)
- **Data**: Calibri, 10pt, ink color
- **KPI big number**: Cambria, 24pt, ink, bold
- **KPI label**: Calibri, 9pt, ink-with-50%-opacity (or just slightly lighter), uppercase

## 🏗️ Standard layout

```
A1: [merged] Report Title (22pt Cambria bold)
A2: Subtitle / period · As of YYYY-MM-DD (11pt Calibri italic)
A3: (blank, 6pt row height for breathing room)

A4: [section banner] "SUMMARY"  (banner row, merged)
A5-D5: KPI cards row — each KPI in a 1×2 cell block
  - Row 5: KPI label (small caps)
  - Row 6: KPI big number (24pt Cambria)
  - Row 7: delta vs previous period (Calibri 10pt, accent if positive, ink-50 if negative)

A9: (blank)
A10: [section banner] "DETAIL"
A11: Table header (frozen at row 12)
A12+: Data rows with alternating paper-tint
```

## 📊 Number formatting (Excel format codes)

| Type | Format string |
|---|---|
| Integer | `#,##0` |
| Currency (USD) | `$#,##0.00;-$#,##0.00` (no parens, no red) |
| Currency (CNY) | `¥#,##0.00;-¥#,##0.00` |
| Percentage | `0.0%` |
| Delta percentage | `+0.0%;-0.0%;0.0%` (explicit sign) |
| Large numbers | `#,##0,"K";-#,##0,"K"` or `#,##0,,"M"` |
| Date | `yyyy-mm-dd` |

### 💱 Currency default — pick carefully

The user's name / locale is NOT a reliable signal for which currency the
*report's subject company* uses. Pick currency by **the subject domain**, not
by who's asking:

- **Default → USD (`$`)** for: SaaS, B2B software, fintech, US/global
  companies, crypto, VC metrics (ARR / MRR / NRR / CAC / LTV are
  industry-standard USD).
- **Use CNY (`¥`)** only when: the user explicitly says CNY/RMB, the
  company is named as Chinese (e.g. Alibaba / Pinduoduo / ByteDance), or
  the data source is a Chinese filing.
- **Use EUR (`€`)** for EU-based companies explicitly mentioned.
- **Use GBP (`£`)** for UK-based companies explicitly mentioned.

When in doubt, **default to USD** and note the assumption in cell A2
subtitle (e.g. `Q1 2026 Quarterly Review · As of 2026-03-31 · USD`).

## 📈 Chart styling (openpyxl charts)

- Chart types allowed: `LineChart`, `BarChart` (`type="col"`, **never** 3-D),
  `ScatterChart`. **NEVER** import `BarChart3D`, `PieChart3D`, etc.
- Series colors: only `ink` and `accent` (max 2 series). For more, use line
  variations (solid vs dashed) in `ink`.
- No legend if only 1 series. If 2 series, legend at right, no border, no fill.
- No axis title unless absolutely necessary. Axis labels font: Calibri 9pt.
- Chart border: none. Chart background: paper.
- Title: 12pt Calibri bold, ink, left-aligned above chart.

### ⚠️ openpyxl 3.x chart API gotchas — copy this verbatim, don't improvise

The openpyxl chart API is full of footguns. Use ONLY the imports below and
ONLY the kwargs shown. Most common mistakes:

| ❌ Wrong | ✅ Right |
|---|---|
| `from openpyxl.chart.data_source import StrDataSource` | not needed — `Reference` covers this |
| `from openpyxl.drawing.fill import SolidColorFill` | not needed — pass hex string to `solidFill="0F172A"` |
| `GraphicalProperties(line=...)` | `GraphicalProperties(ln=...)` (kwarg is `ln`, two letters) |
| `LineProperties(width=25000)` | `LineProperties(w=25000)` (kwarg is `w`) |
| `chart.y_axis.majorGridlines = GraphicalProperties(...)` | wrap in `ChartLines(spPr=GraphicalProperties(...))` |
| `BarChart3D` | `BarChart(type="col")` |

### ✅ Tested line-chart snippet (openpyxl 3.x — copy as-is)

> **Before the snippet:** define your chosen palette once. Example for
> Indigo Sheet:
>
> ```python
> palette = {"ink": "0F172A", "paper": "FFFFFF",
>            "paper_tint": "F1F5F9", "accent": "4F46E5"}
> ```
>
> Reference `palette["ink"]` / `palette["accent"]` / `palette["paper_tint"]`
> in the styling code below instead of copying hex literals.


```python
from openpyxl.chart import LineChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.chart.marker import Marker
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.chart.axis import ChartLines
from openpyxl.drawing.line import LineProperties

# Assume helper columns H (categories) and I (values) at rows 11..14
ch = LineChart()
ch.title = "Monthly Agent Runs (Jan–Apr 2026)"
ch.width, ch.height = 20, 10
ch.legend = None  # 1-series chart -> no legend

vals = Reference(ws, min_col=9, min_row=11, max_row=14)  # column I
cats = Reference(ws, min_col=8, min_row=11, max_row=14)  # column H
ch.add_data(vals, titles_from_data=False)
ch.set_categories(cats)

# Series styling — kwarg is `ln=`, not `line=`. `w` is in EMU (12700 EMU = 1pt).
s = ch.series[0]
s.graphicalProperties = GraphicalProperties(
    ln=LineProperties(solidFill=palette["ink"], w=25000)  # ~2pt
)
s.marker = Marker(symbol="circle", size=7)
s.marker.graphicalProperties = GraphicalProperties(solidFill=palette["accent"])

# Data labels (optional)
s.dLbls = DataLabelList(showVal=True)
s.dLbls.numFmt = '#,##0'

# Value-axis gridlines — must wrap in ChartLines(spPr=...). The
# gridline tint comes from the palette's paper_tint, not a hard-coded
# grey.
ch.y_axis.majorGridlines = ChartLines(
    spPr=GraphicalProperties(ln=LineProperties(solidFill=palette["paper_tint"], w=6350))
)
# No category-axis gridlines
ch.x_axis.majorGridlines = None

# Backgrounds (optional — paper is usually already default)
ch.plot_area.graphicalProperties = GraphicalProperties(solidFill=palette["paper"])

ws.add_chart(ch, "A22")
```

### ✅ Tested bar-chart snippet

```python
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.shapes import GraphicalProperties

ch = BarChart()
ch.type = "col"     # vertical bars; "bar" = horizontal
ch.style = 2        # minimal style; we override colors anyway
ch.title = "Skill Invocations"
ch.width, ch.height = 20, 10
ch.legend = None

ch.add_data(Reference(ws, min_col=3, min_row=11, max_row=20), titles_from_data=False)
ch.set_categories(Reference(ws, min_col=2, min_row=11, max_row=20))

# Fill the bars with `ink` (from your palette dict)
ch.series[0].graphicalProperties = GraphicalProperties(solidFill=palette["ink"])

ws.add_chart(ch, "F22")
```

## 🧊 Frozen panes + alternating rows + conditional formatting

```python
ws.freeze_panes = 'A12'  # header at row 11 stays visible

# Alternating row bands (apply via conditional formatting MOD)
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import PatternFill
band = PatternFill(start_color=palette["paper_tint"],
                   end_color=palette["paper_tint"], fill_type='solid')
ws.conditional_formatting.add(
    f'A12:Z{last_row}',
    FormulaRule(formula=['MOD(ROW(),2)=0'], fill=band),
)

# Data bar (monochrome only) for a KPI column
from openpyxl.formatting.rule import DataBarRule
ws.conditional_formatting.add(
    'D12:D' + str(last_row),
    DataBarRule(start_type='min', end_type='max', color=palette["ink"]),
)
```

## 📝 Output checklist

- [ ] One palette, only its 4 hex values appear anywhere
- [ ] Only Cambria + Calibri (verify no Arial/Times slipped in)
- [ ] No 3-D, no shadows, no gradients, no rainbow conditional formatting
- [ ] Title row merged, frozen header at correct row
- [ ] All numbers have explicit Excel format codes
- [ ] Charts: ≤ 2 series, ink + accent only
- [ ] Alternating row bands applied
- [ ] No fabricated data; if KPIs require deltas, deltas come from real prior-period numbers

Then write the .xlsx and report path + which palette + which sections.

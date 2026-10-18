import * as XLSX from "xlsx"
import DOMPurify from "dompurify"

export interface ExcelSheetPreview {
  html: string
  missingFormulaResults: number
}

const INLINE_TAGS = ["a", "b", "strong", "i", "em", "u", "s", "sub", "sup", "span", "br"]

function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;")
}

/** Render a preview copy without treating missing formula caches as blank results. */
export function createExcelSheetPreview(
  sheet: XLSX.WorkSheet,
  missingResultText: string,
): ExcelSheetPreview {
  const preview = { ...sheet }
  let missingFormulaResults = 0

  for (const [address, cell] of Object.entries(sheet)) {
    if (address.startsWith("!")) continue
    // Format a copy before dropping raw v/z attributes, preserving cell text
    // and table structure. Link targets follow the export escaping path below.
    let html = ""
    if (cell.v != null) {
      // Rich-text runs can contain table markup. Keep them inline before
      // insertion so they cannot close a cell or inject rows into the sheet.
      html = cell.h ? DOMPurify.sanitize(cell.h, {
        ALLOWED_TAGS: INLINE_TAGS,
        ALLOWED_ATTR: ["href", "title"],
        ALLOW_DATA_ATTR: false,
        ALLOW_ARIA_ATTR: false,
      }) : escapeHtml(XLSX.utils.format_cell({ ...cell })).replace(/\r?\n/g, "<br/>")
    }
    // With sheetStubs enabled, SheetJS represents an uncached XLSX formula as
    // a type-z cell with v: 0. That is not a calculated zero. Empty strings,
    // false, zero and cached errors on other cell types are actual results.
    if ((cell.f || cell.F) && (cell.t === "z" || cell.v == null)) {
      missingFormulaResults++
      // Array-formula followers can have F (the array range) but no formula
      // text of their own. Show a placeholder rather than inventing a formula.
      const text = cell.f ? `=${cell.f}` : missingResultText
      html = escapeHtml(text)
    }
    preview[address] = {
      t: cell.t === "n" ? "n" : "s",
      v: "",
      h: html,
    }
    if (cell.l?.Target) {
      // SheetJS 0.20.3 decodes XML entities on read and escapes HTML attributes
      // on export. Keep the original target: pre-escaping would double-encode
      // literal quotes/entities. DOMPurify below still rejects unsafe schemes.
      preview[address].l = { Target: cell.l.Target }
    }
  }

  // Sanitize the complete output, including rich text and link protocols. Only
  // keep spreadsheet presentation markup; no embedded media or workbook CSS.
  const html = DOMPurify.sanitize(XLSX.utils.sheet_to_html(preview), {
    ALLOWED_TAGS: [
      "table", "thead", "tbody", "tfoot", "tr", "th", "td",
      ...INLINE_TAGS,
    ],
    ALLOWED_ATTR: ["id", "colspan", "rowspan", "href", "title", "data-t"],
    ALLOW_DATA_ATTR: false,
    ALLOW_ARIA_ATTR: false,
  })
  return { html, missingFormulaResults }
}

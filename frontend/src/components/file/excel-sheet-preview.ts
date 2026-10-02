import * as XLSX from "xlsx"

export interface ExcelSheetPreview {
  html: string
  missingFormulaResults: number
}

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
    preview[address] = { ...cell }
    // With sheetStubs enabled, SheetJS represents an uncached XLSX formula as
    // a type-z cell with v: 0. That is not a calculated zero. Empty strings,
    // false, zero and cached errors on other cell types are actual results.
    if ((cell.f || cell.F) && (cell.t === "z" || cell.v == null)) {
      missingFormulaResults++
      // Array-formula followers can have F (the array range) but no formula
      // text of their own. Show a placeholder rather than inventing a formula.
      // SheetJS does not escape its data-v attribute. Keep formula text out of
      // that attribute and supply only escaped display HTML on a fresh cell.
      const text = cell.f ? `=${cell.f}` : missingResultText
      preview[address] = { t: "s", v: "", h: escapeHtml(text) }
    }
  }

  return { html: XLSX.utils.sheet_to_html(preview), missingFormulaResults }
}

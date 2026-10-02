/// <reference types="@testing-library/jest-dom/vitest" />

import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"
import * as XLSX from "xlsx"
import JSZip from "jszip"
import { I18nProvider } from "@/contexts/i18n-context"
import { ExcelPreviewRenderer } from "./excel-preview-renderer"
import { createExcelSheetPreview } from "./excel-sheet-preview"

function workbookContent(sheets: Record<string, XLSX.WorkSheet>) {
  const workbook = XLSX.utils.book_new()
  for (const [name, sheet] of Object.entries(sheets)) {
    XLSX.utils.book_append_sheet(workbook, sheet, name)
  }
  // Exercise real XLSX bytes and the real reader, not a mocked parsed workbook.
  return XLSX.write(workbook, { type: "base64", bookType: "xlsx" }) as string
}

function preview(content: string, locale: "en" | "zh" = "en") {
  return (
    <I18nProvider initialLocale={locale}>
      <ExcelPreviewRenderer base64Content={content} />
    </I18nProvider>
  )
}

function cell(container: HTMLElement, address: string) {
  return container.querySelector(`#sjs-${address}`)
}

describe("ExcelPreviewRenderer formula results", () => {
  afterEach(cleanup)

  it("keeps uncached formulas visible instead of dropping them or displaying zero", () => {
    const content = workbookContent({
      Inventory: {
        A1: { t: "s", v: "Quantity" },
        B1: { t: "s", v: "Inventory value" },
        A2: { t: "n", v: 7 },
        B2: { t: "n", f: "A2*12", z: "$0.00" },
        B3: { t: "n", f: "SUM(B2:B2)" },
        "!ref": "A1:B3",
      },
    })
    const { container } = render(preview(content))

    expect(cell(container, "B2")).toHaveTextContent("=A2*12")
    expect(cell(container, "B3")).toHaveTextContent("=SUM(B2:B2)")
    expect(screen.getByRole("status")).toHaveTextContent("Missing formula results on this sheet: 2")
    expect(screen.getByRole("status")).toHaveTextContent("this preview does not calculate them")
    expect(screen.getByRole("status")).toHaveTextContent("Download and open")
  })

  it("preserves cached zero, false, empty string, errors and number formats", () => {
    const content = workbookContent({
      Finance: {
        A1: { t: "n", f: "1-1", v: 0 },
        A2: { t: "b", f: "1=2", v: false },
        A3: { t: "s", f: 'IF(1,"","x")', v: "" },
        A4: { t: "e", f: "1/0", v: 7 },
        A5: { t: "n", f: "1/4", v: 0.25, z: "0.0%" },
        A6: { t: "n", f: "2*12", v: 24, z: "$0.00" },
        "!ref": "A1:A7",
      },
    })
    const { container } = render(preview(content))

    expect(cell(container, "A1")).toHaveTextContent(/^0$/)
    expect(cell(container, "A2")).toHaveTextContent(/^FALSE$/)
    expect(cell(container, "A3")).toBeEmptyDOMElement()
    expect(cell(container, "A4")).toHaveTextContent("#DIV/0!")
    expect(cell(container, "A5")).toHaveTextContent("25.0%")
    expect(cell(container, "A6")).toHaveTextContent("$24.00")
    expect(cell(container, "A7")).toBeEmptyDOMElement()
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
  })

  it("reports only the selected sheet and preserves cached cells alongside missing results", () => {
    const content = workbookContent({
      Cached: { A1: { t: "n", f: "1+1", v: 2 }, "!ref": "A1" },
      Pending: {
        A1: { t: "n", f: "Cached!A1*2" },
        A2: { t: "n", f: "0+0", v: 0 },
        "!ref": "A1:A2",
      },
    })
    const { container } = render(preview(content))
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Pending" }))
    expect(screen.getByRole("status")).toHaveTextContent("Missing formula results on this sheet: 1")
    expect(cell(container, "A1")).toHaveTextContent("=Cached!A1*2")
    expect(cell(container, "A2")).toHaveTextContent(/^0$/)
    fireEvent.click(screen.getByRole("button", { name: "Cached" }))
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
    expect(cell(container, "A1")).toHaveTextContent(/^2$/)
  })

  it("handles empty value elements and array-formula followers without inventing results", async () => {
    const sheet: XLSX.WorkSheet = {
      A1: { t: "n", f: "ROW(A1:A2)", F: "A1:A2", v: 0 },
      A2: { t: "n", F: "A1:A2", v: 0 },
      "!ref": "A1:A2",
    }
    // Empty <v/> elements match uncached files produced by openpyxl. Retain an
    // explicit follower cell so the reader can associate it with the array.
    const zip = await JSZip.loadAsync(workbookContent({ Array: sheet }), { base64: true })
    const sheetPath = "xl/worksheets/sheet1.xml"
    const xml = await zip.file(sheetPath)!.async("string")
    zip.file(sheetPath, xml.replace(/<v>0<\/v>/g, "<v/>"))
    const { container } = render(preview(await zip.generateAsync({ type: "base64" }), "zh"))

    expect(cell(container, "A1")).toHaveTextContent("=ROW(A1:A2)")
    expect(cell(container, "A2")).toHaveTextContent("未计算")
    expect(screen.getByRole("status")).toHaveTextContent("2 个公式单元格未保存计算结果")
  })

  it("renders unknown and external formulas as text without executing them", () => {
    const content = workbookContent({
      Unsupported: {
        A1: { t: "n", f: '_xlfn.UNKNOWN("<img src=x onerror=alert(1)>")' },
        A2: { t: "n", f: 'WEBSERVICE("https://example.invalid/data")' },
        "!ref": "A1:A2",
      },
    })
    const { container } = render(preview(content))

    expect(cell(container, "A1")).toHaveTextContent('<img src=x onerror=alert(1)>')
    expect(cell(container, "A1")?.textContent?.startsWith("=")).toBe(true)
    expect(cell(container, "A2")).toHaveTextContent('=WEBSERVICE("https://example.invalid/data")')
    expect(container.querySelector("img")).toBeNull()
    expect(container.querySelector("script")).toBeNull()
    expect(container.querySelector("[onerror]")).toBeNull()
    expect(screen.getByRole("status")).toHaveTextContent("Missing formula results on this sheet: 2")
  })

  it("removes the warning and old cells when switching files or clearing content", () => {
    const pending = workbookContent({ Old: { A1: { t: "n", f: "2+2" }, "!ref": "A1" } })
    const { container, rerender } = render(preview(pending))
    expect(screen.getByRole("status")).toBeInTheDocument()

    rerender(preview(workbookContent({ New: XLSX.utils.aoa_to_sheet([["New value"]]) })))
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
    expect(cell(container, "A1")).toHaveTextContent("New value")
    rerender(preview(""))
    expect(container.querySelector("table")).toBeNull()
  })

  it.each([false, true])("keeps plain CSV previews working (base64=%s)", (base64) => {
    const csv = "item,quantity\nCoffee,0\nTea,7"
    const { container } = render(preview(base64 ? btoa(csv) : csv))
    expect(cell(container, "B2")).toHaveTextContent(/^0$/)
    expect(cell(container, "A3")).toHaveTextContent("Tea")
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
  })

  it("only transforms a preview copy and ignores non-formula blanks", () => {
    const sheet: XLSX.WorkSheet = {
      A1: { t: "z", f: "SUM(B1:C1)", v: 0, w: "0", h: "<b>stale</b>" },
      B1: { t: "n", v: 9 },
      C1: { t: "z" },
      A2: { t: "n", f: "B1*2" },
      "!ref": "A1:C2",
    }
    const original = JSON.stringify(sheet)
    const result = createExcelSheetPreview(sheet, "Not calculated")
    const container = document.createElement("div")
    container.innerHTML = result.html

    expect(cell(container, "A1")).toHaveTextContent("=SUM(B1:C1)")
    expect(cell(container, "A2")).toHaveTextContent("=B1*2")
    expect(cell(container, "C1")).toBeEmptyDOMElement()
    expect(container.querySelector("b")).toBeNull()
    expect(result.missingFormulaResults).toBe(2)
    expect(JSON.stringify(sheet)).toBe(original)
  })
})

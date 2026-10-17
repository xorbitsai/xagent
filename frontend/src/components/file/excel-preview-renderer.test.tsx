/// <reference types="@testing-library/jest-dom/vitest" />

import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"
import * as XLSX from "xlsx"
import JSZip from "jszip"
import { I18nProvider } from "@/contexts/i18n-context"
import { ExcelPreviewRenderer } from "./excel-preview-renderer"
import { createExcelSheetPreview } from "./excel-sheet-preview"

function workbookContent(sheets: Record<string, XLSX.WorkSheet>, bookSST = false) {
  const workbook = XLSX.utils.book_new()
  for (const [name, sheet] of Object.entries(sheets)) {
    XLSX.utils.book_append_sheet(workbook, sheet, name)
  }
  // Exercise real XLSX bytes and the real reader, not a mocked parsed workbook.
  return XLSX.write(workbook, { type: "base64", bookType: "xlsx", bookSST }) as string
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

describe("ExcelPreviewRenderer HTML safety", () => {
  afterEach(cleanup)

  it("loads the patched SheetJS release rather than the stale npm registry version", () => {
    // CVE-2023-30533 requires >=0.19.3; CVE-2024-22363 requires >=0.20.2.
    const [major, minor, patch] = XLSX.version.split(".").map(Number)
    expect(major > 0 || minor > 20 || (minor === 20 && patch >= 2)).toBe(true)
  })

  it("preserves literal XML entity text in hyperlink targets", () => {
    const target = 'https://example.com/report?a="x"&name=&quot;literal&quot;&page=2'
    const content = workbookContent({ Links: {
      A1: { t: "s", v: "Report", l: { Target: target } },
      "!ref": "A1",
    } })
    render(preview(content))
    expect(screen.getByRole("link", { name: "Report" })).toHaveAttribute("href", target)
  })

  it.each([false, true])("sanitizes cell values from real XLSX bytes (cached formula=%s)", (cached) => {
    const values = [
      '\"><img src=x onerror=alert(1)>',
      '\"><svg onload=alert(1)>',
      '\" onmouseover=\"alert(1)',
    ]
    const sheet: XLSX.WorkSheet = { "!ref": "A1:A3" }
    values.forEach((value, index) => {
      sheet[`A${index + 1}`] = { t: "s", v: value, ...(cached ? { f: '"cached"' } : {}) }
    })
    const { container } = render(preview(workbookContent({ Values: sheet })))

    expect(container.querySelector("img, svg, script, [onerror], [onload], [onmouseover]")).toBeNull()
    values.forEach((value, index) => {
      expect(cell(container, `A${index + 1}`)).toHaveTextContent(value)
    })
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
  })

  it.each([false, true])("sanitizes CSV cells (base64=%s)", (base64) => {
    const value = '\"><img src=x onerror=alert(1)>'
    const csv = `note,amount\n"${value.replace(/"/g, '""')}",7`
    const { container } = render(preview(base64 ? btoa(csv) : csv))

    expect(container.querySelector("img, [onerror]")).toBeNull()
    expect(container).toHaveTextContent(value)
    expect(cell(container, "B2")).toHaveTextContent(/^7$/)
  })

  it.each([
    { base64: false, lineBreak: "\n" },
    { base64: true, lineBreak: "\n" },
    { base64: false, lineBreak: "\r\n" },
    { base64: true, lineBreak: "\r\n" },
  ])("preserves line breaks in escaped CSV values (%j)", ({ base64, lineBreak }) => {
    const csv = `address,amount\n"Office <HQ>${lineBreak}Floor 2 & reception",7`
    const { container } = render(preview(base64 ? btoa(csv) : csv))

    expect(cell(container, "A2")?.querySelectorAll("br")).toHaveLength(1)
    expect(cell(container, "A2")).toHaveTextContent("Office <HQ>")
    expect(cell(container, "A2")).toHaveTextContent("Floor 2 & reception")
    expect(container.querySelector("hq")).toBeNull()
    expect(cell(container, "B2")).toHaveTextContent(/^7$/)
  })

  it("preserves line breaks in cached XLSX string formula results", () => {
    const content = workbookContent({ Notes: {
      A1: { t: "s", f: '"Office"&CHAR(10)&"Floor 2"', v: "Office\nFloor 2" },
      "!ref": "A1",
    } })
    const parsed = XLSX.read(content, { type: "base64", sheetStubs: true })
    expect(parsed.Sheets.Notes.A1.h).toContain("<br/>")
    const { container } = render(preview(content))

    expect(cell(container, "A1")?.querySelectorAll("br")).toHaveLength(1)
    expect(cell(container, "A1")).toHaveTextContent("Office")
    expect(cell(container, "A1")).toHaveTextContent("Floor 2")
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
  })

  it.each([
    "javascript:alert(1)",
    "JaVaScRiPt:alert(1)",
    "java\tscript:alert(1)",
    "data:text/html,<script>alert(1)</script>",
    "vbscript:msgbox(1)",
  ])("removes unsafe hyperlink targets: %s", (target) => {
    const content = workbookContent({ Links: {
      A1: { t: "s", v: "Link label", l: { Target: target } },
      "!ref": "A1",
    } })
    const { container } = render(preview(content))

    expect(cell(container, "A1")).toHaveTextContent("Link label")
    expect(container.querySelector("a[href]")).toBeNull()
  })

  it("removes attributes injected through hyperlink targets", () => {
    const target = 'https://example.com/\" onclick=\"alert(1)'
    const content = workbookContent({ Links: {
      A1: { t: "s", v: "Link label", l: { Target: target } },
      "!ref": "A1",
    } })
    const { container } = render(preview(content))

    expect(container.querySelector("[onclick]")).toBeNull()
    expect(cell(container, "A1")).toHaveTextContent("Link label")
    // The payload stays inert URL text, not a separate event-handler attribute.
    expect(cell(container, "A1")?.querySelector("a[href]")).toHaveAttribute("href", target)
  })

  it("preserves merged cells, formatted values, safe links and formula warnings", () => {
    const content = workbookContent({ Report: {
      A1: { t: "s", v: "Report & totals" },
      A2: { t: "n", v: 12.5, z: "$0.00" },
      B2: { t: "n", v: 0.25, z: "0.0%" },
      A3: { t: "s", v: "Source", l: { Target: "https://example.com/report?a=1&b=2" } },
      B3: { t: "s", v: "Contact", l: { Target: "mailto:reports@example.com" } },
      A4: { t: "n", f: "SUM(A2:B2)" },
      "!ref": "A1:B4",
      "!merges": [XLSX.utils.decode_range("A1:B1")],
    } })
    const { container } = render(preview(content))

    expect(cell(container, "A1")).toHaveAttribute("colspan", "2")
    expect(cell(container, "A1")).toHaveTextContent("Report & totals")
    expect(cell(container, "A2")).toHaveAttribute("data-t", "n")
    expect(cell(container, "A2")).toHaveTextContent("$12.50")
    expect(cell(container, "B2")).toHaveTextContent("25.0%")
    expect(screen.getByRole("link", { name: "Source" })).toHaveAttribute("href", "https://example.com/report?a=1&b=2")
    expect(screen.getByRole("link", { name: "Contact" })).toHaveAttribute("href", "mailto:reports@example.com")
    expect(cell(container, "A4")).toHaveTextContent("=SUM(A2:B2)")
    expect(screen.getByRole("status")).toHaveTextContent("Missing formula results on this sheet: 1")
  })

  it("cleans rich HTML and keeps text formatting without mutating the worksheet", () => {
    const sheet: XLSX.WorkSheet = {
      A1: {
        t: "s", v: "Rich text",
        h: '<b>Rich</b> <i>text</i><style>body{display:none}</style><img src=x onerror=alert(1)><svg onload=alert(1)></svg><iframe srcdoc="bad"></iframe><form><input autofocus onfocus=alert(1)></form><span style="position:fixed" onclick="alert(1)"> label</span>',
      },
      "!ref": "A1",
    }
    const original = JSON.stringify(sheet)
    const result = createExcelSheetPreview(sheet, "Not calculated")
    const container = document.createElement("div")
    container.innerHTML = result.html

    expect(container.querySelector("b")).toHaveTextContent("Rich")
    expect(container.querySelector("i")).toHaveTextContent("text")
    expect(container.querySelector("style, img, svg, iframe, form, input, [style], [onclick]")).toBeNull()
    expect(container).toHaveTextContent("label")
    expect(JSON.stringify(sheet)).toBe(original)
  })

  it.each([false, true])("keeps rich-text table markup inside its cell from real XLSX bytes (shared strings=%s)", async (shared) => {
    const content = workbookContent({ Report: XLSX.utils.aoa_to_sheet([
      ["Rich text", "Neighbor"],
      ["Lower", 7],
    ]) }, shared)
    const zip = await JSZip.loadAsync(content, { base64: true })
    const payload = '</td><td id="forged-cell">Injected</td><td><table><tr><td>Nested</td></tr></table>'
    const escapedPayload = payload.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    const richText = `<r><rPr><b/></rPr><t>Rich text</t></r><r><t>${escapedPayload}</t></r>`
    if (shared) {
      const path = "xl/sharedStrings.xml"
      const xml = await zip.file(path)!.async("string")
      zip.file(path, xml.replace("<si><t>Rich text</t></si>", `<si>${richText}</si>`))
    } else {
      const path = "xl/worksheets/sheet1.xml"
      const xml = await zip.file(path)!.async("string")
      zip.file(path, xml.replace(/<c\b[^>]*\br="A1"[^>]*>[\s\S]*?<\/c>/, `<c r="A1" t="inlineStr"><is>${richText}</is></c>`))
    }
    const maliciousContent = await zip.generateAsync({ type: "base64" })
    // Verify that the actual reader, not a hand-built cell.h, emits the markup.
    const parsed = XLSX.read(maliciousContent, { type: "base64", sheetStubs: true })
    expect(parsed.Sheets.Report.A1.h).toContain(payload)
    const { container } = render(preview(maliciousContent))

    expect(container.querySelectorAll("table")).toHaveLength(1)
    expect(container.querySelectorAll("tr")).toHaveLength(2)
    expect(container.querySelectorAll("td")).toHaveLength(4)
    expect(container.querySelector("#forged-cell")).toBeNull()
    expect(cell(container, "A1")?.querySelector("b")).toHaveTextContent("Rich text")
    expect(cell(container, "A1")).toHaveTextContent("Injected")
    expect(cell(container, "A1")).toHaveTextContent("Nested")
    expect(cell(container, "B1")).toHaveTextContent(/^Neighbor$/)
    expect(cell(container, "A2")).toHaveTextContent(/^Lower$/)
    expect(cell(container, "B2")).toHaveTextContent(/^7$/)
  })
})

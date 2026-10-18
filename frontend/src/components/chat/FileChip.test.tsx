import { describe, expect, it } from "vitest"
import { createFileChipHTML, renderFileMentionsHtml } from "./FileChip"

function fragment(html: string) {
  const root = document.createElement("div")
  root.innerHTML = html
  return root
}

function expectInert(root: HTMLElement) {
  expect(root.querySelector("script,img,iframe,object,embed,style,link")).toBeNull()
  for (const element of root.querySelectorAll("*")) {
    expect([...element.attributes].filter((attr) => /^on/i.test(attr.name))).toEqual([])
  }
}

describe("file chip HTML boundary", () => {
  it.each([
    '<img src=x onerror="alert(1)">',
    '"><svg onload="alert(1)"><script>alert(1)</script>',
    "' autofocus onfocus='alert(1)",
    "A&B <预算> \"quoted\" 'name'.txt",
    "literal &lt;img src=x onerror=alert(1)&gt;.txt",
    "`backticks` [file](file:other)",
  ])("keeps raw metadata inert and lossless: %s", (value) => {
    const root = fragment(createFileChipHTML(value, value, value, value))
    const chip = root.querySelector("[data-file-path]")!
    expect(chip.getAttribute("data-file-path")).toBe(value)
    expect(chip.getAttribute("data-file-id")).toBe(value)
    expect(chip.getAttribute("data-filename")).toBe(value)
    expect(chip.textContent?.trim()).toBe(value)
    expectInert(root)
  })

  it("escapes surrounding text once and preserves canonical file metadata", () => {
    const root = fragment(renderFileMentionsHtml('A&B <plain> [预算 & <Q>.txt](file:id&<Q>)'))
    const chip = root.querySelector("[data-file-path]")!
    expect(root.textContent).toContain("A&B <plain>")
    expect(chip.getAttribute("data-filename")).toBe("预算 & <Q>.txt")
    expect(chip.getAttribute("data-file-path")).toBe("id&<Q>")
    expectInert(root)
  })

  it("never reparses backticks from a canonical filename as generated HTML", () => {
    const root = fragment(renderFileMentionsHtml('[a`b`&<c>.txt](file:one) and `legacy&<file>.txt`', true))
    const chips = root.querySelectorAll("[data-file-path]")
    expect(chips).toHaveLength(2)
    expect(chips[0].getAttribute("data-filename")).toBe("a`b`&<c>.txt")
    expect(chips[1].getAttribute("data-file-path")).toBe("legacy&<file>.txt")
    expectInert(root)
  })

  it("prioritizes canonical links even after a stray backtick", () => {
    const root = fragment(renderFileMentionsHtml('don`t forget [report.pdf](file:abc) and `npm i`', true))
    const chips = root.querySelectorAll("[data-file-path]")
    expect(chips).toHaveLength(2)
    expect(chips[0].getAttribute("data-file-path")).toBe("abc")
    expect(chips[0].getAttribute("data-filename")).toBe("report.pdf")
    expect(chips[1].getAttribute("data-file-path")).toBe("npm i")
    expect(root.textContent).toContain('don`t forget ')
    expectInert(root)
  })

  it("does not interpret legacy backticks in builder instructions", () => {
    const root = fragment(renderFileMentionsHtml('use `<img src=x onerror=alert(1)>`'))
    expect(root.querySelector("[data-file-path]")).toBeNull()
    expect(root.textContent).toBe('use `<img src=x onerror=alert(1)>`')
    expectInert(root)
  })
})

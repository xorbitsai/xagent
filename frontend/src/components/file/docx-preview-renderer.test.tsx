/// <reference types="@testing-library/jest-dom/vitest" />

import React from "react"
import { cleanup, render, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import JSZip from "jszip"
import * as docxPreview from "docx-preview"

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

import { DocxPreviewRenderer } from "./docx-preview-renderer"
import {
  renderSanitizedDocx,
  sanitizeDocxCss,
  sanitizeDocxPreview,
} from "./docx-preview-sanitize"

const W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
const R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
const REL = "http://schemas.openxmlformats.org/package/2006/relationships"
// 1x1 transparent PNG.
const PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

function escapeXml(text: string) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
}

async function maliciousDocx(): Promise<ArrayBuffer> {
  const symbolPayload = `41;<img src="x" onerror="window.__docxXss=1">`
  const documentXml = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="${W}" xmlns:r="${R}"
  xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
  xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
  <w:body>
    <w:p><w:r><w:t>Quarterly summary</w:t></w:r></w:p>
    <w:p><w:hyperlink r:id="rIdEvil"><w:r><w:t>evil link</w:t></w:r></w:hyperlink></w:p>
    <w:p><w:hyperlink r:id="rIdSafe"><w:r><w:t>safe link</w:t></w:r></w:hyperlink></w:p>
    <w:p><w:hyperlink w:anchor="intro"><w:r><w:t>jump</w:t></w:r></w:hyperlink></w:p>
    <w:p><w:r><w:sym w:font="Symbol" w:char="${escapeXml(symbolPayload)}"/></w:r></w:p>
    <w:tbl>
      <w:tr><w:tc><w:p><w:r><w:t>Cell A1</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Cell B1</w:t></w:r></w:p></w:tc></w:tr>
    </w:tbl>
    <w:p><w:r><w:drawing><wp:inline><wp:extent cx="9525" cy="9525"/>
      <a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
        <pic:pic><pic:blipFill><a:blip r:embed="rIdImg"/></pic:blipFill>
          <pic:spPr><a:xfrm><a:ext cx="9525" cy="9525"/></a:xfrm></pic:spPr></pic:pic>
      </a:graphicData></a:graphic>
    </wp:inline></w:drawing></w:r></w:p>
  </w:body>
</w:document>`

  const zip = new JSZip()
  zip.file("_rels/.rels", `<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="${REL}">
  <Relationship Id="rId1" Type="${R}/officeDocument" Target="word/document.xml"/>
</Relationships>`)
  zip.file("word/document.xml", documentXml)
  zip.file("word/_rels/document.xml.rels", `<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="${REL}">
  <Relationship Id="rIdEvil" Type="${R}/hyperlink" Target="javascript:alert(document.domain)" TargetMode="External"/>
  <Relationship Id="rIdSafe" Type="${R}/hyperlink" Target="https://example.com/report" TargetMode="External"/>
  <Relationship Id="rIdImg" Type="${R}/image" Target="media/image1.png"/>
</Relationships>`)
  zip.file("word/media/image1.png", PNG_BASE64, { base64: true })
  return zip.generateAsync({ type: "arraybuffer" })
}

function arrayBufferToBase64(buffer: ArrayBuffer) {
  return Buffer.from(new Uint8Array(buffer)).toString("base64")
}

function staged(html: string) {
  const container = document.createElement("div")
  // Test fixture only: a detached element in jsdom, which loads no resources.
  container.innerHTML = html
  return container
}

function wrap(fragment: DocumentFragment) {
  const div = document.createElement("div")
  div.appendChild(fragment)
  return div
}

describe("sanitizeDocxPreview", () => {
  it("neutralizes active content while keeping document structure", () => {
    const output = wrap(sanitizeDocxPreview(staged(`
      <section class="docx-preview"><article>
        <p class="docx-preview_normal">Plain paragraph</p>
        <a href="javascript:alert(1)">js link</a>
        <a href="  JaVaScRiPt:alert(1)">spaced js link</a>
        <a href="data:text/html,<script>alert(1)</script>">data link</a>
        <a href="https://example.com/a">https link</a>
        <a href="mailto:team@example.com">mail link</a>
        <a href="#bookmark">bookmark</a>
        <a href="relative/path">relative link</a>
        <img src="x" onerror="alert(1)">
        <img src="data:image/png;base64,${PNG_BASE64}" onload="alert(1)" style="width: 10px">
        <img src="https://tracker.example/pixel.gif">
        <svg onload="alert(1)"><rect width="10" height="10" fill="url(https://evil.example/p.svg#x)" onclick="alert(1)"></rect></svg>
        <script>alert(1)</script>
        <iframe srcdoc="<script>alert(1)</script>"></iframe>
        <object data="evil.swf"></object><embed src="evil.swf">
        <form action="https://evil.example"><input name="pw"><button>Go</button></form>
        <table><tbody><tr><td colspan="2">Cell</td></tr></tbody></table>
        <span id="bookmark" lang="en-US">Anchor target</span>
      </article></section>`)))

    expect(output.querySelector("script, iframe, object, embed, form, input, button")).toBeNull()
    for (const element of Array.from(output.querySelectorAll("*"))) {
      for (const attr of Array.from(element.attributes)) {
        expect(attr.name.startsWith("on"), `${element.tagName} ${attr.name}`).toBe(false)
      }
    }

    const links = Array.from(output.querySelectorAll("a"))
    const hrefs = links.map(link => link.getAttribute("href"))
    expect(hrefs).toEqual([null, null, null, "https://example.com/a", "mailto:team@example.com", "#bookmark", null])
    const external = links.find(link => link.getAttribute("href") === "https://example.com/a")
    expect(external).toHaveAttribute("target", "_blank")
    expect(external).toHaveAttribute("rel", "noopener noreferrer")
    const bookmark = links.find(link => link.getAttribute("href") === "#bookmark")
    expect(bookmark).not.toHaveAttribute("target")

    const images = Array.from(output.querySelectorAll("img"))
    expect(images).toHaveLength(3)
    expect(images.map(image => image.getAttribute("src"))).toEqual([
      null,
      `data:image/png;base64,${PNG_BASE64}`,
      null,
    ])
    expect(images[1]).toHaveStyle({ width: "10px" })

    expect(output.querySelector("svg rect")).toHaveAttribute("fill", "none")
    expect(output.querySelector("p.docx-preview_normal")).toHaveTextContent("Plain paragraph")
    expect(output.querySelector("td[colspan='2']")).toHaveTextContent("Cell")
    expect(output.querySelector("#bookmark")).toHaveAttribute("lang", "en-US")
  })

  it("keeps docx-preview styles but strips CSS that can load or run anything", () => {
    const output = wrap(sanitizeDocxPreview(staged(`<style>
      @import url(https://evil.example/a.css);
      .docx-preview p { margin: 0pt; min-height: 1em; }
      .docx-preview { background: url(https://evil.example/bg.png); }
      .docx-preview span { background: u\\72l(https://evil.example/escaped.png); }
      @font-face { font-family: Embedded; src: url(data:application/octet-stream;base64,AAAA); }
    </style><p style="background-image: url('https://evil.example/inline.png'); color: red">Text</p>`)))

    const css = output.querySelector("style")?.textContent ?? ""
    expect(css).toContain(".docx-preview p { margin: 0pt; min-height: 1em; }")
    expect(css).toContain('src: url("data:application/octet-stream;base64,AAAA")')
    expect(css).not.toMatch(/@import|url\(\s*(?!["']?data:)/i)
    expect(css).toContain("background: none;")
    const paragraph = output.querySelector("p")
    expect(paragraph?.getAttribute("style")).not.toContain("evil.example")
    expect(paragraph).toHaveStyle({ color: "rgb(255, 0, 0)" })
  })
})

describe("sanitizeDocxCss", () => {
  it.each([
    ["url(javascript:alert(1))", "none)"],
    ["url( 'https://evil.example/x' )", "none"],
    ["image-set('https://evil.example/x' 1x)", "invalid('https://evil.example/x' 1x)"],
    ["src('https://evil.example/x')", "invalid('https://evil.example/x')"],
    // ASCII escapes are dropped, so they cannot spell url( or @import.
    ["@\\69mport 'https://evil.example/a.css';", "@mport 'https://evil.example/a.css';"],
    ["u\\72l(https://evil.example/x)", "ul(https://evil.example/x)"],
    ["u\\\\72l(https://evil.example/x)", "u72l(https://evil.example/x)"],
    ["u\\41rl(https://evil.example/x)", "none"],
    ["ur/**/l(https://evil.example/x)", "none"],
    ["content: \"</style><script>\"", "content: \"\\3c /style>\\3c script>\""],
    ["content: \"\\9\"; content: \"\\a0\"", "content: \"\\9\"; content: \"\\a0\""],
    ["src: url(data:font/woff2;base64,AAAA)", "src: url(\"data:font/woff2;base64,AAAA\")"],
    ["color: red", "color: red"],
  ])("rewrites %s", (input, expected) => {
    expect(sanitizeDocxCss(input)).toBe(expected)
  })
})

describe("renderSanitizedDocx", () => {
  it("renders a hostile .docx without unsafe markup reaching the staging or output DOM", async () => {
    let staging: HTMLElement | null = null
    const fragment = await renderSanitizedDocx({
      parseAsync: docxPreview.parseAsync,
      renderDocument: (model, container, styleContainer, options) => {
        staging = container
        return docxPreview.renderDocument(model, container, styleContainer, options)
      },
    }, await maliciousDocx())

    // The w:sym innerHTML sink must not have produced markup, even detached.
    expect(staging).not.toBeNull()
    expect(staging!.isConnected).toBe(false)
    expect(staging!.querySelector("img[onerror]")).toBeNull()

    const output = wrap(fragment)
    expect(output.querySelector("[onerror], script, iframe")).toBeNull()
    expect(output.innerHTML).not.toContain("javascript:")
    expect(output).toHaveTextContent("Quarterly summary")
    expect(output.querySelector("table")).toHaveTextContent("Cell A1")
    expect(output.querySelector("table")).toHaveTextContent("Cell B1")
    expect(output.querySelector("style")).not.toBeNull()
    expect(output.querySelector(".docx-preview-wrapper section.docx-preview")).not.toBeNull()

    const links = Array.from(output.querySelectorAll("a"))
    expect(links.find(link => link.textContent === "evil link")).not.toHaveAttribute("href")
    expect(links.find(link => link.textContent === "safe link")).toHaveAttribute("href", "https://example.com/report")
    expect(links.find(link => link.textContent === "safe link")).toHaveAttribute("rel", "noopener noreferrer")
    expect(links.find(link => link.textContent === "jump")).toHaveAttribute("href", "#intro")

    const image = output.querySelector("img")
    expect(image?.getAttribute("src")).toMatch(/^data:/)
  })
})

describe("DocxPreviewRenderer", () => {
  afterEach(() => {
    cleanup()
  })

  it("mounts only sanitized content inside a contained shadow root", async () => {
    const content = arrayBufferToBase64(await maliciousDocx())
    const { getByTestId } = render(<DocxPreviewRenderer base64Content={content} />)
    const host = getByTestId("docx-preview-host")

    await waitFor(() => expect(host.shadowRoot?.textContent).toContain("Quarterly summary"))
    const root = host.shadowRoot!
    expect(host.childNodes).toHaveLength(0)
    expect(host.parentElement).toHaveStyle({ contain: "layout paint" })
    expect(root.querySelector("[onerror], script, iframe")).toBeNull()
    expect(Array.from(root.querySelectorAll("a")).map(link => link.getAttribute("href")))
      .not.toContain("javascript:alert(document.domain)")
    expect((window as { __docxXss?: number }).__docxXss).toBeUndefined()
  })
})

import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { MarkdownDescription } from "./markdown-description"

describe("untrusted connector descriptions", () => {
  it("renders no element for an empty description", () => {
    const { container } = render(<MarkdownDescription content="" />)
    expect(container.firstChild).toBeNull()
  })

  it("renders formatting as React elements", () => {
    const { container } = render(<MarkdownDescription content="**Trusted-looking** and `code`" />)
    expect(container.querySelector("strong")?.textContent).toBe("Trusted-looking")
    expect(container.querySelector("code")?.textContent).toBe("code")
  })

  it("never inserts raw HTML, executable links, remote images or embedded documents", () => {
    const { container } = render(<MarkdownDescription content={[
      '<img src=x onerror="window.compromised=true">',
      '<script>window.compromised=true</script>',
      '<iframe srcdoc="<script>alert(1)</script>"></iframe>',
      '[bad](javascript:alert%281%29)',
      '[data](data:text/html;base64,PHNjcmlwdD4=)',
      '![tracking](https://attacker.example/pixel)',
    ].join("\n\n")} />)
    expect(container.querySelector("script,img,iframe,object,embed,[onerror]")).toBeNull()
    expect(screen.getByText("bad").getAttribute("href") || "").not.toMatch(/^javascript:/i)
    expect(screen.getByText("data").getAttribute("href") || "").not.toMatch(/^data:/i)
  })

  it("isolates safe outbound links and drops prior content on change", () => {
    const { rerender, container } = render(<MarkdownDescription content="[docs](https://example.com)" />)
    expect(screen.getByRole("link").getAttribute("rel")).toBe("noopener noreferrer")
    rerender(<MarkdownDescription content="replacement" />)
    expect(container.querySelector("a")).toBeNull()
    expect(screen.getByText("replacement")).toBeTruthy()
  })
})

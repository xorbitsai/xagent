import React from "react"
import { renderToString } from "react-dom/server"
import { afterEach, describe, expect, it, vi } from "vitest"

vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t: (key: string) => key }) }))

import { viewport } from "@/app/widget/chat/link/page"

import { LinkSessionPage } from "./link-session-page"

describe("LinkSessionPage rendered on the server", () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  // The standalone Node server renders this page for /widget/chat/link, where
  // no window exists.
  it("renders the loading state without a window", () => {
    vi.stubGlobal("window", undefined)

    let html = ""
    expect(() => {
      html = renderToString(<LinkSessionPage />)
    }).not.toThrow()
    expect(html).toContain("widgetLink.loading")
    expect(html).not.toContain("<iframe")
  })
})

describe("link page entry", () => {
  // The page insets its content with env(safe-area-inset-*), which iOS reports
  // as 0 unless the viewport covers the whole screen.
  it("exports a viewport that covers the whole screen", () => {
    expect(viewport).toEqual({ width: "device-width", initialScale: 1, viewportFit: "cover" })
  })
})

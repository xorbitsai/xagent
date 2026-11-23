import React from "react"
import { renderToString } from "react-dom/server"
import { afterEach, describe, expect, it, vi } from "vitest"

vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t: (key: string) => key }) }))

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

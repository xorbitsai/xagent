import React from "react"
import { render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { I18nProvider, useI18n } from "./i18n-context"

function TranslationProbe() {
  const { t, tDynamic } = useI18n()
  return (
    <>
      <span>{t("tools.mcp.runtime.title")}</span>
      <span>{tDynamic("tools.mcp.runtime.missing", "Unavailable")}</span>
      <span>{tDynamic("tools.mcp.runtime.missing", "Unavailable")}</span>
    </>
  )
}

describe("I18nProvider", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("resolves typed keys and reports a missing dynamic key once", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined)

    render(
      <I18nProvider initialLocale="en">
        <TranslationProbe />
      </I18nProvider>,
    )

    expect(screen.getByText("Runtime Inputs")).toBeInTheDocument()
    expect(screen.getAllByText("Unavailable")).toHaveLength(2)
    expect(warn).toHaveBeenCalledTimes(1)
    expect(warn).toHaveBeenCalledWith(
      "Missing translation key: tools.mcp.runtime.missing",
    )
  })
})

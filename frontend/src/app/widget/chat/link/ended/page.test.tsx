import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { translations } from "@/i18n/translations"

let reason: string | null = null
vi.mock("next/navigation", () => ({
  useSearchParams: () => ({ get: (name: string) => (name === "reason" ? reason : null) }),
}))
vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key.split(".").reduce<unknown>((node, part) => (node as Record<string, unknown>)[part], translations.en),
  }),
}))

import LinkEndedPage from "./page"

const en = translations.en.widgetLink.ended

describe("link end page", () => {
  afterEach(cleanup)

  it.each([
    ["link_invalid", en.link_invalid],
    ["session_ended", en.session_ended],
    ["unavailable", en.unavailable],
    ["network", en.network],
    ["closed", en.closed],
  ])("shows its own copy for reason %s", (value, copy) => {
    reason = value
    render(<LinkEndedPage />)
    expect(screen.getByRole("heading")).toHaveTextContent(copy.title)
    expect(screen.getByText(copy.description)).toBeInTheDocument()
  })

  it.each([["unknown_x"], [""], [null], ["CLOSED"]])("shows the generic copy for reason %s", (value) => {
    reason = value
    render(<LinkEndedPage />)
    expect(screen.getByRole("heading")).toHaveTextContent(en.generic.title)
    expect(screen.getByText(en.generic.description)).toBeInTheDocument()
  })

  it("has distinct copy for the five reasons and the generic case", () => {
    const titles = Object.values(en).map((copy) => copy.title)
    // unavailable intentionally repeats the chat page's own wording.
    expect(new Set(titles).size).toBe(titles.length)
  })
})

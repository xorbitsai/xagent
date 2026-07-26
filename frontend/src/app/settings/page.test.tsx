import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const i18nMock = vi.hoisted(() => ({
  setLocale: vi.fn(),
  t: (key: string) => key,
}))

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/lib/utils")>()
  return {
    ...original,
    getApiUrl: () => "http://api.local",
  }
})

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({
    user: {
      id: "user-1",
      username: "alice",
      email: "alice@example.com",
    },
  }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    locale: "en",
    ...i18nMock,
  }),
}))

vi.mock("@/components/settings/browser-relay", () => ({
  BrowserRelaySettings: () => <section id="browser-relay">browser-relay-panel</section>,
}))

vi.mock("@/components/settings/desktop-relay", () => ({
  DesktopRelaySettings: () => <section id="desktop-relay">desktop-relay-panel</section>,
}))

import SettingsPage from "./page"

describe("SettingsPage tabs", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", "/settings")
    apiRequestMock.mockReset()
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          success: true,
          user: {
            id: "user-1",
            username: "alice",
            email: "alice@example.com",
          },
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    )
  })

  afterEach(() => {
    cleanup()
  })

  it("keeps account settings on the General tab", () => {
    render(<SettingsPage />)

    expect(screen.getByRole("tab", { name: "settings.tabs.general" }))
      .toHaveAttribute("aria-selected", "true")
    expect(screen.getAllByText("settings.language.title")).not.toHaveLength(0)
    expect(screen.queryByText("browser-relay-panel")).not.toBeInTheDocument()
  })

  it("shows both local device connections on the Computer access tab", () => {
    render(<SettingsPage />)

    fireEvent.mouseDown(
      screen.getByRole("tab", { name: "settings.tabs.computerAccess" }),
      { button: 0, ctrlKey: false },
    )

    expect(screen.getByText("browser-relay-panel")).toBeInTheDocument()
    expect(screen.getByText("desktop-relay-panel")).toBeInTheDocument()
    expect(window.location.search).toBe("?tab=computer")
  })

  it("opens legacy device anchors on the Computer access tab", async () => {
    window.history.replaceState({}, "", "/settings#desktop-relay")

    render(<SettingsPage />)

    await waitFor(() => {
      expect(
        screen.getByRole("tab", { name: "settings.tabs.computerAccess" }),
      ).toHaveAttribute("aria-selected", "true")
    })
    expect(screen.getByText("desktop-relay-panel")).toBeInTheDocument()
    expect(screen.queryByText("settings.language.title")).not.toBeInTheDocument()
  })
})

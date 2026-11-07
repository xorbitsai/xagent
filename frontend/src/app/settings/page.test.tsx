import { cleanup, render, waitFor } from "@testing-library/react"
import React from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import SettingsPage from "./page"

const navigationState = vi.hoisted(() => ({ replace: vi.fn() }))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: navigationState.replace }),
}))

afterEach(() => {
  cleanup()
  navigationState.replace.mockReset()
})

describe("SettingsPage", () => {
  it("opens the Voice settings dialog from a direct settings link", async () => {
    window.history.replaceState({}, "", "/settings?tab=voice")
    render(<SettingsPage />)
    await waitFor(() => expect(navigationState.replace).toHaveBeenCalledWith("/task?settings=voice"))
  })
  it("redirects the legacy settings route to the Computer use dialog", async () => {
    window.history.replaceState({}, "", "/settings?tab=computer#desktop-relay")

    render(<SettingsPage />)

    await waitFor(() => expect(navigationState.replace).toHaveBeenCalledWith(
      "/task?settings=computer#desktop-relay",
    ))
  })
})

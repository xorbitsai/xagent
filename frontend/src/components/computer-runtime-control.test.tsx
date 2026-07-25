import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", () => ({
  cn: (...classes: Array<string | false | null | undefined>) =>
    classes.filter(Boolean).join(" "),
  getApiUrl: () => "http://api.local",
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
  }),
}))

import { ComputerRuntimeControl } from "./computer-runtime-control"

describe("ComputerRuntimeControl", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          targets: {
            extension_relay: {
              runtime_kind: "extension_relay",
              ready: true,
              connected: true,
              attached: true,
              issues: [],
            },
            desktop_relay: {
              runtime_kind: "desktop_relay",
              ready: false,
              connected: true,
              attached: true,
              issues: [
                {
                  code: "accessibility_permission_missing",
                  message: "Accessibility permission is missing.",
                },
              ],
            },
          },
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }
      )
    )
  })

  afterEach(() => {
    cleanup()
  })

  it("shows live readiness without blocking target selection", async () => {
    const onValueChange = vi.fn()
    render(
      <ComputerRuntimeControl
        value="extension_relay"
        onValueChange={onValueChange}
      />
    )

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/computer/readiness"
      )
    })
    expect(
      await screen.findByLabelText("computerRuntime.status.ready")
    ).toBeInTheDocument()

    fireEvent.click(screen.getByTitle("computerRuntime.title"))
    expect(
      screen.getByText(
        "computerRuntime.issues.accessibility_permission_missing"
      )
    ).toBeInTheDocument()

    fireEvent.click(screen.getByText("computerRuntime.desktop.label"))
    expect(onValueChange).toHaveBeenCalledWith("desktop_relay")
  })

  it("links setup guidance to the selected relay settings", async () => {
    render(
      <ComputerRuntimeControl
        value="desktop_relay"
        onValueChange={vi.fn()}
      />
    )

    fireEvent.click(screen.getByTitle("computerRuntime.title"))

    const link = screen.getByRole("link", {
      name: "computerRuntime.manageConnections",
    })
    expect(link).toHaveAttribute("href", "/settings#desktop-relay")
  })

  it("keeps a bound target inspectable but not changeable", () => {
    const onValueChange = vi.fn()
    render(
      <ComputerRuntimeControl
        value="desktop_relay"
        onValueChange={onValueChange}
        disabled
      />
    )

    fireEvent.click(
      screen.getByTitle("computerRuntime.boundHint")
    )
    fireEvent.click(screen.getByText("computerRuntime.browser.label"))

    expect(onValueChange).not.toHaveBeenCalled()
    expect(
      screen.getByRole("link", {
        name: "computerRuntime.manageConnections",
      })
    ).toHaveAttribute("href", "/settings#desktop-relay")
  })
})

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

import { ComposerAddMenu } from "./computer-runtime-control"

describe("ComposerAddMenu", () => {
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

  it("grants a ready local target from the optional add menu", async () => {
    const onValueChange = vi.fn()
    render(
      <ComposerAddMenu
        onValueChange={onValueChange}
      />
    )

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/computer/readiness"
      )
    })
    fireEvent.click(
      screen.getByRole("button", {
        name: "computerRuntime.addMenu.title",
      })
    )
    expect(
      screen.queryByText("computerRuntime.browser.label")
    ).not.toBeInTheDocument()
    fireEvent.click(
      screen.getByRole("button", {
        name: "computerRuntime.addMenu.computerAccess",
      })
    )
    expect(
      await screen.findByText("computerRuntime.status.ready")
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        "computerRuntime.issues.accessibility_permission_missing"
      )
    ).toBeInTheDocument()

    fireEvent.click(screen.getByText("computerRuntime.browser.label"))
    expect(onValueChange).toHaveBeenCalledWith("extension_relay")
  })

  it("takes an unavailable target to its connection settings", async () => {
    render(
      <ComposerAddMenu
        onValueChange={vi.fn()}
      />
    )

    fireEvent.click(
      screen.getByRole("button", {
        name: "computerRuntime.addMenu.title",
      })
    )
    fireEvent.click(
      screen.getByRole("button", {
        name: "computerRuntime.addMenu.computerAccess",
      })
    )

    const link = await screen.findByRole("link", {
      name: /computerRuntime\.desktop\.label/,
    })
    expect(link).toHaveAttribute(
      "href",
      "/settings?tab=computer#desktop-relay"
    )
  })

  it("shows a removable chip for an explicit task grant", async () => {
    const onValueChange = vi.fn()
    render(
      <ComposerAddMenu
        value="desktop_relay"
        onValueChange={onValueChange}
      />
    )

    fireEvent.click(
      screen.getByRole("button", {
        name: "computerRuntime.removeAccess",
      })
    )
    expect(onValueChange).toHaveBeenCalledWith(undefined)
  })

  it("keeps an existing task grant inspectable but not changeable", () => {
    const onValueChange = vi.fn()
    render(
      <ComposerAddMenu
        value="desktop_relay"
        onValueChange={onValueChange}
        selectionLocked
      />
    )

    fireEvent.click(
      screen.getByRole("button", {
        name: "computerRuntime.addMenu.title",
      })
    )
    fireEvent.click(
      screen.getByRole("button", {
        name: "computerRuntime.addMenu.computerAccess",
      })
    )
    fireEvent.click(screen.getByText("computerRuntime.browser.label"))

    expect(onValueChange).not.toHaveBeenCalled()
    expect(
      screen.getByRole("link", {
        name: "computerRuntime.manageConnections",
      })
    ).toHaveAttribute("href", "/settings?tab=computer#desktop-relay")
  })
})

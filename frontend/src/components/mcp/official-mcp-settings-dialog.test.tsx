import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { OfficialMcpSettingsDialog } from "./official-mcp-settings-dialog"
import type { AppIntegration } from "./types"

const apiRequestMock = vi.hoisted(() => vi.fn())
const translateMock = vi.hoisted(() => (key: string) => key)

vi.mock("@/lib/api-wrapper", () => ({ apiRequest: apiRequestMock }))

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return { ...actual, getApiUrl: () => "http://api.local" }
})

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ token: "token", inTeam: false }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: translateMock }),
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: { error: vi.fn(), success: vi.fn() },
}))

vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ open, children }: { open: boolean; children: React.ReactNode }) =>
    open ? <div role="dialog">{children}</div> : null,
  DialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogTitle: ({ children }: { children: React.ReactNode }) => <h1>{children}</h1>,
}))

function app(overrides: Partial<AppIntegration> = {}): AppIntegration {
  return {
    id: "chrome",
    name: "Chrome",
    description: "Automate a Chrome browser.",
    icon: "",
    is_connected: true,
    transport: "stdio",
    auth_type: "keyless",
    ...overrides,
  } as AppIntegration
}

function renderDialog(props: Partial<React.ComponentProps<typeof OfficialMcpSettingsDialog>> = {}) {
  return render(
    <OfficialMcpSettingsDialog
      open
      onOpenChange={vi.fn()}
      app={app()}
      isGloballyConnected
      {...props}
    />,
  )
}

describe("OfficialMcpSettingsDialog connected-state actions", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it("offers only disconnect for a connected keyless app", () => {
    renderDialog()

    // No key to manage and no editable config — the configure/manage-key
    // button must be suppressed for keyless apps.
    expect(
      screen.queryByRole("button", { name: /manageKey|configure/ }),
    ).not.toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /disconnect/ }),
    ).toBeInTheDocument()
  })

  it("still offers manage-key for a connected api_key app", () => {
    const onManageKey = vi.fn()
    renderDialog({
      app: app({ id: "google-maps", name: "Google Maps", auth_type: "api_key" }),
      onManageKey,
    })

    const manageButton = screen.getByRole("button", { name: /manageKey/ })
    fireEvent.click(manageButton)
    expect(onManageKey).toHaveBeenCalledTimes(1)
  })
})

describe("OfficialMcpSettingsDialog connect trigger", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it("fires onConnectStart once and shows a pending state while connecting", () => {
    const onConnectStart = vi.fn()
    const { rerender } = renderDialog({
      app: app({ is_connected: false }),
      isGloballyConnected: false,
      onConnectStart,
    })

    const connectButton = screen.getByRole("button", { name: /dialog\.connect$/ })
    fireEvent.click(connectButton)
    expect(onConnectStart).toHaveBeenCalledTimes(1)

    // Parent flips isConnecting while its POST is in flight: the trigger
    // must disable (rapid double-clicks fired overlapping requests before)
    // and show the pending label — the catalog card spinner is hidden
    // behind this dialog, so this is the flow's only visible feedback.
    rerender(
      <OfficialMcpSettingsDialog
        open
        onOpenChange={vi.fn()}
        app={app({ is_connected: false })}
        isGloballyConnected={false}
        onConnectStart={onConnectStart}
        isConnecting
      />,
    )

    const pendingButton = screen.getByRole("button", { name: /connecting/ })
    expect(pendingButton).toBeDisabled()
    fireEvent.click(pendingButton)
    expect(onConnectStart).toHaveBeenCalledTimes(1)
  })
})

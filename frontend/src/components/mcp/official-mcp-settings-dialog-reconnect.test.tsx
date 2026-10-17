import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest"

import { OfficialMcpSettingsDialog } from "./official-mcp-settings-dialog"
import type { AppIntegration } from "./types"

// A builtin OAuth entry whose /api/mcp/servers row reports
// connection_status: "needs_reconnect" (#2931). The dialog is reached two ways:
// the Tools page hands it a connected entry (isGloballyConnected), and the
// agent builder hands it the catalog entry, which /api/mcp/apps already
// reports as not connected. Both must name the action Reconnect and run the
// same app-scoped login.

vi.mock("@/lib/api-wrapper", () => ({ apiRequest: vi.fn() }))
vi.mock("@/lib/oauth-launch", async () => ({
  openOAuthAuthorizationPopup: (await import("@/test/oauth-popup-double")).openOAuthPopupDouble,
  isOAuthSuccessMessage: (await import("@/test/oauth-popup-double")).isOAuthSuccessMessageDouble,
}))

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return { ...actual, getApiUrl: () => "http://api.local" }
})

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ token: "token", inTeam: false }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
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

const NOTICE = "tools.mcp.dialog.reconnectNotice"
const RECONNECT = "tools.mcp.dialog.reconnect"
const CONFIGURE = "tools.mcp.dialog.configure"
const CONNECT = "tools.mcp.dialog.connect"

function app(overrides: Partial<AppIntegration> = {}): AppIntegration {
  return {
    id: "gmail",
    name: "Gmail",
    description: "",
    icon: "",
    is_connected: true,
    provider: "google",
    transport: "oauth",
    auth_type: "builtin_oauth",
    ...overrides,
  }
}

function renderDialog(props: Partial<React.ComponentProps<typeof OfficialMcpSettingsDialog>>) {
  return render(<OfficialMcpSettingsDialog open onOpenChange={vi.fn()} app={app()} {...props} />)
}

function loginUrl(openSpy: MockInstance<typeof window.open>): URL {
  expect(openSpy).toHaveBeenCalledTimes(1)
  return new URL(openSpy.mock.calls[0][0] as string)
}

describe("OfficialMcpSettingsDialog needs_reconnect", () => {
  let openSpy: MockInstance<typeof window.open>

  beforeEach(() => {
    openSpy = vi.spyOn(window, "open").mockReturnValue({ closed: false, close: vi.fn() } as unknown as Window)
  })

  afterEach(() => {
    cleanup()
    openSpy.mockRestore()
  })

  it("replaces Configure with Reconnect for a connected entry", () => {
    renderDialog({ app: app({ connection_status: "needs_reconnect" }), isGloballyConnected: true })

    expect(screen.getByText(NOTICE)).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: CONFIGURE })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: RECONNECT }))

    const url = loginUrl(openSpy)
    expect(url.pathname).toBe("/api/auth/google/login")
    expect(url.searchParams.get("app_id")).toBe("gmail")
  })

  it("names the Connect action Reconnect for the not-connected catalog entry", () => {
    renderDialog({
      app: app({ is_connected: false, connection_status: "needs_reconnect" }),
      isGloballyConnected: false,
    })

    expect(screen.getByText(NOTICE)).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: CONNECT })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: RECONNECT }))

    expect(loginUrl(openSpy).searchParams.get("app_id")).toBe("gmail")
  })

  it("runs the account step before the reconnect login on the agent builder's path", () => {
    renderDialog({
      app: app({
        id: "zendesk",
        name: "Zendesk",
        provider: "zendesk",
        is_connected: false,
        connection_status: "needs_reconnect",
        oauth_account_input: {
          label: "Zendesk subdomain",
          placeholder: "acme",
          help: "The part before .zendesk.com.",
          suffix: ".zendesk.com",
        },
      }),
      isGloballyConnected: false,
    })

    fireEvent.click(screen.getByRole("button", { name: RECONNECT }))
    expect(openSpy).not.toHaveBeenCalled()
    fireEvent.change(screen.getByLabelText("Zendesk subdomain"), { target: { value: "acme" } })
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.accountStep.continue" }))

    const url = loginUrl(openSpy)
    expect(url.pathname).toBe("/api/auth/zendesk/login")
    expect(url.searchParams.get("app_id")).toBe("zendesk")
    expect(url.searchParams.get("account")).toBe("acme")
  })

  it("hides the Google Drive picker, which needs the grant's token", () => {
    renderDialog({
      app: app({ id: "google-drive", name: "Google Drive", connection_status: "needs_reconnect" }),
      isGloballyConnected: true,
    })

    expect(screen.getByRole("button", { name: RECONNECT })).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "kb.dialog.cloudConnect.picker.open" })).not.toBeInTheDocument()
  })

  it.each([
    ["connected", "connected" as const],
    ["absent", undefined],
    ["null", null],
  ])("leaves a %s status unchanged", (_label, status) => {
    renderDialog({ app: app({ connection_status: status }), isGloballyConnected: true })

    expect(screen.queryByText(NOTICE)).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: RECONNECT })).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: CONFIGURE })).toBeInTheDocument()
  })

  it("ignores the status on an entry that is not builtin OAuth", () => {
    renderDialog({
      app: app({ id: "maps", auth_type: "api_key", connection_status: "needs_reconnect" }),
      isGloballyConnected: true,
      onManageKey: vi.fn(),
    })

    expect(screen.queryByText(NOTICE)).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "tools.mcp.dialog.manageKey" })).toBeInTheDocument()
  })
})

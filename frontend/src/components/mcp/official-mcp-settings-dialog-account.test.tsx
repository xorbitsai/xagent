import React from "react"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest"

import { OfficialMcpSettingsDialog } from "./official-mcp-settings-dialog"
import type { AppIntegration } from "./types"

const translateMock = vi.hoisted(() => (key: string) => key)

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

const zendeskInput = {
  label: "Zendesk subdomain",
  placeholder: "acme",
  help: "The part before .zendesk.com.",
  suffix: ".zendesk.com",
}

function zendeskApp(overrides: Partial<AppIntegration> = {}): AppIntegration {
  return {
    id: "zendesk",
    name: "Zendesk",
    description: "",
    icon: "",
    is_connected: false,
    transport: "oauth",
    auth_type: "builtin_oauth",
    provider: "zendesk",
    oauth_account_input: zendeskInput,
    ...overrides,
  }
}

function submitAccount(value: string) {
  fireEvent.change(screen.getByLabelText("Zendesk subdomain"), { target: { value } })
  fireEvent.click(screen.getByRole("button", { name: "tools.mcp.accountStep.continue" }))
}

function loginUrl(call: unknown[]) {
  return new URL(call[0] as string)
}

describe("OfficialMcpSettingsDialog account step", () => {
  let popup: { closed: boolean; close: ReturnType<typeof vi.fn> }
  let openSpy: MockInstance<typeof window.open>

  beforeEach(() => {
    popup = { closed: false, close: vi.fn() }
    openSpy = vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
  })

  afterEach(() => {
    cleanup()
    openSpy.mockRestore()
    vi.useRealTimers()
  })

  it("asks for the account before its own Connect popup", () => {
    render(<OfficialMcpSettingsDialog open onOpenChange={vi.fn()} app={zendeskApp()} />)

    fireEvent.click(screen.getByRole("button", { name: /tools\.mcp\.dialog\.connect/ }))
    expect(openSpy).not.toHaveBeenCalled()

    submitAccount("acme")

    expect(openSpy).toHaveBeenCalledTimes(1)
    const url = loginUrl(openSpy.mock.calls[0])
    expect(url.pathname).toBe("/api/auth/zendesk/login")
    expect(url.searchParams.get("app_id")).toBe("zendesk")
    expect(url.searchParams.get("account")).toBe("acme")
  })

  it("asks for the account when reconnecting through Configure on the Tools page", () => {
    // The Tools page passes a connected app and no onConnectStart, so
    // Configure re-runs this dialog's own login.
    render(
      <OfficialMcpSettingsDialog
        open
        onOpenChange={vi.fn()}
        app={zendeskApp({ is_connected: true, server_id: 7 })}
        isGloballyConnected
      />,
    )

    fireEvent.click(screen.getByRole("button", { name: /tools\.mcp\.dialog\.configure/ }))
    expect(openSpy).not.toHaveBeenCalled()
    submitAccount("other-account")

    expect(loginUrl(openSpy.mock.calls[0]).searchParams.get("account")).toBe("other-account")
  })

  it("returns to the account step with the last label when the popup closes without connecting", async () => {
    render(<OfficialMcpSettingsDialog open onOpenChange={vi.fn()} app={zendeskApp()} />)
    fireEvent.click(screen.getByRole("button", { name: /tools\.mcp\.dialog\.connect/ }))
    vi.useFakeTimers()
    submitAccount("acme")

    popup.closed = true
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500)
    })

    expect(screen.getByLabelText("Zendesk subdomain")).toHaveValue("acme")
    expect(screen.getByText("tools.mcp.accountStep.retryHint")).toBeInTheDocument()

    // Resubmitting from the reopened step starts a second popup.
    submitAccount("acme2")

    expect(openSpy).toHaveBeenCalledTimes(2)
    expect(new URL(openSpy.mock.calls[1][0] as string).searchParams.get("account")).toBe("acme2")
  })

  it("does not reopen the account step after the settings dialog was closed", async () => {
    const onOpenChange = vi.fn()
    const app = zendeskApp()
    const { rerender } = render(<OfficialMcpSettingsDialog open onOpenChange={onOpenChange} app={app} />)
    fireEvent.click(screen.getByRole("button", { name: /tools\.mcp\.dialog\.connect/ }))
    vi.useFakeTimers()
    submitAccount("acme")

    // The Tools page keeps this component mounted with a non-null app.
    rerender(<OfficialMcpSettingsDialog open={false} onOpenChange={onOpenChange} app={app} />)
    popup.closed = true
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500)
    })

    expect(screen.queryByLabelText("Zendesk subdomain")).not.toBeInTheDocument()
  })

  it("does not reopen the account step after another app's settings were opened", async () => {
    const onOpenChange = vi.fn()
    const { rerender } = render(<OfficialMcpSettingsDialog open onOpenChange={onOpenChange} app={zendeskApp()} />)
    fireEvent.click(screen.getByRole("button", { name: /tools\.mcp\.dialog\.connect/ }))
    vi.useFakeTimers()
    submitAccount("acme")

    rerender(
      <OfficialMcpSettingsDialog
        open
        onOpenChange={onOpenChange}
        app={zendeskApp({ id: "zendesk-sandbox", name: "Zendesk Sandbox" })}
      />,
    )
    popup.closed = true
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500)
    })

    expect(screen.queryByLabelText("Zendesk subdomain")).not.toBeInTheDocument()
  })

  it("does not reopen the account step once the popup reported success", async () => {
    const onSuccess = vi.fn()
    render(<OfficialMcpSettingsDialog open onOpenChange={vi.fn()} app={zendeskApp()} onSuccess={onSuccess} />)
    fireEvent.click(screen.getByRole("button", { name: /tools\.mcp\.dialog\.connect/ }))
    vi.useFakeTimers()
    submitAccount("acme")

    await act(async () => {
      window.dispatchEvent(Object.assign(new Event("message"), { source: popup, origin: "http://api.local", data: { type: "oauth-success", provider: "zendesk" } }))
    })
    // The callback page closes itself right after posting success.
    popup.closed = true
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })

    expect(onSuccess).toHaveBeenCalledTimes(1)
    expect(screen.queryByLabelText("Zendesk subdomain")).not.toBeInTheDocument()
  })

  it("leaves the account step to the parent when onConnectStart is provided", () => {
    const onConnectStart = vi.fn()
    render(
      <OfficialMcpSettingsDialog open onOpenChange={vi.fn()} app={zendeskApp()} onConnectStart={onConnectStart} />,
    )

    fireEvent.click(screen.getByRole("button", { name: /tools\.mcp\.dialog\.connect/ }))

    expect(onConnectStart).toHaveBeenCalledTimes(1)
    expect(screen.queryByLabelText("Zendesk subdomain")).not.toBeInTheDocument()
    expect(openSpy).not.toHaveBeenCalled()
  })

  it("keeps the direct popup for apps that do not declare an account input", () => {
    render(
      <OfficialMcpSettingsDialog
        open
        onOpenChange={vi.fn()}
        app={zendeskApp({ id: "zoom", name: "Zoom", provider: "zoom", oauth_account_input: undefined })}
      />,
    )

    fireEvent.click(screen.getByRole("button", { name: /tools\.mcp\.dialog\.connect/ }))

    expect(openSpy).toHaveBeenCalledTimes(1)
    expect(loginUrl(openSpy.mock.calls[0]).searchParams.has("account")).toBe(false)
  })
})

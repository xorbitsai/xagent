import React from "react"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest"

import { ConnectMcpDialog } from "./connect-mcp-dialog"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const connectStartApp = vi.hoisted(() => ({ current: null as Record<string, unknown> | null }))

vi.mock("@/lib/api-wrapper", () => ({ apiRequest: apiRequestMock }))
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

// Stable across renders: the dialog's load effect depends on `t`, so a fresh
// function per render would refetch forever.
const translateMock = vi.hoisted(() => (key: string) => key)
vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: translateMock }),
}))

vi.mock("@/contexts/mcp-apps-context", () => ({
  useMcpApps: () => ({ apps: [] }),
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: { error: toastErrorMock, success: vi.fn(), warning: vi.fn() },
}))

vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ open, children }: { open: boolean; children: React.ReactNode }) =>
    open ? <div role="dialog">{children}</div> : null,
  DialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogTitle: ({ children }: { children: React.ReactNode }) => <h1>{children}</h1>,
}))

vi.mock("./custom-api-form", () => ({ CustomApiForm: () => null }))
vi.mock("./custom-mcp-form", () => ({ CustomMcpForm: () => null }))

// The detail modal's Connect button reaches the dialog's handleConnectApp
// through onConnectStart, the same route the real modal takes.
vi.mock("./official-mcp-settings-dialog", () => ({
  OfficialMcpSettingsDialog: ({ onConnectStart }: { onConnectStart: (app: object) => void }) => (
    <div>
      <button type="button" onClick={() => onConnectStart(connectStartApp.current as object)}>
        connect-start
      </button>
    </div>
  ),
}))

const zendeskInput = {
  label: "Zendesk subdomain",
  placeholder: "acme",
  help: "The part before .zendesk.com.",
  suffix: ".zendesk.com",
}

function builtinApp(overrides: Record<string, unknown> = {}) {
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

function renderDialog() {
  const result = render(
    <ConnectMcpDialog open onOpenChange={vi.fn()} selectedMcpServers={[]} onSuccess={vi.fn()} />,
  )
  // Fake timers only after the render, so the popup poll's setInterval is
  // advanceable; enabling them before the render hangs the dialog's mount.
  // Only synchronous queries from here on.
  vi.useFakeTimers()
  return result
}

function submitAccount(value: string) {
  fireEvent.change(screen.getByLabelText("Zendesk subdomain"), { target: { value } })
  fireEvent.click(screen.getByRole("button", { name: "tools.mcp.accountStep.continue" }))
}

describe("ConnectMcpDialog account step", () => {
  let popup: { closed: boolean; close: ReturnType<typeof vi.fn> }
  let openSpy: MockInstance<typeof window.open>

  beforeEach(() => {
    apiRequestMock.mockReset().mockResolvedValue({ ok: true, json: async () => [] })
    toastErrorMock.mockReset()
    popup = { closed: false, close: vi.fn() }
    openSpy = vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    connectStartApp.current = builtinApp()
  })

  afterEach(() => {
    cleanup()
    openSpy.mockRestore()
    vi.useRealTimers()
  })

  it("asks for the account first, then opens the login with it from the submit click", () => {
    renderDialog()

    fireEvent.click(screen.getByRole("button", { name: "connect-start" }))

    expect(openSpy).not.toHaveBeenCalled()

    submitAccount("Acme")

    expect(openSpy).toHaveBeenCalledTimes(1)
    const url = new URL(openSpy.mock.calls[0][0] as string)
    expect(url.pathname).toBe("/api/auth/zendesk/login")
    expect(url.searchParams.get("app_id")).toBe("zendesk")
    expect(url.searchParams.get("account")).toBe("acme")
  })

  it("does not take the app's connect slot when the account step is cancelled", () => {
    renderDialog()

    fireEvent.click(screen.getByRole("button", { name: "connect-start" }))
    fireEvent.click(screen.getByRole("button", { name: "common.cancel" }))

    expect(openSpy).not.toHaveBeenCalled()
    expect(screen.queryByLabelText("Zendesk subdomain")).not.toBeInTheDocument()

    // Had the cancelled step claimed the in-flight slot, this second attempt
    // would be refused by the double-connect guard.
    fireEvent.click(screen.getByRole("button", { name: "connect-start" }))
    submitAccount("acme")
    expect(openSpy).toHaveBeenCalledTimes(1)
  })

  it("returns to the account step with the last label when the popup closes without connecting", async () => {
    renderDialog()
    fireEvent.click(screen.getByRole("button", { name: "connect-start" }))
    submitAccount("acme")
    expect(screen.queryByLabelText("Zendesk subdomain")).not.toBeInTheDocument()

    popup.closed = true
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500)
    })

    expect(screen.getByLabelText("Zendesk subdomain")).toHaveValue("acme")
    expect(screen.getByText("tools.mcp.accountStep.retryHint")).toBeInTheDocument()

    // The failed attempt released its slot, so the retry opens a new popup.
    submitAccount("acme-support")
    expect(openSpy).toHaveBeenCalledTimes(2)
    expect(new URL(openSpy.mock.calls[1][0] as string).searchParams.get("account")).toBe("acme-support")
  })

  it("does not reopen the account step after a successful connect", async () => {
    renderDialog()
    fireEvent.click(screen.getByRole("button", { name: "connect-start" }))
    submitAccount("acme")

    await act(async () => {
      window.dispatchEvent(Object.assign(new Event("message"), { source: popup, origin: "http://api.local", data: { type: "oauth-success", provider: "zendesk" } }))
    })
    popup.closed = true
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })

    expect(screen.queryByLabelText("Zendesk subdomain")).not.toBeInTheDocument()
  })

  it("keeps the direct popup without an account for apps that do not declare one", () => {
    connectStartApp.current = builtinApp({ id: "zoom", name: "Zoom", provider: "zoom", oauth_account_input: undefined })
    renderDialog()

    fireEvent.click(screen.getByRole("button", { name: "connect-start" }))

    expect(screen.queryByLabelText("Zendesk subdomain")).not.toBeInTheDocument()
    expect(openSpy).toHaveBeenCalledTimes(1)
    const url = new URL(openSpy.mock.calls[0][0] as string)
    expect(url.searchParams.has("account")).toBe(false)
    expect(url.searchParams.get("app_id")).toBe("zoom")
  })
})

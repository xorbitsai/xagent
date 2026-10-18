import React from "react"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest"

import type { McpApp } from "@/contexts/mcp-apps-context"

const apiRequestMock = vi.hoisted(() => vi.fn())
const routerMock = vi.hoisted(() => ({ replace: vi.fn(), push: vi.fn() }))
const authValue = vi.hoisted(() => ({ token: "token", inTeam: true, user: { is_admin: true } }))
const mcpAppsValue = vi.hoisted(() => ({ getAppIcon: () => null, apps: [] as McpApp[] }))
const translate = vi.hoisted(
  () => (key: string, vars?: Record<string, string | number>) =>
    vars ? `${key}:${JSON.stringify(vars)}` : key,
)
const i18nValue = vi.hoisted(() => ({
  t: translate,
  tDynamic: (_key: string, fallback: string) => fallback,
  locale: "en",
}))

vi.mock("@/lib/api-wrapper", () => ({ apiRequest: apiRequestMock }))

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return { ...actual, getApiUrl: () => "http://api.local" }
})

vi.mock("next/navigation", () => ({
  useRouter: () => routerMock,
  useSearchParams: () => new URLSearchParams(),
}))

vi.mock("@/contexts/auth-context", () => ({ useAuth: () => authValue }))
vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => i18nValue }))
vi.mock("@/contexts/mcp-apps-context", () => ({ useMcpApps: () => mcpAppsValue }))
vi.mock("@/components/ui/sonner", () => ({ toast: { error: vi.fn(), success: vi.fn() } }))

vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ open, children }: { open: boolean; children: React.ReactNode }) =>
    open ? <div role="dialog">{children}</div> : null,
  DialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogTitle: ({ children }: { children: React.ReactNode }) => <h1>{children}</h1>,
  DialogDescription: ({ children }: { children: React.ReactNode }) => <p>{children}</p>,
  DialogFooter: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogTrigger: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

import ToolsPage from "./page"

const zendeskInput = {
  label: "Zendesk subdomain",
  placeholder: "acme",
  help: "The part before .zendesk.com.",
  suffix: ".zendesk.com",
}

function catalogApp(overrides: Partial<McpApp> = {}): McpApp {
  return {
    id: "zendesk",
    name: "Zendesk",
    description: "",
    icon: "",
    users: "",
    transport: "oauth",
    provider: "zendesk",
    category: "Support",
    auth_type: "builtin_oauth",
    oauth_account_input: zendeskInput,
    ...overrides,
  }
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } })
}

// A connected Zendesk server row: /api/mcp/servers carries no
// oauth_account_input, so the page has to take it from the catalog.
function installApiMock() {
  apiRequestMock.mockImplementation(async (url: string) => {
    if (url.endsWith("/api/tools/available")) return jsonResponse({ tools: [] })
    if (url.endsWith("/api/mcp/servers")) {
      return jsonResponse([
        {
          id: 9,
          user_id: 1,
          name: "Zendesk",
          app_id: "zendesk",
          provider: "zendesk",
          transport: "oauth",
          description: "",
          config: {},
          is_active: true,
          is_default: false,
          transport_display: "OAuth",
          created_at: "2024-01-01T00:00:00Z",
          updated_at: "2024-01-01T00:00:00Z",
        },
      ])
    }
    if (url.endsWith("/api/connectors/status")) return jsonResponse({})
    if (url.endsWith("/api/tools/configurable")) return jsonResponse({ tools: [] })
    if (url.endsWith("/api/tools/sql-connections")) return jsonResponse({ connections: [] })
    throw new Error(`Unexpected request: ${url}`)
  })
}

async function openSettingsAndReconnect() {
  render(<ToolsPage />)
  await waitFor(() => expect(screen.getByText("Zendesk")).toBeInTheDocument())
  fireEvent.click(screen.getByText("Zendesk"))
  const dialog = await screen.findByRole("dialog")
  fireEvent.click(within(dialog).getByRole("button", { name: /tools\.mcp\.dialog\.configure/ }))
}

describe("ToolsPage reconnect for an account-hosted provider", () => {
  let openSpy: MockInstance<typeof window.open>

  beforeEach(() => {
    vi.stubGlobal("React", React)
    apiRequestMock.mockReset()
    installApiMock()
    openSpy = vi.spyOn(window, "open").mockReturnValue({ closed: false, close: vi.fn() } as unknown as Window)
  })

  afterEach(() => {
    cleanup()
    openSpy.mockRestore()
    vi.unstubAllGlobals()
    mcpAppsValue.apps = []
  })

  it("asks for the account using the catalog's input field before the popup", async () => {
    mcpAppsValue.apps = [catalogApp()]
    await openSettingsAndReconnect()

    expect(openSpy).not.toHaveBeenCalled()
    fireEvent.change(screen.getByLabelText("Zendesk subdomain"), { target: { value: "acme" } })
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.accountStep.continue" }))

    expect(openSpy).toHaveBeenCalledTimes(1)
    expect(openSpy.mock.calls[0][0]).toBe("about:blank")
    expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/auth/zendesk/login", expect.objectContaining({
      method: "POST", body: JSON.stringify({ app_id: "zendesk", account: "acme", redirect: window.location.origin }),
    }))
  })

  it("reconnects directly when the catalog app declares no account input", async () => {
    mcpAppsValue.apps = [catalogApp({ oauth_account_input: undefined })]
    await openSettingsAndReconnect()

    expect(screen.queryByLabelText("Zendesk subdomain")).not.toBeInTheDocument()
    expect(openSpy).toHaveBeenCalledTimes(1)
    expect(openSpy.mock.calls[0][0]).toBe("about:blank")
    const call = apiRequestMock.mock.calls.find(([url]) => url.endsWith("/api/auth/zendesk/login"))!
    expect(JSON.parse(call[1].body).account).toBeUndefined()
  })
})

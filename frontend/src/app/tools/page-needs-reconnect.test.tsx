import React from "react"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest"

import type { McpApp } from "@/contexts/mcp-apps-context"

// GET /api/mcp/servers keeps a builtin OAuth row listed after its grant stops
// working and reports that as connection_status: "needs_reconnect" (#2931).
// The card and its settings dialog have to say so and offer the app-scoped
// login, while connected and absent statuses render as before.

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

const BADGE = "tools.mcp.needsReconnect"
const NOTICE = /tools\.mcp\.dialog\.reconnectNotice/
const RECONNECT = /tools\.mcp\.dialog\.reconnect$/
const CONFIGURE = /tools\.mcp\.dialog\.configure/

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

type ServerExtras = { connection_status?: "connected" | "needs_reconnect" | null; connected_account?: string }

function installApiMock(extras: ServerExtras, connectorStatus: Record<string, unknown> = {}) {
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
          ...extras,
        },
      ])
    }
    if (url.endsWith("/api/connectors/status")) return jsonResponse(connectorStatus)
    if (url.endsWith("/api/tools/configurable")) return jsonResponse({ tools: [] })
    if (url.endsWith("/api/tools/sql-connections")) return jsonResponse({ connections: [] })
    throw new Error(`Unexpected request: ${url}`)
  })
}

async function renderPage() {
  render(<ToolsPage />)
  await waitFor(() => expect(screen.getByText("Zendesk")).toBeInTheDocument())
}

async function openSettings() {
  await renderPage()
  fireEvent.click(screen.getByText("Zendesk"))
  return screen.findByRole("dialog")
}

describe("ToolsPage needs_reconnect state", () => {
  let openSpy: MockInstance<typeof window.open>

  beforeEach(() => {
    vi.stubGlobal("React", React)
    apiRequestMock.mockReset()
    mcpAppsValue.apps = [catalogApp()]
    openSpy = vi.spyOn(window, "open").mockReturnValue({ closed: false, close: vi.fn() } as unknown as Window)
  })

  afterEach(() => {
    cleanup()
    openSpy.mockRestore()
    vi.unstubAllGlobals()
    mcpAppsValue.apps = []
  })

  it("marks the card and offers Reconnect, which runs the app-scoped login through the account step", async () => {
    installApiMock({ connection_status: "needs_reconnect" })
    await renderPage()
    expect(screen.getByText(BADGE)).toBeInTheDocument()

    fireEvent.click(screen.getByText("Zendesk"))
    const dialog = await screen.findByRole("dialog")
    expect(within(dialog).getByText(NOTICE)).toBeInTheDocument()
    expect(within(dialog).queryByRole("button", { name: CONFIGURE })).not.toBeInTheDocument()
    fireEvent.click(within(dialog).getByRole("button", { name: RECONNECT }))

    expect(openSpy).not.toHaveBeenCalled()
    fireEvent.change(screen.getByLabelText("Zendesk subdomain"), { target: { value: "acme" } })
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.accountStep.continue" }))

    expect(openSpy).toHaveBeenCalledTimes(1)
    const url = new URL(openSpy.mock.calls[0][0] as string)
    expect(url.pathname).toBe("/api/auth/zendesk/login")
    expect(url.searchParams.get("app_id")).toBe("zendesk")
    expect(url.searchParams.get("account")).toBe("acme")
  })

  it("keeps Disconnect available while the connection needs a reconnect", async () => {
    installApiMock({ connection_status: "needs_reconnect" })
    const dialog = await openSettings()
    expect(within(dialog).getByRole("button", { name: /tools\.mcp\.dialog\.disconnect/ })).toBeInTheDocument()
  })

  it("renders a connected server unchanged", async () => {
    installApiMock({ connection_status: "connected", connected_account: "agent@acme.example" })
    await renderPage()
    expect(screen.queryByText(BADGE)).not.toBeInTheDocument()

    fireEvent.click(screen.getByText("Zendesk"))
    const dialog = await screen.findByRole("dialog")
    expect(within(dialog).getByText("agent@acme.example")).toBeInTheDocument()
    expect(within(dialog).queryByText(NOTICE)).not.toBeInTheDocument()
    expect(within(dialog).getByRole("button", { name: CONFIGURE })).toBeInTheDocument()
    expect(within(dialog).queryByRole("button", { name: RECONNECT })).not.toBeInTheDocument()
  })

  it.each([
    ["without the field", {}],
    ["with a null status", { connection_status: null }],
  ])("renders a server %s unchanged", async (_label, extras: ServerExtras) => {
    installApiMock(extras)
    const dialog = await openSettings()
    expect(screen.queryByText(BADGE)).not.toBeInTheDocument()
    expect(within(dialog).queryByText(NOTICE)).not.toBeInTheDocument()
    expect(within(dialog).getByRole("button", { name: CONFIGURE })).toBeInTheDocument()
  })

  it("still marks a team connector the viewer does not own, which has no settings dialog", async () => {
    installApiMock({ connection_status: "needs_reconnect" }, { "mcp:9": { shared: true, is_owner: false } })
    await renderPage()
    await waitFor(() => expect(screen.getByText("tools.mcp.sharing.teamTool")).toBeInTheDocument())
    expect(screen.getByText(BADGE)).toBeInTheDocument()

    fireEvent.click(screen.getByText("Zendesk"))
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })
})

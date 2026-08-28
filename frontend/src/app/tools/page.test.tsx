import React from "react"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const routerReplaceMock = vi.hoisted(() => vi.fn())
const routerPushMock = vi.hoisted(() => vi.fn())
const searchParamsMock = vi.hoisted(() => ({ value: new URLSearchParams() }))
// Stable across renders on purpose: the page has a useEffect keyed on
// [isAdmin, user] (loadConfigurableTools/loadSqlConnections). A mock that
// returns a fresh object literal on every call gives that effect a new
// `user` reference every render, which never stabilizes -- each run sets
// state, which re-renders, which calls this mock again, forever.
const routerMock = vi.hoisted(() => ({ replace: routerReplaceMock, push: routerPushMock }))
const authValue = vi.hoisted(() => ({
  token: "token",
  inTeam: true,
  user: { is_admin: true },
}))
const mcpAppsValue = vi.hoisted(() => ({ getAppIcon: () => null, apps: [] }))
// t/tDynamic must be stable function references, not freshly-created
// closures: ConnectMcpDialog (mounted unconditionally by this page, gated
// closed via its own `open` prop) has a useEffect keyed on [open, t,
// selectedMcpServers]. A `t` that is a new function every render never lets
// that effect settle, and its close-branch cleanup (clearMcpOauthPollState)
// unconditionally calls setLoadingApps(new Set()) -- an unstable `t` alone
// is enough to spin that into an infinite render loop.
const translate = vi.hoisted(
  () => (key: string, vars?: Record<string, string | number>) =>
    vars ? `${key}:${JSON.stringify(vars)}` : key,
)
const translateDynamic = vi.hoisted(() => (_key: string, fallback: string) => fallback)
const i18nValue = vi.hoisted(() => ({ t: translate, tDynamic: translateDynamic, locale: "en" }))

// ./page.tsx has no top-level React import (it relies on the automatic JSX
// runtime everywhere else in this app config), while this harness's esbuild
// transform resolves its React.createElement calls against a global binding.
// The binding is read when the component renders, not when the module is
// imported, so stubbing it per test is early enough -- and vi.stubGlobal is
// tracked, so the afterEach below restores the global instead of leaving one
// behind for the other files sharing this worker.

vi.mock("@/lib/api-wrapper", () => ({ apiRequest: apiRequestMock }))

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return { ...actual, getApiUrl: () => "http://api.local" }
})

vi.mock("next/navigation", () => ({
  useRouter: () => routerMock,
  useSearchParams: () => searchParamsMock.value,
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => authValue,
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => i18nValue,
}))

vi.mock("@/contexts/mcp-apps-context", () => ({
  useMcpApps: () => mcpAppsValue,
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: { error: vi.fn(), success: vi.fn() },
}))

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
import type { MCPServer } from "./page"

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  })
}

// The one server shape handleEditMcpServer's official branch requires:
// transport === 'oauth'.
function officialMcpServer(overrides: Partial<MCPServer> = {}): MCPServer {
  return {
    id: 9,
    user_id: 1,
    name: "Records MCP",
    transport: "oauth",
    description: "",
    config: {},
    is_active: true,
    is_default: false,
    transport_display: "OAuth",
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
    ...overrides,
  }
}

/** Dispatches the page's five mount fetches by URL. */
function installApiMock(connectorStatus: unknown = {}) {
  apiRequestMock.mockImplementation(async (url: string) => {
    if (url.endsWith("/api/tools/available")) return jsonResponse({ tools: [] })
    if (url.endsWith("/api/mcp/servers")) return jsonResponse([officialMcpServer()])
    if (url.endsWith("/api/connectors/status")) return jsonResponse(connectorStatus)
    if (url.endsWith("/api/tools/configurable")) return jsonResponse({ tools: [] })
    if (url.endsWith("/api/tools/sql-connections")) return jsonResponse({ connections: [] })
    throw new Error(`Unexpected request: ${url}`)
  })
}

async function renderPage() {
  render(<ToolsPage />)
  await waitFor(() => {
    expect(screen.getByText("Records MCP")).toBeInTheDocument()
  })
}

beforeEach(() => {
  vi.stubGlobal("React", React)
  apiRequestMock.mockReset()
  routerReplaceMock.mockReset()
  routerPushMock.mockReset()
  searchParamsMock.value = new URLSearchParams()
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe("ToolsPage official settings dialog ownership badge (#1623)", () => {
  it("shows the matching three-way label when the server carries a real sharing status", async () => {
    // is_owner: true, not false: the card's own (unrelated, untouched)
    // isNonOwnedTeamTool check reads a team-shared entry the viewer does not
    // own as unclickable, which would silently skip the dialog entirely
    // while still leaving the *card's* own badge in the document to satisfy
    // an unscoped query -- scoping to the dialog below is what would have
    // caught that.
    installApiMock({ "mcp:9": { shared: true, is_owner: true, needs_config: false } })
    await renderPage()

    fireEvent.click(screen.getByText("Records MCP"))

    const dialog = await screen.findByRole("dialog")
    expect(within(dialog).getByText("tools.mcp.sharing.shared")).toBeInTheDocument()
    // Spreading the sanitized status onto this entry has a second consumer
    // besides the badge: the share-toggle button reads app.shared to decide
    // its own label and the direction it POSTs (handleToggleShare's share
    // argument is !app.shared). Before this change app.shared was always
    // undefined here, so this button read "Share" even for an
    // already-shared connector -- pinning the correct "Unshare" direction.
    expect(
      within(dialog).getByRole("button", { name: "tools.mcp.sharing.unshare" }),
    ).toBeInTheDocument()
  })

  it("shows the private label and the share direction for an unshared server", async () => {
    installApiMock({ "mcp:9": { shared: false, is_owner: true, needs_config: false } })
    await renderPage()

    fireEvent.click(screen.getByText("Records MCP"))

    const dialog = await screen.findByRole("dialog")
    expect(within(dialog).getByText("tools.mcp.sharing.private")).toBeInTheDocument()
    expect(
      within(dialog).getByRole("button", { name: "tools.mcp.sharing.share" }),
    ).toBeInTheDocument()
  })

  it("shows no ownership label when the sharing status is malformed", async () => {
    // shared is not a boolean. is_owner: true and needs_config: false keep
    // the card's own isNonOwnedTeamTool check (which reads the raw status
    // and is deliberately left alone by this change) reading this as the
    // viewer's own tool, so the card stays clickable, and keep the badge
    // div's needs_config suffix from ever appending -- both deliberate, so
    // this isolates the assertion to the dialog's sanitized lookup, which
    // must reject the whole entry and withhold its badge.
    installApiMock({ "mcp:9": { shared: "yes", is_owner: true, needs_config: false } })
    await renderPage()

    fireEvent.click(screen.getByText("Records MCP"))

    const dialog = await screen.findByRole("dialog")
    // Scoped to the dialog: the card behind it renders its own (unsanitized,
    // untouched) badge off the same malformed status and would otherwise
    // make this assertion about a different element entirely.
    const dialogQueries = within(dialog)
    expect(dialogQueries.queryByText("tools.mcp.sharing.private")).toBeNull()
    expect(dialogQueries.queryByText("tools.mcp.sharing.shared")).toBeNull()
    expect(dialogQueries.queryByText("tools.mcp.sharing.teamTool")).toBeNull()
    // The badge labels above pass whether or not the sanitizer runs -- both
    // a rejected entry and a never-spread one leave app.shared undefined,
    // so a malformed entry alone can't tell the two apart. The share
    // button's direction can: it reads app.shared for truthiness, so an
    // unsanitized "yes" would read truthy and show "Unshare" here instead.
    expect(
      within(dialog).getByRole("button", { name: "tools.mcp.sharing.share" }),
    ).toBeInTheDocument()
  })

  it("shows no ownership label when needs_config is not a boolean", async () => {
    // needs_config: "yes" is the only malformed field here; shared and
    // is_owner are well-formed. sanitizeConnectorStatusEntry rejects the
    // whole entry when any one field fails its boolean check, so this pins
    // that field-level rejection is entry-wide, not per-field.
    installApiMock({ "mcp:9": { shared: true, is_owner: true, needs_config: "yes" } })
    await renderPage()

    fireEvent.click(screen.getByText("Records MCP"))

    const dialog = await screen.findByRole("dialog")
    const dialogQueries = within(dialog)
    expect(dialogQueries.queryByText("tools.mcp.sharing.private")).toBeNull()
    expect(dialogQueries.queryByText("tools.mcp.sharing.shared")).toBeNull()
    expect(dialogQueries.queryByText("tools.mcp.sharing.teamTool")).toBeNull()
    // Positive anchor: the dialog did render past the malformed status,
    // it just withheld every sharing badge.
    expect(dialogQueries.getByRole("button", { name: "tools.mcp.sharing.share" })).toBeInTheDocument()
  })
})

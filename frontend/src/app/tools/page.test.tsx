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

import * as fs from "node:fs"
import * as path from "node:path"

import { SIGNED_UPLOAD_TRANSPORTS } from "@/lib/mcp-utils"
import {
  signedUploadRowsFromValue,
  signedUploadsFromRows,
  type SignedUploadMode,
} from "@/components/mcp/signed-uploads-form"

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

it("shows the migration reason for a retired SQL connection without hiding its record", async () => {
  installApiMock()
  const fallback = apiRequestMock.getMockImplementation()!
  const reason = "File-backed SQLite is retired; migrate to PostgreSQL or MySQL."
  apiRequestMock.mockImplementation(async (url: string, init?: RequestInit) => {
    if (url.endsWith("/api/tools/available")) {
      return jsonResponse({ tools: [{
        name: "execute_sql_query", description: "SQL", type: "builtin", category: "database",
        enabled: true, requires_configuration: true,
      }] })
    }
    if (url.endsWith("/api/tools/sql-connections")) {
      return jsonResponse({ connections: [{
        name: "legacy", source: "db", masked: "sqlite:///old.db", configured: false, error: reason,
      }] })
    }
    return fallback(url, init)
  })
  await renderPage()
  fireEvent.click(screen.getByRole("button", { name: "tools.database.manageConnections" }))
  const dialog = within(await screen.findByRole("dialog"))
  expect(dialog.getByRole("alert")).toHaveTextContent(reason)
  expect(dialog.getByText("legacy")).toBeInTheDocument()
  expect(dialog.getByText("sqlite:///old.db")).toBeInTheDocument()
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

describe("ToolsPage connector delete from the edit dialog", () => {
  function customMcpServer(overrides: Partial<MCPServer> = {}): MCPServer {
    return officialMcpServer({
      id: 5,
      name: "Records MCP",
      transport: "stdio",
      transport_display: "stdio",
      can_edit_global: true,
      ...overrides,
    })
  }

  function mcpDetail(server: MCPServer) {
    return {
      id: server.id,
      user_id: server.user_id,
      name: server.name,
      transport: server.transport,
      description: server.description,
      config: { command: "run", args: [] },
      user_env: null,
      runtime_input_schema: null,
      runtime_bindings: null,
      allow_delegated_authorization: false,
      can_edit_global: server.can_edit_global ?? false,
      signed_uploads: null,
      signed_uploads_error: null,
    }
  }

  function customApiDetail(server: MCPServer) {
    return {
      id: server.id,
      name: server.name,
      description: server.description,
      url: "https://api.example.com",
      method: "GET",
      headers: null,
      body: null,
      env: null,
      runtime_input_schema: null,
      runtime_bindings: null,
      allow_delegated_authorization: false,
    }
  }

  function installDeleteApiMock(
    server: MCPServer,
    deleteResponse: () => Response | Promise<Response> = () => new Response(null, { status: 204 }),
    statusResponse: () => Response | Promise<Response> = () => jsonResponse({}),
  ) {
    let deleted = false
    apiRequestMock.mockImplementation(async (url: string, init?: RequestInit) => {
      if (init?.method === "DELETE") {
        const response = await deleteResponse()
        if (response.ok) deleted = true
        return response
      }
      if (url.endsWith("/api/tools/available")) return jsonResponse({ tools: [] })
      if (url.endsWith("/api/mcp/servers")) return jsonResponse(deleted ? [] : [server])
      if (url.endsWith("/api/connectors/status")) return statusResponse()
      if (url.endsWith("/api/tools/configurable")) return jsonResponse({ tools: [] })
      if (url.endsWith("/api/tools/sql-connections")) return jsonResponse({ connections: [] })
      if (url.endsWith(`/api/mcp/servers/${server.id}`)) return jsonResponse(mcpDetail(server))
      if (url.endsWith(`/api/custom-apis/${server.id}`)) return jsonResponse(customApiDetail(server))
      throw new Error(`Unexpected request: ${url}`)
    })
  }

  async function openEditDialog() {
    await renderPage()
    fireEvent.click(screen.getByText("Records MCP"))
    return within(await screen.findByRole("dialog"))
  }

  function deleteCalls() {
    return apiRequestMock.mock.calls.filter(([, init]) => init?.method === "DELETE")
  }

  it("deletes an owned MCP server after confirmation and refreshes the list", async () => {
    const { toast } = await import("@/components/ui/sonner")
    vi.mocked(toast.success).mockClear()
    vi.stubGlobal("confirm", vi.fn(() => true))
    installDeleteApiMock(customMcpServer())

    const dialog = await openEditDialog()
    fireEvent.click(dialog.getByRole("button", { name: /tools\.mcp\.buttons\.delete/ }))

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(deleteCalls().map(([url]) => url)).toEqual(["http://api.local/api/mcp/servers/5"])
    await waitFor(() => expect(screen.queryByText("Records MCP")).not.toBeInTheDocument())
    expect(toast.success).toHaveBeenCalledWith(
      'tools.mcp.dialog.deleteSuccess:{"name":"Records MCP"}',
    )
  })

  it("deletes a Custom API through the Custom API route", async () => {
    const { toast } = await import("@/components/ui/sonner")
    vi.mocked(toast.success).mockClear()
    vi.stubGlobal("confirm", vi.fn(() => true))
    installDeleteApiMock(customMcpServer({ id: 7, transport: "custom_api" }))

    const dialog = await openEditDialog()
    fireEvent.click(dialog.getByRole("button", { name: /tools\.mcp\.buttons\.delete/ }))

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(deleteCalls().map(([url]) => url)).toEqual(["http://api.local/api/custom-apis/7"])
    await waitFor(() => expect(screen.queryByText("Records MCP")).not.toBeInTheDocument())
    expect(toast.success).toHaveBeenCalledWith(
      'tools.mcp.dialog.deleteSuccess:{"name":"Records MCP"}',
    )
  })

  it("keeps the dialog and its buttons locked while a delete is in flight", async () => {
    vi.stubGlobal("confirm", vi.fn(() => true))
    let finishDelete: (response: Response) => void = () => {}
    installDeleteApiMock(
      customMcpServer(),
      () => new Promise<Response>((resolve) => { finishDelete = resolve }),
    )

    const dialog = await openEditDialog()
    const deleteButton = dialog.getByRole("button", { name: /tools\.mcp\.buttons\.delete/ })
    fireEvent.click(deleteButton)

    await waitFor(() => expect(deleteButton).toBeDisabled())
    expect(dialog.getByRole("button", { name: /tools\.mcp\.buttons\.cancel/ })).toBeDisabled()
    expect(dialog.getByRole("button", { name: /tools\.mcp\.buttons\.save/ })).toBeDisabled()
    fireEvent.click(deleteButton)
    expect(deleteCalls()).toHaveLength(1)

    finishDelete(new Response(null, { status: 204 }))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(deleteCalls()).toHaveLength(1)
  })

  it.each([
    ["a network error", () => Promise.reject(new Error("offline"))],
    ["a non-string detail", () => jsonResponse({ detail: [{ msg: "bad id" }] }, 422)],
    ["a body without detail", () => new Response("upstream failure", { status: 502 })],
  ])("falls back to the generic failure message on %s", async (_label, respond) => {
    const { toast } = await import("@/components/ui/sonner")
    vi.mocked(toast.error).mockClear()
    vi.stubGlobal("confirm", vi.fn(() => true))
    installDeleteApiMock(customMcpServer(), respond)

    const dialog = await openEditDialog()
    fireEvent.click(dialog.getByRole("button", { name: /tools\.mcp\.buttons\.delete/ }))

    await waitFor(() =>
      expect(toast.error).toHaveBeenCalledWith('tools.mcp.dialog.deleteFailed:{"name":"Records MCP"}'),
    )
    expect(screen.getByRole("dialog")).toBeInTheDocument()
  })

  it.each([
    ["has not answered yet", () => new Promise<Response>(() => {})],
    ["failed", () => jsonResponse({ detail: "unavailable" }, 500)],
  ])("hides delete in a team while the ownership status %s", async (_label, respondStatus) => {
    installDeleteApiMock(customMcpServer(), undefined, respondStatus)

    const dialog = await openEditDialog()

    expect(dialog.getByRole("button", { name: /tools\.mcp\.buttons\.cancel/ })).toBeInTheDocument()
    expect(dialog.queryByRole("button", { name: /tools\.mcp\.buttons\.delete/ })).not.toBeInTheDocument()
  })

  it("sends nothing when the confirmation is declined", async () => {
    vi.stubGlobal("confirm", vi.fn(() => false))
    installDeleteApiMock(customMcpServer())

    const dialog = await openEditDialog()
    fireEvent.click(dialog.getByRole("button", { name: /tools\.mcp\.buttons\.delete/ }))

    expect(deleteCalls()).toHaveLength(0)
    expect(screen.getByRole("dialog")).toBeInTheDocument()
  })

  it("shows the backend refusal and keeps the dialog open", async () => {
    const { toast } = await import("@/components/ui/sonner")
    vi.mocked(toast.error).mockClear()
    vi.stubGlobal("confirm", vi.fn(() => true))
    installDeleteApiMock(customMcpServer(), () =>
      jsonResponse({ detail: "MCP server has actor-owned OAuth connections" }, 409),
    )

    const dialog = await openEditDialog()
    fireEvent.click(dialog.getByRole("button", { name: /tools\.mcp\.buttons\.delete/ }))

    await waitFor(() =>
      expect(toast.error).toHaveBeenCalledWith("MCP server has actor-owned OAuth connections"),
    )
    expect(screen.getByRole("dialog")).toBeInTheDocument()
    expect(screen.getAllByText("Records MCP").length).toBeGreaterThan(0)
  })

  it("hides the delete button when the viewer cannot edit the shared config", async () => {
    installDeleteApiMock(customMcpServer({ can_edit_global: false }))

    const dialog = await openEditDialog()

    expect(dialog.getByRole("button", { name: /tools\.mcp\.buttons\.cancel/ })).toBeInTheDocument()
    expect(dialog.queryByRole("button", { name: /tools\.mcp\.buttons\.delete/ })).not.toBeInTheDocument()
  })
})

// These checks live under src/app so the app-pages CI lane runs them; they hold the signed-upload checks that must run in CI.
describe("ToolsPage signed upload settings", () => {
  const SIGNED_UPLOADS = [
    {
      tool_name: "attach_document",
      file_argument: "document_file_id",
      issuing_tool_name: "request_upload",
      allowed_upload_hosts: ["uploads.example.test"],
      upload_path_prefix: "/staged/",
      content_types: { ".pdf": "application/pdf" },
    },
  ]
  const SETTINGS_ERROR = "signed_uploads[0] must have exactly the documented keys"
  // Stored entries that cannot be used: the detail echoes them as they are.
  const BAD_SIGNED_UPLOADS = [{ tool_name: "attach_document", file_argument: "document_file_id" }]
  // A stored value that is not a list: the detail reads it as null.
  const NOT_A_LIST = { tool_name: "attach_document" }
  const NOT_A_LIST_ERROR = "signed_uploads must be a list"
  const KEY = "tools.mcp.signedUploads"
  // An issuer entry: exactly the four issuer keys.
  const ISSUER_ENTRY = {
    issuing_tool_name: "request_upload",
    allowed_upload_hosts: ["uploads.example.test"],
    upload_path_prefix: "/staged/",
    content_types: { ".pdf": "application/pdf" },
  }
  // A replacement entry whose issuing tool differs from ISSUER_ENTRY's.
  const REPLACEMENT_ENTRY = { ...SIGNED_UPLOADS[0], issuing_tool_name: "request_image_upload" }
  const BOTH_KINDS = [REPLACEMENT_ENTRY, ISSUER_ENTRY]

  function uploadsServer(overrides: Partial<MCPServer> = {}): MCPServer {
    return officialMcpServer({
      id: 5,
      name: "records_mcp",
      transport: "streamable_http",
      transport_display: "Streamable HTTP",
      can_edit_global: true,
      ...overrides,
    })
  }

  function uploadsDetail(server: MCPServer, overrides: Record<string, unknown> = {}) {
    return {
      id: server.id,
      user_id: server.user_id,
      name: server.name,
      transport: server.transport,
      description: server.description,
      config: { url: "https://mcp.example.com/mcp" },
      user_env: null,
      runtime_input_schema: null,
      runtime_bindings: null,
      allow_delegated_authorization: false,
      can_edit_global: server.can_edit_global ?? false,
      signed_uploads: null,
      signed_uploads_error: null,
      ...overrides,
    }
  }

  function installUploadsApiMock(
    servers: MCPServer[],
    detail: Record<string, unknown> = {},
    putResponse: () => Response = () => jsonResponse({}),
  ) {
    const requests = { put: [] as Record<string, unknown>[], post: [] as Record<string, unknown>[] }
    apiRequestMock.mockImplementation(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET"
      if (url.endsWith("/api/tools/available")) return jsonResponse({ tools: [] })
      if (url.endsWith("/api/connectors/status")) return jsonResponse({})
      if (url.endsWith("/api/tools/configurable")) return jsonResponse({ tools: [] })
      if (url.endsWith("/api/tools/sql-connections")) return jsonResponse({ connections: [] })
      if (url.includes("/api/mcp/apps")) return jsonResponse([])
      if (url.endsWith("/api/mcp/servers")) {
        if (method === "POST") {
          requests.post.push(JSON.parse(String(init?.body)))
          return jsonResponse({ id: 99 })
        }
        return jsonResponse(servers)
      }
      if (url.endsWith("/api/mcp/servers/5")) {
        if (method === "PUT") {
          requests.put.push(JSON.parse(String(init?.body)))
          return putResponse()
        }
        return jsonResponse(uploadsDetail(servers[0], detail))
      }
      throw new Error(`Unexpected request: ${method} ${url}`)
    })
    return requests
  }

  async function openUploadsEditor() {
    render(<ToolsPage />)
    await waitFor(() => expect(screen.getByText("records_mcp")).toBeInTheDocument())
    fireEvent.click(screen.getByText("records_mcp"))
    return within(await screen.findByRole("dialog"))
  }

  function saveButton(dialog: ReturnType<typeof within>) {
    return dialog.getByRole("button", { name: /tools\.mcp\.buttons\.save/ })
  }

  it("keeps the frontend transport set equal to the backend one", () => {
    // A plain regex read of the Python source, not a Python parse.
    const source = fs.readFileSync(
      path.resolve(__dirname, "../../../../src/xagent/core/tools/adapters/vibe/signed_upload.py"),
      "utf-8",
    )
    const match = /^SIGNED_UPLOAD_TRANSPORTS = frozenset\(\{([^}]*)\}\)$/m.exec(source)
    expect(match).not.toBeNull()
    const backend = Array.from((match as RegExpExecArray)[1].matchAll(/"(\w+)"/g), (item) => item[1])

    expect(backend.length).toBeGreaterThan(0)
    expect([...backend].sort()).toEqual([...SIGNED_UPLOAD_TRANSPORTS].sort())
  })

  it("marks a connector whose upload settings are invalid", async () => {
    installUploadsApiMock([
      uploadsServer({ id: 5, name: "records_mcp", signed_uploads_error: SETTINGS_ERROR }),
      uploadsServer({ id: 6, name: "healthy_mcp", signed_uploads_error: null }),
    ])
    render(<ToolsPage />)
    await waitFor(() => expect(screen.getByText("records_mcp")).toBeInTheDocument())

    const invalidCard = screen.getByText("records_mcp").parentElement as HTMLElement
    const healthyCard = screen.getByText("healthy_mcp").parentElement as HTMLElement
    const badge = within(invalidCard).getByText(`${KEY}.invalidBadge`)
    expect(badge).toHaveAttribute("title", `${KEY}.invalidBadgeDescription`)
    expect(within(healthyCard).queryByText(`${KEY}.invalidBadge`)).not.toBeInTheDocument()
  })

  it("saves edited upload settings", async () => {
    const requests = installUploadsApiMock([uploadsServer()], { signed_uploads: SIGNED_UPLOADS })
    const dialog = await openUploadsEditor()

    fireEvent.change(dialog.getByLabelText(`${KEY}.uploadPathPrefix`), {
      target: { value: "/incoming/" },
    })
    fireEvent.click(saveButton(dialog))

    await waitFor(() => expect(requests.put).toHaveLength(1))
    expect(requests.put[0]).toEqual({
      signed_uploads: [{ ...SIGNED_UPLOADS[0], upload_path_prefix: "/incoming/" }],
    })
  })

  it.each([
    ["entries that cannot be used", BAD_SIGNED_UPLOADS, SETTINGS_ERROR],
    ["a value that is not a list", NOT_A_LIST, NOT_A_LIST_ERROR],
  ])("clears invalid upload settings with Clear", async (_name, stored, error) => {
    const requests = installUploadsApiMock([uploadsServer()], {
      signed_uploads: stored,
      signed_uploads_error: error,
    })
    const dialog = await openUploadsEditor()

    if (stored === BAD_SIGNED_UPLOADS) {
      expect(dialog.getByLabelText(`${KEY}.toolName`)).toHaveValue("attach_document")
    }
    fireEvent.click(dialog.getByRole("button", { name: "common.clear" }))
    fireEvent.click(saveButton(dialog))

    await waitFor(() => expect(requests.put).toHaveLength(1))
    expect(requests.put[0]).toEqual({ signed_uploads: [] })
  })

  it.each([
    ["entries that cannot be used", BAD_SIGNED_UPLOADS, SETTINGS_ERROR],
    ["a value that is not a list", NOT_A_LIST, NOT_A_LIST_ERROR],
  ])("leaves invalid upload settings alone when the owner edits something else", async (_name, stored, error) => {
    const requests = installUploadsApiMock([uploadsServer()], {
      signed_uploads: stored,
      signed_uploads_error: error,
    })
    const dialog = await openUploadsEditor()

    fireEvent.change(dialog.getByLabelText("tools.mcp.form.descriptionLabel"), {
      target: { value: "Updated" },
    })
    fireEvent.click(saveButton(dialog))

    await waitFor(() => expect(requests.put).toHaveLength(1))
    expect(requests.put[0]).toEqual({ description: "Updated" })
  })

  it("clears upload settings when the owner switches to stdio", async () => {
    const requests = installUploadsApiMock([uploadsServer()], { signed_uploads: SIGNED_UPLOADS })
    const dialog = await openUploadsEditor()

    fireEvent.click(dialog.getByRole("button", { name: "STDIO" }))
    expect(dialog.getByText(`${KEY}.willBeCleared`)).toBeInTheDocument()
    fireEvent.click(saveButton(dialog))

    await waitFor(() => expect(requests.put).toHaveLength(1))
    expect(requests.put[0]).toEqual({ transport: "stdio", signed_uploads: [] })
  })

  it.each([
    ["replacement entries", SIGNED_UPLOADS, `${KEY}.modeIssuer`],
    ["entries of both kinds", BOTH_KINDS, `${KEY}.modeReplacement`],
  ])("sends no upload settings for a viewer who cannot edit the shared config: %s", async (_name, stored, otherMode) => {
    const requests = installUploadsApiMock(
      [uploadsServer({ can_edit_global: false })],
      { signed_uploads: stored },
    )
    const dialog = await openUploadsEditor()

    const block = dialog.getByText(`${KEY}.title`).closest("fieldset")
    if (!block) throw new Error("File uploads block not found")
    const fields = block.querySelectorAll("input, textarea")
    expect(fields.length).toBeGreaterThan(0)
    fields.forEach((field) => expect(field).toBeDisabled())
    expect(dialog.getByText("attach_document", { selector: "span" })).toBeInTheDocument()
    const radios = within(block).getAllByRole("radio")
    expect(radios).toHaveLength(stored.length * 2)
    radios.forEach((radio) => expect(radio).toBeDisabled())
    const lastRowOther = dialog.getAllByRole("radio", { name: otherMode })[stored.length - 1]
    fireEvent.click(lastRowOther)
    expect(lastRowOther).not.toBeChecked()
    fireEvent.click(saveButton(dialog))

    await waitFor(() => expect(requests.put).toHaveLength(1))
    expect(requests.put[0]).toEqual({})
  })

  it("sends no upload settings when a new connector leaves sse or streamable_http", async () => {
    const requests = installUploadsApiMock([uploadsServer()])
    render(<ToolsPage />)
    await waitFor(() => expect(screen.getByText("records_mcp")).toBeInTheDocument())

    fireEvent.click(screen.getByRole("button", { name: /tools\.mcp\.addConnector/ }))
    const customTab = (await screen.findAllByRole("tab")).find((tab) =>
      (tab.textContent ?? "").includes("tools.mcp.dialog.customMcp"),
    )
    if (!customTab) throw new Error("Custom MCP tab not found")
    fireEvent.mouseDown(customTab)
    fireEvent.click(customTab)

    fireEvent.change(await screen.findByLabelText("tools.mcp.form.nameLabel"), {
      target: { value: "records_mcp" },
    })
    fireEvent.click(screen.getByRole("button", { name: "HTTP" }))
    fireEvent.click(screen.getByRole("button", { name: `${KEY}.addTool` }))
    fireEvent.change(screen.getByLabelText(`${KEY}.toolName`), {
      target: { value: "attach_document" },
    })
    fireEvent.click(screen.getByRole("button", { name: "WebSocket" }))
    expect(screen.queryByText(`${KEY}.title`)).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "STDIO" }))
    fireEvent.change(screen.getByLabelText("tools.mcp.dialog.command"), {
      target: { value: "run" },
    })
    const saveButtons = screen.getAllByRole("button", { name: /tools\.mcp\.buttons\.save/ })
    fireEvent.click(saveButtons[saveButtons.length - 1])

    await waitFor(() => expect(requests.post).toHaveLength(1))
    expect(requests.post[0].transport).toBe("stdio")
    expect("signed_uploads" in requests.post[0]).toBe(false)
  })

  it.each([
    ["a rejected setting", "Invalid configuration: signed_uploads[0].content_types keys must contain only one '.'"],
    [
      "a refused transport change",
      "Invalid configuration: signed_uploads are supported only for the sse and streamable_http transports; to use another transport, set signed_uploads to [] in the same request",
    ],
  ])("shows the backend refusal verbatim", async (_name, detailText) => {
    const { toast } = await import("@/components/ui/sonner")
    vi.mocked(toast.error).mockClear()
    const requests = installUploadsApiMock(
      [uploadsServer()],
      { signed_uploads: SIGNED_UPLOADS },
      () => jsonResponse({ detail: detailText }, 400),
    )
    const dialog = await openUploadsEditor()

    fireEvent.change(dialog.getByLabelText("tools.mcp.form.descriptionLabel"), {
      target: { value: "Updated" },
    })
    fireEvent.click(saveButton(dialog))

    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(detailText))
    expect(requests.put).toHaveLength(1)
    expect(screen.getByRole("dialog")).toBeInTheDocument()
  })

  function rowIds() {
    let next = 0
    return () => `row-${next++}`
  }

  it.each([
    ["replacement entries only", SIGNED_UPLOADS, ["replacement"]],
    ["issuer entries only", [ISSUER_ENTRY], ["issuer"]],
    ["a replacement entry before an issuer entry", BOTH_KINDS, ["replacement", "issuer"]],
    ["an issuer entry before a replacement entry", [ISSUER_ENTRY, REPLACEMENT_ENTRY], ["issuer", "replacement"]],
  ])("round-trips stored entries of both kinds: %s", (_name, stored, modes) => {
    const rows = signedUploadRowsFromValue(stored, rowIds())
    expect(rows.map((row) => row.mode)).toEqual(modes)
    expect(signedUploadsFromRows(rows)).toStrictEqual(stored)
  })

  const ISSUER_KEY_NAMES = [
    "issuing_tool_name",
    "allowed_upload_hosts",
    "upload_path_prefix",
    "content_types",
  ] as const
  const KEY_SET_CASES: [string, unknown, SignedUploadMode][] = [
    ["exactly the issuer keys", ISSUER_ENTRY, "issuer"],
    ["the issuer keys with values of the wrong type", { ...ISSUER_ENTRY, allowed_upload_hosts: "uploads.example.test" }, "issuer"],
    ["the six replacement keys", SIGNED_UPLOADS[0], "replacement"],
    ["the issuer keys and tool_name", { ...ISSUER_ENTRY, tool_name: "attach_document" }, "replacement"],
    ["the issuer keys and file_argument", { ...ISSUER_ENTRY, file_argument: "document_file_id" }, "replacement"],
    ["the issuer keys and an unknown key", { ...ISSUER_ENTRY, note: "x" }, "replacement"],
    ...ISSUER_KEY_NAMES.map((key): [string, unknown, SignedUploadMode] => {
      const entry: Record<string, unknown> = { ...ISSUER_ENTRY }
      delete entry[key]
      return [`the issuer keys without ${key}`, entry, "replacement"]
    }),
    ["four keys that are not the issuer keys", { tool_name: "attach_document", allowed_upload_hosts: ["uploads.example.test"], upload_path_prefix: "/staged/", content_types: { ".pdf": "application/pdf" } }, "replacement"],
    ["an empty object", {}, "replacement"],
    ["a value that is not an object", "request_upload", "replacement"],
  ]

  it.each(KEY_SET_CASES)("reads an entry as an issuer row only when it has exactly the issuer keys: %s", (_name, entry, mode) => {
    expect(signedUploadRowsFromValue([entry], rowIds())[0].mode).toBe(mode)
  })

  it("shows only the issuer fields for an issuer entry", async () => {
    installUploadsApiMock([uploadsServer()], { signed_uploads: [ISSUER_ENTRY] })
    const dialog = await openUploadsEditor()

    expect(dialog.getByRole("radio", { name: `${KEY}.modeIssuer` })).toBeChecked()
    expect(dialog.getByRole("radio", { name: `${KEY}.modeReplacement` })).not.toBeChecked()
    expect(dialog.getByRole("radiogroup", { name: `${KEY}.mode` })).toBeInTheDocument()
    expect(dialog.queryByLabelText(`${KEY}.toolName`)).not.toBeInTheDocument()
    expect(dialog.queryByLabelText(`${KEY}.fileArgument`)).not.toBeInTheDocument()
    expect(dialog.getByLabelText(`${KEY}.issuingToolName`)).toHaveValue("request_upload")
    expect(dialog.getByText(`${KEY}.modeIssuerHint`)).toBeInTheDocument()
    expect(dialog.getByText("request_upload", { selector: "span" })).toBeInTheDocument()
  })

  it("keeps a stored issuer entry when the owner saves without touching uploads", async () => {
    const requests = installUploadsApiMock([uploadsServer()], { signed_uploads: BOTH_KINDS })
    const dialog = await openUploadsEditor()
    fireEvent.change(dialog.getByLabelText("tools.mcp.form.descriptionLabel"), {
      target: { value: "Updated" },
    })
    fireEvent.click(saveButton(dialog))
    await waitFor(() => expect(requests.put).toHaveLength(1))
    expect(requests.put[0]).toStrictEqual({ description: "Updated" })
  })

  it.each([
    ["the replacement row", 0, [{ ...REPLACEMENT_ENTRY, upload_path_prefix: "/incoming/" }, ISSUER_ENTRY]],
    ["the issuer row", 1, [REPLACEMENT_ENTRY, { ...ISSUER_ENTRY, upload_path_prefix: "/incoming/" }]],
  ])("keeps a stored issuer entry when the owner edits %s", async (_name, index, expected) => {
    const requests = installUploadsApiMock([uploadsServer()], { signed_uploads: BOTH_KINDS })
    const dialog = await openUploadsEditor()
    fireEvent.change(dialog.getAllByLabelText(`${KEY}.uploadPathPrefix`)[index], {
      target: { value: "/incoming/" },
    })
    fireEvent.click(saveButton(dialog))
    await waitFor(() => expect(requests.put).toHaveLength(1))
    expect(requests.put[0]).toStrictEqual({ signed_uploads: expected })
  })

  it("sends no upload settings after an upload edit is undone", async () => {
    const requests = installUploadsApiMock([uploadsServer()], { signed_uploads: BOTH_KINDS })
    const dialog = await openUploadsEditor()
    const prefix = dialog.getAllByLabelText(`${KEY}.uploadPathPrefix`)[0]
    fireEvent.change(prefix, { target: { value: "/incoming/" } })
    fireEvent.change(prefix, { target: { value: "/staged/" } })
    fireEvent.change(dialog.getByLabelText("tools.mcp.form.descriptionLabel"), {
      target: { value: "Updated" },
    })
    fireEvent.click(saveButton(dialog))
    await waitFor(() => expect(requests.put).toHaveLength(1))
    expect(requests.put[0]).toStrictEqual({ description: "Updated" })
  })

  it.each([
    ["the replacement row", 0, [ISSUER_ENTRY]],
    ["the issuer row", 1, [REPLACEMENT_ENTRY]],
  ])("removes only %s", async (_name, index, expected) => {
    const requests = installUploadsApiMock([uploadsServer()], { signed_uploads: BOTH_KINDS })
    const dialog = await openUploadsEditor()
    fireEvent.click(dialog.getAllByRole("button", { name: `${KEY}.removeTool` })[index])
    fireEvent.click(saveButton(dialog))
    await waitFor(() => expect(requests.put).toHaveLength(1))
    expect(requests.put[0]).toStrictEqual({ signed_uploads: expected })
  })

  it("writes a new issuer entry with exactly the issuer keys", async () => {
    const requests = installUploadsApiMock([uploadsServer()], { signed_uploads: [REPLACEMENT_ENTRY] })
    const dialog = await openUploadsEditor()
    fireEvent.click(dialog.getByRole("button", { name: `${KEY}.addTool` }))
    expect(dialog.getAllByRole("radio", { name: `${KEY}.modeReplacement` })[1]).toBeChecked()
    fireEvent.click(dialog.getAllByRole("radio", { name: `${KEY}.modeIssuer` })[1])
    expect(dialog.getAllByLabelText(`${KEY}.toolName`)).toHaveLength(1)
    fireEvent.change(dialog.getAllByLabelText(`${KEY}.issuingToolName`)[1], { target: { value: " request_upload " } })
    fireEvent.change(dialog.getAllByLabelText(`${KEY}.allowedUploadHosts`)[1], { target: { value: "uploads.example.test\n" } })
    fireEvent.change(dialog.getAllByLabelText(`${KEY}.uploadPathPrefix`)[1], { target: { value: "/staged/" } })
    fireEvent.change(dialog.getAllByLabelText(`${KEY}.extension`)[1], { target: { value: ".pdf" } })
    fireEvent.change(dialog.getAllByLabelText(`${KEY}.contentType`)[1], { target: { value: "application/pdf" } })
    fireEvent.click(saveButton(dialog))
    await waitFor(() => expect(requests.put).toHaveLength(1))
    expect(requests.put[0]).toStrictEqual({ signed_uploads: [REPLACEMENT_ENTRY, ISSUER_ENTRY] })
  })

  it.each([
    ["a replacement entry to an issuer entry", [SIGNED_UPLOADS[0]], ["issuer"], [ISSUER_ENTRY]],
    ["an issuer entry to a replacement entry", [ISSUER_ENTRY], ["replacement"], [{ ...ISSUER_ENTRY, tool_name: "", file_argument: "" }]],
    ["a replacement entry to an issuer entry and back", [SIGNED_UPLOADS[0]], ["issuer", "replacement"], [{ ...ISSUER_ENTRY, tool_name: "", file_argument: "" }]],
  ])("switching %s keeps only that mode's keys", async (_name, stored, clicks, expected) => {
    const requests = installUploadsApiMock([uploadsServer()], { signed_uploads: stored })
    const dialog = await openUploadsEditor()
    for (const mode of clicks) {
      fireEvent.click(dialog.getByRole("radio", { name: mode === "issuer" ? `${KEY}.modeIssuer` : `${KEY}.modeReplacement` }))
    }
    if (clicks[clicks.length - 1] === "issuer") {
      expect(dialog.queryByLabelText(`${KEY}.toolName`)).not.toBeInTheDocument()
    } else {
      expect(dialog.getByLabelText(`${KEY}.toolName`)).toHaveValue("")
      expect(dialog.getByLabelText(`${KEY}.fileArgument`)).toHaveValue("")
    }
    fireEvent.click(saveButton(dialog))
    await waitFor(() => expect(requests.put).toHaveLength(1))
    expect(requests.put[0]).toStrictEqual({ signed_uploads: expected })
  })
})

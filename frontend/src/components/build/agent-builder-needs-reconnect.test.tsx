import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

// A selected connector whose /api/mcp/servers row reports
// connection_status: "needs_reconnect" (#2931) used to render as connected,
// because the chip treated the row's presence as connected. The chip must say
// it needs a reconnect and offer one, without the row leaving mcpServers: the
// saved mcp: selector is resolved from that row.

const apiRequestMock = vi.hoisted(() => vi.fn())
const authUser = vi.hoisted(() => ({ current: { id: "1", is_admin: false } as { id: string; is_admin: boolean } }))

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>("@/lib/api-wrapper")
  return { ...actual, apiRequest: apiRequestMock }
})

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return {
    ...actual,
    getApiUrl: () => "http://api.local",
    getUploadApiUrl: () => "http://api.local",
    getWsUrl: () => "ws://api.local",
  }
})

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    state: {
      messages: [],
      traceEvents: [],
      currentTask: null,
      isProcessing: false,
      isHistoryLoading: false,
      taskId: null,
      filePreview: { isOpen: false },
      dagExecution: null,
      steps: [],
    },
    setTaskId: vi.fn(),
    sendMessage: vi.fn(),
    dispatch: vi.fn(),
    closeFilePreview: vi.fn(),
    pauseTask: vi.fn(),
    resumeTask: vi.fn(),
    openFilePreview: vi.fn(),
    requestStatus: vi.fn(),
  }),
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ token: "token", user: authUser.current }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    locale: "en",
    t: (key: string, vars?: Record<string, string>) => (vars?.appName ? `${key}:${vars.appName}` : key),
  }),
}))

// /api/mcp/apps reports the catalog entry as not connected once its grant
// stops working; the row in /api/mcp/servers is what still says otherwise.
const zendeskApp = {
  id: "zendesk",
  name: "Zendesk",
  description: "Tickets",
  icon: "",
  users: "",
  transport: "oauth",
  provider: "zendesk",
  category: "Support",
  auth_type: "builtin_oauth" as const,
  is_connected: false,
}

const catalog = vi.hoisted(() => ({ authType: "builtin_oauth" as string }))

vi.mock("@/contexts/mcp-apps-context", () => ({
  useMcpApps: () => ({
    apps: [{ ...zendeskApp, auth_type: catalog.authType }],
    getAppIcon: () => null,
    refresh: vi.fn(async () => {}),
  }),
}))

vi.mock("@/lib/branding", () => ({
  getBrandingFromEnv: () => ({ appName: "Xagent" }),
}))

vi.mock("sonner", () => ({ toast: { error: vi.fn(), success: vi.fn() } }))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useSearchParams: () => ({ get: () => null }),
}))

vi.mock("@/components/layout/resizable-three-column-layout", () => ({
  ResizableThreeColumnLayout: ({ middlePanel }: { middlePanel: React.ReactNode }) => <div>{middlePanel}</div>,
}))

vi.mock("@/components/task/task-conversation-panel", () => ({
  TaskConversationPanel: () => null,
}))

vi.mock("@/components/build/agent-builder-chat", () => ({ AgentBuilderChat: () => null }))
vi.mock("@/components/kb/knowledge-base-creation-dialog", () => ({
  KnowledgeBaseCreationDialog: () => null,
}))
vi.mock("@/components/mcp/connect-mcp-dialog", () => ({
  ConnectMcpDialog: () => null,
}))
vi.mock("@/components/mcp/official-mcp-settings-dialog", () => ({
  OfficialMcpSettingsDialog: ({
    open,
    app,
  }: {
    open: boolean
    app: { id: string; connection_status?: string | null } | null
  }) => (open ? <output data-testid="reconnect-dialog">{`${app?.id}:${app?.connection_status}`}</output> : null),
}))
vi.mock("@/components/chat/FileMentionDropdown", () => ({ FileMentionDropdown: () => null }))
vi.mock("@/hooks/use-file-mention", () => ({
  useFileMention: () => ({
    checkTrigger: vi.fn(),
    isOpen: false,
    items: [],
    selectedIndex: 0,
    selectItem: vi.fn(),
    close: vi.fn(),
  }),
}))
vi.mock("@/components/ui/multi-select", () => ({ MultiSelect: () => null }))
vi.mock("@/components/ui/select", () => ({ Select: () => null }))
vi.mock("@/components/build/build-file-preview-sheet", () => ({
  BuildFilePreviewSheet: () => null,
}))

import { AgentBuilder } from "./agent-builder"

const AGENT_ID = "5"
const BADGE = "tools.mcp.needsReconnect"
const RECONNECT = "tools.mcp.dialog.reconnect"

type Status = "connected" | "needs_reconnect" | null | undefined

function agentResponse(toolCategories: string[], opts: { ownerId?: number; canEdit?: boolean } = {}) {
  const canEdit = opts.canEdit ?? true
  return {
    id: Number(AGENT_ID),
    user_id: opts.ownerId ?? 1,
    team_id: null,
    name: "Some Agent",
    description: "",
    instructions: "Do the thing",
    execution_mode: "balanced",
    models: { general: 1, small_fast: null, visual: null, compact: null },
    knowledge_bases: [],
    skills: [],
    tool_categories: toolCategories,
    suggested_prompts: [],
    logo_url: null,
    status: "draft",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    widget_enabled: false,
    allowed_domains: [],
    share_enabled: false,
    share_updated_at: null,
    readonly: !canEdit,
    can_edit: canEdit,
  }
}

// Saved under the catalog display name. The chip finds the row through its
// app_id, and the row's own name differs, so a selector resolved from the row
// is distinguishable from the unresolved one.
const INITIAL_TOOL_CATEGORIES = ["mcp:Zendesk"]

let putBody: Record<string, unknown> | undefined

// ownerStatus, when set, is what the admin's owner-scoped (?user_id=) fetch
// reports, so a test can tell the owner's rows from the viewer's own.
function installApi(status: Status, opts: { ownerId?: number; canEdit?: boolean; ownerStatus?: Status } = {}) {
  putBody = undefined
  const rowWith = (rowStatus: Status) => {
    const row: Record<string, unknown> = { id: 9, name: "zendesk-row", app_id: "zendesk", description: "Zendesk row" }
    if (rowStatus !== undefined) row.connection_status = rowStatus
    return row
  }
  apiRequestMock.mockImplementation((url: string, init?: RequestInit) => {
    const json = (body: unknown) => Promise.resolve(new Response(JSON.stringify(body), { status: 200 }))
    if (url.endsWith("/api/kb/collections")) return json({ collections: [] })
    if (url.endsWith("/api/skills/")) return json([])
    if (url.endsWith("/api/tools/available?include_usage=false")) return json({ tools: [] })
    if (url.endsWith("/api/models/?category=llm")) return json([])
    if (url.endsWith("/api/models/user-default")) return json([])
    if (url.includes(`/api/agents/${AGENT_ID}/triggers`)) return json([])
    if (url.includes("/api/mcp/servers?user_id=")) return json([rowWith(opts.ownerStatus ?? status)])
    if (url.includes("/api/mcp/servers")) return json([rowWith(status)])
    if (url.endsWith(`/api/agents/${AGENT_ID}`) && init?.method === "PUT") {
      putBody = JSON.parse(init.body as string)
      return json(agentResponse(putBody!.tool_categories as string[], opts))
    }
    if (url.endsWith(`/api/agents/${AGENT_ID}`)) return json(agentResponse(INITIAL_TOOL_CATEGORIES, opts))
    return json({})
  })
}

async function renderBuilder() {
  render(<AgentBuilder agentId={AGENT_ID} />)
  const nameInput = await screen.findByPlaceholderText("builds.configForm.name.placeholder")
  await waitFor(() => expect(nameInput).toHaveValue("Some Agent"))
  return nameInput
}

describe("AgentBuilder selected connector needs_reconnect", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    authUser.current = { id: "1", is_admin: false }
    catalog.authType = "builtin_oauth"
    ;(globalThis as unknown as { WebSocket: unknown }).WebSocket = vi.fn()
  })

  afterEach(() => cleanup())

  it("marks the connector and opens the app's reconnect dialog", async () => {
    installApi("needs_reconnect")
    await renderBuilder()

    await waitFor(() => expect(screen.getByText(BADGE)).toBeInTheDocument())
    expect(screen.getByText("tools.mcp.needsReconnectDescription")).toBeInTheDocument()
    expect(screen.queryByText("tools.mcp.mcpUnavailable")).not.toBeInTheDocument()
    expect(screen.queryByTestId("reconnect-dialog")).not.toBeInTheDocument()

    // The row is not connected, but its Reconnect action must not look disabled.
    const row = screen.getByRole("button", { name: RECONNECT }).closest("div.rounded-md.border")
    expect(row).not.toBeNull()
    expect(row).not.toHaveClass("opacity-50")

    fireEvent.click(screen.getByRole("button", { name: RECONNECT }))
    expect(screen.getByTestId("reconnect-dialog")).toHaveTextContent("zendesk:needs_reconnect")
  })

  it.each([
    ["connected", "connected" as const],
    ["absent", undefined],
    ["null", null],
  ])("renders a %s status as connected, unchanged", async (_label, status) => {
    installApi(status)
    await renderBuilder()
    // The row's own description is what a connected chip shows.
    await waitFor(() => expect(screen.getByText("Zendesk row")).toBeInTheDocument())

    expect(screen.queryByText(BADGE)).not.toBeInTheDocument()
    expect(screen.queryByText("tools.mcp.mcpUnavailable")).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: RECONNECT })).not.toBeInTheDocument()
  })

  it("keeps saving the selector resolved from the row", async () => {
    installApi("needs_reconnect")
    const nameInput = await renderBuilder()
    await waitFor(() => expect(screen.getByText(BADGE)).toBeInTheDocument())

    const updateButton = screen.getByRole("button", { name: "builds.editor.header.update" })
    fireEvent.change(nameInput, { target: { value: "Some Agent Renamed" } })
    await waitFor(() => expect(updateButton).not.toBeDisabled())
    fireEvent.click(updateButton)

    await waitFor(() => expect(putBody).toBeDefined())
    expect(putBody?.tool_categories).toContain("mcp:zendesk-row")
    expect(putBody?.tool_categories).not.toContain("mcp:Zendesk")
  })

  it("shows the owner's state without a Reconnect when an admin views another user's agent", async () => {
    authUser.current = { id: "1", is_admin: true }
    // The admin's own row is connected; only the owner's needs a reconnect.
    installApi("connected", { ownerId: 99, ownerStatus: "needs_reconnect" })
    await renderBuilder()
    await waitFor(() =>
      expect(apiRequestMock.mock.calls.map(([u]) => String(u))).toContain(
        "http://api.local/api/mcp/servers?user_id=99",
      ),
    )

    await waitFor(() => expect(screen.getByText(BADGE)).toBeInTheDocument())
    expect(screen.queryByRole("button", { name: RECONNECT })).not.toBeInTheDocument()
  })

  it("does not offer Reconnect for an app the reconnect dialog would not treat as builtin OAuth", async () => {
    catalog.authType = "api_key"
    installApi("needs_reconnect")
    await renderBuilder()

    await waitFor(() => expect(screen.getByText(BADGE)).toBeInTheDocument())
    expect(screen.queryByRole("button", { name: RECONNECT })).not.toBeInTheDocument()
  })

  it("does not offer Reconnect on a read-only agent", async () => {
    installApi("needs_reconnect", { canEdit: false })
    render(<AgentBuilder agentId={AGENT_ID} />)
    await screen.findByText("builds.editor.header.readOnly")
    await waitFor(() => expect(screen.getByText(BADGE)).toBeInTheDocument())

    expect(screen.queryByRole("button", { name: RECONNECT })).not.toBeInTheDocument()
  })
})

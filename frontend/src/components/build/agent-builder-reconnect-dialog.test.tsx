import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest"

// The chip's Reconnect (#2931) opens the real OfficialMcpSettingsDialog, runs
// the app-scoped login, and a successful login must refresh the catalog and
// the server rows so the chip leaves the needs_reconnect state. The sibling
// needs-reconnect test stubs the dialog and cannot see that wiring.

const apiRequestMock = vi.hoisted(() => vi.fn())
const refreshMock = vi.hoisted(() => vi.fn(async () => {}))
const serverStatus = vi.hoisted(() => ({ current: "needs_reconnect" as string }))
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
// stops working; the row in /api/mcp/servers is what carries needs_reconnect.
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

vi.mock("@/contexts/mcp-apps-context", () => ({
  useMcpApps: () => ({
    apps: [zendeskApp],
    getAppIcon: () => null,
    refresh: refreshMock,
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
vi.mock("@/components/ui/dialog", () => {
  const passthrough = ({ children }: { children?: React.ReactNode }) => <div>{children}</div>
  return {
    Dialog: ({ open, children }: { open?: boolean; children?: React.ReactNode }) =>
      open ? <div role="dialog">{children}</div> : null,
    DialogContent: passthrough,
    DialogHeader: passthrough,
    DialogFooter: passthrough,
    DialogTitle: ({ children }: { children?: React.ReactNode }) => <h1>{children}</h1>,
    DialogDescription: ({ children }: { children?: React.ReactNode }) => <p>{children}</p>,
  }
})
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
const DISCONNECT = "tools.mcp.dialog.disconnect"

function agentResponse() {
  return {
    id: Number(AGENT_ID),
    user_id: 1,
    team_id: null,
    name: "Some Agent",
    description: "",
    instructions: "Do the thing",
    execution_mode: "balanced",
    models: { general: 1, small_fast: null, visual: null, compact: null },
    knowledge_bases: [],
    skills: [],
    tool_categories: ["mcp:Zendesk"],
    suggested_prompts: [],
    logo_url: null,
    status: "draft",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    widget_enabled: false,
    allowed_domains: [],
    share_enabled: false,
    share_updated_at: null,
    readonly: false,
    can_edit: true,
  }
}

function installApi() {
  apiRequestMock.mockImplementation((url: string) => {
    const json = (body: unknown) => Promise.resolve(new Response(JSON.stringify(body), { status: 200 }))
    if (url.endsWith("/api/kb/collections")) return json({ collections: [] })
    if (url.endsWith("/api/skills/")) return json([])
    if (url.endsWith("/api/tools/available?include_usage=false")) return json({ tools: [] })
    if (url.endsWith("/api/models/?category=llm")) return json([])
    if (url.endsWith("/api/models/user-default")) return json([])
    if (url.includes(`/api/agents/${AGENT_ID}/triggers`)) return json([])
    if (url.includes("/api/mcp/servers")) {
      return json([
        {
          id: 9,
          name: "zendesk-row",
          app_id: "zendesk",
          description: "Zendesk row",
          connection_status: serverStatus.current,
        },
      ])
    }
    if (url.endsWith(`/api/agents/${AGENT_ID}`)) return json(agentResponse())
    return json({})
  })
}

const serversCalls = () =>
  apiRequestMock.mock.calls.filter(([u]) => String(u).endsWith("/api/mcp/servers")).length

describe("AgentBuilder reconnect dialog", () => {
  let openSpy: MockInstance<typeof window.open>

  beforeEach(() => {
    apiRequestMock.mockReset()
    refreshMock.mockClear()
    serverStatus.current = "needs_reconnect"
    authUser.current = { id: "1", is_admin: false }
    ;(globalThis as unknown as { WebSocket: unknown }).WebSocket = vi.fn()
    openSpy = vi.spyOn(window, "open").mockReturnValue({ closed: false, close: vi.fn() } as unknown as Window)
  })

  afterEach(() => {
    cleanup()
    openSpy.mockRestore()
  })

  it("logs in app-scoped and refreshes the connectors once the login succeeds", async () => {
    installApi()
    render(<AgentBuilder agentId={AGENT_ID} />)
    await waitFor(() => expect(screen.getByText(BADGE)).toBeInTheDocument())

    fireEvent.click(screen.getByRole("button", { name: RECONNECT }))
    const dialog = within(screen.getByRole("dialog"))
    expect(dialog.queryByRole("button", { name: DISCONNECT })).not.toBeInTheDocument()
    fireEvent.click(dialog.getByRole("button", { name: RECONNECT }))

    expect(openSpy).toHaveBeenCalledTimes(1)
    const loginUrl = new URL(String(openSpy.mock.calls[0][0]))
    expect(loginUrl.pathname).toBe("/api/auth/zendesk/login")
    expect(loginUrl.searchParams.get("app_id")).toBe("zendesk")

    const callsBefore = serversCalls()
    serverStatus.current = "connected"
    act(() => {
      window.dispatchEvent(new MessageEvent("message", { data: { type: "oauth-success" } }))
    })

    await waitFor(() => expect(screen.queryByText(BADGE)).not.toBeInTheDocument())
    // Connected, not merely badge-free: an unresolved chip would read "MCP unavailable".
    expect(screen.queryByText("tools.mcp.mcpUnavailable")).not.toBeInTheDocument()
    expect(screen.getByText("Zendesk row")).toBeInTheDocument()
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(serversCalls()).toBeGreaterThan(callsBefore)
    expect(refreshMock).toHaveBeenCalled()
  })
})

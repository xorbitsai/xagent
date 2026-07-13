import React from "react"
import { cleanup, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

// Exercises the admin cross-user MCP-list path in AgentBuilder's edit-mode
// loadAgent effect: an admin opening another user's agent must fetch that
// owner's MCP servers (?user_id=), a self-owned agent must not, and a load that
// is torn down mid-flight must not fire the owner fetch (active cleanup flag).

const apiRequestMock = vi.hoisted(() => vi.fn())
const authUser = vi.hoisted(() => ({ current: { id: "1", is_admin: true } as any }))

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper"
  )
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
    t: (key: string, vars?: Record<string, string>) =>
      vars?.appName ? `${key}:${vars.appName}` : key,
  }),
}))

vi.mock("@/contexts/mcp-apps-context", () => ({
  useMcpApps: () => ({ apps: [], getAppIcon: () => null }),
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
  ResizableThreeColumnLayout: ({ middlePanel }: { middlePanel: React.ReactNode }) => (
    <div>{middlePanel}</div>
  ),
}))

vi.mock("@/components/task/task-conversation-panel", () => ({
  TaskConversationPanel: () => null,
}))

vi.mock("@/components/build/agent-builder-chat", () => ({ AgentBuilderChat: () => null }))
vi.mock("@/components/kb/knowledge-base-creation-dialog", () => ({
  KnowledgeBaseCreationDialog: () => null,
}))
vi.mock("@/components/mcp/connect-mcp-dialog", () => ({ ConnectMcpDialog: () => null }))
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

function agentResponse(ownerId: number, canEdit = true) {
  return {
    id: Number(AGENT_ID),
    user_id: ownerId,
    name: "Some Agent",
    description: "",
    instructions: "",
    execution_mode: "balanced",
    models: null,
    knowledge_bases: [],
    skills: [],
    tool_categories: ["mcp:foo"],
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

// Base handler for the mount-time fetchData resources + agent detail. `ownerId`
// controls who owns the loaded agent; `agentPromise` lets a test defer the agent
// response to drive the mid-flight teardown case.
function installApi(
  ownerId: number,
  opts: { agentPromise?: Promise<Response>; canEdit?: boolean } = {}
) {
  apiRequestMock.mockImplementation((url: string) => {
    if (url.endsWith("/api/kb/collections"))
      return Promise.resolve(new Response(JSON.stringify({ collections: [] }), { status: 200 }))
    if (url.endsWith("/api/skills/"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.endsWith("/api/tools/available"))
      return Promise.resolve(new Response(JSON.stringify({ tools: [] }), { status: 200 }))
    if (url.endsWith("/api/models/?category=llm"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.endsWith("/api/models/user-default"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.includes(`/api/agents/${AGENT_ID}/triggers`))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.endsWith(`/api/agents/${AGENT_ID}`))
      return opts.agentPromise ??
        Promise.resolve(
          new Response(JSON.stringify(agentResponse(ownerId, opts.canEdit ?? true)), {
            status: 200,
          })
        )
    // Both the mount-time (no user_id) and owner-scoped (?user_id=) MCP fetches.
    if (url.includes("/api/mcp/servers"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    return Promise.resolve(new Response(JSON.stringify({}), { status: 200 }))
  })
}

const mcpCalls = () =>
  apiRequestMock.mock.calls.map(([u]) => String(u)).filter((u) => u.includes("/api/mcp/servers"))

describe("AgentBuilder admin cross-user MCP list", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    authUser.current = { id: "1", is_admin: true }
    ;(globalThis as any).WebSocket = vi.fn()
  })

  afterEach(() => cleanup())

  it("fetches the owner's MCP servers when an admin opens another user's agent", async () => {
    installApi(99) // agent owned by user 99, admin is user 1
    render(<AgentBuilder agentId={AGENT_ID} />)

    await waitFor(() =>
      expect(mcpCalls()).toContain("http://api.local/api/mcp/servers?user_id=99")
    )
  })

  it("does not scope the MCP fetch when an admin opens their own agent", async () => {
    installApi(1) // agent owned by the admin themselves
    render(<AgentBuilder agentId={AGENT_ID} />)

    await waitFor(() => expect(mcpCalls()).toContain("http://api.local/api/mcp/servers"))
    expect(mcpCalls().some((u) => u.includes("user_id="))).toBe(false)
  })

  it("locks the builder read-only and hides Save when can_edit is false", async () => {
    installApi(99, { canEdit: false }) // admin opening a read-only agent
    const { container } = render(<AgentBuilder agentId={AGENT_ID} />)

    // Read-only badge appears once the agent detail loads.
    await screen.findByText("builds.editor.header.readOnly")
    // The Save/Publish buttons are gone, so there's no "edits but save fails" trap.
    expect(screen.queryByText("builds.editor.header.update")).toBeNull()
    expect(screen.queryByText("builds.editor.header.create")).toBeNull()
    expect(screen.queryByText("builds.editor.header.publish")).toBeNull()

    // The form is actually locked, not just badge-swapped: native fields are
    // disabled via the fieldset and the instructions editor drops contentEditable.
    expect(screen.getByPlaceholderText("builds.configForm.name.placeholder")).toBeDisabled()
    const editor = container.querySelector('div[role="textbox"]')
    expect(editor).not.toBeNull()
    expect(editor).toHaveAttribute("contenteditable", "false")
  })

  it("does not fire the owner-scoped fetch if unmounted before the agent load resolves", async () => {
    let resolveAgent: (r: Response) => void = () => {}
    const agentPromise = new Promise<Response>((resolve) => {
      resolveAgent = resolve
    })
    installApi(99, { agentPromise })

    const { unmount } = render(<AgentBuilder agentId={AGENT_ID} />)
    unmount()
    // Agent detail resolves only after teardown; the active flag must gate it.
    resolveAgent(new Response(JSON.stringify(agentResponse(99)), { status: 200 }))
    await Promise.resolve()
    await Promise.resolve()

    expect(mcpCalls().some((u) => u.includes("user_id=99"))).toBe(false)
  })
})

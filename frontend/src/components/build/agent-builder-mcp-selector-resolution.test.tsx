import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

// #1280 round-3 review, Major finding 2: the pure resolveMcpToolSelector unit
// tests cover the helper in isolation, but nothing exercised the two things
// this PR's fix actually depends on end to end: (a) the outbound save
// request's tool_categories carries the *resolved* selector, not whatever
// selectedMcpServers holds, and (b) isDirty returns to false after a
// successful save instead of staying permanently true because
// selectedMcpServers was never re-seeded to match. This drives both through
// the real AgentBuilder component and a mocked PUT /api/agents/{id}.

const apiRequestMock = vi.hoisted(() => vi.fn())

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
  useAuth: () => ({ token: "token", user: { id: "1", is_admin: false } }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    locale: "en",
    t: (key: string, vars?: Record<string, string>) =>
      vars?.appName ? `${key}:${vars.appName}` : key,
  }),
}))

// The catalog app: display name "Chrome" deliberately diverging from its
// real connected server row's name "chrome-devtools" (installApi below) --
// the exact divergence this whole fix exists for.
vi.mock("@/contexts/mcp-apps-context", () => ({
  useMcpApps: () => ({
    apps: [
      {
        id: "chrome-devtools",
        name: "Chrome",
        description: "",
        icon: "",
        users: "",
        transport: "stdio",
        provider: "",
        category: "Productivity",
        is_connected: true,
      },
    ],
    getAppIcon: () => null,
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
vi.mock("@/components/mcp/connect-mcp-dialog", () => ({
  ConnectMcpDialog: () => null,
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

function agentResponse(toolCategories: string[]) {
  return {
    id: Number(AGENT_ID),
    user_id: 1,
    team_id: null,
    name: "Some Agent",
    description: "",
    instructions: "Do the thing",
    execution_mode: "balanced",
    // handleCreate's save validation requires a general model to be set.
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
    readonly: false,
    can_edit: true,
  }
}

// The agent was saved before this fix landed, with the unresolved display
// name -- exactly the shape the original Chrome bug produced.
const INITIAL_TOOL_CATEGORIES = ["mcp:Chrome"]

let putBody: Record<string, unknown> | undefined

function installApi() {
  putBody = undefined
  apiRequestMock.mockImplementation((url: string, init?: RequestInit) => {
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
    // The real connected MCPServer row: named after the app_id
    // ("chrome-devtools"), not the display name -- Chrome's actual
    // shared-server-catalog convention (_ensure_catalog_app_server).
    if (url.includes("/api/mcp/servers"))
      return Promise.resolve(
        new Response(JSON.stringify([{ id: 1, name: "chrome-devtools" }]), { status: 200 })
      )
    if (url.endsWith(`/api/agents/${AGENT_ID}`) && init?.method === "PUT") {
      putBody = JSON.parse(init.body as string)
      return Promise.resolve(
        new Response(JSON.stringify(agentResponse(putBody!.tool_categories as string[])), {
          status: 200,
        })
      )
    }
    if (url.endsWith(`/api/agents/${AGENT_ID}`))
      return Promise.resolve(
        new Response(JSON.stringify(agentResponse(INITIAL_TOOL_CATEGORIES)), { status: 200 })
      )
    return Promise.resolve(new Response(JSON.stringify({}), { status: 200 }))
  })
}

describe("AgentBuilder mcp: selector resolution on save", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    installApi()
    ;(globalThis as unknown as { WebSocket: unknown }).WebSocket = vi.fn()
  })

  afterEach(() => cleanup())

  it("saves the resolved server-row name, not the unresolved display name, and clears isDirty afterward", async () => {
    render(<AgentBuilder agentId={AGENT_ID} />)

    const nameInput = await screen.findByPlaceholderText(
      "builds.configForm.name.placeholder"
    )
    await waitFor(() => expect(nameInput).toHaveValue("Some Agent"))

    // The Update button starts disabled: nothing has changed yet, and
    // selectedMcpServers/originalData's MCP extraction agree on the loaded
    // (still-unresolved) "Chrome" -- loading an already-broken agent
    // doesn't self-heal it without an edit, by design (round-1 review).
    const updateButton = screen.getByRole("button", {
      name: "builds.editor.header.update",
    })
    expect(updateButton).toBeDisabled()

    // A trivial, unrelated edit is enough to make isDirty true and reach
    // the save path -- this test isn't about the name change itself.
    fireEvent.change(nameInput, { target: { value: "Some Agent Renamed" } })
    await waitFor(() => expect(updateButton).not.toBeDisabled())

    fireEvent.click(updateButton)

    await waitFor(() => expect(putBody).toBeDefined())

    // (a) The outbound PUT resolved "Chrome" to the real connected row's
    // name, not the display name that was actually in selectedMcpServers.
    expect(putBody?.tool_categories).toContain("mcp:chrome-devtools")
    expect(putBody?.tool_categories).not.toContain("mcp:Chrome")

    // (b) isDirty clears back to false once the save succeeds, observed via
    // the Update button re-disabling with no further edits. Before the
    // selectedMcpServers re-seed fix, this stayed enabled forever: the PUT
    // response's tool_categories rewrote originalData to hold
    // "chrome-devtools" while selectedMcpServers kept holding "Chrome",
    // and isDirty's plain string comparison of the two never matched again.
    await waitFor(() => expect(updateButton).toBeDisabled())
  })
})

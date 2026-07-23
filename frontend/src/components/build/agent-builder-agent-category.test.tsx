import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

// Issue #802: the `agent` tool category (multi-agent delegation) is a
// Workforce concern and must not be assignable from the agent builder.
// Covers the two frontend guarantees: the category never appears in the
// tool selector, and a legacy agent that still has it saved neither
// shows it as selected nor writes it back on save.

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())

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

vi.mock("@/contexts/mcp-apps-context", () => ({
  useMcpApps: () => ({ apps: [], getAppIcon: () => null }),
}))

vi.mock("@/lib/branding", () => ({
  getBrandingFromEnv: () => ({ appName: "Xagent" }),
}))

vi.mock("sonner", () => ({ toast: { error: toastErrorMock, success: vi.fn() } }))

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
// Render each MultiSelect's option values so tests can assert on what the
// tool-category selector actually offers.
vi.mock("@/components/ui/multi-select", () => ({
  MultiSelect: (props: any) => (
    <div data-testid="multi-select" data-placeholder={props.placeholder}>
      {(props.options || []).map((o: any) => o.value).join("|")}
    </div>
  ),
}))
vi.mock("@/components/ui/select", () => ({ Select: () => null }))
vi.mock("@/components/build/build-file-preview-sheet", () => ({
  BuildFilePreviewSheet: () => null,
}))

import { AgentBuilder } from "./agent-builder"

const AGENT_ID = "5"

const AVAILABLE_TOOLS = [
  { name: "calculator", description: "", category: "basic", enabled: true },
  { name: "web_search", description: "", category: "web_search", enabled: true },
  { name: "agent_7", description: "", category: "agent", enabled: true },
  { name: "misc_tool", description: "", category: "other", enabled: true },
]

function agentResponse(toolCategories: string[]) {
  return {
    id: Number(AGENT_ID),
    user_id: 1,
    team_id: null,
    name: "Legacy Agent",
    description: "",
    instructions: "You are a legacy agent.",
    execution_mode: "balanced",
    models: { general: "10" },
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
    can_edit: true,
  }
}

function installApi(
  toolCategories: string[],
  putResponse: () => Response = () =>
    new Response(JSON.stringify(agentResponse(toolCategories)), { status: 200 })
) {
  apiRequestMock.mockImplementation((url: string, opts?: { method?: string }) => {
    if (opts?.method === "PUT")
      return Promise.resolve(putResponse())
    if (url.endsWith("/api/kb/collections"))
      return Promise.resolve(new Response(JSON.stringify({ collections: [] }), { status: 200 }))
    if (url.endsWith("/api/skills/"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.endsWith("/api/tools/available"))
      return Promise.resolve(
        new Response(JSON.stringify({ tools: AVAILABLE_TOOLS }), { status: 200 })
      )
    if (url.endsWith("/api/models/?category=llm"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.endsWith("/api/models/user-default"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.includes(`/api/agents/${AGENT_ID}/triggers`))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.endsWith(`/api/agents/${AGENT_ID}`))
      return Promise.resolve(
        new Response(JSON.stringify(agentResponse(toolCategories)), { status: 200 })
      )
    if (url.includes("/api/mcp/servers"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    return Promise.resolve(new Response(JSON.stringify({}), { status: 200 }))
  })
}

const toolCategorySelector = () =>
  screen
    .getAllByTestId("multi-select")
    .find((el) => el.getAttribute("data-placeholder") === "builds.configForm.tools.placeholder")

beforeEach(() => {
  apiRequestMock.mockReset()
  toastErrorMock.mockReset()
  ;(globalThis as any).WebSocket = vi.fn()
})

afterEach(() => cleanup())

describe("AgentBuilder agent tool category (issue #802)", () => {
  it("does not offer unassignable categories even when such tools exist", async () => {
    installApi(["basic"])
    render(<AgentBuilder agentId={AGENT_ID} />)

    await waitFor(() => {
      const selector = toolCategorySelector()
      expect(selector).toBeTruthy()
      expect(selector!.textContent).toContain("basic")
    })
    const offered = toolCategorySelector()!.textContent!.split("|")
    expect(offered).not.toContain("agent")
    expect(offered).not.toContain("other")
  })

  it("saves a legacy agent without writing the agent category back", async () => {
    installApi(["basic", "agent"])
    render(<AgentBuilder agentId={AGENT_ID} />)

    // Wait for the agent to load (name lands in the header input).
    await waitFor(() =>
      expect(
        screen.getByPlaceholderText("builds.configForm.name.placeholder")
      ).toHaveValue("Legacy Agent")
    )

    fireEvent.click(screen.getByText("builds.editor.header.update"))

    await waitFor(() => {
      const putCall = apiRequestMock.mock.calls.find(
        ([, opts]) => (opts as any)?.method === "PUT"
      )
      expect(putCall).toBeTruthy()
      const body = JSON.parse((putCall![1] as any).body)
      expect(body.tool_categories).toEqual(["basic"])
    })
  })
})

describe("AgentBuilder update error handling (issue #956)", () => {
  it.each([
    {
      name: "a plain string detail",
      payload: { detail: " Agent update failed with string detail " },
      expected: "Agent update failed with string detail",
    },
    {
      name: "a top-level message",
      payload: { message: " Agent update failed with top-level message " },
      expected: "Agent update failed with top-level message",
    },
    {
      name: "a detail object without a readable message",
      payload: { detail: { code: 123 } },
      expected: "builds.editor.error.unknown",
    },
    {
      name: "a detail array without readable entries",
      payload: {
        detail: [1, true, null, { msg: " " }, { message: " " }],
      },
      expected: "builds.editor.error.unknown",
    },
  ])("handles $name without unmounting the builder", async ({ payload, expected }) => {
    installApi(
      ["basic"],
      () =>
        new Response(JSON.stringify(payload), {
          status: 422,
          headers: { "Content-Type": "application/json" },
        })
    )
    render(<AgentBuilder agentId={AGENT_ID} />)

    const nameInput = await screen.findByPlaceholderText(
      "builds.configForm.name.placeholder"
    )
    fireEvent.change(nameInput, { target: { value: "Updated Agent" } })
    fireEvent.click(screen.getByText("builds.editor.header.update"))

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalled()
      expect(toastErrorMock.mock.calls.at(-1)?.[0]).toBe(expected)
    })
    expect(screen.getByDisplayValue("Updated Agent")).toBeInTheDocument()
  })

  it("renders a structured detail message without unmounting the builder", async () => {
    installApi(
      ["basic"],
      () =>
        new Response(
          JSON.stringify({
            detail: {
              message: "Agent update failed",
              context: [],
            },
          }),
          { status: 422, headers: { "Content-Type": "application/json" } }
        )
    )
    render(<AgentBuilder agentId={AGENT_ID} />)

    const nameInput = await screen.findByPlaceholderText(
      "builds.configForm.name.placeholder"
    )
    fireEvent.change(nameInput, { target: { value: "Updated Agent" } })
    fireEvent.click(screen.getByText("builds.editor.header.update"))

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalled()
      expect(toastErrorMock.mock.calls.at(-1)?.[0]).toBe("Agent update failed")
    })
    expect(screen.getByDisplayValue("Updated Agent")).toBeInTheDocument()
  })

  it("uses the localized fallback for an empty response without unmounting the builder", async () => {
    installApi(["basic"], () => new Response(null, { status: 500 }))
    render(<AgentBuilder agentId={AGENT_ID} />)

    const nameInput = await screen.findByPlaceholderText(
      "builds.configForm.name.placeholder"
    )
    fireEvent.change(nameInput, { target: { value: "Updated Agent" } })
    fireEvent.click(screen.getByText("builds.editor.header.update"))

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalled()
      expect(toastErrorMock.mock.calls.at(-1)?.[0]).toBe(
        "builds.editor.error.unknown"
      )
    })
    expect(screen.getByDisplayValue("Updated Agent")).toBeInTheDocument()
  })

  it("renders FastAPI validation detail messages without unmounting the builder", async () => {
    installApi(
      ["basic"],
      () =>
        new Response(
          JSON.stringify({
            detail: [
              { msg: " Name must be 200 characters or fewer " },
              " Invalid model selection ",
              { message: " Unsupported execution mode " },
              { msg: " " },
            ],
          }),
          { status: 422, headers: { "Content-Type": "application/json" } }
        )
    )
    render(<AgentBuilder agentId={AGENT_ID} />)

    const nameInput = await screen.findByPlaceholderText(
      "builds.configForm.name.placeholder"
    )
    fireEvent.change(nameInput, { target: { value: "Updated Agent" } })
    fireEvent.click(screen.getByText("builds.editor.header.update"))

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalled()
      expect(toastErrorMock.mock.calls.at(-1)?.[0]).toBe(
        "Name must be 200 characters or fewer; Invalid model selection; Unsupported execution mode"
      )
    })
    expect(screen.getByDisplayValue("Updated Agent")).toBeInTheDocument()
  })
})

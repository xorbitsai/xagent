import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { Task } from "@/contexts/app-context-chat"
import { toast } from "sonner"

const apiRequestMock = vi.hoisted(() => vi.fn())
const setTaskIdMock = vi.hoisted(() => vi.fn())
const sendMessageMock = vi.hoisted(() => vi.fn())
const previewSendErrorMock = vi.hoisted(() => vi.fn())
const dispatchMock = vi.hoisted(() => vi.fn())
const taskConversationPanelMock = vi.hoisted(() => vi.fn())
const closeFilePreviewMock = vi.hoisted(() => vi.fn())
const connectMcpDialogMock = vi.hoisted(() => vi.fn())
const multiSelectMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper"
  )
  return {
    ...actual,
    apiRequest: apiRequestMock,
  }
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
      traceEvents: [],
      isHistoryLoading: false,
      filePreview: { isOpen: false },
      dagExecution: null,
      steps: [],
      ...previewState,
    },
    setTaskId: setTaskIdMock,
    sendMessage: sendMessageMock,
    dispatch: dispatchMock,
    closeFilePreview: closeFilePreviewMock,
    pauseTask: vi.fn(),
    resumeTask: vi.fn(),
    openFilePreview: vi.fn(),
    requestStatus: vi.fn(),
  }),
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ token: "token" }),
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

vi.mock("sonner", () => ({
  toast: {
    error: vi.fn(),
    success: vi.fn(),
  },
}))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useSearchParams: () => ({ get: (key: string) => key === "template" ? selectedTemplateId : null }),
}))

vi.mock("@/components/layout/resizable-three-column-layout", () => ({
  ResizableThreeColumnLayout: ({ leftPanel, middlePanel, rightPanel }: { leftPanel: React.ReactNode; middlePanel: React.ReactNode; rightPanel: React.ReactNode }) => (
    <div>
      {leftPanel}
      <div data-testid="middle-panel">{middlePanel}</div>
      <div data-testid="right-panel">{rightPanel}</div>
    </div>
  ),
}))

vi.mock("@/components/task/task-conversation-panel", () => ({
  TaskConversationPanel: (props: { onSend?: (message: string, config?: any, files?: File[]) => Promise<void> }) => {
    taskConversationPanelMock(props)
    return (
      <button type="button" onClick={() => props.onSend?.("Preview this").catch(previewSendErrorMock)}>
        send-preview-message
      </button>
    )
  },
}))

vi.mock("@/components/chat/ChatInput", () => ({
  ChatInput: ({ onSend }: { onSend?: (message: string) => void }) => (
    <button type="button" onClick={() => onSend?.("add web search")}>send-chat-input</button>
  ),
}))

vi.mock("@/components/chat/ChatMessage", () => ({
  ChatMessage: () => null,
}))

class MockWebSocket {
  static OPEN = 1
  static instances: MockWebSocket[] = []
  readyState = 0
  sentMessages: string[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  constructor() { MockWebSocket.instances.push(this) }
  send(message: string) { this.sentMessages.push(message) }
  close() {}
  open() { this.readyState = MockWebSocket.OPEN; this.onopen?.() }
}

vi.mock("@/components/kb/knowledge-base-creation-dialog", () => ({
  KnowledgeBaseCreationDialog: () => null,
}))

vi.mock("@/components/mcp/connect-mcp-dialog", () => ({
  ConnectMcpDialog: (props: unknown) => {
    connectMcpDialogMock(props)
    return null
  },
}))

vi.mock("@/components/chat/FileMentionDropdown", () => ({
  FileMentionDropdown: () => null,
}))

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

vi.mock("@/components/ui/multi-select", () => ({
  MultiSelect: (props: unknown) => {
    multiSelectMock(props)
    return null
  },
}))

vi.mock("@/components/ui/select", () => ({
  Select: () => null,
}))

vi.mock("@/components/build/build-file-preview-sheet", () => ({
  BuildFilePreviewSheet: () => null,
}))

import { AgentBuilder } from "./agent-builder"

let storedToolCategories: string[] = ["ssh"]
let putBody: { tool_categories?: string[] } | undefined
let availableTools: unknown[] = []
let selectedTemplateId: string | null = null
let savedPreview: Record<string, unknown> | null = null
let previewState: {
  messages: Array<{ role: string }>
  currentTask: Pick<Task, "id" | "status" | "completionOutcome"> | null
  taskId: number | null
  isProcessing: boolean
}

describe("AgentBuilder preview", () => {
  const originalWebSocket = globalThis.WebSocket

  const expectSectionHighlighted = (section: string, highlighted: boolean) => {
    const label = section === "mcp" ? "tools.mcp.dialog.connector" : `builds.configForm.${section}.label`
    const element = screen.getByText(label, { selector: "label" }).closest(".transition-all")
    expect(element).not.toBeNull()
    expect(element?.classList.contains("border-primary/30")).toBe(highlighted)
  }

  beforeEach(() => {
    storedToolCategories = ["ssh"]
    putBody = undefined
    availableTools = []
    selectedTemplateId = null
    savedPreview = null
    previewState = { messages: [], currentTask: null, taskId: null, isProcessing: false }
    apiRequestMock.mockReset()
    setTaskIdMock.mockReset()
    sendMessageMock.mockReset()
    previewSendErrorMock.mockReset()
    vi.mocked(toast.error).mockClear()
    dispatchMock.mockReset()
    taskConversationPanelMock.mockReset()
    sendMessageMock.mockResolvedValue(undefined)
    globalThis.WebSocket = vi.fn() as any

    apiRequestMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith("/api/kb/collections")) {
        return Promise.resolve(new Response(JSON.stringify({ collections: [] }), { status: 200 }))
      }
      if (url.endsWith("/api/skills/")) {
        return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
      }
      if (url.endsWith("/api/tools/available?include_usage=false")) {
        return Promise.resolve(new Response(JSON.stringify({ tools: availableTools }), { status: 200 }))
      }
      if (url.endsWith("/api/models/?category=llm")) {
        return Promise.resolve(
          new Response(JSON.stringify([{ id: 7, model_id: "gpt-test", model_name: "GPT Test", model_provider: "test", category: "llm" }]), {
            status: 200,
          })
        )
      }
      if (url.endsWith("/api/models/user-default")) {
        return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
      }
      if (url.endsWith("/api/mcp/servers")) {
        return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
      }
      if (url.endsWith("/api/agents/42/triggers")) {
        return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
      }
      if (url.endsWith("/preview-task")) {
        return Promise.resolve(new Response(JSON.stringify(savedPreview)))
      }
      if (url.endsWith("/api/agents/42") && init?.method === "PUT") {
        putBody = JSON.parse(init.body as string)
        return Promise.resolve(new Response(JSON.stringify({ id: 42, ...putBody, logo_url: null }), { status: 200 }))
      }
      if (url.endsWith("/api/agents/42")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              id: 42,
              user_id: 1,
              name: "Existing SSH agent",
              description: "Saved description",
              instructions: "Saved instructions",
              execution_mode: "balanced",
              suggested_prompts: [],
              visibility: "team",
              team_id: null,
              knowledge_bases: [],
              skills: [],
              tool_categories: storedToolCategories,
              logo_url: null,
              models: {
                general: 7,
                small_fast: null,
                visual: null,
                compact: null,
              },
              can_edit: true,
              status: "draft",
              origin: "user",
              widget_enabled: false,
              allowed_domains: [],
              share_enabled: false,
            }),
            { status: 200 },
          ),
        )
      }
      if (url.endsWith("/api/chat/task/create")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              task_id: 123,
              title: "Preview this",
              description: "Preview this",
              status: "pending",
              created_at: "2026-01-01T00:00:00Z",
              updated_at: "2026-01-01T00:00:00Z",
              model_id: "gpt-test",
              execution_mode: "balanced",
              is_dag: true,
            }),
            { status: 200 }
          )
        )
      }
      return Promise.resolve(new Response(JSON.stringify({}), { status: 200 }))
    })
  })

  afterEach(() => {
    cleanup()
    globalThis.WebSocket = originalWebSocket
  })

  it("creates a hidden normal task and sends through the app task path", async () => {
    render(<AgentBuilder />)

    fireEvent.click(await screen.findByText("send-preview-message"))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/chat/task/create",
        expect.objectContaining({
          method: "POST",
          body: expect.any(String),
        })
      )
    })

    const createCall = apiRequestMock.mock.calls.find(([url]) => String(url).endsWith("/api/chat/task/create"))
    expect(JSON.parse(createCall?.[1]?.body as string)).toMatchObject({
      title: "Preview this",
      description: "Preview this",
      execution_mode: "balanced",
      is_visible: false,
      agent_config: {
        is_preview: true,
      },
    })

    await waitFor(() => {
      expect(setTaskIdMock).toHaveBeenCalledWith(123, { navigate: false })
      expect(sendMessageMock).toHaveBeenCalledWith("Preview this", expect.objectContaining({ force: true }), undefined)
    })
    expect(globalThis.WebSocket).not.toHaveBeenCalled()
  })

  it("keeps an edit-mode agent identity inside preview config only", async () => {
    render(<AgentBuilder agentId="42" />)

    await screen.findByDisplayValue("Existing SSH agent")
    fireEvent.click(screen.getByText("send-preview-message"))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/chat/task/create",
        expect.objectContaining({ method: "POST", body: expect.any(String) }),
      )
    })

    const createCall = apiRequestMock.mock.calls.find(([url]) =>
      String(url).endsWith("/api/chat/task/create"),
    )
    const payload = JSON.parse(createCall?.[1]?.body as string)
    expect(payload.agent_id).toBeUndefined()
    expect(payload.agent_config).toMatchObject({
      preview_agent_id: 42,
      is_preview: true,
      tool_categories: ["ssh"],
    })
  })

  it("defers uploads until the preview send", async () => {
    render(<AgentBuilder agentId="42" />)
    await screen.findByDisplayValue("Existing SSH agent")
    expect(taskConversationPanelMock.mock.lastCall![0]).toMatchObject({
      deferFileUpload: true,
      onSend: expect.any(Function),
    })
  })

  it("rejects a preview without a model and shows the specific error", async () => {
    const base = apiRequestMock.getMockImplementation()!
    apiRequestMock.mockImplementation((url, init) => url.endsWith("/api/models/user-default") || url.endsWith("/api/models/?category=llm")
      ? Promise.resolve(new Response(url.endsWith("user-default") ? "{}" : "[]")) : base(url, init))
    render(<AgentBuilder />)
    fireEvent.click(await screen.findByText("send-preview-message"))
    await screen.findByText("builds.preview.errors.noModel")
    expect(previewSendErrorMock).toHaveBeenCalledWith(expect.objectContaining({ notificationHandled: true }))
    expect(sendMessageMock).not.toHaveBeenCalled()
    expect(apiRequestMock.mock.calls.some(([url]) => url.endsWith("/api/chat/task/create"))).toBe(false)
  })

  it.each(["not_sent", "rejected"])("returns to an empty preview when the first send is %s", async disposition => {
    sendMessageMock.mockRejectedValueOnce(Object.assign(new Error("Delivery failed"), { disposition }))
    render(<AgentBuilder agentId="42" />)
    await screen.findByDisplayValue("Existing SSH agent")
    fireEvent.click(screen.getByText("send-preview-message"))
    await screen.findByText(`builds.preview.errors.${disposition === "not_sent" ? "notSent" : "rejected"}`)
    expect(previewSendErrorMock).toHaveBeenCalled()
    expect(setTaskIdMock).toHaveBeenLastCalledWith(null, { navigate: false })
    expect(dispatchMock).toHaveBeenCalledWith({ type: "SET_HISTORY_LOADING", payload: false })
    expect(screen.queryByText("builds.preview.rerunSample")).not.toBeInTheDocument()
    fireEvent.click(screen.getByText("send-preview-message"))
    await waitFor(() => expect(sendMessageMock).toHaveBeenCalledTimes(2))
    expect(apiRequestMock.mock.calls.filter(([url]) => url.endsWith("/api/chat/task/create"))).toHaveLength(2)
  })

  describe("preview completion", () => {
    const expectPreviewComplete = (complete: boolean) => {
      const step = screen.getByRole("button", { name: /builds.editor.stepGuide.preview/ })
      expect(step.querySelector("svg") !== null).toBe(complete)
    }

    it.each(["edit", "create"])("preserves a completed preview after saving in %s mode", async (mode) => {
      const baseImpl = apiRequestMock.getMockImplementation()!
      let createdAgent: Record<string, unknown> | undefined
      apiRequestMock.mockImplementation((url: string, init?: RequestInit) => {
        if (url.endsWith("/api/agents") && init?.method === "POST") {
          createdAgent = { ...JSON.parse(init.body as string), id: 43, team_id: null, can_edit: true, status: "draft", logo_url: null }
          return Promise.resolve(new Response(JSON.stringify(createdAgent)))
        }
        if (url.endsWith("/api/agents/43")) {
          return Promise.resolve(new Response(JSON.stringify(createdAgent)))
        }
        if (url.endsWith("/api/agents/43/triggers")) {
          return Promise.resolve(new Response("[]"))
        }
        return baseImpl(url, init)
      })
      const builder = () => <AgentBuilder agentId={mode === "edit" ? "42" : undefined} />
      const { rerender } = render(builder())
      if (mode === "edit") {
        await screen.findByDisplayValue("Existing SSH agent")
      } else {
        fireEvent.change(await screen.findByPlaceholderText("builds.configForm.name.placeholder"), { target: { value: "New preview agent" } })
        const editor = document.querySelector("[contenteditable]") as HTMLElement
        editor.textContent = "Saved instructions"
        fireEvent.input(editor)
      }
      // Let the default model finish loading before attempting the preview.
      await waitFor(() => {
        if (!sendMessageMock.mock.calls.length) fireEvent.click(screen.getByText("send-preview-message"))
        expect(sendMessageMock).toHaveBeenCalled()
      })
      previewState = { messages: [{ role: "user" }], currentTask: { id: "123", status: "completed" }, taskId: 123, isProcessing: false }
      rerender(builder())
      expectPreviewComplete(true)
      if (mode === "edit") {
        // Update is disabled for a clean form; renaming enables it without changing execution config.
        fireEvent.change(screen.getByDisplayValue("Existing SSH agent"), { target: { value: "Renamed preview agent" } })
      }
      fireEvent.click(screen.getByText(`builds.editor.header.${mode === "edit" ? "update" : "create"}`))
      await waitFor(() => expect(screen.getByText("builds.editor.header.update")).toBeDisabled())
      if (mode === "create") {
        expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/agents/43/preview-task", expect.objectContaining({
          method: "PUT", body: JSON.stringify({ task_id: 123 }),
        }))
        await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/agents/43"))
        expect(screen.getByText("builds.editor.success.createdDesc")).toBeInTheDocument()
        expect(screen.queryByText("builds.editor.header.startChat")).not.toBeInTheDocument()
        fireEvent.click(screen.getByText("builds.editor.success.keepEditing"))
        expect(apiRequestMock.mock.calls.some(([url]) => String(url).endsWith("/publish"))).toBe(false)
      }
      expectPreviewComplete(true)
    })

    it.each(["pending", "running", "failed", "paused", "waiting_for_user"] as const)(
      "does not count a user message on a %s task as a completed preview",
      async (status) => {
        const { rerender } = render(<AgentBuilder agentId="42" />)
        await screen.findByDisplayValue("Existing SSH agent")
        fireEvent.click(screen.getByText("send-preview-message"))
        await waitFor(() => expect(sendMessageMock).toHaveBeenCalled())
        previewState = { messages: [{ role: "user" }], currentTask: { id: "123", status }, taskId: 123, isProcessing: status === "running" }
        rerender(<AgentBuilder agentId="42" />)
        expectPreviewComplete(false)
      },
    )

    it.each(["partial", "blocked"] as const)("does not count %s as preview completion and permits a completed follow-up", async (completionOutcome) => {
      const { rerender } = render(<AgentBuilder agentId="42" />)
      await screen.findByDisplayValue("Existing SSH agent")
      fireEvent.click(screen.getByText("send-preview-message"))
      await waitFor(() => expect(sendMessageMock).toHaveBeenCalled())
      previewState = { messages: [{ role: "user" }], currentTask: { id: "123", status: "completed", completionOutcome }, taskId: 123, isProcessing: false }
      rerender(<AgentBuilder agentId="42" />)
      expectPreviewComplete(false)
      fireEvent.click(screen.getByText("send-preview-message"))
      await waitFor(() => expect(sendMessageMock).toHaveBeenCalledTimes(2))
      previewState.currentTask = { id: "123", status: "completed", completionOutcome: "completed" }
      rerender(<AgentBuilder agentId="42" />)
      expectPreviewComplete(true)
    })

    it("requires this configuration's completed task and invalidates it on config edits or Clear", async () => {
      const { rerender } = render(<AgentBuilder agentId="42" />)
      await screen.findByDisplayValue("Existing SSH agent")
      fireEvent.click(screen.getByText("send-preview-message"))
      await waitFor(() => expect(sendMessageMock).toHaveBeenCalled())
      previewState.taskId = 123
      rerender(<AgentBuilder agentId="42" />)
      expectPreviewComplete(false)
      previewState = { messages: [{ role: "user" }], currentTask: { id: "999", status: "completed" }, taskId: 999, isProcessing: false }
      rerender(<AgentBuilder agentId="42" />)
      expectPreviewComplete(false)
      previewState = { ...previewState, currentTask: { id: "123", status: "completed" }, taskId: 123 }
      rerender(<AgentBuilder agentId="42" />)
      expectPreviewComplete(true)

      fireEvent.change(screen.getByDisplayValue("Existing SSH agent"), { target: { value: "Renamed" } })
      expectPreviewComplete(true)
      fireEvent.click(screen.getByText("builds.configForm.executionMode.think.title"))
      expectPreviewComplete(false)
      const baseImpl = apiRequestMock.getMockImplementation()!
      apiRequestMock.mockImplementation((url: string, init?: RequestInit) =>
        url.endsWith("/api/chat/task/create")
          ? Promise.resolve(new Response(JSON.stringify({ task_id: 456, status: "pending" })))
          : baseImpl(url, init),
      )
      fireEvent.click(screen.getByText("send-preview-message"))
      await waitFor(() => expect(sendMessageMock).toHaveBeenCalledTimes(2))
      expectPreviewComplete(false)
      previewState = { ...previewState, taskId: 456, currentTask: { id: "456", status: "completed" } }
      rerender(<AgentBuilder agentId="42" />)
      expectPreviewComplete(true)
      fireEvent.click(screen.getByTitle("common.clear"))
      expectPreviewComplete(false)
    })

    it("only completes after a waiting or running preview finishes", async () => {
      const { rerender } = render(<AgentBuilder agentId="42" />)
      await screen.findByDisplayValue("Existing SSH agent")
      fireEvent.click(screen.getByText("send-preview-message"))
      await waitFor(() => expect(sendMessageMock).toHaveBeenCalled())
      for (const status of ["waiting_for_user", "running", "completed"] as const) {
        previewState = { messages: [{ role: "user" }], currentTask: { id: "123", status }, taskId: 123, isProcessing: status === "running" }
        rerender(<AgentBuilder agentId="42" />)
        expectPreviewComplete(status === "completed")
      }
      previewState.isProcessing = true
      rerender(<AgentBuilder agentId="42" />)
      expectPreviewComplete(false)
    })

    it.each(["config", "clear"])("does not restore completion after %s changes before the send is acknowledged", async (change) => {
      let acknowledgeSend!: () => void
      sendMessageMock.mockReturnValueOnce(new Promise<void>((resolve) => { acknowledgeSend = resolve }))
      const { rerender } = render(<AgentBuilder agentId="42" />)
      await screen.findByDisplayValue("Existing SSH agent")
      fireEvent.click(screen.getByText("send-preview-message"))
      await waitFor(() => expect(sendMessageMock).toHaveBeenCalled())
      fireEvent.click(change === "config"
        ? screen.getByText("builds.configForm.executionMode.think.title")
        : screen.getByTitle("common.clear"))
      previewState = { messages: [{ role: "user" }], currentTask: { id: "123", status: "completed" }, taskId: 123, isProcessing: false }
      await act(async () => acknowledgeSend())
      rerender(<AgentBuilder agentId="42" />)
      expectPreviewComplete(false)
    })

    it("withdraws completion while a follow-up is being sent and after delivery fails", async () => {
      const { rerender } = render(<AgentBuilder agentId="42" />)
      await screen.findByDisplayValue("Existing SSH agent")
      fireEvent.click(screen.getByText("send-preview-message"))
      await waitFor(() => expect(sendMessageMock).toHaveBeenCalled())
      previewState = { messages: [{ role: "user" }], currentTask: { id: "123", status: "completed" }, taskId: 123, isProcessing: false }
      rerender(<AgentBuilder agentId="42" />)
      expectPreviewComplete(true)
      let rejectSend!: (error: Error) => void
      sendMessageMock.mockReturnValueOnce(new Promise((_, reject) => { rejectSend = reject }))
      fireEvent.click(screen.getByText("send-preview-message"))
      expectPreviewComplete(false)
      await act(async () => rejectSend(new Error("not delivered")))
      expectPreviewComplete(false)
    })
  })

  describe("configuration completion", () => {
    const configStep = () => screen.getByRole("button", { name: /builds.editor.stepGuide.configure/ })

    it("allows a model-only agent without optional tools, skills, or connections", async () => {
      storedToolCategories = []
      render(<AgentBuilder agentId="42" />)
      await screen.findByDisplayValue("Existing SSH agent")
      await waitFor(() => expect(configStep().querySelector("svg")).not.toBeNull())
      for (const section of ["model", "knowledgeBase", "skills", "tools", "mcp"]) {
        expectSectionHighlighted(section, false)
      }
    })

    it("does not complete configuration just because a tool is selected without a model", async () => {
      const baseImpl = apiRequestMock.getMockImplementation()!
      apiRequestMock.mockImplementation(async (url: string, init?: RequestInit) => {
        if (url.endsWith("/api/agents/42")) {
          const response = await baseImpl(url, init)
          return new Response(JSON.stringify({ ...await response.json(), models: {} }))
        }
        return baseImpl(url, init)
      })
      render(<AgentBuilder agentId="42" />)
      await screen.findByDisplayValue("Existing SSH agent")
      expect(configStep().querySelector("svg")).toBeNull()
      expectSectionHighlighted("model", true)
      for (const section of ["knowledgeBase", "skills", "tools", "mcp"]) {
        expectSectionHighlighted(section, false)
      }
    })

    it.each(["knowledge", "mcp:missing-connector"])("still requires the template's %s capability", async (category) => {
      selectedTemplateId = "required-capability"
      const baseImpl = apiRequestMock.getMockImplementation()!
      apiRequestMock.mockImplementation((url: string, init?: RequestInit) => {
        if (url.endsWith("/api/templates/required-capability")) {
          return Promise.resolve(new Response(JSON.stringify({
            name: "Template agent",
            agent_config: { instructions: "Use the required data source", tool_categories: [category] },
          })))
        }
        return baseImpl(url, init)
      })
      render(<AgentBuilder />)
      await screen.findByDisplayValue("Template agent")
      // The name lands before the default model and MCP catalog settle; requirements are only
      // highlighted once both have.
      await waitFor(() => {
        expectSectionHighlighted("model", false)
        expectSectionHighlighted("knowledgeBase", category === "knowledge")
        expectSectionHighlighted("mcp", category === "mcp:missing-connector")
      })
      expect(configStep().querySelector("svg")).toBeNull()
      expectSectionHighlighted("skills", false)
      expectSectionHighlighted("tools", false)
    })
  })

  describe("binding pre-save previews", () => {
    beforeEach(() => {
      MockWebSocket.instances = []
      globalThis.WebSocket = MockWebSocket as unknown as typeof WebSocket
      const base = apiRequestMock.getMockImplementation()!
      apiRequestMock.mockImplementation((url, init) => {
        if (url.endsWith("/api/agents") && init?.method === "POST") {
          return Promise.resolve(new Response(JSON.stringify({
            ...JSON.parse(init.body), id: 42, team_id: null, can_edit: true, status: "draft",
          })))
        }
        return base(url, init)
      })
    })

    it.each([
      ["manual", "not_sent"], ["chat", "not_sent"],
      ["manual", "rejected"], ["chat", "rejected"],
      ["manual", "accepted"], ["chat", "accepted"],
    ])("binds the settled preview after %s creation during a %s send", async (mode, outcome) => {
      render(<AgentBuilder />)
      fireEvent.change(await screen.findByPlaceholderText("builds.configForm.name.placeholder"), { target: { value: "New agent" } })
      const editor = document.querySelector("[contenteditable]") as HTMLElement
      editor.textContent = "Saved instructions"
      fireEvent.input(editor)
      await waitFor(() => {
        if (!sendMessageMock.mock.calls.length) fireEvent.click(screen.getByText("send-preview-message"))
        expect(sendMessageMock).toHaveBeenCalledOnce()
      })
      if (mode === "chat") {
        fireEvent.click(screen.getByText("send-chat-input"))
        act(() => MockWebSocket.instances[0].open())
        await waitFor(() => expect(MockWebSocket.instances[0].sentMessages).toHaveLength(1))
      }
      let resolveSend!: () => void
      let rejectSend!: (error: Error) => void
      sendMessageMock.mockReturnValueOnce(new Promise<void>((resolve, reject) => { resolveSend = resolve; rejectSend = reject }))
      const base = apiRequestMock.getMockImplementation()!
      apiRequestMock.mockImplementation((url, init) => url.endsWith("/api/chat/task/create")
        ? Promise.resolve(new Response(JSON.stringify({ task_id: 456 }))) : base(url, init))
      fireEvent.click(screen.getByText("builds.preview.rerunSample"))
      await waitFor(() => expect(sendMessageMock).toHaveBeenCalledTimes(2))
      if (mode === "manual") {
        fireEvent.click(screen.getByText("builds.editor.header.create"))
        await waitFor(() => expect(apiRequestMock.mock.calls.some(([url, init]) => url.endsWith("/api/agents") && init?.method === "POST")).toBe(true))
      } else {
        act(() => MockWebSocket.instances[0].onmessage?.({ data: JSON.stringify({
          type: "trace_event", event_id: "created", event_type: "tool_execution_end",
          data: { tool_name: "create_agent", tool_params: { name: "New agent" }, result: { status: "success", agent_id: 42 } },
        }) }))
      }
      const bindings = () => apiRequestMock.mock.calls.filter(([url, init]) => url.endsWith("/preview-task") && init?.method === "PUT")
      if (mode === "manual") {
        // Saving the agent must finish even if the preview upload never does.
        await screen.findByText("builds.editor.success.createdDesc")
        expect(screen.queryByText("builds.editor.header.creating")).not.toBeInTheDocument()
      }
      expect(bindings()).toHaveLength(0)
      await act(async () => {
        if (outcome === "accepted") resolveSend()
        else rejectSend(Object.assign(new Error("Delivery failed"), { disposition: outcome }))
      })
      await waitFor(() => expect(bindings()).toHaveLength(1))
      expect(JSON.parse(bindings()[0][1].body)).toEqual({ task_id: outcome === "accepted" ? 456 : 123 })
    })

    it.each(["manual", "chat"])("keeps an edited preview visible after %s creation", async (mode) => {
      render(<AgentBuilder />)
      fireEvent.change(await screen.findByPlaceholderText("builds.configForm.name.placeholder"), { target: { value: "New agent" } })
      const editor = document.querySelector("[contenteditable]") as HTMLElement
      editor.textContent = "Saved instructions"
      fireEvent.input(editor)
      if (mode === "chat") {
        // Start chat first: its response must bind the latest preview, not the
        // empty record captured when this WebSocket callback was installed.
        fireEvent.click(screen.getByText("send-chat-input"))
        act(() => MockWebSocket.instances[0].open())
        await waitFor(() => expect(MockWebSocket.instances[0].sentMessages).toHaveLength(1))
      }
      await waitFor(() => {
        if (!sendMessageMock.mock.calls.length) fireEvent.click(screen.getByText("send-preview-message"))
        expect(sendMessageMock).toHaveBeenCalled()
      })
      fireEvent.click(screen.getByText("builds.configForm.executionMode.think.title"))
      setTaskIdMock.mockClear()
      if (mode === "manual") {
        fireEvent.click(screen.getByText("builds.editor.header.create"))
      } else {
        act(() => MockWebSocket.instances[0].onmessage?.({ data: JSON.stringify({
          type: "trace_event", event_id: "created", event_type: "tool_execution_end",
          data: { tool_name: "create_agent", tool_params: { name: "New agent" }, result: { status: "success", agent_id: 42 } },
        }) }))
      }
      await screen.findByDisplayValue("Existing SSH agent")
      await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/agents/42/preview-task", expect.objectContaining({
        method: "PUT", body: JSON.stringify({ task_id: 123 }),
      })))
      expect(apiRequestMock.mock.calls.filter(([url, init]) => url.endsWith("/preview-task") && !init?.method)).toHaveLength(0)
      expect(setTaskIdMock).not.toHaveBeenCalled()
      expect(screen.getByText("builds.preview.configChanged")).toBeInTheDocument()
    })

    it("reports binding failure without rolling back the saved draft", async () => {
      const base = apiRequestMock.getMockImplementation()!
      apiRequestMock.mockImplementation((url, init) => url.endsWith("/preview-task") && init?.method === "PUT"
        ? Promise.resolve(new Response("unavailable", { status: 503 })) : base(url, init))
      render(<AgentBuilder />)
      fireEvent.change(await screen.findByPlaceholderText("builds.configForm.name.placeholder"), { target: { value: "New agent" } })
      const editor = document.querySelector("[contenteditable]") as HTMLElement
      editor.textContent = "Saved instructions"
      fireEvent.input(editor)
      await waitFor(() => {
        if (!sendMessageMock.mock.calls.length) fireEvent.click(screen.getByText("send-preview-message"))
        expect(sendMessageMock).toHaveBeenCalled()
      })
      fireEvent.click(screen.getByText("builds.editor.header.create"))
      await waitFor(() => expect(toast.error).toHaveBeenCalledWith("builds.preview.errors.linkFailed", { duration: 8000 }))
      await screen.findByText("builds.editor.success.createdDesc")
      expect(apiRequestMock.mock.calls.some(([, init]) => init?.method === "DELETE")).toBe(false)
      expect(apiRequestMock.mock.calls.some(([url]) => url.endsWith("/publish"))).toBe(false)
    })
  })

  describe("persisted preview", () => {
    const configKey = () => JSON.stringify({
      instructions: "Saved instructions", executionMode: "balanced",
      models: ["7", null, null, null], knowledgeBases: [], skills: [], toolCategories: ["ssh"],
    })

    beforeEach(() => {
      savedPreview = { task_id: 99, config_key: configKey(), message: "Original sample", attachments: [] }
    })

    it("restores on remount via the normal task socket without sending or publishing", async () => {
      const first = render(<AgentBuilder agentId="42" />)
      await waitFor(() => expect(setTaskIdMock).toHaveBeenCalledWith(99, { navigate: false }))
      first.unmount()
      setTaskIdMock.mockClear()
      const { rerender } = render(<AgentBuilder agentId="42" />)
      await waitFor(() => expect(setTaskIdMock).toHaveBeenCalledWith(99, { navigate: false }))
      previewState = { taskId: 99, currentTask: { id: "99", status: "completed" }, messages: [], isProcessing: false }
      rerender(<AgentBuilder agentId="42" />)
      expect(screen.getByRole("button", { name: /builds.editor.stepGuide.preview/ }).querySelector("svg")).not.toBeNull()
      expect(screen.getByText("builds.preview.currentConfig")).toBeInTheDocument()
      expect(sendMessageMock).not.toHaveBeenCalled()
      expect(apiRequestMock.mock.calls.every(([, init]) => !init?.method)).toBe(true)
    })

    it.each(["old-config", null])("does not certify a restored result with snapshot %s", async (key) => {
      savedPreview!.config_key = key
      const { rerender } = render(<AgentBuilder agentId="42" />)
      await waitFor(() => expect(setTaskIdMock).toHaveBeenCalledWith(99, { navigate: false }))
      previewState = { taskId: 99, currentTask: { id: "99", status: "completed" }, messages: [], isProcessing: false }
      rerender(<AgentBuilder agentId="42" />)
      expect(screen.getByRole("button", { name: /builds.editor.stepGuide.preview/ }).querySelector("svg")).toBeNull()
      expect(screen.getByText(key === null ? "builds.preview.configUnknown" : "builds.preview.configChanged")).toBeInTheDocument()
    })

    it("marks execution edits stale but ignores renaming, and reruns the original sample in a fresh task", async () => {
      const attachment = { file_id: "file-99", name: "sample.csv", size: 12, type: "text/csv" }
      savedPreview!.attachments = [attachment]
      const base = apiRequestMock.getMockImplementation()!
      apiRequestMock.mockImplementation((url, init) => url.endsWith("/api/files/download/file-99")
        ? Promise.resolve(new Response("amount\n80\n")) : base(url, init))
      render(<AgentBuilder agentId="42" />)
      await screen.findByText("builds.preview.currentConfig")
      fireEvent.change(screen.getByDisplayValue("Existing SSH agent"), { target: { value: "Renamed" } })
      expect(screen.getByText("builds.preview.currentConfig")).toBeInTheDocument()
      fireEvent.click(screen.getByText("builds.configForm.executionMode.think.title"))
      expect(screen.getByText("builds.preview.configChanged")).toBeInTheDocument()
      expect(sendMessageMock).not.toHaveBeenCalled()
      fireEvent.click(screen.getByText("builds.preview.rerunSample"))
      await waitFor(() => expect(sendMessageMock).toHaveBeenCalledWith("Original sample", { force: true, targetTaskId: 123 }, [expect.any(File)]))
      const copied = sendMessageMock.mock.calls[0][2][0]
      expect(copied.name).toBe("sample.csv")
      expect(copied.type).toBe("text/csv")
      expect(copied.size).toBe(10)
      expect(copied.file_id).toBeUndefined()
      const body = JSON.parse(apiRequestMock.mock.calls.find(([url]) => url.endsWith("/api/chat/task/create"))![1].body)
      expect(body.execution_mode).toBe("think")
      expect(JSON.parse(body.agent_config.preview_config_key).executionMode).toBe("think")
      expect(screen.getByText("builds.preview.currentConfig")).toBeInTheDocument()
    })

    it("reruns in a fresh task even when the configuration is unchanged", async () => {
      render(<AgentBuilder agentId="42" />)
      await screen.findByText("builds.preview.currentConfig")
      fireEvent.click(screen.getByText("builds.preview.rerunSample"))
      await waitFor(() => expect(sendMessageMock).toHaveBeenCalledWith("Original sample", { force: true, targetTaskId: 123 }, []))
    })

    it("rejects an unsent upload and restores the previous preview without consuming the draft", async () => {
      previewState = { taskId: 99, currentTask: { id: "99", status: "completed" }, messages: [{ role: "user" }], isProcessing: false }
      const uploadError = Object.assign(new Error("Upload failed"), { disposition: "not_sent", errorCode: "upload_failed" })
      sendMessageMock.mockRejectedValueOnce(uploadError)
      render(<AgentBuilder agentId="42" />)
      await screen.findByText("builds.preview.currentConfig")
      fireEvent.click(screen.getByText("builds.configForm.executionMode.think.title"))
      const file = new File(["amount\n80"], "draft.csv", { type: "text/csv" })
      await act(async () => {
        await expect(taskConversationPanelMock.mock.lastCall![0].onSend("Keep this draft", undefined, [file]))
          .rejects.toBe(uploadError)
      })
      expect(setTaskIdMock).toHaveBeenLastCalledWith(99, { navigate: false })
      expect(screen.getByText("builds.preview.configChanged")).toBeInTheDocument()
      expect(screen.getByRole("alert")).toHaveTextContent("builds.preview.errors.notSent clientErrors.uploadFailed")
      expect(apiRequestMock.mock.calls.some(([, init]) => init?.method === "DELETE")).toBe(false)
      // A retry keeps the original input/file and allocates a new task.
      await act(async () => {
        await taskConversationPanelMock.mock.lastCall![0].onSend("Keep this draft", undefined, [file])
      })
      expect(apiRequestMock.mock.calls.filter(([url]) => url.endsWith("/api/chat/task/create"))).toHaveLength(2)
      expect(sendMessageMock).toHaveBeenLastCalledWith("Keep this draft", { force: true, targetTaskId: 123 }, [file])
      expect(sendMessageMock.mock.lastCall![2][0].file_id).toBeUndefined()
      expect(screen.queryByText("builds.preview.errors.notSent")).not.toBeInTheDocument()
    })

    it.each([
      ["outcome_unknown", "clientErrors.messageOutcomeUnknown"],
      [undefined, "builds.preview.errors.requestFailed"],
    ])("does not roll back a %s send as an upload failure", async (disposition, hint) => {
      previewState = { taskId: 99, currentTask: { id: "99", status: "completed" }, messages: [], isProcessing: false }
      const error = Object.assign(new Error("Delivery failed"), { disposition })
      sendMessageMock.mockRejectedValueOnce(error)
      render(<AgentBuilder agentId="42" />)
      await screen.findByText("builds.preview.currentConfig")
      fireEvent.click(screen.getByText("builds.configForm.executionMode.think.title"))
      await act(async () => {
        await expect(taskConversationPanelMock.mock.lastCall![0].onSend("New sample", { clientMessageId: "stable-preview-turn" }))
          .rejects.toBe(error)
      })
      expect(setTaskIdMock).toHaveBeenLastCalledWith(123, { navigate: false })
      expect(apiRequestMock.mock.calls.some(([, init]) => init?.method === "DELETE")).toBe(false)
      expect(screen.getByText(hint!)).toBeInTheDocument()
      expect(error).toMatchObject({ notificationHandled: true })
      await act(async () => {
        await taskConversationPanelMock.mock.lastCall![0].onSend("New sample", { clientMessageId: "stable-preview-turn" })
      })
      expect(apiRequestMock.mock.calls.filter(([url]) => url.endsWith("/api/chat/task/create"))).toHaveLength(1)
      expect(sendMessageMock.mock.calls.map(([, config]) => config)).toEqual([
        { force: true, targetTaskId: 123, clientMessageId: "stable-preview-turn" },
        { force: true, targetTaskId: 123, clientMessageId: "stable-preview-turn" },
      ])
    })

    it.each(["accepted", "outcome_unknown"])("retains an existing %s task when a later retry is rejected", async initialOutcome => {
      if (initialOutcome === "outcome_unknown") {
        sendMessageMock.mockRejectedValueOnce(Object.assign(new Error("Ack lost"), { disposition: "outcome_unknown" }))
      }
      render(<AgentBuilder agentId="42" />)
      await screen.findByText("builds.preview.currentConfig")
      fireEvent.click(screen.getByText("builds.configForm.executionMode.think.title"))
      fireEvent.click(screen.getByText("send-preview-message"))
      await waitFor(() => expect(sendMessageMock).toHaveBeenCalledOnce())
      const error = Object.assign(new Error("Changed retry payload"), { disposition: "rejected", errorCode: "message_id_conflict" })
      sendMessageMock.mockRejectedValueOnce(error)
      fireEvent.click(screen.getByText("send-preview-message"))
      await screen.findByText("builds.preview.errors.rejected clientErrors.messageIdConflict")
      expect(setTaskIdMock).toHaveBeenLastCalledWith(123, { navigate: false })
      expect(apiRequestMock.mock.calls.filter(([url]) => url.endsWith("/api/chat/task/create"))).toHaveLength(1)
      expect(screen.getByText("builds.preview.currentConfig")).toBeInTheDocument()
    })

    it.each(["open", "config"])("clears a stale send error after %s changes", async change => {
      sendMessageMock.mockRejectedValueOnce(Object.assign(new Error("Upload failed"), { disposition: "not_sent" }))
      render(<AgentBuilder agentId="42" />)
      await screen.findByText("builds.preview.currentConfig")
      fireEvent.click(screen.getByText("builds.preview.rerunSample"))
      await screen.findByText("builds.preview.errors.notSent")
      fireEvent.click(screen.getByText(change === "open"
        ? "builds.preview.viewLast" : "builds.configForm.executionMode.think.title"))
      expect(screen.queryByRole("alert")).not.toBeInTheDocument()
      if (change === "open") expect(setTaskIdMock).toHaveBeenLastCalledWith(99, { navigate: false })
    })

    it("keeps the previous record without reopening the cleared UI when an upload fails", async () => {
      let rejectSend!: (error: Error) => void
      sendMessageMock.mockReturnValueOnce(new Promise((_, reject) => { rejectSend = reject }))
      render(<AgentBuilder agentId="42" />)
      await screen.findByText("builds.preview.currentConfig")
      fireEvent.click(screen.getByText("builds.preview.rerunSample"))
      await waitFor(() => expect(sendMessageMock).toHaveBeenCalledOnce())
      fireEvent.click(screen.getByTitle("common.clear"))
      await act(async () => rejectSend(Object.assign(new Error("Upload failed"), { disposition: "not_sent" })))
      expect(setTaskIdMock).toHaveBeenLastCalledWith(null, { navigate: false })
      expect(screen.queryByRole("alert")).not.toBeInTheDocument()
      fireEvent.click(screen.getByText("builds.preview.viewLast"))
      expect(setTaskIdMock).toHaveBeenLastCalledWith(99, { navigate: false })
    })

    it.each([false, true])("restores completion only if config did not change during upload (edited: %s)", async edited => {
      previewState = { taskId: 99, currentTask: { id: "99", status: "completed" }, messages: [], isProcessing: false }
      let rejectSend!: (error: Error) => void
      sendMessageMock.mockReturnValueOnce(new Promise((_, reject) => { rejectSend = reject }))
      render(<AgentBuilder agentId="42" />)
      await screen.findByText("builds.preview.currentConfig")
      fireEvent.click(screen.getByText("builds.preview.rerunSample"))
      await waitFor(() => expect(sendMessageMock).toHaveBeenCalled())
      if (edited) fireEvent.click(screen.getByText("builds.configForm.executionMode.think.title"))
      await act(async () => rejectSend(Object.assign(new Error("Upload failed"), { disposition: "not_sent" })))
      expect(setTaskIdMock).toHaveBeenLastCalledWith(99, { navigate: false })
      const step = screen.getByRole("button", { name: /builds.editor.stepGuide.preview/ })
      expect(step.querySelector("svg") !== null).toBe(!edited)
      expect(screen.getByText(edited ? "builds.preview.configChanged" : "builds.preview.currentConfig")).toBeInTheDocument()
    })

    it("does not start a task if an original attachment is deleted or inaccessible", async () => {
      savedPreview!.attachments = [{ file_id: "gone", name: "sample.csv", size: 12 }]
      const base = apiRequestMock.getMockImplementation()!
      apiRequestMock.mockImplementation((url, init) => url.endsWith("/api/files/download/gone")
        ? Promise.resolve(new Response("not found", { status: 404 })) : base(url, init))
      render(<AgentBuilder agentId="42" />)
      await screen.findByText("builds.preview.currentConfig")
      fireEvent.click(screen.getByText("builds.preview.rerunSample"))
      await screen.findByText("builds.preview.errors.filesUnavailable")
      expect(toast.error).not.toHaveBeenCalledWith("clientErrors.taskBusy")
      expect(sendMessageMock).not.toHaveBeenCalled()
      expect(apiRequestMock.mock.calls.some(([url]) => url.endsWith("/api/chat/task/create"))).toBe(false)
    })

    it("rejects a panel send while rerun preparation holds the lock", async () => {
      let finish!: (response: Response) => void
      const base = apiRequestMock.getMockImplementation()!
      apiRequestMock.mockImplementation((url, init) => url.endsWith("/api/chat/task/create")
        ? new Promise<Response>(resolve => { finish = resolve }) : base(url, init))
      render(<AgentBuilder agentId="42" />)
      await screen.findByText("builds.preview.currentConfig")
      fireEvent.click(screen.getByText("builds.preview.rerunSample"))
      await waitFor(() => expect(finish).toBeDefined())
      await expect(taskConversationPanelMock.mock.lastCall![0].onSend("Keep my draft", undefined, [new File(["sample"], "sample.csv")]))
        .rejects.toMatchObject({ errorCode: "task_busy" })
      expect(sendMessageMock).not.toHaveBeenCalled()
      await act(async () => { finish(new Response(JSON.stringify({ task_id: 123 }))) })
      expect(sendMessageMock).toHaveBeenCalledWith("Original sample", { force: true, targetTaskId: 123 }, [])
    })

    it.each(["clear", "send", "unmount"])("ignores late history after %s", async (action) => {
      let finish!: (response: Response) => void
      const base = apiRequestMock.getMockImplementation()!
      apiRequestMock.mockImplementation((url, init) => url.endsWith("/preview-task")
        ? new Promise<Response>(resolve => { finish = resolve }) : base(url, init))
      const view = render(<AgentBuilder agentId="42" />)
      await waitFor(() => expect(finish).toBeDefined())
      if (action === "clear") fireEvent.click(screen.getByTitle("common.clear"))
      if (action === "send") {
        fireEvent.click(screen.getByText("send-preview-message"))
        await waitFor(() => expect(sendMessageMock).toHaveBeenCalled())
      }
      if (action === "unmount") view.unmount()
      await act(async () => { finish(new Response(JSON.stringify(savedPreview))) })
      expect(setTaskIdMock).not.toHaveBeenCalledWith(99, { navigate: false })
    })

    it("reports history fetch failure and permits retry without starting a task", async () => {
      const base = apiRequestMock.getMockImplementation()!
      let fail = true
      apiRequestMock.mockImplementation((url, init) => url.endsWith("/preview-task") && fail
        ? Promise.resolve(new Response("unavailable", { status: 503 })) : base(url, init))
      render(<AgentBuilder agentId="42" />)
      await screen.findByText("builds.preview.errors.historyFailed")
      fail = false
      fireEvent.click(screen.getByText("common.retry"))
      await waitFor(() => expect(setTaskIdMock).toHaveBeenCalledWith(99, { navigate: false }))
      expect(sendMessageMock).not.toHaveBeenCalled()
    })
  })

  describe("after a builder-chat category update", () => {
    beforeEach(() => {
      MockWebSocket.instances = []
      globalThis.WebSocket = MockWebSocket as unknown as typeof WebSocket
    })

    const chatUpdatesCategories = async (
      categories: string[],
      toolParams: Record<string, unknown> = { tool_categories: categories },
    ) => {
      fireEvent.click(screen.getByText("send-chat-input"))
      const ws = MockWebSocket.instances[0]
      act(() => ws.open())
      await waitFor(() => expect(ws.sentMessages).toHaveLength(1))
      act(() => {
        ws.onmessage?.({
          data: JSON.stringify({
            type: "trace_event",
            event_id: "tool-end",
            event_type: "tool_execution_end",
            step_id: "react-1",
            timestamp: 3,
            data: {
              tool_name: "update_agent",
              tool_params: { agent_id: 42, ...toolParams },
              result: { status: "success", agent_id: 42, tool_categories: categories },
            },
          }),
        })
      })
    }

    const previewCategories = async () => {
      fireEvent.click(screen.getByText("send-preview-message"))
      await waitFor(() => {
        expect(apiRequestMock).toHaveBeenCalledWith(
          "http://api.local/api/chat/task/create",
          expect.objectContaining({ method: "POST" }),
        )
      })
      const createCall = apiRequestMock.mock.calls.find(([url]) =>
        String(url).endsWith("/api/chat/task/create"),
      )
      return JSON.parse(createCall?.[1]?.body as string).agent_config.tool_categories
    }

    const saveCategories = async () => {
      const updateButton = screen.getByRole("button", { name: "builds.editor.header.update" })
      await waitFor(() => expect(updateButton).not.toBeDisabled())
      fireEvent.click(updateButton)
      await waitFor(() => expect(putBody).toBeDefined())
      return putBody?.tool_categories
    }

    it.each([
      ["an unsaved connector pick", ["file"], ["github"], ["file", "web_search"], ["file", "web_search", "mcp:github"]],
      ["an unsaved connector removal", ["file", "mcp:github"], [], ["file", "web_search"], ["file", "web_search"]],
      ["a bare mcp grant", ["file", "mcp"], null, ["web_search"], ["web_search", "mcp"]],
      ["stored connectors on an empty result", ["basic", "mcp:github"], null, [], ["mcp:github"]],
    ])("keeps %s", async (_label, stored, picked, chatResult, expected) => {
      storedToolCategories = stored
      render(<AgentBuilder agentId="42" />)
      await screen.findByDisplayValue("Existing SSH agent")
      if (picked) act(() => connectMcpDialogMock.mock.lastCall?.[0].onConnectSelected(picked))

      await chatUpdatesCategories(chatResult)

      expect(await previewCategories()).toEqual(expected)
      expect(await saveCategories()).toEqual(expected)
    })

    it("keeps unsaved picks when the chat call leaves tool_categories null", async () => {
      storedToolCategories = ["file"]
      availableTools = ["basic", "file"].map((category) => ({ name: category, category, enabled: true }))
      render(<AgentBuilder agentId="42" />)
      await screen.findByDisplayValue("Existing SSH agent")
      const toolPicker = () =>
        multiSelectMock.mock.calls
          .filter(([props]) => props.placeholder === "builds.configForm.tools.placeholder")
          .pop()?.[0]
      await waitFor(() => expect(toolPicker()?.values).toEqual(["file"]))
      act(() => toolPicker().onValuesChange(["file", "basic"]))
      act(() => connectMcpDialogMock.mock.lastCall?.[0].onConnectSelected(["github"]))

      await chatUpdatesCategories(["file"], { name: "Renamed", tool_categories: null })

      expect(await previewCategories()).toEqual(["file", "basic", "mcp:github"])
      expect(await saveCategories()).toEqual(["file", "basic", "mcp:github"])
    })
  })

  it("derives preview tool categories the same way as save", async () => {
    const baseImpl = apiRequestMock.getMockImplementation()!
    apiRequestMock.mockImplementation(async (url: string, opts?: RequestInit) => {
      const response = await baseImpl(url, opts)
      if (!url.endsWith("/api/agents/42")) return response
      const agent = await response.json()
      return new Response(
        JSON.stringify({ ...agent, knowledge_bases: ["kb1"], tool_categories: ["mcp:foo"] }),
        { status: 200 },
      )
    })
    render(<AgentBuilder agentId="42" />)

    fireEvent.change(await screen.findByDisplayValue("Existing SSH agent"), {
      target: { value: "Renamed agent" },
    })
    fireEvent.click(screen.getByText("send-preview-message"))
    fireEvent.click(screen.getByText("builds.editor.header.update"))

    const findBody = (match: (url: string, opts?: RequestInit) => boolean) => {
      const call = apiRequestMock.mock.calls.find(([url, opts]) => match(String(url), opts))
      return call ? JSON.parse(call[1].body as string) : undefined
    }
    await waitFor(() => {
      expect(findBody((url) => url.endsWith("/api/chat/task/create"))).toBeDefined()
      expect(findBody((_, opts) => opts?.method === "PUT")).toBeDefined()
    })
    const previewCategories: string[] = findBody((url) => url.endsWith("/api/chat/task/create")).agent_config.tool_categories
    const saveCategories: string[] = findBody((_, opts) => opts?.method === "PUT").tool_categories
    expect(previewCategories).toEqual(expect.arrayContaining(["knowledge", "mcp:foo"]))
    expect([...previewCategories].sort()).toEqual([...saveCategories].sort())
  })

  it("shows task file management in the embedded preview panel", async () => {
    render(<AgentBuilder />)

    await waitFor(() => {
      expect(taskConversationPanelMock).toHaveBeenCalledWith(
        expect.objectContaining({
          mode: "embedded-preview",
          showTaskActions: true,
          showTaskFiles: true,
          showDagPreview: false,
          showTokenUsage: false,
        })
      )
    })
  })

  it("keeps the current preview visible after config changes and recreates the task on the next send", async () => {
    render(<AgentBuilder />)

    fireEvent.click(await screen.findByText("send-preview-message"))

    await waitFor(() => {
      expect(setTaskIdMock).toHaveBeenCalledWith(123, { navigate: false })
    })

    apiRequestMock.mockClear()
    dispatchMock.mockClear()
    setTaskIdMock.mockClear()
    sendMessageMock.mockClear()
    closeFilePreviewMock.mockClear()

    fireEvent.click(screen.getByText("builds.configForm.executionMode.think.title"))

    await waitFor(() => {
      expect(apiRequestMock).not.toHaveBeenCalled()
      expect(setTaskIdMock).not.toHaveBeenCalled()
      expect(sendMessageMock).not.toHaveBeenCalled()
      expect(dispatchMock).not.toHaveBeenCalled()
    })

    fireEvent.click(screen.getByText("send-preview-message"))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/chat/task/create",
        expect.objectContaining({
          method: "POST",
          body: expect.any(String),
        })
      )
    })

    const createCall = apiRequestMock.mock.calls.find(([url]) => String(url).endsWith("/api/chat/task/create"))
    expect(JSON.parse(createCall?.[1]?.body as string)).toMatchObject({
      execution_mode: "think",
      is_visible: false,
      agent_config: {
        is_preview: true,
      },
    })
    expect(closeFilePreviewMock).toHaveBeenCalledTimes(1)
  })

  it("nulls the shared task when the preview resets on mount", async () => {
    render(<AgentBuilder />)
    await screen.findByText("send-preview-message")

    expect(setTaskIdMock).toHaveBeenCalledWith(null, { navigate: false })
  })

  it("nulls the shared task when the preview is cleared", async () => {
    render(<AgentBuilder />)
    await screen.findByText("send-preview-message")
    setTaskIdMock.mockClear()

    fireEvent.click(screen.getByTitle("common.clear"))

    expect(setTaskIdMock).toHaveBeenCalledWith(null, { navigate: false })
  })

  it("does not show App Widget in the builder form (widget moved to Deploy dialog)", async () => {
    // App Widget was removed from the Configure form and is now only accessible
    // via the Deploy Agent dialog. Verify it is absent from the builder UI.
    render(<AgentBuilder />)

    // Wait for the form to render (check for a known element)
    await screen.findByText("builds.configForm.executionMode.balanced.title")

    expect(screen.queryByText("appWidget.builder.title")).not.toBeInTheDocument()
    expect(screen.queryByRole("switch", { name: "appWidget.builder.toggle" })).not.toBeInTheDocument()
  })

  describe("while preview task creation is pending", () => {
    let pendingCreates: Array<(response: Response) => void>

    beforeEach(() => {
      pendingCreates = []
      const baseImpl = apiRequestMock.getMockImplementation()!
      apiRequestMock.mockImplementation((url: string, init?: RequestInit) =>
        url.endsWith("/api/chat/task/create")
          ? new Promise<Response>((resolve) => pendingCreates.push(resolve))
          : baseImpl(url, init),
      )
    })

    const sendPreview = async () => {
      const button = await screen.findByText("send-preview-message")
      // A click before the default model loads hits the no-model guard and rejects into
      // previewSendErrorMock, so wait until the model section is no longer flagged.
      await waitFor(() => expectSectionHighlighted("model", false))
      fireEvent.click(button)
      await waitFor(() => expect(pendingCreates).toHaveLength(1))
    }

    const settle = (fn: () => void) =>
      act(async () => {
        fn()
        await new Promise((resolve) => setTimeout(resolve, 0))
      })

    const resolveCreate = (taskId: number) =>
      settle(() =>
        pendingCreates.shift()!(
          new Response(JSON.stringify({ task_id: taskId, title: "Preview this", status: "pending" }), { status: 200 }),
        ),
      )

    const expectDropped = (taskId: number) => {
      expect(setTaskIdMock).not.toHaveBeenCalledWith(taskId, expect.anything())
      expect(dispatchMock).not.toHaveBeenCalledWith(
        expect.objectContaining({ type: "SET_CURRENT_TASK", payload: expect.objectContaining({ id: String(taskId) }) }),
      )
      expect(sendMessageMock).not.toHaveBeenCalled()
    }

    it("drops a task that resolves after Clear", async () => {
      render(<AgentBuilder />)
      await sendPreview()

      fireEvent.click(screen.getByTitle("common.clear"))
      await resolveCreate(123)

      expectDropped(123)
    })

    it.each(["create", "send"])("serializes sends across Clear while %s is pending", async (phase) => {
      let acknowledgeSend!: () => void
      if (phase === "send") {
        sendMessageMock.mockReturnValueOnce(new Promise<void>((resolve) => { acknowledgeSend = resolve }))
      }
      render(<AgentBuilder />)
      await sendPreview()
      if (phase === "send") await resolveCreate(123)

      // Neither a repeated click nor Clear releases the in-flight send's lock.
      fireEvent.click(screen.getByText("send-preview-message"))
      fireEvent.click(screen.getByTitle("common.clear"))
      fireEvent.click(screen.getByText("send-preview-message"))
      const createCalls = () => apiRequestMock.mock.calls.filter(([url]) => String(url).endsWith("/api/chat/task/create"))
      expect(createCalls()).toHaveLength(1)
      expect(sendMessageMock).toHaveBeenCalledTimes(phase === "send" ? 1 : 0)
      await waitFor(() => expect(previewSendErrorMock).toHaveBeenCalledTimes(2))
      expect(previewSendErrorMock.mock.calls[0][0]).toMatchObject({ errorCode: "task_busy" })

      if (phase === "send") {
        await settle(acknowledgeSend)
      } else {
        await resolveCreate(123)
        expectDropped(123)
      }

      // A fresh send can start only after the old operation has settled.
      await sendPreview()
      fireEvent.click(screen.getByText("send-preview-message"))
      expect(createCalls()).toHaveLength(2)
      await resolveCreate(456)
      expect(sendMessageMock.mock.calls.map(([, config]) => config.targetTaskId))
        .toEqual(phase === "send" ? [123, 456] : [456])
    })

    it("drops a task that resolves after the builder unmounts", async () => {
      const { unmount } = render(<AgentBuilder />)
      await sendPreview()

      unmount()
      await resolveCreate(123)

      expectDropped(123)
    })

    it("reuses the created task for the next send when the config is unchanged", async () => {
      render(<AgentBuilder />)
      await sendPreview()
      await resolveCreate(123)

      fireEvent.click(screen.getByText("send-preview-message"))

      await waitFor(() => expect(sendMessageMock).toHaveBeenCalledTimes(2))
      expect(apiRequestMock.mock.calls.filter(([url]) => String(url).endsWith("/api/chat/task/create"))).toHaveLength(1)
      expect(sendMessageMock.mock.calls.map(([, config]) => config.targetTaskId)).toEqual([123, 123])
    })

    it("delivers the pending message after a config edit but starts the next send on a new task", async () => {
      render(<AgentBuilder />)
      await sendPreview()

      fireEvent.click(screen.getByText("builds.configForm.executionMode.think.title"))
      await resolveCreate(123)

      expect(setTaskIdMock).toHaveBeenCalledWith(123, { navigate: false })
      expect(sendMessageMock).toHaveBeenCalledWith("Preview this", expect.objectContaining({ targetTaskId: 123 }), undefined)

      await sendPreview()
      const createModes = apiRequestMock.mock.calls
        .filter(([url]) => String(url).endsWith("/api/chat/task/create"))
        .map(([, init]) => JSON.parse(init.body as string).execution_mode)
      expect(createModes).toEqual(["balanced", "think"])
    })

    it.each([undefined, "not_sent", "rejected"])("does not restore a %s send that fails after Clear", async disposition => {
      let rejectSend!: (error: Error) => void
      sendMessageMock.mockReturnValue(new Promise((_, reject) => { rejectSend = reject }))
      render(<AgentBuilder />)
      await sendPreview()
      await resolveCreate(123)
      expect(sendMessageMock).toHaveBeenCalled()

      fireEvent.click(screen.getByTitle("common.clear"))
      await settle(() => rejectSend(Object.assign(new Error("reset before delivery"), { disposition })))

      expect(setTaskIdMock).toHaveBeenLastCalledWith(null, { navigate: false })
      expect(screen.queryByText("builds.preview.errors.notSent")).not.toBeInTheDocument()
      expect(screen.queryByText("builds.preview.errors.requestFailed")).not.toBeInTheDocument()
      expect(screen.queryByText("builds.preview.rerunSample") !== null).toBe(disposition === undefined)

      expect(dispatchMock).not.toHaveBeenCalledWith(
        expect.objectContaining({
          type: "ADD_MESSAGE",
          payload: expect.objectContaining({ content: "builds.preview.errors.requestFailed" }),
        }),
      )
    })

    it("still reports a send that fails without a reset", async () => {
      sendMessageMock.mockRejectedValue(new Error("send failed"))
      render(<AgentBuilder />)
      await sendPreview()
      await resolveCreate(123)

      await screen.findByText("builds.preview.errors.requestFailed")
      expect(previewSendErrorMock).toHaveBeenCalled()
    })
  })
})

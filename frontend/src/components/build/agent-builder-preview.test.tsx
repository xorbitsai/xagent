import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { Task } from "@/contexts/app-context-chat"

const apiRequestMock = vi.hoisted(() => vi.fn())
const setTaskIdMock = vi.hoisted(() => vi.fn())
const sendMessageMock = vi.hoisted(() => vi.fn())
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
  useSearchParams: () => ({ get: () => null }),
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
  TaskConversationPanel: (props: { onSend?: (message: string, config?: any, files?: File[]) => void }) => {
    taskConversationPanelMock(props)
    return (
      <button type="button" onClick={() => props.onSend?.("Preview this")}>
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
let previewState: {
  messages: Array<{ role: string }>
  currentTask: Pick<Task, "id" | "status" | "completionOutcome"> | null
  taskId: number | null
  isProcessing: boolean
}

describe("AgentBuilder preview", () => {
  const originalWebSocket = globalThis.WebSocket

  beforeEach(() => {
    storedToolCategories = ["ssh"]
    putBody = undefined
    availableTools = []
    previewState = { messages: [], currentTask: null, taskId: null, isProcessing: false }
    apiRequestMock.mockReset()
    setTaskIdMock.mockReset()
    sendMessageMock.mockReset()
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
      if (url.endsWith("/api/tools/available")) {
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
        await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/agents/43"))
        fireEvent.click(screen.getByText("common.cancel"))
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
      // A click before the default model loads hits the no-model guard, so retry until one reaches task/create.
      await waitFor(() => {
        if (pendingCreates.length === 0) fireEvent.click(button)
        expect(pendingCreates).toHaveLength(1)
      })
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

    it("does not report a send that fails after Clear", async () => {
      let rejectSend!: (error: Error) => void
      sendMessageMock.mockReturnValue(new Promise((_, reject) => { rejectSend = reject }))
      render(<AgentBuilder />)
      await sendPreview()
      await resolveCreate(123)
      expect(sendMessageMock).toHaveBeenCalled()

      fireEvent.click(screen.getByTitle("common.clear"))
      await settle(() => rejectSend(new Error("reset before delivery")))

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

      await waitFor(() =>
        expect(dispatchMock).toHaveBeenCalledWith(
          expect.objectContaining({
            type: "ADD_MESSAGE",
            payload: expect.objectContaining({ content: "builds.preview.errors.requestFailed" }),
          }),
        ),
      )
    })
  })
})

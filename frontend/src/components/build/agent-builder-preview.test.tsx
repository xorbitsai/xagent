import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const setTaskIdMock = vi.hoisted(() => vi.fn())
const sendMessageMock = vi.hoisted(() => vi.fn())
const dispatchMock = vi.hoisted(() => vi.fn())
const taskConversationPanelMock = vi.hoisted(() => vi.fn())
const closeFilePreviewMock = vi.hoisted(() => vi.fn())

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
  ResizableThreeColumnLayout: ({ middlePanel, rightPanel }: { middlePanel: React.ReactNode; rightPanel: React.ReactNode }) => (
    <div>
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

vi.mock("@/components/build/agent-builder-chat", () => ({
  AgentBuilderChat: () => null,
}))

vi.mock("@/components/kb/knowledge-base-creation-dialog", () => ({
  KnowledgeBaseCreationDialog: () => null,
}))

vi.mock("@/components/mcp/connect-mcp-dialog", () => ({
  ConnectMcpDialog: () => null,
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
  MultiSelect: () => null,
}))

vi.mock("@/components/ui/select", () => ({
  Select: () => null,
}))

vi.mock("@/components/build/build-file-preview-sheet", () => ({
  BuildFilePreviewSheet: () => null,
}))

import { AgentBuilder } from "./agent-builder"

describe("AgentBuilder preview", () => {
  const originalWebSocket = globalThis.WebSocket

  beforeEach(() => {
    apiRequestMock.mockReset()
    setTaskIdMock.mockReset()
    sendMessageMock.mockReset()
    dispatchMock.mockReset()
    taskConversationPanelMock.mockReset()
    sendMessageMock.mockResolvedValue(undefined)
    globalThis.WebSocket = vi.fn() as any

    apiRequestMock.mockImplementation((url: string) => {
      if (url.endsWith("/api/kb/collections")) {
        return Promise.resolve(new Response(JSON.stringify({ collections: [] }), { status: 200 }))
      }
      if (url.endsWith("/api/skills/")) {
        return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
      }
      if (url.endsWith("/api/tools/available")) {
        return Promise.resolve(new Response(JSON.stringify({ tools: [] }), { status: 200 }))
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

  it("does not show App Widget in the builder form (widget moved to Deploy dialog)", async () => {
    // App Widget was removed from the Configure form and is now only accessible
    // via the Deploy Agent dialog. Verify it is absent from the builder UI.
    render(<AgentBuilder />)

    // Wait for the form to render (check for a known element)
    await screen.findByText("builds.configForm.executionMode.balanced.title")

    expect(screen.queryByText("appWidget.builder.title")).not.toBeInTheDocument()
    expect(screen.queryByRole("switch", { name: "appWidget.builder.toggle" })).not.toBeInTheDocument()
  })
})

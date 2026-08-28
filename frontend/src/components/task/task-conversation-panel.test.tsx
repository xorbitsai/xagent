import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

type PanelTraceEvent = {
  event_id?: string
  event_type?: string
  timestamp?: string | number
  data?: Record<string, unknown>
  [key: string]: unknown
}

const appState = vi.hoisted(() => ({
  messages: [] as Array<Record<string, unknown>>,
  traceEvents: [] as PanelTraceEvent[],
  currentTask: null as null | Record<string, unknown>,
  taskRuntimeExtensions: {} as Record<string, Record<string, unknown>>,
  isProcessing: false,
  isHistoryLoading: false,
  taskId: 42,
  filePreview: { isOpen: false, fileId: "", fileName: "", viewMode: "preview" },
  dagExecution: null,
  steps: [],
}))
const openFilePreviewMock = vi.hoisted(() => vi.fn())
const sendMessageMock = vi.hoisted(() => vi.fn())
const pauseTaskMock = vi.hoisted(() => vi.fn())
const resumeTaskMock = vi.hoisted(() => vi.fn())
const getFileDownloadUrlMock = vi.hoisted(() => vi.fn())
const fileAccessRequestMock = vi.hoisted(() => vi.fn())
const appControls = vi.hoisted(() => ({
  filesDisabled: false,
  voiceInputEnabled: true,
  taskControlsEnabled: true,
  isConversationResetPending: false,
  isMessageDeliveryPending: false,
  isSessionInteractionLocked: false,
}))
const chatInputProps = vi.hoisted(() => ({
  current: null as null | {
    files?: File[]
    filesDisabled?: boolean
    voiceInputEnabled?: boolean
    hideFileUpload?: boolean
    isLoading?: boolean
    currentInteractionRequestId?: string
    onFilesChange?: (files: File[]) => void
    onPause?: () => void
    onResume?: () => void
    onSend: (message: string, config?: unknown, files?: File[]) => Promise<void> | void
  },
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    state: appState,
    filesDisabled: appControls.filesDisabled,
    voiceInputEnabled: appControls.voiceInputEnabled,
    taskControlsEnabled: appControls.taskControlsEnabled,
    sendMessage: sendMessageMock,
    pauseTask: pauseTaskMock,
    resumeTask: resumeTaskMock,
    openFilePreview: openFilePreviewMock,
    closeFilePreview: vi.fn(),
    requestStatus: vi.fn(),
    dispatch: vi.fn(),
    getFileDownloadUrl: getFileDownloadUrlMock,
    isConversationResetPending: appControls.isConversationResetPending,
    isMessageDeliveryPending: appControls.isMessageDeliveryPending,
    isSessionInteractionLocked: appControls.isSessionInteractionLocked,
  }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

vi.mock("@/contexts/file-access-context", () => ({
  useFileAccess: () => ({ request: fileAccessRequestMock }),
}))

vi.mock("dagre", () => {
  class Graph {
    nodes = new Map<string, unknown>()
    edges: Array<{ source: string; target: string; data?: unknown }> = []

    setGraph() { }
    setDefaultEdgeLabel() { }
    setNode(id: string, data: unknown) {
      if (id === "throw-step") {
        throw new Error("bad node")
      }
      this.nodes.set(id, data)
    }
    setEdge(source: string, target: string, data?: unknown) {
      this.edges.push({ source, target, data })
    }
    node(id: string) {
      return this.nodes.get(id) || { x: 0, y: 0 }
    }
  }

  return {
    default: {
      graphlib: { Graph },
      layout: () => undefined,
    },
  }
})

vi.mock("@/components/chat/ChatMessage", () => ({
  ChatMessage: ({
    content,
    interactionsActive,
    traceEvents,
    taskStatus,
    processStatus,
    showEmptyStatus,
    showProcessView,
    onOpenExecutionPlan,
    contextBadges,
    taskRuntimeExtensionMetadata,
    interactionRequestId,
    interactions,
  }: {
    content?: string | null
    interactionsActive?: boolean
    traceEvents?: unknown[]
    taskStatus?: string
    processStatus?: string
    showEmptyStatus?: boolean
    showProcessView?: boolean
    onOpenExecutionPlan?: () => void
    contextBadges?: Array<{ kind: string; label: string; detail: string }>
    taskRuntimeExtensionMetadata?: {
      bindings: string[]
      publicMetadata: Record<string, Record<string, unknown>>
    }
    interactionRequestId?: string
    interactions?: unknown[]
  }) => (
    <div
      data-testid="chat-message"
      data-active={interactionsActive ? "true" : "false"}
      data-trace-count={traceEvents?.length ?? 0}
      data-task-status={taskStatus || ""}
      data-process-status={processStatus || ""}
      data-show-empty-status={showEmptyStatus ? "true" : "false"}
      data-show-process-view={showProcessView ? "true" : "false"}
      data-context-badges={JSON.stringify(contextBadges || [])}
      data-runtime-extension-metadata={JSON.stringify(taskRuntimeExtensionMetadata || {})}
      data-request-id={interactionRequestId || ""}
      data-interactions={JSON.stringify(interactions || [])}
    >
      {content}
      {onOpenExecutionPlan && traceEvents?.some((event) => {
        if (!event || typeof event !== "object" || !("event_type" in event)) return false
        return typeof event.event_type === "string" && (
          event.event_type === "dag_execution" ||
          event.event_type === "dag_execute_start" ||
          event.event_type === "dag_execute_end" ||
          event.event_type === "dag_plan_start" ||
          event.event_type === "dag_plan_end" ||
          event.event_type.startsWith("dag_step_")
        )
      }) && (
        <button type="button" title="chatPage.executionPlan.tooltip" onClick={onOpenExecutionPlan}>
          chatPage.executionPlan.view
        </button>
      )}
    </div>
  ),
}))

vi.mock("@/components/chat/ChatInput", () => ({
  ChatInput: (props: NonNullable<typeof chatInputProps.current>) => {
    chatInputProps.current = props
    return (
      <div data-testid="chat-input">
        <button
          type="button"
          disabled={props.isLoading}
          onClick={() => props.onFilesChange?.([
            new File(["draft"], "draft.txt", { type: "text/plain" }),
          ])}
        >
          stage file
        </button>
        <button
          type="button"
          disabled={props.isLoading}
          onClick={() => void props.onSend("send draft", { mode: "balanced" })}
        >
          send draft
        </button>
      </div>
    )
  },
}))

vi.mock("@/components/chat/TokenUsageDisplay", () => ({
  TokenUsageDisplay: () => null,
}))

vi.mock("@/components/file/task-file-manager", () => ({
  TaskFileManager: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="task-file-manager">{children}</div>
  ),
}))

vi.mock("@/components/file/file-preview-content", () => ({
  FilePreviewContent: () => <div data-testid="file-preview-content" />,
}))

vi.mock("@/components/file/file-preview-action-buttons", () => ({
  FilePreviewActionButtons: ({ onDownload }: { onDownload: () => void }) => (
    <button type="button" data-testid="file-preview-actions" onClick={onDownload}>
      download
    </button>
  ),
}))

vi.mock("@/components/preview-sheet", () => ({
  PreviewSheet: ({ children, actions }: { children: React.ReactNode; actions?: React.ReactNode }) => (
    <>{actions}{children}</>
  ),
}))

vi.mock("@/components/layout/center-panel", () => ({
  CenterPanel: ({
    dagNodes,
    dagEdges,
  }: {
    dagNodes?: unknown[]
    dagEdges?: unknown[]
  }) => (
    <div
      data-testid="center-panel"
      data-node-count={dagNodes?.length ?? 0}
      data-edge-count={dagEdges?.length ?? 0}
    />
  ),
}))

import { TaskConversationPanel, findWaitingPromptAndInteractions } from "./task-conversation-panel"

describe("TaskConversationPanel", () => {
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => undefined)
    openFilePreviewMock.mockReset()
    sendMessageMock.mockReset()
    sendMessageMock.mockResolvedValue(undefined)
    pauseTaskMock.mockReset()
    resumeTaskMock.mockReset()
    getFileDownloadUrlMock.mockReset()
    fileAccessRequestMock.mockReset()
    chatInputProps.current = null
    appControls.isConversationResetPending = false
    appControls.isMessageDeliveryPending = false
    appControls.isSessionInteractionLocked = false
    appControls.filesDisabled = false
    appControls.voiceInputEnabled = true
    appControls.taskControlsEnabled = true
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    appState.messages = []
    appState.traceEvents = []
    appState.currentTask = null
    appState.taskRuntimeExtensions = {}
    appState.taskId = 42
    appState.isProcessing = false
    appState.isHistoryLoading = false
    appState.filePreview = { isOpen: false, fileId: "", fileName: "", viewMode: "preview" }
  })

  it("marks user turns with the task's Local browser context", () => {
    appState.messages = [{
      id: "user-1",
      role: "user",
      content: "Inspect this window",
      timestamp: "2026-08-07T07:00:00Z",
    }]
    appState.taskRuntimeExtensions = {
      local_browser: { kind: "local_browser" },
    }

    render(<TaskConversationPanel mode="page" />)

    expect(screen.getByTestId("chat-message")).toHaveAttribute(
      "data-context-badges",
      JSON.stringify([{
        kind: "computer_use",
        label: "chatPage.input.localBrowser.chipLabel",
        detail: "chatPage.input.localBrowser.label",
      }]),
    )
  })

  it("passes bindings and public metadata through the message extension slot", () => {
    appState.messages = [{
      id: "user-1",
      role: "user",
      content: "Inspect my browser",
      timestamp: "2026-08-07T07:00:00Z",
    }]
    appState.currentTask = {
      id: "42",
      runtimeExtensionBindings: ["browser_relay"],
    }
    appState.taskRuntimeExtensions = {
      browser_relay: { kind: "browser_relay", connected: true },
    }
    render(<TaskConversationPanel mode="page" />)

    expect(screen.getByTestId("chat-message")).toHaveAttribute(
      "data-runtime-extension-metadata",
      JSON.stringify({
        bindings: ["browser_relay"],
        publicMetadata: {
          browser_relay: { kind: "browser_relay", connected: true },
        },
      }),
    )
  })

  it("restores the Computer use badge from the task's persisted binding", () => {
    appState.messages = [{
      id: "user-1",
      role: "user",
      content: "Inspect this window",
      timestamp: "2026-08-07T07:00:00Z",
    }]
    appState.currentTask = {
      id: "823",
      title: "Local browser task",
      description: "Inspect this window",
      status: "completed",
      createdAt: "2026-08-07T07:00:00Z",
      updatedAt: "2026-08-07T07:01:00Z",
      runtimeExtensionBindings: ["local_browser"],
    }
    appState.taskId = 823

    render(<TaskConversationPanel mode="page" />)

    expect(screen.getByTestId("chat-message")).toHaveAttribute(
      "data-context-badges",
      JSON.stringify([{
        kind: "computer_use",
        label: "chatPage.input.localBrowser.chipLabel",
        detail: "chatPage.input.localBrowser.label",
      }]),
    )
  })

  it("does not reuse the previous task's Computer use badge", () => {
    appState.messages = [{
      id: "user-1",
      role: "user",
      content: "Inspect this window",
      timestamp: "2026-08-07T07:00:00Z",
    }]
    appState.taskId = 824
    appState.currentTask = {
      id: "823",
      title: "Previous local browser task",
      description: "Inspect another window",
      status: "completed",
      createdAt: "2026-08-07T07:00:00Z",
      updatedAt: "2026-08-07T07:01:00Z",
      runtimeExtensionBindings: ["local_browser"],
    }

    render(<TaskConversationPanel mode="page" />)

    expect(screen.getByTestId("chat-message")).toHaveAttribute(
      "data-context-badges",
      JSON.stringify([]),
    )
    expect(screen.getByTestId("chat-message")).toHaveAttribute(
      "data-runtime-extension-metadata",
      JSON.stringify({ bindings: [], publicMetadata: {} }),
    )
  })

  it("disables every file surface and drops staged files when the capability is disabled", async () => {
    appState.filePreview = {
      isOpen: true,
      fileId: "secret-file",
      fileName: "secret.txt",
      viewMode: "preview",
    }
    const { rerender } = render(
      <TaskConversationPanel mode="page" />
    )

    fireEvent.click(screen.getByRole("button", { name: "stage file" }))
    expect(chatInputProps.current?.files).toHaveLength(1)

    appControls.filesDisabled = true
    rerender(<TaskConversationPanel mode="page" />)

    expect(chatInputProps.current?.hideFileUpload).toBe(true)
    expect(chatInputProps.current?.filesDisabled).toBe(true)
    expect(chatInputProps.current?.files).toEqual([])
    expect(screen.queryByTestId("task-file-manager")).not.toBeInTheDocument()
    expect(screen.queryByTestId("file-preview-content")).not.toBeInTheDocument()
    expect(screen.queryByTestId("file-preview-actions")).not.toBeInTheDocument()

    window.dispatchEvent(new CustomEvent("openFilePreview", {
      detail: { filePath: "file-secret", fileName: "secret.txt" },
    }))
    expect(openFilePreviewMock).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole("button", { name: "send draft" }))
    expect(sendMessageMock).toHaveBeenCalledWith(
      "send draft",
      { mode: "balanced" },
      [],
    )
  })

  it("preserves the existing file behavior when the capability is enabled", () => {
    render(<TaskConversationPanel mode="page" />)

    expect(chatInputProps.current?.hideFileUpload).toBe(false)
    expect(chatInputProps.current?.filesDisabled).toBe(false)
    expect(screen.getByTestId("task-file-manager")).toBeInTheDocument()

    window.dispatchEvent(new CustomEvent("openFilePreview", {
      detail: { filePath: "file-default", fileName: "default.txt" },
    }))
    expect(openFilePreviewMock).toHaveBeenCalledWith(
      "file-default",
      "default.txt",
    )
  })

  it("downloads through the provider-scoped file request policy", async () => {
    appState.filePreview = {
      isOpen: true,
      fileId: "public-file-id",
      fileName: "report.pdf",
      viewMode: "preview",
    }
    getFileDownloadUrlMock.mockReturnValue(
      "/api/files/public/download/public-file-id?token=guest-token",
    )
    fileAccessRequestMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(["file"]),
    })

    render(<TaskConversationPanel mode="page" />)
    fireEvent.click(screen.getByTestId("file-preview-actions"))

    await Promise.resolve()
    expect(getFileDownloadUrlMock).toHaveBeenCalledWith("public-file-id")
    expect(fileAccessRequestMock).toHaveBeenCalledWith(
      "/api/files/public/download/public-file-id?token=guest-token",
    )
  })

  it("passes the transport voice capability through to ChatInput", () => {
    const { rerender } = render(<TaskConversationPanel mode="page" />)

    expect(chatInputProps.current?.voiceInputEnabled).toBe(true)

    appControls.voiceInputEnabled = false
    rerender(<TaskConversationPanel mode="page" />)

    expect(chatInputProps.current?.voiceInputEnabled).toBe(false)
  })

  it("omits task controls from ChatInput when the transport disables them", () => {
    appControls.taskControlsEnabled = false

    render(<TaskConversationPanel mode="page" />)

    expect(chatInputProps.current?.onPause).toBeUndefined()
    expect(chatInputProps.current?.onResume).toBeUndefined()
  })

  it("preserves task control callbacks when the transport leaves them enabled", () => {
    render(<TaskConversationPanel mode="page" />)

    expect(chatInputProps.current?.onPause).toBe(pauseTaskMock)
    expect(chatInputProps.current?.onResume).toBe(resumeTaskMock)
  })

  it("does not offer a retained waiting request identity from a different task", () => {
    appState.taskId = 43
    appState.currentTask = {
      id: "42",
      title: "Previous waiting task",
      description: "Previous waiting task",
      status: "waiting_for_user",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
      waitingRequestId: "inputreq_stale",
    }

    render(<TaskConversationPanel mode="page" />)

    expect(chatInputProps.current?.currentInteractionRequestId).toBeUndefined()
  })

  it("keeps the composer busy during Session reset and durable delivery", () => {
    appControls.isConversationResetPending = true
    const { rerender } = render(
      <TaskConversationPanel mode="page" />
    )
    expect(chatInputProps.current?.isLoading).toBe(true)

    appControls.isConversationResetPending = false
    appControls.isMessageDeliveryPending = true
    rerender(<TaskConversationPanel mode="page" />)
    expect(chatInputProps.current?.isLoading).toBe(true)

    appControls.isMessageDeliveryPending = false
    rerender(<TaskConversationPanel mode="page" />)
    expect(chatInputProps.current?.isLoading).toBe(false)
  })

  it("blocks the established-conversation composer when the Session outcome requires reload", () => {
    appControls.isSessionInteractionLocked = true
    render(<TaskConversationPanel mode="page" />)

    expect(screen.getByRole("button", { name: "send draft" })).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "send draft" }))
    expect(sendMessageMock).not.toHaveBeenCalled()
  })

  it("renders waiting-for-user prompts from normal task state", () => {
    appState.messages = []
    appState.traceEvents = []
    appState.currentTask = {
      id: "42",
      title: "Preview",
      description: "Preview",
      status: "waiting_for_user",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
      waitingQuestion: "Which dataset should I use?",
      waitingInteractions: [
        {
          type: "select_one",
          field: "dataset",
          label: "Dataset",
          options: [{ label: "Sales", value: "sales" }],
        },
      ],
    } as any
    appState.isHistoryLoading = false

    render(<TaskConversationPanel mode="embedded-preview" />)

    expect(screen.getByText("Which dataset should I use?")).toBeInTheDocument()
    expect(screen.getByTestId("chat-message")).toHaveAttribute("data-active", "true")
  })

  it("keeps each rendered clarification bound to its own request id", () => {
    appState.messages = [
      {
        id: "q1",
        role: "assistant",
        content: "Which city?",
        timestamp: "1000",
        isResult: true,
        interactions: [{ type: "text_input", field: "city", label: "City" }],
        interactionRequestId: "inputreq_q1",
      },
      {
        id: "q2",
        role: "assistant",
        content: "Which hotel?",
        timestamp: "2000",
        isResult: true,
        interactions: [{ type: "text_input", field: "hotel", label: "Hotel" }],
        interactionRequestId: "inputreq_q2",
      },
    ]
    appState.traceEvents = []
    appState.currentTask = {
      id: "42",
      title: "Preview",
      description: "Preview",
      status: "waiting_for_user",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
      waitingQuestion: "Which hotel?",
      waitingRequestId: "inputreq_q2",
    } as any

    render(<TaskConversationPanel mode="embedded-preview" />)

    const rendered = screen.getAllByTestId("chat-message")
    expect(rendered[0]).toHaveAttribute("data-request-id", "inputreq_q1")
    expect(rendered[1]).toHaveAttribute("data-request-id", "inputreq_q2")
  })

  it("keeps an identified text-only wait separate from stale structured trace interactions", () => {
    appState.messages = [{
      id: "user-r2",
      role: "user",
      content: "Start the next question",
      timestamp: 2000,
    }]
    appState.traceEvents = [{
      event_id: "stale-r1",
      event_type: "agent_message",
      timestamp: 1000,
      data: {
        message: "Choose the old city",
        expect_response: true,
        metadata: {
          interactions: [{
            type: "select_one",
            field: "city",
            label: "City",
            options: [{ label: "Paris", value: "paris" }],
          }],
        },
      },
    }]
    appState.currentTask = {
      id: "42",
      title: "Preview",
      description: "Preview",
      status: "waiting_for_user",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
      waitingQuestion: "Type the current answer",
      waitingRequestId: "inputreq_r2",
    }

    render(<TaskConversationPanel mode="embedded-preview" />)

    const currentWait = screen.getAllByTestId("chat-message").find(
      (message) => message.getAttribute("data-request-id") === "inputreq_r2",
    )
    expect(currentWait).toHaveAttribute("data-interactions", "[]")
    expect(chatInputProps.current?.currentInteractionRequestId).toBe("inputreq_r2")
  })

  it("keeps the historical trace fallback for id-less waiting state", () => {
    appState.messages = [{
      id: "user-legacy",
      role: "user",
      content: "Start the legacy question",
      timestamp: 2000,
    }]
    appState.traceEvents = [{
      event_id: "legacy-wait",
      event_type: "agent_message",
      timestamp: 1000,
      data: {
        message: "Choose a city",
        expect_response: true,
        metadata: {
          interactions: [{ type: "text_input", field: "city", label: "City" }],
        },
      },
    }]
    appState.currentTask = {
      id: "42",
      title: "Legacy preview",
      description: "Legacy preview",
      status: "waiting_for_user",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
      waitingQuestion: "Choose a city",
    }

    render(<TaskConversationPanel mode="embedded-preview" />)

    const activeWait = screen.getAllByTestId("chat-message").find(
      (message) => message.getAttribute("data-active") === "true",
    )
    expect(activeWait).toHaveAttribute(
      "data-interactions",
      JSON.stringify([{ type: "text_input", field: "city", label: "City" }]),
    )
    expect(chatInputProps.current?.currentInteractionRequestId).toBeUndefined()
  })

  it("shows history loading before waiting-for-user content while history is loading", () => {
    appState.messages = []
    appState.traceEvents = []
    appState.currentTask = {
      id: "42",
      title: "Task",
      description: "Task",
      status: "waiting_for_user",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
      waitingQuestion: "Which dataset should I use?",
    } as any
    appState.isHistoryLoading = true

    render(<TaskConversationPanel mode="page" />)

    expect(screen.getByText("common.loading")).toBeInTheDocument()
    expect(screen.queryByText("Which dataset should I use?")).not.toBeInTheDocument()
  })

  it("does not surface ordinary agent messages as waiting prompts", () => {
    appState.messages = []
    appState.traceEvents = [
      {
        event_id: "agent-1",
        event_type: "agent_message",
        timestamp: "1000",
        data: {
          message: "Hello! What can I help you with?",
          message_type: "question",
          expect_response: false,
        },
      },
    ] as any
    appState.currentTask = {
      id: "42",
      title: "Task",
      description: "Task",
      status: "waiting_for_user",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
    } as any

    render(<TaskConversationPanel mode="embedded-preview" />)

    expect(screen.queryByText("Hello! What can I help you with?")).not.toBeInTheDocument()
  })

  it("hides raw Workforce Agent tool calls and delegated child traces", () => {
    appState.messages = []
    appState.traceEvents = [
      {
        event_id: "raw-agent-tool",
        event_type: "tool_execution_start",
        timestamp: 1000,
        data: {
          tool_name: "worker_editor_agent__a20",
          tool_params: { task: "Delegated task instructions" },
        },
      },
      {
        event_id: "raw-semantic-agent-tool",
        event_type: "tool_execution_start",
        timestamp: 1000.5,
        data: {
          tool_name: "agent_reviewer_agent__a7",
          tool_params: { task: "Review the delegated work" },
        },
      },
      {
        event_id: "raw-legacy-agent-tool",
        event_type: "tool_execution_start",
        timestamp: 1000.75,
        data: {
          tool_name: "call_agent_7",
          tool_params: { task: "Review the delegated work" },
        },
      },
      {
        event_id: "delegation-start",
        event_type: "workforce_delegation_start",
        timestamp: 1001,
        data: {
          worker_task_id: "agent_20_run",
          worker_alias: "Editor Agent",
          tool_name: "worker_editor_agent__a20",
        },
      },
      {
        event_id: "child-tool",
        event_type: "tool_execution_start",
        timestamp: 1002,
        data: {
          source: "xagent-agent-tool-child",
          worker_task_id: "agent_20_run",
          tool_name: "transcribe_audio",
        },
      },
      {
        event_id: "manager-progress",
        event_type: "agent_progress",
        timestamp: 1003,
        data: { message: "Editor is working" },
      },
    ] as any
    appState.currentTask = {
      id: "42",
      title: "Workforce run",
      description: "Workforce run",
      status: "running",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
    } as any

    render(
      <TaskConversationPanel
        mode="embedded-preview"
        onAgentExecutionClick={vi.fn()}
      />
    )

    const processMessage = screen.getByTestId("chat-message")
    expect(processMessage).toHaveAttribute("data-trace-count", "2")
  })

  it("does not leave a closed historical turn running", () => {
    appState.messages = [
      { id: "turn-1", role: "user", content: "First turn", timestamp: 1000 },
      { id: "turn-2", role: "user", content: "Continue", timestamp: 3000 },
    ] as any
    appState.traceEvents = [
      {
        event_id: "unfinished-tool",
        event_type: "tool_execution_start",
        timestamp: 2000,
        data: { tool_name: "long_running_tool" },
      },
    ] as any
    appState.currentTask = {
      id: "42",
      title: "Task",
      description: "Task",
      status: "running",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
    } as any

    render(<TaskConversationPanel mode="embedded-preview" />)

    const processMessage = screen.getAllByTestId("chat-message").find(
      (element) => element.getAttribute("data-trace-count") === "1",
    )
    expect(processMessage).toHaveAttribute("data-process-status", "completed")
  })

  it("renders trace process events as separate timeline items between messages", () => {
    appState.messages = [
      {
        id: "msg-user",
        role: "user",
        content: "Run analysis",
        timestamp: "1000",
      },
      {
        id: "msg-result",
        role: "assistant",
        content: "Done",
        timestamp: "3000",
        isResult: true,
      },
    ] as any
    appState.traceEvents = [
      {
        event_id: "trace-1",
        event_type: "tool_call",
        timestamp: 2000,
        data: { message: "Using tool" },
      },
    ] as any
    appState.currentTask = {
      id: "42",
      title: "Task",
      description: "Task",
      status: "completed",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
    } as any
    appState.isHistoryLoading = false

    render(<TaskConversationPanel mode="page" />)

    const renderedMessages = screen.getAllByTestId("chat-message")
    expect(renderedMessages).toHaveLength(3)
    expect(renderedMessages[0]).toHaveTextContent("Run analysis")
    expect(renderedMessages[1]).toHaveAttribute("data-trace-count", "1")
    expect(renderedMessages[1]).toHaveAttribute("data-process-status", "completed")
    expect(renderedMessages[2]).toHaveTextContent("Done")
  })

  it("renders a terminal failure reason instead of a virtual unknown-error placeholder", () => {
    // A quota-gate refusal produces a failed result bubble and no trace events.
    // The bubble must render as the turn's outcome; without it the panel would
    // fall back to a virtual failed message that shows "unknown error".
    const quotaReason = "Team quota exhausted for this billing period."
    appState.messages = [
      {
        id: "msg-user",
        role: "user",
        content: "Run analysis",
        timestamp: "1000",
      },
      {
        id: "msg-task-failed",
        role: "assistant",
        content: quotaReason,
        timestamp: "2000",
        status: "failed",
        isResult: true,
      },
    ] as any
    appState.traceEvents = []
    appState.currentTask = {
      id: "42",
      title: "Task",
      description: "Task",
      status: "failed",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
    } as any
    appState.isHistoryLoading = false

    render(<TaskConversationPanel mode="page" />)

    const renderedMessages = screen.getAllByTestId("chat-message")
    // Only the user turn and the failure reason — no extra virtual message.
    expect(renderedMessages).toHaveLength(2)
    expect(renderedMessages[1]).toHaveTextContent(quotaReason)
    // Trace visible: the reason renders as plain content, not the failed path.
    expect(renderedMessages[1]).toHaveAttribute("data-task-status", "")
    cleanup()

    // With the trace hidden the same verbatim reason must reach ChatMessage
    // flagged as failed, so its raw text gets the generic replacement there.
    render(<TaskConversationPanel mode="page" showProcessView={false} />)
    const hiddenTraceMessages = screen.getAllByTestId("chat-message")
    expect(hiddenTraceMessages[1]).toHaveAttribute("data-task-status", "failed")
  })

  it("applies current task status only to the latest trace process group", () => {
    appState.messages = [
      {
        id: "msg-user-1",
        role: "user",
        content: "First turn",
        timestamp: "1000",
      },
      {
        id: "msg-result-1",
        role: "assistant",
        content: "First result",
        timestamp: "3000",
        isResult: true,
      },
      {
        id: "msg-user-2",
        role: "user",
        content: "Second turn",
        timestamp: "4000",
      },
    ] as any
    appState.traceEvents = [
      {
        event_id: "trace-old",
        event_type: "tool_call",
        timestamp: 2000,
        data: { message: "Old turn work" },
      },
      {
        event_id: "trace-latest",
        event_type: "tool_call",
        timestamp: 5000,
        data: { message: "Latest turn work" },
      },
    ] as any
    appState.currentTask = {
      id: "42",
      title: "Task",
      description: "Task",
      status: "completed",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
    } as any

    render(<TaskConversationPanel mode="page" />)

    const processMessages = screen
      .getAllByTestId("chat-message")
      .filter((message) => message.getAttribute("data-trace-count") === "1")
    expect(processMessages).toHaveLength(2)
    expect(processMessages[0]).toHaveAttribute("data-process-status", "completed")
    expect(processMessages[1]).toHaveAttribute("data-process-status", "completed")
  })

  it("keeps late react_task_end events in the same thinking group after the assistant result", () => {
    appState.messages = [
      {
        id: "msg-user",
        role: "user",
        content: "Hello",
        timestamp: "1000",
      },
      {
        id: "msg-result",
        role: "assistant",
        content: "Hi there",
        timestamp: "3000",
        isResult: true,
        status: "completed",
      },
    ] as any
    appState.traceEvents = [
      {
        event_id: "react-start",
        event_type: "react_task_start",
        step_id: "step-1",
        timestamp: 2000,
        data: {},
      },
      {
        event_id: "react-end",
        event_type: "react_task_end",
        step_id: "step-1",
        timestamp: 4000,
        data: {},
      },
    ] as any
    appState.currentTask = {
      id: "42",
      title: "Preview",
      description: "Preview",
      status: "running",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
    } as any

    render(<TaskConversationPanel mode="embedded-preview" />)

    const processMessages = screen
      .getAllByTestId("chat-message")
      .filter((message) => message.getAttribute("data-trace-count") !== "0")

    expect(processMessages).toHaveLength(1)
    expect(processMessages[0]).toHaveAttribute("data-trace-count", "2")
    expect(processMessages[0]).toHaveAttribute("data-process-status", "completed")
    expect(processMessages[0]).toHaveAttribute("data-show-empty-status", "false")
  })

  it("does not apply current task status to a previous turn when the new turn has no trace yet", () => {
    appState.messages = [
      {
        id: "msg-user-1",
        role: "user",
        content: "First turn",
        timestamp: "1000",
      },
      {
        id: "msg-result-1",
        role: "assistant",
        content: "First result",
        timestamp: "3000",
        isResult: true,
      },
      {
        id: "msg-user-2",
        role: "user",
        content: "Second turn",
        timestamp: "4000",
      },
    ] as any
    appState.traceEvents = [
      {
        event_id: "trace-old",
        event_type: "tool_call",
        timestamp: 2000,
        data: { message: "Old turn work" },
      },
    ] as any
    appState.currentTask = {
      id: "42",
      title: "Task",
      description: "Task",
      status: "running",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
    } as any
    appState.isProcessing = true

    render(<TaskConversationPanel mode="page" />)

    const historicalProcessMessages = screen
      .getAllByTestId("chat-message")
      .filter((message) => message.getAttribute("data-trace-count") === "1")
    expect(historicalProcessMessages).toHaveLength(1)
    expect(historicalProcessMessages[0]).toHaveAttribute("data-process-status", "completed")

    const virtualProcessMessages = screen
      .getAllByTestId("chat-message")
      .filter((message) => message.getAttribute("data-trace-count") === "0")
    expect(
      virtualProcessMessages.some(
        (message) => message.getAttribute("data-process-status") === "running"
      )
    ).toBe(true)
  })

  it("normalizes invalid timestamps to zero for deterministic ordering", () => {
    appState.messages = [
      {
        id: "msg-valid",
        role: "user",
        content: "Valid timestamp",
        timestamp: "1000",
      },
      {
        id: "msg-invalid",
        role: "assistant",
        content: "Invalid timestamp",
        timestamp: {},
        isResult: true,
      },
    ] as any
    appState.traceEvents = []
    appState.currentTask = {
      id: "42",
      title: "Task",
      description: "Task",
      status: "completed",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
    } as any

    render(<TaskConversationPanel mode="page" />)

    const renderedMessages = screen.getAllByTestId("chat-message")
    expect(renderedMessages[0]).toHaveTextContent("Invalid timestamp")
    expect(renderedMessages[1]).toHaveTextContent("Valid timestamp")
  })

  it("keeps a process event with a missing timestamp after the user message", () => {
    appState.messages = [
      {
        id: "msg-user",
        role: "user",
        content: "Ask something",
        timestamp: "1000",
      },
      {
        id: "msg-result",
        role: "assistant",
        content: "Answer",
        timestamp: "3000",
        isResult: true,
      },
    ] as any
    appState.traceEvents = [
      {
        event_id: "trace-no-ts",
        event_type: "tool_call",
        // No timestamp: must not drag the process group above the user message.
        data: { message: "Untimed work" },
      },
    ] as any
    appState.currentTask = {
      id: "42",
      title: "Task",
      description: "Task",
      status: "completed",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
    } as any
    appState.isHistoryLoading = false

    render(<TaskConversationPanel mode="page" />)

    const renderedMessages = screen.getAllByTestId("chat-message")
    expect(renderedMessages).toHaveLength(3)
    expect(renderedMessages[0]).toHaveTextContent("Ask something")
    expect(renderedMessages[1]).toHaveAttribute("data-trace-count", "1")
    expect(renderedMessages[2]).toHaveTextContent("Answer")
  })

  it("renders a system notice as an inline line between messages", () => {
    appState.messages = [
      {
        id: "msg-user",
        role: "user",
        content: "Long task",
        timestamp: "1000",
      },
      {
        id: "compact-notice",
        role: "assistant",
        content: "Context compacted (56860→449 tokens)",
        timestamp: "2000",
        isSystemNotice: true,
      },
      {
        id: "msg-result",
        role: "assistant",
        content: "Answer",
        timestamp: "3000",
        isResult: true,
      },
    ] as any
    appState.traceEvents = []
    appState.currentTask = {
      id: "42",
      title: "Task",
      description: "Task",
      status: "completed",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
    } as any
    appState.isHistoryLoading = false

    render(<TaskConversationPanel mode="page" />)

    // The notice is a plain inline line, not a ChatMessage bubble.
    const renderedMessages = screen.getAllByTestId("chat-message")
    expect(renderedMessages).toHaveLength(2)
    expect(screen.getByText("Context compacted (56860→449 tokens)")).toBeInTheDocument()
  })

  it("scopes the execution plan action to the DAG turn instead of task actions", () => {
    appState.messages = [
      {
        id: "user-react",
        role: "user",
        content: "Use tools",
        timestamp: 1000,
      },
      {
        id: "result-react",
        role: "assistant",
        content: "Done",
        timestamp: 1500,
        isResult: true,
      },
      {
        id: "user-dag",
        role: "user",
        content: "Create a plan",
        timestamp: 2000,
      },
    ] as any
    appState.traceEvents = [
      {
        event_id: "react-start",
        event_type: "react_task_start",
        step_id: "react-step",
        timestamp: 1100,
        data: {},
      },
      {
        event_id: "dag-execution",
        event_type: "dag_execution",
        timestamp: 2100,
        data: {
          phase: "executing",
          steps: [{ id: "dag-step", task: "Create output", dependencies: [] }],
        },
      },
    ] as any
    appState.currentTask = {
      id: "42",
      title: "Task",
      description: "Task",
      status: "completed",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
      isDag: true,
    } as any

    render(<TaskConversationPanel mode="page" />)

    expect(screen.getAllByTitle("chatPage.executionPlan.tooltip")).toHaveLength(1)
  })

  it("ignores malformed DAG layout failures without throwing", async () => {
    appState.messages = []
    appState.traceEvents = [
      {
        event_id: "dag-execution",
        event_type: "dag_execution",
        timestamp: 1000,
        data: { phase: "executing", steps: [] },
      },
    ] as any
    appState.currentTask = {
      id: "42",
      title: "Task",
      description: "Task",
      status: "running",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
      isDag: true,
    } as any
    appState.steps = [
      {
        id: "throw-step",
        name: "Throwing step",
        status: "pending",
        dependencies: [],
      },
      {
        id: "valid-step",
        name: "Valid step",
        status: "running",
        dependencies: [null, "", "missing-step"],
      },
      {
        id: "",
        name: "Malformed step",
        status: "pending",
        dependencies: ["valid-step"],
      },
    ] as any
    appState.filePreview = { isOpen: false } as any

    expect(() => render(<TaskConversationPanel mode="page" />)).not.toThrow()
    fireEvent.click(screen.getByTitle("chatPage.executionPlan.tooltip"))

    expect(await screen.findByTestId("center-panel")).toHaveAttribute("data-node-count", "3")
    expect(screen.getByTestId("center-panel")).toHaveAttribute("data-edge-count", "0")
  })

  it("shows the process view by default and hides it when asked to", () => {
    const runningTask = {
      id: "42",
      title: "Task",
      description: "Task",
      status: "running",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
    } as any
    appState.messages = []
    appState.traceEvents = [
      {
        event_id: "tool-1",
        event_type: "tool_call",
        timestamp: 1000,
        data: { tool_name: "web_search", args: { query: "secret" } },
      },
    ] as any
    appState.currentTask = runningTask
    appState.isProcessing = true

    const { unmount } = render(<TaskConversationPanel mode="page" />)
    for (const message of screen.getAllByTestId("chat-message")) {
      expect(message).toHaveAttribute("data-show-process-view", "true")
    }
    unmount()

    render(<TaskConversationPanel mode="page" showProcessView={false} />)
    for (const message of screen.getAllByTestId("chat-message")) {
      expect(message).toHaveAttribute("data-show-process-view", "false")
    }
    cleanup()

    // Unlike the mode-derived sibling flags, showProcessView defaults to true
    // for every mode; only the widget opts out explicitly.
    render(<TaskConversationPanel mode="embedded-preview" />)
    for (const message of screen.getAllByTestId("chat-message")) {
      expect(message).toHaveAttribute("data-show-process-view", "true")
    }
  })
})

describe("findWaitingPromptAndInteractions", () => {
  const waitingTask = { status: "waiting_for_user" } as any

  it("returns null/undefined when the task is not waiting_for_user", () => {
    expect(findWaitingPromptAndInteractions({ status: "running" } as any, [])).toEqual({
      message: null,
      interactions: undefined,
    })
  })

  it("prefers currentTask's own waitingQuestion/waitingInteractions fields when set", () => {
    const task = {
      status: "waiting_for_user",
      waitingQuestion: "Which dataset?",
      waitingInteractions: [{ type: "select_one", field: "dataset" }],
    } as any
    expect(findWaitingPromptAndInteractions(task, [])).toEqual({
      message: "Which dataset?",
      interactions: task.waitingInteractions,
    })
  })

  it("falls back to a trace-scanned message when only waitingInteractions is set on currentTask, instead of losing the prompt text", () => {
    const task = {
      status: "waiting_for_user",
      waitingInteractions: [{ type: "connect_apps", field: "connect_apps", apps: ["Gmail"] }],
    } as any
    const traceEvents = [
      {
        event_type: "agent_message",
        data: { expect_response: true, message: "I need access to Gmail to continue." },
      },
    ]

    expect(findWaitingPromptAndInteractions(task, traceEvents)).toEqual({
      message: "I need access to Gmail to continue.",
      interactions: task.waitingInteractions,
    })
  })

  it("falls back to trace-scanned interactions when only waitingQuestion is set on currentTask, instead of losing the widget", () => {
    const task = {
      status: "waiting_for_user",
      waitingQuestion: "Which dataset?",
      waitingInteractions: [],
    } as any
    const traceEvents = [
      {
        event_type: "agent_message",
        data: {
          expect_response: true,
          metadata: { interactions: [{ type: "select_one", field: "dataset" }] },
        },
      },
    ]

    expect(findWaitingPromptAndInteractions(task, traceEvents)).toEqual({
      message: "Which dataset?",
      interactions: [{ type: "select_one", field: "dataset" }],
    })
  })

  it("pairs the message and interactions from the SAME trace event, not two independently-found events", () => {
    const traceEvents = [
      {
        event_type: "agent_message",
        data: {
          expect_response: true,
          message: "I need access to Gmail to continue.",
          metadata: {
            interactions: [{ type: "connect_apps", field: "connect_apps", apps: ["Gmail"] }],
          },
        },
      },
      {
        event_type: "agent_message",
        data: { expect_response: true, message: "Which dataset should I use?" },
      },
    ]

    // The most recent qualifying event (the plain question) has no
    // interactions of its own - it must not inherit the older connect_apps
    // event's interactions just because that one has some.
    expect(findWaitingPromptAndInteractions(waitingTask, traceEvents)).toEqual({
      message: "Which dataset should I use?",
      interactions: undefined,
    })
  })

  it("takes interactions from the most recent event even without a message on it", () => {
    const traceEvents = [
      {
        event_type: "agent_message",
        data: { expect_response: true, message: "An older question." },
      },
      {
        event_type: "agent_message",
        data: {
          expect_response: true,
          metadata: {
            interactions: [{ type: "connect_apps", field: "connect_apps", apps: ["Gmail"] }],
          },
        },
      },
    ]

    expect(findWaitingPromptAndInteractions(waitingTask, traceEvents)).toEqual({
      message: null,
      interactions: [{ type: "connect_apps", field: "connect_apps", apps: ["Gmail"] }],
    })
  })
})

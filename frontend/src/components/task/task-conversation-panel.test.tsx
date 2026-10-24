import React from "react"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { ComposerAttachmentStatus } from "@/components/chat/ChatInput"
import type { SessionStopState } from "@/contexts/app-context-chat"

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
  stoppedTurnAnchorIds: [] as string[],
  filePreview: { isOpen: false, fileId: "", fileName: "", viewMode: "preview" },
  dagExecution: null,
  steps: [],
}))
const realChatMessage = vi.hoisted(() => ({ enabled: false }))
const openFilePreviewMock = vi.hoisted(() => vi.fn())
const sendMessageMock = vi.hoisted(() => vi.fn())
const pauseTaskMock = vi.hoisted(() => vi.fn())
const resumeTaskMock = vi.hoisted(() => vi.fn())
const stopTaskMock = vi.hoisted(() => vi.fn())
const getFileDownloadUrlMock = vi.hoisted(() => vi.fn())
const fileAccessRequestMock = vi.hoisted(() => vi.fn())
const appControls = vi.hoisted(() => ({
  filesDisabled: false,
  voiceInputEnabled: true,
  taskControlsEnabled: true,
  canStopTask: false,
  stopState: "idle" as SessionStopState,
  isConversationResetPending: false,
  isMessageDeliveryPending: false,
  isSessionInteractionLocked: false,
  connectionNotice: null as "reconnecting" | "lost" | null,
}))
const chatInputProps = vi.hoisted(() => ({
  current: null as null | {
    files?: File[]
    filesDisabled?: boolean
    voiceInputEnabled?: boolean
    hideFileUpload?: boolean
    isLoading?: boolean
    currentInteractionRequestId?: string
    attachments?: { accept: string; maxFiles: number; maxFileBytes: number }
    attachmentStatuses?: ReadonlyMap<File, string>
    onRetryAttachment?: (file: File) => void
    onFilesChange?: (files: File[]) => void
    onPause?: () => void
    onResume?: () => void
    onStop?: () => void
    stopState?: string
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
    canStopTask: appControls.canStopTask,
    stopTask: stopTaskMock,
    stopState: appControls.stopState,
    openFilePreview: openFilePreviewMock,
    closeFilePreview: vi.fn(),
    requestStatus: vi.fn(),
    dispatch: vi.fn(),
    getFileDownloadUrl: getFileDownloadUrlMock,
    isConversationResetPending: appControls.isConversationResetPending,
    isMessageDeliveryPending: appControls.isMessageDeliveryPending,
    isSessionInteractionLocked: appControls.isSessionInteractionLocked,
    connectionNotice: appControls.connectionNotice,
  }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key, tDynamic: (key: string) => key }),
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

vi.mock("@/components/chat/ChatMessage", async () => {
  const actual = await vi.importActual<typeof import("@/components/chat/ChatMessage")>(
    "@/components/chat/ChatMessage",
  )
  const MockChatMessage = ({
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
    failureTaskId,
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
    failureTaskId?: number | null
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
      data-failure-task-id={failureTaskId ?? ""}
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
  )
  // Tests that assert on the real trace rendering opt in through realChatMessage.
  return {
    ...actual,
    ChatMessage: (props: React.ComponentProps<typeof actual.ChatMessage>) =>
      realChatMessage.enabled ? <actual.ChatMessage {...props} /> : <MockChatMessage {...(props as React.ComponentProps<typeof MockChatMessage>)} />,
  }
})

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }))
vi.mock("@/lib/api-wrapper", () => ({ apiRequest: vi.fn() }))
vi.mock("@/components/file/docx-preview-renderer", () => ({ DocxPreviewRenderer: () => null }))
vi.mock("@/components/file/excel-preview-renderer", () => ({ ExcelPreviewRenderer: () => null }))
vi.mock("@/components/file/pptx-preview-renderer", () => ({ PptxPreviewRenderer: () => null }))

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

import { TaskConversationPanel } from "./task-conversation-panel"

describe("TaskConversationPanel", () => {
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => undefined)
    openFilePreviewMock.mockReset()
    sendMessageMock.mockReset()
    sendMessageMock.mockResolvedValue(undefined)
    pauseTaskMock.mockReset()
    resumeTaskMock.mockReset()
    stopTaskMock.mockReset()
    getFileDownloadUrlMock.mockReset()
    fileAccessRequestMock.mockReset()
    chatInputProps.current = null
    appControls.isConversationResetPending = false
    appControls.isMessageDeliveryPending = false
    appControls.isSessionInteractionLocked = false
    appControls.connectionNotice = null
    appControls.filesDisabled = false
    appControls.voiceInputEnabled = true
    appControls.taskControlsEnabled = true
    appControls.canStopTask = false
    appControls.stopState = "idle"
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    realChatMessage.enabled = false
    appState.messages = []
    appState.traceEvents = []
    appState.stoppedTurnAnchorIds = []
    appState.currentTask = null
    appState.taskRuntimeExtensions = {}
    appState.taskId = 42
    appState.isProcessing = false
    appState.isHistoryLoading = false
    appState.filePreview = { isOpen: false, fileId: "", fileName: "", viewMode: "preview" }
  })

  it.each(["completed", "partial", "blocked"])("shows the current settled outcome %s, including on history load", (completionOutcome) => {
    appState.currentTask = { id: "42", status: "completed", completionOutcome }
    const { rerender } = render(<TaskConversationPanel mode="page" />)
    expect(screen.getByText(`taskCompletion.${completionOutcome}`)).toBeInTheDocument()
    appState.currentTask = { ...appState.currentTask, status: "running" }
    rerender(<TaskConversationPanel mode="page" />)
    expect(screen.queryByText(`taskCompletion.${completionOutcome}`)).not.toBeInTheDocument()
  })

  it.each([
    ["lease_expired", "taskInterruption.leaseExpired"],
    ["persistence_failure", "taskInterruption.persistenceFailure"],
    ["llm_unavailable", "taskInterruption.llmUnavailable"],
    ["shutdown", "taskInterruption.paused"],
  ])("explains a %s pause and drops the notice once the task leaves paused", (reason, key) => {
    appState.currentTask = { id: "42", status: "paused", autoRecovery: { reason } }
    const { rerender } = render(<TaskConversationPanel mode="page" />)
    expect(screen.getByTestId("task-interruption-notice")).toHaveTextContent(key)
    appState.currentTask = { ...appState.currentTask, status: "running" }
    rerender(<TaskConversationPanel mode="page" />)
    expect(screen.queryByTestId("task-interruption-notice")).not.toBeInTheDocument()
  })

  it.each([
    [{ status: "scheduled" }, "taskInterruption.autoResumeScheduled"],
    [{ status: "scheduled", nextAttemptAt: "2026-10-10T08:00:12+00:00" }, "taskInterruption.autoResumeScheduledAt"],
    [{ status: "resuming" }, "taskInterruption.autoResumeResuming"],
    [{ status: "stopped", stopReason: "limit_reached" }, "taskInterruption.autoResumeStoppedLimit"],
    [{ status: "stopped", stopReason: "disabled" }, "taskInterruption.autoResumeStoppedDisabled"],
    [{ status: "stopped", stopReason: "expired" }, "taskInterruption.autoResumeStoppedExpired"],
    [{ status: "stopped", stopReason: "superseded" }, "taskInterruption.autoResumeStoppedSuperseded"],
    [{ status: "stopped", stopReason: "failed" }, "taskInterruption.autoResumeStoppedFailed"],
    // A time that does not parse falls back to the plain scheduled copy.
    [{ status: "scheduled", nextAttemptAt: "not a time" }, "taskInterruption.autoResumeScheduled"],
    [{ status: "stopped", stopReason: "ineligible" }, "taskInterruption.autoResumeStopped"],
  ])("says what automatic resume is doing (%o)", (autoResume, key) => {
    appState.currentTask = {
      id: "42",
      status: "paused",
      autoRecovery: { reason: "lease_expired" },
      autoResume: { runId: "run-1", stateVersion: 4, ...autoResume },
    }
    const { rerender } = render(<TaskConversationPanel mode="page" />)
    expect(screen.getByTestId("task-interruption-notice")).toHaveTextContent(key)
    appState.currentTask = { ...appState.currentTask, status: "running" }
    rerender(<TaskConversationPanel mode="page" />)
    expect(screen.queryByTestId("task-interruption-notice")).not.toBeInTheDocument()
  })

  it("does not show the previous task's automatic-resume notice after a switch", () => {
    appState.currentTask = {
      id: "41",
      status: "paused",
      autoResume: { status: "scheduled", runId: "run-1", stateVersion: 4 },
    }
    appState.taskId = 42
    render(<TaskConversationPanel mode="page" />)
    expect(screen.queryByTestId("task-interruption-notice")).not.toBeInTheDocument()
  })

  it("does not show the previous task's interruption notice after a switch", () => {
    appState.currentTask = { id: "41", status: "paused", autoRecovery: { reason: "lease_expired" } }
    appState.taskId = 42
    render(<TaskConversationPanel mode="page" />)
    expect(screen.queryByTestId("task-interruption-notice")).not.toBeInTheDocument()
  })

  it("shows no interruption notice for the user's own pause or an unexplained one", () => {
    appState.currentTask = { id: "42", status: "paused", autoRecovery: { reason: "user_pause" } }
    const { rerender } = render(<TaskConversationPanel mode="page" />)
    expect(screen.queryByTestId("task-interruption-notice")).not.toBeInTheDocument()
    appState.currentTask = { id: "42", status: "paused" }
    rerender(<TaskConversationPanel mode="page" />)
    expect(screen.queryByTestId("task-interruption-notice")).not.toBeInTheDocument()
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

  it("keeps pause and resume closed while task stop is open", () => {
    appControls.canStopTask = true
    appControls.taskControlsEnabled = false

    render(<TaskConversationPanel mode="page" />)

    expect(chatInputProps.current?.onPause).toBeUndefined()
    expect(chatInputProps.current?.onResume).toBeUndefined()
    expect(chatInputProps.current?.onStop).toBe(stopTaskMock)
  })

  it("passes the stop callback when task stop is available", () => {
    appControls.canStopTask = true

    const { rerender } = render(<TaskConversationPanel mode="page" />)

    expect(chatInputProps.current?.onStop).toBe(stopTaskMock)

    appControls.canStopTask = false
    rerender(<TaskConversationPanel mode="page" />)

    expect(chatInputProps.current?.onStop).toBeUndefined()
  })

  it("forwards the stop state to the composer", () => {
    appControls.canStopTask = true
    appControls.stopState = "stopping"

    const { rerender } = render(<TaskConversationPanel mode="page" />)

    expect(chatInputProps.current?.stopState).toBe("stopping")

    appControls.stopState = "timed_out"
    rerender(<TaskConversationPanel mode="page" />)

    expect(chatInputProps.current?.stopState).toBe("timed_out")
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

  it("shows a status banner only once the connection is lost", () => {
    realChatMessage.enabled = true
    appState.messages = [{ id: "u1", role: "user", content: "Hello", timestamp: "1000" }]
    appState.isProcessing = true
    appControls.connectionNotice = "reconnecting"
    const { rerender } = render(<TaskConversationPanel mode="page" />)
    expect(screen.queryByText("clientErrors.connectionLost")).not.toBeInTheDocument()
    expect(screen.getByText("clientErrors.connectionReconnecting")).toBeInTheDocument()

    appControls.connectionNotice = "lost"
    rerender(<TaskConversationPanel mode="page" />)
    expect(screen.getAllByText("clientErrors.connectionLost")).toHaveLength(1)
    expect(screen.getByRole("status")).toHaveTextContent("clientErrors.connectionLost")
    expect(screen.getByText("clientErrors.connectionLostStatus")).toBeInTheDocument()
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

  it("passes the task id only to failed assistant messages", () => {
    appState.messages = [
      { id: "msg-user", role: "user", content: "Run analysis", timestamp: "1000" },
      {
        id: "msg-failed",
        role: "assistant",
        content: "Failure reason",
        timestamp: "2000",
        status: "failed",
        isResult: true,
      },
      { id: "msg-user-2", role: "user", content: "Try again", timestamp: "3000" },
      {
        id: "msg-done",
        role: "assistant",
        content: "Answer",
        timestamp: "4000",
        status: "completed",
        isResult: true,
      },
    ] as any
    appState.traceEvents = []
    appState.currentTask = { id: "42", title: "Task", status: "completed" } as any

    render(<TaskConversationPanel mode="page" />)

    const renderedMessages = screen.getAllByTestId("chat-message")
    expect(renderedMessages).toHaveLength(4)
    expect(renderedMessages.map((message) => message.getAttribute("data-failure-task-id")))
      .toEqual(["", "42", "", ""])
  })

  it("passes the task id to a failed trace process group", () => {
    appState.messages = [
      { id: "msg-user", role: "user", content: "Run analysis", timestamp: "1000" },
    ] as any
    appState.traceEvents = [
      { event_id: "start", event_type: "task_start", timestamp: 1500, data: {} },
      { event_id: "fail", event_type: "task_failed", timestamp: 1600, data: { error: "boom" } },
    ] as any
    appState.currentTask = { id: "42", title: "Task", status: "failed" } as any

    render(<TaskConversationPanel mode="page" showProcessView={false} />)

    const renderedMessages = screen.getAllByTestId("chat-message")
    expect(renderedMessages.at(-1)).toHaveAttribute("data-failure-task-id", "42")
    expect(renderedMessages[0]).toHaveAttribute("data-failure-task-id", "")
  })

  it("labels a failed embedded turn once when trace group and result row both fail", () => {
    appState.messages = [
      { id: "msg-user", role: "user", content: "Run analysis", timestamp: "1000" },
      {
        id: "msg-failed",
        role: "assistant",
        content: "Task execution failed.",
        timestamp: "1700",
        status: "failed",
        isResult: true,
      },
    ] as any
    appState.traceEvents = [
      { event_id: "start", event_type: "task_start", timestamp: 1500, data: {} },
      { event_id: "fail", event_type: "task_failed", timestamp: 1600, data: { error: "boom" } },
    ] as any
    appState.currentTask = { id: "42", title: "Task", status: "failed" } as any

    render(<TaskConversationPanel mode="page" showProcessView={false} />)

    const labels = screen
      .getAllByTestId("chat-message")
      .map((message) => message.getAttribute("data-failure-task-id"))
    expect(labels.filter((label) => label === "42")).toHaveLength(1)
    const failedRow = screen
      .getAllByTestId("chat-message")
      .find((message) => message.textContent?.includes("Task execution failed."))
    expect(failedRow).toHaveAttribute("data-failure-task-id", "42")
  })

  it("passes the task id to the virtual failure placeholder of the current task", () => {
    appState.messages = [
      { id: "msg-user", role: "user", content: "Run analysis", timestamp: "1000" },
    ] as any
    appState.traceEvents = []
    appState.currentTask = { id: "42", title: "Task", status: "failed" } as any

    render(<TaskConversationPanel mode="page" />)

    const renderedMessages = screen.getAllByTestId("chat-message")
    expect(renderedMessages).toHaveLength(2)
    expect(renderedMessages[1]).toHaveAttribute("data-failure-task-id", "42")
  })

  it("does not label the virtual placeholder with a mismatched task id", () => {
    appState.messages = [
      { id: "msg-user", role: "user", content: "Run analysis", timestamp: "1000" },
    ] as any
    appState.traceEvents = []
    appState.currentTask = { id: "7", title: "Previous task", status: "failed" } as any

    render(<TaskConversationPanel mode="page" />)

    const renderedMessages = screen.getAllByTestId("chat-message")
    expect(renderedMessages).toHaveLength(2)
    expect(renderedMessages[1]).toHaveAttribute("data-failure-task-id", "")
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

  describe("trace process rendering of paused and interrupted runs", () => {
    const traceEvent = (
      id: string,
      type: string,
      stepId: string,
      timestamp: number,
      data: Record<string, unknown> = {},
    ) => ({ event_id: id, event_type: type, step_id: stepId, timestamp: String(timestamp), data })
    const task = (status: string) => ({
      id: "42",
      title: "Task",
      description: "Task",
      status,
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
    })
    const turnOne = { id: "u1", role: "user", content: "First", timestamp: "1000" }

    beforeEach(() => {
      realChatMessage.enabled = true
    })

    it("passes the connection notice to a running turn that already has a trace", () => {
      appState.messages = [turnOne]
      appState.traceEvents = [
        traceEvent("a-start", "react_task_start", "react_a", 1100, { pattern: "ReActPattern" }),
      ]
      appState.currentTask = task("running")
      appState.isProcessing = true
      appControls.connectionNotice = "reconnecting"

      render(<TaskConversationPanel mode="page" />)

      expect(screen.getByText("clientErrors.connectionReconnecting")).toBeInTheDocument()
    })

    it("keeps a resumed run spinning after a pause in the same turn", () => {
      appState.messages = [turnOne]
      appState.traceEvents = [
        traceEvent("a-start", "react_task_start", "react_a", 1100, { pattern: "ReActPattern" }),
        traceEvent("a-tool", "tool_execution_start", "react_a", 1400, { tool_name: "web_search" }),
        traceEvent("a-end", "react_task_end", "react_a", 1500, {
          pattern: "ReActPattern",
          status: "interrupted",
        }),
        traceEvent("b-start", "react_task_start", "react_b", 2000, { pattern: "ReActPattern" }),
        traceEvent("b-tool", "tool_execution_start", "react_b", 2160, { tool_name: "web_search" }),
      ]
      appState.currentTask = task("running")

      const { container } = render(<TaskConversationPanel mode="page" />)

      expect(container.querySelector(".animate-spin")).not.toBeNull()
      expect(container.querySelector("svg[data-icon='interrupted']")).toBeNull()
      expect(screen.queryByText("agent.layout.status.interrupted")).toBeNull()
    })

    it("renders a paused-and-abandoned DAG step as stopped in a closed turn", () => {
      appState.messages = [
        turnOne,
        { id: "u2", role: "user", content: "Second", timestamp: "3000" },
        { id: "r2", role: "assistant", content: "ok", timestamp: "3500", isResult: true },
      ]
      appState.traceEvents = [
        traceEvent("d-start", "dag_execute_start", "root", 1100),
        traceEvent("d1-s", "dag_step_start", "s1", 1200, { step_name: "Step one" }),
        traceEvent("d1-e", "dag_step_end", "s1", 1300, { step_name: "Step one", result: "ok" }),
        traceEvent("d2-s", "dag_step_start", "s2", 1400, { step_name: "Step two" }),
        traceEvent("d2-tool", "tool_execution_start", "s2", 1450, { tool_name: "web_search" }),
        traceEvent("d-end", "dag_execute_end", "root", 1500, { status: "interrupted" }),
        traceEvent("c-start", "react_task_start", "react_c", 3100),
        traceEvent("c-end", "react_task_end", "react_c", 3400, { status: "completed" }),
      ]
      appState.currentTask = task("completed")

      const { container } = render(<TaskConversationPanel mode="page" />)

      const stepTwo = screen.getByRole("button", { name: /Step two/ })
      const icon = stepTwo.querySelector("svg[data-icon='interrupted']")
      expect(icon).not.toBeNull()
      expect(icon?.querySelector("line[x1='9'][y1='15'][x2='15'][y2='9']")).not.toBeNull()
      expect(stepTwo.querySelector(".animate-spin")).toBeNull()
      expect(container.querySelector(".animate-spin")).toBeNull()
    })
  })
  describe("a turn the visitor stopped", () => {
    const traceEvent = (id: string, type: string, stepId: string, timestamp: number, data: Record<string, unknown> = {}) => (
      { event_id: id, event_type: type, step_id: stepId, timestamp: String(timestamp), data }
    )
    const failedTask = {
      id: "42",
      title: "Task",
      description: "Task",
      status: "failed",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
    }
    const STOPPED_BUBBLE = "This response was interrupted."
    const user = (id: string, timestamp: number) => ({ id, role: "user", content: `question ${id}`, timestamp: String(timestamp) })
    const stoppedBubble = (timestamp: number) => ({
      id: "msg-stopped-1",
      role: "assistant",
      content: STOPPED_BUBBLE,
      timestamp: String(timestamp),
      status: "completed",
      isResult: true,
    })
    // A turn cut off while a tool was still running: no terminal trace event.
    const cutOffTurn = (prefix: string, start: number) => [
      traceEvent(`${prefix}-start`, "react_task_start", `react_${prefix}`, start, { pattern: "ReActPattern" }),
      traceEvent(`${prefix}-tool`, "tool_execution_start", `react_${prefix}`, start + 100, { tool_name: "web_search" }),
    ]
    const FAILURE_TEXTS = ["traceEventRenderer.unknownError", "common.errors.unknown"]

    beforeEach(() => {
      realChatMessage.enabled = true
      appState.currentTask = failedTask
    })

    it("shows the stopped turn's process as interrupted, not as a red failure", () => {
      appState.messages = [user("u1", 1000), stoppedBubble(2000)]
      appState.traceEvents = cutOffTurn("a", 1100)
      appState.stoppedTurnAnchorIds = ["u1"]

      const { container } = render(<TaskConversationPanel mode="page" />)

      const icon = container.querySelector("svg[data-icon='interrupted']")
      expect(icon).not.toBeNull()
      expect(icon?.querySelector("line[x1='9'][y1='15'][x2='15'][y2='9']")).not.toBeNull()
      expect(container.querySelector(".text-red-500")).toBeNull()
      expect(container.querySelector(".animate-spin")).toBeNull()
      for (const text of FAILURE_TEXTS) expect(screen.queryByText(text)).toBeNull()
    })

    it("still shows a failure for a turn the visitor did not stop", () => {
      appState.messages = [user("u1", 1000), stoppedBubble(2000)]
      appState.traceEvents = cutOffTurn("a", 1100)
      appState.stoppedTurnAnchorIds = []

      const { container } = render(<TaskConversationPanel mode="page" />)

      expect(container.querySelector("svg[data-icon='interrupted']")).toBeNull()
      expect(container.querySelector(".text-red-500")).not.toBeNull()
    })

    it("marks the turn when no bubble was added for its terminal frame", () => {
      appState.messages = [user("u1", 1000)]
      appState.traceEvents = cutOffTurn("a", 1100)
      appState.stoppedTurnAnchorIds = ["u1"]

      const { container } = render(<TaskConversationPanel mode="page" />)

      expect(container.querySelector("svg[data-icon='interrupted']")).not.toBeNull()
      expect(container.querySelector(".text-red-500")).toBeNull()
      for (const text of FAILURE_TEXTS) expect(screen.queryByText(text)).toBeNull()
      expect(screen.queryByTestId("task-failure-reference")).toBeNull()
      // No bubble says the turn was interrupted, so the process says it once.
      // The step icon's screen-reader label carries the same words.
      expect(
        screen
          .queryAllByText("agent.layout.status.interrupted")
          .filter((element) => !element.classList.contains("sr-only"))
      ).toHaveLength(1)
    })

    it("shows the interruption once, in the turn's own bubble", () => {
      appState.messages = [user("u1", 1000), stoppedBubble(2000)]
      appState.traceEvents = cutOffTurn("a", 1100)
      appState.stoppedTurnAnchorIds = ["u1"]

      render(<TaskConversationPanel mode="page" />)

      expect(screen.getAllByText(STOPPED_BUBBLE)).toHaveLength(1)
      // The step icon's screen-reader label carries the same words; only the
      // visible "Interrupted" line must be absent.
      expect(
        screen
          .queryAllByText("agent.layout.status.interrupted")
          .filter((element) => !element.classList.contains("sr-only"))
      ).toHaveLength(0)
    })

    it("says the turn was interrupted when there is no trace and no bubble to say it", () => {
      appState.messages = [user("u1", 1000)]
      appState.traceEvents = []
      appState.stoppedTurnAnchorIds = ["u1"]

      const { container } = render(<TaskConversationPanel mode="page" />)

      expect(screen.getByText("agent.layout.status.interrupted")).toBeInTheDocument()
      expect(container.querySelector(".text-red-500")).toBeNull()
      for (const text of FAILURE_TEXTS) expect(screen.queryByText(text)).toBeNull()
      expect(screen.queryByTestId("task-failure-reference")).toBeNull()
    })

    it("re-renders the process as interrupted when only the record changes", () => {
      appState.messages = [user("u1", 1000), stoppedBubble(2000)]
      appState.traceEvents = cutOffTurn("a", 1100)
      appState.stoppedTurnAnchorIds = []

      const { container, rerender } = render(<TaskConversationPanel mode="page" />)
      expect(container.querySelector("svg[data-icon='interrupted']")).toBeNull()

      appState.stoppedTurnAnchorIds = ["u1"]
      rerender(<TaskConversationPanel mode="page" />)

      expect(container.querySelector("svg[data-icon='interrupted']")).not.toBeNull()
      expect(container.querySelector(".text-red-500")).toBeNull()
    })

    it("leaves a later turn running normally after the stopped one", () => {
      appState.messages = [user("u1", 1000), stoppedBubble(2000), user("u2", 3000)]
      appState.traceEvents = [...cutOffTurn("a", 1100), ...cutOffTurn("b", 3100)]
      appState.stoppedTurnAnchorIds = ["u1"]
      appState.currentTask = { ...failedTask, status: "running" }

      const { container } = render(<TaskConversationPanel mode="page" />)

      expect(container.querySelectorAll("svg[data-icon='interrupted']")).toHaveLength(1)
      expect(container.querySelector(".animate-spin")).not.toBeNull()
      expect(container.querySelector(".text-red-500")).toBeNull()
    })
  })
  describe("upload-only attachments", () => {
    const attachments = { accept: ".txt", maxFiles: 3, maxFileBytes: 100 }

    it("keeps staged files and the upload control while files stay disabled", () => {
      appControls.filesDisabled = true
      render(<TaskConversationPanel mode="page" hideFileUpload={false} attachments={attachments} />)

      fireEvent.click(screen.getByRole("button", { name: "stage file" }))

      expect(chatInputProps.current?.files).toHaveLength(1)
      expect(chatInputProps.current?.onFilesChange).toEqual(expect.any(Function))
      expect(chatInputProps.current?.hideFileUpload).toBe(false)
      expect(chatInputProps.current?.filesDisabled).toBe(true)
      expect(chatInputProps.current?.attachments).toBe(attachments)
      expect(screen.queryByTestId("task-file-manager")).not.toBeInTheDocument()
      expect(screen.queryByTestId("file-preview-content")).not.toBeInTheDocument()
    })

    it("still hides the upload control when the caller hides it", () => {
      appControls.filesDisabled = true
      render(<TaskConversationPanel mode="page" hideFileUpload attachments={attachments} />)

      expect(chatInputProps.current?.hideFileUpload).toBe(true)
    })

    it("hands the staged files to onSend and clears them only after it succeeds", async () => {
      appControls.filesDisabled = true
      const onSend = vi.fn().mockResolvedValue(undefined)
      render(<TaskConversationPanel mode="page" attachments={attachments} onSend={onSend} />)
      fireEvent.click(screen.getByRole("button", { name: "stage file" }))
      const staged = chatInputProps.current?.files ?? []

      await act(async () => {
        await chatInputProps.current?.onSend("send draft", { mode: "balanced" })
      })

      expect(staged).toHaveLength(1)
      expect(onSend).toHaveBeenCalledWith("send draft", { mode: "balanced" }, staged)
      expect(sendMessageMock).not.toHaveBeenCalled()
      expect(chatInputProps.current?.files).toEqual([])
    })

    it("keeps the staged files when onSend fails", async () => {
      appControls.filesDisabled = true
      const onSend = vi.fn().mockRejectedValue(new Error("not sent"))
      render(<TaskConversationPanel mode="page" attachments={attachments} onSend={onSend} />)
      fireEvent.click(screen.getByRole("button", { name: "stage file" }))

      await act(async () => {
        await expect(chatInputProps.current?.onSend("send draft")).rejects.toThrow("not sent")
      })

      expect(chatInputProps.current?.files).toHaveLength(1)
    })

    const txt = (name: string) => new File([name], name, { type: "text/plain" })
    const pick = (next: File[]) => act(() => { chatInputProps.current?.onFilesChange?.(next) })
    const pendingSend = () => {
      let finish!: () => void
      const onSend = vi.fn(() => new Promise<void>((resolve) => { finish = resolve }))
      return { onSend, finish: () => finish() }
    }

    it.each([
      ["with files", (a: File) => [a], (a: File) => [a]],
      ["text-only", () => [], () => []],
    ])("keeps files added while a %s send is in flight and removes only the files it carried", async (_kind, before, sent) => {
      appControls.filesDisabled = true
      const { onSend, finish } = pendingSend()
      render(<TaskConversationPanel mode="page" attachments={attachments} onSend={onSend} />)
      const [a, b] = [txt("a.txt"), txt("b.txt")]
      pick(before(a))
      let sending!: Promise<void> | void
      act(() => { sending = chatInputProps.current?.onSend("draft") })
      pick([...(chatInputProps.current?.files ?? []), b])

      await act(async () => { finish(); await sending })

      expect(onSend).toHaveBeenCalledWith("draft", undefined, sent(a))
      expect(chatInputProps.current?.files).toEqual([b])
    })

    it("keeps files hidden while uploads are not offered through a text-only send", async () => {
      appControls.filesDisabled = true
      const onSend = vi.fn().mockResolvedValue(undefined)
      const { rerender } = render(
        <TaskConversationPanel mode="page" attachments={attachments} onSend={onSend} />,
      )
      const a = txt("a.txt")
      pick([a])
      rerender(<TaskConversationPanel mode="page" />)
      expect(chatInputProps.current?.files).toEqual([])

      await act(async () => { await chatInputProps.current?.onSend("text only") })
      rerender(<TaskConversationPanel mode="page" attachments={attachments} onSend={onSend} />)

      expect(sendMessageMock).toHaveBeenCalledWith("text only", undefined, [])
      expect(chatInputProps.current?.files).toEqual([a])
    })

    it("still clears every file after a send when files are enabled", async () => {
      const { onSend, finish } = pendingSend()
      render(<TaskConversationPanel mode="page" onSend={onSend} />)
      const [a, b] = [txt("a.txt"), txt("b.txt")]
      pick([a])
      let sending!: Promise<void> | void
      act(() => { sending = chatInputProps.current?.onSend("draft") })
      pick([a, b])

      await act(async () => { finish(); await sending })

      expect(onSend).toHaveBeenCalledWith("draft", undefined, [a])
      expect(chatInputProps.current?.files).toEqual([])
    })

    it("shows the files the caller holds and reports every change to it", async () => {
      appControls.filesDisabled = true
      const { onSend, finish } = pendingSend()
      const onAttachmentFilesChange = vi.fn()
      const [a, b, c] = [txt("a.txt"), txt("b.txt"), txt("c.txt")]
      const panel = (held: File[]) => (
        <TaskConversationPanel
          mode="page"
          attachments={attachments}
          attachmentFiles={held}
          onAttachmentFilesChange={onAttachmentFilesChange}
          onSend={onSend}
        />
      )
      const { rerender } = render(panel([a]))
      expect(chatInputProps.current?.files).toEqual([a])
      pick([a, b])
      expect(onAttachmentFilesChange).toHaveBeenLastCalledWith([a, b])
      expect(chatInputProps.current?.files).toEqual([a])

      let sending!: Promise<void> | void
      act(() => { sending = chatInputProps.current?.onSend("draft") })
      rerender(panel([a, b, c]))
      await act(async () => { finish(); await sending })

      expect(onSend).toHaveBeenCalledWith("draft", undefined, [a])
      expect(onAttachmentFilesChange).toHaveBeenLastCalledWith([b, c])
    })

    it("reports the staged files to the caller as the visitor changes them", async () => {
      appControls.filesDisabled = true
      const onAttachmentFilesChange = vi.fn()
      render(
        <TaskConversationPanel
          mode="page"
          attachments={attachments}
          onAttachmentFilesChange={onAttachmentFilesChange}
          onSend={vi.fn().mockResolvedValue(undefined)}
        />,
      )

      fireEvent.click(screen.getByRole("button", { name: "stage file" }))
      expect(onAttachmentFilesChange).toHaveBeenLastCalledWith([expect.any(File)])
      await act(async () => {
        await chatInputProps.current?.onSend("send draft")
      })

      expect(onAttachmentFilesChange).toHaveBeenLastCalledWith([])
    })

    it("counts the caller's attachment work as a message in flight and passes the statuses through", () => {
      appControls.filesDisabled = true
      const statuses = new Map<File, ComposerAttachmentStatus>()
      const onRetryAttachment = vi.fn()
      const { rerender } = render(
        <TaskConversationPanel
          mode="page"
          attachments={attachments}
          attachmentStatuses={statuses}
          onRetryAttachment={onRetryAttachment}
        />,
      )
      expect(chatInputProps.current?.isLoading).toBe(false)
      expect(chatInputProps.current?.attachmentStatuses).toBe(statuses)
      expect(chatInputProps.current?.onRetryAttachment).toBe(onRetryAttachment)

      rerender(<TaskConversationPanel mode="page" attachments={attachments} attachmentsBusy />)

      expect(chatInputProps.current?.isLoading).toBe(true)
    })
  })
})

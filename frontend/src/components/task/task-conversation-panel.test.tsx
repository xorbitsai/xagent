import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const appState = vi.hoisted(() => ({
  messages: [],
  traceEvents: [],
  currentTask: null,
  isProcessing: false,
  isHistoryLoading: false,
  taskId: 42,
  filePreview: { isOpen: false },
  dagExecution: null,
  steps: [],
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    state: appState,
    sendMessage: vi.fn(),
    pauseTask: vi.fn(),
    resumeTask: vi.fn(),
    openFilePreview: vi.fn(),
    closeFilePreview: vi.fn(),
    requestStatus: vi.fn(),
    dispatch: vi.fn(),
  }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
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
    showEmptyStatus,
  }: {
    content?: string | null
    interactionsActive?: boolean
    traceEvents?: unknown[]
    taskStatus?: string
    showEmptyStatus?: boolean
  }) => (
    <div
      data-testid="chat-message"
      data-active={interactionsActive ? "true" : "false"}
      data-trace-count={traceEvents?.length ?? 0}
      data-task-status={taskStatus || ""}
      data-show-empty-status={showEmptyStatus ? "true" : "false"}
    >
      {content}
    </div>
  ),
}))

vi.mock("@/components/chat/ChatInput", () => ({
  ChatInput: () => <div data-testid="chat-input" />,
}))

vi.mock("@/components/chat/TokenUsageDisplay", () => ({
  TokenUsageDisplay: () => null,
}))

vi.mock("@/components/file/task-file-manager", () => ({
  TaskFileManager: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}))

vi.mock("@/components/file/file-preview-content", () => ({
  FilePreviewContent: () => null,
}))

vi.mock("@/components/file/file-preview-action-buttons", () => ({
  FilePreviewActionButtons: () => null,
}))

vi.mock("@/components/preview-sheet", () => ({
  PreviewSheet: ({ children }: { children: React.ReactNode }) => <>{children}</>,
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
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    appState.messages = []
    appState.traceEvents = []
    appState.currentTask = null
    appState.isProcessing = false
    appState.isHistoryLoading = false
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
    expect(renderedMessages[2]).toHaveTextContent("Done")
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

  it("ignores malformed DAG layout failures without throwing", async () => {
    appState.messages = []
    appState.traceEvents = []
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
    fireEvent.click(screen.getByTitle("chatPage.executionPlan.title"))

    expect(await screen.findByTestId("center-panel")).toHaveAttribute("data-node-count", "3")
    expect(screen.getByTestId("center-panel")).toHaveAttribute("data-edge-count", "0")
  })
})

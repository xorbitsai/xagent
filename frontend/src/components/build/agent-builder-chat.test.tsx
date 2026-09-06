import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper"
  )
  return {
    ...actual,
    apiRequest: apiRequestMock,
  }
})

vi.mock("@/lib/utils", () => ({
  getApiUrl: () => "http://api.local",
  getUploadApiUrl: () => "http://api.local",
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ token: "token" }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string>) =>
      vars?.appName ? `${key}:${vars.appName}` : key,
  }),
}))

vi.mock("@/lib/branding", () => ({
  getBrandingFromEnv: () => ({ appName: "Xagent" }),
}))

vi.mock("sonner", () => ({
  toast: {
    error: toastErrorMock,
  },
}))

vi.mock("@/components/chat/ChatInput", () => ({
  ChatInput: ({
    onSend,
    files = [],
    onFilesChange,
  }: {
    onSend?: (message: string) => void | Promise<void>
    files?: File[]
    onFilesChange?: (files: File[]) => void
  }) => (
    <div>
      <button
        type="button"
        onClick={() =>
          onFilesChange?.([
            ...files,
            new File(["chat-input"], "chat-input.txt", { type: "text/plain" }),
          ])
        }
      >
        attach-chat-input-file
      </button>
      <button
        type="button"
        onClick={() => {
          const file = new File(["chat-input"], "chat-input.txt", { type: "text/plain" }) as File & { file_id?: string }
          file.file_id = "existing-file-id"
          onFilesChange?.([...files, file])
        }}
      >
        attach-preuploaded-chat-input-file
      </button>
      <button
        type="button"
        onClick={() => {
          const preUploaded = new File(["existing"], "existing.txt", {
            type: "text/plain",
          }) as File & { file_id?: string }
          preUploaded.file_id = "duplicate-file-id"
          onFilesChange?.([
            preUploaded,
            new File(["new"], "new.txt", { type: "text/plain" }),
          ])
        }}
      >
        attach-mixed-chat-input-files
      </button>
      <button type="button" onClick={() => onSend?.("chat input message")}>
        send-chat-input
      </button>
    </div>
  ),
}))

vi.mock("@/components/chat/ChatMessage", () => ({
  ChatMessage: ({
    content,
    onSendInteraction,
    processStatus,
    traceEvents,
  }: {
    content?: React.ReactNode
    onSendInteraction?: (text: string, files?: File[]) => Promise<void> | void
    processStatus?: string
    traceEvents?: unknown[]
  }) => {
    const [status, setStatus] = React.useState("idle")

    if (!onSendInteraction) {
      return (
        <div
          data-testid="chat-message"
          data-process-status={processStatus || ""}
          data-trace-count={traceEvents?.length ?? 0}
        >
          {content || "message"}
        </div>
      )
    }

    return (
      <div
        data-testid="chat-message"
        data-process-status={processStatus || ""}
        data-trace-count={traceEvents?.length ?? 0}
      >
        <span data-testid="chat-content">{content}</span>
        <button
          type="button"
          onClick={async () => {
            try {
              await onSendInteraction("upload this", [
                new File(["data"], "data.txt", { type: "text/plain" }),
              ])
              setStatus("resolved")
            } catch {
              setStatus("rejected")
            }
          }}
        >
          send-file-interaction
        </button>
        <span>{status}</span>
      </div>
    )
  },
}))

vi.mock("@/components/ui/scroll-area", () => ({
  ScrollArea: React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
    ({ children, ...props }, ref) => (
      <div ref={ref} {...props}>
        {children}
      </div>
    )
  ),
}))

vi.mock("@/components/file/file-attachment", () => ({
  FileAttachment: () => <div>attachment</div>,
}))

vi.mock("lucide-react", () => ({
  Bot: (props: React.SVGProps<SVGSVGElement>) => <svg {...props} />,
}))

import { AgentBuilderChat, type AgentConfig } from "./agent-builder-chat"

class MockWebSocket {
  static OPEN = 1
  static CONNECTING = 0
  static instances: MockWebSocket[] = []

  readyState = MockWebSocket.CONNECTING
  sentMessages: string[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onerror: ((event?: unknown) => void) | null = null
  onclose: (() => void) | null = null

  constructor(_url: string) {
    MockWebSocket.instances.push(this)
  }

  send(message: string) {
    this.sentMessages.push(message)
  }

  close() {
    this.readyState = 3
    this.onclose?.()
  }

  open() {
    this.readyState = MockWebSocket.OPEN
    this.onopen?.()
  }
}

const originalWebSocket = globalThis.WebSocket

const agentConfig: AgentConfig = {
  name: "Demo",
  description: "Demo",
  instructions: "Help",
  executionMode: "balanced",
  suggestedPrompts: [],
  selectedToolCategories: [],
  modelConfig: {
    general: null,
    small_fast: null,
    visual: null,
    compact: null,
  },
}

const successfulUploadResponse = (files: unknown[]) =>
  new Response(JSON.stringify({ success: true, files }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })

const renderBuilderChat = () =>
  render(
    <AgentBuilderChat
      agentConfig={agentConfig}
      onUpdateConfig={vi.fn()}
    />
  )

const expectUploadRejectedWithoutSend = async () => {
  await waitFor(() => {
    expect(apiRequestMock).toHaveBeenCalledTimes(1)
  })
  MockWebSocket.instances.forEach((ws) => ws.open())
  await waitFor(() => {
    expect(toastErrorMock.mock.calls[0]?.[0]).toBe("Failed to upload files")
  })
  expect(MockWebSocket.instances.flatMap((ws) => ws.sentMessages)).toEqual([])
}

describe("AgentBuilderChat", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
    MockWebSocket.instances = []
    globalThis.WebSocket = MockWebSocket as unknown as typeof WebSocket
  })

  afterEach(() => {
    cleanup()
    globalThis.WebSocket = originalWebSocket
  })

  it("renders ordinary agent messages in chat without requiring a response", async () => {
    renderBuilderChat()
    fireEvent.click(screen.getByText("send-chat-input"))

    const ws = MockWebSocket.instances[0]
    ws.open()
    ws.onmessage?.({
      data: JSON.stringify({
        type: "trace_event",
        event_type: "agent_message",
        data: {
          message: "Ordinary visible update",
          message_type: "info",
          expect_response: false,
          display: "chat",
        },
      }),
    })

    await waitFor(() => {
      expect(screen.getByText("Ordinary visible update")).toBeInTheDocument()
    })
    let messages = screen.getAllByTestId("chat-message")
    expect(messages[messages.length - 2]).toHaveAttribute("data-trace-count", "1")

    ws.onmessage?.({
      data: JSON.stringify({
        type: "trace_event",
        event_type: "ai_message",
        data: { content: "Final builder answer", display: "chat" },
      }),
    })

    await waitFor(() => {
      expect(screen.getByText("Ordinary visible update")).toBeInTheDocument()
      expect(screen.getByText("Final builder answer")).toBeInTheDocument()
    })
    messages = screen.getAllByTestId("chat-message")
    expect(messages[messages.length - 2]).toHaveTextContent("Ordinary visible update")
    expect(messages[messages.length - 1]).toHaveTextContent("Final builder answer")
  })

  it("preserves a final answer when a non-waiting message arrives afterwards", async () => {
    renderBuilderChat()
    fireEvent.click(screen.getByText("send-chat-input"))

    const ws = MockWebSocket.instances[0]
    ws.open()
    ws.onmessage?.({
      data: JSON.stringify({
        type: "trace_event",
        event_type: "ai_message",
        data: { content: "Final answer first", display: "chat" },
      }),
    })
    ws.onmessage?.({
      data: JSON.stringify({
        type: "trace_event",
        event_type: "agent_message",
        data: {
          message: "Late durable note",
          expect_response: false,
          display: "chat",
        },
      }),
    })
    ws.onmessage?.({
      data: JSON.stringify({
        type: "task_completed",
        status: "completed",
        result: { content: "Final answer first" },
      }),
    })

    await waitFor(() => {
      expect(screen.getByText("Final answer first")).toBeInTheDocument()
      expect(screen.getByText("Late durable note")).toBeInTheDocument()
    })
    expect(screen.getAllByText("Final answer first")).toHaveLength(1)
  })

  it("renders native final-answer stream events from the builder bridge", async () => {
    renderBuilderChat()
    fireEvent.click(screen.getByText("send-chat-input"))

    const ws = MockWebSocket.instances[0]
    ws.open()
    for (const event of [
      { type: "final_answer_start", message_id: "answer-1" },
      { type: "final_answer_delta", message_id: "answer-1", delta: "Streamed " },
      { type: "final_answer_delta", message_id: "answer-1", delta: "answer" },
      { type: "final_answer_end", message_id: "answer-1", content: "Streamed answer" },
    ]) {
      ws.onmessage?.({ data: JSON.stringify(event) })
    }

    await waitFor(() => {
      expect(screen.getByText("Streamed answer")).toBeInTheDocument()
    })
    expect(screen.getAllByText("Streamed answer")).toHaveLength(1)
  })

  it("keeps each streamed final answer inside its own user turn", async () => {
    renderBuilderChat()
    fireEvent.click(screen.getByText("send-chat-input"))

    const ws = MockWebSocket.instances[0]
    ws.open()
    ws.onmessage?.({
      data: JSON.stringify({
        type: "trace_event",
        event_type: "ai_message",
        data: { content: "First answer", display: "chat" },
      }),
    })
    ws.onmessage?.({
      data: JSON.stringify({
        type: "task_completed",
        status: "completed",
        result: { content: "First answer" },
      }),
    })

    await waitFor(() => {
      expect(screen.getByText("First answer")).toBeInTheDocument()
    })
    fireEvent.click(screen.getByText("send-chat-input"))
    for (const event of [
      { type: "final_answer_start", message_id: "answer-2" },
      { type: "final_answer_delta", message_id: "answer-2", delta: "Second answer" },
      { type: "final_answer_end", message_id: "answer-2", content: "Second answer" },
    ]) {
      ws.onmessage?.({ data: JSON.stringify(event) })
    }

    await waitFor(() => {
      expect(screen.getByText("First answer")).toBeInTheDocument()
      expect(screen.getByText("Second answer")).toBeInTheDocument()
    })
  })

  it("removes an empty completion placeholder after a durable message", async () => {
    renderBuilderChat()
    fireEvent.click(screen.getByText("send-chat-input"))

    const ws = MockWebSocket.instances[0]
    ws.open()
    ws.onmessage?.({
      data: JSON.stringify({
        type: "trace_event",
        event_type: "agent_message",
        data: {
          message: "Only durable update",
          expect_response: false,
          display: "chat",
        },
      }),
    })
    ws.onmessage?.({
      data: JSON.stringify({
        type: "task_completed",
        status: "completed",
        result: "",
      }),
    })

    await waitFor(() => {
      expect(screen.getByText("Only durable update")).toBeInTheDocument()
      expect(screen.getAllByTestId("chat-message")).toHaveLength(3)
    })
    expect(
      screen.queryByText("builds.configForm.chat.defaultReply"),
    ).not.toBeInTheDocument()
  })

  it("keeps timeline agent messages out of builder chat content", async () => {
    renderBuilderChat()
    fireEvent.click(screen.getByText("send-chat-input"))

    const ws = MockWebSocket.instances[0]
    ws.open()
    ws.onmessage?.({
      data: JSON.stringify({
        type: "trace_event",
        event_type: "agent_message",
        data: { message: "Timeline only", display: "timeline" },
      }),
    })

    await waitFor(() => {
      const messages = screen.getAllByTestId("chat-message")
      expect(messages[messages.length - 1]).toHaveAttribute("data-trace-count", "1")
    })
    expect(screen.queryByText("Timeline only")).not.toBeInTheDocument()
  })

  it("renders waiting questions in builder chat", async () => {
    renderBuilderChat()
    fireEvent.click(screen.getByText("send-chat-input"))

    const ws = MockWebSocket.instances[0]
    ws.open()
    ws.onmessage?.({
      data: JSON.stringify({
        type: "trace_event",
        event_type: "agent_message",
        data: {
          message: "Choose an option",
          message_type: "question",
          expect_response: true,
          display: "chat",
        },
      }),
    })

    await waitFor(() => {
      expect(screen.getByText("Choose an option")).toBeInTheDocument()
    })
  })

  it("shows backend upload error details when file upload is unavailable", async () => {
    apiRequestMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "Startup file storage sync failed" }), {
        status: 503,
        statusText: "Service Unavailable",
        headers: { "Content-Type": "application/json" },
      })
    )

    render(
      <AgentBuilderChat
        agentConfig={agentConfig}
        onUpdateConfig={vi.fn()}
      />
    )

    fireEvent.click(await screen.findByText("send-file-interaction"))

    await waitFor(() => {
      expect(toastErrorMock.mock.calls[0]?.[0]).toBe(
        "Startup file storage sync failed"
      )
    })

    await waitFor(() => {
      expect(screen.getByText("rejected")).toBeInTheDocument()
    })
  })

  it.each([
    {
      caseName: "contains a blank file id",
      files: [{ file_id: "   ", filename: "data.txt" }],
    },
    { caseName: "has the wrong result count", files: [] },
  ])("does not send when an upload response $caseName", async ({ files }) => {
    apiRequestMock.mockResolvedValueOnce(successfulUploadResponse(files))
    renderBuilderChat()

    fireEvent.click(await screen.findByText("send-file-interaction"))

    await expectUploadRejectedWithoutSend()
  })

  it("does not send when a new upload duplicates a pre-uploaded file id", async () => {
    apiRequestMock.mockResolvedValueOnce(
      successfulUploadResponse([
        { file_id: "duplicate-file-id", filename: "new.txt" },
      ])
    )
    renderBuilderChat()

    fireEvent.click(screen.getByText("attach-mixed-chat-input-files"))
    fireEvent.click(screen.getByText("send-chat-input"))

    await expectUploadRejectedWithoutSend()
  })

  it("reuses pre-uploaded file ids from compact chat input instead of uploading again", async () => {
    render(
      <AgentBuilderChat
        agentConfig={agentConfig}
        onUpdateConfig={vi.fn()}
      />
    )

    fireEvent.click(screen.getByText("attach-preuploaded-chat-input-file"))
    fireEvent.click(screen.getByText("send-chat-input"))

    expect(MockWebSocket.instances).toHaveLength(1)
    MockWebSocket.instances[0].open()

    await waitFor(() => {
      expect(MockWebSocket.instances[0].sentMessages).toHaveLength(1)
    })

    const payload = JSON.parse(MockWebSocket.instances[0].sentMessages[0])
    expect(payload.files).toEqual([
      {
        file_id: "existing-file-id",
        name: "chat-input.txt",
        size: 10,
        type: "text/plain",
      },
    ])
    expect(apiRequestMock).not.toHaveBeenCalled()
  })

  it("passes failed task completion status into the process renderer", async () => {
    render(
      <AgentBuilderChat
        agentConfig={agentConfig}
        onUpdateConfig={vi.fn()}
      />
    )

    fireEvent.click(screen.getByText("send-chat-input"))

    expect(MockWebSocket.instances).toHaveLength(1)
    const ws = MockWebSocket.instances[0]
    ws.open()

    await waitFor(() => {
      expect(ws.sentMessages).toHaveLength(1)
    })

    ws.onmessage?.({
      data: JSON.stringify({
        type: "trace_event",
        event_id: "react-start",
        event_type: "react_task_start",
        step_id: "react-1",
        timestamp: 1,
        data: { pattern: "ReActPattern" },
      }),
    })
    ws.onmessage?.({
      data: JSON.stringify({
        type: "trace_event",
        event_id: "llm-start",
        event_type: "llm_call_start",
        step_id: "react-1",
        timestamp: 2,
        data: { model_name: "demo-model" },
      }),
    })
    ws.onmessage?.({
      data: JSON.stringify({
        type: "task_completed",
        success: false,
        result: "All patterns failed",
      }),
    })

    await waitFor(() => {
      const messages = screen.getAllByTestId("chat-message")
      expect(messages[messages.length - 1]).toHaveAttribute(
        "data-process-status",
        "failed"
      )
    })
  })

  it("handles null task completion results without crashing", async () => {
    render(
      <AgentBuilderChat
        agentConfig={agentConfig}
        onUpdateConfig={vi.fn()}
      />
    )

    fireEvent.click(screen.getByText("send-chat-input"))

    expect(MockWebSocket.instances).toHaveLength(1)
    const ws = MockWebSocket.instances[0]
    ws.open()

    await waitFor(() => {
      expect(ws.sentMessages).toHaveLength(1)
    })

    ws.onmessage?.({
      data: JSON.stringify({
        type: "task_completed",
        success: true,
        status: "completed",
        result: null,
      }),
    })

    await waitFor(() => {
      const messages = screen.getAllByTestId("chat-message")
      const latestMessage = messages[messages.length - 1]
      expect(latestMessage).toHaveAttribute("data-process-status", "completed")
    })
  })
})

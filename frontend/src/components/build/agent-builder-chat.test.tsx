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
        <button
          type="button"
          onClick={async () => {
            try {
              await onSendInteraction("upload this", [
                new File(["data"], "data.txt", { type: "text/plain" }),
              ])
              setStatus("resolved")
            } catch (error) {
              // Surface the declared delivery contract (#1485): the form
              // probes `disposition` off whatever this callback rejects with.
              const disposition = (error as { disposition?: unknown })?.disposition
              setStatus(
                `rejected:${typeof disposition === "string" ? disposition : "untyped"}`,
              )
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
      // The upload failed before anything reached the agent, and the typed
      // failure must say so - "not_sent" is what lets ClarificationForm tell
      // the visitor a resubmit is safe.
      expect(screen.getByText("rejected:not_sent")).toBeInTheDocument()
    })
    // The resubmit that hint invites must not stack a duplicate answer: both
    // optimistic bubbles (user + assistant placeholder) are rolled back,
    // leaving only the initial greeting.
    expect(screen.getAllByTestId("chat-message")).toHaveLength(1)
  })

  it("rolls back both optimistic bubbles when the connection setup throws", async () => {
    apiRequestMock.mockResolvedValueOnce(
      successfulUploadResponse([{ file_id: "file-1", filename: "data.txt" }])
    )
    class ThrowingWebSocket {
      static OPEN = 1
      constructor(_url: string) {
        throw new Error("SecurityError: insecure WebSocket")
      }
    }
    globalThis.WebSocket = ThrowingWebSocket as unknown as typeof WebSocket

    renderBuilderChat()
    fireEvent.click(await screen.findByText("send-file-interaction"))

    // Nothing reached the wire, so the interaction rejects as not_sent and
    // the transcript returns to just the greeting - a hinted resubmit must
    // not find a stranded answer bubble or blank placeholder.
    await waitFor(() => {
      expect(screen.getByText("rejected:not_sent")).toBeInTheDocument()
    })
    expect(screen.getAllByTestId("chat-message")).toHaveLength(1)
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

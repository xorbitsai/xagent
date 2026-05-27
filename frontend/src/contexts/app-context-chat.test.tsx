import React from "react"
import { act, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

type TestWebSocketMessage = {
  type: string
  timestamp: string
  data: unknown
}

const webSocketOptions = vi.hoisted(() => ({
  current: null as null | { onMessage?: (message: TestWebSocketMessage) => void },
}))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ token: "token" }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

vi.mock("@/hooks/use-websocket", () => ({
  useWebSocket: (options: {
    onMessage?: (message: TestWebSocketMessage) => void
  }) => {
    webSocketOptions.current = options
    return {
      isConnected: true,
      connectionError: null,
      sendChatMessage: vi.fn(),
      executeTask: vi.fn(),
      pauseTask: vi.fn(),
      resumeTask: vi.fn(),
      requestStatus: vi.fn(),
      connect: vi.fn(),
    }
  },
}))

vi.mock("sonner", () => ({
  toast: {
    error: vi.fn(),
    success: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
  },
}))

import { AppProvider, useApp } from "./app-context-chat"

function StateProbe() {
  const { state } = useApp()
  const allTraceEvents = [
    ...state.traceEvents,
    ...state.messages.flatMap((message) => message.traceEvents || []),
  ]
  return (
    <>
      <div data-testid="messages">
        {JSON.stringify(
          state.messages.map((message) => ({
            role: message.role,
            content:
              typeof message.content === "string" ? message.content : "react-node",
          }))
        )}
      </div>
      <div data-testid="trace-events">
        {JSON.stringify(
          allTraceEvents.map((event) => {
            const data = event.data as { message?: string } | undefined
            return {
              event_type: event.event_type,
              message: data?.message,
            }
          })
        )}
      </div>
    </>
  )
}

describe("AppProvider websocket message routing", () => {
  beforeEach(() => {
    webSocketOptions.current = null
  })

  it("routes historical assistant transcript rows to chat and progress events to trace", async () => {
    render(
      <AppProvider token="token">
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    act(() => {
      onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:00:00Z",
        data: {
          event_id: "chat-message-1",
          event_type: "agent_message",
          data: {
            message: "Final answer",
            content: "Final answer",
            role: "assistant",
            expect_response: false,
          },
        },
      })
    })

    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "Final answer"
      )
    })
    expect(screen.getByTestId("trace-events").textContent).not.toContain(
      "Final answer"
    )

    act(() => {
      onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:00:01Z",
        data: {
          event_id: "progress-1",
          event_type: "agent_progress",
          step_id: "react",
          data: {
            message: "Searching",
            display: "timeline",
          },
        },
      })
    })

    await waitFor(() => {
      expect(screen.getByTestId("trace-events").textContent).toContain(
        "agent_progress"
      )
    })
    expect(screen.getByTestId("messages").textContent).not.toContain("Searching")
  })
})

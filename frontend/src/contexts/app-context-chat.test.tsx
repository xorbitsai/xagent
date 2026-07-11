import React from "react"
import { act, cleanup, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

type TestWebSocketMessage = {
  type: string
  timestamp: string
  data?: unknown
  task_id?: number
  task?: Record<string, unknown>
  status?: string
  run_id?: string | null
  state_version?: number
  control_state?: string
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

import {
  AppProvider,
  extractTaskControlEnvelope,
  useApp,
} from "./app-context-chat"

type TaskControlMessage = Parameters<typeof extractTaskControlEnvelope>[0]

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
      <div data-testid="task-status">{state.currentTask?.status || ""}</div>
      <div data-testid="processing">{String(state.isProcessing)}</div>
    </>
  )
}

function SeedRunningTask() {
  const { dispatch } = useApp()

  React.useEffect(() => {
    dispatch({
      type: "SET_CURRENT_TASK",
      payload: {
        id: "1",
        title: "Test task",
        status: "running",
        description: "Test task",
        createdAt: "2026-05-27T05:00:00Z",
        updatedAt: "2026-05-27T05:00:00Z",
      },
    })
    dispatch({ type: "SET_PROCESSING", payload: true })
  }, [dispatch])

  return null
}

describe("task control envelope parsing", () => {
  it("does not coerce null, boolean, or empty identifiers to integers", () => {
    const nullEnvelope = extractTaskControlEnvelope({
      type: "task_paused",
      timestamp: "2026-05-27T05:00:00Z",
      task_id: null,
      state_version: null,
    } as unknown as TaskControlMessage)
    const coercedEnvelope = extractTaskControlEnvelope({
      type: "task_paused",
      timestamp: "2026-05-27T05:00:00Z",
      task_id: true,
      state_version: "",
    } as unknown as TaskControlMessage)

    expect(nullEnvelope.taskId).toBeUndefined()
    expect(nullEnvelope.stateVersion).toBeUndefined()
    expect(coercedEnvelope.taskId).toBeUndefined()
    expect(coercedEnvelope.stateVersion).toBeUndefined()
  })

  it("accepts positive task IDs and non-negative versions", () => {
    const envelope = extractTaskControlEnvelope({
      type: "task_paused",
      timestamp: "2026-05-27T05:00:00Z",
      task_id: "12",
      state_version: "0",
    } as unknown as TaskControlMessage)

    expect(envelope.taskId).toBe(12)
    expect(envelope.stateVersion).toBe(0)
  })
})

describe("AppProvider websocket message routing", () => {
  beforeEach(() => {
    webSocketOptions.current = null
  })

  afterEach(() => {
    cleanup()
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

  it("handles top-level failed task completion payloads", async () => {
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("running")
      expect(screen.getByTestId("processing").textContent).toBe("true")
    })

    act(() => {
      onMessage?.({
        type: "task_completed",
        timestamp: "2026-05-27T05:00:02Z",
        task: {
          id: 1,
          status: "failed",
        },
        success: false,
        result: "Task failed",
      } as TestWebSocketMessage)
    })

    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("failed")
      expect(screen.getByTestId("processing").textContent).toBe("false")
    })
  })

  it("ignores out-of-order and semantically stale task state events", async () => {
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    act(() => {
      onMessage?.({
        type: "task_paused",
        timestamp: "2026-05-27T05:00:01Z",
        task_id: 1,
        status: "paused",
        run_id: "run-1",
        state_version: 4,
        control_state: "paused",
      })
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("paused")
    })

    act(() => {
      onMessage?.({
        type: "task_resumed",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        status: "running",
        run_id: "run-1",
        state_version: 5,
        control_state: "running",
      })
      onMessage?.({
        type: "task_paused",
        timestamp: "2026-05-27T05:00:03Z",
        task_id: 1,
        status: "paused",
        run_id: "run-1",
        state_version: 4,
        control_state: "paused",
      })
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("running")
    })

    act(() => {
      onMessage?.({
        type: "task_paused",
        timestamp: "2026-05-27T05:00:04Z",
        task_id: 1,
        status: "running",
        run_id: "run-1",
        state_version: 6,
        control_state: "running",
      })
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("running")
    })
  })

  it("normalizes uppercase task info status before syncing processing state", async () => {
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("running")
      expect(screen.getByTestId("processing").textContent).toBe("true")
    })

    act(() => {
      onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:00:02Z",
        data: {
          event_id: "task-info-1",
          event_type: "task_info",
          data: {
            id: 1,
            title: "Test task",
            description: "Test task",
            status: "FAILED",
            created_at: "2026-05-27T05:00:00Z",
            updated_at: "2026-05-27T05:00:02Z",
          },
        },
      })
    })

    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("failed")
      expect(screen.getByTestId("processing").textContent).toBe("false")
    })
  })

  it("shows websocket error payloads and syncs task status when provided", async () => {
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("running")
      expect(screen.getByTestId("processing").textContent).toBe("true")
    })

    act(() => {
      onMessage?.({
        type: "task_started",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        status: "running",
        run_id: "run-1",
        state_version: 4,
        control_state: "running",
      })
      onMessage?.({
        type: "error",
        timestamp: "2026-05-27T05:00:03Z",
        message: "No live execution found to pause",
        task: {
          id: 1,
          status: "failed",
        },
      } as TestWebSocketMessage)
    })

    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("failed")
      expect(screen.getByTestId("processing").textContent).toBe("false")
      expect(screen.getByTestId("messages").textContent).toContain(
        "No live execution found to pause"
      )
    })
  })

  it("keeps running state for non-terminal agent errors without task status", async () => {
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("running")
      expect(screen.getByTestId("processing").textContent).toBe("true")
    })

    act(() => {
      onMessage?.({
        type: "agent_error",
        timestamp: "2026-05-27T05:00:04Z",
        data: {
          type: "agent_error",
          message:
            "Task is currently busy; please wait for the previous turn to finish before sending another message.",
        },
      })
    })

    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("running")
      expect(screen.getByTestId("processing").textContent).toBe("true")
      expect(screen.getByTestId("messages").textContent).toContain(
        "Task is currently busy"
      )
    })
  })

  it("syncs terminal agent errors when task status is provided", async () => {
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("running")
      expect(screen.getByTestId("processing").textContent).toBe("true")
    })

    act(() => {
      onMessage?.({
        type: "agent_error",
        timestamp: "2026-05-27T05:00:05Z",
        data: {
          type: "agent_error",
          message: "Runtime error",
          task: {
            id: 1,
            status: "failed",
          },
        },
      })
    })

    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("failed")
      expect(screen.getByTestId("processing").textContent).toBe("false")
      expect(screen.getByTestId("messages").textContent).toContain(
        "Runtime error"
      )
    })
  })

  it("stops processing when a task waits for user input", async () => {
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("running")
      expect(screen.getByTestId("processing").textContent).toBe("true")
    })

    act(() => {
      onMessage?.({
        type: "task_waiting_for_user",
        timestamp: "2026-05-27T05:00:06Z",
        data: {
          question: "Which file should I use?",
          interactions: [],
        },
      })
    })

    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe(
        "waiting_for_user"
      )
      expect(screen.getByTestId("processing").textContent).toBe("false")
      expect(screen.getByTestId("messages").textContent).toContain(
        "Which file should I use?"
      )
    })
  })
})

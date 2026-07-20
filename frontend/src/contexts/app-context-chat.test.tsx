import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
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
  current: null as null | {
    onMessage?: (message: TestWebSocketMessage) => void
    onConnect?: () => void
  },
}))
const sendChatMessageMock = vi.hoisted(() => vi.fn())
const wsHarness = vi.hoisted(() => ({ isConnected: true }))
const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api-wrapper")>()
  return {
    ...actual,
    apiRequest: (...args: unknown[]) => apiRequestMock(...args),
  }
})

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
    onConnect?: () => void
  }) => {
    webSocketOptions.current = options
    return {
      isConnected: wsHarness.isConnected,
      connectionError: null,
      sendChatMessage: sendChatMessageMock,
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
import { TASK_ERROR_EVENT, type TaskErrorEventDetail } from "@/lib/task-error-events"

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
            id: message.id,
            role: message.role,
            content:
              typeof message.content === "string" ? message.content : "react-node",
            isOptimistic: message.isOptimistic,
            isResult: message.isResult,
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

function SeedExistingTask() {
  const { dispatch } = useApp()

  React.useEffect(() => {
    dispatch({ type: "SET_TASK_ID", payload: 1 })
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
  }, [dispatch])

  return null
}

function SendMessageProbe() {
  const { sendMessage } = useApp()

  return (
    <button
      type="button"
      onClick={() => {
        void sendMessage("Optimistic round trip", {
          clientMessageId: "turn-optimistic",
        })
      }}
    >
      Send message
    </button>
  )
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
    wsHarness.isConnected = true
    apiRequestMock.mockReset()
    sendChatMessageMock.mockReset()
    sendChatMessageMock.mockResolvedValue({
      client_message_id: "turn-optimistic",
      turn_id: "turn-optimistic",
    })
    ;(window as typeof window & { clearDuplicateMessageCache?: () => void })
      .clearDuplicateMessageCache?.()
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

  it("preserves workforce delegation event types for agent execution links", async () => {
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
        timestamp: "2026-05-27T05:00:02Z",
        data: {
          event_id: "delegation-1",
          event_type: "workforce_delegation_start",
          data: {
            worker_task_id: "agent_20_564c4340",
            agent_name: "Editor Agent",
          },
        },
      })
    })

    await waitFor(() => {
      expect(screen.getByTestId("trace-events").textContent).toContain(
        "workforce_delegation_start"
      )
    })
  })

  it("keeps delegated child prompts and answers out of the parent chat", async () => {
    render(
      <AppProvider token="token">
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    for (const [eventType, content] of [
      ["user_message", "Delegated task instructions"],
      ["agent_message", "Child Agent clarification"],
      ["ai_message", "Child Agent final answer"],
    ] as const) {
      act(() => {
        onMessage?.({
          type: "trace_event",
          timestamp: "2026-05-27T05:00:03Z",
          data: {
            event_id: `child-${eventType}`,
            event_type: eventType,
            data: {
              source: "xagent-agent-tool-child",
              worker_task_id: "agent_20_run",
              message: content,
              content,
              role: eventType === "user_message" ? "user" : "assistant",
              display: "chat",
            },
          },
        })
      })
    }

    await waitFor(() => {
      const traceText = screen.getByTestId("trace-events").textContent || ""
      expect(traceText).toContain("user_message")
      expect(traceText).toContain("agent_message")
      expect(traceText).toContain("ai_message")
    })
    const messageText = screen.getByTestId("messages").textContent || ""
    expect(messageText).not.toContain("Delegated task instructions")
    expect(messageText).not.toContain("Child Agent clarification")
    expect(messageText).not.toContain("Child Agent final answer")
  })

  it("deduplicates the same user turn when history is replayed after reconnect", async () => {
    render(
      <AppProvider token="token">
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()
    const userTurn = {
      type: "trace_event",
      timestamp: "2026-05-27T05:00:00Z",
      data: {
        event_id: "user-event-1",
        event_type: "user_message",
        data: {
          message: "Repeated after reconnect",
          turn_id: "turn-1",
        },
      },
    }

    act(() => {
      onMessage?.(userTurn)
    })
    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "Repeated after reconnect"
      )
    })

    // A hot reload can reset the old content cache before the socket replays
    // the same persisted turn.
    act(() => {
      ;(window as typeof window & { clearDuplicateMessageCache?: () => void })
        .clearDuplicateMessageCache?.()
      onMessage?.(userTurn)
    })

    await waitFor(() => {
      const messages = JSON.parse(
        screen.getByTestId("messages").textContent || "[]"
      ) as Array<{ content: string }>
      expect(messages.filter(message => message.content === "Repeated after reconnect"))
        .toHaveLength(1)
    })
  })

  it("keeps identical text from distinct user turns", async () => {
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
          event_id: "user-event-1",
          event_type: "user_message",
          data: { message: "Send it again", turn_id: "turn-1" },
        },
      })
      onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:00:01Z",
        data: {
          event_id: "user-event-2",
          event_type: "user_message",
          data: { message: "Send it again", turn_id: "turn-2" },
        },
      })
    })

    await waitFor(() => {
      const messages = JSON.parse(
        screen.getByTestId("messages").textContent || "[]"
      ) as Array<{ content: string }>
      expect(messages.filter(message => message.content === "Send it again"))
        .toHaveLength(2)
    })
  })

  it("reconciles an optimistic send with its persisted user turn", async () => {
    render(
      <AppProvider token="token">
        <SeedExistingTask />
        <SendMessageProbe />
        <StateProbe />
      </AppProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("running")
    })

    fireEvent.click(screen.getByRole("button", { name: "Send message" }))

    await waitFor(() => {
      const messages = JSON.parse(
        screen.getByTestId("messages").textContent || "[]"
      ) as Array<{ content: string; isOptimistic?: boolean }>
      expect(messages).toEqual([
        expect.objectContaining({
          content: "Optimistic round trip",
          isOptimistic: true,
        }),
      ])
    })

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()
    act(() => {
      onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:00:02Z",
        data: {
          event_id: "user-event-optimistic",
          event_type: "user_message",
          data: {
            message: "Optimistic round trip",
            turn_id: "turn-optimistic",
          },
        },
      })
    })

    await waitFor(() => {
      const messages = JSON.parse(
        screen.getByTestId("messages").textContent || "[]"
      ) as Array<{ content: string; isOptimistic?: boolean }>
      expect(messages).toEqual([
        expect.objectContaining({
          content: "Optimistic round trip",
          isOptimistic: false,
        }),
      ])
    })
  })

  it("does not append an acknowledged optimistic message after switching tasks", async () => {
    let acknowledgeDelivery: (() => void) | undefined
    sendChatMessageMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          acknowledgeDelivery = () =>
            resolve({
              client_message_id: "turn-switch",
              turn_id: "turn-switch",
            })
        })
    )

    let send: (() => Promise<void>) | undefined
    let switchTask: (() => void) | undefined
    function SwitchingTaskProbe() {
      const { sendMessage, setTaskId } = useApp()
      send = () =>
        sendMessage("Message for task one", {
          clientMessageId: "turn-switch",
        })
      switchTask = () => setTaskId(2, { navigate: false })
      return null
    }

    render(
      <AppProvider token="token">
        <SeedExistingTask />
        <SwitchingTaskProbe />
        <StateProbe />
      </AppProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("running")
    })

    let delivery: Promise<void> | undefined
    await act(async () => {
      delivery = send?.()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(sendChatMessageMock).toHaveBeenCalledOnce()

    act(() => {
      switchTask?.()
    })
    await act(async () => {
      acknowledgeDelivery?.()
      await delivery
    })

    const messages = JSON.parse(
      screen.getByTestId("messages").textContent || "[]"
    ) as Array<{ content: string }>
    expect(messages).toEqual([])
  })

  it("shows the sender's message live when a new task's run dies before tracing", async () => {
    // A run refused at the quota gate returns before agent tracing starts, so
    // the live user_message trace event is never emitted. The sender's bubble
    // must come from the optimistic copy added once delivery is acknowledged —
    // without it the message only appears after a reload replays the transcript.
    apiRequestMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        task_id: 7,
        title: "hello quota",
        description: "hello quota",
        status: "pending",
      }),
    })

    let send: ((message: string) => Promise<void>) | undefined
    function CreateTaskProbe() {
      const { sendMessage } = useApp()
      send = (message: string) =>
        sendMessage(message, { clientMessageId: "turn-create" })
      return null
    }

    // The freshly created task's socket has not connected yet.
    wsHarness.isConnected = false
    render(
      <AppProvider token="token">
        <CreateTaskProbe />
        <StateProbe />
      </AppProvider>
    )

    let delivery: Promise<void> | undefined
    await act(async () => {
      delivery = send?.("hello quota")
      await new Promise((resolve) => setTimeout(resolve, 0))
    })

    // The socket for task 7 connects; the queued message is delivered and acked.
    await act(async () => {
      wsHarness.isConnected = true
      webSocketOptions.current?.onConnect?.()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    await act(async () => {
      await delivery
    })

    const messages = JSON.parse(
      screen.getByTestId("messages").textContent || "[]"
    ) as Array<{ role: string; content: string; isOptimistic?: boolean }>
    expect(messages).toEqual([
      expect.objectContaining({
        role: "user",
        content: "hello quota",
        isOptimistic: true,
      }),
    ])
  })

  it("does not crash on a trace event without data", () => {
    render(
      <AppProvider token="token">
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    expect(() => {
      act(() => {
        onMessage?.({
          type: "trace_event",
          event_type: "unknown_event",
          timestamp: "2026-05-27T05:00:02Z",
        } as TestWebSocketMessage)
      })
    }).not.toThrow()
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

  it("surfaces the failure reason from a failed task_completed payload", async () => {
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
    })

    // Quota-gate refusals never stream a message; the reason arrives only in the
    // terminal event's output. It must render live, not just after a reload.
    const quotaReason =
      "Team quota exhausted for this billing period."
    act(() => {
      onMessage?.({
        type: "task_completed",
        timestamp: "2026-05-27T05:00:02Z",
        task: {
          id: 1,
          status: "failed",
        },
        success: false,
        output: quotaReason,
      } as TestWebSocketMessage)
    })

    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("failed")
      expect(screen.getByTestId("messages").textContent).toContain(quotaReason)
    })
    // The bubble must be flagged as the turn's result; the conversation panel
    // filters out assistant messages without it, so an unflagged reason never
    // renders and the UI degrades to a generic "unknown error" until reload.
    const messages = JSON.parse(screen.getByTestId("messages").textContent || "[]")
    const failureBubble = messages.find((m: { content: string }) =>
      m.content.includes(quotaReason)
    )
    expect(failureBubble?.isResult).toBe(true)
    // Verbatim, no live-only prefix: reload replays the persisted transcript
    // row with this exact text, and the two views must match.
    expect(failureBubble?.content).toBe(quotaReason)
  })

  it("does not suppress a failure reason contained within the user's message", async () => {
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
    })

    act(() => {
      onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:00:01Z",
        data: {
          event_id: "user-event-with-reason",
          event_type: "user_message",
          data: {
            message: "Why did this quota failure happen?",
            turn_id: "turn-with-reason",
          },
        },
      })
    })

    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "Why did this quota failure happen?"
      )
    })

    act(() => {
      onMessage?.({
        type: "task_completed",
        timestamp: "2026-05-27T05:00:02Z",
        task: { id: 1, status: "failed" },
        success: false,
        output: "quota failure",
      } as TestWebSocketMessage)
    })

    await waitFor(() => {
      const messages = JSON.parse(
        screen.getByTestId("messages").textContent || "[]"
      ) as Array<{ role: string; content: string; isResult?: boolean }>
      expect(messages).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            role: "assistant",
            content: "quota failure",
            isResult: true,
          }),
        ])
      )
    })
  })

  it("emits a coded-error event for the app layer and still shows the reason", async () => {
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
    })

    const events: TaskErrorEventDetail[] = []
    const listener = (e: Event) => events.push((e as CustomEvent<TaskErrorEventDetail>).detail)
    window.addEventListener(TASK_ERROR_EVENT, listener)

    // Real coded-gate terminal events omit output/result; the reason rides in
    // error_details.message.
    const details = {
      code: "quota_exceeded",
      metric: "runs_per_month",
      limit: 0,
      message: "Team quota exhausted.",
    }
    act(() => {
      onMessage?.({
        type: "task_completed",
        timestamp: "2026-05-27T05:00:02Z",
        task: { id: 1, status: "failed" },
        success: false,
        error_code: "quota_exceeded",
        error_details: details,
      } as TestWebSocketMessage)
    })

    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("failed")
      // The code is handed to the app layer via the event (drives the dialog)...
      expect(events).toHaveLength(1)
    })
    expect(events[0].code).toBe("quota_exceeded")
    expect(events[0].details).toEqual(details)
    // ...and the reason (from error_details.message) still shows live in chat,
    // matching a page reload, instead of an empty "unknown error" turn.
    expect(screen.getByTestId("messages").textContent).toContain(
      "Team quota exhausted."
    )
    const codedMessages = JSON.parse(screen.getByTestId("messages").textContent || "[]")
    const codedBubble = codedMessages.find((m: { content: string }) =>
      m.content.includes("Team quota exhausted.")
    )
    expect(codedBubble?.isResult).toBe(true)
    expect(codedBubble?.content).toBe("Team quota exhausted.")

    window.removeEventListener(TASK_ERROR_EVENT, listener)
  })

  it("tags the coded-error event with the event's own task id, not the viewed one", async () => {
    // SeedExistingTask puts the viewer on task 1. A terminal event for a
    // different task (99) must attribute its dialog to 99 — using the
    // currently-viewed id would pop the dialog against the wrong task under a
    // reconnect/task-switch race.
    render(
      <AppProvider token="token">
        <SeedExistingTask />
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    const events: TaskErrorEventDetail[] = []
    const listener = (e: Event) => events.push((e as CustomEvent<TaskErrorEventDetail>).detail)
    window.addEventListener(TASK_ERROR_EVENT, listener)

    act(() => {
      onMessage?.({
        type: "task_completed",
        timestamp: "2026-05-27T05:00:02Z",
        task: { id: 99, status: "failed" },
        success: false,
        error_code: "quota_exceeded",
        error_details: { code: "quota_exceeded", limit: 0 },
      } as TestWebSocketMessage)
    })

    await waitFor(() => {
      expect(events).toHaveLength(1)
    })
    expect(events[0].taskId).toBe(99)

    window.removeEventListener(TASK_ERROR_EVENT, listener)
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

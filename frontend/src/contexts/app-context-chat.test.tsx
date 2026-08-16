import React from "react"
import { flushSync } from "react-dom"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

type TestWebSocketMessage = {
  type: string
  timestamp: string
  data?: unknown
  task_id?: number
  step_id?: string
  task?: Record<string, unknown>
  status?: string
  run_id?: string | null
  state_version?: number
  control_state?: string
}

const webSocketOptions = vi.hoisted(() => ({
  current: null as null | {
    connection?: {
      identity: string
      url: string
      protocols?: string[]
      expectedProtocol?: string
      taskId?: number
      chatTaskIdMode: "required" | "omit"
      credentialOwner: { kind: "external" }
    } | null
    deliveryGeneration?: number
    onConnectionClose?: (event: CloseEvent) => "handled" | "default"
    onConnectionFailure?: (failure: {
      recoverable: boolean
      error: Error
    }) => void
    onSessionConnectionClose?: (
      event: CloseEvent,
      connectionIdentity: string,
    ) => "handled" | "default"
    onSessionConnectionFailure?: (failure: {
      recoverable: boolean
      error: Error
    }, connectionIdentity: string) => void
    onMessage?: (message: TestWebSocketMessage) => void
    onConnect?: () => void
    uploadFiles?: unknown
    token?: string
  },
  all: [] as Array<{
    onMessage?: (message: TestWebSocketMessage) => void
    token?: string
  }>,
}))
const sendChatMessageMock = vi.hoisted(() => vi.fn())
const sendRawMessageMock = vi.hoisted(() => vi.fn())
const wsHarness = vi.hoisted(() => ({ isConnected: true }))
const apiRequestMock = vi.hoisted(() => vi.fn())
const routerPushMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api-wrapper")>()
  return {
    ...actual,
    apiRequest: (...args: unknown[]) => apiRequestMock(...args),
  }
})

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: routerPushMock, replace: vi.fn(), refresh: vi.fn() }),
  usePathname: () => "/",
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ token: "token" }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

vi.mock("@/hooks/use-websocket", () => ({
  useWebSocket: (options: {
    connection?: {
      identity: string
      url: string
      protocols?: string[]
      expectedProtocol?: string
      taskId?: number
      chatTaskIdMode: "required" | "omit"
      credentialOwner: { kind: "external" }
    } | null
    deliveryGeneration?: number
    onConnectionClose?: (event: CloseEvent) => "handled" | "default"
    onConnectionFailure?: (failure: {
      recoverable: boolean
      error: Error
    }) => void
    onSessionConnectionClose?: (
      event: CloseEvent,
      connectionIdentity: string,
    ) => "handled" | "default"
    onSessionConnectionFailure?: (failure: {
      recoverable: boolean
      error: Error
    }, connectionIdentity: string) => void
    onMessage?: (message: TestWebSocketMessage) => void
    onConnect?: () => void
    uploadFiles?: unknown
    token?: string
  }) => {
    webSocketOptions.current = options
    webSocketOptions.all.push(options)
    return {
      isConnected: wsHarness.isConnected,
      connectionError: null,
      sendMessage: sendRawMessageMock,
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
import { ChatStartScreen } from "@/components/chat/ChatStartScreen"
import { MarkdownRenderer } from "@/components/ui/markdown-renderer"
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
      <div data-testid="task-title">{state.currentTask?.title || ""}</div>
      <div data-testid="task-id">{state.taskId ?? ""}</div>
      <div data-testid="task-runtime-extensions">
        {JSON.stringify(state.taskRuntimeExtensions)}
      </div>
      <div data-testid="task-runtime-extension-bindings">
        {JSON.stringify(state.currentTask?.runtimeExtensionBindings || [])}
      </div>
      <div data-testid="steps-count">{state.steps.length}</div>
      <div data-testid="steps">{JSON.stringify(state.steps.map((step) => ({
        id: step.id,
        status: step.status,
        dependencies: step.dependencies,
        started_at: step.started_at,
        completed_at: step.completed_at,
      })))}</div>
      <div data-testid="dag-phase">{state.dagExecution?.phase ?? ""}</div>
      <div data-testid="history-loading">{String(state.isHistoryLoading)}</div>
      <div data-testid="preview-open">{String(state.filePreview.isOpen)}</div>
      <div data-testid="processing">{String(state.isProcessing)}</div>
    </>
  )
}

function MessageContentProbe() {
  const { state } = useApp()
  return (
    <div data-testid="message-content">
      {state.messages.map(message => (
        <React.Fragment key={message.id}>{message.content}</React.Fragment>
      ))}
    </div>
  )
}

function TaskRuntimeMetadataProbe() {
  const { dispatch, setTaskId } = useApp()
  return (
    <button type="button" onClick={() => {
      dispatch({
        type: "SET_CURRENT_TASK",
        payload: {
          id: "823",
          title: "Local browser task",
          status: "completed",
          description: "Inspect the selected window",
          createdAt: "2026-08-07T07:00:00Z",
          updatedAt: "2026-08-07T07:01:00Z",
          runtimeExtensionBindings: ["local_browser"],
        },
      })
      setTaskId(823, { navigate: false })
    }}>
      select task
    </button>
  )
}

function ScopedMessagesProbe({ testId }: { testId: string }) {
  const { state } = useApp()
  return <div data-testid={testId}>{JSON.stringify(state.messages.map(message => message.content))}</div>
}

type SessionControls = ReturnType<typeof useApp> & {
  startNewConversation: () => Promise<void>
  isConversationResetPending: boolean
  isMessageDeliveryPending: boolean
  sessionConversationState: string
}

let sessionControls: SessionControls | null = null

function SessionControlsProbe() {
  sessionControls = useApp() as SessionControls
  return (
    <>
      <div data-testid="reset-pending">
        {String(sessionControls.isConversationResetPending)}
      </div>
      <div data-testid="message-delivery-pending">
        {String(sessionControls.isMessageDeliveryPending)}
      </div>
      <div data-testid="session-conversation-state">
        {sessionControls.sessionConversationState}
      </div>
      <div data-testid="files-disabled">
        {String(sessionControls.filesDisabled)}
      </div>
      <div data-testid="voice-input-enabled">
        {String(sessionControls.voiceInputEnabled)}
      </div>
      <div data-testid="agent-cards-enabled">
        {String(sessionControls.agentCardsEnabled)}
      </div>
      <div data-testid="task-controls-enabled">
        {String(sessionControls.taskControlsEnabled)}
      </div>
    </>
  )
}

function TransportCapabilityConsumerProbe() {
  const { filesDisabled, voiceInputEnabled } = useApp()
  return (
    <ChatStartScreen
      title="Public support"
      onSend={vi.fn()}
      hideConfig
      filesDisabled={filesDisabled}
      voiceInputEnabled={voiceInputEnabled}
    />
  )
}

function getSessionControls(): SessionControls {
  if (!sessionControls) {
    throw new Error("Session controls are not mounted")
  }
  return sessionControls
}

const makeSessionConnection = (identity = "session-one") => ({
  identity,
  url: `wss://au.cloud.xagent.co/v1/widget/sessions/${identity}/ws`,
  protocols: ["xagent-session", `credential-${identity}`],
  expectedProtocol: "xagent-session",
  chatTaskIdMode: "omit" as const,
  credentialOwner: { kind: "external" as const },
})

const makeSessionTransport = (
  connection: ReturnType<typeof makeSessionConnection> | null = makeSessionConnection(),
) => ({
  uploadFiles: vi.fn(),
  session: {
    connection,
    onConnectionClose: () => "handled" as const,
    onConnectionFailure: vi.fn(),
    onConnectionOpen: vi.fn(),
    allowTasklessChat: true as const,
    supportsConversationReset: true as const,
    history: "none" as const,
    files: "disabled" as const,
    agentCards: "disabled" as const,
    voice: "disabled" as const,
    taskControls: "disabled" as const,
  },
})

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, reject, resolve }
}

const taskInfoMessage = (
  taskId: unknown,
  overrides: Record<string, unknown> = {},
): TestWebSocketMessage => ({
  type: "trace_event",
  timestamp: "2026-05-27T05:00:00Z",
  data: {
    event_id: `task-info-${String(taskId)}`,
    event_type: "task_info",
    data: {
      id: taskId,
      title: `Task ${String(taskId)}`,
      status: "running",
      description: "Session task",
      created_at: "2026-05-27T05:00:00Z",
      updated_at: "2026-05-27T05:00:00Z",
      ...overrides,
    },
  },
})

const assistantMessage = (
  content: string,
  taskId?: number,
): TestWebSocketMessage => ({
  type: "trace_event",
  timestamp: "2026-05-27T05:00:01Z",
  ...(taskId === undefined ? {} : { task_id: taskId }),
  data: {
    event_id: `assistant-${content}`,
    event_type: "agent_message",
    data: {
      message: content,
      content,
      role: "assistant",
    },
  },
})

const dagTraceMessage = (
  eventType: string,
  taskId: number,
  data: Record<string, unknown> = {},
): TestWebSocketMessage => ({
  type: "trace_event",
  timestamp: "2026-05-27T05:00:02Z",
  task_id: taskId,
  step_id: typeof data.step_id === "string" ? data.step_id : undefined,
  data: {
    event_id: `${eventType}-${taskId}-${String(data.step_id ?? "task")}`,
    event_type: eventType,
    data,
  },
})

const dagBurst = (taskId: number): TestWebSocketMessage[] => [
  dagTraceMessage("dag_plan_start", taskId, { phase: "planning" }),
  dagTraceMessage("dag_plan_end", taskId, {
    plan_id: `plan-${taskId}`,
    steps_count: 1,
    plan_data: {
      steps: [{
        id: "step-one",
        name: "Step one",
        dependencies: ["dependency-one"],
      }],
    },
  }),
  dagTraceMessage("dag_execute_start", taskId, { iteration: 1 }),
  dagTraceMessage("dag_step_start", taskId, {
    step_id: "step-one",
    step_name: "Step one",
    dependencies: ["dependency-one"],
    started_at: "2026-05-27T05:00:03Z",
  }),
  dagTraceMessage("dag_step_end", taskId, {
    step_id: "step-one",
    step_name: "Step one",
    completed_at: "2026-05-27T05:00:04Z",
  }),
  dagTraceMessage("dag_step_failed", taskId, {
    step_id: "step-one",
    step_name: "Step one",
    completed_at: "2026-05-27T05:00:05Z",
  }),
]

function SeedRunningTask() {
  const { dispatch } = useApp()

  React.useEffect(() => {
    // Real navigation always sets taskId alongside currentTask (setTaskId,
    // ADOPT_SESSION_TASK, and every builder/workforce dispatch site do both
    // together) - seed both here too, matching SeedExistingTask below, so
    // this fixture doesn't exercise a "viewing task 1 but taskId is null"
    // shape that never happens in production.
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
    webSocketOptions.all = []
    sessionControls = null
    wsHarness.isConnected = true
    apiRequestMock.mockReset()
    routerPushMock.mockReset()
    sendRawMessageMock.mockReset()
    sendRawMessageMock.mockReturnValue("sent")
    sendChatMessageMock.mockReset()
    sendChatMessageMock.mockResolvedValue({
      client_message_id: "turn-optimistic",
      turn_id: "turn-optimistic",
    })
    localStorage.clear()
    ;(window as typeof window & { clearDuplicateMessageCache?: () => void })
      .clearDuplicateMessageCache?.()
  })

  afterEach(() => {
    cleanup()
    localStorage.clear()
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

  it("rejects a queued message once the conversation is reset before delivery", async () => {
    let send: (() => Promise<void> | undefined) | undefined
    let reset: (() => void) | undefined
    function ResetProbe() {
      const { sendMessage, setTaskId } = useApp()
      send = () => {
        // Mirrors the widget's bootstrap: taskId is set in the same tick the
        // opening message is queued for it.
        setTaskId(9, { navigate: false })
        return sendMessage("hello there", {
          clientMessageId: "turn-reset",
          targetTaskId: 9,
        })
      }
      reset = () => setTaskId(null, { navigate: false })
      return null
    }

    // The new task's socket never connects, so the message stays queued.
    wsHarness.isConnected = false
    render(
      <AppProvider token="token">
        <ResetProbe />
      </AppProvider>
    )

    let deliveryError: Error | undefined
    let delivery: Promise<void | undefined> | undefined
    await act(async () => {
      delivery = Promise.resolve(send?.()).catch((error: Error) => {
        deliveryError = error
        return undefined
      })
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(sendChatMessageMock).not.toHaveBeenCalled()
    expect(deliveryError).toBeUndefined()

    // "New conversation" nulls the taskId: the queued message must fail now,
    // not sit out the 30s timeout as an unhandled rejection.
    await act(async () => {
      reset?.()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    await act(async () => {
      await delivery
    })

    expect(deliveryError?.message).toMatch(/reset/)
    expect(sendChatMessageMock).not.toHaveBeenCalled()
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

  it("sends selected task runtime extensions in the create request", async () => {
    let createBody: Record<string, unknown> | undefined
    apiRequestMock.mockImplementation(
      async (url: string, options?: RequestInit) => {
        if (url.endsWith("/api/chat/task/create")) {
          createBody = JSON.parse(String(options?.body || "{}"))
          return {
            ok: true,
            json: async () => ({
              task_id: 8,
              title: "inspect browser",
              description: "inspect browser",
              status: "pending",
            }),
          }
        }
        return { ok: true, json: async () => ({}) }
      },
    )

    let send: (() => Promise<void>) | undefined
    function RuntimeExtensionProbe() {
      const { sendMessage } = useApp()
      send = () => sendMessage("inspect browser", {
        clientMessageId: "turn-local-browser",
        runtimeExtensions: { local_browser: {} },
      })
      return null
    }

    wsHarness.isConnected = false
    render(
      <AppProvider token="token">
        <RuntimeExtensionProbe />
      </AppProvider>
    )

    let delivery: Promise<void> | undefined
    await act(async () => {
      delivery = send?.()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })

    expect(createBody).toEqual(
      expect.objectContaining({
        runtime_extensions: { local_browser: {} },
      })
    )

    await act(async () => {
      wsHarness.isConnected = true
      webSocketOptions.current?.onConnect?.()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    await act(async () => {
      await delivery
    })
  })

  it("loads task runtime metadata for history replay", async () => {
    apiRequestMock.mockImplementation(async (url: string) => {
      if (url.endsWith("/api/chat/task/823/runtime-extensions")) {
        return {
          ok: true,
          json: async () => ({
            task_id: 823,
            runtime_extensions: {
              local_browser: { kind: "local_browser" },
            },
          }),
        }
      }
      return { ok: true, json: async () => ({}) }
    })

    render(
      <AppProvider token="token">
        <TaskRuntimeMetadataProbe />
        <StateProbe />
      </AppProvider>,
    )

    fireEvent.click(screen.getByRole("button", { name: "select task" }))

    await waitFor(() => {
      expect(screen.getByTestId("task-runtime-extensions")).toHaveTextContent(
        JSON.stringify({ local_browser: { kind: "local_browser" } }),
      )
    })
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
    // This task never entered DAG mode (no dagExecution was ever seeded) -
    // a failed non-DAG task completion must not fabricate one just because
    // the completion payload includes a status.
    expect(screen.getByTestId("dag-phase").textContent).toBe("")
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

  it("drops DAG/chat/task events for a background task while viewing a different one", async () => {
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    // A background task (2) is still running while the viewer stays on task
    // 1 (seeded by SeedRunningTask) - its DAG burst, chat reply, and task_info
    // must not repaint task 1's view.
    act(() => {
      dagBurst(2).forEach((message) => onMessage?.(message))
      onMessage?.(assistantMessage("Stray reply from task 2", 2))
      // Unlike the shared taskInfoMessage() helper (which several
      // session-adoption tests rely on omitting), a real task_info frame for
      // task 2 also carries a top-level task_id (see
      // ws_trace_handlers.py's create_stream_event) - set it explicitly here
      // so this exercises the same envelope shape the guard actually checks.
      onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 2,
        data: {
          event_id: "task-info-2",
          event_type: "task_info",
          data: {
            id: 2,
            title: "Task 2",
            status: "completed",
            description: "Session task",
            created_at: "2026-05-27T05:00:00Z",
            updated_at: "2026-05-27T05:00:00Z",
          },
        },
      })
      // A real task_completed broadcast nests the task id at task.id, and
      // useWebSocket's normalizer (use-websocket.ts) lifts it into a
      // top-level task_id - exercise that exact shape too, not just the
      // trace_event envelope above. Also carries a file output that would
      // auto-open the file preview (dispatchAutoOpenPreview) if the stray
      // dispatch weren't dropped - OPEN_FILE_PREVIEW must be task-scoped too.
      onMessage?.({
        type: "task_completed",
        timestamp: "2026-05-27T05:00:03Z",
        task_id: 2,
        task: { id: 2, status: "completed" },
        success: true,
        // Must be object-shaped ({file_id, filename}), matching what the
        // real backend broadcast sends (websocket.py's normalized_outputs) -
        // normalizeGeneratedPreviewFiles filters out any entry without a
        // file_id, so a bare string here would silently never reach
        // dispatchAutoOpenPreview at all, making the assertion below pass
        // even if the OPEN_FILE_PREVIEW guard were broken.
        file_outputs: [{ file_id: "task-2-report", filename: "report.pdf" }],
      } as TestWebSocketMessage)
    })
    await waitFor(() => {
      // dagBurst's dag_step_failed would otherwise flip this to "failed".
      expect(screen.getByTestId("dag-phase").textContent).toBe("")
    })
    expect(screen.getByTestId("steps-count").textContent).toBe("0")
    expect(screen.getByTestId("task-status").textContent).toBe("running")
    expect(screen.getByTestId("task-title").textContent).toBe("Test task")
    expect(screen.getByTestId("messages").textContent).not.toContain("Stray reply from task 2")
    expect(screen.getByTestId("preview-open").textContent).toBe("false")

    // The viewed task's own events must still apply normally - the guard
    // isn't blanket-dropping DAG/chat state, only cross-task events.
    act(() => {
      dagBurst(1).forEach((message) => onMessage?.(message))
    })
    await waitFor(() => {
      expect(screen.getByTestId("dag-phase").textContent).toBe("failed")
    })
    expect(screen.getByTestId("steps-count").textContent).toBe("1")
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

  it("retains persisted runtime-extension bindings from task info", async () => {
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )

    act(() => {
      webSocketOptions.current?.onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:00:02Z",
        data: {
          event_id: "task-info-runtime-1",
          event_type: "task_info",
          data: {
            id: 1,
            title: "Test task",
            description: "Test task",
            status: "COMPLETED",
            created_at: "2026-05-27T05:00:00Z",
            updated_at: "2026-05-27T05:00:02Z",
            runtime_extension_bindings: ["local_browser"],
          },
        },
      })
    })

    await waitFor(() => {
      expect(
        screen.getByTestId("task-runtime-extension-bindings"),
      ).toHaveTextContent(JSON.stringify(["local_browser"]))
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

  it("passes the persistent Session transport contract and sends the first taskless turn only through the socket", async () => {
    const transport = makeSessionTransport()
    const acknowledgement = deferred<{
      client_message_id: string
      turn_id: string
    }>()
    sendChatMessageMock.mockReturnValueOnce(acknowledgement.promise)

    render(
      <AppProvider token="token" transport={transport}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )

    expect(webSocketOptions.current?.connection).toEqual(
      transport.session.connection
    )
    expect(webSocketOptions.current?.deliveryGeneration).toBe(0)
    expect(webSocketOptions.current?.onSessionConnectionClose).toBe(
      transport.session.onConnectionClose
    )
    expect(webSocketOptions.current?.onSessionConnectionFailure).toBe(
      transport.session.onConnectionFailure
    )
    act(() => webSocketOptions.current?.onConnect?.())
    expect(transport.session.onConnectionOpen).toHaveBeenCalledWith(
      transport.session.connection?.identity,
    )
    expect(webSocketOptions.current?.uploadFiles).toEqual(expect.any(Function))

    let delivery!: Promise<void>
    act(() => {
      delivery = getSessionControls().sendMessage("First Session turn", {
        clientMessageId: "session-turn-1",
      })
    })

    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(sendChatMessageMock).toHaveBeenCalledWith(
      "First Session turn",
      undefined,
      undefined,
      "session-turn-1"
    )
    expect(screen.getByTestId("messages").textContent).not.toContain(
      "First Session turn"
    )

    await act(async () => {
      acknowledgement.resolve({
        client_message_id: "session-turn-1",
        turn_id: "session-turn-1",
      })
      await delivery
    })

    expect(screen.getByTestId("messages").textContent).toContain(
      "First Session turn"
    )
    expect(apiRequestMock).not.toHaveBeenCalled()
  })

  it("applies explicit public transport capabilities without disabling public files", () => {
    const ambientAuthCache = JSON.stringify({
      schemaVersion: 2,
      sessionId: "personal-session",
      credentialRevision: 0,
      profileRevision: 0,
      user: { id: "personal-user", username: "personal" },
      token: "personal-access-token",
      refreshToken: "personal-refresh-token",
      timestamp: Date.now(),
    })
    localStorage.setItem("auth_cache", ambientAuthCache)

    render(
      <AppProvider
        token="public-token"
        transport={{
          capabilities: {
            agentCards: "disabled",
            voice: "disabled",
          },
          fileAccess: {
            previewUrl: (fileId) => `/api/widget/files/preview/${fileId}`,
            downloadUrl: (fileId) => `/api/widget/files/download/${fileId}`,
            inlinePreviewUrl: (fileId) => `/api/widget/files/public/preview/${fileId}`,
            inlineDownloadUrl: (fileId) => `/api/widget/files/public/download/${fileId}`,
            relativePreviewUrl: (fileId, relativePath) =>
              `/api/widget/files/public/preview/${fileId}?relative_path=${encodeURIComponent(relativePath)}`,
            request: vi.fn(),
          },
          uploadFiles: vi.fn(),
        }}
      >
        <SessionControlsProbe />
        <TransportCapabilityConsumerProbe />
        <MarkdownRenderer content="[Specialist](agent://42)" />
      </AppProvider>
    )

    expect(screen.getByTestId("files-disabled")).toHaveTextContent("false")
    expect(screen.getByTestId("voice-input-enabled")).toHaveTextContent("false")
    expect(screen.getByTestId("agent-cards-enabled")).toHaveTextContent("false")
    expect(screen.getByText("Specialist")).not.toHaveAttribute("data-agent-id")
    expect(screen.queryByLabelText("voiceInput.start")).not.toBeInTheDocument()
    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(localStorage.getItem("auth_cache")).toBe(ambientAuthCache)
  })

  it("renders Session events live and adopts task_info without legacy navigation or conversation clearing", async () => {
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )

    act(() => {
      webSocketOptions.current?.onConnect?.()
    })
    expect(screen.getByTestId("history-loading").textContent).toBe("false")

    act(() => {
      const tasklessLiveEvent = assistantMessage("Live before task_info")
      webSocketOptions.current?.onMessage?.({
        ...tasklessLiveEvent,
        task_id: undefined,
        task: { id: undefined },
        data: {
          ...(tasklessLiveEvent.data as object),
          task_id: undefined,
          task: { id: undefined },
        },
      })
    })
    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "Live before task_info"
      )
    })

    for (const invalidId of [0, -1, true, ""] as const) {
      act(() => {
        webSocketOptions.current?.onMessage?.(taskInfoMessage(invalidId))
      })
      expect(screen.getByTestId("task-id").textContent).toBe("")
    }

    act(() => {
      webSocketOptions.current?.onMessage?.(taskInfoMessage("37"))
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-id").textContent).toBe("37")
    })
    expect(screen.getByTestId("messages").textContent).toContain(
      "Live before task_info"
    )
    expect(routerPushMock).not.toHaveBeenCalled()

    act(() => {
      webSocketOptions.current?.onMessage?.({
        ...assistantMessage("Null-bound frame"),
        task_id: null,
      } as unknown as TestWebSocketMessage)
      webSocketOptions.current?.onMessage?.({
        ...assistantMessage("Boolean-bound frame"),
        task_id: true,
      } as unknown as TestWebSocketMessage)
      webSocketOptions.current?.onMessage?.({
        ...assistantMessage("Conflicting-bound frame", 37),
        data: {
          ...(assistantMessage("Conflicting-bound frame", 37).data as object),
          task_id: 38,
        },
      })
    })
    expect(screen.getByTestId("messages").textContent).not.toContain(
      "Null-bound frame"
    )
    expect(screen.getByTestId("messages").textContent).not.toContain(
      "Boolean-bound frame"
    )
    expect(screen.getByTestId("messages").textContent).not.toContain(
      "Conflicting-bound frame"
    )
  })

  it("preserves Session conversation on connection refresh and rejects callbacks from the replaced socket", async () => {
    const firstTransport = makeSessionTransport(
      makeSessionConnection("session-before-refresh")
    )
    const { rerender } = render(
      <AppProvider token="token" transport={firstTransport}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )

    const oldOnMessage = webSocketOptions.current?.onMessage
    act(() => {
      oldOnMessage?.(taskInfoMessage(41))
      oldOnMessage?.(assistantMessage("Preserved answer", 41))
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-id").textContent).toBe("41")
      expect(screen.getByTestId("messages").textContent).toContain(
        "Preserved answer"
      )
    })

    const refreshedTransport = makeSessionTransport(
      makeSessionConnection("session-after-refresh")
    )
    rerender(
      <AppProvider token="token" transport={refreshedTransport}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    act(() => {
      webSocketOptions.current?.onConnect?.()
    })

    expect(screen.getByTestId("task-id").textContent).toBe("41")
    expect(screen.getByTestId("messages").textContent).toContain(
      "Preserved answer"
    )
    expect(screen.getByTestId("history-loading").textContent).toBe("false")

    act(() => {
      oldOnMessage?.(assistantMessage("Stale socket answer", 41))
      webSocketOptions.current?.onMessage?.(
        assistantMessage("Current socket answer", 41)
      )
    })
    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "Current socket answer"
      )
    })
    expect(screen.getByTestId("messages").textContent).not.toContain(
      "Stale socket answer"
    )
  })

  it("rebinds a bound Session to a refreshed connection before a later reset", async () => {
    const { rerender } = render(
      <AppProvider token="token" transport={makeSessionTransport(makeSessionConnection("rebind-old"))}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    act(() => webSocketOptions.current?.onMessage?.(taskInfoMessage(83)))
    rerender(
      <AppProvider token="token" transport={makeSessionTransport(null)}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    rerender(
      <AppProvider token="token" transport={makeSessionTransport(makeSessionConnection("rebind-new"))}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    const reset = getSessionControls().startNewConversation()
    await act(async () => {
      webSocketOptions.current?.onMessage?.({ type: "conversation_reset", timestamp: "2026-05-27T05:00:03Z", data: {} })
      await reset
    })
    await getSessionControls().sendMessage("Replacement", { clientMessageId: "rebind-replacement" })
    act(() => webSocketOptions.current?.onMessage?.(taskInfoMessage(84)))
    await waitFor(() => expect(screen.getByTestId("task-id").textContent).toBe("84"))
  })

  it("serializes Session reset, clears only on the owned acknowledgement, and gates the next task lineage", async () => {
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    const physicalOnMessage = webSocketOptions.current?.onMessage
    expect(physicalOnMessage).toBeDefined()

    act(() => {
      physicalOnMessage?.(taskInfoMessage(51))
      physicalOnMessage?.(
        assistantMessage("Old conversation", 51)
      )
      getSessionControls().dispatch({
        type: "SET_STEPS",
        payload: [{
          id: "old-step",
          name: "Old step",
          description: "Old step",
          status: "running",
          dependencies: [],
        }],
      })
      getSessionControls().dispatch({
        type: "SET_TRACE_EVENTS",
        payload: [{
          event_id: "old-trace",
          event_type: "tool_call",
          timestamp: "2026-05-27T05:00:02Z",
          data: {},
        }],
      })
      getSessionControls().dispatch({ type: "SET_PROCESSING", payload: true })
      getSessionControls().dispatch({
        type: "OPEN_FILE_PREVIEW",
        payload: { fileId: "old-file", fileName: "old.pdf" },
      })
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-id").textContent).toBe("51")
      expect(screen.getByTestId("steps-count").textContent).toBe("1")
      expect(screen.getByTestId("preview-open").textContent).toBe("true")
    })

    let firstReset!: Promise<void>
    let duplicateReset!: Promise<void>
    const preResetState = getSessionControls().state
    act(() => {
      firstReset = getSessionControls().startNewConversation()
      duplicateReset = getSessionControls().startNewConversation()
    })
    expect(duplicateReset).toBe(firstReset)
    expect(sendRawMessageMock).toHaveBeenCalledTimes(1)
    expect(sendRawMessageMock).toHaveBeenCalledWith({
      type: "new_conversation",
    })
    expect(screen.getByTestId("reset-pending").textContent).toBe("true")
    expect(screen.getByTestId("messages").textContent).toContain(
      "Old conversation"
    )
    await expect(
      getSessionControls().sendMessage("Blocked during reset")
    ).rejects.toThrow(/reset/i)
    expect(sendChatMessageMock).not.toHaveBeenCalled()

    await act(async () => {
      physicalOnMessage?.({
        type: "conversation_reset",
        timestamp: "2026-05-27T05:00:03Z",
        data: {},
      })
      await firstReset
    })

    expect(screen.getByTestId("reset-pending").textContent).toBe("false")
    expect(screen.getByTestId("task-id").textContent).toBe("")
    expect(screen.getByTestId("messages").textContent).toBe("[]")
    expect(screen.getByTestId("trace-events").textContent).toBe("[]")
    expect(screen.getByTestId("steps-count").textContent).toBe("0")
    expect(screen.getByTestId("preview-open").textContent).toBe("false")
    expect(screen.getByTestId("processing").textContent).toBe("false")
    expect(getSessionControls().state.messages).not.toBe(preResetState.messages)
    expect(getSessionControls().state.traceEvents).not.toBe(preResetState.traceEvents)
    expect(getSessionControls().state.filePreview).not.toBe(preResetState.filePreview)
    expect(getSessionControls().state.filePreview.availableFiles).not.toBe(
      preResetState.filePreview.availableFiles,
    )
    expect(webSocketOptions.current?.deliveryGeneration).toBe(1)

    act(() => {
      physicalOnMessage?.(
        assistantMessage("Retired task output", 51)
      )
      physicalOnMessage?.(taskInfoMessage(52))
      physicalOnMessage?.(
        assistantMessage("Premature new task output", 52)
      )
    })
    expect(screen.getByTestId("task-id").textContent).toBe("")
    expect(screen.getByTestId("messages").textContent).not.toContain(
      "Retired task output"
    )
    expect(screen.getByTestId("messages").textContent).not.toContain(
      "Premature new task output"
    )

    await act(async () => {
      await getSessionControls().sendMessage("Start replacement conversation", {
        clientMessageId: "session-turn-after-reset",
      })
    })
    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(sendChatMessageMock).toHaveBeenLastCalledWith(
      "Start replacement conversation",
      undefined,
      undefined,
      "session-turn-after-reset"
    )

    act(() => {
      physicalOnMessage?.(taskInfoMessage(52))
      physicalOnMessage?.(
        assistantMessage("Replacement answer", 52)
      )
      physicalOnMessage?.(
        assistantMessage("Late retired answer", 51)
      )
      physicalOnMessage?.(
        assistantMessage("Wrong task answer", 99)
      )
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-id").textContent).toBe("52")
      expect(screen.getByTestId("messages").textContent).toContain(
        "Replacement answer"
      )
    })
    expect(screen.getByTestId("messages").textContent).not.toContain(
      "Late retired answer"
    )
    expect(screen.getByTestId("messages").textContent).not.toContain(
      "Wrong task answer"
    )
  })

  it("allows only one first replacement-conversation send until task_info adoption", async () => {
    const firstAcknowledgement = deferred<{
      client_message_id: string
      turn_id: string
    }>()
    sendChatMessageMock
      .mockReturnValueOnce(firstAcknowledgement.promise)
      .mockImplementationOnce(() => (
        firstAcknowledgement.promise.then(() => {
          throw new Error("Later concurrent delivery failed")
        })
      ))
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    const physicalOnMessage = webSocketOptions.current?.onMessage
    act(() => {
      physicalOnMessage?.(taskInfoMessage(54))
    })

    const reset = getSessionControls().startNewConversation()
    await act(async () => {
      physicalOnMessage?.({
        type: "conversation_reset",
        timestamp: "2026-05-27T05:00:03Z",
        data: {},
      })
      await reset
    })

    const acceptedSend = getSessionControls().sendMessage(
      "Accepted replacement turn",
      { clientMessageId: "accepted-replacement-turn" },
    )
    const concurrentOutcome = getSessionControls().sendMessage(
      "Concurrent replacement turn",
      { clientMessageId: "concurrent-replacement-turn" },
    ).then(
      () => "resolved",
      error => error instanceof Error ? error.message : String(error),
    )

    await act(async () => {
      firstAcknowledgement.resolve({
        client_message_id: "accepted-replacement-turn",
        turn_id: "accepted-replacement-turn",
      })
      await acceptedSend
      await concurrentOutcome
    })

    expect(sendChatMessageMock).toHaveBeenCalledTimes(1)
    await expect(concurrentOutcome).resolves.toMatch(
      /replacement|conversation|pending/i
    )
    expect(apiRequestMock).not.toHaveBeenCalled()

    act(() => {
      physicalOnMessage?.(taskInfoMessage(55))
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-id").textContent).toBe("55")
    })
  })

  it("ignores a late replacement acceptance after its connection owner changes", async () => {
    const acknowledgement = deferred<{ client_message_id: string; turn_id: string }>()
    sendChatMessageMock.mockReturnValueOnce(acknowledgement.promise)
    const { rerender } = render(
      <AppProvider token="token" transport={makeSessionTransport(makeSessionConnection("late-owner-old"))}>
        <SessionControlsProbe />
      </AppProvider>
    )
    act(() => webSocketOptions.current?.onMessage?.(taskInfoMessage(86)))
    const reset = getSessionControls().startNewConversation()
    await act(async () => {
      webSocketOptions.current?.onMessage?.({ type: "conversation_reset", timestamp: "2026-05-27T05:00:03Z", data: {} })
      await reset
    })
    const replacement = getSessionControls().sendMessage("Late acceptance")
    rerender(
      <AppProvider token="token" transport={makeSessionTransport(makeSessionConnection("late-owner-new"))}>
        <SessionControlsProbe />
      </AppProvider>
    )
    await act(async () => {
      acknowledgement.resolve({ client_message_id: "late", turn_id: "late" })
      await replacement
    })
    expect(screen.getByTestId("session-conversation-state").textContent).toBe("reload_required")
  })

  it("does not reopen replacement state when a late replacement rejection settles after unmount", async () => {
    const acknowledgement = deferred<{ client_message_id: string; turn_id: string }>()
    sendChatMessageMock.mockReturnValueOnce(acknowledgement.promise)
    const { unmount } = render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
      </AppProvider>
    )
    act(() => webSocketOptions.current?.onMessage?.(taskInfoMessage(87)))
    const reset = getSessionControls().startNewConversation()
    await act(async () => {
      webSocketOptions.current?.onMessage?.({ type: "conversation_reset", timestamp: "2026-05-27T05:00:03Z", data: {} })
      await reset
    })
    const replacement = getSessionControls().sendMessage("Late rejection")
    unmount()
    acknowledgement.reject(new Error("late rejection"))
    await expect(replacement).rejects.toThrow("late rejection")
  })

  it("rejects Session reset while a durable message delivery is pending", async () => {
    const acknowledgement = deferred<{
      client_message_id: string
      turn_id: string
    }>()
    sendChatMessageMock.mockReturnValueOnce(acknowledgement.promise)
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )

    let delivery!: Promise<void>
    act(() => {
      delivery = getSessionControls().sendMessage("Still delivering", {
        clientMessageId: "pending-delivery",
      })
    })
    expect(
      screen.getByTestId("message-delivery-pending").textContent
    ).toBe("true")
    await expect(
      getSessionControls().startNewConversation()
    ).rejects.toThrow(/delivery|message/i)
    expect(sendRawMessageMock).not.toHaveBeenCalled()

    await act(async () => {
      acknowledgement.resolve({
        client_message_id: "pending-delivery",
        turn_id: "pending-delivery",
      })
      await delivery
    })
    expect(
      screen.getByTestId("message-delivery-pending").textContent
    ).toBe("false")
    act(() => {
      webSocketOptions.current?.onMessage?.(taskInfoMessage(58))
    })

    let reset!: Promise<void>
    act(() => {
      reset = getSessionControls().startNewConversation()
    })
    expect(sendRawMessageMock).toHaveBeenCalledWith({
      type: "new_conversation",
    })
    await act(async () => {
      webSocketOptions.current?.onMessage?.({
        type: "conversation_reset",
        timestamp: "2026-05-27T05:00:03Z",
        data: {},
      })
      await reset
    })
  })

  it("rejects reset before the Session has adopted an established task", async () => {
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )

    let reset!: Promise<void>
    act(() => {
      reset = getSessionControls().startNewConversation()
    })
    const resetOutcome = reset.then(
      () => "resolved",
      error => error instanceof Error ? error.message : String(error),
    )
    await expect(Promise.race([
      resetOutcome,
      new Promise<string>(resolve => {
        window.setTimeout(() => resolve("timeout"), 20)
      }),
    ])).resolves.toMatch(/established|task/i)
  })

  it("fails closed when the reset command throws synchronously instead of stranding controls", async () => {
    sendRawMessageMock.mockImplementationOnce(() => {
      throw new Error("socket write failed")
    })
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
      </AppProvider>
    )
    act(() => webSocketOptions.current?.onMessage?.(taskInfoMessage(85)))
    await expect(getSessionControls().startNewConversation()).rejects.toThrow("socket write failed")
    expect(screen.getByTestId("session-conversation-state").textContent).toBe("reload_required")
    await expect(getSessionControls().startNewConversation()).rejects.toThrow(/reload required/i)
  })

  it("keeps an unsent reset bound and lets the caller retry without retiring its task", async () => {
    sendRawMessageMock.mockReturnValueOnce("not_sent")
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    act(() => webSocketOptions.current?.onMessage?.(taskInfoMessage(88)))

    const unsentReset = getSessionControls().startNewConversation()
    void unsentReset.catch(() => {})

    await waitFor(() => {
      expect(screen.getByTestId("session-conversation-state").textContent).toBe("bound")
      expect(screen.getByTestId("reset-pending").textContent).toBe("false")
    })
    await expect(unsentReset).rejects.toThrow(/not sent|retry/i)
    expect(screen.getByTestId("task-id").textContent).toBe("88")

    const retry = getSessionControls().startNewConversation()
    await act(async () => {
      webSocketOptions.current?.onMessage?.({
        type: "conversation_reset",
        timestamp: "2026-05-27T05:00:03Z",
        data: {},
      })
      await retry
    })
  })

  it("commits an early replacement task candidate and its ordered frames only after delivery acceptance", async () => {
    const acknowledgement = deferred<{ client_message_id: string; turn_id: string }>()
    sendChatMessageMock.mockReturnValueOnce(acknowledgement.promise)
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    act(() => onMessage?.(taskInfoMessage(89)))
    const reset = getSessionControls().startNewConversation()
    await act(async () => {
      onMessage?.({ type: "conversation_reset", timestamp: "2026-05-27T05:00:03Z", data: {} })
      await reset
    })

    const replacement = getSessionControls().sendMessage("replacement", {
      clientMessageId: "early-replacement",
    })
    act(() => {
      onMessage?.(taskInfoMessage(90))
      onMessage?.(assistantMessage("Early ordered output", 90))
    })
    expect(screen.getByTestId("task-id").textContent).toBe("")
    expect(screen.getByTestId("messages").textContent).not.toContain("Early ordered output")

    await act(async () => {
      acknowledgement.resolve({ client_message_id: "early-replacement", turn_id: "early-replacement" })
      await replacement
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-id").textContent).toBe("90")
      expect(screen.getByTestId("messages").textContent).toContain("Early ordered output")
    })
  })

  it("requires reload when a pre-acceptance candidate accompanies an untyped replacement failure", async () => {
    const acknowledgement = deferred<{ client_message_id: string; turn_id: string }>()
    sendChatMessageMock.mockReturnValueOnce(acknowledgement.promise)
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    act(() => onMessage?.(taskInfoMessage(91)))
    const reset = getSessionControls().startNewConversation()
    await act(async () => {
      onMessage?.({ type: "conversation_reset", timestamp: "2026-05-27T05:00:03Z", data: {} })
      await reset
    })

    const replacement = getSessionControls().sendMessage("rejected replacement", {
      clientMessageId: "rejected-replacement",
    })
    act(() => {
      onMessage?.(taskInfoMessage(92))
      onMessage?.(assistantMessage("Rejected candidate output", 92))
    })
    await act(async () => {
      acknowledgement.reject(new Error("delivery rejected"))
      await expect(replacement).rejects.toThrow("delivery rejected")
    })

    expect(screen.getByTestId("session-conversation-state").textContent).toBe("reload_required")
    expect(screen.getByTestId("task-id").textContent).toBe("")
    expect(screen.getByTestId("messages").textContent).not.toContain("Rejected candidate output")
  })

  it.each([
    ["not_sent", false, "replacement_ready"],
    ["rejected", false, "replacement_ready"],
    ["outcome_unknown", false, "reload_required"],
    ["not_sent", true, "reload_required"],
    ["rejected", true, "reload_required"],
    ["outcome_unknown", true, "reload_required"],
  ] as const)("settles replacement %s with candidate=%s as %s", async (disposition, candidatePresent, expectedPhase) => {
    const acknowledgement = deferred<{ client_message_id: string; turn_id: string }>()
    sendChatMessageMock.mockReturnValueOnce(acknowledgement.promise)
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    act(() => onMessage?.(taskInfoMessage(190)))
    const reset = getSessionControls().startNewConversation()
    await act(async () => {
      onMessage?.({ type: "conversation_reset", timestamp: "2026-05-27T05:00:03Z", data: {} })
      await reset
    })

    const replacement = getSessionControls().sendMessage("replacement", {
      clientMessageId: `replacement-${disposition}-${String(candidatePresent)}`,
    })
    if (candidatePresent) {
      act(() => onMessage?.(taskInfoMessage(191)))
    }
    const error = Object.assign(new Error(`replacement ${disposition}`), { disposition })
    await act(async () => {
      acknowledgement.reject(error)
      await expect(replacement).rejects.toBe(error)
    })

    expect(screen.getByTestId("session-conversation-state").textContent).toBe(expectedPhase)
  })

  it("fails closed when a bound Session receives a different task id", async () => {
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
      </AppProvider>
    )
    act(() => {
      webSocketOptions.current?.onMessage?.(taskInfoMessage(93))
      webSocketOptions.current?.onMessage?.(taskInfoMessage(94))
    })
    expect(screen.getByTestId("session-conversation-state").textContent).toBe("reload_required")
  })

  it("keeps reload-required absorbing across generic state resets", async () => {
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
      </AppProvider>
    )
    act(() => {
      webSocketOptions.current?.onMessage?.(taskInfoMessage(95))
      webSocketOptions.current?.onMessage?.(taskInfoMessage(96))
      getSessionControls().dispatch({ type: "RESET_STATE" })
    })
    expect(screen.getByTestId("session-conversation-state").textContent).toBe("reload_required")
  })

  it("drops retired frames before replacement buffering can consume its frame budget", async () => {
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    act(() => onMessage?.(taskInfoMessage(97)))
    const reset = getSessionControls().startNewConversation()
    await act(async () => {
      onMessage?.({ type: "conversation_reset", timestamp: "2026-05-27T05:00:03Z", data: {} })
      await reset
    })
    await getSessionControls().sendMessage("replacement", { clientMessageId: "retired-budget" })

    act(() => {
      for (let index = 0; index < 65; index += 1) {
        onMessage?.(assistantMessage(`Retired ${index}`, 97))
      }
      onMessage?.(taskInfoMessage(98))
      onMessage?.(assistantMessage("Current replacement output", 98))
    })
    await waitFor(() => {
      expect(screen.getByTestId("session-conversation-state").textContent).toBe("bound")
      expect(screen.getByTestId("task-id").textContent).toBe("98")
      expect(screen.getByTestId("messages").textContent).toContain("Current replacement output")
    })
  })

  it("projects every normal Session DAG burst frame against the prior frame", async () => {
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <StateProbe />
      </AppProvider>
    )
    act(() => {
      webSocketOptions.current?.onMessage?.(taskInfoMessage(99))
      dagBurst(99).forEach((message) => webSocketOptions.current?.onMessage?.(message))
    })

    await waitFor(() => {
      expect(screen.getByTestId("dag-phase").textContent).toBe("failed")
      expect(screen.getByTestId("steps-count").textContent).toBe("1")
    })
    expect(screen.getByTestId("steps").textContent).toContain('"status":"failed"')
    expect(screen.getByTestId("steps").textContent).toContain('"dependencies":["dependency-one"]')
    expect(screen.getByTestId("steps").textContent).toContain('"started_at":"2026-05-27T05:00:03Z"')
    expect(screen.getByTestId("steps").textContent).toContain('"completed_at":"2026-05-27T05:00:05Z"')
  })

  it("projects every buffered Session DAG burst frame against the prior frame", async () => {
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    act(() => onMessage?.(taskInfoMessage(100)))
    const reset = getSessionControls().startNewConversation()
    await act(async () => {
      onMessage?.({ type: "conversation_reset", timestamp: "2026-05-27T05:00:03Z", data: {} })
      await reset
    })
    await getSessionControls().sendMessage("replacement", { clientMessageId: "buffered-dag-burst" })

    act(() => {
      dagBurst(101).forEach((message) => onMessage?.(message))
      onMessage?.(taskInfoMessage(101))
    })

    await waitFor(() => {
      expect(screen.getByTestId("task-id").textContent).toBe("101")
      expect(screen.getByTestId("dag-phase").textContent).toBe("failed")
      expect(screen.getByTestId("steps-count").textContent).toBe("1")
    })
    expect(screen.getByTestId("steps").textContent).toContain('"status":"failed"')
    expect(screen.getByTestId("steps").textContent).toContain('"dependencies":["dependency-one"]')
    expect(screen.getByTestId("steps").textContent).toContain('"started_at":"2026-05-27T05:00:03Z"')
    expect(screen.getByTestId("steps").textContent).toContain('"completed_at":"2026-05-27T05:00:05Z"')
  })

  for (const strictMode of [false, true]) {
    for (const order of ["ordinary-first", "session-first"] as const) {
      it(`preserves ${order} actions in one batch${strictMode ? " under StrictMode" : ""}`, async () => {
        const captured = { current: null as null | ReturnType<typeof useApp> }
        function Capture() {
          captured.current = useApp()
          return <StateProbe />
        }
        const provider = (
          <AppProvider token="token" transport={makeSessionTransport()}>
            <Capture />
          </AppProvider>
        )
        render(strictMode ? <React.StrictMode>{provider}</React.StrictMode> : provider)

        const onMessage = webSocketOptions.current?.onMessage
        const sessionTaskId = order === "ordinary-first"
          ? (strictMode ? 104 : 103)
          : (strictMode ? 106 : 105)
        const ordinaryAction = {
          type: "ADD_MESSAGE" as const,
          payload: {
            id: `${order}-${String(strictMode)}-ordinary`,
            role: "assistant" as const,
            content: "ordinary action",
            timestamp: "2026-05-27T05:00:00Z",
            status: "completed" as const,
          },
        }

        act(() => {
          if (order === "ordinary-first") {
            captured.current?.dispatch(ordinaryAction)
            onMessage?.(taskInfoMessage(sessionTaskId))
            return
          }
          onMessage?.(taskInfoMessage(sessionTaskId))
          captured.current?.dispatch(ordinaryAction)
        })

        await waitFor(() => {
          expect(screen.getByTestId("messages").textContent).toContain("ordinary action")
          expect(screen.getByTestId("task-id").textContent).toBe(String(sessionTaskId))
        })
      })
    }
  }

  it("projects an effectful replay action once under StrictMode", () => {
    const captured = { current: null as null | ReturnType<typeof useApp> }
    const replayScheduler = { setPlaybackSpeed: vi.fn() }
    function Capture() {
      captured.current = useApp()
      return null
    }

    render(
      <React.StrictMode>
        <AppProvider token="token">
          <Capture />
        </AppProvider>
      </React.StrictMode>,
    )

    act(() => {
      captured.current?.dispatch({
        type: "SET_REPLAY_SCHEDULER",
        payload: replayScheduler as never,
      })
      captured.current?.dispatch({ type: "SET_REPLAY_SPEED", payload: 2 })
    })

    expect(replayScheduler.setPlaybackSpeed).toHaveBeenCalledTimes(1)
    expect(replayScheduler.setPlaybackSpeed).toHaveBeenCalledWith(2)
    expect(captured.current?.state.replaySpeed).toBe(2)
  })

  it("keeps the projection head through a priority-split provider render", async () => {
    const captured = { current: null as null | ReturnType<typeof useApp> }
    let renderAtHigherPriority!: () => void
    function Capture() {
      captured.current = useApp()
      return <StateProbe />
    }
    function PriorityHarness() {
      const [revision, setRevision] = React.useState(0)
      renderAtHigherPriority = () => setRevision(current => current + 1)
      return (
        <AppProvider token="token">
          <div data-testid="priority-revision">{revision}</div>
          <Capture />
        </AppProvider>
      )
    }

    render(<PriorityHarness />)

    act(() => {
      React.startTransition(() => {
        captured.current?.dispatch({
          type: "ADD_MESSAGE",
          payload: {
            id: "pending-low-priority",
            role: "assistant",
            content: "pending low priority",
            timestamp: "2026-05-27T05:00:00Z",
            status: "completed",
          },
        })
      })
      flushSync(renderAtHigherPriority)
      captured.current?.dispatch({
        type: "ADD_MESSAGE",
        payload: {
          id: "later-action",
          role: "assistant",
          content: "later action",
          timestamp: "2026-05-27T05:00:01Z",
          status: "completed",
        },
      })
    })

    await waitFor(() => {
      expect(screen.getByTestId("priority-revision").textContent).toBe("1")
      expect(screen.getByTestId("messages").textContent).toContain("pending low priority")
      expect(screen.getByTestId("messages").textContent).toContain("later action")
    })
  })

  it("keeps a reconnected Session bound when same-task task_info is replayed and accepts its newer follow-up state", async () => {
    const { rerender } = render(
      <AppProvider token="token" transport={makeSessionTransport(makeSessionConnection("same-task-old"))}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    act(() => webSocketOptions.current?.onMessage?.(taskInfoMessage(102, {
      run_id: "same-task-run",
      state_version: 5,
    })))

    rerender(
      <AppProvider token="token" transport={makeSessionTransport(makeSessionConnection("same-task-new"))}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    act(() => {
      webSocketOptions.current?.onMessage?.(taskInfoMessage(102, {
        run_id: "same-task-run",
        state_version: 5,
        title: "Refreshed same task",
        status: "paused",
      }))
      webSocketOptions.current?.onMessage?.({
        type: "task_paused",
        timestamp: "2026-05-27T05:00:06Z",
        task_id: 102,
        run_id: "same-task-run",
        state_version: 4,
        control_state: "paused",
        status: "paused",
      })
      webSocketOptions.current?.onMessage?.({
        type: "task_paused",
        timestamp: "2026-05-27T05:00:07Z",
        task_id: 102,
        run_id: "same-task-run",
        state_version: 6,
        control_state: "paused",
        status: "paused",
      })
    })

    await waitFor(() => {
      expect(screen.getByTestId("session-conversation-state").textContent).toBe("bound")
      expect(screen.getByTestId("task-id").textContent).toBe("102")
      expect(screen.getByTestId("task-title").textContent).toBe("Refreshed same task")
      expect(screen.getByTestId("task-status").textContent).toBe("paused")
    })
  })

  it("requires reload when the current replacement connection publishes a different task", async () => {
    const { rerender } = render(
      <AppProvider token="token" transport={makeSessionTransport(makeSessionConnection("different-task-old"))}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    act(() => webSocketOptions.current?.onMessage?.(taskInfoMessage(102)))

    rerender(
      <AppProvider token="token" transport={makeSessionTransport(makeSessionConnection("different-task-new"))}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    act(() => webSocketOptions.current?.onMessage?.(taskInfoMessage(103)))

    await waitFor(() => {
      expect(screen.getByTestId("session-conversation-state").textContent).toBe("reload_required")
    })
  })

  it("rejects an outstanding reset when AppProvider unmounts", async () => {
    const { unmount } = render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    act(() => {
      webSocketOptions.current?.onMessage?.(taskInfoMessage(62))
    })
    const resetOutcome = getSessionControls()
      .startNewConversation()
      .then(() => "resolved", () => "rejected")

    unmount()

    await expect(Promise.race([
      resetOutcome,
      new Promise<string>(resolve => {
        window.setTimeout(() => resolve("timeout"), 20)
      }),
    ])).resolves.toBe("rejected")
  })

  it("rejects the current reset when the same Session connection disconnects", async () => {
    const transport = makeSessionTransport()
    const sessionView = () => (
      <AppProvider token="token" transport={transport}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    const { rerender } = render(sessionView())
    act(() => {
      webSocketOptions.current?.onMessage?.(taskInfoMessage(64))
      webSocketOptions.current?.onMessage?.(
        assistantMessage("Conversation survives disconnect", 64)
      )
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-id").textContent).toBe("64")
      expect(screen.getByTestId("messages").textContent).toContain(
        "Conversation survives disconnect"
      )
    })

    let reset!: Promise<void>
    act(() => {
      reset = getSessionControls().startNewConversation()
    })
    const resetOutcome = reset.then(
      () => "resolved",
      error => error instanceof Error ? error.message : String(error),
    )
    expect(screen.getByTestId("reset-pending").textContent).toBe("true")
    expect(sendRawMessageMock).toHaveBeenCalledTimes(1)

    act(() => {
      wsHarness.isConnected = false
      rerender(sessionView())
    })

    await expect(Promise.race([
      resetOutcome,
      new Promise<string>(resolve => {
        window.setTimeout(() => resolve("timeout"), 20)
      }),
    ])).resolves.toMatch(/disconnect|connection/i)
    expect(screen.getByTestId("reset-pending").textContent).toBe("false")
    expect(screen.getByTestId("task-id").textContent).toBe("64")
    expect(screen.getByTestId("messages").textContent).toContain(
      "Conversation survives disconnect"
    )
  })

  it("does not mutate AppProvider state when a pending Session delivery settles after unmount", async () => {
    const acknowledgement = deferred<{
      client_message_id: string
      turn_id: string
    }>()
    sendChatMessageMock.mockReturnValueOnce(acknowledgement.promise)
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => {})
    const { unmount } = render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
      </AppProvider>
    )
    const delivery = getSessionControls().sendMessage("Unmounted delivery", {
      clientMessageId: "unmounted-delivery",
    })

    unmount()
    acknowledgement.resolve({
      client_message_id: "unmounted-delivery",
      turn_id: "unmounted-delivery",
    })
    await delivery

    expect(
      consoleError.mock.calls.flat().join(" ")
    ).not.toMatch(/state update.*unmounted|unmounted.*state update/i)
    consoleError.mockRestore()
  })

  it("uses adopted task_info version as the Session state-ordering baseline", async () => {
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )

    act(() => {
      webSocketOptions.current?.onMessage?.(
        taskInfoMessage(63, {
          run_id: "session-run",
          state_version: 5,
        })
      )
      webSocketOptions.current?.onMessage?.({
        type: "task_paused",
        timestamp: "2026-05-27T05:00:01Z",
        task_id: 63,
        run_id: "session-run",
        state_version: 4,
        control_state: "paused",
        status: "paused",
      })
    })
    expect(screen.getByTestId("task-status").textContent).toBe("running")

    act(() => {
      webSocketOptions.current?.onMessage?.({
        type: "task_paused",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 63,
        run_id: "session-run",
        state_version: 6,
        control_state: "paused",
        status: "paused",
      })
    })
    expect(screen.getByTestId("task-status").textContent).toBe("paused")
  })

  it("ignores stale and unsolicited Session reset acknowledgements", async () => {
    const firstTransport = makeSessionTransport(
      makeSessionConnection("session-reset-old")
    )
    const { rerender } = render(
      <AppProvider token="token" transport={firstTransport}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    act(() => {
      webSocketOptions.current?.onMessage?.(taskInfoMessage(61))
      webSocketOptions.current?.onMessage?.(
        assistantMessage("Keep on refresh", 61)
      )
    })
    const oldOnMessage = webSocketOptions.current?.onMessage
    const resetOutcome = getSessionControls()
      .startNewConversation()
      .then(() => "resolved", () => "rejected")

    rerender(
      <AppProvider
        token="token"
        transport={makeSessionTransport(
          makeSessionConnection("session-reset-new")
        )}
      >
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    act(() => {
      oldOnMessage?.({
        type: "conversation_reset",
        timestamp: "2026-05-27T05:00:04Z",
        data: {},
      })
      webSocketOptions.current?.onMessage?.({
        type: "conversation_reset",
        timestamp: "2026-05-27T05:00:05Z",
        data: {},
      })
    })

    await expect(resetOutcome).resolves.toBe("rejected")
    expect(screen.getByTestId("task-id").textContent).toBe("61")
    expect(screen.getByTestId("messages").textContent).toContain(
      "Keep on refresh"
    )
    expect(screen.getByTestId("reset-pending").textContent).toBe("false")
  })

  it("fails closed with reload-required when the reset acknowledgement deadline expires", async () => {
    vi.useFakeTimers()
    try {
      render(
        <AppProvider token="token" transport={makeSessionTransport()}>
          <SessionControlsProbe />
          <StateProbe />
        </AppProvider>
      )
      act(() => {
        webSocketOptions.current?.onMessage?.(taskInfoMessage(73))
      })

      let outcome = "pending"
      act(() => {
        void getSessionControls().startNewConversation().then(
          () => { outcome = "resolved" },
          error => { outcome = error instanceof Error ? error.message : String(error) },
        )
      })
      await act(async () => {
        vi.advanceTimersByTime(30_001)
        await Promise.resolve()
      })

      expect(outcome).toMatch(/reload|required|unknown/i)
      expect(screen.getByTestId("session-conversation-state").textContent).toBe("reload_required")
      expect(screen.getByTestId("task-id").textContent).toBe("73")
    } finally {
      vi.useRealTimers()
    }
  })

  it("fails closed rather than carrying a reset acknowledgement across a connection refresh", async () => {
    const { rerender } = render(
      <AppProvider token="token" transport={makeSessionTransport(makeSessionConnection("reset-old"))}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    act(() => {
      webSocketOptions.current?.onMessage?.(taskInfoMessage(74))
    })
    const reset = getSessionControls().startNewConversation()
    rerender(
      <AppProvider token="token" transport={makeSessionTransport(makeSessionConnection("reset-new"))}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )

    await expect(reset).rejects.toThrow(/reload|required|unknown/i)
    expect(screen.getByTestId("session-conversation-state").textContent).toBe("reload_required")
    expect(screen.getByTestId("task-id").textContent).toBe("74")
  })

  it("buffers ordered frames only after a replacement send is accepted and flushes them after task_info", async () => {
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    act(() => {
      onMessage?.(taskInfoMessage(75))
    })
    const reset = getSessionControls().startNewConversation()
    await act(async () => {
      onMessage?.({ type: "conversation_reset", timestamp: "2026-05-27T05:00:03Z", data: {} })
      await reset
    })

    act(() => {
      onMessage?.(assistantMessage("Before acceptance", 76))
    })
    await getSessionControls().sendMessage("Replacement", { clientMessageId: "replacement-buffer" })
    expect(screen.getByTestId("session-conversation-state").textContent).toBe("replacement_awaiting_task")

    act(() => {
      onMessage?.(assistantMessage("Buffered first", 76))
      onMessage?.(assistantMessage("Buffered second", 76))
      onMessage?.(taskInfoMessage(76))
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-id").textContent).toBe("76")
      expect(screen.getByTestId("messages").textContent).toContain("Buffered first")
      expect(screen.getByTestId("messages").textContent).toContain("Buffered second")
    })
    expect(screen.getByTestId("messages").textContent).not.toContain("Before acceptance")
  })

  it("enters reload-required when the post-acceptance frame buffer overflows", async () => {
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    act(() => onMessage?.(taskInfoMessage(77)))
    const reset = getSessionControls().startNewConversation()
    await act(async () => {
      onMessage?.({ type: "conversation_reset", timestamp: "2026-05-27T05:00:03Z", data: {} })
      await reset
    })
    await getSessionControls().sendMessage("Replacement", { clientMessageId: "replacement-overflow" })
    act(() => {
      for (let index = 0; index < 65; index += 1) {
        onMessage?.(assistantMessage(`Buffered ${index}`, 78))
      }
    })
    expect(screen.getByTestId("session-conversation-state").textContent).toBe("reload_required")
  })

  it("enters reload-required when one buffered frame exceeds the serialized byte cap", async () => {
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    act(() => onMessage?.(taskInfoMessage(80)))
    const reset = getSessionControls().startNewConversation()
    await act(async () => {
      onMessage?.({ type: "conversation_reset", timestamp: "2026-05-27T05:00:03Z", data: {} })
      await reset
    })
    await getSessionControls().sendMessage("Replacement", { clientMessageId: "replacement-byte-cap" })
    act(() => {
      onMessage?.(assistantMessage("x".repeat(256 * 1024), 81))
    })
    expect(screen.getByTestId("session-conversation-state").textContent).toBe("reload_required")
  })

  it("counts a task_info candidate in the bounded replacement buffer", async () => {
    const acknowledgement = deferred<{ client_message_id: string; turn_id: string }>()
    sendChatMessageMock.mockReturnValueOnce(acknowledgement.promise)
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    act(() => onMessage?.(taskInfoMessage(82)))
    const reset = getSessionControls().startNewConversation()
    await act(async () => {
      onMessage?.({ type: "conversation_reset", timestamp: "2026-05-27T05:00:03Z", data: {} })
      await reset
    })
    const replacement = getSessionControls().sendMessage("Replacement", { clientMessageId: "candidate-budget" })
    act(() => {
      for (let index = 0; index < 63; index += 1) {
        onMessage?.(assistantMessage(`Buffered ${index}`, 83))
      }
      onMessage?.(taskInfoMessage(83))
      onMessage?.(assistantMessage("one frame too many", 83))
    })
    expect(screen.getByTestId("session-conversation-state").textContent).toBe("reload_required")
    await act(async () => {
      acknowledgement.reject(Object.assign(new Error("ambiguous replacement"), {
        disposition: "outcome_unknown",
      }))
      await expect(replacement).rejects.toThrow("ambiguous replacement")
    })
  })

  it("applies the byte cap to an oversized task_info candidate before acceptance", async () => {
    const acknowledgement = deferred<{ client_message_id: string; turn_id: string }>()
    sendChatMessageMock.mockReturnValueOnce(acknowledgement.promise)
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    act(() => onMessage?.(taskInfoMessage(84)))
    const reset = getSessionControls().startNewConversation()
    await act(async () => {
      onMessage?.({ type: "conversation_reset", timestamp: "2026-05-27T05:00:03Z", data: {} })
      await reset
    })
    const replacement = getSessionControls().sendMessage("Replacement", { clientMessageId: "oversized-candidate" })
    act(() => onMessage?.(taskInfoMessage(85, { description: "x".repeat(256 * 1024) })))
    expect(screen.getByTestId("session-conversation-state").textContent).toBe("reload_required")
    await act(async () => {
      acknowledgement.reject(Object.assign(new Error("ambiguous replacement"), {
        disposition: "outcome_unknown",
      }))
      await expect(replacement).rejects.toThrow("ambiguous replacement")
    })
  })

  it("expires an accepted replacement awaiting task_info instead of guessing after its deadline", async () => {
    vi.useFakeTimers()
    try {
      render(
        <AppProvider token="token" transport={makeSessionTransport()}>
          <SessionControlsProbe />
        </AppProvider>
      )
      const onMessage = webSocketOptions.current?.onMessage
      act(() => onMessage?.(taskInfoMessage(82)))
      const reset = getSessionControls().startNewConversation()
      await act(async () => {
        onMessage?.({ type: "conversation_reset", timestamp: "2026-05-27T05:00:03Z", data: {} })
        await reset
      })
      await getSessionControls().sendMessage("Replacement", { clientMessageId: "replacement-deadline" })
      await act(async () => {
        vi.advanceTimersByTime(30_001)
        await Promise.resolve()
      })
      expect(screen.getByTestId("session-conversation-state").textContent).toBe("reload_required")
    } finally {
      vi.useRealTimers()
    }
  })

  it("does not share Session message dedupe state between provider instances", async () => {
    render(
      <>
        <AppProvider token="first" transport={makeSessionTransport(makeSessionConnection("provider-one"))}>
          <ScopedMessagesProbe testId="provider-one-messages" />
        </AppProvider>
        <AppProvider token="second" transport={makeSessionTransport(makeSessionConnection("provider-two"))}>
          <ScopedMessagesProbe testId="provider-two-messages" />
        </AppProvider>
      </>
    )
    const [first, second] = webSocketOptions.all
    act(() => {
      first?.onMessage?.(assistantMessage("Same content, separate provider"))
      second?.onMessage?.(assistantMessage("Same content, separate provider"))
    })
    await waitFor(() => {
      expect(screen.getByTestId("provider-one-messages").textContent).toContain("Same content, separate provider")
      expect(screen.getByTestId("provider-two-messages").textContent).toContain("Same content, separate provider")
    })
  })

  it("runs delayed historical replay only for the provider that received completion", async () => {
    vi.useFakeTimers()
    try {
      const first = { current: null as null | ReturnType<typeof useApp> }
      const second = { current: null as null | ReturnType<typeof useApp> }
      function Capture({ target }: { target: typeof first }) {
        target.current = useApp()
        return null
      }
      render(<><AppProvider token="replay-one"><Capture target={first} /></AppProvider><AppProvider token="replay-two"><Capture target={second} /></AppProvider></>)
      act(() => {
        first.current?.dispatch({ type: "START_REPLAY", payload: { taskId: 1, events: [] } })
        first.current?.dispatch({ type: "ADD_TO_REPLAY_CACHE", payload: assistantMessage("first replay") as never })
      })
      act(() => {
        second.current?.dispatch({ type: "SET_HISTORY_LOADING", payload: false })
      })
      expect(webSocketOptions.all.at(-1)?.token).toBe("replay-two")
      const firstHandler = webSocketOptions.all.filter(option => option.token === "replay-one").at(-1)?.onMessage
      act(() => firstHandler?.({ type: "historical_data_complete", timestamp: "2026-05-27T05:00:00Z" }))
      act(() => vi.advanceTimersByTime(500))
      expect(first.current?.state.replayScheduler).not.toBeNull()
      expect(second.current?.state.replayScheduler).toBeNull()
    } finally { vi.useRealTimers() }
  })

  it("stops a replay scheduler before publishing the fresh replacement state", async () => {
    const replayScheduler = { stop: vi.fn() }
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    act(() => {
      onMessage?.(taskInfoMessage(79))
      getSessionControls().dispatch({ type: "SET_REPLAY_SCHEDULER", payload: replayScheduler as never })
    })
    const reset = getSessionControls().startNewConversation()
    await act(async () => {
      onMessage?.({ type: "conversation_reset", timestamp: "2026-05-27T05:00:03Z", data: {} })
      await reset
    })
    expect(replayScheduler.stop).toHaveBeenCalledOnce()
    expect(screen.getByTestId("session-conversation-state").textContent).toBe("replacement_ready")
  })

  it("fails closed for every Session file entry point without issuing HTTP or socket egress", async () => {
    const transport = makeSessionTransport()
    render(
      <AppProvider token="token" transport={transport}>
        <SessionControlsProbe />
        <StateProbe />
        <MessageContentProbe />
      </AppProvider>
    )
    expect(screen.getByTestId("files-disabled").textContent).toBe("true")
    expect(screen.getByTestId("voice-input-enabled").textContent).toBe("false")
    expect(screen.getByTestId("task-controls-enabled").textContent).toBe("false")
    const file = new File(["secret"], "secret.txt", { type: "text/plain" })

    await expect(
      getSessionControls().sendMessage("Taskless file", undefined, [file])
    ).rejects.toThrow(/files.*disabled|disabled.*files/i)
    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(sendChatMessageMock).not.toHaveBeenCalled()
    expect(transport.uploadFiles).not.toHaveBeenCalled()

    act(() => {
      webSocketOptions.current?.onMessage?.(taskInfoMessage(71))
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-id").textContent).toBe("71")
    })
    await expect(
      getSessionControls().sendMessage("Task file", undefined, [file])
    ).rejects.toThrow(/files.*disabled|disabled.*files/i)
    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(sendChatMessageMock).not.toHaveBeenCalled()
    expect(transport.uploadFiles).not.toHaveBeenCalled()

    act(() => {
      getSessionControls().openFilePreview("manual-file", "manual.pdf")
    })
    expect(screen.getByTestId("preview-open").textContent).toBe("false")
    expect(() =>
      getSessionControls().getFilePreviewUrl("manual-file")
    ).toThrow(/files.*disabled|disabled.*files/i)
    expect(() =>
      getSessionControls().getFileDownloadUrl("manual-file")
    ).toThrow(/files.*disabled|disabled.*files/i)

    const previewEvents: Event[] = []
    const previewListener = (event: Event) => previewEvents.push(event)
    window.addEventListener("openFilePreview", previewListener)
    act(() => {
      webSocketOptions.current?.onMessage?.({
        type: "task_completed",
        timestamp: "2026-05-27T05:00:06Z",
        task_id: 71,
        task: { id: 71, status: "completed" },
        status: "completed",
        file_outputs: [{
          file_id: "generated-file",
          filename: "generated.pdf",
        }],
      } as TestWebSocketMessage)
    })
    expect(screen.getByTestId("preview-open").textContent).toBe("false")
    fireEvent.click(
      screen.getByRole("button", {
        name: "agent.logs.event.messages.previewLabel",
      })
    )
    expect(previewEvents).toHaveLength(0)
    expect(screen.getByTestId("preview-open").textContent).toBe("false")
    expect(apiRequestMock).not.toHaveBeenCalled()
    window.removeEventListener("openFilePreview", previewListener)
  })

  it("uses a top-level disabled files capability at every transport and presentation boundary", async () => {
    const transport = {
      capabilities: { files: "disabled" as const },
      uploadFiles: vi.fn(),
    }
    render(
      <AppProvider token="token" transport={transport}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>,
    )

    const file = new File(["secret"], "secret.txt", { type: "text/plain" })
    expect(screen.getByTestId("files-disabled")).toHaveTextContent("true")

    const disabledUpload = webSocketOptions.current?.uploadFiles as (
      files: File[],
      params: { taskId?: number | null; taskType: string },
    ) => Promise<unknown>
    await expect(disabledUpload([file], { taskType: "task" })).rejects.toThrow(
      /files.*disabled|disabled.*files/i,
    )
    await expect(
      getSessionControls().sendMessage("Disabled file", undefined, [file]),
    ).rejects.toThrow(/files.*disabled|disabled.*files/i)
    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(sendChatMessageMock).not.toHaveBeenCalled()
    expect(transport.uploadFiles).not.toHaveBeenCalled()

    act(() => {
      getSessionControls().openFilePreview("manual-file", "manual.pdf")
    })
    expect(screen.getByTestId("preview-open")).toHaveTextContent("false")
    expect(() => getSessionControls().getFilePreviewUrl("manual-file")).toThrow(
      /files.*disabled|disabled.*files/i,
    )
    expect(() => getSessionControls().getFileDownloadUrl("manual-file")).toThrow(
      /files.*disabled|disabled.*files/i,
    )

    act(() => {
      webSocketOptions.current?.onMessage?.({
        type: "task_completed",
        timestamp: "2026-05-27T05:00:06Z",
        task_id: 71,
        task: { id: 71, status: "completed" },
        status: "completed",
        file_outputs: [{ file_id: "generated-file", filename: "generated.pdf" }],
      } as TestWebSocketMessage)
    })
    expect(screen.getByTestId("preview-open")).toHaveTextContent("false")
  })

  it("fails closed when a malformed Session descriptor omits files", async () => {
    const transport = makeSessionTransport() as unknown as {
      uploadFiles: ReturnType<typeof vi.fn>
      session: Record<string, unknown>
    }
    delete transport.session.files
    render(
      <AppProvider token="token" transport={transport as never}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>,
    )

    const file = new File(["secret"], "secret.txt", { type: "text/plain" })
    expect(screen.getByTestId("files-disabled")).toHaveTextContent("true")
    const disabledUpload = webSocketOptions.current?.uploadFiles as (
      files: File[],
      params: { taskId?: number | null; taskType: string },
    ) => Promise<unknown>
    await expect(disabledUpload([file], { taskType: "task" })).rejects.toThrow(
      /files.*disabled|disabled.*files/i,
    )
    await expect(
      getSessionControls().sendMessage("Malformed Session file", undefined, [file]),
    ).rejects.toThrow(/files.*disabled|disabled.*files/i)
    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(sendChatMessageMock).not.toHaveBeenCalled()
    expect(transport.uploadFiles).not.toHaveBeenCalled()
  })

  it("disables agent cards when Session files are disabled even if the transport requests cards", () => {
    const transport = makeSessionTransport() as unknown as {
      session: Record<string, unknown>
    }
    transport.session.agentCards = "enabled"

    const { container } = render(
      <AppProvider token="token" transport={transport as never}>
        <SessionControlsProbe />
        <MarkdownRenderer content="[Specialist](agent://42)" />
      </AppProvider>,
    )

    expect(screen.getByTestId("files-disabled")).toHaveTextContent("true")
    expect(screen.getByTestId("agent-cards-enabled")).toHaveTextContent("false")
    expect(screen.getByText("Specialist")).not.toHaveAttribute("data-agent-id")
    expect(container.querySelector("[data-agent-card-wrapper]")).toBeNull()
    expect(apiRequestMock).not.toHaveBeenCalled()
  })

  it("fails closed for agent cards when a malformed Session transport omits the capability", () => {
    const malformedTransport = makeSessionTransport() as unknown as {
      session: Record<string, unknown>
    }
    delete malformedTransport.session.agentCards

    const { container } = render(
      <AppProvider token="token" transport={malformedTransport as never}>
        <MarkdownRenderer content="[Specialist](agent://42)" />
      </AppProvider>,
    )

    expect(screen.getByText("Specialist")).not.toHaveAttribute("data-agent-id")
    expect(container.querySelector("[data-agent-card-wrapper]")).toBeNull()
    expect(apiRequestMock).not.toHaveBeenCalled()
  })

  it("opens markdown links in a new tab only when the transport enables linksOpenInNewTab", () => {
    const { unmount } = render(
      <AppProvider
        token="public-token"
        transport={{ capabilities: { linksOpenInNewTab: "enabled" } }}
      >
        <MarkdownRenderer content="[Docs](https://example.com/docs)" />
      </AppProvider>,
    )

    const widgetLink = screen.getByRole("link", { name: "Docs" })
    expect(widgetLink).toHaveAttribute("target", "_blank")
    expect(widgetLink).toHaveAttribute("rel", "noopener noreferrer")
    unmount()

    render(
      <AppProvider token="public-token" transport={{ capabilities: {} }}>
        <MarkdownRenderer content="[Docs](https://example.com/docs)" />
      </AppProvider>,
    )

    const defaultLink = screen.getByRole("link", { name: "Docs" })
    expect(defaultLink).not.toHaveAttribute("target")
    expect(defaultLink).not.toHaveAttribute("rel")
  })
})

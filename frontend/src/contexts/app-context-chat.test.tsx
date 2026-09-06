import React from "react"
import { flushSync } from "react-dom"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

type TestWebSocketMessage = {
  type: string
  timestamp: string
  data?: unknown
  error_code?: unknown
  message?: string
  task_id?: number
  step_id?: string
  task?: Record<string, unknown>
  status?: string
  run_id?: string | null
  state_version?: number
  control_state?: string
  request_id?: string
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
    legacyErrorProse?: "trusted" | "untrusted"
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
    legacyErrorProse?: "trusted" | "untrusted"
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
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string | number>) =>
      vars ? `${key}:${JSON.stringify(vars)}` : key,
  }),
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
    legacyErrorProse?: "trusted" | "untrusted"
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
  projectErrorFrameForDisplay,
  type AppProviderTransportConfig,
  useApp,
} from "./app-context-chat"
import { ChatStartScreen } from "@/components/chat/ChatStartScreen"
import { MarkdownRenderer } from "@/components/ui/markdown-renderer"
import { TASK_ERROR_EVENT, type TaskErrorEventDetail } from "@/lib/task-error-events"
import type { Translate } from "@/contexts/i18n-context"

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
            interactionRequestId: message.interactionRequestId,
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
      <div data-testid="waiting-request-id">{state.currentTask?.waitingRequestId || ""}</div>
      <div data-testid="waiting-interactions">{JSON.stringify(state.currentTask?.waitingInteractions || [])}</div>
      <div data-testid="task-dag-terminated-at">{state.currentTask?.dagTerminatedAt ?? ""}</div>
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
      <div data-testid="dag-created-at">{state.dagExecution?.created_at ?? ""}</div>
      <div data-testid="dag-turn-id">{state.dagExecution?.turn_id ?? ""}</div>
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

function SeedCompletedTask() {
  const { dispatch } = useApp()

  React.useEffect(() => {
    dispatch({ type: "SET_TASK_ID", payload: 1 })
    dispatch({
      type: "SET_CURRENT_TASK",
      payload: {
        id: "1",
        title: "Completed task",
        status: "completed",
        description: "Completed task",
        createdAt: "2026-05-27T05:00:00Z",
        updatedAt: "2026-05-27T05:01:00Z",
      },
    })
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

  it("does not turn a completed task into waiting from question metadata alone", async () => {
    render(
      <AppProvider token="token">
        <SeedCompletedTask />
        <StateProbe />
      </AppProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("completed")
    })

    act(() => {
      webSocketOptions.current?.onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:00:00Z",
        task_id: 1,
        data: {
          event_id: "stray-question-row",
          event_type: "agent_message",
          data: {
            message: "A question-shaped note",
            message_type: "question",
            expect_response: false,
            display: "chat",
          },
        },
      })
    })

    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "A question-shaped note",
      )
    })
    expect(screen.getByTestId("task-status").textContent).toBe("completed")
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

  it("does not retarget a different task's status when a failed-retry ack arrives after switching away", async () => {
    let acknowledgeDelivery: (() => void) | undefined
    sendChatMessageMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          acknowledgeDelivery = () =>
            resolve({ client_message_id: "turn-retry", turn_id: "turn-retry" })
        })
    )

    let retry: (() => Promise<void>) | undefined
    let markTaskOneFailed: (() => void) | undefined
    let switchToTaskTwo: (() => void) | undefined
    function RetryAckProbe() {
      const { sendMessage, dispatch, setTaskId } = useApp()
      retry = () => sendMessage("Retry the failed turn", { clientMessageId: "turn-retry" })
      markTaskOneFailed = () => {
        dispatch({
          type: "SET_CURRENT_TASK",
          payload: {
            id: "1",
            title: "Test task",
            status: "failed",
            description: "Test task",
            createdAt: "2026-05-27T05:00:00Z",
            updatedAt: "2026-05-27T05:00:00Z",
          },
        })
      }
      switchToTaskTwo = () => {
        setTaskId(2, { navigate: false })
        dispatch({
          type: "SET_CURRENT_TASK",
          payload: {
            id: "2",
            title: "Task two",
            status: "completed",
            description: "Task two",
            createdAt: "2026-05-27T05:00:00Z",
            updatedAt: "2026-05-27T05:00:00Z",
          },
        })
      }
      return null
    }

    render(
      <AppProvider token="token">
        <SeedExistingTask />
        <RetryAckProbe />
        <StateProbe />
      </AppProvider>
    )

    // SeedExistingTask starts task 1 as "running" - flip it to "failed" so
    // the retry's optimistic completed/failed -> running branch is armed.
    act(() => {
      markTaskOneFailed?.()
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("failed")
    })

    let delivery: Promise<void> | undefined
    await act(async () => {
      delivery = retry?.()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(sendChatMessageMock).toHaveBeenCalledOnce()

    // The user navigates to task 2 (already completed) while task 1's retry
    // is still in flight.
    act(() => {
      switchToTaskTwo?.()
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-id").textContent).toBe("2")
      expect(screen.getByTestId("task-status").textContent).toBe("completed")
    })

    // Task 1's delayed ack now resolves - it must not flip task 2 (the task
    // actually being viewed) to "running".
    await act(async () => {
      acknowledgeDelivery?.()
      await delivery
    })
    expect(screen.getByTestId("task-status").textContent).toBe("completed")
  })

  it("clears a stale dagTerminatedAt once a rerun's task_info reports running again", async () => {
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    // Run 1 completes via UPDATE_TASK_STATUS, which stamps dagTerminatedAt.
    act(() => {
      onMessage?.({
        type: "task_completed",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        task: { id: 1, status: "completed" },
        success: true,
      } as TestWebSocketMessage)
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("completed")
    })
    expect(screen.getByTestId("task-dag-terminated-at").textContent).not.toBe("")

    // Run 2 starts, observed only via a task_info event (SET_CURRENT_TASK,
    // not UPDATE_TASK_STATUS) - the stale run-1 timestamp must not survive.
    act(() => {
      onMessage?.(taskInfoMessage(1, { status: "running" }))
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("running")
    })
    expect(screen.getByTestId("task-dag-terminated-at").textContent).toBe("")
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

  it("rejects an unsuccessful 2xx pre-task upload instead of dropping files", async () => {
    apiRequestMock.mockResolvedValue(new Response(JSON.stringify({
      success: false,
      error_code: "upload_too_large",
      detail: "private storage detail",
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }))

    let send: (() => Promise<void>) | undefined
    function CreateTaskWithFileProbe() {
      const { sendMessage } = useApp()
      send = () => sendMessage(
        "analyze attachment",
        { clientMessageId: "turn-upload-failure" },
        [new File(["data"], "data.txt")],
      )
      return null
    }

    render(
      <AppProvider token="token">
        <CreateTaskWithFileProbe />
      </AppProvider>
    )

    await expect(send?.()).rejects.toThrow("clientErrors.uploadTooLarge")
    expect(apiRequestMock).toHaveBeenCalledOnce()
    expect(sendChatMessageMock).not.toHaveBeenCalled()
  })

  it.each([
    {
      name: "blank",
      files: [{ file_id: "   " }, { file_id: "file-2" }],
    },
    {
      name: "duplicate",
      files: [{ file_id: "file-1" }, { file_id: " file-1 " }],
    },
  ])("rejects $name pre-task upload identifiers before task creation", async ({ files }) => {
    apiRequestMock.mockResolvedValue(new Response(JSON.stringify({
      success: true,
      files,
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }))

    let send: (() => Promise<void>) | undefined
    function CreateTaskWithMalformedFilesProbe() {
      const { sendMessage } = useApp()
      send = () => sendMessage(
        "analyze attachments",
        { clientMessageId: "turn-malformed-upload" },
        [
          new File(["first"], "first.txt"),
          new File(["second"], "second.txt"),
        ],
      )
      return null
    }

    render(
      <AppProvider token="token">
        <CreateTaskWithMalformedFilesProbe />
      </AppProvider>
    )

    await expect(send?.()).rejects.toThrow("clientErrors.uploadFailed")
    expect(apiRequestMock).toHaveBeenCalledOnce()
    expect(sendChatMessageMock).not.toHaveBeenCalled()
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

  it("drops a foreign task_completed sent in the real production wire shape, not just the flat test shape", async () => {
    // use-websocket.ts's own normalizer (the "type": "task_completed" branch)
    // never hands handleMessage a flat message - it wraps the ENTIRE raw
    // frame under `.data` and lifts only `task_id` (from `data.task.id ??
    // data.task_id`) to the top level. The flat shape used elsewhere in this
    // file happens to also work (normalizeTaskCompletedMessage falls back to
    // the root object when `.data` is absent), but it never actually
    // exercises that `.data` fallback branch, or proves the top-level
    // ownership guard (which only ever reads the top-level task_id) still
    // works once task/success/file_outputs move under `.data` instead of
    // sitting next to it.
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
        type: "task_completed",
        timestamp: "2026-05-27T05:00:03Z",
        task_id: 2,
        data: {
          type: "task_completed",
          timestamp: "2026-05-27T05:00:03Z",
          task: { id: 2, status: "completed" },
          success: true,
          file_outputs: [{ file_id: "task-2-report", filename: "report.pdf" }],
        },
      } as unknown as TestWebSocketMessage)
    })
    // Give the (would-be, if the guard failed) dispatches a tick to land.
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(screen.getByTestId("task-status").textContent).toBe("running")
    expect(screen.getByTestId("preview-open").textContent).toBe("false")

    // The viewed task's own production-shaped completion must still apply.
    act(() => {
      onMessage?.({
        type: "task_completed",
        timestamp: "2026-05-27T05:00:04Z",
        task_id: 1,
        data: {
          type: "task_completed",
          timestamp: "2026-05-27T05:00:04Z",
          task: { id: 1, status: "completed" },
          success: true,
        },
      } as unknown as TestWebSocketMessage)
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("completed")
    })
  })

  it("resets created_at/steps for a new turn_id instead of inheriting the prior run's", async () => {
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    const dagExecutionMessage = (
      timestamp: string,
      data: Record<string, unknown>,
    ): TestWebSocketMessage => ({
      type: "trace_event",
      timestamp,
      task_id: 1,
      data: {
        event_id: `dag-execution-${timestamp}`,
        event_type: "dag_execution",
        data,
      },
    })

    // Run A: planning, then executing with one step.
    act(() => {
      onMessage?.(dagExecutionMessage("2026-05-27T05:00:02Z", { phase: "planning", turn_id: "turn-A" }))
    })
    await waitFor(() => {
      expect(screen.getByTestId("dag-turn-id").textContent).toBe("turn-A")
    })
    expect(screen.getByTestId("dag-created-at").textContent).toBe("2026-05-27T05:00:02Z")

    act(() => {
      onMessage?.(dagExecutionMessage("2026-05-27T05:00:03Z", {
        phase: "executing",
        turn_id: "turn-A",
        steps: [{ id: "step-one", name: "Step one", dependencies: [] }],
      }))
    })
    await waitFor(() => {
      expect(screen.getByTestId("steps-count").textContent).toBe("1")
    })
    // Later events for the SAME run must not reset created_at, even though
    // this event's own timestamp differs from run A's first event.
    expect(screen.getByTestId("dag-created-at").textContent).toBe("2026-05-27T05:00:02Z")

    // Run B starts (e.g. sendMessage's RESET_DAG_STATE guard raced and lost,
    // so the reducer never got to clear run A's dagExecution/steps first) -
    // a DIFFERENT turn_id must reset created_at to this event's own timestamp
    // and clear run A's steps, not inherit either.
    act(() => {
      onMessage?.(dagExecutionMessage("2026-05-27T05:00:04Z", { phase: "planning", turn_id: "turn-B" }))
    })
    await waitFor(() => {
      expect(screen.getByTestId("dag-turn-id").textContent).toBe("turn-B")
    })
    expect(screen.getByTestId("dag-created-at").textContent).toBe("2026-05-27T05:00:04Z")
    expect(screen.getByTestId("steps-count").textContent).toBe("0")
  })

  it("does not treat a one-sided missing turn_id as a new run, in either direction", async () => {
    // A turn_id present on only one side (a legacy/history event that
    // predates this field, mixed with a modern one) can't be reliably
    // compared - both directions must fall back to "not a new run" rather
    // than guessing, or a legacy event arriving mid-run would wipe live
    // steps out from under it.
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    const dagExecutionMessage = (
      timestamp: string,
      data: Record<string, unknown>,
    ): TestWebSocketMessage => ({
      type: "trace_event",
      timestamp,
      task_id: 1,
      data: {
        event_id: `dag-execution-${timestamp}`,
        event_type: "dag_execution",
        data,
      },
    })

    // Tracked turn_id present ("turn-A"), incoming event has none (legacy
    // shape) - must not reset, AND must not wipe the tracked turn_id either
    // (SET_DAG_EXECUTION replaces the whole object, so without an explicit
    // carry-forward the legacy event would silently erase the run's
    // identity, blinding the new-run detection for the rest of the run).
    act(() => {
      onMessage?.(dagExecutionMessage("2026-05-27T05:00:02Z", {
        phase: "executing",
        turn_id: "turn-A",
        steps: [{ id: "step-one", name: "Step one", dependencies: [] }],
      }))
    })
    await waitFor(() => {
      expect(screen.getByTestId("steps-count").textContent).toBe("1")
    })
    act(() => {
      onMessage?.(dagExecutionMessage("2026-05-27T05:00:03Z", { phase: "executing" }))
    })
    await waitFor(() => {
      expect(screen.getByTestId("dag-created-at").textContent).toBe("2026-05-27T05:00:02Z")
    })
    expect(screen.getByTestId("steps-count").textContent).toBe("1")
    expect(screen.getByTestId("dag-turn-id").textContent).toBe("turn-A")

    // The tracked identity survived the legacy event above, so a further
    // same-run event must still read as a continuation...
    act(() => {
      onMessage?.(dagExecutionMessage("2026-05-27T05:00:04Z", { phase: "executing", turn_id: "turn-A" }))
    })
    await waitFor(() => {
      expect(screen.getByTestId("dag-turn-id").textContent).toBe("turn-A")
    })
    expect(screen.getByTestId("dag-created-at").textContent).toBe("2026-05-27T05:00:02Z")
    expect(screen.getByTestId("steps-count").textContent).toBe("1")

    // ...and a genuinely NEW run must still be detected as new - the exact
    // regression the carry-forward exists to prevent: had the legacy event
    // wiped the tracked turn_id, this differing id would compare against
    // undefined and fall back to "not a new run", inheriting turn-A's
    // created_at (a wildly wrong elapsed time) and stale steps.
    act(() => {
      onMessage?.(dagExecutionMessage("2026-05-27T05:00:05Z", { phase: "planning", turn_id: "turn-B" }))
    })
    await waitFor(() => {
      expect(screen.getByTestId("dag-turn-id").textContent).toBe("turn-B")
    })
    expect(screen.getByTestId("dag-created-at").textContent).toBe("2026-05-27T05:00:05Z")
    expect(screen.getByTestId("steps-count").textContent).toBe("0")
  })

  it("resets created_at/steps for a new turn_id via the bare dag_execution message type too", async () => {
    // The trace_event-wrapped shape above and this bare-message-type shape
    // are handled by two SEPARATE branches in handleMessage (this one is
    // what a live single-connection tracer actually sends while a task is
    // running) - a past version of this fix updated only the trace_event
    // branch and missed this one, so this exercises it directly.
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    const bareDagExecutionMessage = (
      timestamp: string,
      data: Record<string, unknown>,
    ): TestWebSocketMessage => ({
      type: "dag_execution",
      timestamp,
      task_id: 1,
      data,
    })

    act(() => {
      onMessage?.(bareDagExecutionMessage("2026-05-27T05:00:02Z", {
        phase: "executing",
        turn_id: "turn-A",
        steps: [{ id: "step-one", name: "Step one", dependencies: [] }],
      }))
    })
    await waitFor(() => {
      expect(screen.getByTestId("steps-count").textContent).toBe("1")
    })
    expect(screen.getByTestId("dag-turn-id").textContent).toBe("turn-A")
    expect(screen.getByTestId("dag-created-at").textContent).toBe("2026-05-27T05:00:02Z")

    act(() => {
      onMessage?.(bareDagExecutionMessage("2026-05-27T05:00:05Z", { phase: "planning", turn_id: "turn-B" }))
    })
    await waitFor(() => {
      expect(screen.getByTestId("dag-turn-id").textContent).toBe("turn-B")
    })
    expect(screen.getByTestId("dag-created-at").textContent).toBe("2026-05-27T05:00:05Z")
    expect(screen.getByTestId("steps-count").textContent).toBe("0")
  })

  it("resets dagExecution/steps once an ordinary new send's ack resolves on an existing DAG task", async () => {
    // The plain-path counterpart to the race-condition test below: no
    // task_completed clone in the middle, just an existing task with a
    // finished DAG plan and a normal new message sent into it - the ack
    // resolving alone must be enough to clear the stale run.
    let acknowledgeDelivery: (() => void) | undefined
    sendChatMessageMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          acknowledgeDelivery = () =>
            resolve({ client_message_id: "turn-2", turn_id: "turn-2" })
        })
    )

    let sendTurnTwo: (() => Promise<void> | undefined) | undefined
    let markTaskOneCompleted: (() => void) | undefined
    function OrdinaryResetProbe() {
      const { sendMessage, dispatch } = useApp()
      sendTurnTwo = () => sendMessage("A plain follow-up", { clientMessageId: "turn-2" })
      markTaskOneCompleted = () => {
        dispatch({
          type: "SET_CURRENT_TASK",
          payload: {
            id: "1",
            title: "Test task",
            status: "completed",
            description: "Test task",
            createdAt: "2026-05-27T05:00:00Z",
            updatedAt: "2026-05-27T05:00:00Z",
          },
        })
      }
      return null
    }

    render(
      <AppProvider token="token">
        <SeedExistingTask />
        <OrdinaryResetProbe />
        <StateProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    act(() => {
      markTaskOneCompleted?.()
      onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        data: {
          event_id: "dag-execution-1",
          event_type: "dag_execution",
          data: {
            phase: "completed",
            turn_id: "turn-1",
            steps: [{ id: "step-one", name: "Step one", dependencies: [] }],
          },
        },
      })
    })
    await waitFor(() => {
      expect(screen.getByTestId("dag-turn-id").textContent).toBe("turn-1")
    })
    expect(screen.getByTestId("steps-count").textContent).toBe("1")

    let delivery: Promise<void> | undefined
    await act(async () => {
      delivery = sendTurnTwo?.()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(sendChatMessageMock).toHaveBeenCalledOnce()

    await act(async () => {
      acknowledgeDelivery?.()
      await delivery
    })
    expect(screen.getByTestId("dag-turn-id").textContent).toBe("")
    expect(screen.getByTestId("steps-count").textContent).toBe("0")
  })

  it("does not wipe a turn_id-less DAG run that arrives before the ack of the send that started it", async () => {
    // turn_id is optional (older backends/patterns never set it), and with
    // no dagExecution at all before the send, both the pre-send capture and
    // the post-ack read of `?.turn_id` are undefined - "undefined ===
    // undefined" can't tell "nothing happened" apart from "a genuinely new,
    // turn_id-less run just arrived". The guard must fall back to object
    // reference equality for that case (the pre-turn_id behavior), which
    // correctly sees the newly-arrived object as new and skips the reset.
    let acknowledgeDelivery: (() => void) | undefined
    sendChatMessageMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          acknowledgeDelivery = () =>
            resolve({ client_message_id: "turn-legacy", turn_id: "turn-legacy" })
        })
    )

    let sendTurn: (() => Promise<void> | undefined) | undefined
    let markTaskOneCompleted: (() => void) | undefined
    function LegacyRunProbe() {
      const { sendMessage, dispatch } = useApp()
      sendTurn = () => sendMessage("Start a DAG task", { clientMessageId: "turn-legacy" })
      markTaskOneCompleted = () => {
        dispatch({
          type: "SET_CURRENT_TASK",
          payload: {
            id: "1",
            title: "Test task",
            status: "completed",
            description: "Test task",
            createdAt: "2026-05-27T05:00:00Z",
            updatedAt: "2026-05-27T05:00:00Z",
          },
        })
      }
      return null
    }

    render(
      <AppProvider token="token">
        <SeedExistingTask />
        <LegacyRunProbe />
        <StateProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    // Terminal status so the send exercises the reset path (a running/paused
    // task short-circuits it as a continuation), and NO dagExecution at all.
    act(() => {
      markTaskOneCompleted?.()
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("completed")
    })
    expect(screen.getByTestId("dag-phase").textContent).toBe("")

    let delivery: Promise<void> | undefined
    await act(async () => {
      delivery = sendTurn?.()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(sendChatMessageMock).toHaveBeenCalledOnce()

    // The new turn's own DAG run starts broadcasting before the ack resolves,
    // from a backend that doesn't set turn_id.
    act(() => {
      onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        data: {
          event_id: "dag-execution-legacy-1",
          event_type: "dag_execution",
          data: {
            phase: "executing",
            steps: [{ id: "step-one", name: "Step one", dependencies: [] }],
          },
        },
      })
    })
    await waitFor(() => {
      expect(screen.getByTestId("steps-count").textContent).toBe("1")
    })

    await act(async () => {
      acknowledgeDelivery?.()
      await delivery
    })
    // The freshly-arrived run survives the ack's reset attempt.
    expect(screen.getByTestId("dag-phase").textContent).toBe("executing")
    expect(screen.getByTestId("steps-count").textContent).toBe("1")
  })

  it("still resets stale dagExecution when a task_completed clone races ahead of a new turn's ack", async () => {
    // Regression coverage for the sendMessage reset guard: it used to compare
    // dagExecution by OBJECT REFERENCE, which the (unrelated, unmodified)
    // task_completed handler defeats - that handler unconditionally clones
    // currentState.dagExecution into a NEW object whenever it's truthy, even
    // for a turn that isn't a DAG turn at all. If that clone lands before this
    // send's own ack resolves, reference equality would wrongly read as
    // "something new arrived" and skip the reset, leaving turn 1's stale
    // plan/steps stuck in view. Comparing turn_id instead of the object
    // itself isn't fooled by the clone, since the id doesn't change.
    let acknowledgeDelivery: (() => void) | undefined
    sendChatMessageMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          acknowledgeDelivery = () =>
            resolve({ client_message_id: "turn-2", turn_id: "turn-2" })
        })
    )

    let sendTurnTwo: (() => Promise<void> | undefined) | undefined
    let markTaskOneCompleted: (() => void) | undefined
    function StaleCloneProbe() {
      const { sendMessage, dispatch } = useApp()
      sendTurnTwo = () => sendMessage("A plain follow-up", { clientMessageId: "turn-2" })
      markTaskOneCompleted = () => {
        dispatch({
          type: "SET_CURRENT_TASK",
          payload: {
            id: "1",
            title: "Test task",
            status: "completed",
            description: "Test task",
            createdAt: "2026-05-27T05:00:00Z",
            updatedAt: "2026-05-27T05:00:00Z",
          },
        })
      }
      return null
    }

    render(
      <AppProvider token="token">
        <SeedExistingTask />
        <StaleCloneProbe />
        <StateProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    // Turn 1's DAG run finished, and the task is now done - the state a
    // second, unrelated turn would be sent into.
    act(() => {
      markTaskOneCompleted?.()
      onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        data: {
          event_id: "dag-execution-1",
          event_type: "dag_execution",
          data: {
            phase: "completed",
            turn_id: "turn-1",
            steps: [{ id: "step-one", name: "Step one", dependencies: [] }],
          },
        },
      })
    })
    await waitFor(() => {
      expect(screen.getByTestId("dag-turn-id").textContent).toBe("turn-1")
    })
    expect(screen.getByTestId("steps-count").textContent).toBe("1")

    let delivery: Promise<void> | undefined
    await act(async () => {
      delivery = sendTurnTwo?.()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(sendChatMessageMock).toHaveBeenCalledOnce()

    // Before turn 2's ack resolves, a task_completed broadcast for task 1
    // arrives (e.g. a duplicate/replayed completion notice) - the existing
    // task_completed handler clones the still-turn-1 dagExecution into a new
    // object purely to sync its phase, without changing its turn_id.
    act(() => {
      onMessage?.({
        type: "task_completed",
        timestamp: "2026-05-27T05:00:03Z",
        task_id: 1,
        task: { id: 1, status: "completed" },
        success: true,
      } as TestWebSocketMessage)
    })
    await waitFor(() => {
      expect(screen.getByTestId("dag-phase").textContent).toBe("completed")
    })
    // Still turn 1's data - the clone didn't introduce a new run.
    expect(screen.getByTestId("dag-turn-id").textContent).toBe("turn-1")
    expect(screen.getByTestId("steps-count").textContent).toBe("1")

    // Turn 2's ack now resolves - since nothing turn-2-specific arrived (no
    // new turn_id), the guard must still recognize turn 1's data as stale and
    // reset it, despite the object having been recloned in between.
    await act(async () => {
      acknowledgeDelivery?.()
      await delivery
    })
    expect(screen.getByTestId("dag-turn-id").textContent).toBe("")
    expect(screen.getByTestId("steps-count").textContent).toBe("0")
  })

  it("still resets a turn_id-LESS stale dagExecution when a task_completed clone races ahead of the ack", async () => {
    // The no-turn_id twin of the clone-race test above: with no turn_id on
    // either side, the reset guard falls back to comparing created_at (NOT
    // object reference - the clone changes the reference while keeping
    // created_at, which is exactly how a reference fallback wrongly read the
    // clone as "something new arrived" and skipped the reset).
    let acknowledgeDelivery: (() => void) | undefined
    sendChatMessageMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          acknowledgeDelivery = () =>
            resolve({ client_message_id: "turn-2", turn_id: "turn-2" })
        })
    )

    let sendTurnTwo: (() => Promise<void> | undefined) | undefined
    let markTaskOneCompleted: (() => void) | undefined
    function LegacyStaleCloneProbe() {
      const { sendMessage, dispatch } = useApp()
      sendTurnTwo = () => sendMessage("A plain follow-up", { clientMessageId: "turn-2" })
      markTaskOneCompleted = () => {
        dispatch({
          type: "SET_CURRENT_TASK",
          payload: {
            id: "1",
            title: "Test task",
            status: "completed",
            description: "Test task",
            createdAt: "2026-05-27T05:00:00Z",
            updatedAt: "2026-05-27T05:00:00Z",
          },
        })
      }
      return null
    }

    render(
      <AppProvider token="token">
        <SeedExistingTask />
        <LegacyStaleCloneProbe />
        <StateProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    // Turn 1's DAG finished - as a legacy event with NO turn_id.
    act(() => {
      markTaskOneCompleted?.()
      onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        data: {
          event_id: "dag-execution-1",
          event_type: "dag_execution",
          data: {
            phase: "completed",
            steps: [{ id: "step-one", name: "Step one", dependencies: [] }],
          },
        },
      })
    })
    await waitFor(() => {
      expect(screen.getByTestId("steps-count").textContent).toBe("1")
    })
    expect(screen.getByTestId("dag-turn-id").textContent).toBe("")

    let delivery: Promise<void> | undefined
    await act(async () => {
      delivery = sendTurnTwo?.()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(sendChatMessageMock).toHaveBeenCalledOnce()

    // The racing task_completed clones the (no-id) dagExecution into a new
    // object before the ack resolves.
    act(() => {
      onMessage?.({
        type: "task_completed",
        timestamp: "2026-05-27T05:00:03Z",
        task_id: 1,
        task: { id: 1, status: "completed" },
        success: true,
      } as TestWebSocketMessage)
    })
    await waitFor(() => {
      expect(screen.getByTestId("dag-phase").textContent).toBe("completed")
    })
    expect(screen.getByTestId("steps-count").textContent).toBe("1")

    await act(async () => {
      acknowledgeDelivery?.()
      await delivery
    })
    expect(screen.getByTestId("dag-phase").textContent).toBe("")
    expect(screen.getByTestId("steps-count").textContent).toBe("0")
  })

  it("does not wipe a turn_id-less dagExecution that newly arrived from nothing during the ack await", async () => {
    // The case the no-turn_id fallback exists to protect: nothing was in
    // state when the send began, and the new turn's own (legacy, no-id) DAG
    // events land before the ack resolves. created_at going from undefined
    // to a value reads as "something new arrived" - the reset must skip.
    let acknowledgeDelivery: (() => void) | undefined
    sendChatMessageMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          acknowledgeDelivery = () =>
            resolve({ client_message_id: "turn-2", turn_id: "turn-2" })
        })
    )

    let sendTurnTwo: (() => Promise<void> | undefined) | undefined
    let markTaskOneCompleted: (() => void) | undefined
    function FreshArrivalProbe() {
      const { sendMessage, dispatch } = useApp()
      sendTurnTwo = () => sendMessage("Kick off a new run", { clientMessageId: "turn-2" })
      markTaskOneCompleted = () => {
        dispatch({
          type: "SET_CURRENT_TASK",
          payload: {
            id: "1",
            title: "Test task",
            status: "completed",
            description: "Test task",
            createdAt: "2026-05-27T05:00:00Z",
            updatedAt: "2026-05-27T05:00:00Z",
          },
        })
      }
      return null
    }

    render(
      <AppProvider token="token">
        <SeedExistingTask />
        <FreshArrivalProbe />
        <StateProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    act(() => {
      markTaskOneCompleted?.()
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("completed")
    })
    expect(screen.getByTestId("dag-phase").textContent).toBe("")

    let delivery: Promise<void> | undefined
    await act(async () => {
      delivery = sendTurnTwo?.()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(sendChatMessageMock).toHaveBeenCalledOnce()

    // The new turn's own DAG starts broadcasting (legacy shape, no turn_id)
    // before the ack resolves.
    act(() => {
      onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:00:05Z",
        task_id: 1,
        data: {
          event_id: "dag-execution-new",
          event_type: "dag_execution",
          data: {
            phase: "planning",
          },
        },
      })
    })
    await waitFor(() => {
      expect(screen.getByTestId("dag-phase").textContent).toBe("planning")
    })

    await act(async () => {
      acknowledgeDelivery?.()
      await delivery
    })
    // The freshly-arrived run survived the ack's reset attempt.
    expect(screen.getByTestId("dag-phase").textContent).toBe("planning")
    expect(screen.getByTestId("dag-created-at").textContent).toBe("2026-05-27T05:00:05Z")
  })

  it("does not flip a task back to running when its own terminal event landed during the ack await", async () => {
    // The optimistic terminal->running flip is a pre-send GUESS - a terminal
    // event accepted during the await (a fast non-DAG turn completing before
    // the ack resolves) is newer truth. Flipping over it would mark the
    // finished run as running again and clear the dagTerminatedAt the
    // reducer just stamped, unfreezing the Progress panel's elapsed clock.
    let acknowledgeDelivery: (() => void) | undefined
    sendChatMessageMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          acknowledgeDelivery = () =>
            resolve({ client_message_id: "turn-2", turn_id: "turn-2" })
        })
    )

    let sendTurnTwo: (() => Promise<void> | undefined) | undefined
    let markTaskOneFailed: (() => void) | undefined
    function TerminalDuringAwaitProbe() {
      const { sendMessage, dispatch } = useApp()
      sendTurnTwo = () => sendMessage("Run it again", { clientMessageId: "turn-2" })
      markTaskOneFailed = () => {
        dispatch({
          type: "SET_CURRENT_TASK",
          payload: {
            id: "1",
            title: "Test task",
            status: "failed",
            description: "Test task",
            createdAt: "2026-05-27T05:00:00Z",
            updatedAt: "2026-05-27T05:00:00Z",
          },
        })
      }
      return null
    }

    render(
      <AppProvider token="token">
        <SeedExistingTask />
        <TerminalDuringAwaitProbe />
        <StateProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    act(() => {
      markTaskOneFailed?.()
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("failed")
    })

    let delivery: Promise<void> | undefined
    await act(async () => {
      delivery = sendTurnTwo?.()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(sendChatMessageMock).toHaveBeenCalledOnce()

    // The new turn completes (fast non-DAG run) BEFORE the ack resolves.
    act(() => {
      onMessage?.({
        type: "task_completed",
        timestamp: "2026-05-27T05:00:07Z",
        task_id: 1,
        task: { id: 1, status: "completed" },
        success: true,
      } as TestWebSocketMessage)
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("completed")
    })
    expect(screen.getByTestId("task-dag-terminated-at").textContent).toBe("2026-05-27T05:00:07Z")

    await act(async () => {
      acknowledgeDelivery?.()
      await delivery
    })
    // The ack must NOT override the newer terminal truth.
    expect(screen.getByTestId("task-status").textContent).toBe("completed")
    expect(screen.getByTestId("task-dag-terminated-at").textContent).toBe("2026-05-27T05:00:07Z")
  })

  it("lets a terminal event's own timestamp replace a provisional task-metadata dagTerminatedAt", async () => {
    // Cold history load: the Task row arrives first, already terminal, and
    // dagTerminatedAt gets backfilled from mutable updatedAt (which may by
    // then reflect a post-execution title edit, NOT the real end time). The
    // replayed terminal event knows the real end time - it must replace the
    // provisional guess, while a second terminal delivery afterwards must
    // NOT push the now-authoritative value forward (first-write-wins).
    let seedColdCompletedTask: (() => void) | undefined
    function ColdHistoryProbe() {
      const { dispatch } = useApp()
      seedColdCompletedTask = () => {
        dispatch({
          type: "SET_CURRENT_TASK",
          payload: {
            id: "1",
            title: "Test task (renamed after completion)",
            status: "completed",
            description: "Test task",
            createdAt: "2026-05-27T05:00:00Z",
            // A post-execution title edit bumped this well past the run's
            // actual 05:01:00 end.
            updatedAt: "2026-05-27T09:00:00Z",
          },
        })
      }
      return null
    }

    render(
      <AppProvider token="token">
        <SeedExistingTask />
        <ColdHistoryProbe />
        <StateProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    act(() => {
      seedColdCompletedTask?.()
    })
    // Backfilled (provisionally) from updatedAt - the only value available
    // before any terminal event replays.
    await waitFor(() => {
      expect(screen.getByTestId("task-dag-terminated-at").textContent).toBe("2026-05-27T09:00:00Z")
    })

    // The terminal event replays with the run's REAL end time.
    act(() => {
      onMessage?.({
        type: "task_completed",
        timestamp: "2026-05-27T05:01:00Z",
        task_id: 1,
        task: { id: 1, status: "completed" },
        success: true,
      } as TestWebSocketMessage)
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-dag-terminated-at").textContent).toBe("2026-05-27T05:01:00Z")
    })

    // A duplicate terminal delivery must not move it again.
    act(() => {
      onMessage?.({
        type: "task_completed",
        timestamp: "2026-05-27T05:02:30Z",
        task_id: 1,
        task: { id: 1, status: "completed" },
        success: true,
      } as TestWebSocketMessage)
    })
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(screen.getByTestId("task-dag-terminated-at").textContent).toBe("2026-05-27T05:01:00Z")
  })

  it("lets a task-level trace_error replace a provisional dagTerminatedAt on an ALREADY-failed task, but never decides the outcome itself", async () => {
    // trace_error alone must never be what marks a task failed (a global
    // trace_error can be logged without the task actually stopping) - it
    // only backstops the TIMESTAMP once the task is independently already
    // known failed (task_info established that on cold history load,
    // backfilling dagTerminatedAt from mutable updatedAt as a provisional
    // guess) and no proper terminal broadcast ever arrived to replace it.
    let seedColdFailedTask: (() => void) | undefined
    let seedColdRunningTask: (() => void) | undefined
    function ColdTraceErrorProbe() {
      const { dispatch } = useApp()
      seedColdFailedTask = () => {
        dispatch({
          type: "SET_CURRENT_TASK",
          payload: {
            id: "1",
            title: "Test task",
            status: "failed",
            description: "Test task",
            createdAt: "2026-05-27T05:00:00Z",
            updatedAt: "2026-05-27T09:00:00Z",
          },
        })
      }
      seedColdRunningTask = () => {
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
      }
      return null
    }

    render(
      <AppProvider token="token">
        <SeedExistingTask />
        <ColdTraceErrorProbe />
        <StateProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    // A task-level trace_error arriving while the task is still RUNNING must
    // not flip it to failed - some other event decides that, not this one.
    act(() => {
      seedColdRunningTask?.()
      onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:00:30Z",
        task_id: 1,
        data: {
          event_id: "trace-error-recoverable",
          event_type: "trace_error",
          data: { error_message: "a recoverable hiccup" },
        },
      })
    })
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(screen.getByTestId("task-status").textContent).toBe("running")
    expect(screen.getByTestId("task-dag-terminated-at").textContent).toBe("")

    // Now the task is independently known failed (cold history's task_info),
    // with a provisional dagTerminatedAt backfilled from mutable updatedAt.
    act(() => {
      seedColdFailedTask?.()
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-dag-terminated-at").textContent).toBe("2026-05-27T09:00:00Z")
    })

    // The replayed global trace_error carries the REAL failure timestamp -
    // it must replace the provisional backfill.
    act(() => {
      onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:01:15Z",
        task_id: 1,
        data: {
          event_id: "trace-error-global",
          event_type: "trace_error",
          data: { error_message: "fatal error" },
        },
      })
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-dag-terminated-at").textContent).toBe("2026-05-27T05:01:15Z")
    })

    // A second trace_error afterward must not push the now-authoritative
    // value forward (first-write-wins).
    act(() => {
      onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:03:00Z",
        task_id: 1,
        data: {
          event_id: "trace-error-duplicate",
          event_type: "trace_error",
          data: { error_message: "fatal error" },
        },
      })
    })
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(screen.getByTestId("task-dag-terminated-at").textContent).toBe("2026-05-27T05:01:15Z")
  })

  it("does not let replayed liveness events flip a terminal task back to running during a history load", async () => {
    // Replayed activity events (llm_call_start & co) optimistically infer
    // "running" from the fact that work is happening - but during a history
    // load that work already finished. Flipping the authoritative terminal
    // status would clear dagTerminatedAt and let the Progress panel auto-open
    // for a task that is merely being VIEWED.
    let seedCompletedTask: (() => void) | undefined
    function TerminalHistoryProbe() {
      const { dispatch } = useApp()
      seedCompletedTask = () => {
        dispatch({
          type: "SET_CURRENT_TASK",
          payload: {
            id: "1",
            title: "Test task",
            status: "completed",
            description: "Test task",
            createdAt: "2026-05-27T05:00:00Z",
            updatedAt: "2026-05-27T05:01:00Z",
          },
        })
      }
      return null
    }

    render(
      <AppProvider token="token">
        <SeedExistingTask />
        <TerminalHistoryProbe />
        <StateProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    act(() => {
      seedCompletedTask?.()
      // Connecting is what arms the history load (isHistoricalDataLoadingRef).
      webSocketOptions.current?.onConnect?.()
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("completed")
    })

    // A replayed llm_call_start lands mid-history-load.
    act(() => {
      onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:00:30Z",
        task_id: 1,
        data: {
          event_id: "llm-start-replayed",
          event_type: "llm_call_start",
          data: { model_name: "test-model" },
        },
      })
    })
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(screen.getByTestId("task-status").textContent).toBe("completed")
    expect(screen.getByTestId("task-dag-terminated-at").textContent).toBe("2026-05-27T05:01:00Z")

    // Once the history load actually completes, LIVE activity events must
    // still be able to flip a (genuinely re-run) terminal task to running -
    // the gate is scoped to the load, not to terminal tasks forever.
    act(() => {
      onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:02:00Z",
        task_id: 1,
        data: {
          event_id: "history-done",
          event_type: "historical_data_complete",
          data: {},
        },
      })
    })
    act(() => {
      onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:02:01Z",
        task_id: 1,
        data: {
          event_id: "llm-start-live",
          event_type: "llm_call_start",
          data: { model_name: "test-model" },
        },
      })
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("running")
    })
  })

  it("drops a dag_execution update whose phase is absent or unknown instead of synthesizing an active run", async () => {
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )
    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    // A bare legacy dag_execution frame with an EMPTY payload - an earlier
    // version normalized this to phase "executing", conjuring a blank
    // in-progress DAG (and an auto-opened Progress panel) out of nothing.
    act(() => {
      onMessage?.({
        type: "dag_execution",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        data: {},
      } as TestWebSocketMessage)
    })
    // An unknown future/malformed phase string must be dropped too.
    act(() => {
      onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:00:03Z",
        task_id: 1,
        data: {
          event_id: "dag-execution-mystery",
          event_type: "dag_execution",
          data: { phase: "mystery_phase", steps: [{ id: "s1", name: "Step", dependencies: [] }] },
        },
      })
    })
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(screen.getByTestId("dag-phase").textContent).toBe("")
    expect(screen.getByTestId("steps-count").textContent).toBe("0")

    // A well-formed update right after still applies - the drop is per-frame,
    // not sticky.
    act(() => {
      onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:00:04Z",
        task_id: 1,
        data: {
          event_id: "dag-execution-valid",
          event_type: "dag_execution",
          data: { phase: "executing", turn_id: "turn-A" },
        },
      })
    })
    await waitFor(() => {
      expect(screen.getByTestId("dag-phase").textContent).toBe("executing")
    })
  })

  it("clears dagExecution/steps when switching to a different task", async () => {
    let switchTask: (() => void) | undefined
    function SwitchingTaskProbe() {
      const { setTaskId } = useApp()
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
    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    act(() => {
      dagBurst(1).forEach((message) => onMessage?.(message))
    })
    await waitFor(() => {
      expect(screen.getByTestId("steps-count").textContent).toBe("1")
    })
    expect(screen.getByTestId("dag-phase").textContent).toBe("failed")

    act(() => {
      switchTask?.()
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-id").textContent).toBe("2")
    })
    expect(screen.getByTestId("dag-phase").textContent).toBe("")
    expect(screen.getByTestId("steps-count").textContent).toBe("0")
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
          error_code: "task_execution_failed",
          message: "provider token=secret",
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
        "clientErrors.taskExecutionFailed"
      )
      expect(screen.getByTestId("messages").textContent).not.toContain("token=secret")
    })
  })

  it.each(
    [
      { name: "object", value: { code: "task_execution_failed" } },
      { name: "number", value: 42 },
      { name: "array", value: ["task_execution_failed"] },
      { name: "boolean", value: true },
      { name: "null", value: null },
    ].flatMap(({ name, value }) => [
      { location: "nested", name, value },
      { location: "root", name, value },
    ]),
  )("fails closed for a $location $name WebSocket error code", async ({ location, value }) => {
    render(
      <AppProvider token="token">
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    act(() => {
      onMessage?.(location === "nested"
        ? {
            type: "agent_error",
            timestamp: "2026-05-27T05:00:05Z",
            data: {
              type: "agent_error",
              error_code: value,
              message: "provider token=secret",
            },
          }
        : {
            type: "agent_error",
            timestamp: "2026-05-27T05:00:05Z",
            error_code: value,
            message: "provider token=secret",
          })
    })

    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain("Unknown error")
    })
    expect(screen.getByTestId("messages").textContent).not.toContain("token=secret")
  })

  it.each(
    (["error", "agent_error", "task_error"] as const).flatMap((eventType) => [
      { eventType, location: "nested" as const },
      { eventType, location: "root" as const },
    ]),
  )("hides absent-code $eventType prose at the untrusted $location boundary", async ({ eventType, location }) => {
    render(
      <AppProvider
        token="public-token"
        transport={{ legacyErrorProse: "untrusted" }}
      >
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()
    expect(webSocketOptions.current?.legacyErrorProse).toBe("untrusted")
    const secret = "provider=openai path=/srv/private token=secret"

    act(() => {
      onMessage?.(location === "nested"
        ? {
            type: eventType,
            timestamp: "2026-05-27T05:00:05Z",
            data: { type: eventType, message: secret },
          }
        : {
            type: eventType,
            timestamp: "2026-05-27T05:00:05Z",
            message: secret,
          } as TestWebSocketMessage)
    })

    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain("Unknown error")
    })
    expect(screen.getByTestId("messages").textContent).not.toContain(secret)
  })

  it("preserves absent-code legacy prose for the authenticated default transport", async () => {
    render(
      <AppProvider token="authenticated-token">
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()
    expect(webSocketOptions.current?.legacyErrorProse).toBe("trusted")

    act(() => {
      onMessage?.({
        type: "agent_error",
        timestamp: "2026-05-27T05:00:05Z",
        data: {
          type: "agent_error",
          message: "Legacy actionable authenticated error",
        },
      })
    })

    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "Legacy actionable authenticated error",
      )
    })
  })

  it.each(
    (["error", "agent_error", "task_error"] as const).flatMap((eventType) => [
      { eventType, location: "nested" as const },
      { eventType, location: "root" as const },
    ]),
  )("localizes recognized $location codes for untrusted $eventType events", async ({ eventType, location }) => {
    render(
      <AppProvider
        token="public-token"
        transport={{ legacyErrorProse: "untrusted" }}
      >
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    act(() => {
      onMessage?.(location === "nested"
        ? {
            type: eventType,
            timestamp: "2026-05-27T05:00:05Z",
            data: {
              type: eventType,
              error_code: "task_execution_failed",
              message: "provider token=secret",
            },
          }
        : {
            type: eventType,
            timestamp: "2026-05-27T05:00:05Z",
            error_code: "task_execution_failed",
            message: "provider token=secret",
          } as TestWebSocketMessage)
    })

    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "clientErrors.taskExecutionFailed",
      )
    })
    expect(screen.getByTestId("messages").textContent).not.toContain("token=secret")
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
        question: "Which file should I use?",
        interactions: [],
        request_id: "inputreq_0011223344556677889900aabbccddee",
      } as TestWebSocketMessage)
    })

    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe(
        "waiting_for_user"
      )
      expect(screen.getByTestId("processing").textContent).toBe("false")
      expect(screen.getByTestId("messages").textContent).toContain(
        "Which file should I use?"
      )
      expect(screen.getByTestId("messages").textContent).toContain(
        "inputreq_0011223344556677889900aabbccddee"
      )
      expect(screen.getByTestId("waiting-request-id").textContent).toBe(
        "inputreq_0011223344556677889900aabbccddee"
      )
    })
  })

  it("keeps the request identity when paired waiting producers publish the same prompt", async () => {
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("running")
    })

    act(() => {
      webSocketOptions.current?.onMessage?.({
        type: "trace_event",
        timestamp: "2026-05-27T05:00:05Z",
        task_id: 1,
        data: {
          event_id: "agent-r1",
          event_type: "agent_message",
          data: {
            message: "Which file should I use?",
            content: "Which file should I use?",
            role: "assistant",
            expect_response: true,
            request_id: "inputreq_r1",
          },
        },
      })
      webSocketOptions.current?.onMessage?.({
        type: "task_waiting_for_user",
        timestamp: "2026-05-27T05:00:06Z",
        task_id: 1,
        question: "Which file should I use?",
        interactions: [],
        request_id: "inputreq_r1",
      } as TestWebSocketMessage)
    })

    await waitFor(() => {
      const rendered = JSON.parse(
        screen.getByTestId("messages").textContent || "[]",
      ) as Array<{
        role: string
        content: string
        interactionRequestId?: string
      }>
      const matchingAssistantMessages = rendered.filter(
        (message) =>
          message.role === "assistant" &&
          message.content === "Which file should I use?",
      )
      expect(matchingAssistantMessages).toEqual([expect.objectContaining({
        interactionRequestId: "inputreq_r1",
      })])
    })
  })

  const renderRunningTaskProbe = async () => {
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )
    await waitFor(() => expect(screen.getByTestId("task-status").textContent).toBe("running"))
  }

  const publishTraceEvent = (
    event_type: string,
    data: Record<string, unknown>,
    timestamp = "2026-05-27T05:00:05Z",
  ) => webSocketOptions.current?.onMessage?.({
    type: "trace_event",
    timestamp,
    task_id: 1,
    data: { event_type, data },
  })

  const publishWaitingOccurrence = (
    prompt: string,
    requestId?: string,
    timestamp = "2026-05-27T05:00:05Z",
  ) => {
    publishTraceEvent("agent_message", {
      message: prompt, role: "assistant", expect_response: true, request_id: requestId,
    }, timestamp)
    webSocketOptions.current?.onMessage?.({
      type: "task_waiting_for_user",
      timestamp,
      task_id: 1,
      question: prompt,
      interactions: [],
      request_id: requestId,
    } as TestWebSocketMessage)
  }

  const assistantMessagesFor = (prompt: string) => (
    JSON.parse(screen.getByTestId("messages").textContent || "[]") as Array<{
      role: string
      content: string
      interactionRequestId?: string
    }>
  ).filter((message) => message.role === "assistant" && message.content === prompt)

  it("keeps identical prompts from different waiting occurrences separate", async () => {
    await renderRunningTaskProbe()
    act(() => {
      publishWaitingOccurrence("Which file should I use?", "inputreq_r1")
      publishWaitingOccurrence("Which file should I use?", "inputreq_r2", "2026-05-27T05:00:07Z")
    })

    await waitFor(() => {
      expect(assistantMessagesFor("Which file should I use?")).toEqual([
        expect.objectContaining({ interactionRequestId: "inputreq_r1" }),
        expect.objectContaining({ interactionRequestId: "inputreq_r2" }),
      ])
      expect(screen.getByTestId("waiting-request-id").textContent).toBe(
        "inputreq_r2",
      )
    })
  })

  it("keeps ai_message dedupe independent of request identity", async () => {
    await renderRunningTaskProbe()
    act(() => {
      for (const request_id of ["inputreq_r1", "inputreq_r2"]) {
        publishTraceEvent("ai_message", {
          message: "Same final answer", role: "assistant", request_id,
        })
      }
    })
    expect(assistantMessagesFor("Same final answer")).toHaveLength(1)
  })

  it("keeps delimiter-like legacy content distinct from an identified occurrence", async () => {
    await renderRunningTaskProbe()
    act(() => {
      publishTraceEvent("agent_message", {
        message: "Choose file:occurrence:inputreq_r2", role: "assistant",
      })
      publishWaitingOccurrence("Choose file", "inputreq_r2")
    })
    expect(assistantMessagesFor("Choose file:occurrence:inputreq_r2")).toHaveLength(1)
    expect(assistantMessagesFor("Choose file")).toEqual([
      expect.objectContaining({ interactionRequestId: "inputreq_r2" }),
    ])
  })

  it("continues to content-dedupe id-less legacy waiting siblings", async () => {
    await renderRunningTaskProbe()
    act(() => publishWaitingOccurrence("Which legacy file should I use?"))
    await waitFor(() => expect(
      assistantMessagesFor("Which legacy file should I use?"),
    ).toHaveLength(1))
  })

  it("lets the latest identified occurrence suppress an id-less legacy sibling for 30 seconds", async () => {
    await renderRunningTaskProbe()
    const prompt = "Which compatibility file should I use?"
    vi.useFakeTimers()
    try {
      act(() => {
        publishWaitingOccurrence(prompt, "inputreq_r1")
        vi.advanceTimersByTime(20_000)
        publishWaitingOccurrence(prompt, "inputreq_r2")
        vi.advanceTimersByTime(11_000)
        publishTraceEvent("react_task_end", {
          result: { status: "waiting_for_user", message: prompt },
        })
      })

      expect(assistantMessagesFor(prompt)).toEqual([
        expect.objectContaining({ interactionRequestId: "inputreq_r1" }),
        expect.objectContaining({ interactionRequestId: "inputreq_r2" }),
      ])
    } finally {
      vi.useRealTimers()
    }
  })

  it("keeps a projected waiting occurrence together over stale nested legacy fields", async () => {
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )

    await waitFor(() => expect(screen.getByTestId("task-status").textContent).toBe("running"))

    act(() => {
      webSocketOptions.current?.onMessage?.({
        type: "task_waiting_for_user",
        timestamp: "2026-05-27T05:00:06Z",
        question: "Which projected hotel should I use?",
        interactions: [{ type: "text_input", field: "hotel", label: "Hotel" }],
        request_id: "inputreq_0011223344556677889900aabbccddee",
        data: {
          question: "Which stale city should I use?",
          interactions: [{ type: "text_input", field: "city", label: "City" }],
        },
      } as TestWebSocketMessage)
    })

    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain("Which projected hotel should I use?")
      expect(screen.getByTestId("messages").textContent).not.toContain("Which stale city should I use?")
      expect(screen.getByTestId("waiting-interactions").textContent).toContain("hotel")
      expect(screen.getByTestId("waiting-interactions").textContent).not.toContain("city")
      expect(screen.getByTestId("waiting-request-id").textContent).toBe(
        "inputreq_0011223344556677889900aabbccddee"
      )
    })
  })

  it("replaces structured interactions when a concurrent wait becomes text-only", async () => {
    await renderRunningTaskProbe()

    act(() => {
      webSocketOptions.current?.onMessage?.({
        type: "task_waiting_for_user",
        timestamp: "2026-05-27T05:00:06Z",
        task_id: 1,
        question: "Which city should I use?",
        interactions: [{ type: "text_input", field: "city", label: "City" }],
        request_id: "inputreq_r1",
      } as TestWebSocketMessage)
      webSocketOptions.current?.onMessage?.({
        type: "task_waiting_for_user",
        timestamp: "2026-05-27T05:00:07Z",
        task_id: 1,
        question: "Anything else I should know?",
        interactions: [],
        request_id: "inputreq_r2",
      } as TestWebSocketMessage)
    })

    await waitFor(() => {
      expect(screen.getByTestId("waiting-request-id").textContent).toBe(
        "inputreq_r2"
      )
      expect(screen.getByTestId("waiting-interactions").textContent).toBe("[]")
      expect(assistantMessagesFor("Anything else I should know?")).toEqual([
        expect.objectContaining({ interactionRequestId: "inputreq_r2" }),
      ])
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
      "session-turn-1",
      undefined,
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

  it("forwards the rendered interaction request id to the Session socket", async () => {
    const transport = makeSessionTransport()
    render(
      <AppProvider token="token" transport={transport}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    act(() => webSocketOptions.current?.onConnect?.())
    act(() => webSocketOptions.current?.onMessage?.(taskInfoMessage(42)))
    await waitFor(() => expect(screen.getByTestId("task-id").textContent).toBe("42"))

    await act(async () => {
      await getSessionControls().sendMessage("City: Sydney", {
        clientMessageId: "answer-q1",
        metadata: {
          request_id: "inputreq_0011223344556677889900aabbccddee",
        },
      })
    })

    expect(sendChatMessageMock).toHaveBeenLastCalledWith(
      "City: Sydney",
      undefined,
      undefined,
      "answer-q1",
      "inputreq_0011223344556677889900aabbccddee",
    )
  })

  it("forwards taskless Session files so unsupported delivery rejects instead of sending text alone", async () => {
    const transport = makeSessionTransport()
    // Exercise the branch-level contract independently from today's
    // intentionally literal-disabled Session descriptor.
    const enabledTransport = {
      ...transport,
      session: {
        ...transport.session,
        files: "enabled",
      },
    } as unknown as AppProviderTransportConfig
    const file = new File(["secret"], "secret.txt", { type: "text/plain" })
    const deliveryError = Object.assign(
      new Error("File delivery is not supported for this connection."),
      { disposition: "not_sent" as const },
    )
    sendChatMessageMock.mockImplementationOnce(async (_message: string, files?: File[]) => {
      if (files?.length) throw deliveryError
      return {
        client_message_id: "session-file-turn",
        turn_id: "session-file-turn",
      }
    })

    render(
      <AppProvider token="token" transport={enabledTransport}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )

    expect(screen.getByTestId("files-disabled").textContent).toBe("false")
    await expect(
      getSessionControls().sendMessage(
        "Read the attached file",
        { clientMessageId: "session-file-turn" },
        [file],
      )
    ).rejects.toMatchObject({ disposition: "not_sent" })
    expect(sendChatMessageMock).toHaveBeenCalledWith(
      "Read the attached file",
      [file],
      undefined,
      "session-file-turn",
      undefined,
    )
    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(transport.uploadFiles).not.toHaveBeenCalled()
    expect(screen.getByTestId("messages").textContent).not.toContain(
      "Read the attached file"
    )
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
      "session-turn-after-reset",
      undefined,
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

  it("replaces a cached waiting identity with the request replayed after reconnect", async () => {
    const { rerender } = render(
      <AppProvider token="token" transport={makeSessionTransport(makeSessionConnection("waiting-old"))}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    act(() => {
      webSocketOptions.current?.onMessage?.(taskInfoMessage(103))
      webSocketOptions.current?.onMessage?.({
        type: "task_waiting_for_user",
        timestamp: "2026-05-27T05:00:05Z",
        task_id: 103,
        question: "Which city?",
        interactions: [],
        request_id: "inputreq_0011223344556677889900aabbccddee",
      } as TestWebSocketMessage)
    })
    await waitFor(() => {
      expect(screen.getByTestId("waiting-request-id").textContent).toBe(
        "inputreq_0011223344556677889900aabbccddee"
      )
    })

    rerender(
      <AppProvider token="token" transport={makeSessionTransport(makeSessionConnection("waiting-new"))}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    act(() => {
      webSocketOptions.current?.onMessage?.(taskInfoMessage(103))
      webSocketOptions.current?.onMessage?.({
        type: "task_waiting_for_user",
        timestamp: "2026-05-27T05:00:06Z",
        task_id: 103,
        question: "Which hotel?",
        interactions: [],
        request_id: "inputreq_ffeeddccbbaa00998877665544332211",
      } as TestWebSocketMessage)
    })

    await waitFor(() => {
      expect(screen.getByTestId("session-conversation-state").textContent).toBe("bound")
      expect(screen.getByTestId("waiting-request-id").textContent).toBe(
        "inputreq_ffeeddccbbaa00998877665544332211"
      )
      expect(screen.getByTestId("messages").textContent).toContain(
        "inputreq_ffeeddccbbaa00998877665544332211"
      )
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

  it("requires reload before replacing a rebound Session task the server cannot replay", async () => {
    const { rerender } = render(
      <AppProvider token="token" transport={makeSessionTransport(makeSessionConnection("unreplayable-old"))}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    act(() => {
      webSocketOptions.current?.onMessage?.(taskInfoMessage(102))
      webSocketOptions.current?.onMessage?.(assistantMessage("Preserved transcript", 102))
    })

    rerender(
      <AppProvider token="token" transport={makeSessionTransport(makeSessionConnection("unreplayable-new"))}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    act(() => {
      webSocketOptions.current?.onMessage?.({
        type: "conversation_reload_required",
        timestamp: "2026-05-27T05:00:03Z",
        task_id: 102,
        data: {},
      })
    })

    expect(screen.getByTestId("session-conversation-state").textContent).toBe("reload_required")
    expect(screen.getByTestId("task-id").textContent).toBe("102")
    expect(screen.getByTestId("messages").textContent).toContain("Preserved transcript")
    await expect(getSessionControls().sendMessage("Replacement")).rejects.toThrow(/reload required/i)
    await expect(getSessionControls().startNewConversation()).rejects.toThrow(/reload required/i)
    expect(sendChatMessageMock).not.toHaveBeenCalled()
    expect(sendRawMessageMock).not.toHaveBeenCalled()
  })

  it("requires reload when a stale sibling reset is fenced before acknowledgement", async () => {
    render(
      <AppProvider token="token" transport={makeSessionTransport(makeSessionConnection("stale-reset"))}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    act(() => webSocketOptions.current?.onMessage?.(taskInfoMessage(102)))

    let reset!: Promise<void>
    act(() => {
      reset = getSessionControls().startNewConversation()
    })
    act(() => {
      webSocketOptions.current?.onMessage?.({
        type: "conversation_reload_required",
        timestamp: "2026-05-27T05:00:03Z",
        task_id: 102,
        data: {},
      })
    })

    await expect(reset).rejects.toThrow(/reload required/i)
    expect(screen.getByTestId("session-conversation-state").textContent).toBe("reload_required")
    expect(screen.getByTestId("task-id").textContent).toBe("102")
    await expect(getSessionControls().sendMessage("Must not send")).rejects.toThrow(/reload required/i)
    await expect(getSessionControls().startNewConversation()).rejects.toThrow(/reload required/i)
  })

  it("ignores a stale, invalid, or mismatched server reload requirement", () => {
    const { rerender } = render(
      <AppProvider token="token" transport={makeSessionTransport(makeSessionConnection("unreplayable-old"))}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )
    act(() => webSocketOptions.current?.onMessage?.(taskInfoMessage(102)))
    const staleOnMessage = webSocketOptions.current?.onMessage

    rerender(
      <AppProvider token="token" transport={makeSessionTransport(makeSessionConnection("unreplayable-current"))}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )

    act(() => staleOnMessage?.({
      type: "conversation_reload_required",
      timestamp: "2026-05-27T05:00:03Z",
      task_id: 102,
      data: {},
    }))

    for (const taskId of [undefined, true, 0, 103]) {
      act(() => {
        webSocketOptions.current?.onMessage?.({
          type: "conversation_reload_required",
          timestamp: "2026-05-27T05:00:03Z",
          ...(taskId === undefined ? {} : { task_id: taskId as unknown as number }),
          data: {},
        })
      })
    }

    expect(screen.getByTestId("session-conversation-state").textContent).toBe("bound")
    expect(screen.getByTestId("task-id").textContent).toBe("102")
  })

  it("ignores a server reload requirement before the Session adopts a task", () => {
    render(
      <AppProvider token="token" transport={makeSessionTransport()}>
        <SessionControlsProbe />
        <StateProbe />
      </AppProvider>
    )

    act(() => webSocketOptions.current?.onMessage?.({
      type: "conversation_reload_required",
      timestamp: "2026-05-27T05:00:03Z",
      task_id: 102,
      data: {},
    }))

    expect(screen.getByTestId("session-conversation-state").textContent).toBe("unbound")
    expect(screen.getByTestId("task-id").textContent).toBe("")
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

describe("terminal error frames", () => {
  // Same reset as the routing suite above: the websocket harness ref and the
  // duplicate-message cache both outlive a single render, so without this the
  // second test in this block would keep talking to the first one's provider.
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

  // The conversation panel renders only user / isResult / system-notice
  // messages. Without the flag the bubble is filtered out and the UI falls
  // back to a generic "unknown error" placeholder until the page reloads.
  it("flags the terminal task_error bubble as the turn's result", async () => {
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
        type: "task_error",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        task: { id: 1, status: "failed" },
        message: "Task execution failed.",
        error: "Task execution failed.",
      } as TestWebSocketMessage)
    })

    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "Task execution failed."
      )
    })

    const messages = JSON.parse(
      screen.getByTestId("messages").textContent || "[]"
    )
    const bubble = messages.find((m: { content: string }) =>
      m.content.includes("Task execution failed.")
    )
    expect(bubble?.isResult).toBe(true)
  })

  // Rejections arrive on the root "error" type while the task keeps running:
  // websocket.py refuses a chat message (:5521-5554) and a pause command
  // (:8491, :8502) that way. Flagging one as this turn's result makes the
  // conversation panel treat the turn as answered -- it renders only user /
  // isResult / system-notice messages, so a flagged rejection ends the live
  // progress indicator of a turn that is still running.
  it("does not treat a rejection on a running task as the turn's result", async () => {
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
        type: "error",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        task: { id: 1, status: "running" },
        message: "Task is currently busy; please wait for the previous turn to finish.",
        error_code: "task_busy",
      } as TestWebSocketMessage)
    })

    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "clientErrors.taskBusy"
      )
    })

    const messages = JSON.parse(screen.getByTestId("messages").textContent || "[]")
    const bubble = messages.find((m: { content: string }) =>
      m.content.includes("clientErrors.taskBusy")
    )
    expect(bubble).toBeDefined()
    expect(bubble?.isResult).not.toBe(true)
    // The turn is untouched: still running, still processing.
    expect(screen.getByTestId("task-status").textContent).toBe("running")
    expect(screen.getByTestId("processing").textContent).toBe("true")
  })

  // The waiting half. A refused resume arrives on the root "error" type
  // carrying the task's real current status (built from TaskControlSnapshot
  // in websocket.py's _handle_resume_task_unserialized, near where
  // resume_control_state is assembled -- that function is long, so look for
  // the assignment rather than a fixed offset). The question the user still
  // has to answer lives on the panel's virtual bubble, which a flagged
  // rejection would remove.
  // The fixture below combines fields from two producers: `task` comes from
  // the resume-refusal branch, websocket.py's
  // _handle_resume_task_unserialized; `error_code` comes from the
  // pause-refusal branch, websocket.py's handle_pause_task, near its
  // ClientErrorCode.MESSAGE_PROCESSING_FAILED fallback. Neither path emits
  // both fields together today; each field is genuinely emitted by its own
  // path.
  // Carrying `task` drives the reducer's preservation branch (it is what
  // makes `UPDATE_TASK_STATUS` dispatch at all) -- without it the assertions
  // below would pass vacuously instead of exercising that branch.
  it("does not treat a rejection on a waiting task as the turn's result", async () => {
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    // Put the task where a resume can be refused: waiting on a question that
    // carries an interaction the user has to fill in.
    act(() => {
      onMessage?.({
        type: "task_waiting_for_user",
        timestamp: "2026-05-27T05:00:01Z",
        task_id: 1,
        task: { id: 1, status: "waiting_for_user" },
        message: "Which region should I use?",
        request_id: "req-1",
        interactions: [
          { type: "text", request_id: "req-1", prompt: "Which region should I use?" },
        ],
      } as TestWebSocketMessage)
    })
    await waitFor(() => {
      expect(screen.getByTestId("task-status").textContent).toBe("waiting_for_user")
    })
    expect(screen.getByTestId("waiting-interactions").textContent).not.toBe("[]")

    act(() => {
      onMessage?.({
        type: "error",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        task: { id: 1, status: "waiting_for_user" },
        message: "Task pause is still being applied; please retry shortly.",
        error_code: "task_pause_in_progress",
      } as TestWebSocketMessage)
    })

    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "clientErrors.taskPauseInProgress"
      )
    })

    const messages = JSON.parse(screen.getByTestId("messages").textContent || "[]")
    const bubble = messages.find((m: { content: string }) =>
      m.content.includes("clientErrors.taskPauseInProgress")
    )
    expect(bubble).toBeDefined()
    expect(bubble?.isResult).not.toBe(true)
    // The question the user still owes an answer to is untouched.
    expect(screen.getByTestId("task-status").textContent).toBe("waiting_for_user")
    expect(screen.getByTestId("waiting-interactions").textContent).not.toBe("[]")
  })

  // The frame's code alone decides the wording -- nothing the raise site
  // attached beyond the code reaches the client, so a missing-value code
  // gets its own table entry regardless of what it would have named.
  it("resolves wording from the code alone", async () => {
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
        type: "task_error",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        task: { id: 1, status: "failed" },
        message: "Connector secrets are unavailable.",
        error: "Connector secrets are unavailable.",
        code: "runtime_secret_unavailable",
      } as TestWebSocketMessage)
    })

    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "clientErrors.runtimeSecretUnavailable"
      )
    })

    const messages = JSON.parse(screen.getByTestId("messages").textContent || "[]")
    const bubble = messages.find((m: { content: string }) =>
      m.content.includes("clientErrors.runtimeSecretUnavailable")
    )
    // Same toBe as the test below, on a different code: an interpolation
    // regression has to be caught on both.
    expect(bubble?.content).toBe("clientErrors.runtimeSecretUnavailable")
  })

  // What the server now sends for a missing declared context key: the code
  // survives, and nothing else about the failure -- the key name included --
  // ever reaches this frame at all. The bubble therefore says a value is
  // missing without saying which -- the key name is owner configuration and
  // this frame reaches anonymous widget and share-link visitors. Asserted
  // with toBe, under a variable-aware i18n mock: had the wording
  // interpolated anything, the content would read "<key>:{...}" and this
  // assertion would fail.
  it("names no declared key when a runtime value is missing", async () => {
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
        type: "task_error",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        task: { id: 1, status: "failed" },
        message: "Required connector runtime context is missing.",
        error: "Required connector runtime context is missing.",
        code: "missing_runtime_context",
      } as TestWebSocketMessage)
    })

    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "clientErrors.missingRuntimeContext"
      )
    })

    const messages = JSON.parse(screen.getByTestId("messages").textContent || "[]")
    const bubble = messages.find((m: { content: string }) =>
      m.content.includes("clientErrors.missingRuntimeContext")
    )
    // No prefix: this wording replaces the server sentence rather than
    // decorating it. Exactly the key, with nothing appended: no variable was
    // interpolated, so no key name can be in the rendered text.
    expect(bubble?.content).toBe("clientErrors.missingRuntimeContext")
    expect(bubble?.isResult).toBe(true)
  })

  // The frame's code decides the wording -- this code reports a server-side
  // component being down, and it has its own table entry rather than the
  // generic prefix.
  it("uses the code's own wording when a server-side component is down", async () => {
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
        type: "task_error",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        task: { id: 1, status: "failed" },
        message: "Connector runtime is unavailable.",
        error: "Connector runtime is unavailable.",
        code: "connector_runtime_unavailable",
      } as TestWebSocketMessage)
    })

    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "clientErrors.connectorRuntimeUnavailable"
      )
    })

    // The server sentence is no longer relayed: this code has its own table
    // entry now, on every transport, so the logged-in audience no longer sees
    // a different wording than the one an untrusted transport gets.
    expect(screen.getByTestId("messages").textContent).not.toContain(
      "Connector runtime is unavailable."
    )
  })

  // The dedup identity is the frame's own (run_id, state_version), not its
  // code. These two turns fail under the same code -- the same runtime
  // secret, both times unavailable -- and each settlement bumps
  // state_version at least once (the retry takes the lease FAILED ->
  // RUNNING, then settles RUNNING -> FAILED), so the second turn's version
  // is strictly greater. Two distinct versions mean two distinct
  // identities, and the bubble is the turn's result.
  it("keeps both bubbles when one code fails twice at different state versions", async () => {
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    const frameAtVersion = (timestamp: string, stateVersion: number) => ({
      type: "task_error",
      timestamp,
      task_id: 1,
      task: { id: 1, status: "failed" },
      // Identical on both frames, and that is the production shape:
      // _message_for_code returns one string per code.
      message: "Required runtime secret is unavailable.",
      error: "Required runtime secret is unavailable.",
      code: "runtime_secret_unavailable",
      run_id: "run-1",
      state_version: stateVersion,
    }) as TestWebSocketMessage

    act(() => {
      onMessage?.(frameAtVersion("2026-05-27T05:00:02Z", 12))
    })
    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "clientErrors.runtimeSecretUnavailable"
      )
    })

    act(() => {
      onMessage?.(frameAtVersion("2026-05-27T05:00:03Z", 14))
    })

    await waitFor(() => {
      const messages = JSON.parse(
        screen.getByTestId("messages").textContent || "[]"
      )
      const bubbles = messages.filter(
        (m: { content: string }) =>
          m.content === "clientErrors.runtimeSecretUnavailable"
      )
      expect(bubbles).toHaveLength(2)
    })
  })

  // The other half of the same contract: a genuine redelivery of one
  // settlement (same run_id, same state_version) is still collapsed, so the
  // identity did not simply disable deduplication.
  it("still collapses a redelivery of one settlement", async () => {
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    const frame = {
      type: "task_error",
      timestamp: "2026-05-27T05:00:02Z",
      task_id: 1,
      task: { id: 1, status: "failed" },
      message: "Required runtime secret is unavailable.",
      error: "Required runtime secret is unavailable.",
      code: "runtime_secret_unavailable",
      run_id: "run-1",
      state_version: 12,
    } as TestWebSocketMessage

    act(() => {
      onMessage?.(frame)
    })
    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "clientErrors.runtimeSecretUnavailable"
      )
    })

    act(() => {
      onMessage?.({ ...frame, timestamp: "2026-05-27T05:00:03Z" })
    })

    await waitFor(() => {
      const messages = JSON.parse(
        screen.getByTestId("messages").textContent || "[]"
      )
      const bubbles = messages.filter(
        (m: { content: string }) =>
          m.content === "clientErrors.runtimeSecretUnavailable"
      )
      expect(bubbles).toHaveLength(1)
    })
  })

  // Two failed turns under the same code, distinguished only by their
  // state_version. Keying on the failure's class -- the code or the
  // rendered sentence -- cannot tell these apart; keying on the frame's own
  // state tuple can.
  it("keeps both bubbles when one failure repeats on the next turn", async () => {
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    const frameAtVersion = (stateVersion: number, timestamp: string) => ({
      type: "task_error",
      timestamp,
      task_id: 1,
      task: { id: 1, status: "failed" },
      message: "Required runtime secret is unavailable.",
      error: "Required runtime secret is unavailable.",
      code: "runtime_secret_unavailable",
      run_id: "run-1",
      state_version: stateVersion,
    }) as TestWebSocketMessage

    act(() => {
      onMessage?.(frameAtVersion(12, "2026-05-27T05:00:02Z"))
    })
    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "clientErrors.runtimeSecretUnavailable"
      )
    })

    act(() => {
      onMessage?.(frameAtVersion(14, "2026-05-27T05:00:03Z"))
    })

    await waitFor(() => {
      const messages = JSON.parse(
        screen.getByTestId("messages").textContent || "[]"
      )
      const bubbles = messages.filter(
        (m: { content: string }) =>
          m.content === "clientErrors.runtimeSecretUnavailable"
      )
      expect(bubbles).toHaveLength(2)
    })
  })

  // Two failed turns that carry no code at all -- the rendered sentence is
  // identical on both -- distinguished only by their state_version. Before
  // this change the dedup key was the rendered sentence alone, so the
  // second of these vanished and that bubble
  // is the turn's result.
  it("keeps both bubbles for two generic failures on consecutive turns", async () => {
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    const frameAtVersion = (stateVersion: number, timestamp: string) => ({
      type: "task_error",
      timestamp,
      task_id: 1,
      task: { id: 1, status: "failed" },
      message: "Task execution failed.",
      error: "Task execution failed.",
      run_id: "run-1",
      state_version: stateVersion,
    }) as TestWebSocketMessage

    act(() => {
      onMessage?.(frameAtVersion(12, "2026-05-27T05:00:02Z"))
    })
    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "Task execution failed."
      )
    })

    act(() => {
      onMessage?.(frameAtVersion(14, "2026-05-27T05:00:03Z"))
    })

    await waitFor(() => {
      const messages = JSON.parse(
        screen.getByTestId("messages").textContent || "[]"
      )
      const bubbles = messages.filter(
        (m: { content: string }) =>
          typeof m.content === "string" && m.content.includes("Task execution failed.")
      )
      expect(bubbles).toHaveLength(2)
    })
  })

  // The widest form of the same case: a generic failure on an untrusted
  // transport reads the same fixed "Unknown error" constant regardless of
  // what precedes it, so a version-blind identity would collapse it into
  // whatever coded failure happened to precede it within the window. The
  // frame's own state tuple tells these two turns apart even though their
  // rendered sentence -- and, on this transport, their entire dedup text --
  // is identical.
  it("keeps a generic failure that follows a coded one on an untrusted transport", async () => {
    render(
      <AppProvider token="public-token" transport={{ legacyErrorProse: "untrusted" }}>
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    act(() => {
      onMessage?.({
        type: "task_error",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        task: { id: 1, status: "failed" },
        message: "Required connector runtime context is missing.",
        error: "Required connector runtime context is missing.",
        code: "missing_runtime_context",
        run_id: "run-1",
        state_version: 12,
      } as TestWebSocketMessage)
    })
    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "clientErrors.missingRuntimeContext"
      )
    })

    act(() => {
      onMessage?.({
        type: "task_error",
        timestamp: "2026-05-27T05:00:03Z",
        task_id: 1,
        task: { id: 1, status: "failed" },
        message: "Task execution failed.",
        error: "Task execution failed.",
        run_id: "run-1",
        state_version: 14,
      } as TestWebSocketMessage)
    })

    await waitFor(() => {
      const messages = JSON.parse(
        screen.getByTestId("messages").textContent || "[]"
      )
      expect(messages).toHaveLength(2)
      const codedBubbles = messages.filter(
        (m: { content: string }) =>
          m.content === "clientErrors.missingRuntimeContext"
      )
      const genericBubbles = messages.filter(
        (m: { content: string }) =>
          typeof m.content === "string" && m.content.includes("Unknown error")
      )
      expect(codedBubbles).toHaveLength(1)
      expect(genericBubbles).toHaveLength(1)
    })
  })

  // The witness at the integration level: two non-terminal rejections each
  // carry a version ("error" is in VERSIONED_TASK_EVENT_TYPES too), but the
  // terminal-only identity must not leak into this channel -- if it did, two
  // different versions would make these look like two distinct rejections
  // instead of one repeated one.
  it("still collapses two identical non-terminal rejections", async () => {
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    const rejectionAtVersion = (stateVersion: number, timestamp: string) => ({
      type: "error",
      timestamp,
      task_id: 1,
      task: { id: 1, status: "running" },
      message: "Task is currently busy; please wait for the previous turn to finish.",
      error_code: "task_busy",
      run_id: "run-1",
      state_version: stateVersion,
    }) as TestWebSocketMessage

    act(() => {
      onMessage?.(rejectionAtVersion(12, "2026-05-27T05:00:02Z"))
    })
    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "clientErrors.taskBusy"
      )
    })

    act(() => {
      onMessage?.(rejectionAtVersion(14, "2026-05-27T05:00:03Z"))
    })

    await waitFor(() => {
      const messages = JSON.parse(
        screen.getByTestId("messages").textContent || "[]"
      )
      const bubbles = messages.filter(
        (m: { content: string }) =>
          typeof m.content === "string" && m.content.includes("clientErrors.taskBusy")
      )
      expect(bubbles).toHaveLength(1)
    })
  })

  // connector_runtime_unavailable has its own table entry, so it survives a
  // transport that marks legacy prose untrusted the same way the three
  // previously-curated codes always did. Before this, only those three had
  // client-side wording; every other code, this one included, fell through
  // to "Unknown error" for an anonymous widget or share-link visitor.
  it("localizes a connector-runtime code for an untrusted transport", async () => {
    render(
      <AppProvider token="public-token" transport={{ legacyErrorProse: "untrusted" }}>
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    act(() => {
      onMessage?.({
        type: "task_error",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        task: { id: 1, status: "failed" },
        message: "Connector runtime is unavailable.",
        error: "Connector runtime is unavailable.",
        code: "connector_runtime_unavailable",
      } as TestWebSocketMessage)
    })

    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "clientErrors.connectorRuntimeUnavailable"
      )
    })

    expect(screen.getByTestId("messages").textContent).not.toContain(
      "Unknown error"
    )
  })

  // The boundary the client wording table draws: a code the table does not
  // list keeps the generic prefixed wording, even though it is a member of
  // the same connector-runtime family. connector_not_found is real -- it is
  // one of the ten V1ErrorCode connector-runtime members -- but its only
  // raise site is in the payload-validation stage before a task row exists,
  // so it never reaches a terminal frame in production; this pins what the
  // table does when a code outside its five members shows up anyway, not a
  // claim that this specific code will arrive on this path.
  it("falls back to generic wording for a code the table does not list", async () => {
    render(
      <AppProvider token="public-token" transport={{ legacyErrorProse: "untrusted" }}>
        <SeedRunningTask />
        <StateProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    act(() => {
      onMessage?.({
        type: "task_error",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        task: { id: 1, status: "failed" },
        message: "Connector could not be found.",
        error: "Connector could not be found.",
        code: "connector_not_found",
      } as TestWebSocketMessage)
    })

    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "agent.logs.event.messages.errorPrefix Unknown error"
      )
    })
  })

  // getTaskErrorProjection reads `data?.code` before `root.code` -- this
  // pins the nested half of that fallback, which no existing fixture
  // exercises (they all carry code on the root).
  it("reads code nested under data", async () => {
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
        type: "task_error",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        data: {
          code: "missing_runtime_context",
        },
      } as TestWebSocketMessage)
    })

    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "clientErrors.missingRuntimeContext"
      )
    })
  })

  // A cancellation carries no code (external_task_cancel.py:404 passes only
  // a message), so isTerminal alone -- not a code -- has to make this frame
  // the turn's result and route it through ADD_MESSAGE's isResult branch,
  // the one place trace events accumulated on state.traceEvents move onto
  // the settling message and state.traceEvents is cleared.
  it("drains accumulated trace events onto the cancellation bubble", async () => {
    render(
      <AppProvider token="token">
        <SeedRunningTask />
        <StateProbe />
        <SessionControlsProbe />
      </AppProvider>
    )

    const onMessage = webSocketOptions.current?.onMessage
    expect(onMessage).toBeDefined()

    act(() => {
      getSessionControls().dispatch({
        type: "ADD_TRACE_EVENT",
        payload: {
          event_id: "trace-1",
          event_type: "agent_progress",
          timestamp: "2026-05-27T05:00:01Z",
          data: { message: "Reading the connector config" },
        },
      })
      getSessionControls().dispatch({
        type: "ADD_TRACE_EVENT",
        payload: {
          event_id: "trace-2",
          event_type: "agent_progress",
          timestamp: "2026-05-27T05:00:01.500Z",
          data: { message: "Calling the tool" },
        },
      })
    })
    expect(getSessionControls().state.traceEvents).toHaveLength(2)

    act(() => {
      onMessage?.({
        type: "task_error",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        task: { id: 1, status: "failed" },
        message: "This response was interrupted.",
        error: "This response was interrupted.",
      } as TestWebSocketMessage)
    })

    await waitFor(() => {
      expect(screen.getByTestId("messages").textContent).toContain(
        "This response was interrupted."
      )
    })

    expect(getSessionControls().state.traceEvents).toEqual([])
    const bubble = getSessionControls().state.messages.find(
      (message) =>
        typeof message.content === "string" &&
        message.content.includes("This response was interrupted.")
    )
    expect(bubble?.traceEvents?.map((event) => event.event_id)).toEqual([
      "trace-1",
      "trace-2",
    ])
  })
})

describe("error frame display projection", () => {
  const translate = ((key: string) => key) as unknown as Translate

  it.each([
    {
      name: "a terminal frame with a missing-value code on a trusted transport",
      frame: {
        type: "task_error",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        task: { id: 1, status: "failed" },
        message: "Required connector runtime context is missing.",
        error: "Required connector runtime context is missing.",
        code: "missing_runtime_context",
      } as unknown as TaskControlMessage,
      trustLegacyErrorProse: true,
      expected: {
        isTerminal: true,
        taskStatus: "failed",
        stopsProcessing: true,
        dedupText: "Required connector runtime context is missing.",
        // No run_id/state_version on this fixture, same as production for a
        // frame the version gate would drop once any versioned event has
        // been seen; the second case below is the witness for that fallback.
        occurrenceIdentity: undefined,
        bubbleContent: "clientErrors.missingRuntimeContext",
        isResult: true,
      },
    },
    {
      name: "a terminal frame with no code on a trusted transport",
      frame: {
        type: "task_error",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        task: { id: 1, status: "failed" },
        message: "Task execution failed.",
        error: "Task execution failed.",
      } as unknown as TaskControlMessage,
      trustLegacyErrorProse: true,
      expected: {
        isTerminal: true,
        taskStatus: "failed",
        stopsProcessing: true,
        dedupText: "Task execution failed.",
        // This is the witness for withholding the identity when the frame
        // has no state version.
        occurrenceIdentity: undefined,
        bubbleContent: "agent.logs.event.messages.errorPrefix Task execution failed.",
        isResult: true,
      },
    },
    {
      // A resume that settles the task before the caller can hand it a code:
      // websocket.py's resume-settlement broadcast (:3038) carries error_code
      // on the root, the same field the non-terminal rejection channel
      // uses -- not the code field create_terminal_task_error_event writes.
      // getWebSocketErrorCode reads it regardless of which channel it came
      // from, so dedupText picks up the coded wording; the bubble still gets
      // the generic prefix, because that only drops for a frame carrying a
      // `code` field, which this one does not.
      name: "a resume-settlement frame with a root error_code on a trusted transport",
      frame: {
        type: "task_error",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        task: { id: 1, status: "failed" },
        message: "Task execution failed.",
        error: "Task execution failed.",
        error_code: "task_execution_failed",
      } as unknown as TaskControlMessage,
      trustLegacyErrorProse: true,
      expected: {
        isTerminal: true,
        taskStatus: "failed",
        stopsProcessing: true,
        dedupText: "clientErrors.taskExecutionFailed",
        occurrenceIdentity: undefined,
        bubbleContent: "agent.logs.event.messages.errorPrefix clientErrors.taskExecutionFailed",
        isResult: true,
      },
    },
    {
      name: "a terminal frame with a missing-value code on an untrusted transport",
      frame: {
        type: "task_error",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        task: { id: 1, status: "failed" },
        message: "Required connector runtime context is missing.",
        error: "Required connector runtime context is missing.",
        code: "missing_runtime_context",
      } as unknown as TaskControlMessage,
      trustLegacyErrorProse: false,
      expected: {
        isTerminal: true,
        taskStatus: "failed",
        stopsProcessing: true,
        dedupText: "Unknown error",
        occurrenceIdentity: undefined,
        bubbleContent: "clientErrors.missingRuntimeContext",
        isResult: true,
      },
    },
    {
      name: "a terminal frame with a state version on a trusted transport",
      frame: {
        type: "task_error",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        task: { id: 1, status: "failed" },
        message: "Required runtime secret is unavailable.",
        error: "Required runtime secret is unavailable.",
        code: "runtime_secret_unavailable",
        run_id: "run-1",
        state_version: 12,
      } as unknown as TaskControlMessage,
      trustLegacyErrorProse: true,
      expected: {
        isTerminal: true,
        taskStatus: "failed",
        stopsProcessing: true,
        dedupText: "Required runtime secret is unavailable.",
        occurrenceIdentity: "run-1:12",
        bubbleContent: "clientErrors.runtimeSecretUnavailable",
        isResult: true,
      },
    },
    {
      name: "a terminal frame with a state version and no code on a trusted transport",
      frame: {
        type: "task_error",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        task: { id: 1, status: "failed" },
        message: "Task execution failed.",
        error: "Task execution failed.",
        run_id: "run-1",
        state_version: 12,
      } as unknown as TaskControlMessage,
      trustLegacyErrorProse: true,
      expected: {
        isTerminal: true,
        taskStatus: "failed",
        stopsProcessing: true,
        dedupText: "Task execution failed.",
        occurrenceIdentity: "run-1:12",
        bubbleContent: "agent.logs.event.messages.errorPrefix Task execution failed.",
        isResult: true,
      },
    },
    {
      name: "a non-terminal frame with a code and a state version on a trusted transport",
      frame: {
        type: "error",
        timestamp: "2026-05-27T05:00:02Z",
        task_id: 1,
        task: { id: 1, status: "running" },
        message: "Task is currently busy; please wait for the previous turn to finish.",
        error: "Task is currently busy; please wait for the previous turn to finish.",
        code: "connector_runtime_unavailable",
        state_version: 12,
      } as unknown as TaskControlMessage,
      trustLegacyErrorProse: true,
      expected: {
        isTerminal: false,
        taskStatus: "running",
        stopsProcessing: false,
        dedupText: "Task is currently busy; please wait for the previous turn to finish.",
        // This is the witness for keeping the terminal-only identity out of
        // the rejection channel.
        occurrenceIdentity: undefined,
        bubbleContent: "agent.logs.event.messages.errorPrefix Task is currently busy; please wait for the previous turn to finish.",
        isResult: false,
      },
    },
  ])("derives $name", ({ frame, trustLegacyErrorProse, expected }) => {
    // The envelope is parsed here rather than hand-built, matching the one
    // call site in production (app-context-chat.tsx, before the switch): a
    // hand-built envelope would be non-production-shaped input, and this is
    // also what makes the no-version cell above (see its comment) actually
    // exercise stateVersion being undefined rather than a value we chose.
    const controlEnvelope = extractTaskControlEnvelope(frame)
    expect(
      projectErrorFrameForDisplay(frame, { trustLegacyErrorProse, translate, controlEnvelope }),
    ).toEqual(expected)
  })
})

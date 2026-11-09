import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { AppProviderTransportConfig } from "@/contexts/app-context-chat"
import { MessageDeliveryError } from "@/hooks/use-websocket"
import type { WidgetSession } from "./use-widget-session"

const bridge = vi.hoisted(() => ({
  value: {
    status: "waiting" as "waiting" | "active" | "refreshing" | "degraded" | "terminal",
    session: null as WidgetSession | null,
    agent: null as WidgetSession["agent"] | null,
    terminalCode: null as string | null,
    isAbsoluteExpiryWarningVisible: false,
    requestReconnect: vi.fn(),
    handleConnectionClose: vi.fn(() => "handled" as const),
    handleConnectionFailure: vi.fn(),
    ensureFreshSession: vi.fn(),
    refreshSession: vi.fn(),
    waitForOpenConnection: vi.fn(),
  },
}))

const toastMock = vi.hoisted(() => ({ error: vi.fn() }))

vi.mock("@/components/ui/sonner", () => ({ toast: toastMock }))

const i18n = vi.hoisted(() => ({ locale: "en" }))

const app = vi.hoisted(() => ({
  provider: null as null | {
    token?: string
    transport?: AppProviderTransportConfig
  },
  startProps: null as null | Record<string, unknown>,
  panelProps: null as null | Record<string, unknown>,
  listProps: null as null | Record<string, unknown>,
  viewProps: null as null | Record<string, unknown>,
  sendMessage: vi.fn(),
  startNewConversation: vi.fn(() => Promise.resolve()),
  state: {
    taskId: null as number | null,
    currentTask: null as null | { status?: string },
    isProcessing: false,
  },
  isConnected: false,
  filesDisabled: true,
  voiceInputEnabled: false,
  isConversationResetPending: false,
  isMessageDeliveryPending: false,
  sessionConversationState: "unbound",
  sessionConversationAttachment: "pending" as "pending" | "attached" | "unsupported",
  isSessionInteractionLocked: false,
}))

vi.mock("./use-widget-session", () => ({
  useWidgetSession: () => bridge.value,
  buildWidgetSessionWebSocketUrl: (origin: string) => {
    const url = new URL(origin)
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:"
    url.pathname = "/v1/external/chat/sessions/ws"
    url.search = ""
    url.hash = ""
    return url.toString()
  },
}))

vi.mock("@/contexts/app-context-chat", () => ({
  AppProvider: ({
    children,
    token,
    transport,
  }: {
    children: React.ReactNode
    token?: string
    transport?: AppProviderTransportConfig
  }) => {
    app.provider = { token, transport }
    return <>{children}</>
  },
  useApp: () => ({
    state: app.state,
    filesDisabled: app.filesDisabled,
    voiceInputEnabled: app.voiceInputEnabled,
    isConnected: app.isConnected,
    sendMessage: app.sendMessage,
    startNewConversation: app.startNewConversation,
    isConversationResetPending: app.isConversationResetPending,
    isMessageDeliveryPending: app.isMessageDeliveryPending,
    sessionConversationState: app.sessionConversationState,
    sessionConversationAttachment: app.sessionConversationAttachment,
    isSessionInteractionLocked: app.isSessionInteractionLocked,
  }),
}))

vi.mock("@/components/chat/ChatStartScreen", () => ({
  ChatStartScreen: (props: Record<string, unknown>) => {
    app.startProps = props
    const onSend = props.onSend as (
      message: string,
      files: File[],
      config?: Record<string, unknown>,
    ) => Promise<void>
    return (
      <button
        type="button"
        data-files-disabled={String(props.filesDisabled)}
        onClick={() => { void onSend("hello", [], { mode: "balanced" }).catch(() => undefined) }}
      >
        start:{String(props.title)}
      </button>
    )
  },
}))

vi.mock("@/components/task/task-conversation-panel", () => ({
  TaskConversationPanel: (props: Record<string, unknown>) => {
    app.panelProps = props
    return (
      <div
        data-testid="session-conversation-panel"
        data-files-disabled={String(app.filesDisabled)}
      />
    )
  },
}))

vi.mock("./widget-history-list", () => ({
  WidgetHistoryList: (props: Record<string, unknown>) => {
    app.listProps = props
    return (
      <div data-testid="history-list">
        <button type="button" onClick={() => (props.onOpen as (id: number, title: string) => void)(7, "Earlier chat")}>
          list-open-other
        </button>
        {/* The real list answers a click on the current conversation with onBack. */}
        <button type="button" onClick={() => (props.onBack as () => void)()}>list-open-current</button>
        <button type="button" onClick={() => (props.onBack as () => void)()}>list-back</button>
      </div>
    )
  },
}))

vi.mock("./widget-history-view", () => ({
  WidgetHistoryView: (props: Record<string, unknown>) => {
    app.viewProps = props
    return (
      <div data-testid="history-view">
        <button
          type="button"
          onClick={() => (props.onContinue as (message: string, id: number) => void)("a follow-up", props.taskId as number)}
        >
          view-continue
        </button>
        <button type="button" onClick={() => (props.onBack as () => void)()}>view-back</button>
      </div>
    )
  },
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => (i18n.locale === "en" ? key : `${i18n.locale}:${key}`),
  }),
}))

import { SessionAgentChatPage } from "./session-agent-chat-page"

const SESSION_TOKEN = "st_secret_session_token"

function activeSession(overrides: Partial<WidgetSession> = {}): WidgetSession {
  return {
    token: SESSION_TOKEN,
    tokenExpiresAt: "2026-07-26T01:00:00.000Z",
    absoluteExpiresAt: "2026-07-26T02:00:00.000Z",
    generation: 3,
    agent: {
      id: 42,
      name: "Support Agent",
      description: "Answers schedules",
      logoUrl: "https://cdn.example/support.png",
      suggestedPrompts: ["Show my schedule"],
    },
    ...overrides,
  }
}

function setBridge(
  status: typeof bridge.value.status,
  session: WidgetSession | null,
  terminalCode: string | null = null,
) {
  bridge.value.status = status
  bridge.value.session = session
  if (session) {
    bridge.value.agent = session.agent
  } else if (status !== "refreshing" && status !== "degraded") {
    bridge.value.agent = null
  }
  bridge.value.terminalCode = terminalCode
}

describe("SessionAgentChatPage", () => {
  beforeEach(() => {
    setBridge("waiting", null)
    bridge.value.isAbsoluteExpiryWarningVisible = false
    bridge.value.handleConnectionClose.mockClear()
    bridge.value.handleConnectionFailure.mockClear()
    bridge.value.ensureFreshSession.mockReset()
    bridge.value.refreshSession.mockReset()
    bridge.value.waitForOpenConnection.mockReset()
    toastMock.error.mockReset()
    app.provider = null
    app.startProps = null
    app.panelProps = null
    app.listProps = null
    app.viewProps = null
    app.sendMessage.mockReset()
    app.startNewConversation.mockReset()
    app.startNewConversation.mockResolvedValue()
    app.state.taskId = null
    app.state.currentTask = null
    app.state.isProcessing = false
    app.isConnected = false
    app.isConversationResetPending = false
    app.isMessageDeliveryPending = false
    app.sessionConversationState = "unbound"
    app.sessionConversationAttachment = "pending"
    app.isSessionInteractionLocked = false
    localStorage.clear()
    sessionStorage.clear()
    vi.stubGlobal("fetch", vi.fn())
  })

  afterEach(() => {
    i18n.locale = "en"
    cleanup()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it("keeps the Session capability provider mounted while waiting with no connection", () => {
    render(<SessionAgentChatPage />)

    expect(screen.getByText("widgetChat.status.initializing")).toBeInTheDocument()
    // This "waiting" branch renders no header at all -- widget.js's mobile
    // full-screen FAB-hiding guard depends on WidgetChromeControls (and thus
    // this close button) never rendering here; a regression that added it
    // without the panel actually being able to close would go uncaught
    // without this.
    expect(screen.queryByRole("button", { name: "widgetChat.close" })).not.toBeInTheDocument()
    expect(app.provider?.token).toBeUndefined()
    expect(app.provider?.transport?.legacyErrorProse).toBe("untrusted")
    expect(app.provider?.transport?.session).toEqual({
      connection: null,
      onConnectionClose: bridge.value.handleConnectionClose,
      onConnectionFailure: bridge.value.handleConnectionFailure,
      allowTasklessChat: true,
      supportsConversationReset: true,
      history: "none",
      files: "disabled",
      agentCards: "disabled",
      voice: "disabled",
      taskControls: "disabled",
    })
    // The session page always serves the widget surface, so links must leave
    // the conversation in place by opening a new tab.
    expect(app.provider?.transport?.capabilities).toEqual({
      linksOpenInNewTab: "enabled",
      taskStop: "enabled",
    })
  })

  it("constructs the exact active Session transport and does not leak credentials", () => {
    // Same fix as widget-chrome.test.ts: this suite's localStorage is a
    // LocalStorageMock with its own prototype (vitest.setup.ts), unrelated
    // to native Storage.prototype -- spying there never intercepted this
    // component's actual localStorage.setItem calls, so this credential-leak
    // assertion was vacuous.
    const storageSet = vi.spyOn(localStorage, "setItem")
    setBridge("active", activeSession())
    app.isConnected = true

    render(<SessionAgentChatPage />)

    const expectedUrl = new URL(window.location.origin)
    expectedUrl.protocol = expectedUrl.protocol === "https:" ? "wss:" : "ws:"
    expectedUrl.pathname = "/v1/external/chat/sessions/ws"
    expect(app.provider?.transport?.session?.connection).toEqual({
      identity: "widget-session:3",
      url: expectedUrl.toString(),
      protocols: [
        "xagent-session-v1",
        `xagent-session-token.${SESSION_TOKEN}`,
        "xagent-session-liveness-v1",
        "xagent-session-attach-v1",
      ],
      expectedProtocol: "xagent-session-v1",
      chatTaskIdMode: "omit",
      taskBindingMode: "session-subprotocol",
      credentialOwner: { kind: "external" },
    })
    expect(app.provider?.transport?.session?.onConnectionFailure).toBe(
      bridge.value.handleConnectionFailure,
    )
    expect(app.provider?.token).toBeUndefined()
    expect(document.body).not.toHaveTextContent(SESSION_TOKEN)
    expect(document.documentElement.innerHTML).not.toContain(SESSION_TOKEN)
    expect(storageSet).not.toHaveBeenCalled()
    expect(fetch).not.toHaveBeenCalled()
  })

  it("shows a reload-required outcome and disables reset and composer actions", () => {
    setBridge("active", activeSession())
    app.isConnected = true
    app.state.taskId = 91
    app.sessionConversationState = "reload_required"
    app.isSessionInteractionLocked = true

    render(<SessionAgentChatPage />)

    expect(screen.getByRole("alert")).toHaveTextContent("widgetSession.reloadRequired")
    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    expect(screen.getByRole("menuitem", {
      name: "widgetSession.startNewConversation",
    })).toBeDisabled()
    expect(app.startProps).toBeNull()

    app.state.taskId = null
    render(<SessionAgentChatPage />)
    expect(app.startProps?.isSending).toBe(true)
  })

  it("shows only the reload instruction when a reset rejection is followed by reload-required", async () => {
    setBridge("active", activeSession())
    app.isConnected = true
    app.state.taskId = 92
    app.startNewConversation.mockRejectedValueOnce(new Error("timeout"))
    const { rerender } = render(<SessionAgentChatPage />)
    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    fireEvent.click(screen.getByRole("menuitem", { name: "widgetSession.startNewConversation" }))
    await screen.findByText("widgetSession.resetFailed")

    app.sessionConversationState = "reload_required"
    app.isSessionInteractionLocked = true
    rerender(<SessionAgentChatPage />)
    expect(screen.getAllByRole("alert")).toHaveLength(1)
    expect(screen.getByRole("alert")).toHaveTextContent("widgetSession.reloadRequired")
  })

  it("renders allowlisted Agent metadata and a taskless, file-disabled start screen", () => {
    setBridge("active", activeSession())
    app.isConnected = true

    render(<SessionAgentChatPage />)

    expect(screen.getByRole("heading", { name: "Support Agent" })).toBeInTheDocument()
    expect(screen.getByAltText("Support Agent")).toHaveAttribute(
      "src",
      "https://cdn.example/support.png",
    )
    expect(screen.getByText("widgetChat.status.online")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "start:Support Agent" })).toHaveAttribute(
      "data-files-disabled",
      "true",
    )
    expect(app.startProps).toEqual(expect.objectContaining({
      title: "Support Agent",
      description: "Answers schedules",
      prompts: ["Show my schedule"],
      sendPromptOnClick: true,
      filesDisabled: true,
      voiceInputEnabled: false,
    }))

    fireEvent.click(screen.getByRole("button", { name: "start:Support Agent" }))
    expect(app.sendMessage).toHaveBeenCalledWith(
      "hello",
      { mode: "balanced" },
      [],
    )
    // No conversation yet — nothing to reset, but the "..." trigger still
    // renders since expand/collapse is always offered regardless.
    expect(screen.getByRole("button", { name: "widgetChat.moreOptions" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "widgetChat.close" })).toBeInTheDocument()
  })

  it("renders the conversation with files disabled and gates reset on pending work", () => {
    setBridge("active", activeSession())
    app.state.taskId = 71
    app.isConnected = true
    app.isMessageDeliveryPending = true

    const { rerender } = render(<SessionAgentChatPage />)

    expect(screen.getByTestId("session-conversation-panel")).toHaveAttribute(
      "data-files-disabled",
      "true",
    )
    expect(app.panelProps).toEqual(expect.objectContaining({
      showTaskFiles: false,
      showTaskActions: false,
    }))
    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    const reset = screen.getByRole("menuitem", {
      name: "widgetSession.startNewConversation",
    })
    expect(reset).toBeDisabled()

    app.isMessageDeliveryPending = false
    rerender(<SessionAgentChatPage />)
    fireEvent.click(screen.getByRole("menuitem", {
      name: "widgetSession.startNewConversation",
    }))
    expect(app.startNewConversation).toHaveBeenCalledTimes(1)
  })

  it("keeps the reset visibly pending on the trigger after the menu auto-closes on click", () => {
    // The menuitem click that starts the reset also closes the menu (normal
    // menu UX), so the "Resetting..." label is only reachable here if the
    // trigger itself carries the pending state too.
    setBridge("active", activeSession())
    app.state.taskId = 71
    app.isConnected = true

    const { rerender } = render(<SessionAgentChatPage />)
    const trigger = screen.getByRole("button", { name: "widgetChat.moreOptions" })
    fireEvent.click(trigger)
    fireEvent.click(screen.getByRole("menuitem", { name: "widgetSession.startNewConversation" }))
    expect(screen.queryByRole("menu")).toBeNull()
    expect(app.startNewConversation).toHaveBeenCalledTimes(1)

    app.isConversationResetPending = true
    rerender(<SessionAgentChatPage />)

    expect(screen.queryByRole("menu")).toBeNull()
    // The spinner is status feedback, not a lockout -- sizing has no
    // dependency on the conversation reset, so the trigger itself must stay
    // reachable even while pending.
    expect(trigger).not.toBeDisabled()
    expect(trigger.querySelector("svg.animate-spin")).not.toBeNull()
  })

  it("shows connecting and the non-blocking absolute-expiry warning", () => {
    setBridge("active", activeSession())
    bridge.value.isAbsoluteExpiryWarningVisible = true

    render(<SessionAgentChatPage />)

    expect(screen.getByText("widgetChat.status.connecting")).toBeInTheDocument()
    expect(screen.getByText("widgetSession.expiryWarning")).toBeInTheDocument()
  })

  it("keeps the start-message promise contract and surfaces a delivery failure", async () => {
    setBridge("active", activeSession())
    app.sendMessage.mockRejectedValueOnce(new Error("delivery failed"))

    render(<SessionAgentChatPage />)
    fireEvent.click(screen.getByRole("button", { name: "start:Support Agent" }))

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent("widgetSession.startMessageFailed")
    })
  })

  async function failStartMessage(error: unknown): Promise<unknown> {
    app.sendMessage.mockRejectedValueOnce(error)
    const onSend = app.startProps?.onSend as (
      message: string,
      files: File[],
      config?: Record<string, unknown>,
    ) => Promise<void>
    let thrown: unknown
    await act(async () => {
      thrown = await onSend("hi", [], {}).then(() => null, (caught: unknown) => caught)
    })
    return thrown
  }

  it.each([
    [
      "a coded delivery error",
      new MessageDeliveryError("diagnostic", "not_sent", { errorCode: "connection_not_ready" }),
      "clientErrors.connectionNotReady",
    ],
    [
      "an error carrying an unrecognised code",
      Object.assign(new Error("x"), { errorCode: "not_a_code" }),
      "widgetSession.startMessageFailed",
    ],
  ])("shows the start-message failure for %s", async (_label, error, expectedText) => {
    setBridge("active", activeSession())
    render(<SessionAgentChatPage />)

    const thrown = await failStartMessage(error)

    expect(screen.getByRole("alert")).toHaveTextContent(expectedText)
    expect(thrown).toBe(error)
    expect((thrown as { notificationHandled?: unknown }).notificationHandled).toBeUndefined()
  })

  it("re-renders the start-message failure in the new locale", async () => {
    setBridge("active", activeSession())
    const { rerender } = render(<SessionAgentChatPage />)

    await failStartMessage(
      new MessageDeliveryError("diagnostic", "not_sent", { errorCode: "connection_not_ready" }),
    )
    expect(screen.getByRole("alert")).toHaveTextContent("clientErrors.connectionNotReady")
    expect(screen.getByRole("alert")).not.toHaveTextContent("zh:")

    i18n.locale = "zh"
    rerender(<SessionAgentChatPage />)

    expect(screen.getByRole("alert")).toHaveTextContent("zh:clientErrors.connectionNotReady")
  })

  it("does not show a start-message failure that arrives after the conversation is established", async () => {
    setBridge("active", activeSession())
    const { rerender } = render(<SessionAgentChatPage />)
    app.state.taskId = 71
    rerender(<SessionAgentChatPage />)

    await failStartMessage(new Error("late rejection"))

    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("does not bring an old start-message failure back when the conversation returns to the start page", async () => {
    setBridge("active", activeSession())
    const { rerender } = render(<SessionAgentChatPage />)
    await failStartMessage(new Error("early rejection"))
    expect(screen.getByRole("alert")).toBeInTheDocument()

    app.state.taskId = 71
    rerender(<SessionAgentChatPage />)
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()

    app.state.taskId = null
    rerender(<SessionAgentChatPage />)
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("does not show a start-message failure stored while established after a reset to the start page", async () => {
    setBridge("active", activeSession())
    const { rerender } = render(<SessionAgentChatPage />)
    app.state.taskId = 71
    rerender(<SessionAgentChatPage />)

    await failStartMessage(
      new MessageDeliveryError("diagnostic", "outcome_unknown"),
    )
    app.state.taskId = null
    rerender(<SessionAgentChatPage />)

    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("never inserts a stale start-message failure while returning to the start page", async () => {
    setBridge("active", activeSession())
    const { rerender } = render(<SessionAgentChatPage />)
    app.state.taskId = 71
    rerender(<SessionAgentChatPage />)
    await failStartMessage(
      new MessageDeliveryError("diagnostic", "outcome_unknown"),
    )

    // rerender flushes effects before it returns, so the final DOM cannot show
    // a line that was committed and then cleared; the observer records every
    // node inserted on the way.
    const inserted: Element[] = []
    const collect = (records: MutationRecord[]) => {
      for (const record of records) {
        record.addedNodes.forEach((node) => {
          if (node instanceof Element) inserted.push(node)
        })
      }
    }
    const observer = new MutationObserver(collect)
    observer.observe(document.body, { childList: true, subtree: true })
    app.state.taskId = null
    rerender(<SessionAgentChatPage />)
    collect(observer.takeRecords())
    observer.disconnect()

    expect(inserted.some((node) => node.textContent?.includes("start:Support Agent"))).toBe(true)
    expect(inserted.filter((node) => (
      node.matches('[role="alert"]') || node.querySelector('[role="alert"]') !== null
    ))).toEqual([])
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("clears the previous start-message failure while a resend is pending", async () => {
    setBridge("active", activeSession())
    render(<SessionAgentChatPage />)
    await failStartMessage(new Error("first failure"))
    expect(screen.getByRole("alert")).toHaveTextContent("widgetSession.startMessageFailed")

    app.sendMessage.mockReturnValueOnce(new Promise<void>(() => undefined))
    const onSend = app.startProps?.onSend as (
      message: string,
      files: File[],
      config?: Record<string, unknown>,
    ) => Promise<void>
    act(() => {
      void onSend("hi again", [], {})
    })

    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("lets the reload-required notice take priority over the start-message failure", async () => {
    setBridge("active", activeSession())
    const { rerender } = render(<SessionAgentChatPage />)
    await failStartMessage(new Error("early rejection"))

    app.isSessionInteractionLocked = true
    rerender(<SessionAgentChatPage />)

    expect(screen.getAllByRole("alert")).toHaveLength(1)
    expect(screen.getByRole("alert")).toHaveTextContent("widgetSession.reloadRequired")
  })

  it("preserves validated Agent metadata and the conversation while refreshing", () => {
    setBridge("active", activeSession())
    app.state.taskId = 71
    app.isConnected = true

    const { rerender } = render(<SessionAgentChatPage />)

    expect(screen.getByRole("heading", { name: "Support Agent" })).toBeInTheDocument()
    expect(screen.getByTestId("session-conversation-panel")).toBeInTheDocument()
    expect(app.provider?.transport?.session?.connection).not.toBeNull()

    setBridge("refreshing", null)
    app.isConnected = false
    rerender(<SessionAgentChatPage />)

    expect(screen.getByRole("heading", { name: "Support Agent" })).toBeInTheDocument()
    expect(screen.getByTestId("session-conversation-panel")).toBeInTheDocument()
    expect(screen.getByText("widgetChat.status.connecting")).toBeInTheDocument()
    expect(app.provider?.transport?.session?.connection).toBeNull()
  })

  it("renders first-message degradation as unavailable without a spinner", () => {
    setBridge("degraded", null)

    render(<SessionAgentChatPage />)

    expect(screen.getByRole("heading", {
      name: "widgetSession.unavailable.title",
    })).toBeInTheDocument()
    expect(screen.getByText("widgetSession.unavailable.description")).toBeInTheDocument()
    expect(screen.queryByText("widgetChat.status.initializing")).not.toBeInTheDocument()
    expect(screen.queryByText("widgetChat.status.connecting")).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "widgetChat.close" })).not.toBeInTheDocument()
  })

  it("removes all conversation controls after degradation with an existing Agent", () => {
    setBridge("active", activeSession())
    app.state.taskId = 71
    setBridge("degraded", null)

    render(<SessionAgentChatPage />)

    expect(screen.getByRole("heading", {
      name: "widgetSession.unavailable.title",
    })).toBeInTheDocument()
    expect(screen.getByText("widgetSession.unavailable.description")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "start:Support Agent" })).not.toBeInTheDocument()
    expect(screen.queryByRole("button", {
      name: "widgetSession.startNewConversation",
    })).not.toBeInTheDocument()
    // This branch renders no header at all -- including WidgetChromeControls
    // itself, not just its menu. A regression that only dropped the menu
    // item's own render path (leaving the close button behind) would slip
    // past the two assertions above alone.
    expect(screen.queryByRole("button", { name: "widgetChat.close" })).not.toBeInTheDocument()
    expect(screen.queryByTestId("session-conversation-panel")).not.toBeInTheDocument()
    expect(app.provider?.transport?.session?.connection).toBeNull()
  })

  it("keeps an accepted expansion across a degraded/active remount instead of resetting to Expand", () => {
    // WidgetChromeControls is torn down and rebuilt across this transition
    // (the degraded branch renders a completely different subtree with no
    // header at all), so its own isExpanded state can't survive it -- this
    // is why SessionAgentChatPage lifts and passes it down as a controlled
    // prop instead of leaving it as the component's internal useState.
    const postMessageSpy = vi.fn()
    vi.stubGlobal("parent", { postMessage: postMessageSpy })
    setBridge("active", activeSession())
    app.state.taskId = 71
    app.isConnected = true

    const { rerender } = render(<SessionAgentChatPage />)
    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    fireEvent.click(screen.getByRole("menuitem", { name: "widgetChat.expandWindow" }))
    expect(postMessageSpy).toHaveBeenCalledWith(
      { xagent: true, v: 1, type: "widget_expand" },
      "*",
    )

    setBridge("degraded", null)
    rerender(<SessionAgentChatPage />)
    expect(screen.getByRole("heading", {
      name: "widgetSession.unavailable.title",
    })).toBeInTheDocument()

    setBridge("active", activeSession())
    app.state.taskId = 71
    rerender(<SessionAgentChatPage />)

    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    expect(screen.getByRole("menuitem", { name: "widgetChat.collapseWindow" })).toBeInTheDocument()
    fireEvent.click(screen.getByRole("menuitem", { name: "widgetChat.collapseWindow" }))
    expect(postMessageSpy).toHaveBeenCalledWith(
      { xagent: true, v: 1, type: "widget_collapse" },
      "*",
    )
  })

  it.each([
    ["session_expired", "widgetSession.expired.title", "widgetSession.expired.description"],
    ["grant_expired", "widgetSession.expired.title", "widgetSession.expired.description"],
    ["grant_already_used", "widgetSession.expired.title", "widgetSession.expired.description"],
    ["ws_4408", "widgetSession.expired.title", "widgetSession.expired.description"],
    ["reconnect_invalid", "widgetSession.unavailable.title", "widgetSession.unavailable.description"],
    ["identity_mismatch", "widgetSession.unavailable.title", "widgetSession.unavailable.description"],
    ["ws_4403", "widgetSession.unavailable.title", "widgetSession.unavailable.description"],
    ["unexpected_error", "widgetSession.unavailable.title", "widgetSession.unavailable.description"],
    ["unknown_catalog_failure", "widgetSession.unavailable.title", "widgetSession.unavailable.description"],
  ])(
    "maps terminal code %s to localized UI while keeping the provider fail closed",
    (terminalCode, title, description) => {
      setBridge("terminal", null, terminalCode)

      render(<SessionAgentChatPage />)

      expect(screen.getByRole("heading", { name: title })).toBeInTheDocument()
      expect(screen.getByText(description)).toBeInTheDocument()
      expect(app.provider?.transport?.session?.connection).toBeNull()
      expect(app.provider?.transport?.session?.files).toBe("disabled")
    },
  )
  describe("file attachments", () => {
    const withUploads = (overrides: Record<string, unknown> = {}) => activeSession({
      capabilities: {
        fileUploads: {
          allowedExtensions: [".pdf", ".png"],
          maxFileBytes: 4096,
          maxFilesPerMessage: 3,
          ...overrides,
        },
      },
    })
    type StartSend = (message: string, files: File[], config?: Record<string, unknown>) => Promise<void>
    const startSend = () => app.startProps?.onSend as StartSend
    const pdf = () => new File(["x"], "a.pdf", { type: "application/pdf" })
    const retry = (file: File) =>
      (app.startProps?.onRetryAttachment as (file: File) => Promise<void>)(file)

    it("adds an attachments option to the transport only when uploads are advertised", () => {
      setBridge("active", withUploads())
      const { unmount } = render(<SessionAgentChatPage />)

      expect(app.provider?.transport?.session?.attachments).toEqual({
        accept: ".pdf,.png",
        maxFiles: 3,
        maxFileBytes: 4096,
      })
      expect(app.provider?.transport?.session?.files).toBe("disabled")
      expect(app.startProps?.attachments).toEqual({ accept: ".pdf,.png", maxFiles: 3, maxFileBytes: 4096 })
      unmount()

      setBridge("active", activeSession())
      render(<SessionAgentChatPage />)
      expect(app.provider?.transport?.session).not.toHaveProperty("attachments")
      expect(app.startProps?.attachments).toBeUndefined()
      expect(app.startProps?.files).toEqual([])
    })

    it("keeps the last advertisement while refreshing and drops it once the session degrades", () => {
      setBridge("active", withUploads())
      const { rerender } = render(<SessionAgentChatPage />)

      setBridge("refreshing", null)
      bridge.value.agent = activeSession().agent
      rerender(<SessionAgentChatPage />)
      expect(app.provider?.transport?.session?.attachments).toBeDefined()

      setBridge("degraded", null)
      rerender(<SessionAgentChatPage />)
      expect(app.provider?.transport?.session).not.toHaveProperty("attachments")
    })

    it.each([
      ["reset_requested", false, false],
      ["reload_required", false, false],
      ["replacement_sending", false, false],
      ["replacement_awaiting_task", false, false],
      ["unbound", true, false],
      ["unbound", false, true],
    ])("refuses a send made in the %s phase without calling sendMessage or uploading", async (phase, locked, resetPending) => {
      setBridge("active", withUploads())
      app.sessionConversationState = phase
      app.isSessionInteractionLocked = locked
      app.isConversationResetPending = resetPending
      bridge.value.ensureFreshSession.mockResolvedValue(withUploads())
      render(<SessionAgentChatPage />)

      const failure = await startSend()("hello", [pdf()], {}).then(() => null, (error: unknown) => error)

      expect(bridge.value.ensureFreshSession).not.toHaveBeenCalled()
      expect(fetch).not.toHaveBeenCalled()
      expect(app.sendMessage).not.toHaveBeenCalled()
      expect((failure as Error | null)?.message).toContain("not ready for attachments")
    })

    it("tells the visitor why a retried upload failed and stays quiet when it was cancelled", async () => {
      setBridge("active", withUploads())
      render(<SessionAgentChatPage />)

      bridge.value.ensureFreshSession.mockRejectedValueOnce(
        Object.assign(new Error("gone"), { errorCode: "upload_failed" }),
      )
      await act(async () => { await retry(pdf()) })
      expect(toastMock.error).toHaveBeenLastCalledWith("clientErrors.uploadFailed")

      bridge.value.ensureFreshSession.mockRejectedValueOnce(new Error("plain"))
      await act(async () => { await retry(pdf()) })
      expect(toastMock.error).toHaveBeenCalledTimes(2)
      expect(toastMock.error).toHaveBeenLastCalledWith("clientErrors.uploadFailed")

      bridge.value.ensureFreshSession.mockRejectedValueOnce(
        Object.assign(new Error("cancelled"), { name: "AbortError" }),
      )
      await act(async () => { await retry(pdf()) })
      expect(toastMock.error).toHaveBeenCalledTimes(2)
    })

    it("passes the token the upload server rejected to refreshSession", async () => {
      setBridge("active", withUploads())
      render(<SessionAgentChatPage />)
      const session = withUploads()
      bridge.value.ensureFreshSession.mockResolvedValue({ ...session, token: "st_current" })
      bridge.value.refreshSession.mockResolvedValue({ ...session, token: "st_refreshed" })
      vi.mocked(fetch)
        .mockResolvedValueOnce(new Response(
          JSON.stringify({ error: { code: "invalid_api_key", message: "no" } }),
          { status: 401, headers: { "Content-Type": "application/json" } },
        ))
        .mockResolvedValueOnce(new Response(
          JSON.stringify({ file_id: "0b1e5c9e-3f1a-4a63-9b0e-6c1d2f3a4b5c" }),
          { status: 201, headers: { "Content-Type": "application/json" } },
        ))

      await act(async () => { await retry(pdf()) })

      expect(bridge.value.refreshSession).toHaveBeenCalledTimes(1)
      expect(bridge.value.refreshSession).toHaveBeenCalledWith("st_current")
    })

    it("counts a retry as busy from the click until its token is settled", async () => {
      setBridge("active", withUploads())
      render(<SessionAgentChatPage />)
      expect(app.startProps?.attachmentsBusy).toBe(false)
      let rejectToken!: (reason: unknown) => void
      bridge.value.ensureFreshSession.mockReturnValueOnce(
        new Promise((_resolve, reject) => { rejectToken = reject }),
      )

      let retrying!: Promise<void>
      act(() => { retrying = retry(pdf()) })

      expect(app.startProps?.attachmentsBusy).toBe(true)
      expect(fetch).not.toHaveBeenCalled()

      await act(async () => {
        rejectToken(Object.assign(new Error("gone"), { errorCode: "upload_failed" }))
        await retrying
      })

      expect(toastMock.error).toHaveBeenCalledWith("clientErrors.uploadFailed")
      expect(app.startProps?.attachmentsBusy).toBe(false)
    })

    it("tells the visitor a retry was refused because the latest session withdrew uploads", async () => {
      setBridge("active", withUploads())
      render(<SessionAgentChatPage />)
      const file = pdf()
      bridge.value.ensureFreshSession.mockResolvedValueOnce(activeSession())

      await act(async () => { await retry(file) })

      expect(toastMock.error).toHaveBeenLastCalledWith("widgetSession.attachments.notAvailable")
      expect(fetch).not.toHaveBeenCalled()
      expect((app.startProps?.attachmentStatuses as ReadonlyMap<File, string>).get(file)).toBeUndefined()
    })

    it("offers the conversation panel's upload entry while uploads are advertised", () => {
      setBridge("active", withUploads())
      app.state.taskId = 5
      render(<SessionAgentChatPage />)

      expect(app.panelProps?.hideFileUpload).toBe(false)
      expect(app.panelProps?.attachments).toEqual({ accept: ".pdf,.png", maxFiles: 3, maxFileBytes: 4096 })
    })

    it("holds the panel's files, hides them while uploads are not advertised, and drops them with the conversation", () => {
      const a = pdf()
      const panelFiles = () => app.panelProps?.attachmentFiles
      setBridge("active", withUploads())
      app.state.taskId = 5
      const { rerender } = render(<SessionAgentChatPage />)
      act(() => { (app.panelProps?.onAttachmentFilesChange as (files: File[]) => void)([a]) })
      expect(panelFiles()).toEqual([a])

      setBridge("active", activeSession())
      rerender(<SessionAgentChatPage />)
      expect(panelFiles()).toEqual([])
      setBridge("active", withUploads())
      rerender(<SessionAgentChatPage />)
      expect(panelFiles()).toEqual([a])

      app.state.taskId = null
      rerender(<SessionAgentChatPage />)
      app.state.taskId = 5
      rerender(<SessionAgentChatPage />)
      expect(panelFiles()).toEqual([])
    })

    it("passes no upload callbacks to the surfaces when uploads are not advertised", () => {
      setBridge("active", activeSession())
      app.state.taskId = 5
      render(<SessionAgentChatPage />)

      expect(app.panelProps?.hideFileUpload).toBe(true)
      expect(app.panelProps?.onSend).toBeUndefined()
      expect(app.panelProps?.attachments).toBeUndefined()
      expect(app.panelProps?.onAttachmentFilesChange).toBeUndefined()
    })
  })

  describe("initial message from the host page", () => {
    function deliverInitialMessage(value: unknown) {
      window.dispatchEvent(new MessageEvent("message", {
        data: { xagent: true, v: 1, type: "initial_message", value },
        source: window.parent as unknown as MessageEventSource,
      }))
    }

    function stubParent() {
      const post = vi.fn()
      vi.stubGlobal("parent", { postMessage: post })
      return post
    }

    const initialRequests = (post: ReturnType<typeof vi.fn>) => (
      post.mock.calls.filter(([data]) => data.type === "initial_message_request")
    )

    it("sends the delivered message once the socket has attached", async () => {
      const post = stubParent()
      setBridge("active", activeSession())
      app.isConnected = true
      app.sessionConversationAttachment = "attached"
      app.sendMessage.mockResolvedValue(undefined)

      render(<SessionAgentChatPage />)
      await waitFor(() => expect(initialRequests(post)).toHaveLength(1))

      deliverInitialMessage("hello agent")

      await waitFor(() => {
        expect(app.sendMessage).toHaveBeenCalledWith("hello agent", undefined, [])
      })
      deliverInitialMessage("again")
      expect(app.sendMessage).toHaveBeenCalledTimes(1)
    })

    it("does not send after a first message queued before the socket attached", () => {
      const post = stubParent()
      setBridge("active", activeSession())
      app.isConnected = false
      app.isMessageDeliveryPending = true
      const { rerender } = render(<SessionAgentChatPage />)

      app.isConnected = true
      app.sessionConversationAttachment = "attached"
      rerender(<SessionAgentChatPage />)

      expect(initialRequests(post)).toEqual([])
    })

    it("waits for an open connection before asking", () => {
      const post = stubParent()
      setBridge("active", activeSession())
      app.isConnected = false

      render(<SessionAgentChatPage />)

      expect(initialRequests(post)).toEqual([])
    })

    it("waits for the open socket to attach before asking", () => {
      const post = stubParent()
      setBridge("active", activeSession())
      app.isConnected = true
      app.sessionConversationAttachment = "pending"

      render(<SessionAgentChatPage />)

      expect(initialRequests(post)).toEqual([])
    })

    it("asks on the open socket when its server does not announce attach completion", async () => {
      const post = stubParent()
      setBridge("active", activeSession())
      app.isConnected = true
      app.sessionConversationAttachment = "unsupported"

      render(<SessionAgentChatPage />)

      await waitFor(() => expect(initialRequests(post)).toHaveLength(1))
    })

    it("does not ask while a conversation already exists", () => {
      const post = stubParent()
      setBridge("active", activeSession())
      app.isConnected = true
      app.sessionConversationAttachment = "attached"
      app.state.taskId = 7

      render(<SessionAgentChatPage />)

      expect(initialRequests(post)).toEqual([])
    })

    it("surfaces a failed send on the start screen", async () => {
      const post = stubParent()
      setBridge("active", activeSession())
      app.isConnected = true
      app.sessionConversationAttachment = "attached"
      app.sendMessage.mockRejectedValue(new Error("delivery failed"))

      render(<SessionAgentChatPage />)
      await waitFor(() => expect(initialRequests(post)).toHaveLength(1))
      deliverInitialMessage("hello agent")

      expect(await screen.findByText("widgetSession.startMessageFailed")).toBeInTheDocument()
    })
  })

  describe("open message from the host page", () => {
    function deliverOpenMessage(value: string, id = "1") {
      act(() => {
        window.dispatchEvent(new MessageEvent("message", {
          data: { xagent: true, v: 1, type: "open_message", id, value },
          source: window.parent as unknown as MessageEventSource,
        }))
      })
    }

    const openRequests = (post: ReturnType<typeof vi.fn>) => (
      post.mock.calls.filter(([data]) => data.type === "open_message_request")
    )

    const mountBound = async () => {
      const post = vi.fn()
      vi.stubGlobal("parent", { postMessage: post })
      setBridge("active", activeSession())
      app.isConnected = true
      app.state.taskId = 61
      app.sessionConversationState = "bound"
      const view = render(<SessionAgentChatPage />)
      await waitFor(() => expect(openRequests(post)).toHaveLength(1))
      return view
    }

    it.each([
      ["attached", 1],
      ["pending", 0],
    ] as const)("wires the open message driver (attachment %s)", async (attachment, resets) => {
      app.sessionConversationAttachment = attachment
      // Held open: this mock's phase never moves on from bound, so a settled
      // reset would only start another one.
      app.startNewConversation.mockReturnValue(new Promise(() => {}))
      await mountBound()

      deliverOpenMessage("Book a shift")

      await waitFor(() => expect(app.startNewConversation).toHaveBeenCalledTimes(resets))
      expect(app.sendMessage).not.toHaveBeenCalled()
    })

    it.each([
      ["the conversation stays bound", false, ["widgetSession.resetFailed"]],
      ["the session then needs a reload", true, ["widgetSession.reloadRequired"]],
    ] as const)(
      "shows the reset failure line when an open message cannot start a new conversation: %s",
      async (_name, reloadRequired, alerts) => {
        vi.spyOn(console, "warn").mockImplementation(() => {})
        app.sessionConversationAttachment = "attached"
        app.startNewConversation.mockRejectedValueOnce(new Error("reset refused"))
        const { rerender } = await mountBound()

        deliverOpenMessage("Book a shift")
        await waitFor(() => expect(screen.queryAllByRole("alert")).not.toHaveLength(0))
        if (reloadRequired) {
          app.sessionConversationState = "reload_required"
          app.isSessionInteractionLocked = true
          rerender(<SessionAgentChatPage />)
        }

        expect(screen.getAllByRole("alert").map((alert) => alert.textContent)).toEqual(alerts)
      },
    )
  })

  describe("conversation history", () => {
    type HistoryAuth = {
      getToken: () => Promise<string>
      refreshToken?: (rejected: string) => Promise<string>
    }
    const historySession = () => activeSession({ capabilities: { conversationHistory: true } })

    const mountWithHistory = (overrides: Record<string, unknown> = {}) => {
      setBridge("active", historySession())
      app.isConnected = true
      app.sessionConversationAttachment = "attached"
      Object.assign(app, overrides)
      return render(<SessionAgentChatPage />)
    }

    const openMenu = () => fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    const historyItem = () => screen.queryByRole("menuitem", { name: "widgetSession.history.menuLabel" })
    const goToList = () => {
      openMenu()
      fireEvent.click(screen.getByRole("menuitem", { name: "widgetSession.history.menuLabel" }))
    }
    const goToView = () => {
      goToList()
      fireEvent.click(screen.getByRole("button", { name: "list-open-other" }))
    }
    const hand = () => fireEvent.click(screen.getByRole("button", { name: "view-continue" }))
    const authOf = (props: Record<string, unknown> | null) => (props?.getAuth as () => HistoryAuth)()
    const deliverOpenMessage = (value = "Book a shift") => act(() => {
      window.dispatchEvent(new MessageEvent("message", {
        data: { xagent: true, v: 1, type: "open_message", id: "1", value },
        source: window.parent as unknown as MessageEventSource,
      }))
    })
    const refreshingWithAgent = () => {
      setBridge("refreshing", null)
      bridge.value.agent = historySession().agent
    }

    beforeEach(() => {
      vi.stubGlobal("parent", { postMessage: vi.fn() })
    })

    describe("the menu entry", () => {
      const uploads = { allowedExtensions: [".pdf"], maxFileBytes: 10, maxFilesPerMessage: 1 }

      it.each([
        ["the server offers history", { conversationHistory: true as const }, true],
        ["the server offers history and uploads", { conversationHistory: true as const, fileUploads: uploads }, true],
        ["the server offers only uploads", { fileUploads: uploads }, false],
        ["the server offers nothing", undefined, false],
      ])("is shown only when %s", (_label, capabilities, shown) => {
        setBridge("active", activeSession({ capabilities }))

        render(<SessionAgentChatPage />)
        openMenu()

        expect(Boolean(historyItem())).toBe(shown)
      })

      it("keeps the entry while the session refreshes", () => {
        setBridge("active", historySession())
        const { rerender } = render(<SessionAgentChatPage />)

        refreshingWithAgent()
        rerender(<SessionAgentChatPage />)
        openMenu()

        expect(historyItem()).toBeInTheDocument()
      })

      it("drops the entry when the refreshed session no longer offers history", () => {
        setBridge("active", historySession())
        const { rerender } = render(<SessionAgentChatPage />)

        setBridge("active", activeSession())
        rerender(<SessionAgentChatPage />)
        openMenu()

        expect(historyItem()).not.toBeInTheDocument()
      })

      it("offers no entry for a session that is neither active nor refreshing", () => {
        setBridge("active", historySession())
        const { rerender } = render(<SessionAgentChatPage />)

        setBridge("degraded", null, "network_unavailable")
        bridge.value.agent = historySession().agent
        rerender(<SessionAgentChatPage />)

        expect(screen.queryByRole("button", { name: "widgetChat.moreOptions" })).not.toBeInTheDocument()
      })

      it("sits between the new-conversation item and the expand toggle", () => {
        app.state.taskId = 61
        mountWithHistory({ sessionConversationState: "bound" })
        openMenu()

        expect(screen.getAllByRole("menuitem").map((item) => item.textContent)).toEqual([
          "widgetSession.startNewConversation",
          "widgetSession.history.menuLabel",
          "widgetChat.expandWindow",
        ])
      })

      it.each([
        ["a reset is under way", { isConversationResetPending: true }],
        ["a message is being delivered", { isMessageDeliveryPending: true }],
        ["the session needs a reload", {
          isSessionInteractionLocked: true,
          sessionConversationState: "reload_required",
        }],
      ])("is withdrawn while %s", (_label, overrides) => {
        mountWithHistory(overrides)
        openMenu()

        expect(historyItem()).not.toBeInTheDocument()
      })

      it("is withdrawn while a file for a start-page message uploads, and comes back once it is done", async () => {
        setBridge("active", activeSession({ capabilities: { conversationHistory: true, fileUploads: uploads } }))
        app.isConnected = true
        app.sessionConversationAttachment = "attached"
        let releaseToken!: (reason: unknown) => void
        bridge.value.ensureFreshSession.mockReturnValueOnce(
          new Promise((_resolve, reject) => { releaseToken = reject }),
        )
        render(<SessionAgentChatPage />)
        const onSend = app.startProps?.onSend as (message: string, files: File[]) => Promise<void>

        let sending!: Promise<void>
        act(() => { sending = onSend("with a file", [new File(["x"], "a.pdf", { type: "application/pdf" })]) })
        openMenu()
        expect(historyItem()).not.toBeInTheDocument()
        openMenu()

        await act(async () => {
          releaseToken(Object.assign(new Error("gone"), { errorCode: "upload_failed" }))
          await sending.catch(() => undefined)
        })
        openMenu()
        expect(historyItem()).toBeInTheDocument()
      })

      it("is withdrawn while a host page message waits to be sent", () => {
        mountWithHistory({ sessionConversationAttachment: "pending" })
        deliverOpenMessage()
        openMenu()

        expect(historyItem()).not.toBeInTheDocument()
      })

      it("is shown again once the pending work has finished", () => {
        const { rerender } = mountWithHistory({ isMessageDeliveryPending: true })

        app.isMessageDeliveryPending = false
        rerender(<SessionAgentChatPage />)
        openMenu()

        expect(historyItem()).toBeInTheDocument()
      })
    })

    describe("moving between the chat, the list and one conversation", () => {
      it("opens the list from the menu and hides the chat without unmounting it", () => {
        mountWithHistory()
        const startButton = screen.getByRole("button", { name: "start:Support Agent" })

        goToList()

        expect(screen.getByTestId("history-list")).toBeInTheDocument()
        // Hidden elements have no accessible name, so look it up by its text.
        expect(screen.getByText("start:Support Agent")).toBe(startButton)
        const layer = startButton.closest("[aria-hidden='true']") as HTMLElement
        expect(layer.style.visibility).toBe("hidden")
        expect(screen.queryByRole("button", { name: "start:Support Agent" })).not.toBeInTheDocument()
      })

      it("gives the chat back, unhidden and still the same element, when the visitor goes back", () => {
        mountWithHistory()
        const startButton = screen.getByRole("button", { name: "start:Support Agent" })

        goToList()
        fireEvent.click(screen.getByRole("button", { name: "list-back" }))

        expect(screen.queryByTestId("history-list")).not.toBeInTheDocument()
        expect(screen.getByRole("button", { name: "start:Support Agent" })).toBe(startButton)
        expect(startButton.closest("[aria-hidden='true']")).toBeNull()
        expect(startButton.closest("[style*='visibility']")).toBeNull()
      })

      it("keeps the conversation panel mounted and hidden while history is open", () => {
        app.state.taskId = 61
        mountWithHistory({ sessionConversationState: "bound" })
        const panel = screen.getByTestId("session-conversation-panel")

        goToList()

        expect(screen.getByTestId("session-conversation-panel", { })).toBe(panel)
        expect(panel.closest("[aria-hidden='true']")).not.toBeNull()
      })

      it("passes the list the origin, the current conversation and the notice for the phase", () => {
        app.state.taskId = 61
        mountWithHistory({ sessionConversationState: "bound" })

        goToList()

        expect(app.listProps).toMatchObject({
          origin: window.location.origin,
          currentTaskId: 61,
          unauthorizedNotice: "tryLater",
        })
      })

      it("goes from the list to one earlier conversation, back to the list, and back to the chat", () => {
        mountWithHistory()

        goToView()
        expect(screen.queryByTestId("history-list")).not.toBeInTheDocument()
        expect(app.viewProps).toMatchObject({
          origin: window.location.origin,
          taskId: 7,
          title: "Earlier chat",
          sessionPhase: "unbound",
          sessionAttachment: "attached",
        })

        fireEvent.click(screen.getByRole("button", { name: "view-back" }))
        expect(screen.getByTestId("history-list")).toBeInTheDocument()
        fireEvent.click(screen.getByRole("button", { name: "list-back" }))
        expect(screen.queryByTestId("history-list")).not.toBeInTheDocument()
        expect(screen.queryByTestId("history-view")).not.toBeInTheDocument()
      })

      it("returns to the chat without reading or sending anything when the current conversation is chosen", () => {
        app.state.taskId = 61
        mountWithHistory({ sessionConversationState: "bound" })

        goToList()
        fireEvent.click(screen.getByRole("button", { name: "list-open-current" }))

        expect(screen.queryByTestId("history-list")).not.toBeInTheDocument()
        expect(screen.queryByTestId("history-view")).not.toBeInTheDocument()
        expect(fetch).not.toHaveBeenCalled()
        expect(app.startNewConversation).not.toHaveBeenCalled()
        expect(app.sendMessage).not.toHaveBeenCalled()
      })

      it.each(["list", "view"] as const)(
        "starts a new conversation from the %s and shows the chat again",
        async (where) => {
          app.state.taskId = 61
          mountWithHistory({ sessionConversationState: "bound" })

          if (where === "list") goToList()
          else goToView()
          openMenu()
          fireEvent.click(screen.getByRole("menuitem", { name: "widgetSession.startNewConversation" }))

          await waitFor(() => expect(app.startNewConversation).toHaveBeenCalledTimes(1))
          expect(screen.queryByTestId("history-list")).not.toBeInTheDocument()
          expect(screen.queryByTestId("history-view")).not.toBeInTheDocument()
        },
      )

      it("opens the list again from the menu while one conversation is open", () => {
        mountWithHistory()

        goToView()
        openMenu()
        fireEvent.click(historyItem() as HTMLElement)

        expect(screen.getByTestId("history-list")).toBeInTheDocument()
        expect(screen.queryByTestId("history-view")).not.toBeInTheDocument()
      })

      it("shows the same conversation again after the session was degraded and came back", () => {
        const { rerender } = mountWithHistory()
        goToView()

        setBridge("degraded", null, "network_unavailable")
        bridge.value.agent = historySession().agent
        rerender(<SessionAgentChatPage />)
        expect(screen.queryByTestId("history-view")).not.toBeInTheDocument()

        setBridge("active", historySession())
        rerender(<SessionAgentChatPage />)
        expect(screen.getByTestId("history-view")).toBeInTheDocument()
        expect(app.viewProps).toMatchObject({ taskId: 7, title: "Earlier chat" })
      })

      it("keeps the start screen from taking focus while history is open, and lets it back on return", () => {
        mountWithHistory()
        expect(app.startProps?.autoFocus).toBe(true)

        goToList()
        expect(app.startProps?.autoFocus).toBe(false)
        fireEvent.click(screen.getByRole("button", { name: "list-back" }))
        expect(app.startProps?.autoFocus).toBe(true)
      })

      it("keeps the conversation panel from taking focus while history is open", () => {
        app.state.taskId = 61
        mountWithHistory({ sessionConversationState: "bound" })
        expect(app.panelProps?.autoFocusInput).toBe(true)

        goToList()
        expect(app.panelProps?.autoFocusInput).toBe(false)
      })
    })

    describe("going back to the chat by itself", () => {
      it.each(["list", "view"] as const)(
        "leaves the %s when the session needs a reload, and offers no way back in",
        (where) => {
          const { rerender } = mountWithHistory()
          if (where === "list") goToList()
          else goToView()

          app.sessionConversationState = "reload_required"
          app.isSessionInteractionLocked = true
          rerender(<SessionAgentChatPage />)

          expect(screen.queryByTestId("history-list")).not.toBeInTheDocument()
          expect(screen.queryByTestId("history-view")).not.toBeInTheDocument()
          openMenu()
          expect(historyItem()).not.toBeInTheDocument()
        },
      )

      it.each([
        ["a reset starts", { isConversationResetPending: true }],
        ["a message is being delivered", { isMessageDeliveryPending: true }],
      ])("leaves the list and the view when %s", (_label, overrides) => {
        const { rerender } = mountWithHistory()
        goToView()
        expect(screen.getByTestId("history-view")).toBeInTheDocument()

        Object.assign(app, overrides)
        rerender(<SessionAgentChatPage />)

        expect(screen.queryByTestId("history-view")).not.toBeInTheDocument()
        expect(screen.queryByTestId("history-list")).not.toBeInTheDocument()
      })

      it.each(["list", "view"] as const)("leaves the %s when the session stops offering history", (where) => {
        const { rerender } = mountWithHistory()
        if (where === "list") goToList()
        else goToView()

        setBridge("active", activeSession())
        rerender(<SessionAgentChatPage />)

        expect(screen.queryByTestId("history-list")).not.toBeInTheDocument()
        expect(screen.queryByTestId("history-view")).not.toBeInTheDocument()
        expect(screen.getByRole("button", { name: "start:Support Agent" })).toBeVisible()
      })

      it.each(["list", "view"] as const)("leaves the %s when a host page message arrives", (where) => {
        mountWithHistory({ sessionConversationAttachment: "pending" })
        if (where === "list") goToList()
        else goToView()

        deliverOpenMessage()

        expect(screen.queryByTestId("history-list")).not.toBeInTheDocument()
        expect(screen.queryByTestId("history-view")).not.toBeInTheDocument()
      })

      it("closes a list opened just before the host page message starts sending", () => {
        // Between the host message's reset and its send there is one render in
        // which nothing is pending, so the entry is still offered. The send
        // starts in that render's own effect, and the next render closes the
        // list again: nothing can be handed over from it in between.
        const { rerender } = mountWithHistory()
        goToList()
        expect(screen.getByTestId("history-list")).toBeInTheDocument()

        app.isMessageDeliveryPending = true
        rerender(<SessionAgentChatPage />)

        expect(screen.queryByTestId("history-list")).not.toBeInTheDocument()
        openMenu()
        expect(historyItem()).not.toBeInTheDocument()
      })

      it("leaves the view for the chat when the host page's initial message arrives", async () => {
        const post = vi.fn()
        vi.stubGlobal("parent", { postMessage: post })
        app.sendMessage.mockResolvedValue(undefined)
        setBridge("active", historySession())
        app.isConnected = true
        app.sessionConversationAttachment = "pending"
        const { rerender } = render(<SessionAgentChatPage />)
        goToView()

        app.sessionConversationAttachment = "attached"
        rerender(<SessionAgentChatPage />)
        await waitFor(() => expect(
          post.mock.calls.filter(([data]) => data.type === "initial_message_request"),
        ).toHaveLength(1))
        act(() => {
          window.dispatchEvent(new MessageEvent("message", {
            data: { xagent: true, v: 1, type: "initial_message", value: "hello agent" },
            source: window.parent as unknown as MessageEventSource,
          }))
        })

        expect(screen.queryByTestId("history-view")).not.toBeInTheDocument()
        await waitFor(() => expect(app.sendMessage).toHaveBeenCalledWith("hello agent", undefined, []))
      })
    })

    describe("how the list and the view read with the session token", () => {
      it.each(["unbound", "bound"] as const)(
        "refreshes a token that is nearly out of date and retries a rejected one in phase %s",
        async (phase) => {
          mountWithHistory({ sessionConversationState: phase })
          bridge.value.ensureFreshSession.mockResolvedValue({ token: "fresh_token" })
          bridge.value.refreshSession.mockResolvedValue({ token: "newer_token" })
          goToList()

          const auth = authOf(app.listProps)

          await expect(auth.getToken()).resolves.toBe("fresh_token")
          await expect(auth.refreshToken?.("rejected_token")).resolves.toBe("newer_token")
          expect(bridge.value.refreshSession).toHaveBeenCalledWith("rejected_token")
          expect(app.listProps?.unauthorizedNotice).toBe("tryLater")
        },
      )

      it.each([
        ["replacement_ready", "refreshToLoad"],
        ["reset_requested", "tryLater"],
        ["replacement_sending", "tryLater"],
        ["replacement_awaiting_task", "tryLater"],
      ] as const)("reads with the current token and never refreshes in phase %s", async (phase, notice) => {
        const { rerender } = mountWithHistory()
        goToList()
        // A phase the page would not open history in, reached while it is open.
        app.sessionConversationState = phase
        rerender(<SessionAgentChatPage />)
        const auth = authOf(app.listProps)

        await expect(auth.getToken()).resolves.toBe(SESSION_TOKEN)
        expect(auth.refreshToken).toBeUndefined()
        expect(bridge.value.ensureFreshSession).not.toHaveBeenCalled()
        expect(app.listProps?.unauthorizedNotice).toBe(notice)
      })

      it("fails the token request when the page has no session token to offer", async () => {
        const { rerender } = mountWithHistory()
        goToList()
        app.sessionConversationState = "replacement_ready"
        rerender(<SessionAgentChatPage />)
        const getAuth = app.listProps?.getAuth as () => HistoryAuth

        refreshingWithAgent()
        rerender(<SessionAgentChatPage />)

        await expect(getAuth().getToken()).rejects.toThrow()
      })

      it("reads the phase at the moment of the request, not at the moment the list opened", () => {
        const { rerender } = mountWithHistory()
        goToList()
        const getAuth = app.listProps?.getAuth as () => HistoryAuth
        expect(getAuth().refreshToken).toBeDefined()

        app.sessionConversationState = "replacement_ready"
        rerender(<SessionAgentChatPage />)

        expect(getAuth().refreshToken).toBeUndefined()
      })

      it("hands the view the current phase, replay state and notice", () => {
        mountWithHistory({ sessionConversationAttachment: "unsupported", sessionConversationState: "replacement_ready" })
        goToView()

        expect(app.viewProps).toMatchObject({
          sessionPhase: "replacement_ready",
          sessionAttachment: "unsupported",
          unauthorizedNotice: "refreshToLoad",
        })
      })
    })

    describe("continuing from an earlier conversation", () => {
      it("sends the message into a new conversation with the earlier conversation as its source", async () => {
        app.sendMessage.mockResolvedValue(undefined)
        mountWithHistory()
        goToView()

        hand()

        expect(screen.queryByTestId("history-view")).not.toBeInTheDocument()
        await waitFor(() => expect(app.sendMessage).toHaveBeenCalledTimes(1))
        expect(app.sendMessage).toHaveBeenCalledWith("a follow-up", { sourceTaskId: 7 }, [])
        expect(app.startNewConversation).not.toHaveBeenCalled()
      })

      it("starts a new conversation first when one is bound, and sends once it is ready", async () => {
        app.sendMessage.mockResolvedValue(undefined)
        // The reset is acknowledged before it settles, as the provider does:
        // the conversation is gone and the new one is ready for its first message.
        app.startNewConversation.mockImplementation(async () => {
          app.state.taskId = null
          app.sessionConversationState = "replacement_ready"
        })
        app.state.taskId = 61
        mountWithHistory({ sessionConversationState: "bound" })
        goToView()

        hand()

        await waitFor(() => expect(app.sendMessage).toHaveBeenCalledWith("a follow-up", { sourceTaskId: 7 }, []))
        expect(app.startNewConversation).toHaveBeenCalledTimes(1)
      })

      it("sends straight away when the new conversation is already open for the first message", async () => {
        app.sendMessage.mockResolvedValue(undefined)
        mountWithHistory({ sessionConversationState: "replacement_ready" })
        goToView()

        hand()

        await waitFor(() => expect(app.sendMessage).toHaveBeenCalledTimes(1))
        expect(app.startNewConversation).not.toHaveBeenCalled()
      })

      it("waits for the replay of the current socket before doing anything", () => {
        mountWithHistory({ sessionConversationAttachment: "pending" })
        goToView()

        hand()

        expect(app.sendMessage).not.toHaveBeenCalled()
        expect(app.startNewConversation).not.toHaveBeenCalled()
      })

      it("keeps the start screen from sending another message while one is handed over", () => {
        mountWithHistory({ sessionConversationAttachment: "pending" })
        goToView()
        expect(app.startProps?.isSending).toBe(false)

        hand()

        expect(app.startProps?.isSending).toBe(true)
      })

      it("lets a message in delivery finish before the hand-over resets the conversation", async () => {
        app.sendMessage.mockResolvedValue(undefined)
        app.state.taskId = 61
        const { rerender } = mountWithHistory({ sessionConversationState: "bound" })
        goToView()
        const onContinue = app.viewProps?.onContinue as (message: string, sourceTaskId: number) => void

        // The visitor sent a message in the old conversation; the view closes.
        app.isMessageDeliveryPending = true
        rerender(<SessionAgentChatPage />)
        act(() => onContinue("a follow-up", 7))
        expect(app.startNewConversation).not.toHaveBeenCalled()

        app.isMessageDeliveryPending = false
        rerender(<SessionAgentChatPage />)
        await waitFor(() => expect(app.startNewConversation).toHaveBeenCalledTimes(1))
      })

      it("never asks for the host page's initial message once a hand-over is under way, even if it fails", async () => {
        const post = vi.fn()
        vi.stubGlobal("parent", { postMessage: post })
        app.sendMessage.mockRejectedValue(new Error("delivery failed"))
        setBridge("active", historySession())
        app.isConnected = true
        app.sessionConversationAttachment = "pending"
        const { rerender } = render(<SessionAgentChatPage />)
        goToView()
        hand()

        app.sessionConversationAttachment = "attached"
        rerender(<SessionAgentChatPage />)
        await waitFor(() => expect(app.startProps?.isSending).toBe(false))
        await act(async () => undefined)

        expect(app.sendMessage).toHaveBeenCalledTimes(1)
        expect(post.mock.calls.filter(([data]) => data.type === "initial_message_request")).toEqual([])
      })

      describe("what the visitor is told when it does not go through", () => {
        const noticeText = "widgetSession.history.continueNotSent"

        it("says the message was not sent when the conversation is bound again after the reset", async () => {
          app.state.taskId = 61
          mountWithHistory({ sessionConversationState: "bound" })
          goToView()

          hand()

          expect(await screen.findByText(noticeText)).toBeInTheDocument()
          expect(screen.getByRole("alert")).toHaveTextContent(noticeText)
          expect(app.sendMessage).not.toHaveBeenCalled()
          expect(app.startNewConversation).toHaveBeenCalledTimes(1)
        })

        it("says the message was not sent when the server stops announcing replay completion", async () => {
          const { rerender } = mountWithHistory({ sessionConversationAttachment: "pending" })
          goToView()
          hand()

          app.sessionConversationAttachment = "unsupported"
          rerender(<SessionAgentChatPage />)

          expect(await screen.findByText(noticeText)).toBeInTheDocument()
          // The dropped message leaves nothing in hand: the start screen sends
          // again and the history entry is back.
          expect(app.startProps?.isSending).toBe(false)
          openMenu()
          expect(historyItem()).toBeInTheDocument()
        })

        it("clears the line when the conversation changes", async () => {
          const { rerender } = mountWithHistory({ sessionConversationAttachment: "pending" })
          goToView()
          hand()
          app.sessionConversationAttachment = "unsupported"
          rerender(<SessionAgentChatPage />)
          await screen.findByText(noticeText)

          app.state.taskId = 62
          rerender(<SessionAgentChatPage />)

          expect(screen.queryByText(noticeText)).not.toBeInTheDocument()
        })

        it("clears the line when another hand-over begins", async () => {
          const { rerender } = mountWithHistory({ sessionConversationAttachment: "pending" })
          goToView()
          hand()
          app.sessionConversationAttachment = "unsupported"
          rerender(<SessionAgentChatPage />)
          await screen.findByText(noticeText)

          goToView()
          app.sessionConversationAttachment = "pending"
          rerender(<SessionAgentChatPage />)
          hand()

          expect(screen.queryByText(noticeText)).not.toBeInTheDocument()
        })

        it("shows only the reload instruction when the session needs a reload", async () => {
          const { rerender } = mountWithHistory({ sessionConversationAttachment: "pending" })
          goToView()
          hand()
          app.sessionConversationAttachment = "unsupported"
          rerender(<SessionAgentChatPage />)
          await screen.findByText(noticeText)

          app.sessionConversationState = "reload_required"
          app.isSessionInteractionLocked = true
          rerender(<SessionAgentChatPage />)

          expect(screen.queryByText(noticeText)).not.toBeInTheDocument()
          expect(screen.getByRole("alert")).toHaveTextContent("widgetSession.reloadRequired")
        })

        it("does not say anything when the message was sent", async () => {
          app.sendMessage.mockResolvedValue(undefined)
          mountWithHistory()
          goToView()

          hand()
          await waitFor(() => expect(app.sendMessage).toHaveBeenCalledTimes(1))

          expect(screen.queryByText(noticeText)).not.toBeInTheDocument()
        })

        it("says the message was not sent when it is refused after a conversation opened", async () => {
          let refuse!: (reason: unknown) => void
          app.sendMessage.mockReturnValueOnce(new Promise((_resolve, reject) => { refuse = reject }))
          const { rerender } = mountWithHistory()
          goToView()
          hand()
          await waitFor(() => expect(app.sendMessage).toHaveBeenCalledTimes(1))

          // Another message opened a conversation before the refusal came back,
          // so the start screen and its failure line are gone.
          app.state.taskId = 42
          rerender(<SessionAgentChatPage />)
          await act(async () => {
            refuse(Object.assign(new Error("refused"), { errorCode: "invalid_message" }))
          })

          expect(await screen.findByText(noticeText)).toBeInTheDocument()
          expect(screen.getByRole("alert")).toHaveTextContent(noticeText)
          expect(screen.queryByText("clientErrors.invalidMessage")).not.toBeInTheDocument()
        })

        it("leaves a refusal on the start page to the failure line there", async () => {
          app.sendMessage.mockRejectedValueOnce(Object.assign(new Error("refused"), { errorCode: "invalid_message" }))
          mountWithHistory()
          goToView()

          hand()

          expect(await screen.findByText("clientErrors.invalidMessage")).toBeInTheDocument()
          expect(screen.queryByText(noticeText)).not.toBeInTheDocument()
        })

        it("shows the reset failure line when the new conversation cannot be started", async () => {
          app.startNewConversation.mockRejectedValue(new Error("reset refused"))
          app.state.taskId = 61
          mountWithHistory({ sessionConversationState: "bound" })
          goToView()

          hand()

          expect(await screen.findByText("widgetSession.resetFailed")).toBeInTheDocument()
          expect(app.sendMessage).not.toHaveBeenCalled()
        })
      })
    })

    describe("the failure line on the start screen", () => {
      const sendFrom = async (config: Record<string, unknown>, error: unknown) => {
        app.sendMessage.mockRejectedValueOnce(error)
        const onSend = app.startProps?.onSend as (
          message: string,
          files: File[],
          config?: Record<string, unknown>,
        ) => Promise<void>
        await act(async () => {
          await onSend("hi", [], config).catch(() => undefined)
        })
      }
      const coded = (code: string, handled = false) => Object.assign(new Error("failed"), {
        errorCode: code,
        ...(handled ? { notificationHandled: true } : {}),
      })

      it.each([
        ["with a source and task_unavailable", { sourceTaskId: 5 }, "task_unavailable",
          "widgetSession.history.continueUnavailable"],
        ["with a source and invalid_message", { sourceTaskId: 5 }, "invalid_message",
          "clientErrors.invalidMessage"],
        ["with a source and no known code", { sourceTaskId: 5 }, "not_a_code",
          "widgetSession.startMessageFailed"],
        ["with a source and message_processing_failed", { sourceTaskId: 5 }, "message_processing_failed",
          "clientErrors.messageProcessingFailed"],
        ["without a source and task_unavailable", {}, "task_unavailable", "clientErrors.taskUnavailable"],
        ["without a source and invalid_message", {}, "invalid_message", "clientErrors.invalidMessage"],
        ["without a source and no known code", {}, "not_a_code", "widgetSession.startMessageFailed"],
        ["with a source that is not a number", { sourceTaskId: "5" }, "task_unavailable",
          "clientErrors.taskUnavailable"],
      ])("says the right thing for a send %s", async (_label, config, code, expected) => {
        setBridge("active", activeSession())
        render(<SessionAgentChatPage />)

        await sendFrom(config, coded(code))

        expect(screen.getByRole("alert")).toHaveTextContent(expected)
        if (expected !== "widgetSession.history.continueUnavailable") {
          expect(screen.queryByText("widgetSession.history.continueUnavailable")).not.toBeInTheDocument()
        }
      })

      it.each([
        ["with a source", { sourceTaskId: 5 }],
        ["without a source", {}],
      ])("shows nothing for a failure that was already announced, %s", async (_label, config) => {
        setBridge("active", activeSession())
        render(<SessionAgentChatPage />)

        await sendFrom(config, coded("task_unavailable", true))

        expect(screen.queryByRole("alert")).not.toBeInTheDocument()
      })

      it("shows the dedicated line when a handed-over message is refused as unavailable", async () => {
        app.sendMessage.mockRejectedValue(coded("task_unavailable"))
        mountWithHistory()
        goToView()

        hand()

        expect(await screen.findByText("widgetSession.history.continueUnavailable")).toBeInTheDocument()
      })

      it("shows the ordinary line when a handed-over message is refused as invalid", async () => {
        app.sendMessage.mockRejectedValue(coded("invalid_message"))
        mountWithHistory()
        goToView()

        hand()

        expect(await screen.findByText("clientErrors.invalidMessage")).toBeInTheDocument()
      })
    })
  })

})

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

    it("sends the delivered message once the connection is open", async () => {
      const post = stubParent()
      setBridge("active", activeSession())
      app.isConnected = true
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

    it("does not send after a first message queued before the connection opened", () => {
      const post = stubParent()
      setBridge("active", activeSession())
      app.isConnected = false
      app.isMessageDeliveryPending = true
      const { rerender } = render(<SessionAgentChatPage />)

      app.isConnected = true
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

    it("does not ask while a conversation already exists", () => {
      const post = stubParent()
      setBridge("active", activeSession())
      app.isConnected = true
      app.state.taskId = 7

      render(<SessionAgentChatPage />)

      expect(initialRequests(post)).toEqual([])
    })

    it("surfaces a failed send on the start screen", async () => {
      const post = stubParent()
      setBridge("active", activeSession())
      app.isConnected = true
      app.sendMessage.mockRejectedValue(new Error("delivery failed"))

      render(<SessionAgentChatPage />)
      await waitFor(() => expect(initialRequests(post)).toHaveLength(1))
      deliverInitialMessage("hello agent")

      expect(await screen.findByText("widgetSession.startMessageFailed")).toBeInTheDocument()
    })
  })
})

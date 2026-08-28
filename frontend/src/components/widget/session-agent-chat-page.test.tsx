import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { AppProviderTransportConfig } from "@/contexts/app-context-chat"
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
  },
}))

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
  useI18n: () => ({ t: (key: string) => key }),
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
    cleanup()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it("keeps the Session capability provider mounted while waiting with no connection", () => {
    render(<SessionAgentChatPage />)

    expect(screen.getByText("widgetChat.status.initializing")).toBeInTheDocument()
    expect(app.provider?.token).toBeUndefined()
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
      ],
      expectedProtocol: "xagent-session-v1",
      chatTaskIdMode: "omit",
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
    // No conversation yet — nothing to reset, so the "..." menu (which would
    // only ever hold the new-conversation action) doesn't render either. (Not
    // asserting the menuitem itself is absent here: the menu was never
    // opened, so that check can't fail regardless of whether this logic
    // works — the trigger's own absence, below, is what actually proves it.)
    expect(screen.queryByRole("button", { name: "widgetChat.moreOptions" })).toBeNull()
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

    expect(trigger).toBeDisabled()
    expect(screen.queryByRole("menu")).toBeNull()
    // Not just "disabled" -- the whole point of this prop is a visible
    // in-progress indicator on the trigger once the menu itself has closed.
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
    expect(screen.queryByTestId("session-conversation-panel")).not.toBeInTheDocument()
    expect(app.provider?.transport?.session?.connection).toBeNull()
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
})

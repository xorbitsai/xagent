import React from "react"
import { act, cleanup, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const authState = vi.hoisted(() => ({
  token: null as string | null,
  user: null as { id: string } | null,
  refreshAccessToken: vi.fn(async () => false),
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => authState,
}))

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    refresh: vi.fn(),
  }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

vi.mock("sonner", () => ({
  toast: {
    error: vi.fn(),
    success: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
  },
}))

vi.mock("@/components/chat/ChatStartScreen", () => ({
  ChatStartScreen: ({ title }: { title: string }) => (
    <div data-testid="session-start">{title}</div>
  ),
}))

vi.mock("@/components/task/task-conversation-panel", () => ({
  TaskConversationPanel: () => <div data-testid="session-conversation" />,
}))

import { SessionAgentChatPage } from "./session-agent-chat-page"

const PARENT_ORIGIN = "https://shiftcare.example"
const SESSION_TOKEN = "st_integration_secret"
const EMBEDDED_PARENT = {
  postMessage: (...args: Parameters<Window["postMessage"]>) => window.postMessage(...args),
}
const originalParentDescriptor = Object.getOwnPropertyDescriptor(window, "parent")

class MockWebSocket {
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3
  static instances: MockWebSocket[] = []
  static failConstruction = false
  static constructionCount = 0

  readyState = 0
  protocol = ""
  onopen: (() => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  send = vi.fn()
  close = vi.fn(() => {
    this.readyState = MockWebSocket.CLOSED
  })

  constructor(
    public url: string,
    public protocols?: string | string[],
  ) {
    MockWebSocket.constructionCount += 1
    if (MockWebSocket.failConstruction) {
      throw new Error(`native constructor rejected ${SESSION_TOKEN}`)
    }
    MockWebSocket.instances.push(this)
  }

  open(protocol: string) {
    this.protocol = protocol
    this.readyState = MockWebSocket.OPEN
    this.onopen?.()
  }

  triggerError() {
    this.onerror?.(new Event("error"))
  }

  triggerClose(code = 1006) {
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.(new CloseEvent("close", { code }))
  }
}

const sessionUpdate = () => {
  const now = Date.now()
  return {
    xagent: true,
    v: 1,
    type: "session_update",
    session_delivery_id: "delivery-a",
    session_token: SESSION_TOKEN,
    session_token_expires_at: new Date(now + 15 * 60_000).toISOString(),
    absolute_expires_at: new Date(now + 30 * 60_000).toISOString(),
    agent: {
      id: 42,
      name: "Support Agent",
      suggested_prompts: [],
    },
  }
}

const dispatchSession = () => {
  act(() => {
    window.dispatchEvent(new MessageEvent("message", {
      data: sessionUpdate(),
      origin: PARENT_ORIGIN,
      source: EMBEDDED_PARENT as unknown as MessageEventSource,
    }))
  })
}

const dispatchDegradedSession = () => {
  act(() => {
    window.dispatchEvent(new MessageEvent("message", {
      data: {
        xagent: true,
        v: 1,
        type: "session_degraded",
        code: "network_unavailable",
      },
      origin: PARENT_ORIGIN,
      source: EMBEDDED_PARENT as unknown as MessageEventSource,
    }))
  })
}

describe("SessionAgentChatPage connection failure integration", () => {
  beforeEach(() => {
    MockWebSocket.instances = []
    MockWebSocket.failConstruction = false
    MockWebSocket.constructionCount = 0
    localStorage.clear()
    sessionStorage.clear()
    vi.stubGlobal("WebSocket", MockWebSocket)
    Object.defineProperty(window, "parent", {
      configurable: true,
      value: EMBEDDED_PARENT,
    })
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    if (originalParentDescriptor) Object.defineProperty(window, "parent", originalParentDescriptor)
  })

  it.each([
    ["constructor setup", true],
    ["required protocol negotiation", false],
  ])(
    "moves to a terminal credential-free UI after %s failure",
    async (_label, failConstruction) => {
      MockWebSocket.failConstruction = failConstruction
      const storageSet = vi.spyOn(Storage.prototype, "setItem")
      const parentPostMessage = vi.spyOn(window, "postMessage")
        .mockImplementation(() => {})
      const consoleLog = vi.spyOn(console, "log")
        .mockImplementation(() => {})
      const consoleWarn = vi.spyOn(console, "warn")
        .mockImplementation(() => {})
      const consoleError = vi.spyOn(console, "error")
        .mockImplementation(() => {})

      render(<SessionAgentChatPage />)
      dispatchSession()

      if (!failConstruction) {
        await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
        act(() => MockWebSocket.instances[0].open(""))
      }

      await waitFor(() => {
        expect(screen.getByRole("heading", {
          name: "widgetSession.unavailable.title",
        })).toBeInTheDocument()
      })

      expect(screen.queryByText("Support Agent")).not.toBeInTheDocument()
      expect(document.documentElement.innerHTML).not.toContain(SESSION_TOKEN)
      expect(window.location.href).not.toContain(SESSION_TOKEN)
      expect(storageSet).not.toHaveBeenCalled()
      const serializedLogs = [
        ...consoleLog.mock.calls,
        ...consoleWarn.mock.calls,
        ...consoleError.mock.calls,
      ].flat().map(String).join(" ")
      expect(serializedLogs).not.toContain(SESSION_TOKEN)
      expect(parentPostMessage.mock.calls.filter(
        ([message]) => message?.type === "reconnect_request",
      )).toHaveLength(0)
      expect(MockWebSocket.constructionCount).toBe(1)
    },
  )

  it("requests one parent reconnect for a physical failure and its abnormal close", async () => {
    const parentPostMessage = vi.spyOn(window, "postMessage")
      .mockImplementation(() => {})
    vi.spyOn(console, "error").mockImplementation(() => {})

    render(<SessionAgentChatPage />)
    dispatchSession()
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))

    const socket = MockWebSocket.instances[0]
    act(() => {
      socket.open("xagent-session-v1")
      socket.triggerError()
      socket.triggerClose()
    })

    await waitFor(() => {
      expect(parentPostMessage.mock.calls.filter(
        ([message]) => message?.type === "reconnect_request",
      )).toEqual([[
        {
          xagent: true,
          v: 1,
          type: "reconnect_request",
          reason: "ws_closed",
        },
        PARENT_ORIGIN,
      ]])
    })
    expect(screen.getByRole("heading", { name: "Support Agent" }))
      .toBeInTheDocument()
    expect(screen.getByText("widgetChat.status.connecting"))
      .toBeInTheDocument()
    expect(MockWebSocket.constructionCount).toBe(1)
  })

  it("echoes the exact delivery discriminator after the current socket opens", async () => {
    const parentPostMessage = vi.spyOn(window, "postMessage")
      .mockImplementation(() => {})

    render(<SessionAgentChatPage />)
    dispatchSession()
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    act(() => MockWebSocket.instances[0].open("xagent-session-v1"))

    expect(parentPostMessage).toHaveBeenCalledWith(
      {
        xagent: true,
        v: 1,
        type: "session_connection_open",
        session_delivery_id: "delivery-a",
      },
      PARENT_ORIGIN,
    )
  })

  it("renders an existing Session as unavailable after a degraded parent update", async () => {
    render(<SessionAgentChatPage />)
    dispatchSession()
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))

    const socket = MockWebSocket.instances[0]
    act(() => socket.open("xagent-session-v1"))
    dispatchDegradedSession()

    await waitFor(() => {
      expect(screen.getByRole("heading", {
        name: "widgetSession.unavailable.title",
      })).toBeInTheDocument()
    })
    expect(screen.queryByTestId("session-start")).not.toBeInTheDocument()
    expect(MockWebSocket.constructionCount).toBe(1)
    expect(socket.close).toHaveBeenCalledTimes(1)
    expect(socket.readyState).toBe(MockWebSocket.CLOSED)
  })
})

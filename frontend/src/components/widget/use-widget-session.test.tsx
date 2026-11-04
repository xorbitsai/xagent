import { act, cleanup, renderHook } from "@testing-library/react"
import { StrictMode } from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { uploadWidgetSessionFile } from "@/lib/widget-session-upload"

import {
  buildWidgetSessionWebSocketUrl,
  useWidgetSession,
  type WidgetSession,
} from "./use-widget-session"

const PARENT_ORIGIN = "https://embed.example"
const EMBEDDED_PARENT = {
  postMessage: (...args: Parameters<Window["postMessage"]>) => window.postMessage(...args),
}
const originalParentDescriptor = Object.getOwnPropertyDescriptor(window, "parent")

function updateMessage(overrides: Record<string, unknown> = {}) {
  const now = Date.now()
  return {
    xagent: true,
    v: 1,
    type: "session_update",
    session_delivery_id: "delivery-a",
    session_token: "st_session_token",
    session_token_expires_at: new Date(now + 15 * 60_000).toISOString(),
    absolute_expires_at: new Date(now + 30 * 60_000).toISOString(),
    agent: {
      id: 42,
      name: "Support Agent",
      description: "Helps with schedules",
      logo_url: "https://cdn.example/logo.png",
      suggested_prompts: ["Show my schedule"],
    },
    ...overrides,
  }
}

function dispatchFromParent(
  data: Record<string, unknown>,
  origin = PARENT_ORIGIN,
  source: MessageEventSource | null = EMBEDDED_PARENT as unknown as MessageEventSource,
) {
  act(() => {
    window.dispatchEvent(new MessageEvent("message", { data, origin, source }))
  })
}

beforeEach(() => {
  Object.defineProperty(window, "parent", {
    configurable: true,
    value: EMBEDDED_PARENT,
  })
})

afterEach(() => {
  if (originalParentDescriptor) Object.defineProperty(window, "parent", originalParentDescriptor)
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe("useWidgetSession", () => {
  it.each([
    null, [], {},
    { id: 0, name: "Agent", suggested_prompts: [] },
    { id: 1, name: " ", suggested_prompts: [] },
    { id: 1, name: "Agent", description: 1, suggested_prompts: [] },
    { id: 1, name: "Agent", logo_url: {}, suggested_prompts: [] },
    { id: 1, name: "Agent", suggested_prompts: [1] },
  ])("rejects malformed untrusted agent metadata: %p", agent => {
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({ agent }))
    expect(result.current.status).toBe("terminal")
    expect(result.current.session).toBeNull()
    expect(result.current.terminalCode).toBe("unexpected_error")
  })

  it.each([null, [], false, "invalid"])("ignores non-record parent messages: %p", data => {
    const { result } = renderHook(() => useWidgetSession())
    act(() => window.dispatchEvent(new MessageEvent("message", {
      data, origin: PARENT_ORIGIN, source: EMBEDDED_PARENT as unknown as Window,
    })))
    expect(result.current.status).toBe("waiting")
  })

  it("rejects non-HTTP WebSocket origins", () => {
    expect(() => buildWidgetSessionWebSocketUrl("file:///tmp/result")).toThrow("HTTP(S)")
  })

  it("announces credential-free readiness with the bootstrap target origin", () => {
    const postMessage = vi.spyOn(window, "postMessage")

    renderHook(() => useWidgetSession())

    expect(postMessage).toHaveBeenCalledWith(
      { xagent: true, v: 1, type: "ready" },
      "*",
    )
  })

  it("echoes the delivery ID only after the current socket generation opens", () => {
    const postMessage = vi.spyOn(window, "postMessage")
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({ session_delivery_id: "delivery-b" }))

    act(() => result.current.handleConnectionOpen("widget-session:999"))
    act(() => result.current.handleConnectionOpen("widget-session:1"))

    expect(postMessage.mock.calls.filter(([message]) => message?.type === "session_connection_open")).toEqual([[
      { xagent: true, v: 1, type: "session_connection_open", session_delivery_id: "delivery-b" },
      PARENT_ORIGIN,
    ]])
  })

  it.each([undefined, " ", 1, true, {}, []])(
    "fails closed on malformed delivery IDs before replacing the active generation: %p",
    (session_delivery_id) => {
      const postMessage = vi.spyOn(window, "postMessage")
      const { result } = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage({ session_delivery_id: "delivery-a" }))
      const previousGeneration = result.current.session?.generation

      dispatchFromParent(updateMessage({ session_delivery_id }))

      expect(previousGeneration).toBe(1)
      expect(result.current.status).toBe("terminal")
      expect(result.current.session).toBeNull()
      expect(postMessage.mock.calls.filter(([message]) => message?.type === "reconnect_request")).toHaveLength(0)
    },
  )

  it("accepts a legacy v1 update with no delivery ID and never echoes a stale ID", () => {
    const postMessage = vi.spyOn(window, "postMessage")
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({ session_delivery_id: "delivery-a" }))

    const legacyUpdate: Record<string, unknown> = updateMessage()
    delete legacyUpdate.session_delivery_id
    dispatchFromParent(legacyUpdate)

    expect(result.current.status).toBe("active")
    expect(result.current.session?.generation).toBe(2)

    act(() => result.current.handleConnectionOpen("widget-session:2"))

    expect(postMessage.mock.calls.filter(([message]) => message?.type === "session_connection_open")).toHaveLength(0)
  })

  it("pins the first valid parent origin internally and rejects messages from another origin", () => {
    const { result } = renderHook(() => useWidgetSession())

    dispatchFromParent(updateMessage(), PARENT_ORIGIN, null)
    dispatchFromParent({ xagent: true, v: 2, type: "session_terminal", code: "ignored" })
    dispatchFromParent(updateMessage())
    dispatchFromParent(
      { xagent: true, v: 1, type: "session_terminal", code: "session_expired" },
      "https://other.example",
    )

    expect(result.current.status).toBe("active")
    expect(result.current).not.toHaveProperty("parentOrigin")
  })

  it("fails closed in a top-level window without self-handshaking", () => {
    Object.defineProperty(window, "parent", {
      configurable: true,
      value: window,
    })
    const postMessage = vi.spyOn(window, "postMessage")
    const { result } = renderHook(() => useWidgetSession())

    expect(result.current.status).toBe("terminal")
    expect(result.current.terminalCode).toBe("unexpected_error")
    expect(postMessage).not.toHaveBeenCalled()
  })

  it("keeps a degraded parent failure nonterminal for a later session update", () => {
    const { result } = renderHook(() => useWidgetSession())

    dispatchFromParent({
      xagent: true,
      v: 1,
      type: "session_degraded",
      code: "network_unavailable",
    })

    expect(result.current.status).toBe("degraded")
    expect(result.current.session).toBeNull()
    expect(result.current.agent).toBeNull()
    expect(result.current.terminalCode).toBeNull()

    dispatchFromParent(updateMessage())

    expect(result.current.status).toBe("active")
    expect(result.current.session?.token).toBe("st_session_token")
  })

  it.each([undefined, "future_recovery_code"])(
    "fails closed on an invalid session_degraded code %s",
    (code) => {
      const { result } = renderHook(() => useWidgetSession())

      dispatchFromParent({
        xagent: true,
        v: 1,
        type: "session_degraded",
        ...(code === undefined ? {} : { code }),
      })

      expect(result.current.status).toBe("terminal")
      expect(result.current.terminalCode).toBe("unexpected_error")
    },
  )

  it("does not create another reconnect request after session_degraded", () => {
    vi.useFakeTimers()
    const postMessage = vi.spyOn(EMBEDDED_PARENT, "postMessage").mockImplementation(() => undefined)
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage())

    act(() => result.current.requestReconnect("ws_closed"))
    expect(postMessage.mock.calls.filter((call) => call[0]?.type === "reconnect_request")).toHaveLength(1)

    dispatchFromParent({
      xagent: true,
      v: 1,
      type: "session_degraded",
      code: "rate_limited",
    })
    act(() => vi.advanceTimersByTime(60_000))

    expect(result.current.status).toBe("degraded")
    expect(result.current.session).toBeNull()
    expect(result.current.agent?.name).toBe("Support Agent")
    expect(result.current.terminalCode).toBeNull()
    expect(postMessage.mock.calls.filter((call) => call[0]?.type === "reconnect_request")).toHaveLength(1)

    dispatchFromParent(updateMessage({ session_token: "st_recovered" }))
    expect(result.current.status).toBe("active")
    expect(result.current.session?.token).toBe("st_recovered")
  })

  it("sends an exact-origin reconnect request once and removes the usable token", () => {
    const postMessage = vi.spyOn(window, "postMessage")
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage())

    act(() => {
      result.current.requestReconnect("ws_closed")
      result.current.requestReconnect("ws_closed")
    })

    expect(result.current.status).toBe("refreshing")
    expect(result.current.session).toBeNull()
    expect(postMessage).toHaveBeenLastCalledWith(
      { xagent: true, v: 1, type: "reconnect_request", reason: "ws_closed" },
      PARENT_ORIGIN,
    )
    expect(postMessage.mock.calls.filter((call) => call[0]?.type === "reconnect_request")).toHaveLength(1)
  })

  it("owns recoverable transport failures with the existing one-shot reconnect latch", () => {
    const postMessage = vi.spyOn(window, "postMessage")
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage())
    postMessage.mockClear()

    act(() => {
      result.current.handleConnectionFailure({
        recoverable: true,
        error: new Error("physical transport failed"),
      })
      result.current.handleConnectionClose(
        new CloseEvent("close", { code: 1006 }),
      )
      result.current.handleConnectionFailure({
        recoverable: true,
        error: new Error("duplicate physical transport failure"),
      })
    })

    expect(result.current.status).toBe("refreshing")
    expect(result.current.session).toBeNull()
    expect(result.current.agent?.name).toBe("Support Agent")
    expect(postMessage.mock.calls.filter((call) => call[0]?.type === "reconnect_request")).toEqual([
      [{
        xagent: true,
        v: 1,
        type: "reconnect_request",
        reason: "ws_closed",
      }, PARENT_ORIGIN],
    ])
  })

  it(
    "fails terminal and wipes Session credentials for an unrecoverable connection failure",
    () => {
      const postMessage = vi.spyOn(window, "postMessage")
      const { result } = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage())
      postMessage.mockClear()

      act(() => {
        result.current.handleConnectionFailure({
          recoverable: false,
          error: new Error("sanitized failure"),
        })
      })

      expect(result.current.status).toBe("terminal")
      expect(result.current.terminalCode).toBe("unexpected_error")
      expect(result.current.session).toBeNull()
      expect(result.current.agent).toBeNull()
      expect(postMessage).not.toHaveBeenCalled()
    },
  )

  it.each([4401, 1006, 4500])(
    "owns abnormal WebSocket close %s and requests one parent reconnect",
    (code) => {
      const postMessage = vi.spyOn(window, "postMessage")
      const { result } = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage())

      let disposition: "handled" | "default" | undefined
      act(() => {
        disposition = result.current.handleConnectionClose(
          new CloseEvent("close", { code }),
        )
      })

      expect(disposition).toBe("handled")
      expect(result.current.status).toBe("refreshing")
      expect(result.current.session).toBeNull()
      expect(result.current.agent?.name).toBe("Support Agent")
      expect(postMessage).toHaveBeenLastCalledWith(
        { xagent: true, v: 1, type: "reconnect_request", reason: "ws_closed" },
        PARENT_ORIGIN,
      )
    },
  )

  it.each([
    [4403, "ws_4403"],
    [4408, "ws_4408"],
  ])(
    "owns terminal WebSocket close %s without asking the parent to reconnect",
    (code, terminalCode) => {
      const postMessage = vi.spyOn(window, "postMessage")
      const { result } = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage())
      postMessage.mockClear()

      let disposition: "handled" | "default" | undefined
      act(() => {
        disposition = result.current.handleConnectionClose(
          new CloseEvent("close", { code }),
        )
      })

      expect(disposition).toBe("handled")
      expect(result.current.status).toBe("terminal")
      expect(result.current.terminalCode).toBe(terminalCode)
      expect(result.current.session).toBeNull()
      expect(result.current.agent).toBeNull()
      expect(postMessage).not.toHaveBeenCalled()
    },
  )

  it.each([1000, 1001])("terminalizes a current-owner standard WebSocket close %s without reconnecting", (code) => {
    const postMessage = vi.spyOn(window, "postMessage")
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage())
    postMessage.mockClear()

    let disposition: "handled" | "default" | undefined
    act(() => {
      disposition = result.current.handleConnectionClose(
        new CloseEvent("close", { code }),
      )
    })

    expect(disposition).toBe("handled")
    expect(result.current.status).toBe("terminal")
    expect(result.current.terminalCode).toBe("unexpected_error")
    expect(result.current.session).toBeNull()
    expect(result.current.agent).toBeNull()
    expect(postMessage.mock.calls.filter((call) => call[0]?.type === "reconnect_request")).toHaveLength(0)
  })

  it("does nothing when a retained reconnect callback runs after unmount", () => {
    const postMessage = vi.spyOn(window, "postMessage")
    const { result, unmount } = renderHook(() => useWidgetSession(), { wrapper: StrictMode })
    dispatchFromParent(updateMessage())
    const requestReconnect = result.current.requestReconnect
    const snapshot = result.current

    unmount()
    postMessage.mockClear()
    act(() => requestReconnect("ws_closed"))

    expect(postMessage).not.toHaveBeenCalled()
    expect(result.current).toBe(snapshot)
  })

  it("rejects a whitespace-only token without normalizing a nonblank raw token", () => {
    const first = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({ session_token: "   " }))

    expect(first.result.current.status).toBe("terminal")
    expect(first.result.current.session).toBeNull()
    first.unmount()

    const second = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({ session_token: "  st.raw.value  " }))

    expect(second.result.current.status).toBe("active")
    expect(second.result.current.session?.token).toBe("  st.raw.value  ")
  })

  it.each([
    ["missing", undefined],
    ["non-array", "not-an-array"],
  ])("rejects %s suggested prompts", (_label, suggestedPrompts) => {
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({
      agent: {
        id: 42,
        name: "Support Agent",
        suggested_prompts: suggestedPrompts,
      },
    }))

    expect(result.current.status).toBe("terminal")
    expect(result.current.terminalCode).toBe("unexpected_error")
    expect(result.current.session).toBeNull()
  })

  it("accepts nullable optional agent metadata without terminalizing the Session", () => {
    const { result } = renderHook(() => useWidgetSession())

    dispatchFromParent(updateMessage({
      agent: {
        id: 42,
        name: "Support Agent",
        description: null,
        logo_url: null,
        suggested_prompts: [],
      },
    }))

    expect(result.current.status).toBe("active")
    expect(result.current.agent).toEqual({
      id: 42,
      name: "Support Agent",
      suggestedPrompts: [],
    })
  })

  it("requests one refresh instead of exposing a token with less than one minute remaining", () => {
    const postMessage = vi.spyOn(window, "postMessage")
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({ session_token_expires_at: new Date(Date.now() + 59_000).toISOString() }))

    expect(result.current.status).toBe("refreshing")
    expect(result.current.session).toBeNull()
    expect(postMessage).toHaveBeenLastCalledWith(
      { xagent: true, v: 1, type: "reconnect_request", reason: "token_expired" },
      PARENT_ORIGIN,
    )
  })

  it("waits for a parent result when near-expiry updates arrive during recovery", () => {
    vi.useFakeTimers()
    const postMessage = vi.spyOn(EMBEDDED_PARENT, "postMessage").mockImplementation(() => undefined)
    const { result } = renderHook(() => useWidgetSession())

    dispatchFromParent(updateMessage({
      session_token_expires_at: new Date(Date.now() + 59_000).toISOString(),
    }))
    dispatchFromParent(updateMessage({
      session_token_expires_at: new Date(Date.now() + 59_000).toISOString(),
    }))
    act(() => vi.advanceTimersByTime(1_000))
    dispatchFromParent(updateMessage({
      session_token_expires_at: new Date(Date.now() + 59_000).toISOString(),
    }))
    act(() => vi.advanceTimersByTime(2_000))
    dispatchFromParent(updateMessage({
      session_token_expires_at: new Date(Date.now() + 59_000).toISOString(),
    }))

    expect(result.current.status).toBe("refreshing")
    expect(result.current.terminalCode).toBeNull()
    expect(postMessage.mock.calls.filter((call) => call[0]?.type === "reconnect_request")).toHaveLength(1)
  })

  it("waits for the parent-owned retry phase instead of starting its own deadline", () => {
    vi.useFakeTimers()
    const postMessage = vi.spyOn(window, "postMessage")
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage())

    act(() => result.current.requestReconnect("ws_closed"))
    act(() => vi.advanceTimersByTime(30_000))

    expect(result.current.status).toBe("refreshing")
    expect(result.current.terminalCode).toBeNull()
    expect(postMessage.mock.calls.filter((call) => call[0]?.type === "reconnect_request")).toHaveLength(1)
  })

  it("coalesces socket closes while the parent recovery phase is active", () => {
    vi.useFakeTimers()
    const postMessage = vi.spyOn(window, "postMessage")
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage())

    act(() => result.current.requestReconnect("ws_closed"))
    act(() => result.current.handleConnectionClose(new CloseEvent("close", { code: 1006 })))
    act(() => result.current.handleConnectionClose(new CloseEvent("close", { code: 1006 })))
    act(() => vi.advanceTimersByTime(15_000))
    act(() => result.current.handleConnectionClose(new CloseEvent("close", { code: 1006 })))

    expect(result.current.status).toBe("refreshing")
    expect(result.current.terminalCode).toBeNull()
    expect(postMessage.mock.calls.filter((call) => call[0]?.type === "reconnect_request")).toHaveLength(1)
  })

  it("starts the stability window from the validated socket open for the active generation", () => {
    vi.useFakeTimers()
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage())

    act(() => result.current.requestReconnect("ws_closed"))
    dispatchFromParent(updateMessage())
    act(() => result.current.handleConnectionClose(new CloseEvent("close", { code: 1006 })))
    dispatchFromParent(updateMessage())

    act(() => vi.advanceTimersByTime(10_000))
    act(() => result.current.handleConnectionOpen("widget-session:3"))
    act(() => vi.advanceTimersByTime(15_000))
    act(() => result.current.handleConnectionClose(new CloseEvent("close", { code: 1006 })))

    expect(result.current.status).toBe("refreshing")
    expect(result.current.terminalCode).toBeNull()
  })

  it("coalesces socket closes before the parent has replied", () => {
    vi.useFakeTimers()
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage())

    act(() => result.current.requestReconnect("ws_closed"))
    dispatchFromParent(updateMessage())
    act(() => result.current.handleConnectionClose(new CloseEvent("close", { code: 1006 })))
    dispatchFromParent(updateMessage())

    act(() => result.current.handleConnectionOpen("widget-session:3"))
    act(() => vi.advanceTimersByTime(14_999))
    act(() => result.current.handleConnectionClose(new CloseEvent("close", { code: 1006 })))
    dispatchFromParent(updateMessage())
    act(() => result.current.handleConnectionClose(new CloseEvent("close", { code: 1006 })))

    expect(result.current.status).toBe("refreshing")
    expect(result.current.terminalCode).toBeNull()
  })

  it("ignores a stale socket close or failure after accepting a newer Session generation", () => {
    const postMessage = vi.spyOn(window, "postMessage")
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage())
    dispatchFromParent(updateMessage())

    act(() => result.current.handleConnectionClose(
      new CloseEvent("close", { code: 1006 }),
      "widget-session:1",
    ))
    act(() => result.current.handleConnectionFailure(
      {
        recoverable: true,
        error: new Error("stale socket failure"),
      },
      "widget-session:1",
    ))

    expect(result.current.status).toBe("active")
    expect(result.current.session?.generation).toBe(2)
    expect(result.current.terminalCode).toBeNull()
    expect(postMessage.mock.calls.filter((call) => call[0]?.type === "reconnect_request")).toHaveLength(0)
  })

  it("cancels the stability window when a current owner cleanly closes", () => {
    vi.useFakeTimers()
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage())

    act(() => result.current.requestReconnect("ws_closed"))
    dispatchFromParent(updateMessage())
    act(() => result.current.handleConnectionClose(new CloseEvent("close", { code: 1006 })))
    dispatchFromParent(updateMessage())
    act(() => result.current.handleConnectionClose(new CloseEvent("close", { code: 1006 })))
    dispatchFromParent(updateMessage())
    act(() => result.current.handleConnectionClose(new CloseEvent("close", { code: 1000 })))
    expect(result.current.status).toBe("terminal")
    expect(result.current.terminalCode).toBe("unexpected_error")
    act(() => vi.advanceTimersByTime(15_000))
    act(() => result.current.handleConnectionClose(new CloseEvent("close", { code: 1006 })))

    expect(result.current.status).toBe("terminal")
    expect(result.current.terminalCode).toBe("unexpected_error")
  })

  it("fails terminal when absolute expiry is expired", () => {
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({ absolute_expires_at: new Date(Date.now() - 1).toISOString() }))

    expect(result.current.status).toBe("terminal")
    expect(result.current.terminalCode).toBe("session_expired")
    expect(result.current.session).toBeNull()
  })

  it("fails terminal when absolute expiry precedes token expiry in a fresh hook", () => {
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({
      session_token_expires_at: new Date(Date.now() + 20 * 60_000).toISOString(),
      absolute_expires_at: new Date(Date.now() + 10 * 60_000).toISOString(),
    }))

    expect(result.current.status).toBe("terminal")
    expect(result.current.terminalCode).toBe("session_expired")
    expect(result.current.session).toBeNull()
  })

  it("accepts a terminal first response and ignores later updates", () => {
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent({ xagent: true, v: 1, type: "session_terminal", code: "reconnect_invalid" })
    dispatchFromParent(updateMessage())

    expect(result.current.status).toBe("terminal")
    expect(result.current.terminalCode).toBe("reconnect_invalid")
    expect(result.current.session).toBeNull()
  })

  it("clears active token, agent, and warning timer on terminal", () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-07-26T00:00:00.000Z"))
    const clearTimeout = vi.spyOn(globalThis, "clearTimeout")
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({
      session_token_expires_at: new Date(Date.now() + 10 * 60_000).toISOString(),
      absolute_expires_at: new Date(Date.now() + 11 * 60_000).toISOString(),
    }))

    expect(result.current.session?.agent.name).toBe("Support Agent")
    dispatchFromParent({ xagent: true, v: 1, type: "session_terminal", code: "session_expired" })
    act(() => vi.advanceTimersByTime(2 * 60_000))

    expect(result.current.status).toBe("terminal")
    expect(result.current.session).toBeNull()
    expect(result.current.agent).toBeNull()
    expect(result.current.isAbsoluteExpiryWarningVisible).toBe(false)
    expect(clearTimeout).toHaveBeenCalled()
  })

  it("allowlists the complete update and drops unknown fields", () => {
    const tokenExpiresAt = new Date(Date.now() + 15 * 60_000).toISOString()
    const absoluteExpiresAt = new Date(Date.now() + 30 * 60_000).toISOString()
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({
      session_token: "st_exact",
      session_token_expires_at: tokenExpiresAt,
      absolute_expires_at: absoluteExpiresAt,
      ignored_root: "discard",
      agent: {
        id: 42,
        name: "Support Agent",
        description: "Helps with schedules",
        logo_url: "https://cdn.example/logo.png",
        suggested_prompts: ["Show my schedule"],
        ignored_agent: "discard",
      },
    }))

    expect(result.current.session).toEqual({
      token: "st_exact",
      tokenExpiresAt,
      absoluteExpiresAt,
      generation: 1,
      agent: {
        id: 42,
        name: "Support Agent",
        description: "Helps with schedules",
        logoUrl: "https://cdn.example/logo.png",
        suggestedPrompts: ["Show my schedule"],
      },
    })
  })

  it("accepts a token with exactly sixty seconds remaining", () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-07-26T00:00:00.000Z"))
    const postMessage = vi.spyOn(window, "postMessage")
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({
      session_token_expires_at: new Date(Date.now() + 60_000).toISOString(),
      absolute_expires_at: new Date(Date.now() + 20 * 60_000).toISOString(),
    }))

    expect(result.current.status).toBe("active")
    expect(result.current.session?.token).toBe("st_session_token")
    expect(postMessage.mock.calls.filter((call) => call[0]?.type === "reconnect_request")).toHaveLength(0)
  })

  it("shows the expiry warning at the ten-minute boundary", () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-07-26T00:00:00.000Z"))
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({
      session_token_expires_at: new Date(Date.now() + 10 * 60_000).toISOString(),
      absolute_expires_at: new Date(Date.now() + 11 * 60_000).toISOString(),
    }))

    expect(result.current.isAbsoluteExpiryWarningVisible).toBe(false)
    act(() => vi.advanceTimersByTime(60_000))
    expect(result.current.isAbsoluteExpiryWarningVisible).toBe(true)
  })

  it("removes its listener and expiry timer on unmount", () => {
    vi.useFakeTimers()
    const removeEventListener = vi.spyOn(window, "removeEventListener")
    const clearTimeout = vi.spyOn(globalThis, "clearTimeout")
    const { unmount } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({
      session_token_expires_at: new Date(Date.now() + 10 * 60_000).toISOString(),
      absolute_expires_at: new Date(Date.now() + 11 * 60_000).toISOString(),
    }))

    unmount()

    expect(removeEventListener).toHaveBeenCalledWith("message", expect.any(Function))
    expect(clearTimeout).toHaveBeenCalled()
  })

  it("derives the session endpoint from the iframe origin", () => {
    expect(buildWidgetSessionWebSocketUrl("https://chat.example")).toBe(
      "wss://chat.example/v1/external/chat/sessions/ws",
    )
    expect(buildWidgetSessionWebSocketUrl("http://localhost:3000")).toBe(
      "ws://localhost:3000/v1/external/chat/sessions/ws",
    )
  })
})

describe("useWidgetSession capabilities", () => {
  const validCapabilities = () => ({
    file_uploads: {
      allowed_extensions: [".pdf", ".png"],
      max_file_bytes: 4096,
      max_files_per_message: 3,
    },
  })
  const withUploads = (overrides: Record<string, unknown>) => ({
    file_uploads: { ...validCapabilities().file_uploads, ...overrides },
  })

  it("parses a valid file_uploads capability", () => {
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({ capabilities: validCapabilities() }))

    expect(result.current.status).toBe("active")
    expect(result.current.session?.capabilities).toEqual({
      fileUploads: {
        allowedExtensions: [".pdf", ".png"],
        maxFileBytes: 4096,
        maxFilesPerMessage: 3,
      },
    })
  })

  it("treats a missing capabilities field as no capabilities", () => {
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage())

    expect(result.current.status).toBe("active")
    expect(result.current.session?.capabilities).toBeUndefined()
    expect(result.current.session).not.toHaveProperty("capabilities")
  })

  it.each([
    ["uppercase extension", withUploads({ allowed_extensions: [".PDF"] })],
    ["extension without a dot", withUploads({ allowed_extensions: ["pdf"] })],
    ["a bare dot", withUploads({ allowed_extensions: ["."] })],
    ["non-string extension", withUploads({ allowed_extensions: [1] })],
    ["extensions not an array", withUploads({ allowed_extensions: ".pdf" })],
    ["empty extensions", withUploads({ allowed_extensions: [] })],
    ["zero max_file_bytes", withUploads({ max_file_bytes: 0 })],
    ["negative max_file_bytes", withUploads({ max_file_bytes: -1 })],
    ["fractional max_file_bytes", withUploads({ max_file_bytes: 1.5 })],
    ["string max_file_bytes", withUploads({ max_file_bytes: "4096" })],
    ["zero max_files_per_message", withUploads({ max_files_per_message: 0 })],
    ["negative max_files_per_message", withUploads({ max_files_per_message: -2 })],
    ["fractional max_files_per_message", withUploads({ max_files_per_message: 2.5 })],
    ["string max_files_per_message", withUploads({ max_files_per_message: "3" })],
    ["file_uploads not an object", { file_uploads: "yes" }],
    ["no file_uploads key", { other: {} }],
    ["capabilities not an object", "file_uploads"],
    ["capabilities an array", [validCapabilities()]],
    ["capabilities null", null],
  ])("treats %s as no capabilities without failing the session", (_label, capabilities) => {
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({ capabilities }))

    expect(result.current.status).toBe("active")
    expect(result.current.session?.capabilities).toBeUndefined()
  })

  it("ignores unknown capability keys", () => {
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({
      capabilities: { ...validCapabilities(), voice: { enabled: true } },
    }))

    expect(result.current.status).toBe("active")
    expect(result.current.session?.capabilities).toEqual({
      fileUploads: {
        allowedExtensions: [".pdf", ".png"],
        maxFileBytes: 4096,
        maxFilesPerMessage: 3,
      },
    })
  })

  it("replaces the previous capabilities on every session_update", () => {
    const { result } = renderHook(() => useWidgetSession())
    dispatchFromParent(updateMessage({ session_delivery_id: "delivery-a", capabilities: validCapabilities() }))
    expect(result.current.session?.capabilities).toBeDefined()

    dispatchFromParent(updateMessage({ session_delivery_id: "delivery-b" }))

    expect(result.current.status).toBe("active")
    expect(result.current.session?.capabilities).toBeUndefined()
  })
})

describe("useWidgetSession upload coordination", () => {
  // Hooks rendered by earlier tests stay mounted and would answer the parent
  // messages dispatched here, so unmount them before and after each test.
  beforeEach(() => cleanup())
  afterEach(() => cleanup())

  const reconnectRequests = (postMessage: { mock: { calls: unknown[][] } }) =>
    postMessage.mock.calls.filter(([message]) => (message as { type?: string } | null)?.type === "reconnect_request")

  const expiringIn = (ms: number) => new Date(Date.now() + ms).toISOString()

  const settled = async <T,>(promise: Promise<T>) =>
    promise.then((value) => ({ value, error: null as unknown }), (error: unknown) => ({ value: null, error }))

  describe("ensureFreshSession", () => {
    it("returns the current session without asking the parent while the token is fresh", async () => {
      const postMessage = vi.spyOn(window, "postMessage")
      const { result } = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage())

      const session = await result.current.ensureFreshSession()

      expect(session.token).toBe("st_session_token")
      expect(reconnectRequests(postMessage)).toHaveLength(0)
    })

    it("sends one reconnect_request and resolves with the refreshed session", async () => {
      vi.useFakeTimers()
      const postMessage = vi.spyOn(window, "postMessage")
      const { result } = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage({
        session_token_expires_at: expiringIn(90_000),
        absolute_expires_at: expiringIn(30 * 60_000),
      }))
      expect(result.current.status).toBe("active")

      await act(async () => { await vi.advanceTimersByTimeAsync(40_000) })
      let waiting!: Promise<WidgetSession>
      act(() => { waiting = result.current.ensureFreshSession() })

      expect(reconnectRequests(postMessage)).toEqual([[
        { xagent: true, v: 1, type: "reconnect_request", reason: "token_expired" },
        PARENT_ORIGIN,
      ]])
      expect(result.current.status).toBe("refreshing")
      dispatchFromParent(updateMessage({
        session_delivery_id: "delivery-b",
        session_token: "st_renewed",
        session_token_expires_at: expiringIn(15 * 60_000),
        absolute_expires_at: expiringIn(30 * 60_000),
      }))

      await expect(waiting).resolves.toMatchObject({ token: "st_renewed", generation: 2 })
    })

    it("shares one wait between concurrent callers", async () => {
      vi.useFakeTimers()
      const postMessage = vi.spyOn(window, "postMessage")
      const { result } = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage({
        session_token_expires_at: expiringIn(90_000),
        absolute_expires_at: expiringIn(30 * 60_000),
      }))
      await act(async () => { await vi.advanceTimersByTimeAsync(40_000) })

      const first = result.current.ensureFreshSession()
      const second = result.current.ensureFreshSession()
      dispatchFromParent(updateMessage({ session_token: "st_renewed" }))

      expect(reconnectRequests(postMessage)).toHaveLength(1)
      await expect(first).resolves.toMatchObject({ token: "st_renewed" })
      await expect(second).resolves.toMatchObject({ token: "st_renewed" })
    })

    it("rejects when the session ends while waiting", async () => {
      vi.useFakeTimers()
      const { result } = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage({
        session_token_expires_at: expiringIn(90_000),
        absolute_expires_at: expiringIn(30 * 60_000),
      }))
      await act(async () => { await vi.advanceTimersByTimeAsync(40_000) })
      const waiting = settled(result.current.ensureFreshSession())

      dispatchFromParent({ xagent: true, v: 1, type: "session_terminal", code: "session_expired" })

      expect((await waiting).error).toMatchObject({ errorCode: "upload_failed" })
    })

    it("rejects when the session degrades while waiting and when asked while degraded", async () => {
      vi.useFakeTimers()
      const { result } = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage({
        session_token_expires_at: expiringIn(90_000),
        absolute_expires_at: expiringIn(30 * 60_000),
      }))
      await act(async () => { await vi.advanceTimersByTimeAsync(40_000) })
      const waiting = settled(result.current.ensureFreshSession())

      dispatchFromParent({ xagent: true, v: 1, type: "session_degraded", code: "network_unavailable" })

      expect((await waiting).error).toMatchObject({ errorCode: "upload_failed" })
      expect((await settled(result.current.ensureFreshSession())).error).toMatchObject({ errorCode: "upload_failed" })
      expect((await settled(result.current.refreshSession("st_session_token"))).error).toMatchObject({ errorCode: "upload_failed" })
    })

    it("rejects when the parent never answers within the resend deadline", async () => {
      vi.useFakeTimers()
      const { result } = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage({
        session_token_expires_at: expiringIn(90_000),
        absolute_expires_at: expiringIn(30 * 60_000),
      }))
      await act(async () => { await vi.advanceTimersByTimeAsync(40_000) })
      const waiting = settled(result.current.ensureFreshSession())

      await act(async () => { await vi.advanceTimersByTimeAsync(89_000) })
      let done = false
      void waiting.then(() => { done = true })
      await Promise.resolve()
      expect(done).toBe(false)
      await act(async () => { await vi.advanceTimersByTimeAsync(2_000) })

      expect((await waiting).error).toMatchObject({ errorCode: "upload_failed" })
    })

    it("rejects after the hook unmounts", async () => {
      const { result, unmount } = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage())
      const { ensureFreshSession } = result.current
      unmount()

      expect((await settled(ensureFreshSession())).error).toMatchObject({ errorCode: "upload_failed" })
    })
  })

  describe("refreshSession", () => {
    afterEach(() => {
      vi.unstubAllGlobals()
    })

    it("requests a refresh when the current token is the rejected one, even while it looks fresh", async () => {
      const postMessage = vi.spyOn(window, "postMessage")
      const { result } = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage())

      const waiting = result.current.refreshSession("st_session_token")
      expect(reconnectRequests(postMessage)).toHaveLength(1)
      dispatchFromParent(updateMessage({ session_token: "st_after_401" }))

      await expect(waiting).resolves.toMatchObject({ token: "st_after_401" })
    })

    it("sends two reconnect_requests in total when a 401 follows a stale-token refresh", async () => {
      vi.useFakeTimers()
      const postMessage = vi.spyOn(window, "postMessage")
      const { result } = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage({
        session_token_expires_at: expiringIn(90_000),
        absolute_expires_at: expiringIn(30 * 60_000),
      }))
      await act(async () => { await vi.advanceTimersByTimeAsync(40_000) })

      const fresh = result.current.ensureFreshSession()
      dispatchFromParent(updateMessage({ session_token: "st_renewed" }))
      await fresh
      const recovered = result.current.refreshSession("st_renewed")
      dispatchFromParent(updateMessage({ session_token: "st_renewed_again" }))

      await expect(recovered).resolves.toMatchObject({ token: "st_renewed_again" })
      expect(reconnectRequests(postMessage)).toHaveLength(2)
    })

    it("joins a refresh that is already under way instead of sending another request", async () => {
      vi.useFakeTimers()
      const postMessage = vi.spyOn(window, "postMessage")
      const { result } = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage({
        session_token_expires_at: expiringIn(90_000),
        absolute_expires_at: expiringIn(30 * 60_000),
      }))
      await act(async () => { await vi.advanceTimersByTimeAsync(40_000) })

      const fresh = result.current.ensureFreshSession()
      const recovered = result.current.refreshSession("st_session_token")
      dispatchFromParent(updateMessage({ session_token: "st_renewed" }))

      await expect(fresh).resolves.toMatchObject({ token: "st_renewed" })
      await expect(recovered).resolves.toMatchObject({ token: "st_renewed" })
      expect(reconnectRequests(postMessage)).toHaveLength(1)
    })

    it("a caller that joins a pending refresh is rejected at the refresh's deadline, not a full deadline later", async () => {
      vi.useFakeTimers()
      const { result } = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage())

      const first = settled(result.current.refreshSession("st_session_token"))
      await act(async () => { await vi.advanceTimersByTimeAsync(80_000) })
      const joiner = settled(result.current.refreshSession("st_session_token"))
      await act(async () => { await vi.advanceTimersByTimeAsync(11_000) })

      expect((await joiner).error).toMatchObject({ errorCode: "upload_failed" })
      expect((await first).error).toMatchObject({ errorCode: "upload_failed" })
    })

    it("resolves with the current session and sends no reconnect_request when the rejected token was already replaced", async () => {
      const postMessage = vi.spyOn(window, "postMessage")
      const { result } = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage())

      const first = result.current.refreshSession("st_session_token")
      dispatchFromParent(updateMessage({ session_delivery_id: "delivery-b", session_token: "st_renewed" }))
      await first
      const stale = result.current.refreshSession("st_session_token")

      await expect(stale).resolves.toMatchObject({ token: "st_renewed" })
      expect(reconnectRequests(postMessage)).toHaveLength(1)
    })

    it("two uploads whose 401s for the same token arrive before and after the shared refresh cause one reconnect_request", async () => {
      const postMessage = vi.spyOn(window, "postMessage")
      const { result } = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage())

      const fileId = "0b1e5c9e-3f1a-4a63-9b0e-6c1d2f3a4b5c"
      const unauthorized = () => new Response(
        JSON.stringify({ error: { code: "invalid_api_key", message: "no" } }),
        { status: 401, headers: { "Content-Type": "application/json" } },
      )
      const uploaded = () => new Response(
        JSON.stringify({ file_id: fileId }),
        { status: 201, headers: { "Content-Type": "application/json" } },
      )
      let releaseSecond!: () => void
      const gate = new Promise<void>((resolve) => { releaseSecond = resolve })
      let rejections = 0
      const authorizations: string[] = []
      const fetchMock = vi.fn(async (_url: string, init: { headers: { Authorization: string } }) => {
        const authorization = init.headers.Authorization
        authorizations.push(authorization)
        if (authorization !== "Bearer st_session_token") return uploaded()
        rejections += 1
        if (rejections === 2) await gate
        return unauthorized()
      })
      vi.stubGlobal("fetch", fetchMock)

      const upload = () => uploadWidgetSessionFile({
        origin: "https://chat.example.test",
        getToken: async () => (await result.current.ensureFreshSession()).token,
        refreshToken: async (rejected) => (await result.current.refreshSession(rejected)).token,
        file: new File(["hello world"], "report.pdf", { type: "application/pdf" }),
      })
      const first = upload()
      const second = upload()

      await vi.waitFor(() => expect(reconnectRequests(postMessage)).toHaveLength(1))
      dispatchFromParent(updateMessage({ session_delivery_id: "delivery-b", session_token: "st_renewed" }))
      await expect(first).resolves.toMatchObject({ fileId })
      releaseSecond()
      await expect(second).resolves.toMatchObject({ fileId })

      expect(reconnectRequests(postMessage)).toHaveLength(1)
      expect(authorizations).toEqual([
        "Bearer st_session_token",
        "Bearer st_session_token",
        "Bearer st_renewed",
        "Bearer st_renewed",
      ])
    })
  })

  describe("waitForOpenConnection", () => {
    it("resolves immediately when the current generation is already open", async () => {
      const { result } = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage())
      act(() => result.current.handleConnectionOpen("widget-session:1"))

      await expect(result.current.waitForOpenConnection()).resolves.toBeUndefined()
    })

    it("waits for the new generation and ignores an open from an older one", async () => {
      const { result } = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage({ session_delivery_id: "delivery-a" }))
      act(() => result.current.handleConnectionOpen("widget-session:1"))
      dispatchFromParent(updateMessage({ session_delivery_id: "delivery-b" }))

      const waiting = result.current.waitForOpenConnection()
      let done = false
      void waiting.then(() => { done = true })
      act(() => result.current.handleConnectionOpen("widget-session:1"))
      await Promise.resolve()
      expect(done).toBe(false)

      act(() => result.current.handleConnectionOpen("widget-session:2"))
      await expect(waiting).resolves.toBeUndefined()
    })

    it("resolves for a session update that carries no delivery id", async () => {
      const { result } = renderHook(() => useWidgetSession())
      const legacyUpdate: Record<string, unknown> = updateMessage()
      delete legacyUpdate.session_delivery_id
      dispatchFromParent(legacyUpdate)

      const waiting = result.current.waitForOpenConnection()
      act(() => result.current.handleConnectionOpen("widget-session:1"))

      await expect(waiting).resolves.toBeUndefined()
    })

    it("keeps waiting through a refresh and resolves on the next generation's open", async () => {
      const { result } = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage())
      act(() => result.current.handleConnectionOpen("widget-session:1"))
      const refreshed = result.current.refreshSession("st_session_token")
      const waiting = result.current.waitForOpenConnection()
      dispatchFromParent(updateMessage({ session_delivery_id: "delivery-b" }))
      await refreshed

      act(() => result.current.handleConnectionOpen("widget-session:2"))

      await expect(waiting).resolves.toBeUndefined()
    })

    it("rejects with connection_not_ready when the session ends or degrades", async () => {
      const { result } = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage())
      const waitingForTerminal = settled(result.current.waitForOpenConnection())
      dispatchFromParent({ xagent: true, v: 1, type: "session_terminal", code: "session_expired" })
      expect((await waitingForTerminal).error).toMatchObject({ errorCode: "connection_not_ready" })
      expect((await settled(result.current.waitForOpenConnection())).error)
        .toMatchObject({ errorCode: "connection_not_ready" })

      const second = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage())
      const waitingForDegraded = settled(second.result.current.waitForOpenConnection())
      dispatchFromParent({ xagent: true, v: 1, type: "session_degraded", code: "rate_limited" })
      expect((await waitingForDegraded).error).toMatchObject({ errorCode: "connection_not_ready" })
    })

    it("rejects with connection_not_ready when no socket opens within the deadline", async () => {
      vi.useFakeTimers()
      const { result } = renderHook(() => useWidgetSession())
      dispatchFromParent(updateMessage())
      const waiting = settled(result.current.waitForOpenConnection())

      await act(async () => { await vi.advanceTimersByTimeAsync(91_000) })

      expect((await waiting).error).toMatchObject({ errorCode: "connection_not_ready" })
    })
  })
})

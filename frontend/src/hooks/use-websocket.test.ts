import { useLayoutEffect } from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  type WebSocketConnection,
  useWebSocket,
} from "./use-websocket"
import { refreshStoredAccessToken } from "@/lib/api-wrapper"
import { AUTH_CACHE_KEY, readAuthCache, readAuthSessionSnapshot, type AuthSessionSnapshot } from "@/lib/auth-cache"

const authState = vi.hoisted(() => ({
  user: { id: "user-1" } as { id: string } | null,
  token: "token" as string | null,
  refreshToken: "refresh-token" as string | null,
  session: {
    sessionId: "test-lineage",
    credentialRevision: 0,
    profileRevision: 0,
    userId: "user-1",
    accessToken: "token",
    refreshToken: "refresh-token",
    profileFingerprint: '[null,null,null]',
  } as AuthSessionSnapshot,
  refreshAccessToken: vi.fn<(expectedSession?: AuthSessionSnapshot) => Promise<boolean>>(),
}))
vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => authState,
}))

class MockWebSocket {
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3
  static instances: MockWebSocket[] = []
  static constructorError: Error | null = null

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
    if (MockWebSocket.constructorError) throw MockWebSocket.constructorError
    MockWebSocket.instances.push(this)
  }

  open() {
    this.readyState = MockWebSocket.OPEN
    this.onopen?.()
  }

  receive(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent)
  }

  triggerError() {
    this.onerror?.(new Event("error"))
  }

  triggerClose(code = 1006, reason = "network lost") {
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.({ code, reason } as CloseEvent)
  }
}

const sessionConnections = new Map<string, WebSocketConnection>()

const sessionConnection = (
  overrides: Partial<WebSocketConnection> = {},
): WebSocketConnection => {
  const key = JSON.stringify(overrides)
  const existing = sessionConnections.get(key)
  if (existing) return existing
  const connection: WebSocketConnection = {
    identity: "widget-session:1",
    url: "wss://embed.example/v1/external/chat/sessions/ws",
    protocols: ["xagent-session-v1", "xagent-session-token.st_secret"],
    expectedProtocol: "xagent-session-v1",
    chatTaskIdMode: "omit",
    credentialOwner: { kind: "external" },
    ...overrides,
  }
  sessionConnections.set(key, connection)
  return connection
}

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, reject, resolve }
}

function writeAuthCache(
  user: { id: string; username: string; email?: string | null; is_admin?: boolean } | null,
  token: string | null,
  refreshToken: string | null = null,
  expiresIn?: number,
  refreshExpiresIn?: number,
) {
  if (!user || !token) {
    localStorage.removeItem(AUTH_CACHE_KEY)
    return
  }
  const now = Date.now()
  localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify({
    schemaVersion: 2, sessionId: `test-${Math.random()}`, credentialRevision: 0, profileRevision: 0,
    user, token, refreshToken, timestamp: now,
    expiresAt: expiresIn ? now + expiresIn * 1000 : undefined,
    refreshExpiresAt: refreshExpiresIn ? now + refreshExpiresIn * 1000 : undefined,
  }))
}

describe("useWebSocket message delivery", () => {
  beforeEach(() => {
    MockWebSocket.instances = []
    MockWebSocket.constructorError = null
    sessionConnections.clear()
    localStorage.clear()
    authState.user = { id: "user-1" }
    authState.token = "token"
    authState.refreshToken = "refresh-token"
    authState.session = {
      sessionId: "test-lineage",
      credentialRevision: 0,
      profileRevision: 0,
      userId: "user-1",
      accessToken: "token",
      refreshToken: "refresh-token",
      profileFingerprint: '[null,null,null]',
    }
    authState.refreshAccessToken.mockReset()
    vi.stubGlobal("WebSocket", MockWebSocket)
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it("rejects without clearing the caller when the socket is not open", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      autoConnect: false,
    }))

    await expect(result.current.sendChatMessage("keep this draft")).rejects.toMatchObject({
      message: "Message not sent: the connection is not ready.",
      disposition: "not_sent",
    })
  })

  it("returns not_sent when no current open socket owns a raw protocol write", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      autoConnect: false,
    }))

    expect(result.current.sendMessage({ type: "new_conversation" })).toBe("not_sent")
  })

  it("returns not_sent when a socket closes between the open check and raw write", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())
    socket.send.mockImplementationOnce(() => {
      socket.readyState = MockWebSocket.CLOSED
      throw new Error("socket closed during write")
    })

    expect(result.current.sendMessage({ type: "new_conversation" })).toBe("not_sent")
  })

  it("resolves only after the server accepts the durable message", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    let delivery!: Promise<{ client_message_id: string; turn_id: string }>
    act(() => {
      delivery = result.current.sendChatMessage("durable guidance")
    })
    expect(socket.send).toHaveBeenCalledOnce()
    const sent = JSON.parse(socket.send.mock.calls[0][0])
    expect(sent.client_message_id).toBeTruthy()

    let settled = false
    void delivery.finally(() => {
      settled = true
    })
    await Promise.resolve()
    expect(settled).toBe(false)

    act(() => {
      socket.receive({
        type: "message_accepted",
        client_message_id: sent.client_message_id,
        turn_id: sent.client_message_id,
      })
    })

    await expect(delivery).resolves.toEqual({
      client_message_id: sent.client_message_id,
      turn_id: sent.client_message_id,
    })
  })

  it("serializes the interaction request id with the caller's delivery id", async () => {
    const { result } = renderHook(() => useWebSocket({
      connection: sessionConnection(),
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    socket.protocol = "xagent-session-v1"
    act(() => socket.open())

    let delivery!: Promise<{ client_message_id: string; turn_id: string }>
    act(() => {
      delivery = result.current.sendChatMessage(
        "City: Sydney",
        undefined,
        true,
        "answer-q1",
        "inputreq_0011223344556677889900aabbccddee",
      )
    })

    expect(JSON.parse(socket.send.mock.calls[0][0])).toMatchObject({
      type: "chat",
      message: "City: Sydney",
      client_message_id: "answer-q1",
      request_id: "inputreq_0011223344556677889900aabbccddee",
    })
    act(() => socket.receive({
      type: "message_accepted",
      client_message_id: "answer-q1",
      turn_id: "answer-q1",
    }))
    await expect(delivery).resolves.toEqual({
      client_message_id: "answer-q1",
      turn_id: "answer-q1",
    })
  })

  it("assigns idempotency keys to pause and resume commands", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    act(() => {
      result.current.pauseTask()
      result.current.resumeTask()
    })

    const pause = JSON.parse(socket.send.mock.calls[0][0])
    const resume = JSON.parse(socket.send.mock.calls[1][0])
    expect(pause.command_id).toBeTruthy()
    expect(resume.command_id).toBeTruthy()
    expect(resume.command_id).not.toBe(pause.command_id)
  })

  it("allows an unacknowledged draft to retry with the same id", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      legacyErrorProse: "trusted",
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    const first = result.current.sendChatMessage(
      "retry me",
      undefined,
      false,
      "stable-turn-1",
    )
    act(() => {
      socket.receive({
        type: "message_rejected",
        client_message_id: "stable-turn-1",
        message: "temporary failure",
        rejection_outcome: "not_accepted",
      })
    })
    await expect(first).rejects.toMatchObject({
      message: "temporary failure",
      disposition: "rejected",
    })

    const retry = result.current.sendChatMessage(
      "retry me",
      undefined,
      false,
      "stable-turn-1",
    )
    expect(socket.send).toHaveBeenCalledTimes(2)
    act(() => {
      socket.receive({
        type: "message_accepted",
        client_message_id: "stable-turn-1",
        turn_id: "stable-turn-1",
      })
    })
    await expect(retry).resolves.toEqual({
      client_message_id: "stable-turn-1",
      turn_id: "stable-turn-1",
    })
  })

  it("marks a coded backend rejection as localizable and user facing", async () => {
    // The clarification form only shows a rejection reason that is marked
    // user facing; if this flag regresses, the visitor drops back to the
    // generic "Failed to send response" toast.
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    const delivery = result.current.sendChatMessage(
      "answer",
      undefined,
      false,
      "rejected-with-reason",
    )
    act(() => {
      socket.receive({
        type: "message_rejected",
        client_message_id: "rejected-with-reason",
        error_code: "guidance_in_progress",
        message: "A previous guidance message is still being applied. Please wait for it to finish.",
        rejection_outcome: "not_accepted",
      })
    })

    await expect(delivery).rejects.toMatchObject({
      message: "A previous guidance message is still being applied. Please wait for it to finish.",
      disposition: "rejected",
      userFacing: true,
      errorCode: "guidance_in_progress",
    })
  })

  it("preserves absent-code rejection prose for a trusted legacy transport", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      legacyErrorProse: "trusted",
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    const delivery = result.current.sendChatMessage(
      "answer",
      undefined,
      false,
      "trusted-legacy-rejection",
    )
    act(() => {
      socket.receive({
        type: "message_rejected",
        client_message_id: "trusted-legacy-rejection",
        message: "Legacy actionable authenticated rejection",
        rejection_outcome: "not_accepted",
      })
    })

    await expect(delivery).rejects.toMatchObject({
      message: "Legacy actionable authenticated rejection",
      disposition: "rejected",
      userFacing: true,
      errorCode: null,
    })
  })

  it("hides absent-code rejection prose for an untrusted legacy transport", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      legacyErrorProse: "untrusted",
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    const delivery = result.current.sendChatMessage(
      "answer",
      undefined,
      false,
      "untrusted-legacy-rejection",
    )
    act(() => {
      socket.receive({
        type: "message_rejected",
        client_message_id: "untrusted-legacy-rejection",
        message: "provider token=secret",
        rejection_outcome: "not_accepted",
      })
    })

    await expect(delivery).rejects.toMatchObject({
      message: "Message was rejected.",
      disposition: "rejected",
      userFacing: false,
      errorCode: null,
    })
  })

  it("keeps a rejection without a backend reason marked as not user facing", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    const delivery = result.current.sendChatMessage(
      "answer",
      undefined,
      false,
      "rejected-no-reason",
    )
    act(() => {
      socket.receive({
        type: "message_rejected",
        client_message_id: "rejected-no-reason",
        rejection_outcome: "not_accepted",
      })
    })

    await expect(delivery).rejects.toMatchObject({
      message: "Message was rejected.",
      disposition: "rejected",
      userFacing: false,
      errorCode: null,
    })
  })

  it("does not trust a rejection carrying an unknown error code", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      legacyErrorProse: "trusted",
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    const delivery = result.current.sendChatMessage(
      "answer",
      undefined,
      false,
      "rejected-unknown-code",
    )
    act(() => {
      socket.receive({
        type: "message_rejected",
        client_message_id: "rejected-unknown-code",
        error_code: "provider_secret",
        message: "token=secret",
        rejection_outcome: "not_accepted",
      })
    })

    await expect(delivery).rejects.toMatchObject({
      message: "Message was rejected.",
      disposition: "rejected",
      userFacing: false,
      errorCode: null,
    })
  })

  it.each([
    ["object", {}],
    ["number", 7],
    ["array", ["provider_secret"]],
    ["boolean", false],
    ["null", null],
  ])("does not trust a rejection carrying a malformed %s error code", async (_label, malformedCode) => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      legacyErrorProse: "trusted",
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    const delivery = result.current.sendChatMessage(
      "answer",
      undefined,
      false,
      "rejected-malformed-code",
    )
    act(() => {
      socket.receive({
        type: "message_rejected",
        client_message_id: "rejected-malformed-code",
        error_code: malformedCode,
        message: "provider token=secret",
        rejection_outcome: "not_accepted",
      })
    })

    await expect(delivery).rejects.toMatchObject({
      message: "Message was rejected.",
      disposition: "rejected",
      userFacing: false,
      errorCode: null,
    })
  })

  it("rejects concurrent reuse of a pending client message id without replacing its owner", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    const first = result.current.sendChatMessage(
      "first owner",
      undefined,
      false,
      "shared-pending-id",
    )
    const second = result.current.sendChatMessage(
      "second owner",
      undefined,
      false,
      "shared-pending-id",
    )
    await expect(second).rejects.toThrow("already pending")
    expect(socket.send).toHaveBeenCalledOnce()

    act(() => socket.receive({
      type: "message_accepted",
      client_message_id: "shared-pending-id",
    }))
    await expect(first).resolves.toEqual({
      client_message_id: "shared-pending-id",
      turn_id: "shared-pending-id",
    })
  })

  it("marks definitive rejections so the composer can use a fresh id", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      legacyErrorProse: "trusted",
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    const delivery = result.current.sendChatMessage(
      "retry with a new id",
      undefined,
      false,
      "failed-turn-1",
    )
    act(() => {
      socket.receive({
        type: "message_rejected",
        client_message_id: "failed-turn-1",
        message: "previous delivery failed",
        retry_with_new_id: true,
        rejection_outcome: "not_accepted",
      })
    })

    await expect(delivery).rejects.toMatchObject({
      message: "previous delivery failed",
      retryWithNewId: true,
      disposition: "rejected",
    })
  })

  it("allows the same text to be sent again after the first ack", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    const first = result.current.sendChatMessage("ok")
    const firstPayload = JSON.parse(socket.send.mock.calls[0][0])
    act(() => {
      socket.receive({
        type: "message_accepted",
        client_message_id: firstPayload.client_message_id,
      })
    })
    await first

    const second = result.current.sendChatMessage("ok")
    expect(socket.send).toHaveBeenCalledTimes(2)
    const secondPayload = JSON.parse(socket.send.mock.calls[1][0])
    expect(secondPayload.client_message_id).not.toBe(firstPayload.client_message_id)
    act(() => {
      socket.receive({
        type: "message_accepted",
        client_message_id: secondPayload.client_message_id,
      })
    })
    await second
  })

  it("rejects a pending delivery when the socket closes", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    const delivery = result.current.sendChatMessage("keep after disconnect")
    act(() => socket.triggerClose())

    await expect(delivery).rejects.toMatchObject({
      message: "Connection closed before the message was accepted.",
      disposition: "outcome_unknown",
    })
  })

  it("rejects an unacknowledged delivery after 30 seconds", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())
    vi.useFakeTimers()

    try {
      const delivery = result.current.sendChatMessage("timeout draft")
      const rejection = expect(delivery).rejects.toMatchObject({
        message: "Message delivery was not acknowledged. Your draft was kept.",
        disposition: "outcome_unknown",
      })
      await act(async () => {
        vi.advanceTimersByTime(30000)
      })
      await rejection
    } finally {
      vi.useRealTimers()
    }
  })

  it.each([
    [undefined, "outcome_unknown"],
    ["pending", "outcome_unknown"],
    ["unexpected", "outcome_unknown"],
  ])("fails closed for a rejected wire outcome of %s", async (rejectionOutcome, disposition) => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      legacyErrorProse: "trusted",
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    const delivery = result.current.sendChatMessage("ambiguous draft", undefined, false, "ambiguous-turn")
    act(() => socket.receive({
      type: "message_rejected",
      client_message_id: "ambiguous-turn",
      message: "ambiguous delivery",
      ...(rejectionOutcome === undefined ? {} : { rejection_outcome: rejectionOutcome }),
    }))

    await expect(delivery).rejects.toMatchObject({
      message: "ambiguous delivery",
      disposition,
    })
  })
})

describe("useWebSocket normalized connections", () => {
  beforeEach(() => {
    MockWebSocket.instances = []
    MockWebSocket.constructorError = null
    localStorage.clear()
    authState.user = { id: "user-1" }
    authState.token = "token"
    authState.refreshToken = "refresh-token"
    authState.session = {
      sessionId: "test-lineage",
      credentialRevision: 0,
      profileRevision: 0,
      userId: "user-1",
      accessToken: "token",
      refreshToken: "refresh-token",
      profileFingerprint: '[null,null,null]',
    }
    authState.refreshAccessToken.mockReset()
    vi.stubGlobal("WebSocket", MockWebSocket)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it("normalizes an undefined connection through the unchanged legacy URL", async () => {
    renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 7,
      connection: undefined,
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    expect(MockWebSocket.instances[0].url).toBe("ws://localhost/ws/chat/7?token=token")
    expect(MockWebSocket.instances[0].protocols).toBeUndefined()
  })

  it("keeps the task id in the legacy chat frame", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 7,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    const delivery = result.current.sendChatMessage("legacy", undefined, false, "legacy-turn")
    expect(JSON.parse(socket.send.mock.calls[0][0])).toEqual({
      type: "chat",
      message: "legacy",
      task_id: 7,
      client_message_id: "legacy-turn",
      context: { timezone: expect.any(String) },
    })
    act(() => socket.receive({
      type: "message_accepted",
      client_message_id: "legacy-turn",
    }))
    await delivery
  })

  it("reports the browser timezone so the agent clock can render local time", async () => {
    // Pinned, not read back from Intl: asserting against the same source the
    // code reads makes the test tautological on a UTC host.
    const resolvedOptions = vi
      .spyOn(Intl.DateTimeFormat.prototype, "resolvedOptions")
      .mockReturnValue({ timeZone: "America/New_York" } as Intl.ResolvedDateTimeFormatOptions)
    try {
      const { result } = renderHook(() => useWebSocket({
        url: "ws://localhost",
        taskId: 7,
      }))
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
      const socket = MockWebSocket.instances[0]
      act(() => socket.open())

      const delivery = result.current.sendChatMessage("what is tomorrow", undefined, false, "tz-turn")
      const sent = JSON.parse(socket.send.mock.calls[0][0])
      expect(sent.context).toEqual({ timezone: "America/New_York" })
      act(() => socket.receive({
        type: "message_accepted",
        client_message_id: "tz-turn",
      }))
      await delivery
    } finally {
      resolvedOptions.mockRestore()
    }
  })

  it("prefers the embedder-declared timezone from the iframe URL over the browser zone", async () => {
    const resolvedOptions = vi
      .spyOn(Intl.DateTimeFormat.prototype, "resolvedOptions")
      .mockReturnValue({ timeZone: "America/New_York" } as Intl.ResolvedDateTimeFormatOptions)
    window.history.replaceState({}, "", "/widget/chat/session?timezone=Australia%2FPerth")
    try {
      const { result } = renderHook(() => useWebSocket({
        url: "ws://localhost",
        taskId: 7,
      }))
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
      const socket = MockWebSocket.instances[0]
      act(() => socket.open())

      const delivery = result.current.sendChatMessage("hi", undefined, false, "declared-turn")
      const sent = JSON.parse(socket.send.mock.calls[0][0])
      expect(sent.context).toEqual({ timezone: "Australia/Perth" })
      act(() => socket.receive({
        type: "message_accepted",
        client_message_id: "declared-turn",
      }))
      await delivery
    } finally {
      resolvedOptions.mockRestore()
      window.history.replaceState({}, "", "/")
    }
  })

  it("falls back to the browser zone when the declared timezone is blank", async () => {
    const resolvedOptions = vi
      .spyOn(Intl.DateTimeFormat.prototype, "resolvedOptions")
      .mockReturnValue({ timeZone: "America/New_York" } as Intl.ResolvedDateTimeFormatOptions)
    window.history.replaceState({}, "", "/widget/chat/session?timezone=%20%20")
    try {
      const { result } = renderHook(() => useWebSocket({
        url: "ws://localhost",
        taskId: 7,
      }))
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
      const socket = MockWebSocket.instances[0]
      act(() => socket.open())

      const delivery = result.current.sendChatMessage("hi", undefined, false, "blank-declared")
      const sent = JSON.parse(socket.send.mock.calls[0][0])
      expect(sent.context).toEqual({ timezone: "America/New_York" })
      act(() => socket.receive({
        type: "message_accepted",
        client_message_id: "blank-declared",
      }))
      await delivery
    } finally {
      resolvedOptions.mockRestore()
      window.history.replaceState({}, "", "/")
    }
  })

  it("omits the context entirely when the browser cannot resolve a timezone", async () => {
    const resolvedOptions = vi
      .spyOn(Intl.DateTimeFormat.prototype, "resolvedOptions")
      .mockReturnValue({ timeZone: "" } as Intl.ResolvedDateTimeFormatOptions)
    try {
      const { result } = renderHook(() => useWebSocket({
        url: "ws://localhost",
        taskId: 7,
      }))
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
      const socket = MockWebSocket.instances[0]
      act(() => socket.open())

      const delivery = result.current.sendChatMessage("hi", undefined, false, "no-tz-turn")
      const sent = JSON.parse(socket.send.mock.calls[0][0])
      expect("context" in sent).toBe(false)
      act(() => socket.receive({
        type: "message_accepted",
        client_message_id: "no-tz-turn",
      }))
      await delivery
    } finally {
      resolvedOptions.mockRestore()
    }
  })

  it("keeps a same-id retry bound to the zone of its first attempt", async () => {
    const resolvedOptions = vi
      .spyOn(Intl.DateTimeFormat.prototype, "resolvedOptions")
      .mockReturnValue({ timeZone: "Australia/Melbourne" } as Intl.ResolvedDateTimeFormatOptions)
    try {
      const { result } = renderHook(() => useWebSocket({
        url: "ws://localhost",
        taskId: 7,
      }))
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
      const socket = MockWebSocket.instances[0]
      act(() => socket.open())

      const first = result.current.sendChatMessage("same", undefined, false, "retry-id")
      expect(JSON.parse(socket.send.mock.calls[0][0]).context)
        .toEqual({ timezone: "Australia/Melbourne" })
      // Unknown outcome: the command may already be executing server-side, so
      // the client retries under the same id.
      act(() => socket.receive({
        type: "message_rejected",
        client_message_id: "retry-id",
        message: "connection lost",
      }))
      await expect(first).rejects.toThrow()

      resolvedOptions.mockReturnValue(
        { timeZone: "America/New_York" } as Intl.ResolvedDateTimeFormatOptions,
      )

      const retry = result.current.sendChatMessage("same", undefined, true, "retry-id")
      expect(JSON.parse(socket.send.mock.calls[1][0]).context)
        .toEqual({ timezone: "Australia/Melbourne" })
      act(() => socket.receive({
        type: "message_accepted",
        client_message_id: "retry-id",
      }))
      await retry

      // A genuinely new id picks up the current zone.
      const fresh = result.current.sendChatMessage("next", undefined, false, "fresh-id")
      expect(JSON.parse(socket.send.mock.calls[2][0]).context)
        .toEqual({ timezone: "America/New_York" })
      act(() => socket.receive({
        type: "message_accepted",
        client_message_id: "fresh-id",
      }))
      await fresh
    } finally {
      resolvedOptions.mockRestore()
    }
  })

  it("frees the attempt zone after a not-sent failure so a same-id retry re-resolves", async () => {
    const resolvedOptions = vi
      .spyOn(Intl.DateTimeFormat.prototype, "resolvedOptions")
      .mockReturnValue({ timeZone: "Australia/Melbourne" } as Intl.ResolvedDateTimeFormatOptions)
    try {
      const { result } = renderHook(() => useWebSocket({
        url: "ws://localhost",
        taskId: 7,
      }))
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
      const socket = MockWebSocket.instances[0]
      act(() => socket.open())

      socket.send.mockImplementationOnce(() => {
        throw new Error("socket closed during write")
      })
      const failed = result.current.sendChatMessage("same", undefined, false, "reuse-id")
      await expect(failed).rejects.toThrow()

      // Nothing reached the server, so the same id is free to adopt the new zone.
      resolvedOptions.mockReturnValue(
        { timeZone: "America/New_York" } as Intl.ResolvedDateTimeFormatOptions,
      )
      const retry = result.current.sendChatMessage("same", undefined, false, "reuse-id")
      // calls[0] is the throwing send (still records its args); the retry is calls[1].
      expect(JSON.parse(socket.send.mock.calls[1][0]).context)
        .toEqual({ timezone: "America/New_York" })
      act(() => socket.receive({
        type: "message_accepted",
        client_message_id: "reuse-id",
      }))
      await retry
    } finally {
      resolvedOptions.mockRestore()
    }
  })

  it("keeps a same-id retry context-free when the first attempt had no zone", async () => {
    const resolvedOptions = vi
      .spyOn(Intl.DateTimeFormat.prototype, "resolvedOptions")
      .mockReturnValue({ timeZone: "" } as Intl.ResolvedDateTimeFormatOptions)
    try {
      const { result } = renderHook(() => useWebSocket({
        url: "ws://localhost",
        taskId: 7,
      }))
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
      const socket = MockWebSocket.instances[0]
      act(() => socket.open())

      const first = result.current.sendChatMessage("same", undefined, false, "omit-id")
      expect("context" in JSON.parse(socket.send.mock.calls[0][0])).toBe(false)
      act(() => socket.receive({
        type: "message_rejected",
        client_message_id: "omit-id",
        message: "connection lost",
      }))
      await expect(first).rejects.toThrow()

      resolvedOptions.mockReturnValue(
        { timeZone: "America/New_York" } as Intl.ResolvedDateTimeFormatOptions,
      )

      const retry = result.current.sendChatMessage("same", undefined, true, "omit-id")
      expect("context" in JSON.parse(socket.send.mock.calls[1][0])).toBe(false)
      act(() => socket.receive({
        type: "message_accepted",
        client_message_id: "omit-id",
      }))
      await retry
    } finally {
      resolvedOptions.mockRestore()
    }
  })

  it("frees the attempt zone when a late accepted ack follows a timeout", async () => {
    const resolvedOptions = vi
      .spyOn(Intl.DateTimeFormat.prototype, "resolvedOptions")
      .mockReturnValue({ timeZone: "Australia/Melbourne" } as Intl.ResolvedDateTimeFormatOptions)
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 7,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())
    vi.useFakeTimers()
    try {
      const first = result.current.sendChatMessage("same", undefined, false, "late-id")
      first.catch(() => {})
      expect(JSON.parse(socket.send.mock.calls[0][0]).context)
        .toEqual({ timezone: "Australia/Melbourne" })
      await act(async () => {
        vi.advanceTimersByTime(30000)
      })
      await expect(first).rejects.toMatchObject({ disposition: "outcome_unknown" })

      resolvedOptions.mockReturnValue(
        { timeZone: "America/New_York" } as Intl.ResolvedDateTimeFormatOptions,
      )
      // Late terminal ack for the same id after the timeout removed the pending
      // entry: the binding must be released.
      act(() => socket.receive({
        type: "message_accepted",
        client_message_id: "late-id",
      }))

      const retry = result.current.sendChatMessage("same", undefined, true, "late-id")
      expect(JSON.parse(socket.send.mock.calls[1][0]).context)
        .toEqual({ timezone: "America/New_York" })
      act(() => socket.receive({
        type: "message_accepted",
        client_message_id: "late-id",
      }))
      await retry
    } finally {
      vi.useRealTimers()
      resolvedOptions.mockRestore()
    }
  })

  it("frees the attempt zone when a late not_accepted reject follows a timeout", async () => {
    const resolvedOptions = vi
      .spyOn(Intl.DateTimeFormat.prototype, "resolvedOptions")
      .mockReturnValue({ timeZone: "Australia/Melbourne" } as Intl.ResolvedDateTimeFormatOptions)
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 7,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())
    vi.useFakeTimers()
    try {
      const first = result.current.sendChatMessage("same", undefined, false, "late-rej")
      first.catch(() => {})
      expect(JSON.parse(socket.send.mock.calls[0][0]).context)
        .toEqual({ timezone: "Australia/Melbourne" })
      await act(async () => {
        vi.advanceTimersByTime(30000)
      })
      await expect(first).rejects.toMatchObject({ disposition: "outcome_unknown" })

      resolvedOptions.mockReturnValue(
        { timeZone: "America/New_York" } as Intl.ResolvedDateTimeFormatOptions,
      )
      act(() => socket.receive({
        type: "message_rejected",
        client_message_id: "late-rej",
        message: "rejected outright",
        rejection_outcome: "not_accepted",
      }))

      const retry = result.current.sendChatMessage("same", undefined, true, "late-rej")
      expect(JSON.parse(socket.send.mock.calls[1][0]).context)
        .toEqual({ timezone: "America/New_York" })
      act(() => socket.receive({
        type: "message_accepted",
        client_message_id: "late-rej",
      }))
      await retry
    } finally {
      vi.useRealTimers()
      resolvedOptions.mockRestore()
    }
  })

  it("keeps the attempt zone after a timeout until a terminal ack arrives", async () => {
    const resolvedOptions = vi
      .spyOn(Intl.DateTimeFormat.prototype, "resolvedOptions")
      .mockReturnValue({ timeZone: "Australia/Melbourne" } as Intl.ResolvedDateTimeFormatOptions)
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 7,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())
    vi.useFakeTimers()
    try {
      const first = result.current.sendChatMessage("same", undefined, false, "keep-id")
      first.catch(() => {})
      expect(JSON.parse(socket.send.mock.calls[0][0]).context)
        .toEqual({ timezone: "Australia/Melbourne" })
      await act(async () => {
        vi.advanceTimersByTime(30000)
      })
      await expect(first).rejects.toMatchObject({ disposition: "outcome_unknown" })

      resolvedOptions.mockReturnValue(
        { timeZone: "America/New_York" } as Intl.ResolvedDateTimeFormatOptions,
      )
      // A late outcome_unknown reject must NOT release the binding: a same-id
      // retry is still permitted and must reuse the first attempt's zone.
      act(() => socket.receive({
        type: "message_rejected",
        client_message_id: "keep-id",
        message: "still ambiguous",
      }))

      const retry = result.current.sendChatMessage("same", undefined, true, "keep-id")
      expect(JSON.parse(socket.send.mock.calls[1][0]).context)
        .toEqual({ timezone: "Australia/Melbourne" })
      act(() => socket.receive({
        type: "message_accepted",
        client_message_id: "keep-id",
      }))
      await retry
    } finally {
      vi.useRealTimers()
      resolvedOptions.mockRestore()
    }
  })

  it("sends without a context when Intl itself throws", async () => {
    const resolvedOptions = vi
      .spyOn(Intl.DateTimeFormat.prototype, "resolvedOptions")
      .mockImplementation(() => {
        throw new RangeError("ICU unavailable")
      })
    try {
      const { result } = renderHook(() => useWebSocket({
        url: "ws://localhost",
        taskId: 7,
      }))
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
      const socket = MockWebSocket.instances[0]
      act(() => socket.open())

      const delivery = result.current.sendChatMessage("hi", undefined, false, "throw-turn")
      const sent = JSON.parse(socket.send.mock.calls[0][0])
      expect("context" in sent).toBe(false)
      expect(sent.message).toBe("hi")
      act(() => socket.receive({
        type: "message_accepted",
        client_message_id: "throw-turn",
      }))
      await delivery
    } finally {
      resolvedOptions.mockRestore()
    }
  })

  it("treats an explicit null connection as disabled even when legacy inputs exist", async () => {
    renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 7,
      connection: null,
    }))

    await Promise.resolve()
    expect(MockWebSocket.instances).toHaveLength(0)
  })

  it("constructs the exact Session URL and subprotocols and requires the server echo", async () => {
    const onConnect = vi.fn()
    const connection = sessionConnection()
    const { result } = renderHook(() => useWebSocket({ connection, onConnect }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    expect(socket.url).toBe("wss://embed.example/v1/external/chat/sessions/ws")
    expect(socket.url).not.toContain("st_secret")
    expect(socket.protocols).toEqual([
      "xagent-session-v1",
      "xagent-session-token.st_secret",
    ])

    socket.protocol = "xagent-session-v1"
    act(() => socket.open())
    expect(result.current.isConnected).toBe(true)
    expect(onConnect).toHaveBeenCalledOnce()
  })

  it("sanitizes a WebSocket constructor failure before logging or exposing it", async () => {
    const secret = "xagent-session-token.st_constructor_secret"
    MockWebSocket.constructorError = new Error(`constructor rejected ${secret}`)
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {})
    const onError = vi.fn()
    const onConnectionFailure = vi.fn()

    const { result } = renderHook(() => useWebSocket({
      connection: sessionConnection({
        protocols: ["xagent-session-v1", secret],
      }),
      onError,
      onConnectionFailure,
    }))

    await waitFor(() => expect(result.current.connectionError).not.toBeNull())
    expect(onError).toHaveBeenCalledOnce()
    expect(onConnectionFailure).toHaveBeenCalledWith({
      recoverable: false,
      error: result.current.connectionError,
    })
    const exposed = [
      result.current.connectionError?.message,
      ...onError.mock.calls.map(([error]) => (error as Error).message),
      ...consoleError.mock.calls.flat().map(String),
    ].join(" ")
    expect(exposed).not.toContain(secret)
    expect(result.current.connectionError?.message).toBe(
      "Failed to create WebSocket connection.",
    )
  })

  it("fails closed when the server omits the required Session subprotocol echo", async () => {
    const onConnect = vi.fn()
    const onError = vi.fn()
    const onConnectionFailure = vi.fn()
    const { result } = renderHook(() => useWebSocket({
      connection: sessionConnection(),
      onConnect,
      onError,
      onConnectionFailure,
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    expect(result.current.isConnected).toBe(false)
    expect(onConnect).not.toHaveBeenCalled()
    expect(onError).toHaveBeenCalledOnce()
    expect(onConnectionFailure).toHaveBeenCalledWith({
      recoverable: false,
      error: result.current.connectionError,
    })
    expect(socket.close).toHaveBeenCalled()
  })

  it.each(["legacy error", "typed owner"] as const)(
    "isolates a throwing %s callback while closing a protocol mismatch",
    async (throwingCallback) => {
      const secret = `callback-secret-${throwingCallback}`
      const order: string[] = []
      const consoleError = vi.spyOn(console, "error").mockImplementation(() => {})
      const onError = vi.fn(() => {
        order.push("legacy")
        if (throwingCallback === "legacy error") {
          throw new Error(secret)
        }
      })
      const onConnectionFailure = vi.fn(() => {
        order.push("typed")
        if (throwingCallback === "typed owner") {
          throw new Error(secret)
        }
      })
      renderHook(() => useWebSocket({
        connection: sessionConnection(),
        onError,
        onConnectionFailure,
      }))

      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
      const socket = MockWebSocket.instances[0]

      expect(() => act(() => socket.open())).not.toThrow()
      expect(order).toEqual(["typed", "legacy"])
      expect(onConnectionFailure).toHaveBeenCalledOnce()
      expect(onError).toHaveBeenCalledOnce()
      expect(socket.close).toHaveBeenCalledWith(
        1002,
        "WebSocket subprotocol mismatch",
      )
      expect(consoleError.mock.calls.flat().map(String).join(" "))
        .not.toContain(secret)
    },
  )

  it("classifies a physical socket error as a recoverable connection failure", async () => {
    const onError = vi.fn()
    const onConnectionFailure = vi.fn()
    const { result } = renderHook(() => useWebSocket({
      connection: sessionConnection(),
      onError,
      onConnectionFailure,
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    socket.protocol = "xagent-session-v1"
    act(() => socket.open())
    act(() => socket.triggerError())

    expect(result.current.isConnected).toBe(false)
    expect(onError).toHaveBeenCalledOnce()
    expect(onConnectionFailure).toHaveBeenCalledWith({
      recoverable: true,
      error: result.current.connectionError,
    })
  })

  it("sends and acknowledges taskless Session chat without a task id", async () => {
    const { result } = renderHook(() => useWebSocket({
      connection: sessionConnection(),
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    socket.protocol = "xagent-session-v1"
    act(() => socket.open())

    const delivery = result.current.sendChatMessage(
      "create lazily",
      undefined,
      false,
      "session-turn-1",
    )
    const sent = JSON.parse(socket.send.mock.calls[0][0])
    expect(sent).toEqual({
      type: "chat",
      message: "create lazily",
      context: { timezone: expect.any(String) },
      client_message_id: "session-turn-1",
    })

    act(() => socket.receive({
      type: "message_accepted",
      client_message_id: "session-turn-1",
      turn_id: "server-turn-1",
    }))
    await expect(delivery).resolves.toEqual({
      client_message_id: "session-turn-1",
      turn_id: "server-turn-1",
    })
  })

  it("offers an explicit unbound Session binding on the initial physical connection", async () => {
    renderHook(() => useWebSocket({
      connection: sessionConnection({
        taskBindingMode: "session-subprotocol",
      }),
    }))

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    expect(MockWebSocket.instances[0].protocols).toEqual([
      "xagent-session-v1",
      "xagent-session-token.st_secret",
      "xagent-session-binding-v1",
      "xagent-session-task.unbound",
    ])
  })

  it("offers the current bound task when establishing a replacement physical connection", async () => {
    const connection = sessionConnection({
      taskBindingMode: "session-subprotocol",
    })
    const hook = renderHook(
      ({ taskId }: { taskId: number | undefined }) => useWebSocket({
        connection,
        taskId,
      }),
      { initialProps: { taskId: undefined as number | undefined } },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    expect(MockWebSocket.instances[0].protocols).toContain(
      "xagent-session-task.unbound",
    )

    hook.rerender({ taskId: 42 })
    expect(MockWebSocket.instances).toHaveLength(1)
    act(() => hook.result.current.disconnect())
    act(() => hook.result.current.connect())

    expect(MockWebSocket.instances).toHaveLength(2)
    expect(MockWebSocket.instances[1].protocols).toEqual([
      "xagent-session-v1",
      "xagent-session-token.st_secret",
      "xagent-session-binding-v1",
      "xagent-session-task.42",
    ])
  })

  it.each([4001, 4003, 1011])(
    "runs the Session close delegate first and suppresses legacy handling for %s",
    async (code) => {
      vi.useFakeTimers()
      authState.refreshAccessToken.mockResolvedValue(true)
      const order: string[] = []
      const onError = vi.fn()
      const { result } = renderHook(() => useWebSocket({
        connection: sessionConnection(),
        onConnectionClose: () => {
          order.push("delegate")
          return "handled"
        },
        onDisconnect: () => {
          order.push("disconnect")
        },
        onError,
      }))
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })
      const socket = MockWebSocket.instances[0]
      socket.protocol = "xagent-session-v1"
      act(() => socket.open())
      const delivery = result.current.sendChatMessage("pending")
      const rejection = expect(delivery).rejects.toThrow("Connection closed")

      act(() => socket.triggerClose(code))
      await rejection
      await act(async () => {
        await vi.runAllTimersAsync()
      })

      expect(order).toEqual(["delegate"])
      expect(result.current.isConnected).toBe(false)
      expect(authState.refreshAccessToken).not.toHaveBeenCalled()
      expect(onError).not.toHaveBeenCalled()
      expect(MockWebSocket.instances).toHaveLength(1)
    },
  )

  it("retires a pre-open owner before a close delegate synchronously reconnects", async () => {
    let reconnect!: () => void
    const onConnectionClose = vi.fn(() => {
      reconnect()
      return "handled" as const
    })
    const { result } = renderHook(() => {
      const webSocket = useWebSocket({
        connection: sessionConnection(),
        onConnectionClose,
      })
      reconnect = webSocket.connect
      return webSocket
    })
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const oldSocket = MockWebSocket.instances[0]

    act(() => oldSocket.triggerClose(1011))

    expect(onConnectionClose).toHaveBeenCalledOnce()
    expect(MockWebSocket.instances).toHaveLength(2)
    expect(result.current.isConnected).toBe(false)
  })

  it("rejects old delivery and preparation before a close delegate replaces the owner", async () => {
    const upload = deferred<Array<{ file_id: string }>>()
    let reconnect!: () => void
    const hook = renderHook(() => {
      const webSocket = useWebSocket({
        url: "ws://localhost",
        taskId: 1,
        uploadFiles: vi.fn(() => upload.promise),
        onConnectionClose: () => {
          reconnect()
          return "handled" as const
        },
      })
      reconnect = webSocket.connect
      return webSocket
    })
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const oldSocket = MockWebSocket.instances[0]
    act(() => oldSocket.open())
    const pending = hook.result.current.sendChatMessage("pending", undefined, false, "old-delivery")
    const preparing = hook.result.current.sendChatMessage(
      "preparing",
      [new File(["data"], "data.txt")],
      false,
      "old-preparation",
    )

    act(() => oldSocket.triggerClose(4001))

    await expect(pending).rejects.toThrow("Connection closed")
    await expect(preparing).rejects.toThrow("Connection closed")
    expect(MockWebSocket.instances).toHaveLength(2)
    const replacement = MockWebSocket.instances[1]
    act(() => replacement.open())
    await act(async () => {
      upload.resolve([{ file_id: "late-upload" }])
      await Promise.resolve()
    })
    expect(replacement.readyState).toBe(MockWebSocket.OPEN)
  })

  it("fails closed and sanitizes when the Session close delegate throws", async () => {
    const secret = "xagent-session-token.st_close_delegate_secret"
    authState.refreshAccessToken.mockResolvedValue(true)
    const onError = vi.fn()
    const onConnectionFailure = vi.fn()
    const onDisconnect = vi.fn()
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {})
    const { result } = renderHook(() => useWebSocket({
      connection: sessionConnection(),
      onConnectionClose: () => {
        throw new Error(`close delegate leaked ${secret}`)
      },
      onDisconnect,
      onError,
      onConnectionFailure,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    socket.protocol = "xagent-session-v1"
    act(() => socket.open())
    vi.useFakeTimers()
    const delivery = result.current.sendChatMessage(
      "pending",
      undefined,
      false,
      "close-handler-pending",
    )
    const deliveryOutcome = delivery.then(
      () => "resolved",
      error => `rejected:${(error as Error).message}`,
    )

    let escaped: unknown
    try {
      act(() => socket.triggerClose(4001))
    } catch (error) {
      escaped = error
    }

    expect(escaped).toBeUndefined()
    expect(await deliveryOutcome).toContain("Connection closed")
    expect(result.current.isConnected).toBe(false)
    expect(result.current.connectionError).not.toBeNull()
    expect(onError).toHaveBeenCalledOnce()
    expect(onConnectionFailure).toHaveBeenCalledWith({
      recoverable: false,
      error: result.current.connectionError,
    })
    expect(onDisconnect).not.toHaveBeenCalled()
    expect(authState.refreshAccessToken).not.toHaveBeenCalled()
    await act(async () => {
      await vi.runAllTimersAsync()
    })
    expect(MockWebSocket.instances).toHaveLength(1)
    const exposed = [
      result.current.connectionError?.message,
      ...onError.mock.calls.map(([error]) => (error as Error).message),
      ...consoleError.mock.calls.flat().map(String),
    ].join(" ")
    expect(exposed).not.toContain(secret)
  })

  it("keeps legacy auth refresh, retry, pause, resume, and status behavior", async () => {
    authState.refreshAccessToken.mockResolvedValue(false)
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 9,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const first = MockWebSocket.instances[0]
    act(() => first.open())

    act(() => {
      result.current.pauseTask()
      result.current.resumeTask()
      result.current.requestStatus()
    })
    expect(first.send.mock.calls.map(([frame]) => JSON.parse(frame))).toEqual([
      expect.objectContaining({ type: "pause_task", task_id: 9 }),
      expect.objectContaining({ type: "resume_task", task_id: 9 }),
      { type: "status_request", task_id: 9 },
    ])

    act(() => first.triggerClose(4001))
    expect(authState.refreshAccessToken).toHaveBeenCalledOnce()
    expect(authState.refreshAccessToken).toHaveBeenCalledWith(expect.objectContaining({
      sessionId: "test-lineage", credentialRevision: 0, accessToken: "token", userId: "user-1",
    }))

    const retryHook = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 10,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
    const retrySocket = MockWebSocket.instances[1]
    act(() => retrySocket.open())
    vi.useFakeTimers()
    act(() => retrySocket.triggerClose(1011))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })
    expect(MockWebSocket.instances).toHaveLength(3)
    retryHook.unmount()
  })

  it.each([
    ["unmount", "resolve"],
    ["unmount", "reject"],
    ["disconnect", "resolve"],
    ["disconnect", "reject"],
    ["replacement", "resolve"],
    ["replacement", "reject"],
  ] as const)(
    "ignores late auth work after %s when refresh will %s",
    async (lifecycle, settlement) => {
      const refresh = deferred<boolean>()
      authState.refreshAccessToken.mockReturnValue(refresh.promise)
      const onError = vi.fn()
      const consoleError = vi.spyOn(console, "error").mockImplementation(() => {})
      const hook = renderHook(
        ({ taskId }) => useWebSocket({
          url: "ws://localhost",
          taskId,
          onError,
        }),
        { initialProps: { taskId: 1 } },
      )
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
      const oldSocket = MockWebSocket.instances[0]
      act(() => oldSocket.open())
      vi.useFakeTimers()

      act(() => oldSocket.triggerClose(4001))
      expect(authState.refreshAccessToken).toHaveBeenCalledOnce()
      expect(authState.refreshAccessToken).toHaveBeenCalledWith(expect.objectContaining({
        sessionId: "test-lineage", credentialRevision: 0, accessToken: "token", userId: "user-1",
      }))

      let expectedSocketCount = 1
      if (lifecycle === "unmount") {
        hook.unmount()
      } else if (lifecycle === "disconnect") {
        act(() => hook.result.current.disconnect())
      } else {
        hook.rerender({ taskId: 2 })
        expectedSocketCount = 2
        const replacementSocket = MockWebSocket.instances[1]
        act(() => replacementSocket.open())
      }

      await act(async () => {
        if (settlement === "resolve") {
          refresh.resolve(true)
        } else {
          refresh.reject(
            new Error("refresh failed xagent-session-token.st_refresh_secret"),
          )
        }
        await Promise.resolve()
      })
      await act(async () => {
        await vi.runAllTimersAsync()
      })

      expect(MockWebSocket.instances).toHaveLength(expectedSocketCount)
      expect(onError).not.toHaveBeenCalled()
      expect(consoleError.mock.calls.flat().join(" ")).not.toContain(
        "st_refresh_secret",
      )
      if (lifecycle !== "unmount") hook.unmount()
    },
  )

  it.each([
    ["an explicit legacy token", {
      url: "ws://localhost",
      taskId: 1,
      token: "token",
    }, "ws://localhost/ws/chat/1?token=token"],
    ["an explicit empty legacy token", {
      url: "ws://localhost",
      taskId: 1,
      token: "",
    }, "ws://localhost/ws/chat/1"],
    ["an explicit Session descriptor", {
      connection: sessionConnection(),
    }, "wss://embed.example/v1/external/chat/sessions/ws"],
  ] as const)(
    "never refreshes AuthContext credentials for %s",
    async (_name, options, expectedUrl) => {
      vi.useFakeTimers()
      authState.refreshAccessToken.mockResolvedValue(true)
      const onError = vi.fn()
      const hook = renderHook(() => useWebSocket({ ...options, onError }))
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })
      const socket = MockWebSocket.instances[0]
      expect(socket.url).toBe(expectedUrl)
      if ("connection" in options) socket.protocol = "xagent-session-v1"
      act(() => socket.open())

      act(() => socket.triggerClose(4001))
      await act(async () => {
        await vi.runAllTimersAsync()
      })

      expect(authState.refreshAccessToken).not.toHaveBeenCalled()
      expect(onError).toHaveBeenCalledOnce()
      expect((onError.mock.calls[0][0] as Error).message).toBe("Authentication failed")
      expect(MockWebSocket.instances).toHaveLength(1)
      hook.unmount()
    },
  )

  it("reconnects an auth-owned descriptor with the refreshed token only once", async () => {
    const refresh = deferred<boolean>()
    authState.token = "old-auth-token"
    authState.session = {
      ...authState.session,
      accessToken: "old-auth-token",
    }
    authState.refreshAccessToken.mockReturnValue(refresh.promise)
    const hook = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const oldSocket = MockWebSocket.instances[0]
    expect(oldSocket.url).toBe("ws://localhost/ws/chat/1?token=old-auth-token")
    act(() => oldSocket.open())
    vi.useFakeTimers()

    act(() => oldSocket.triggerClose(4001))
    expect(authState.refreshAccessToken).toHaveBeenCalledWith(expect.objectContaining({
      accessToken: "old-auth-token", userId: "user-1",
    }))
    authState.token = "new-auth-token"
    hook.rerender()
    expect(MockWebSocket.instances).toHaveLength(2)
    const refreshedSocket = MockWebSocket.instances[1]
    expect(refreshedSocket.url).toBe("ws://localhost/ws/chat/1?token=new-auth-token")
    act(() => refreshedSocket.open())

    await act(async () => {
      refresh.resolve(true)
      await Promise.resolve()
      await vi.advanceTimersByTimeAsync(1000)
    })

    expect(MockWebSocket.instances).toHaveLength(2)
    hook.unmount()
  })

  it("does not reuse a replacement user's token for an uncommitted old descriptor", async () => {
    const aliceUser = {
      id: "user-a",
      username: "alice",
      email: null,
      is_admin: false,
    }
    const bobUser = {
      id: "user-b",
      username: "bob",
      email: null,
      is_admin: false,
    }
    writeAuthCache(aliceUser, "user-a-token", "user-a-refresh", 120, 240)
    authState.session = readAuthSessionSnapshot()
    authState.user = { id: aliceUser.id }
    authState.token = "user-a-token"
    authState.refreshAccessToken.mockImplementation(
      async (expectedSession) => {
        if (!expectedSession) return false
        const result = await refreshStoredAccessToken(expectedSession)
        return result.accessToken !== null
      },
    )
    const onError = vi.fn()
    const hook = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      onError,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const oldSocket = MockWebSocket.instances[0]
    act(() => oldSocket.open())
    vi.useFakeTimers()

    writeAuthCache(bobUser, "user-b-token", "user-b-refresh", 120, 240)
    act(() => oldSocket.triggerClose(4001))
    await act(async () => {
      await Promise.resolve()
      await vi.advanceTimersByTimeAsync(1000)
    })

    expect(authState.refreshAccessToken).toHaveBeenCalledWith(expect.objectContaining({
      accessToken: "user-a-token", userId: "user-a",
    }))
    expect(readAuthCache()).toMatchObject({
      token: "user-b-token",
      user: { id: "user-b" },
    })
    expect(onError).toHaveBeenCalledOnce()
    expect(MockWebSocket.instances).toHaveLength(1)
    hook.unmount()
  })

  it("retires a closing owner before a same-descriptor replacement claims its id", async () => {
    const oldUpload = deferred<Array<{ file_id: string }>>()
    const uploadFiles = vi.fn(() => oldUpload.promise)
    const onDisconnect = vi.fn()
    const hook = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      uploadFiles,
      onDisconnect,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const oldSocket = MockWebSocket.instances[0]
    act(() => oldSocket.open())
    let oldSettled = false
    const oldOutcome = hook.result.current.sendChatMessage(
      "old upload",
      [new File(["old"], "old.txt")],
      false,
      "same-attempt-id",
    ).then(
      () => "resolved",
      error => `rejected:${(error as Error).message}`,
    ).finally(() => {
      oldSettled = true
    })

    oldSocket.readyState = MockWebSocket.CLOSING
    act(() => hook.result.current.connect())
    expect(MockWebSocket.instances).toHaveLength(2)
    const replacementSocket = MockWebSocket.instances[1]
    act(() => replacementSocket.open())
    await act(async () => {
      await Promise.resolve()
    })
    const oldSettledBeforeUploadCompleted = oldSettled
    const replacementDelivery = hook.result.current.sendChatMessage(
      "replacement",
      undefined,
      false,
      "same-attempt-id",
    )
    const replacementOutcome = replacementDelivery.then(
      ack => ack,
      error => `rejected:${(error as Error).message}`,
    )

    await act(async () => {
      oldUpload.resolve([{ file_id: "stale-upload" }])
      await Promise.resolve()
    })
    act(() => {
      oldSocket.triggerClose(1000, "late old close")
      replacementSocket.receive({
        type: "message_accepted",
        client_message_id: "same-attempt-id",
      })
    })

    expect(oldSettledBeforeUploadCompleted).toBe(true)
    expect(await oldOutcome).toContain("replaced")
    await expect(replacementOutcome).resolves.toMatchObject({
      client_message_id: "same-attempt-id",
    })
    expect(oldSocket.close).toHaveBeenCalledOnce()
    expect(onDisconnect).toHaveBeenCalledOnce()
    hook.unmount()
  })

  it("does not preserve a closing owner when its replacement constructor fails", async () => {
    const oldUpload = deferred<Array<{ file_id: string }>>()
    const uploadFiles = vi.fn(() => oldUpload.promise)
    const onDisconnect = vi.fn()
    const onError = vi.fn()
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {})
    const hook = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      uploadFiles,
      onDisconnect,
      onError,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const oldSocket = MockWebSocket.instances[0]
    act(() => oldSocket.open())
    let oldSettled = false
    const oldOutcome = hook.result.current.sendChatMessage(
      "old upload",
      [new File(["old"], "old.txt")],
      false,
      "constructor-failure-id",
    ).then(
      () => "resolved",
      error => `rejected:${(error as Error).message}`,
    ).finally(() => {
      oldSettled = true
    })

    oldSocket.readyState = MockWebSocket.CLOSING
    MockWebSocket.constructorError = new Error("constructor secret")
    act(() => hook.result.current.connect())
    await act(async () => {
      await Promise.resolve()
    })
    const oldSettledBeforeUploadCompleted = oldSettled
    MockWebSocket.constructorError = null
    act(() => hook.result.current.connect())
    expect(MockWebSocket.instances).toHaveLength(2)
    const replacementSocket = MockWebSocket.instances[1]
    act(() => replacementSocket.open())

    await act(async () => {
      oldUpload.resolve([{ file_id: "stale-upload" }])
      await Promise.resolve()
    })
    act(() => oldSocket.triggerClose(1000, "late old close"))

    expect(oldSettledBeforeUploadCompleted).toBe(true)
    expect(await oldOutcome).toContain("replaced")
    expect(oldSocket.close).toHaveBeenCalledOnce()
    expect(onDisconnect).toHaveBeenCalledOnce()
    expect(onError).toHaveBeenCalledOnce()
    expect(consoleError.mock.calls.flat().join(" ")).not.toContain("constructor secret")
    expect(hook.result.current.isConnected).toBe(true)
    hook.unmount()
  })

  it.each([
    ["failed", false],
    ["successful", true],
  ] as const)(
    "fences a stale %s auth refresh after a same-descriptor manual connect",
    async (_settlement, refreshSucceeded) => {
      const refresh = deferred<boolean>()
      authState.refreshAccessToken.mockReturnValue(refresh.promise)
      const onError = vi.fn()
      const hook = renderHook(() => useWebSocket({
        url: "ws://localhost",
        taskId: 1,
        onError,
      }))
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
      const oldSocket = MockWebSocket.instances[0]
      act(() => oldSocket.open())
      vi.useFakeTimers()

      act(() => oldSocket.triggerClose(4001))
      expect(authState.refreshAccessToken).toHaveBeenCalledOnce()
      expect(authState.refreshAccessToken).toHaveBeenCalledWith(expect.objectContaining({
        sessionId: "test-lineage", credentialRevision: 0, accessToken: "token", userId: "user-1",
      }))
      act(() => hook.result.current.connect())
      expect(MockWebSocket.instances).toHaveLength(2)
      const currentSocket = MockWebSocket.instances[1]
      act(() => currentSocket.open())
      if (refreshSucceeded) {
        act(() => currentSocket.triggerClose(1000, "current socket closed cleanly"))
      }

      await act(async () => {
        refresh.resolve(refreshSucceeded)
        await Promise.resolve()
        await vi.advanceTimersByTimeAsync(1000)
      })

      expect(onError).not.toHaveBeenCalled()
      expect(MockWebSocket.instances).toHaveLength(2)
      hook.unmount()
    },
  )

  it.each([
    "default close",
    "explicit disconnect",
    "descriptor replacement",
    "unmount",
  ] as const)(
    "notifies the socket-owning disconnect callback exactly once on %s",
    async (operation) => {
      const owningDisconnect = vi.fn()
      const latestDisconnect = vi.fn()
      const initialConnection = sessionConnection()
      const hook = renderHook(
        ({ connection, onDisconnect }) => useWebSocket({
          connection,
          onDisconnect,
        }),
        {
          initialProps: {
            connection: initialConnection,
            onDisconnect: owningDisconnect,
          },
        },
      )
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
      const ownedSocket = MockWebSocket.instances[0]
      ownedSocket.protocol = "xagent-session-v1"
      act(() => ownedSocket.open())

      if (operation === "descriptor replacement") {
        hook.rerender({
          connection: sessionConnection({
            url: "wss://embed.example/v1/external/chat/sessions/replacement/ws",
          }),
          onDisconnect: latestDisconnect,
        })
        await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
      } else {
        hook.rerender({
          connection: sessionConnection(),
          onDisconnect: latestDisconnect,
        })
        expect(MockWebSocket.instances).toHaveLength(1)
        if (operation === "default close") {
          act(() => ownedSocket.triggerClose(1006))
        } else if (operation === "explicit disconnect") {
          act(() => hook.result.current.disconnect())
        } else {
          hook.unmount()
        }
      }

      act(() => ownedSocket.triggerClose(1011))
      expect(owningDisconnect).toHaveBeenCalledOnce()
      expect(latestDisconnect).not.toHaveBeenCalled()
      if (operation !== "unmount") hook.unmount()
    },
  )

  it.each([
    "default close",
    "explicit disconnect",
    "descriptor replacement",
    "unmount",
  ] as const)(
    "finishes owner retirement when the disconnect callback throws on %s",
    async (operation) => {
      const secret = "disconnect callback raw secret"
      const oldUpload = deferred<Array<{ file_id: string }>>()
      const uploadFiles = vi.fn(() => oldUpload.promise)
      const throwingDisconnect = vi.fn(() => {
        throw new Error(secret)
      })
      const safeDisconnect = vi.fn()
      const consoleError = vi.spyOn(console, "error").mockImplementation(() => {})
      const hook = renderHook(
        ({ taskId, onDisconnect }) => useWebSocket({
          url: "ws://localhost",
          taskId,
          uploadFiles,
          onDisconnect,
        }),
        {
          initialProps: {
            taskId: 1,
            onDisconnect: throwingDisconnect,
          },
        },
      )
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
      const oldSocket = MockWebSocket.instances[0]
      act(() => oldSocket.open())
      const pendingOutcome = hook.result.current.sendChatMessage(
        "pending",
        undefined,
        false,
        "throwing-disconnect-pending",
      ).then(
        () => "resolved",
        error => `rejected:${(error as Error).message}`,
      )
      const preparationOutcome = hook.result.current.sendChatMessage(
        "preparing",
        [new File(["data"], "data.txt")],
        false,
        "throwing-disconnect-preparation",
      ).then(
        () => "resolved",
        error => `rejected:${(error as Error).message}`,
      )

      let escaped: unknown
      try {
        if (operation === "default close") {
          act(() => oldSocket.triggerClose(1006))
        } else if (operation === "explicit disconnect") {
          act(() => hook.result.current.disconnect())
        } else if (operation === "descriptor replacement") {
          hook.rerender({
            taskId: 2,
            onDisconnect: safeDisconnect,
          })
        } else {
          hook.unmount()
        }
      } catch (error) {
        escaped = error
      }
      await act(async () => {
        await Promise.resolve()
      })
      const stateWasDisconnected = operation === "unmount"
        ? true
        : hook.result.current.isConnected === false

      await act(async () => {
        oldUpload.resolve([{ file_id: "late-file" }])
        await Promise.resolve()
      })

      let replacementContinued = operation === "unmount"
      if (operation !== "unmount" && escaped === undefined) {
        if (operation !== "descriptor replacement") {
          hook.rerender({
            taskId: 1,
            onDisconnect: safeDisconnect,
          })
          act(() => hook.result.current.connect())
        }
        replacementContinued = MockWebSocket.instances.length === 2
        if (replacementContinued) {
          act(() => MockWebSocket.instances[1].open())
        }
        hook.unmount()
      } else if (operation !== "unmount") {
        hook.unmount()
      }

      expect(escaped).toBeUndefined()
      expect(stateWasDisconnected).toBe(true)
      expect(oldSocket.readyState).toBe(MockWebSocket.CLOSED)
      expect(await pendingOutcome).toContain("rejected:")
      expect(await preparationOutcome).toContain("rejected:")
      expect(throwingDisconnect).toHaveBeenCalledOnce()
      expect(replacementContinued).toBe(true)
      expect(consoleError).toHaveBeenCalledWith(
        "WebSocket disconnect handler failed",
      )
      const exposed = [
        String(escaped),
        ...consoleError.mock.calls.flat().map(String),
      ].join(" ")
      expect(exposed).not.toContain(secret)
    },
  )

  it("uses the normalized connection object as lifecycle identity", async () => {
    const connection = sessionConnection()
    const hook = renderHook(
      ({ connection }) => useWebSocket({ connection }),
      { initialProps: { connection } },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))

    hook.rerender({ connection })
    expect(MockWebSocket.instances).toHaveLength(1)

    hook.rerender({ connection: { ...connection } })
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
  })

  it("retires a stalled handshake before reporting its structured transport failure", async () => {
    vi.useFakeTimers()
    const onConnectionFailure = vi.fn()
    const { result } = renderHook(() => useWebSocket({
      connection: sessionConnection(),
      onConnectionFailure,
    }))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(MockWebSocket.instances).toHaveLength(1)
    const socket = MockWebSocket.instances[0]

    act(() => vi.advanceTimersByTime(10_000))

    expect(socket.close).toHaveBeenCalled()
    expect(result.current.isConnected).toBe(false)
    expect(onConnectionFailure).toHaveBeenCalledWith(expect.objectContaining({
      recoverable: true,
    }))
  })

  it("retires an errored owner before its failure callback can synchronously retry", async () => {
    let retry!: () => void
    const onConnectionFailure = vi.fn(() => retry())
    const { result } = renderHook(() => {
      const webSocket = useWebSocket({
        connection: sessionConnection(),
        onConnectionFailure,
      })
      retry = webSocket.connect
      return webSocket
    })
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))

    act(() => MockWebSocket.instances[0].triggerError())

    expect(onConnectionFailure).toHaveBeenCalledOnce()
    expect(MockWebSocket.instances).toHaveLength(2)
    expect(result.current.isConnected).toBe(false)
  })

  it("forwards the owning Session identity with close and transport-failure callbacks", async () => {
    const onSessionConnectionClose = vi.fn(() => "handled" as const)
    const onSessionConnectionFailure = vi.fn()
    const connection = sessionConnection({ identity: "widget-session:owner-a" })
    const closeHook = renderHook(() => useWebSocket({
      connection,
      onSessionConnectionClose,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const closeSocket = MockWebSocket.instances[0]
    closeSocket.protocol = "xagent-session-v1"
    act(() => closeSocket.open())
    act(() => closeSocket.triggerClose(1006))
    expect(onSessionConnectionClose).toHaveBeenCalledWith(
      expect.objectContaining({ code: 1006 }),
      "widget-session:owner-a",
    )
    closeHook.unmount()

    MockWebSocket.instances = []
    const failureHook = renderHook(() => useWebSocket({
      connection,
      onSessionConnectionFailure,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    act(() => MockWebSocket.instances[0].triggerError())
    expect(onSessionConnectionFailure).toHaveBeenCalledWith(
      expect.objectContaining({ recoverable: true }),
      "widget-session:owner-a",
    )
    failureHook.unmount()
  })

  it("bounds 4001 refresh retries across socket opens without resetting on open", async () => {
    authState.refreshAccessToken.mockResolvedValue(true)
    const onConnectionFailure = vi.fn()
    const hook = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      onConnectionFailure,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    vi.useFakeTimers()

    for (let index = 0; index < 3; index += 1) {
      const socket = MockWebSocket.instances[index]
      act(() => socket.open())
      act(() => socket.triggerClose(4001))
      await act(async () => {
        await Promise.resolve()
        await vi.advanceTimersByTimeAsync(1_000)
      })
    }

    expect(MockWebSocket.instances).toHaveLength(4)
    const exhaustedSocket = MockWebSocket.instances[3]
    act(() => exhaustedSocket.open())
    act(() => exhaustedSocket.triggerClose(4001))
    await act(async () => {
      await Promise.resolve()
      await vi.runOnlyPendingTimersAsync()
    })
    expect(MockWebSocket.instances).toHaveLength(4)
    expect(onConnectionFailure).toHaveBeenCalledWith({
      recoverable: false,
      error: expect.objectContaining({
        message: "Authentication failed after token refresh retries",
      }),
    })

    hook.unmount()
  })

  it("keeps an exhausted 4001 retry budget across an auth credential revision", async () => {
    authState.refreshAccessToken.mockResolvedValue(true)
    const onError = vi.fn()
    const hook = renderHook(() => useWebSocket({ url: "ws://localhost", taskId: 1, onError }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    vi.useFakeTimers()

    for (let index = 0; index < 3; index += 1) {
      const socket = MockWebSocket.instances[index]
      act(() => socket.open())
      act(() => socket.triggerClose(4001))
      await act(async () => {
        await Promise.resolve()
        await vi.advanceTimersByTimeAsync(1_000)
      })
    }

    authState.token = "advanced-token"
    authState.session = {
      ...authState.session,
      accessToken: "advanced-token",
      credentialRevision: 1,
    }
    hook.rerender()
    expect(MockWebSocket.instances).toHaveLength(5)
    const revisedSocket = MockWebSocket.instances[4]
    act(() => revisedSocket.open())
    act(() => revisedSocket.triggerClose(4001))

    expect(authState.refreshAccessToken).toHaveBeenCalledTimes(3)
    expect(onError).toHaveBeenCalledWith(expect.objectContaining({
      message: "Authentication failed after token refresh retries",
    }))
    hook.unmount()
  })

  it("keeps an exhausted 4001 retry budget across a null connection gap for the same auth session", async () => {
    authState.refreshAccessToken.mockResolvedValue(true)
    const onError = vi.fn()
    const hook = renderHook(
      ({ connection }: { connection: WebSocketConnection | null | undefined }) => useWebSocket({
        url: "ws://localhost",
        taskId: 1,
        connection,
        onError,
      }),
      { initialProps: { connection: undefined } as { connection: WebSocketConnection | null | undefined } },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    vi.useFakeTimers()

    for (let index = 0; index < 3; index += 1) {
      const socket = MockWebSocket.instances[index]
      act(() => socket.open())
      act(() => socket.triggerClose(4001))
      await act(async () => {
        await Promise.resolve()
        await vi.advanceTimersByTimeAsync(1_000)
      })
    }

    hook.rerender({ connection: null })
    hook.rerender({ connection: undefined })
    expect(MockWebSocket.instances).toHaveLength(5)
    const sameSessionSocket = MockWebSocket.instances[4]
    act(() => sameSessionSocket.open())
    act(() => sameSessionSocket.triggerClose(4001))

    expect(authState.refreshAccessToken).toHaveBeenCalledTimes(3)
    expect(onError).toHaveBeenCalledWith(expect.objectContaining({
      message: "Authentication failed after token refresh retries",
    }))
    hook.unmount()
  })

  it("does not replenish an exhausted auth retry budget from external socket activity", async () => {
    authState.refreshAccessToken.mockResolvedValue(true)
    const onError = vi.fn()
    const hook = renderHook(
      ({ connection }: { connection: WebSocketConnection | undefined }) => useWebSocket({
        url: "ws://localhost",
        taskId: 1,
        connection,
        onError,
      }),
      { initialProps: { connection: undefined } as { connection: WebSocketConnection | undefined } },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    vi.useFakeTimers()

    for (let index = 0; index < 3; index += 1) {
      const socket = MockWebSocket.instances[index]
      act(() => socket.open())
      act(() => socket.triggerClose(4001))
      await act(async () => {
        await Promise.resolve()
        await vi.advanceTimersByTimeAsync(1_000)
      })
    }

    hook.rerender({ connection: sessionConnection() })
    expect(MockWebSocket.instances).toHaveLength(5)
    const externalSocket = MockWebSocket.instances[4]
    externalSocket.protocol = "xagent-session-v1"
    act(() => {
      externalSocket.open()
      externalSocket.receive({ type: "task_info", task_id: 1 })
    })
    hook.rerender({ connection: undefined })
    expect(MockWebSocket.instances).toHaveLength(6)
    const sameSessionSocket = MockWebSocket.instances[5]
    act(() => sameSessionSocket.open())
    act(() => sameSessionSocket.triggerClose(4001))

    expect(authState.refreshAccessToken).toHaveBeenCalledTimes(3)
    expect(onError).toHaveBeenCalledWith(expect.objectContaining({
      message: "Authentication failed after token refresh retries",
    }))
    hook.unmount()
  })

  it("resets an exhausted 4001 retry budget for a new auth session lineage", async () => {
    authState.refreshAccessToken.mockResolvedValue(true)
    const hook = renderHook(() => useWebSocket({ url: "ws://localhost", taskId: 1 }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    vi.useFakeTimers()

    for (let index = 0; index < 3; index += 1) {
      const socket = MockWebSocket.instances[index]
      act(() => socket.open())
      act(() => socket.triggerClose(4001))
      await act(async () => {
        await Promise.resolve()
        await vi.advanceTimersByTimeAsync(1_000)
      })
    }

    authState.token = "replacement-token"
    authState.session = {
      ...authState.session,
      sessionId: "replacement-lineage",
      accessToken: "replacement-token",
      credentialRevision: 0,
    }
    hook.rerender()
    expect(MockWebSocket.instances).toHaveLength(5)
    const replacementSocket = MockWebSocket.instances[4]
    act(() => replacementSocket.open())
    act(() => replacementSocket.triggerClose(4001))
    await act(async () => {
      await Promise.resolve()
      await vi.advanceTimersByTimeAsync(1_000)
    })

    expect(authState.refreshAccessToken).toHaveBeenCalledTimes(4)
    expect(MockWebSocket.instances).toHaveLength(6)
    hook.unmount()
  })

  it("keeps a stale auth refresh result from affecting a replacement session lineage", async () => {
    const refresh = deferred<boolean>()
    authState.refreshAccessToken.mockReturnValue(refresh.promise)
    const onError = vi.fn()
    const hook = renderHook(() => useWebSocket({ url: "ws://localhost", taskId: 1, onError }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const oldSocket = MockWebSocket.instances[0]
    act(() => oldSocket.open())
    vi.useFakeTimers()

    act(() => oldSocket.triggerClose(4001))
    expect(authState.refreshAccessToken).toHaveBeenCalledWith(expect.objectContaining({
      sessionId: "test-lineage",
    }))

    authState.token = "replacement-token"
    authState.session = {
      ...authState.session,
      sessionId: "replacement-lineage",
      accessToken: "replacement-token",
      credentialRevision: 0,
    }
    hook.rerender()
    expect(MockWebSocket.instances).toHaveLength(2)
    act(() => MockWebSocket.instances[1].open())

    await act(async () => {
      refresh.resolve(true)
      await Promise.resolve()
      await vi.advanceTimersByTimeAsync(1_000)
    })

    expect(MockWebSocket.instances).toHaveLength(2)
    expect(onError).not.toHaveBeenCalled()
    hook.unmount()
  })

  it("does not replenish the 4001 retry budget after valid socket activity", async () => {
    authState.refreshAccessToken.mockResolvedValue(true)
    const hook = renderHook(() => useWebSocket({ url: "ws://localhost", taskId: 1 }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    vi.useFakeTimers()

    for (let index = 0; index < 2; index += 1) {
      const socket = MockWebSocket.instances[index]
      act(() => socket.open())
      act(() => socket.triggerClose(4001))
      await act(async () => {
        await Promise.resolve()
        await vi.advanceTimersByTimeAsync(1_000)
      })
    }

    const activitySocket = MockWebSocket.instances[2]
    act(() => {
      activitySocket.open()
      activitySocket.receive({ type: "task_info", task_id: 1 })
    })
    for (let index = 2; index < 3; index += 1) {
      const socket = MockWebSocket.instances[index]
      act(() => socket.triggerClose(4001))
      await act(async () => {
        await Promise.resolve()
        await vi.advanceTimersByTimeAsync(1_000)
      })
    }

    expect(authState.refreshAccessToken).toHaveBeenCalledTimes(3)
    expect(MockWebSocket.instances).toHaveLength(4)
    hook.unmount()
  })

  it.each([
    ["authorization expired", "authorization expired"],
    ["", "Access denied"],
  ])("reports 4003 close reason %j or the access-denied fallback through structured failure", async (reason, expectedMessage) => {
    const onError = vi.fn()
    const onConnectionFailure = vi.fn()
    const { result } = renderHook(() => useWebSocket({
      connection: sessionConnection(),
      onError,
      onConnectionFailure,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    socket.protocol = "xagent-session-v1"
    act(() => socket.open())
    act(() => socket.triggerClose(4003, reason))

    expect(result.current.connectionError?.message).toBe(expectedMessage)
    expect(onError).toHaveBeenCalledWith(expect.objectContaining({ message: expectedMessage }))
    expect(onConnectionFailure).toHaveBeenCalledWith({
      recoverable: false,
      error: expect.objectContaining({ message: expectedMessage }),
    })
  })

  it.each([
    ["external credentials", () => sessionConnection()],
    ["missing auth lineage", () => ({
      ...sessionConnection(),
      credentialOwner: {
        kind: "auth-context" as const,
        accessToken: "access",
        userId: "user-1",
      },
  })],
  ])("reports permanent 4001 %s through structured failure", async (_name, makeConnection) => {
    const onConnectionFailure = vi.fn()
    const onError = vi.fn()
    const connection = makeConnection()
    const { result } = renderHook(() => useWebSocket({
      connection,
      onConnectionFailure,
      onError,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    socket.protocol = "xagent-session-v1"
    act(() => socket.open())

    act(() => socket.triggerClose(4001))

    expect(result.current.connectionError?.message).toMatch(/authentication/i)
    expect(onConnectionFailure).toHaveBeenCalledWith({
      recoverable: false,
      error: expect.objectContaining({ message: result.current.connectionError?.message }),
    })
    expect(onError).toHaveBeenCalledOnce()
  })

  it.each([
    ["returns false", () => authState.refreshAccessToken.mockResolvedValue(false)],
    ["rejects", () => authState.refreshAccessToken.mockRejectedValue(new Error("refresh failed"))],
    ["throws", () => authState.refreshAccessToken.mockImplementation(() => {
      throw new Error("refresh threw")
    })],
  ] as const)("reports a 4001 refresh that %s through structured failure", async (_name, configureRefresh) => {
    configureRefresh()
    const onConnectionFailure = vi.fn()
    const onError = vi.fn()
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      onConnectionFailure,
      onError,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    act(() => socket.triggerClose(4001))

    await waitFor(() => expect(onConnectionFailure).toHaveBeenCalledOnce())
    expect(result.current.connectionError?.message).toMatch(/authentication failed/i)
    expect(onError).toHaveBeenCalledOnce()
  })

  it("resets the transport reconnect delay after explicit disconnect without resetting auth refresh", async () => {
    const { result } = renderHook(() => useWebSocket({ url: "ws://localhost", taskId: 1 }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    vi.useFakeTimers()
    const first = MockWebSocket.instances[0]
    act(() => first.open())
    act(() => first.triggerClose(1011))
    act(() => result.current.disconnect())
    act(() => result.current.connect())
    expect(MockWebSocket.instances).toHaveLength(2)
    const second = MockWebSocket.instances[1]
    act(() => second.open())
    act(() => second.triggerClose(1011))

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000)
    })
    expect(MockWebSocket.instances).toHaveLength(3)
  })

  it("does not suppress a retry for an arbitrary close reason", async () => {
    const hook = renderHook(() => useWebSocket({ url: "ws://localhost", taskId: 1 }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    vi.useFakeTimers()
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())
    act(() => socket.triggerClose(1011, "Component unmounting"))
    await act(async () => vi.advanceTimersByTimeAsync(1_000))

    expect(MockWebSocket.instances).toHaveLength(2)
    hook.unmount()
  })

  it("replaces the socket when an auth-owned credential changes", async () => {
    const secret = "xagent-session-token.st_descriptor_secret"
    const first = sessionConnection({
      credentialOwner: {
        kind: "auth-context",
        accessToken: "first-access-token",
        userId: "user-1",
      },
      protocols: ["xagent-session-v1", secret],
    })
    const { rerender } = renderHook(
      ({ connection }) => useWebSocket({ connection }),
      { initialProps: { connection: first } },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))

    rerender({ connection: {
      ...first,
      credentialOwner: {
        kind: "auth-context",
        accessToken: "second-access-token",
        userId: "user-1",
      },
    } })

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
  })

  it("starts a new lifecycle when a caller replaces the descriptor for a profile-only auth revision", async () => {
    const first = sessionConnection({ credentialOwner: {
      kind: "auth-context", accessToken: "access", userId: "user-1",
      session: { ...authState.session, profileRevision: 0 },
    } })
    const firstOwner = first.credentialOwner
    if (firstOwner.kind !== "auth-context") throw new Error("expected auth owner")
    const { rerender } = renderHook(({ connection }) => useWebSocket({ connection }), { initialProps: { connection: first } })
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    rerender({ connection: { ...first, credentialOwner: {
      ...firstOwner, session: { ...authState.session, profileRevision: 1, profileFingerprint: '["alice",null,null]' },
    } } })
    await Promise.resolve()
    expect(MockWebSocket.instances).toHaveLength(2)
  })

  it("replaces a socket when its auth credential revision changes", async () => {
    const first = sessionConnection({ credentialOwner: {
      kind: "auth-context", accessToken: "access", userId: "user-1",
      session: { ...authState.session, credentialRevision: 0 },
    } })
    const firstOwner = first.credentialOwner
    if (firstOwner.kind !== "auth-context") throw new Error("expected auth owner")
    const { rerender } = renderHook(({ connection }) => useWebSocket({ connection }), { initialProps: { connection: first } })
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    rerender({ connection: { ...first, credentialOwner: {
      ...firstOwner, accessToken: "new-access", session: { ...authState.session, accessToken: "new-access", credentialRevision: 1 },
    } } })
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
  })

  it("replaces a socket for the known former descriptor-hash collision pair", async () => {
    const first = sessionConnection({
      protocols: ["xagent-session-v1", "xagent-session-token.st_collision_0073zx"],
    })
    const { rerender } = renderHook(
      ({ connection }) => useWebSocket({ connection }),
      { initialProps: { connection: first } },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))

    rerender({
      connection: {
        ...first,
        protocols: ["xagent-session-v1", "xagent-session-token.st_collision_00apad"],
      },
    })

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
  })

  it("replaces sockets for every connection field without exposing descriptor material", async () => {
    const secretQuery = "bearer-query-secret"
    const secretProtocol = "xagent-session-token.st_fingerprint_secret"
    const connection: WebSocketConnection = {
      identity: "identity-1",
      url: `wss://embed.example/ws?token=${secretQuery}`,
      protocols: ["xagent-session-v1", secretProtocol],
      expectedProtocol: "xagent-session-v1",
      taskId: 1,
      chatTaskIdMode: "required",
      credentialOwner: {
        kind: "auth-context",
        accessToken: "access-token-1",
        userId: "user-1",
      },
    }
    const variants: WebSocketConnection[] = [
      { ...connection, identity: "identity-2" },
      { ...connection, url: "wss://embed.example/ws?token=changed-query-secret" },
      { ...connection, protocols: ["xagent-session-v1", "xagent-session-token.changed"] },
      { ...connection, expectedProtocol: "xagent-session-v2" },
      { ...connection, taskId: 2 },
      { ...connection, chatTaskIdMode: "omit" },
      { ...connection, credentialOwner: { kind: "external" } },
      { ...connection, credentialOwner: { kind: "auth-context", accessToken: "access-token-2", userId: "user-1" } },
      { ...connection, credentialOwner: { kind: "auth-context", accessToken: "access-token-1", userId: "user-2" } },
    ]

    for (const currentConnection of variants) {
      MockWebSocket.instances = []
      const hook = renderHook(
        ({ descriptor }) => useWebSocket({ connection: descriptor }),
        { initialProps: { descriptor: connection } },
      )
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))

      hook.rerender({ descriptor: currentConnection })
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))

      const exposed = JSON.stringify(hook.result.current)
      expect(exposed).not.toContain(secretQuery)
      expect(exposed).not.toContain(secretProtocol)
      expect(exposed).not.toContain("access-token-1")
      hook.unmount()
    }
  })

  it("keeps the callback snapshot that owned a socket attempt", async () => {
    const firstFailure = vi.fn()
    const replacementFailure = vi.fn()
    const hook = renderHook(
      ({ onConnectionFailure }) => useWebSocket({
        connection: sessionConnection(),
        onConnectionFailure,
      }),
      { initialProps: { onConnectionFailure: firstFailure } },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    hook.rerender({ onConnectionFailure: replacementFailure })

    act(() => MockWebSocket.instances[0].triggerError())

    expect(firstFailure).toHaveBeenCalledOnce()
    expect(replacementFailure).not.toHaveBeenCalled()
  })

  it("makes stale open, message, error, and close callbacks inert after replacement", async () => {
    const onConnect = vi.fn()
    const onMessage = vi.fn()
    const onError = vi.fn()
    const onDisconnect = vi.fn()
    const onConnectionClose = vi.fn(() => "handled" as const)
    const { result, rerender } = renderHook(
      ({ connection }) => useWebSocket({
        connection,
        onConnect,
        onMessage,
        onError,
        onDisconnect,
        onConnectionClose,
      }),
      { initialProps: { connection: sessionConnection() } },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const oldSocket = MockWebSocket.instances[0]

    rerender({ connection: sessionConnection({ identity: "widget-session:2" }) })
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
    const currentSocket = MockWebSocket.instances[1]
    currentSocket.protocol = "xagent-session-v1"
    act(() => currentSocket.open())
    onConnect.mockClear()
    onMessage.mockClear()
    onError.mockClear()
    onDisconnect.mockClear()
    onConnectionClose.mockClear()

    oldSocket.protocol = "xagent-session-v1"
    act(() => {
      oldSocket.open()
      oldSocket.receive({ type: "task_info", task_id: 999 })
      oldSocket.triggerError()
      oldSocket.triggerClose(1011)
    })

    expect(result.current.isConnected).toBe(true)
    expect(result.current.lastMessage).toBeNull()
    expect(onConnect).not.toHaveBeenCalled()
    expect(onMessage).not.toHaveBeenCalled()
    expect(onError).not.toHaveBeenCalled()
    expect(onDisconnect).not.toHaveBeenCalled()
    expect(onConnectionClose).not.toHaveBeenCalled()
    expect(MockWebSocket.instances).toHaveLength(2)
  })

  it("rejects only old pending delivery on replacement and ignores its late ack", async () => {
    const { result, rerender } = renderHook(
      ({ connection }) => useWebSocket({ connection }),
      { initialProps: { connection: sessionConnection() } },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const oldSocket = MockWebSocket.instances[0]
    oldSocket.protocol = "xagent-session-v1"
    act(() => oldSocket.open())
    const oldDelivery = result.current.sendChatMessage(
      "old",
      undefined,
      false,
      "shared-id",
    )
    const oldRejection = expect(oldDelivery).rejects.toThrow("replaced")

    rerender({ connection: sessionConnection({ identity: "widget-session:2" }) })
    await oldRejection
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
    const currentSocket = MockWebSocket.instances[1]
    currentSocket.protocol = "xagent-session-v1"
    act(() => currentSocket.open())
    const currentDelivery = result.current.sendChatMessage(
      "new",
      undefined,
      false,
      "shared-id",
    )
    let currentSettled = false
    void currentDelivery.finally(() => {
      currentSettled = true
    })

    act(() => {
      oldSocket.receive({
        type: "message_accepted",
        client_message_id: "shared-id",
      })
      oldSocket.triggerClose(1011)
    })
    await Promise.resolve()
    expect(currentSettled).toBe(false)

    act(() => currentSocket.receive({
      type: "message_accepted",
      client_message_id: "shared-id",
    }))
    await expect(currentDelivery).resolves.toMatchObject({
      client_message_id: "shared-id",
    })
  })

  it("invalidates pending and dedupe state on delivery generation without reconnecting", async () => {
    const connection = sessionConnection()
    const { result, rerender } = renderHook(
      ({ deliveryGeneration }) => useWebSocket({
        connection,
        deliveryGeneration,
      }),
      { initialProps: { deliveryGeneration: 0 } },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    socket.protocol = "xagent-session-v1"
    act(() => socket.open())
    const oldDelivery = result.current.sendChatMessage("same text")
    const oldRejection = expect(oldDelivery).rejects.toThrow("generation")

    rerender({ deliveryGeneration: 1 })
    await oldRejection
    expect(MockWebSocket.instances).toHaveLength(1)

    const replacementDelivery = result.current.sendChatMessage("same text")
    expect(socket.send).toHaveBeenCalledTimes(2)
    const replacementFrame = JSON.parse(socket.send.mock.calls[1][0])
    act(() => socket.receive({
      type: "message_accepted",
      client_message_id: replacementFrame.client_message_id,
    }))
    await expect(replacementDelivery).resolves.toMatchObject({
      client_message_id: replacementFrame.client_message_id,
    })
  })

  it("commits the generation fence before a consumer layout effect can acknowledge", async () => {
    const connection = sessionConnection()
    const { result, rerender } = renderHook(
      ({ acknowledgeOnCommit, deliveryGeneration }) => {
        const webSocket = useWebSocket({
          connection,
          deliveryGeneration,
        })
        useLayoutEffect(() => {
          if (!acknowledgeOnCommit) return
          MockWebSocket.instances[0]?.receive({
            type: "message_accepted",
            client_message_id: "layout-fence-id",
          })
        }, [acknowledgeOnCommit, deliveryGeneration])
        return webSocket
      },
      {
        initialProps: {
          acknowledgeOnCommit: false,
          deliveryGeneration: 0,
        },
      },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    socket.protocol = "xagent-session-v1"
    act(() => socket.open())
    const oldDelivery = result.current.sendChatMessage(
      "old generation",
      undefined,
      false,
      "layout-fence-id",
    )
    const outcome = oldDelivery.then(
      () => "resolved",
      error => `rejected:${(error as Error).message}`,
    )

    rerender({
      acknowledgeOnCommit: true,
      deliveryGeneration: 1,
    })

    expect(await outcome).toContain("generation")
    expect(MockWebSocket.instances).toHaveLength(1)
  })

  it("keeps duplicate detection local to each hook", async () => {
    const first = renderHook(() => useWebSocket({
      connection: sessionConnection({ identity: "shared-identity" }),
    }))
    const second = renderHook(() => useWebSocket({
      connection: sessionConnection({ identity: "shared-identity" }),
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
    const firstSocket = MockWebSocket.instances[0]
    const secondSocket = MockWebSocket.instances[1]
    firstSocket.protocol = "xagent-session-v1"
    secondSocket.protocol = "xagent-session-v1"
    act(() => {
      firstSocket.open()
      secondSocket.open()
    })

    const firstDelivery = first.result.current.sendChatMessage("same")
    const secondDelivery = second.result.current.sendChatMessage("same")
    expect(firstSocket.send).toHaveBeenCalledOnce()
    expect(secondSocket.send).toHaveBeenCalledOnce()

    const firstFrame = JSON.parse(firstSocket.send.mock.calls[0][0])
    const secondFrame = JSON.parse(secondSocket.send.mock.calls[0][0])
    act(() => {
      firstSocket.receive({
        type: "message_accepted",
        client_message_id: firstFrame.client_message_id,
      })
      secondSocket.receive({
        type: "message_accepted",
        client_message_id: secondFrame.client_message_id,
      })
    })
    await Promise.all([firstDelivery, secondDelivery])
  })

  it("uses the current socket for retained raw protocol sends", async () => {
    const { result, rerender } = renderHook(
      ({ connection }) => useWebSocket({ connection }),
      { initialProps: { connection: sessionConnection() } },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const oldSocket = MockWebSocket.instances[0]
    oldSocket.protocol = "xagent-session-v1"
    act(() => oldSocket.open())
    const sendRaw = result.current.sendMessage

    rerender({ connection: sessionConnection({ identity: "widget-session:2" }) })
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
    const currentSocket = MockWebSocket.instances[1]
    currentSocket.protocol = "xagent-session-v1"
    act(() => currentSocket.open())
    act(() => sendRaw({ type: "new_conversation" }))

    expect(oldSocket.send).not.toHaveBeenCalled()
    expect(currentSocket.send).toHaveBeenCalledWith(
      JSON.stringify({ type: "new_conversation" }),
    )
  })

  it("fails taskless file delivery before any upload", async () => {
    const uploadFiles = vi.fn()
    const { result } = renderHook(() => useWebSocket({
      connection: sessionConnection(),
      uploadFiles,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    socket.protocol = "xagent-session-v1"
    act(() => socket.open())

    const file = new File(["secret"], "secret.txt", { type: "text/plain" })
    await expect(
      result.current.sendChatMessage("with file", [file])
    ).rejects.toMatchObject({
      message: "File delivery is not supported for this connection.",
      disposition: "not_sent",
    })
    expect(uploadFiles).not.toHaveBeenCalled()
    expect(socket.send).not.toHaveBeenCalled()
  })

  it("rejects a coded unsuccessful 2xx batch upload before sending chat", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({
        success: false,
        error_code: "upload_too_large",
        detail: "private proxy detail",
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    )
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    await expect(result.current.sendChatMessage(
      "with file",
      [new File(["data"], "data.txt")],
    )).rejects.toMatchObject({
      errorCode: "upload_too_large",
      message: "File is too large. Please reduce the upload size and try again.",
      userFacing: true,
    })
    expect(socket.send).not.toHaveBeenCalled()
    fetchMock.mockRestore()
  })

  it("rejects a malformed oversized upload result before sending chat", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({
        success: true,
        files: [
          { file_id: 123, filename: "bad.txt" },
          { file_id: "file-2", filename: "second.txt" },
          { file_id: "file-3", filename: "extra.txt" },
        ],
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    )
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    const outcome = result.current.sendChatMessage(
      "with files",
      [
        new File(["one"], "first.txt"),
        new File(["two"], "second.txt"),
      ],
    ).then(
      () => null,
      error => error,
    )

    await waitFor(() => expect(fetchMock).toHaveBeenCalledOnce())
    await expect(outcome).resolves.toMatchObject({
      errorCode: "upload_failed",
      userFacing: true,
    })
    expect(socket.send).not.toHaveBeenCalled()
    fetchMock.mockRestore()
  })

  it("rejects an incomplete custom upload result before sending chat", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      uploadFiles: vi.fn().mockResolvedValue([]),
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    await expect(result.current.sendChatMessage(
      "with file",
      [new File(["data"], "data.txt")],
    )).rejects.toMatchObject({
      errorCode: "upload_failed",
      userFacing: true,
    })
    expect(socket.send).not.toHaveBeenCalled()
  })

  it.each([
    {
      name: "blank",
      uploadResult: [{ file_id: "   " }, { file_id: "file-2" }],
    },
    {
      name: "duplicate",
      uploadResult: [{ file_id: "file-1" }, { file_id: " file-1 " }],
    },
  ])("rejects $name custom upload identifiers before sending chat", async ({ uploadResult }) => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      uploadFiles: vi.fn().mockResolvedValue(uploadResult),
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())

    await expect(result.current.sendChatMessage(
      "with files",
      [
        new File(["first"], "first.txt"),
        new File(["second"], "second.txt"),
      ],
    )).rejects.toMatchObject({
      errorCode: "upload_failed",
      userFacing: true,
    })
    expect(socket.send).not.toHaveBeenCalled()
  })

  it("claims a client message id before awaiting its upload", async () => {
    const upload = deferred<Array<{ file_id: string }>>()
    const uploadFiles = vi.fn(() => upload.promise)
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      uploadFiles,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())
    const file = new File(["data"], "data.txt")

    const first = result.current.sendChatMessage(
      "first owner",
      [file],
      false,
      "shared-upload-id",
    )
    const second = result.current.sendChatMessage(
      "second owner",
      [file],
      false,
      "shared-upload-id",
    )
    const secondOutcome = second.then(
      () => "resolved",
      error => `rejected:${(error as Error).message}`,
    )

    await act(async () => {
      upload.resolve([{ file_id: "uploaded-file" }])
      await Promise.resolve()
    })
    act(() => socket.receive({
      type: "message_accepted",
      client_message_id: "shared-upload-id",
    }))

    await expect(first).resolves.toMatchObject({
      client_message_id: "shared-upload-id",
    })
    expect(await secondOutcome).toContain("already pending")
    expect(uploadFiles).toHaveBeenCalledOnce()
    expect(socket.send).toHaveBeenCalledOnce()
  })

  it("releases an upload claim after preparation fails", async () => {
    const uploadFiles = vi.fn()
      .mockRejectedValueOnce(new Error("upload failed"))
      .mockResolvedValueOnce([{ file_id: "uploaded-file" }])
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      uploadFiles,
    }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())
    const file = new File(["data"], "data.txt")

    await expect(result.current.sendChatMessage(
      "first attempt",
      [file],
      false,
      "retry-upload-id",
    )).rejects.toThrow("upload failed")

    const retry = result.current.sendChatMessage(
      "retry",
      [file],
      false,
      "retry-upload-id",
    )
    await waitFor(() => expect(socket.send).toHaveBeenCalledOnce())
    act(() => socket.receive({
      type: "message_accepted",
      client_message_id: "retry-upload-id",
    }))

    await expect(retry).resolves.toMatchObject({
      client_message_id: "retry-upload-id",
    })
    expect(uploadFiles).toHaveBeenCalledTimes(2)
  })

  it("rejects replaced upload preparation promptly and preserves the replacement claim", async () => {
    const oldUpload = deferred<Array<{ file_id: string }>>()
    const replacementUpload = deferred<Array<{ file_id: string }>>()
    const unexpectedThirdUpload = deferred<Array<{ file_id: string }>>()
    const uploadFiles = vi.fn()
      .mockReturnValueOnce(oldUpload.promise)
      .mockReturnValueOnce(replacementUpload.promise)
      .mockReturnValueOnce(unexpectedThirdUpload.promise)
    const { result, rerender } = renderHook(
      ({ taskId }) => useWebSocket({
        url: "ws://localhost",
        taskId,
        uploadFiles,
      }),
      { initialProps: { taskId: 1 } },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const oldSocket = MockWebSocket.instances[0]
    act(() => oldSocket.open())
    const file = new File(["data"], "data.txt")
    let oldSettled = false
    const oldOutcome = result.current.sendChatMessage(
      "old owner",
      [file],
      false,
      "reclaimed-upload-id",
    ).then(
      () => "resolved",
      error => `rejected:${(error as Error).message}`,
    ).finally(() => {
      oldSettled = true
    })

    rerender({ taskId: 2 })
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
    const replacementSocket = MockWebSocket.instances[1]
    act(() => replacementSocket.open())
    const replacementDelivery = result.current.sendChatMessage(
      "replacement owner",
      [file],
      false,
      "reclaimed-upload-id",
    )
    await act(async () => {
      await Promise.resolve()
    })
    const oldSettledBeforeUploadCompleted = oldSettled

    await act(async () => {
      oldUpload.resolve([{ file_id: "stale-file" }])
      await Promise.resolve()
    })
    const duplicateOutcome = result.current.sendChatMessage(
      "must not steal replacement",
      [file],
      false,
      "reclaimed-upload-id",
    ).then(
      () => "resolved",
      error => `rejected:${(error as Error).message}`,
    )

    await act(async () => {
      replacementUpload.resolve([{ file_id: "replacement-file" }])
      unexpectedThirdUpload.resolve([{ file_id: "unexpected-file" }])
      await Promise.resolve()
    })
    act(() => replacementSocket.receive({
      type: "message_accepted",
      client_message_id: "reclaimed-upload-id",
    }))

    expect(oldSettledBeforeUploadCompleted).toBe(true)
    expect(await oldOutcome).toContain("connection changed")
    expect(await duplicateOutcome).toContain("already pending")
    expect(uploadFiles).toHaveBeenCalledTimes(2)
    expect(oldSocket.send).not.toHaveBeenCalled()
    expect(replacementSocket.send).toHaveBeenCalledOnce()
    await expect(replacementDelivery).resolves.toMatchObject({
      client_message_id: "reclaimed-upload-id",
    })
  })

  it("releases upload preparation on a delivery generation change", async () => {
    const oldUpload = deferred<Array<{ file_id: string }>>()
    const currentUpload = deferred<Array<{ file_id: string }>>()
    const uploadFiles = vi.fn()
      .mockReturnValueOnce(oldUpload.promise)
      .mockReturnValueOnce(currentUpload.promise)
    const { result, rerender } = renderHook(
      ({ deliveryGeneration }) => useWebSocket({
        url: "ws://localhost",
        taskId: 1,
        deliveryGeneration,
        uploadFiles,
      }),
      { initialProps: { deliveryGeneration: 0 } },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())
    const file = new File(["data"], "data.txt")
    let oldSettled = false
    const oldOutcome = result.current.sendChatMessage(
      "old generation",
      [file],
      false,
      "generation-upload-id",
    ).then(
      () => "resolved",
      error => `rejected:${(error as Error).message}`,
    ).finally(() => {
      oldSettled = true
    })

    rerender({ deliveryGeneration: 1 })
    const currentDelivery = result.current.sendChatMessage(
      "current generation",
      [file],
      false,
      "generation-upload-id",
    )
    await act(async () => {
      await Promise.resolve()
    })
    const oldSettledBeforeUploadCompleted = oldSettled

    await act(async () => {
      oldUpload.resolve([{ file_id: "stale-file" }])
      currentUpload.resolve([{ file_id: "current-file" }])
      await Promise.resolve()
    })
    act(() => socket.receive({
      type: "message_accepted",
      client_message_id: "generation-upload-id",
    }))

    expect(oldSettledBeforeUploadCompleted).toBe(true)
    expect(await oldOutcome).toContain("generation")
    expect(uploadFiles).toHaveBeenCalledTimes(2)
    await expect(currentDelivery).resolves.toMatchObject({
      client_message_id: "generation-upload-id",
    })
  })

  it("fails closed when an upload completes after its connection was replaced", async () => {
    let finishUpload!: (files: Array<{ file_id: string }>) => void
    const uploadFiles = vi.fn(() => new Promise<Array<{ file_id: string }>>((resolve) => {
      finishUpload = resolve
    }))
    const { result, rerender } = renderHook(
      ({ taskId }) => useWebSocket({
        url: "ws://localhost",
        taskId,
        uploadFiles,
      }),
      { initialProps: { taskId: 1 } },
    )
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const oldSocket = MockWebSocket.instances[0]
    act(() => oldSocket.open())
    const delivery = result.current.sendChatMessage(
      "upload",
      [new File(["data"], "data.txt")],
    )
    const rejection = expect(delivery).rejects.toThrow("connection changed")
    await waitFor(() => expect(uploadFiles).toHaveBeenCalledOnce())

    rerender({ taskId: 2 })
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
    await act(async () => {
      finishUpload([{ file_id: "uploaded-file" }])
      await Promise.resolve()
    })

    expect(oldSocket.send).not.toHaveBeenCalled()
    await rejection
  })
})

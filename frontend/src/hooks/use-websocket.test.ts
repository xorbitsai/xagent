import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useWebSocket } from "./use-websocket"

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ token: "token", refreshToken: vi.fn() }),
}))

class MockWebSocket {
  static OPEN = 1
  static instances: MockWebSocket[] = []

  readyState = 0
  onopen: (() => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  send = vi.fn()

  constructor(public url: string) {
    MockWebSocket.instances.push(this)
  }

  open() {
    this.readyState = MockWebSocket.OPEN
    this.onopen?.()
  }

  receive(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent)
  }

  close() {
    this.readyState = 3
  }

  triggerClose(code = 1006, reason = "network lost") {
    this.readyState = 3
    this.onclose?.({ code, reason } as CloseEvent)
  }
}

describe("useWebSocket message delivery", () => {
  beforeEach(() => {
    MockWebSocket.instances = []
    vi.stubGlobal("WebSocket", MockWebSocket)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("rejects without clearing the caller when the socket is not open", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
      autoConnect: false,
    }))

    await expect(result.current.sendChatMessage("keep this draft")).rejects.toThrow(
      "connection is not ready",
    )
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

  it("allows an unacknowledged draft to retry with the same id", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
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
      })
    })
    await expect(first).rejects.toThrow("temporary failure")

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

  it("marks definitive rejections so the composer can use a fresh id", async () => {
    const { result } = renderHook(() => useWebSocket({
      url: "ws://localhost",
      taskId: 1,
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
      })
    })

    await expect(delivery).rejects.toMatchObject({
      message: "previous delivery failed",
      retryWithNewId: true,
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

    await expect(delivery).rejects.toThrow("Connection closed")
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
      const rejection = expect(delivery).rejects.toThrow("not acknowledged")
      await act(async () => {
        vi.advanceTimersByTime(30000)
      })
      await rejection
    } finally {
      vi.useRealTimers()
    }
  })
})

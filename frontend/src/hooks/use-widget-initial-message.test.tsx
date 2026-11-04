import { renderHook } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { useWidgetInitialMessage } from "./use-widget-initial-message"

function stubParent() {
  const post = vi.fn()
  vi.stubGlobal("parent", { postMessage: post })
  return post
}

function deliver(value: string) {
  window.dispatchEvent(new MessageEvent("message", {
    data: { xagent: true, v: 1, type: "initial_message", value },
    source: window.parent as unknown as MessageEventSource,
  }))
}

const requests = (post: ReturnType<typeof vi.fn>) => (
  post.mock.calls.filter(([data]) => data.type === "initial_message_request")
)

describe("useWidgetInitialMessage", () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("waits until the page is ready", () => {
    const post = stubParent()
    const { rerender } = renderHook(
      (props) => useWidgetInitialMessage(props),
      { initialProps: { ready: false, hasConversation: false, onMessage: vi.fn() } },
    )
    expect(requests(post)).toEqual([])

    rerender({ ready: true, hasConversation: false, onMessage: vi.fn() })
    expect(requests(post)).toHaveLength(1)
  })

  it("calls the latest handler once and stops listening", () => {
    stubParent()
    const first = vi.fn()
    const latest = vi.fn()
    const { rerender } = renderHook(
      (props) => useWidgetInitialMessage(props),
      { initialProps: { ready: true, hasConversation: false, onMessage: first } },
    )
    rerender({ ready: true, hasConversation: false, onMessage: latest })

    deliver("hello")
    deliver("again")

    expect(first).not.toHaveBeenCalled()
    expect(latest).toHaveBeenCalledTimes(1)
    expect(latest).toHaveBeenCalledWith("hello")
  })

  it("does not ask again after delivery when readiness flaps", () => {
    const post = stubParent()
    const onMessage = vi.fn()
    const { rerender } = renderHook(
      (props) => useWidgetInitialMessage(props),
      { initialProps: { ready: true, hasConversation: false, onMessage } },
    )
    deliver("hello")

    rerender({ ready: false, hasConversation: false, onMessage })
    rerender({ ready: true, hasConversation: false, onMessage })

    expect(requests(post)).toHaveLength(1)
    expect(onMessage).toHaveBeenCalledTimes(1)
  })

  it("asks again when readiness returns before any delivery", () => {
    const post = stubParent()
    const onMessage = vi.fn()
    const { rerender } = renderHook(
      (props) => useWidgetInitialMessage(props),
      { initialProps: { ready: true, hasConversation: false, onMessage } },
    )

    rerender({ ready: false, hasConversation: false, onMessage })
    deliver("lost while not ready")
    expect(onMessage).not.toHaveBeenCalled()

    rerender({ ready: true, hasConversation: false, onMessage })
    expect(requests(post)).toHaveLength(2)
    deliver("hello")
    expect(onMessage).toHaveBeenCalledWith("hello")
  })

  it("swallows a rejected handler", async () => {
    stubParent()
    const onMessage = vi.fn().mockRejectedValue(new Error("send failed"))
    renderHook(() => useWidgetInitialMessage({ ready: true, hasConversation: false, onMessage }))

    deliver("hello")
    await Promise.resolve()

    expect(onMessage).toHaveBeenCalledTimes(1)
  })

  it("never asks once a conversation has been seen, even after it is cleared", () => {
    const post = stubParent()
    const onMessage = vi.fn()
    const { rerender } = renderHook(
      (props) => useWidgetInitialMessage(props),
      { initialProps: { ready: true, hasConversation: true, onMessage } },
    )
    rerender({ ready: true, hasConversation: false, onMessage })

    expect(requests(post)).toEqual([])
    deliver("hello")
    expect(onMessage).not.toHaveBeenCalled()
  })

  it("stops listening when the visitor starts a conversation first", () => {
    stubParent()
    const onMessage = vi.fn()
    const { rerender } = renderHook(
      (props) => useWidgetInitialMessage(props),
      { initialProps: { ready: true, hasConversation: false, onMessage } },
    )
    rerender({ ready: true, hasConversation: true, onMessage })

    deliver("hello")

    expect(onMessage).not.toHaveBeenCalled()
  })
})

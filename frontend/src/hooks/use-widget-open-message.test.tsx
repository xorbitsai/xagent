import { act, cleanup, renderHook } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { useWidgetOpenMessage } from "./use-widget-open-message"

function stubParent() {
  const post = vi.fn()
  vi.stubGlobal("parent", { postMessage: post })
  return post
}

function deliver(id: string, value: string) {
  act(() => {
    window.dispatchEvent(new MessageEvent("message", {
      data: { xagent: true, v: 1, type: "open_message", id, value },
      source: window.parent as unknown as MessageEventSource,
    }))
  })
}

const postedOfType = (post: ReturnType<typeof vi.fn>, type: string) => (
  post.mock.calls.filter(([data]) => data.type === type)
)

function renderOpenMessage(enabled: boolean) {
  return renderHook(
    (props: { enabled: boolean }) => useWidgetOpenMessage(props),
    { initialProps: { enabled } },
  )
}

describe("useWidgetOpenMessage", () => {
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it("neither asks nor accepts while disabled", () => {
    const post = stubParent()
    const { result } = renderOpenMessage(false)

    deliver("1", "A")

    expect(postedOfType(post, "open_message_request")).toEqual([])
    expect(result.current.request).toBeNull()
  })

  it("keeps only the newest message, and clears it only for its own sequence", () => {
    const post = stubParent()
    const { result, rerender } = renderOpenMessage(false)

    rerender({ enabled: true })
    expect(postedOfType(post, "open_message_request")).toHaveLength(1)

    deliver("1", "A")
    expect(result.current.request).toEqual({ seq: 1, message: "A" })
    deliver("2", "B")
    expect(result.current.request).toEqual({ seq: 2, message: "B" })

    act(() => result.current.take(1))
    expect(result.current.request).toEqual({ seq: 2, message: "B" })
    act(() => result.current.take(2))
    expect(result.current.request).toBeNull()
  })

  it("drops a repeat of the last accepted id after subscribing again", () => {
    const post = stubParent()
    const { result, rerender } = renderOpenMessage(true)
    deliver("2", "B")
    act(() => result.current.take(1))
    expect(result.current.request).toBeNull()

    rerender({ enabled: false })
    rerender({ enabled: true })
    expect(postedOfType(post, "open_message_request")).toHaveLength(2)
    deliver("2", "B")

    expect(result.current.request).toBeNull()
  })

  it("stops listening once unmounted", () => {
    const post = stubParent()
    const { result, unmount } = renderOpenMessage(true)
    deliver("1", "A")
    const last = result.current.request
    unmount()
    post.mockClear()

    deliver("2", "B")

    expect(postedOfType(post, "open_message_ack")).toEqual([])
    expect(result.current.request).toBe(last)
  })
})

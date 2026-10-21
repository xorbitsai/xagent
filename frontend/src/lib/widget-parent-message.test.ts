import { afterEach, describe, expect, it, vi } from "vitest"

import { requestPromptContext } from "./widget-parent-message"

function stubParent() {
  const postMessage = vi.fn()
  vi.stubGlobal("parent", { postMessage })
  return postMessage
}

function reply(
  values: unknown,
  envelope: Record<string, unknown> = {},
  source: MessageEventSource = window.parent as unknown as MessageEventSource,
) {
  window.dispatchEvent(new MessageEvent("message", {
    data: { xagent: true, v: 1, type: "prompt_context", values, ...envelope },
    source,
  }))
}

describe("requestPromptContext", () => {
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it("resolves with values replied by the parent", async () => {
    const postMessage = stubParent()

    const pending = requestPromptContext()
    expect(postMessage).toHaveBeenCalledWith(
      { xagent: true, v: 1, type: "prompt_context_request" },
      "*",
    )
    reply({ "product.sku": "A-100" })

    await expect(pending).resolves.toEqual({ "product.sku": "A-100" })
  })

  it("ignores other sources and wrongly enveloped replies", async () => {
    stubParent()

    const pending = requestPromptContext()
    reply({ "product.sku": "WRONG-SOURCE" }, {}, window as unknown as MessageEventSource)
    reply({ "product.sku": "A-100" }, { xagent: false })
    reply({ "product.sku": "A-100" }, { v: 2 })
    reply({ "product.sku": "A-100" }, { type: "widget_expand_rejected" })
    reply({ "product.sku": "A-100" })

    await expect(pending).resolves.toEqual({ "product.sku": "A-100" })
  })

  it("resolves null for non-object values", async () => {
    stubParent()

    const pending = requestPromptContext()
    reply(["A-100"])

    await expect(pending).resolves.toBeNull()
  })

  it("resolves null for non-plain objects", async () => {
    stubParent()

    const pending = requestPromptContext()
    reply(new Date(0))

    await expect(pending).resolves.toBeNull()
  })

  it("clears its timer and listener once the parent replies", async () => {
    vi.useFakeTimers()
    stubParent()
    const removeListener = vi.spyOn(window, "removeEventListener")

    const pending = requestPromptContext(500)
    reply({ "product.sku": "A-100" })

    await expect(pending).resolves.toEqual({ "product.sku": "A-100" })
    expect(vi.getTimerCount()).toBe(0)
    expect(removeListener).toHaveBeenCalledWith("message", expect.any(Function))
  })

  it("resolves null when the parent replies with null values", async () => {
    stubParent()

    const pending = requestPromptContext()
    reply(null)

    await expect(pending).resolves.toBeNull()
  })

  it("resolves null after the timeout and removes its listener", async () => {
    vi.useFakeTimers()
    stubParent()
    const addListener = vi.spyOn(window, "addEventListener")
    const removeListener = vi.spyOn(window, "removeEventListener")

    const pending = requestPromptContext(500)
    const handler = addListener.mock.calls.find(([type]) => type === "message")![1]

    await vi.advanceTimersByTimeAsync(500)

    await expect(pending).resolves.toBeNull()
    expect(removeListener).toHaveBeenCalledWith("message", handler)
  })

  it("resolves null without posting on a direct visit", async () => {
    vi.stubGlobal("parent", window)
    const postMessage = vi.spyOn(window, "postMessage")

    await expect(requestPromptContext()).resolves.toBeNull()
    expect(postMessage).not.toHaveBeenCalled()
  })
})

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

type Listener = (event: Event) => void

function makeScope() {
  const listeners = new Map<string, Listener[]>()
  const posted: unknown[] = []
  const scope = {
    addEventListener: (type: string, listener: Listener) => {
      listeners.set(type, [...(listeners.get(type) ?? []), listener])
    },
    postMessage: (message: unknown) => {
      posted.push(JSON.parse(JSON.stringify(message)))
    },
  }
  function dispatch(type: string, event: Event) {
    for (const listener of listeners.get(type) ?? []) listener(event)
  }
  return { scope, posted, dispatch, listeners }
}

async function loadEntry() {
  vi.resetModules()
  await import("./link-session-worker")
}

describe("link session Worker entry", () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it("posts hello as soon as the script has run", async () => {
    const { scope, posted } = makeScope()
    vi.stubGlobal("self", scope)
    await loadEntry()
    expect(posted).toEqual([{ type: "hello" }])
  })

  it("marks error and unhandledrejection events as handled", async () => {
    const { scope, dispatch } = makeScope()
    vi.stubGlobal("self", scope)
    await loadEntry()
    for (const type of ["error", "unhandledrejection"]) {
      const event = new Event(type, { cancelable: true })
      dispatch(type, event)
      expect(event.defaultPrevented, type).toBe(true)
    }
  })

  it("routes a start message to a same-origin redeem request and never posts the reconnect token", async () => {
    const { scope, posted, dispatch } = makeScope()
    vi.stubGlobal("self", scope)
    const now = Date.now()
    const fetchSpy = vi.fn(() => Promise.resolve(new Response(JSON.stringify({
      session_token: "st_value",
      session_token_expires_at: new Date(now + 15 * 60_000).toISOString(),
      reconnect_token: "rt_SECRET_ENTRY_VALUE",
      session: {
        absolute_expires_at: new Date(now + 8 * 3_600_000).toISOString(),
        agent: { id: 1 },
      },
    }), { status: 200 })))
    vi.stubGlobal("fetch", fetchSpy)
    await loadEntry()
    dispatch("message", new MessageEvent("message", { data: { type: "start", code: `lk_${"B".repeat(43)}` } }))
    await vi.advanceTimersByTimeAsync(0)
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect((fetchSpy.mock.calls[0] as unknown[])[0]).toBe("/v1/external/chat/links/redeem")
    expect(posted.map((m) => (m as { type: string }).type)).toEqual(["hello", "redeeming", "session"])
    expect(JSON.stringify(posted)).not.toContain("rt_SECRET_ENTRY_VALUE")
    expect(JSON.stringify(posted)).not.toContain("reconnect_token")
  })

  it("registers exactly the listeners it needs", async () => {
    const { scope, listeners } = makeScope()
    vi.stubGlobal("self", scope)
    await loadEntry()
    expect([...listeners.keys()].sort()).toEqual(["error", "message", "unhandledrejection"])
  })
})

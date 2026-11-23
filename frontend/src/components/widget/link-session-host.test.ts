import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  buildIframeSrc,
  createLinkSessionHost,
  extractLinkCode,
  type LinkHostSnapshot,
  type LinkSessionHost,
  type LinkWorkerLike,
} from "./link-session-host"

const CODE = `lk_${"C".repeat(43)}`
const ORIGIN = "https://app.example.test"
const MINUTE = 60_000
const HOUR = 60 * MINUTE

type WorkerEventType = "message" | "error" | "messageerror"

class FakeWorker implements LinkWorkerLike {
  posted: Array<Record<string, unknown>> = []
  terminated = false
  private listeners = new Map<string, Array<(event: { data?: unknown; preventDefault?: () => void }) => void>>()

  constructor(private readonly log: string[]) {}

  postMessage(message: unknown) {
    const typed = message as Record<string, unknown>
    this.log.push(`worker.post:${String(typed.type)}`)
    this.posted.push(typed)
  }

  terminate() {
    this.terminated = true
    this.log.push("worker.terminate")
  }

  addEventListener(type: WorkerEventType, listener: (event: { data?: unknown; preventDefault?: () => void }) => void) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener])
  }

  emit(type: WorkerEventType, data?: unknown) {
    for (const listener of this.listeners.get(type) ?? []) listener({ data, preventDefault: () => undefined })
  }

  count(type: string) {
    return this.posted.filter((m) => m.type === type).length
  }
}

function sessionFields(overrides: Record<string, unknown> = {}) {
  const now = Date.now()
  return {
    session_token: "st_value",
    session_token_expires_at: new Date(now + 15 * MINUTE).toISOString(),
    absolute_expires_at: new Date(now + 8 * HOUR).toISOString(),
    agent: { id: 3, name: "Agent" },
    capabilities: { uploads: true },
    ...overrides,
  }
}

interface Options {
  hash?: string
  search?: string
  throwOnCreate?: boolean
}

interface Harness {
  host: LinkSessionHost
  log: string[]
  navs: string[]
  iframePosts: Array<Record<string, unknown>>
  snapshots: LinkHostSnapshot[]
  // Every delay the host asked its timer for.
  timerDelays: number[]
  iframeWindow: object
  worker: () => FakeWorker
}

function makeHost(options: Options = {}): Harness {
  const log: string[] = []
  const navs: string[] = []
  const iframePosts: Array<Record<string, unknown>> = []
  const snapshots: LinkHostSnapshot[] = []
  const timerDelays: number[] = []
  const iframeWindow = {}
  let created: FakeWorker | null = null
  const host = createLinkSessionHost({
    readHash: () => options.hash ?? `#code=${CODE}`,
    readPathname: () => "/widget/chat/link",
    readSearch: () => options.search ?? "",
    readOrigin: () => ORIGIN,
    replaceState: (url) => log.push(`replaceState:${url}`),
    navigate: (url) => {
      log.push(`navigate:${url}`)
      navs.push(url)
    },
    createWorker: () => {
      log.push("createWorker")
      if (options.throwOnCreate) throw new Error("Worker is not available")
      created = new FakeWorker(log)
      return created
    },
    getIframeWindow: () => iframeWindow,
    postToIframe: (message) => iframePosts.push(message),
    now: () => Date.now(),
    setTimer: (callback, ms) => {
      timerDelays.push(ms)
      return setTimeout(callback, ms)
    },
    clearTimer: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
    onChange: (snapshot) => snapshots.push(snapshot),
  })
  return {
    host,
    log,
    navs,
    iframePosts,
    snapshots,
    timerDelays,
    iframeWindow,
    worker: () => {
      if (!created) throw new Error("no worker was created")
      return created
    },
  }
}

function iframeSays(h: Harness, type: string, extra: Record<string, unknown> = {}) {
  h.host.handleMessageEvent({
    source: h.iframeWindow,
    origin: ORIGIN,
    data: { xagent: true, v: 1, type, ...extra },
  })
}

function connect(h: Harness, fields: Record<string, unknown> = {}) {
  h.host.start()
  const worker = h.worker()
  worker.emit("message", { type: "hello" })
  worker.emit("message", { type: "redeeming" })
  worker.emit("message", { type: "session", session: sessionFields(fields) })
  return worker
}

function lastUpdate(h: Harness) {
  return [...h.iframePosts].reverse().find((m) => m.type === "session_update")
}

function updates(h: Harness) {
  return h.iframePosts.filter((m) => m.type === "session_update")
}

function renewCycle(h: Harness, worker: FakeWorker) {
  iframeSays(h, "reconnect_request")
  worker.emit("message", { type: "session", session: sessionFields() })
}

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe("link session host startup", () => {
  it("erases the fragment before it creates the Worker or sends the code", () => {
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)
    const h = makeHost({ search: "?timezone=Australia%2FSydney" })
    h.host.start()
    expect(h.log).toEqual([
      "replaceState:/widget/chat/link?timezone=Australia%2FSydney",
      "createWorker",
      "worker.post:start",
    ])
    expect(h.worker().posted[0]).toEqual({ type: "start", code: CODE })
    expect(extractLinkCode(`#x=1&code=${CODE}`)).toBe(CODE)
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it.each([
    ["no fragment", ""],
    ["a fragment without a code", "#other=1"],
    ["a code of the wrong shape", "#code=lk_short"],
    ["a code with the wrong prefix", `#code=rt_${"C".repeat(43)}`],
    ["an empty code", "#code="],
  ])("still erases the fragment and ends with link_invalid for %s", (_label, hash) => {
    expect(extractLinkCode(hash)).toBeNull()
    const h = makeHost({ hash })
    h.host.start()
    expect(h.log).toEqual([
      "replaceState:/widget/chat/link",
      "navigate:/widget/chat/link/ended?reason=link_invalid",
    ])
  })

  it("ends with unavailable when the Worker cannot be created", () => {
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)
    const h = makeHost({ throwOnCreate: true })
    h.host.start()
    expect(h.navs).toEqual(["/widget/chat/link/ended?reason=unavailable"])
    expect(fetchSpy).not.toHaveBeenCalled()
  })

})

describe("link session host state", () => {
  it("has a fixed key set that contains no link code and no reconnect token", () => {
    const h = makeHost()
    connect(h)
    const state = h.host.getState()
    expect(Object.keys(state).sort()).toEqual([
      "awaitedDeliveryId",
      "chromeReady",
      "currentDeliveryId",
      "deliverySequence",
      "iframeReady",
      "invalidatedDeliveryId",
      "phase",
      "rapidRecoveries",
      "recoverableCode",
      "renewInFlight",
      "session",
      "stableDeliveryId",
    ])
    const serialized = JSON.stringify(state)
    expect(serialized).not.toContain(CODE)
    expect(serialized).not.toContain("reconnect_token")
  })

  it("reports no iframe source until the first session arrives", () => {
    const h = makeHost()
    h.host.start()
    const worker = h.worker()
    worker.emit("message", { type: "hello" })
    worker.emit("message", { type: "redeeming" })
    expect(h.snapshots.every((s) => s.iframeSrc === null)).toBe(true)
    worker.emit("message", { type: "session", session: sessionFields() })
    expect(h.snapshots[h.snapshots.length - 1].iframeSrc).toBe("/widget/chat/session")
    expect(h.host.getState().phase).toBe("active")
  })
})

describe("link session host iframe source", () => {
  it("carries the timezone from the page query, encoded like widget.js", () => {
    expect(buildIframeSrc("?timezone=Australia%2FSydney")).toBe("/widget/chat/session?timezone=Australia%2FSydney")
    expect(buildIframeSrc("?timezone=Australia/Sydney")).toBe("/widget/chat/session?timezone=Australia%2FSydney")
    expect(buildIframeSrc("?timezone=%20Europe%2FParis%20")).toBe("/widget/chat/session?timezone=Europe%2FParis")
  })

  it("leaves the source plain without a usable timezone and never adds voice", () => {
    for (const search of ["", "?timezone=", "?timezone=%20", "?voice=1", "?other=x"]) {
      expect(buildIframeSrc(search)).toBe("/widget/chat/session")
    }
    expect(buildIframeSrc("?voice=1&timezone=UTC")).not.toContain("voice")
  })

})

describe("link session host end reasons", () => {
  it.each(["link_invalid", "session_ended", "unavailable", "network"])(
    "navigates to the end page for a Worker terminal of %s",
    (reason) => {
      const h = makeHost()
      const worker = connect(h)
      worker.emit("message", { type: "terminal", reason })
      expect(h.navs).toEqual([`/widget/chat/link/ended?reason=${reason}`])
      expect(worker.terminated).toBe(true)
    },
  )

  it("maps an unknown Worker reason to unavailable", () => {
    const h = makeHost()
    h.host.start()
    h.worker().emit("message", { type: "terminal", reason: "closed" })
    expect(h.navs).toEqual(["/widget/chat/link/ended?reason=unavailable"])
  })

  it("ends with unavailable when the Worker sends a malformed session", () => {
    const h = makeHost()
    h.host.start()
    h.worker().emit("message", { type: "session", session: { session_token: "x" } })
    expect(h.navs).toEqual(["/widget/chat/link/ended?reason=unavailable"])
  })

  it("navigates exactly once even if more terminations arrive", () => {
    const h = makeHost()
    const worker = connect(h)
    worker.emit("message", { type: "terminal", reason: "network" })
    worker.emit("message", { type: "terminal", reason: "session_ended" })
    h.host.close()
    expect(h.navs).toEqual(["/widget/chat/link/ended?reason=network"])
  })

  it("ends with closed when the close button is used", () => {
    const h = makeHost()
    connect(h)
    h.host.close()
    expect(h.navs).toEqual(["/widget/chat/link/ended?reason=closed"])
  })

  it("ends with session_ended at the absolute expiry and not before", () => {
    const h = makeHost()
    h.host.start()
    const w = h.worker()
    const absolute = new Date(Date.now() + 8 * HOUR).toISOString()
    w.emit("message", { type: "hello" })
    w.emit("message", { type: "redeeming" })
    w.emit("message", { type: "session", session: sessionFields({ absolute_expires_at: absolute }) })
    vi.advanceTimersByTime(8 * HOUR - 1)
    expect(h.navs).toEqual([])
    vi.advanceTimersByTime(1)
    expect(h.navs).toEqual(["/widget/chat/link/ended?reason=session_ended"])
  })

  it("ends at an absolute expiry beyond the timer limit, and not before", () => {
    const h = makeHost()
    const thirtyDays = 30 * 24 * HOUR
    const maxTimer = 2 ** 31 - 1
    connect(h, { absolute_expires_at: new Date(Date.now() + thirtyDays).toISOString() })
    expect(Math.max(...h.timerDelays)).toBeLessThanOrEqual(maxTimer)
    vi.advanceTimersByTime(maxTimer)
    expect(h.navs).toEqual([])
    vi.advanceTimersByTime(thirtyDays - maxTimer - 1)
    expect(h.navs).toEqual([])
    vi.advanceTimersByTime(1)
    expect(h.navs).toEqual(["/widget/chat/link/ended?reason=session_ended"])
  })

  it("restarts the expiry timer when a renewed session carries a different absolute expiry", () => {
    const h = makeHost()
    const worker = connect(h)
    vi.advanceTimersByTime(HOUR)
    worker.emit("message", { type: "session", session: sessionFields({ absolute_expires_at: new Date(Date.now() + 2 * HOUR).toISOString() }) })
    vi.advanceTimersByTime(2 * HOUR - 1)
    expect(h.navs).toEqual([])
    vi.advanceTimersByTime(1)
    expect(h.navs).toEqual(["/widget/chat/link/ended?reason=session_ended"])
  })
})

describe("link session host page cache restore", () => {
  it("terminates the Worker and ends the session on a persisted pageshow", () => {
    const h = makeHost()
    const worker = connect(h)
    h.host.handlePageShow({ persisted: true })
    expect(worker.terminated).toBe(true)
    expect(h.navs).toEqual(["/widget/chat/link/ended?reason=session_ended"])
  })

  it("does nothing on a pageshow that is not a restore", () => {
    const h = makeHost()
    const worker = connect(h)
    h.host.handlePageShow({ persisted: false })
    expect(worker.terminated).toBe(false)
    expect(h.navs).toEqual([])
  })

  it("ends only once when a restore arrives after the session already ended", () => {
    const h = makeHost()
    const worker = connect(h)
    worker.emit("message", { type: "terminal", reason: "network" })
    h.host.handlePageShow({ persisted: true })
    expect(h.navs).toEqual(["/widget/chat/link/ended?reason=network"])
  })
})

describe("link session host fails closed when the Worker cannot run", () => {
  it("ends with unavailable after 30 s without hello", () => {
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)
    const h = makeHost()
    h.host.start()
    vi.advanceTimersByTime(29_999)
    expect(h.navs).toEqual([])
    vi.advanceTimersByTime(1)
    expect(h.navs).toEqual(["/widget/chat/link/ended?reason=unavailable"])
    expect(h.worker().terminated).toBe(true)
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it("ends with unavailable 10 s after hello without redeeming", () => {
    const h = makeHost()
    h.host.start()
    h.worker().emit("message", { type: "hello" })
    vi.advanceTimersByTime(9_999)
    expect(h.navs).toEqual([])
    vi.advanceTimersByTime(1)
    expect(h.navs).toEqual(["/widget/chat/link/ended?reason=unavailable"])
  })

  it("does not apply the 30 s deadline after hello arrived", () => {
    const h = makeHost()
    h.host.start()
    h.worker().emit("message", { type: "hello" })
    h.worker().emit("message", { type: "redeeming" })
    vi.advanceTimersByTime(10 * MINUTE)
    expect(h.navs).toEqual([])
  })

  it("lets a redeem that already started take 14 s, then builds the iframe", () => {
    const h = makeHost()
    h.host.start()
    const worker = h.worker()
    worker.emit("message", { type: "hello" })
    worker.emit("message", { type: "redeeming" })
    vi.advanceTimersByTime(14_000)
    expect(h.navs).toEqual([])
    worker.emit("message", { type: "session", session: sessionFields() })
    vi.advanceTimersByTime(MINUTE)
    expect(h.navs).toEqual([])
    expect(h.snapshots[h.snapshots.length - 1].iframeSrc).toBe("/widget/chat/session")
  })

  it.each(["error", "messageerror"] as const)("a Worker %s before any session ends with unavailable", (type) => {
    const h = makeHost()
    h.host.start()
    h.worker().emit(type)
    expect(h.navs).toEqual(["/widget/chat/link/ended?reason=unavailable"])
  })

  it.each(["error", "messageerror"] as const)("a Worker %s after a session ends with session_ended", (type) => {
    const h = makeHost()
    const worker = connect(h)
    worker.emit(type)
    expect(h.navs).toEqual(["/widget/chat/link/ended?reason=session_ended"])
  })

  it("does not end the page while the Worker backs off silently and a renew is requested", () => {
    const fetchSpy = vi.fn()
    vi.stubGlobal("fetch", fetchSpy)
    const h = makeHost()
    const worker = connect(h)
    iframeSays(h, "ready")
    iframeSays(h, "reconnect_request")
    worker.emit("message", { type: "degraded", code: "network_unavailable" })
    vi.advanceTimersByTime(5 * MINUTE)
    iframeSays(h, "reconnect_request")
    expect(h.navs).toEqual([])
    expect(worker.count("renew")).toBe(1)
    expect(fetchSpy).not.toHaveBeenCalled()
  })
})

describe("link session host iframe protocol", () => {
  it("answers ready with a session update that carries only the session fields and a delivery id", () => {
    const h = makeHost()
    connect(h)
    iframeSays(h, "ready")
    const update = lastUpdate(h)
    expect(update).toBeDefined()
    expect(Object.keys(update as object).sort()).toEqual([
      "absolute_expires_at",
      "agent",
      "capabilities",
      "session_delivery_id",
      "session_token",
      "session_token_expires_at",
      "type",
      "v",
      "xagent",
    ])
    expect(update).toMatchObject({ xagent: true, v: 1, session_token: "st_value" })
    expect(typeof (update as { session_delivery_id: unknown }).session_delivery_id).toBe("string")
  })

  it("re-sends the same delivery id on a repeated ready and mints a new one for each new session", () => {
    const h = makeHost()
    const worker = connect(h)
    iframeSays(h, "ready")
    iframeSays(h, "ready")
    const ids = updates(h).map((m) => m.session_delivery_id)
    expect(ids[0]).toBe(ids[1])
    renewCycle(h, worker)
    const next = (lastUpdate(h) as { session_delivery_id: string }).session_delivery_id
    expect(next).not.toBe(ids[0])
  })

  it("leaves capabilities out of the update when the session has none", () => {
    const h = makeHost()
    h.host.start()
    const worker = h.worker()
    worker.emit("message", { type: "hello" })
    worker.emit("message", { type: "redeeming" })
    const fields: Record<string, unknown> = sessionFields()
    delete fields.capabilities
    worker.emit("message", { type: "session", session: fields })
    iframeSays(h, "ready")
    expect(lastUpdate(h)).not.toHaveProperty("capabilities")
  })

  it("renews first when ready arrives with a token that expires within 60 s, and sends the update afterwards", () => {
    const h = makeHost()
    const worker = connect(h, { session_token_expires_at: new Date(Date.now() + 30_000).toISOString() })
    iframeSays(h, "ready")
    expect(worker.count("renew")).toBe(1)
    expect(updates(h)).toHaveLength(0)
    iframeSays(h, "ready")
    expect(worker.count("renew")).toBe(1)
    worker.emit("message", { type: "session", session: sessionFields() })
    expect(updates(h)).toHaveLength(1)
  })

  it("sends a session update when a renewed session arrives after ready", () => {
    const h = makeHost()
    const worker = connect(h)
    iframeSays(h, "ready")
    iframeSays(h, "reconnect_request")
    expect(updates(h)).toHaveLength(1)
    worker.emit("message", { type: "session", session: sessionFields({ session_token: "st_second" }) })
    expect(updates(h)).toHaveLength(2)
    expect(lastUpdate(h)).toMatchObject({ session_token: "st_second" })
  })

  it("does not send an update after the session has ended", () => {
    const h = makeHost()
    const worker = connect(h)
    worker.emit("message", { type: "terminal", reason: "network" })
    iframeSays(h, "ready")
    expect(h.iframePosts).toEqual([])
  })

  it("forwards degraded to the iframe and repeats it for a late ready", () => {
    const h = makeHost()
    const worker = connect(h)
    iframeSays(h, "ready")
    worker.emit("message", { type: "degraded", code: "network_unavailable" })
    expect(h.iframePosts[h.iframePosts.length - 1]).toEqual({
      xagent: true,
      v: 1,
      type: "session_degraded",
      code: "network_unavailable",
    })
    const before = h.iframePosts.length
    iframeSays(h, "ready")
    expect(h.iframePosts.length).toBe(before + 1)
    expect(h.iframePosts[h.iframePosts.length - 1]).toMatchObject({ type: "session_degraded" })
    worker.emit("message", { type: "session", session: sessionFields() })
    expect(h.iframePosts[h.iframePosts.length - 1]).toMatchObject({ type: "session_update" })
  })

  it("forwards a rate_limited degraded message to the iframe", () => {
    const h = makeHost()
    const worker = connect(h)
    iframeSays(h, "ready")
    worker.emit("message", { type: "degraded", code: "rate_limited" })
    expect(h.iframePosts[h.iframePosts.length - 1]).toEqual({
      xagent: true,
      v: 1,
      type: "session_degraded",
      code: "rate_limited",
    })
    worker.emit("message", { type: "session", session: sessionFields() })
    expect(h.iframePosts[h.iframePosts.length - 1]).toMatchObject({ type: "session_update" })
  })

  it("ignores a degraded message with an unexpected code", () => {
    const h = makeHost()
    const worker = connect(h)
    iframeSays(h, "ready")
    const before = h.iframePosts.length
    worker.emit("message", { type: "degraded", code: "other" })
    expect(h.iframePosts.length).toBe(before)
  })

  it.each([
    ["ws_4403", { code: "ws_4403" }, "session_ended"],
    ["ws_4408", { code: "ws_4408" }, "session_ended"],
    ["session_expired", { code: "session_expired" }, "session_ended"],
    ["unexpected_error", { code: "unexpected_error" }, "unavailable"],
    ["an unknown code", { code: "future_code" }, "unavailable"],
    ["a code that looks like a query string", { code: "session_ended&x=1" }, "unavailable"],
    ["a numeric code", { code: 5 }, "unavailable"],
    ["no code", {}, "unavailable"],
  ])("maps the chat page's terminal report (%s) to its end reason", (_label, extra, reason) => {
    const h = makeHost()
    const worker = connect(h)
    iframeSays(h, "session_terminal_reached", extra)
    expect(h.navs).toEqual([`/widget/chat/link/ended?reason=${reason}`])
    expect(worker.terminated).toBe(true)
  })

  it("ends with closed on widget_close", () => {
    const h = makeHost()
    connect(h)
    iframeSays(h, "widget_close")
    expect(h.navs).toEqual(["/widget/chat/link/ended?reason=closed"])
  })

  it("rejects widget_expand", () => {
    const h = makeHost()
    connect(h)
    iframeSays(h, "widget_expand")
    expect(h.iframePosts).toEqual([{ xagent: true, v: 1, type: "widget_expand_rejected" }])
  })

  it.each([
    ["before a session", (h: Harness) => { h.host.start(); h.host.close() }],
    ["after a session", (h: Harness) => connect(h).emit("message", { type: "terminal", reason: "network" })],
  ])("removes the close button once the session has ended (%s)", (_name, end) => {
    const h = makeHost()
    end(h)
    expect(h.snapshots[h.snapshots.length - 1]).toEqual({ phase: "terminated", iframeSrc: null, closeButtonVisible: false })
  })

  it("answers widget_chrome_ready with a collapsed state and hides the page's own close button", () => {
    const h = makeHost()
    connect(h)
    expect(h.snapshots[h.snapshots.length - 1].closeButtonVisible).toBe(true)
    iframeSays(h, "widget_chrome_ready")
    expect(h.iframePosts).toEqual([{ xagent: true, v: 1, type: "widget_expand_state", expanded: false }])
    expect(h.snapshots[h.snapshots.length - 1].closeButtonVisible).toBe(false)
    iframeSays(h, "widget_chrome_not_ready")
    expect(h.snapshots[h.snapshots.length - 1].closeButtonVisible).toBe(true)
  })

  it.each(["widget_collapse", "initial_message_request", "open_message_request", "prompt_context_request", "something_else"])(
    "ignores %s",
    (type) => {
      const h = makeHost()
      connect(h)
      iframeSays(h, type)
      expect(h.iframePosts).toEqual([])
      expect(h.navs).toEqual([])
    },
  )

  it.each([
    ["another window", { source: {}, origin: ORIGIN }],
    ["another origin", { source: "iframe", origin: "https://evil.example" }],
    ["a null source", { source: null, origin: ORIGIN }],
  ])("ignores messages from %s", (_label, who) => {
    const h = makeHost()
    const worker = connect(h)
    for (const type of [
      "ready",
      "reconnect_request",
      "widget_close",
      "widget_chrome_ready",
      "session_terminal_reached",
    ]) {
      h.host.handleMessageEvent({
        source: who.source === "iframe" ? h.iframeWindow : who.source,
        origin: who.origin,
        data: { xagent: true, v: 1, type },
      })
    }
    expect(h.iframePosts).toEqual([])
    expect(h.navs).toEqual([])
    expect(worker.count("renew")).toBe(0)
    expect(h.snapshots[h.snapshots.length - 1].closeButtonVisible).toBe(true)
  })

  it.each([
    ["without the xagent marker", { v: 1, type: "widget_close" }],
    ["with another version", { xagent: true, v: 2, type: "widget_close" }],
    ["with a non-string type", { xagent: true, v: 1, type: 5 }],
    ["as a string", "widget_close"],
    ["as null", null],
  ])("ignores a message %s", (_label, data) => {
    const h = makeHost()
    connect(h)
    h.host.handleMessageEvent({ source: h.iframeWindow, origin: ORIGIN, data })
    expect(h.navs).toEqual([])
  })

  it("forwards wake to the Worker while it is running and not afterwards", () => {
    const h = makeHost()
    const worker = connect(h)
    h.host.wake()
    expect(worker.count("wake")).toBe(1)
    worker.emit("message", { type: "terminal", reason: "network" })
    h.host.wake()
    expect(worker.count("wake")).toBe(1)
  })

  it("stops listening after dispose and does not navigate", () => {
    const h = makeHost()
    const worker = connect(h)
    h.host.dispose()
    expect(worker.terminated).toBe(true)
    worker.emit("message", { type: "terminal", reason: "network" })
    vi.advanceTimersByTime(9 * HOUR)
    expect(h.navs).toEqual([])
  })
})

describe("link session host renew accounting", () => {
  it("does not count the first redeem", () => {
    const h = makeHost()
    connect(h)
    expect(h.host.getState().rapidRecoveries).toBe(0)
  })

  it("merges a reconnect request into a renew that is already running, without counting it", () => {
    const h = makeHost()
    const worker = connect(h)
    iframeSays(h, "ready")
    iframeSays(h, "reconnect_request")
    iframeSays(h, "reconnect_request")
    iframeSays(h, "reconnect_request")
    expect(worker.count("renew")).toBe(1)
    expect(h.host.getState().rapidRecoveries).toBe(1)
    worker.emit("message", { type: "session", session: sessionFields() })
    expect(h.host.getState().renewInFlight).toBe(false)
  })

  it("ends with network instead of sending the fourth renew inside the stability window", () => {
    const h = makeHost()
    const worker = connect(h)
    iframeSays(h, "ready")
    renewCycle(h, worker)
    renewCycle(h, worker)
    renewCycle(h, worker)
    expect(worker.count("renew")).toBe(3)
    iframeSays(h, "reconnect_request")
    expect(worker.count("renew")).toBe(3)
    expect(h.navs).toEqual(["/widget/chat/link/ended?reason=network"])
  })

  it("resets the count once a connection has stayed open for 60 s", () => {
    const h = makeHost()
    const worker = connect(h)
    iframeSays(h, "ready")
    for (let i = 0; i < 4; i += 1) {
      const id = (lastUpdate(h) as { session_delivery_id: string }).session_delivery_id
      iframeSays(h, "session_connection_open", { session_delivery_id: id })
      vi.advanceTimersByTime(60_000)
      renewCycle(h, worker)
    }
    expect(worker.count("renew")).toBe(4)
    expect(h.navs).toEqual([])
  })

  it("does not reset before 60 s", () => {
    const h = makeHost()
    const worker = connect(h)
    iframeSays(h, "ready")
    for (let i = 0; i < 3; i += 1) {
      const id = (lastUpdate(h) as { session_delivery_id: string }).session_delivery_id
      iframeSays(h, "session_connection_open", { session_delivery_id: id })
      vi.advanceTimersByTime(59_999)
      renewCycle(h, worker)
    }
    iframeSays(h, "reconnect_request")
    expect(h.navs).toEqual(["/widget/chat/link/ended?reason=network"])
  })

  it("voids an unfinished stability timer when a new count starts, so an old delivery cannot reset it", () => {
    const h = makeHost()
    const worker = connect(h)
    iframeSays(h, "ready")
    const firstId = (lastUpdate(h) as { session_delivery_id: string }).session_delivery_id
    iframeSays(h, "session_connection_open", { session_delivery_id: firstId })
    vi.advanceTimersByTime(30_000)
    renewCycle(h, worker)
    // The connection of the old delivery reports itself open again; its id was
    // invalidated when the renew started, so it must not start a stability timer.
    iframeSays(h, "session_connection_open", { session_delivery_id: firstId })
    vi.advanceTimersByTime(5 * MINUTE)
    renewCycle(h, worker)
    renewCycle(h, worker)
    iframeSays(h, "reconnect_request")
    expect(worker.count("renew")).toBe(3)
    expect(h.navs).toEqual(["/widget/chat/link/ended?reason=network"])
  })

  it("lets a repeated ready inside the stability window still reset the count", () => {
    const h = makeHost()
    const worker = connect(h)
    iframeSays(h, "ready")
    renewCycle(h, worker)
    renewCycle(h, worker)
    renewCycle(h, worker)
    expect(h.host.getState().rapidRecoveries).toBe(3)
    const id = (lastUpdate(h) as { session_delivery_id: string }).session_delivery_id
    iframeSays(h, "session_connection_open", { session_delivery_id: id })
    vi.advanceTimersByTime(10_000)
    iframeSays(h, "ready")
    const again = (lastUpdate(h) as { session_delivery_id: string }).session_delivery_id
    expect(again).toBe(id)
    iframeSays(h, "session_connection_open", { session_delivery_id: again })
    vi.advanceTimersByTime(HOUR)
    expect(h.host.getState().rapidRecoveries).toBe(0)
  })

  it("does not let a stability timer that started before a renew reset the count after it", () => {
    const h = makeHost()
    const worker = connect(h)
    iframeSays(h, "ready")
    const id = (lastUpdate(h) as { session_delivery_id: string }).session_delivery_id
    iframeSays(h, "session_connection_open", { session_delivery_id: id })
    vi.advanceTimersByTime(30_000)
    iframeSays(h, "reconnect_request")
    expect(h.host.getState().rapidRecoveries).toBe(1)
    worker.emit("message", { type: "degraded", code: "network_unavailable" })
    vi.advanceTimersByTime(40_000)
    worker.emit("message", { type: "session", session: sessionFields() })
    renewCycle(h, worker)
    renewCycle(h, worker)
    expect(worker.count("renew")).toBe(3)
    iframeSays(h, "reconnect_request")
    expect(worker.count("renew")).toBe(3)
    expect(h.navs).toEqual(["/widget/chat/link/ended?reason=network"])
  })

  it("ignores session_connection_open for an unknown delivery id or without one", () => {
    const h = makeHost()
    const worker = connect(h)
    iframeSays(h, "ready")
    iframeSays(h, "session_connection_open", { session_delivery_id: "unknown" })
    iframeSays(h, "session_connection_open")
    iframeSays(h, "session_connection_open", { session_delivery_id: 5 })
    vi.advanceTimersByTime(5 * MINUTE)
    renewCycle(h, worker)
    renewCycle(h, worker)
    renewCycle(h, worker)
    iframeSays(h, "reconnect_request")
    expect(h.navs).toEqual(["/widget/chat/link/ended?reason=network"])
  })

  it("does not send renew when no session exists yet", () => {
    const h = makeHost()
    h.host.start()
    iframeSays(h, "reconnect_request")
    expect(h.worker().count("renew")).toBe(0)
    expect(h.host.getState().rapidRecoveries).toBe(0)
  })
})

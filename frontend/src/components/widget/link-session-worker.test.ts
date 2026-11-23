import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { WorkerOutboundMessage } from "./link-session-constants"
import {
  createLinkSessionWorkerCore,
  type LinkSessionWorkerCore,
} from "./link-session-worker-core"

const CODE = `lk_${"A".repeat(43)}`
const REDEEM_URL = "/v1/external/chat/links/redeem"
const RENEW_URL = "/v1/external/chat/links/reconnect"
const MINUTE = 60_000
const HOUR = 60 * MINUTE

type FetchHandler = (call: { url: string; body: Record<string, unknown>; init: RequestInit; n: number }) =>
  Promise<Response> | Response

interface Harness {
  core: LinkSessionWorkerCore
  posts: WorkerOutboundMessage[]
  log: string[]
  fetchSpy: ReturnType<typeof vi.fn>
  fetchTimes: number[]
  fetchBodies: Record<string, unknown>[]
  setHandler: (handler: FetchHandler) => void
}

function jsonResponse(status: number, body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(typeof body === "string" ? body : JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  })
}

function tokenResponse(reconnectToken: string, overrides: Record<string, unknown> = {}, withCapabilities = true) {
  const now = Date.now()
  return {
    session_token: "st_value",
    session_token_expires_at: new Date(now + 15 * MINUTE).toISOString(),
    reconnect_token: reconnectToken,
    session: {
      absolute_expires_at: new Date(now + 8 * HOUR).toISOString(),
      agent: { id: 7, name: "Agent" },
      ...(withCapabilities ? { capabilities: { uploads: true } } : {}),
      extra_field: "must not be forwarded",
    },
    top_level_extra: "must not be forwarded",
    ...overrides,
  }
}

function errorResponse(status: number, code: string, headers: Record<string, string> = {}) {
  return jsonResponse(status, { error: { code, message: "fixed" } }, headers)
}

// A fetch that never answers on its own and rejects when its signal aborts.
function hangingFetch(init: RequestInit): Promise<Response> {
  return new Promise((_resolve, reject) => {
    init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))
  })
}

function makeHarness(handler: FetchHandler): Harness {
  let currentHandler = handler
  const posts: WorkerOutboundMessage[] = []
  const log: string[] = []
  const fetchTimes: number[] = []
  const fetchBodies: Record<string, unknown>[] = []
  let n = 0
  const fetchSpy = vi.fn((url: string, init: RequestInit) => {
    n += 1
    log.push("fetch")
    fetchTimes.push(Date.now())
    const body = JSON.parse(String(init.body)) as Record<string, unknown>
    fetchBodies.push(body)
    return Promise.resolve(currentHandler({ url, body, init, n }))
  })
  const core = createLinkSessionWorkerCore({
    fetch: fetchSpy as unknown as (input: string, init: RequestInit) => Promise<Response>,
    post: (message) => {
      log.push(message.type)
      posts.push(JSON.parse(JSON.stringify(message)) as WorkerOutboundMessage)
    },
    now: () => Date.now(),
    random: () => 0.5,
    setTimer: (callback, ms) => setTimeout(callback, ms),
    clearTimer: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
  })
  return { core, posts, log, fetchSpy, fetchTimes, fetchBodies, setHandler: (h) => { currentHandler = h } }
}

async function settle() {
  await vi.advanceTimersByTimeAsync(0)
}

async function startAndSettle(h: Harness) {
  h.core.handleMessage({ type: "start", code: CODE })
  await settle()
}

function types(h: Harness) {
  return h.posts.map((p) => p.type)
}

function terminals(h: Harness) {
  return h.posts.filter((p) => p.type === "terminal")
}

async function advanceUntilFetchCount(h: Harness, count: number, stepMs = 500, maxMs = 60 * MINUTE) {
  let waited = 0
  while (h.fetchSpy.mock.calls.length < count && waited < maxMs) {
    await vi.advanceTimersByTimeAsync(stepMs)
    waited += stepMs
  }
}

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

describe("link session Worker command rules", () => {
  it("announces itself before anything else happens", () => {
    const h = makeHarness(() => jsonResponse(200, tokenResponse("rt_1")))
    h.core.announce()
    expect(h.posts).toEqual([{ type: "hello" }])
    expect(h.fetchSpy).not.toHaveBeenCalled()
  })

  it("posts redeeming before it sends the redeem request", async () => {
    const h = makeHarness(() => jsonResponse(200, tokenResponse("rt_1")))
    await startAndSettle(h)
    expect(h.log.slice(0, 2)).toEqual(["redeeming", "fetch"])
  })

  it("sends the redeem request to the same-origin endpoint without credentials", async () => {
    const h = makeHarness(() => jsonResponse(200, tokenResponse("rt_1")))
    await startAndSettle(h)
    const [url, init] = h.fetchSpy.mock.calls[0] as [string, RequestInit]
    expect(url).toBe(REDEEM_URL)
    expect(init.method).toBe("POST")
    expect(init.credentials).toBe("omit")
    expect(JSON.parse(String(init.body))).toEqual({ code: CODE })
  })

  it("ignores a second start", async () => {
    const h = makeHarness(() => jsonResponse(200, tokenResponse("rt_1")))
    await startAndSettle(h)
    h.core.handleMessage({ type: "start", code: CODE })
    await settle()
    expect(h.fetchSpy).toHaveBeenCalledTimes(1)
  })

  it.each([
    ["an unknown type", { type: "reconnect_token" }],
    ["a message without a type", { code: CODE }],
    ["a string", "start"],
    ["null", null],
    ["an array", [{ type: "start", code: CODE }]],
  ])("ignores %s", async (_label, message) => {
    const h = makeHarness(() => jsonResponse(200, tokenResponse("rt_1")))
    h.core.handleMessage(message)
    await settle()
    expect(h.posts).toEqual([])
    expect(h.fetchSpy).not.toHaveBeenCalled()
  })

  it("ignores renew when there is no session yet", async () => {
    const h = makeHarness(() => jsonResponse(200, tokenResponse("rt_1")))
    h.core.handleMessage({ type: "renew" })
    await settle()
    expect(h.fetchSpy).not.toHaveBeenCalled()
    expect(h.posts).toEqual([])
  })

  it("ignores renew while the first redeem is still in flight", async () => {
    const h = makeHarness(({ init }) => hangingFetch(init))
    await startAndSettle(h)
    h.core.handleMessage({ type: "renew" })
    await settle()
    expect(h.fetchSpy).toHaveBeenCalledTimes(1)
  })

  it.each([
    ["a malformed shape", "lk_short"],
    ["a wrong prefix", `rt_${"A".repeat(43)}`],
    ["a non-string value", 12345],
  ])("ends with link_invalid for %s of the code without any request", async (_label, code) => {
    const h = makeHarness(() => jsonResponse(200, tokenResponse("rt_1")))
    h.core.handleMessage({ type: "start", code })
    await settle()
    expect(h.fetchSpy).not.toHaveBeenCalled()
    expect(terminals(h)).toEqual([{ type: "terminal", reason: "link_invalid" }])
  })

  it("ignores every message after a terminal", async () => {
    const h = makeHarness(() => errorResponse(401, "link_invalid"))
    await startAndSettle(h)
    const before = h.posts.length
    h.core.handleMessage({ type: "start", code: CODE })
    h.core.handleMessage({ type: "renew" })
    h.core.handleMessage({ type: "wake" })
    await settle()
    expect(h.posts.length).toBe(before)
    expect(h.fetchSpy).toHaveBeenCalledTimes(1)
  })
})

describe("link session Worker redeem", () => {
  it("posts only the whitelisted fields, including capabilities when present", async () => {
    const h = makeHarness(() => jsonResponse(200, tokenResponse("rt_1")))
    await startAndSettle(h)
    const session = h.posts.find((p) => p.type === "session")
    expect(session).toBeDefined()
    if (session?.type !== "session") throw new Error("unreachable")
    expect(Object.keys(session.session).sort()).toEqual([
      "absolute_expires_at",
      "agent",
      "capabilities",
      "session_token",
      "session_token_expires_at",
    ])
    expect(session.session.session_token).toBe("st_value")
    expect(session.session.agent).toEqual({ id: 7, name: "Agent" })
  })

  it("leaves capabilities out when the response has none", async () => {
    const h = makeHarness(() => jsonResponse(200, tokenResponse("rt_1", {}, false)))
    await startAndSettle(h)
    const session = h.posts.find((p) => p.type === "session")
    if (session?.type !== "session") throw new Error("expected a session message")
    expect(Object.keys(session.session).sort()).toEqual([
      "absolute_expires_at",
      "agent",
      "session_token",
      "session_token_expires_at",
    ])
  })

  it("retries network failures with backoff and succeeds on the third attempt", async () => {
    const h = makeHarness(({ n }) => {
      if (n < 3) return Promise.reject(new TypeError("Failed to fetch"))
      return jsonResponse(200, tokenResponse("rt_1"))
    })
    await startAndSettle(h)
    await advanceUntilFetchCount(h, 3)
    await settle()
    expect(h.fetchSpy).toHaveBeenCalledTimes(3)
    expect(types(h).filter((t) => t === "redeeming")).toHaveLength(3)
    expect(types(h)).toContain("session")
    expect(terminals(h)).toEqual([])
    // 1 s, then 1.5 s between attempts (random() = 0.5 removes the jitter).
    expect(h.fetchTimes[1] - h.fetchTimes[0]).toBe(1000)
    expect(h.fetchTimes[2] - h.fetchTimes[1]).toBe(1500)
  })

  it("ends with network after five failed attempts", async () => {
    const h = makeHarness(() => Promise.reject(new TypeError("Failed to fetch")))
    await startAndSettle(h)
    await advanceUntilFetchCount(h, 5)
    await vi.advanceTimersByTimeAsync(10 * MINUTE)
    expect(h.fetchSpy).toHaveBeenCalledTimes(5)
    expect(types(h).filter((t) => t === "redeeming")).toHaveLength(5)
    expect(terminals(h)).toEqual([{ type: "terminal", reason: "network" }])
  })

  it("retries an uncoded 5xx but not a 5xx that carries a known code", async () => {
    const retried = makeHarness(({ n }) => (n === 1 ? jsonResponse(503, "<html>down</html>") : jsonResponse(200, tokenResponse("rt_1"))))
    await startAndSettle(retried)
    await advanceUntilFetchCount(retried, 2)
    await settle()
    expect(retried.fetchSpy).toHaveBeenCalledTimes(2)
    expect(types(retried)).toContain("session")

    const final = makeHarness(() => errorResponse(500, "widget_disabled"))
    await startAndSettle(final)
    await vi.advanceTimersByTimeAsync(MINUTE)
    expect(final.fetchSpy).toHaveBeenCalledTimes(1)
    expect(terminals(final)).toEqual([{ type: "terminal", reason: "unavailable" }])
  })

  it("times out each attempt after 15 seconds, covering the whole exchange", async () => {
    const signals: AbortSignal[] = []
    const h = makeHarness(({ init }) => {
      signals.push(init.signal as AbortSignal)
      return hangingFetch(init)
    })
    await startAndSettle(h)
    await vi.advanceTimersByTimeAsync(14_999)
    expect(signals[0].aborted).toBe(false)
    expect(h.fetchSpy).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(1)
    expect(signals[0].aborted).toBe(true)
    // A timed-out redeem is a transport failure: the next attempt follows after the backoff.
    await vi.advanceTimersByTimeAsync(1000)
    expect(h.fetchSpy).toHaveBeenCalledTimes(2)
  })

  it("treats a response body that stalls past the timeout as a transport failure", async () => {
    const stalled = new Response(new ReadableStream({ start: () => undefined }), { status: 200 })
    const h = makeHarness(({ n }) => (n === 1 ? stalled : jsonResponse(200, tokenResponse("rt_1"))))
    await startAndSettle(h)
    await vi.advanceTimersByTimeAsync(15_000 + 1000)
    await settle()
    expect(h.fetchSpy).toHaveBeenCalledTimes(2)
    expect(types(h)).toContain("session")
  })

  it("wakes a redeem that is waiting in backoff, and ignores wake otherwise", async () => {
    const h = makeHarness(({ n }) => (n === 1 ? Promise.reject(new TypeError("offline")) : jsonResponse(200, tokenResponse("rt_1"))))
    await startAndSettle(h)
    expect(h.fetchSpy).toHaveBeenCalledTimes(1)
    h.core.handleMessage({ type: "wake" })
    await settle()
    expect(h.fetchSpy).toHaveBeenCalledTimes(2)

    const idle = makeHarness(() => jsonResponse(200, tokenResponse("rt_1")))
    idle.core.handleMessage({ type: "wake" })
    await settle()
    expect(idle.fetchSpy).not.toHaveBeenCalled()
  })

  it("does not retry a 429 and ends with network", async () => {
    const h = makeHarness(() => jsonResponse(429, "<html>slow down</html>", { "Retry-After": "60" }))
    await startAndSettle(h)
    await vi.advanceTimersByTimeAsync(5 * MINUTE)
    expect(h.fetchSpy).toHaveBeenCalledTimes(1)
    expect(terminals(h)).toEqual([{ type: "terminal", reason: "network" }])
  })

  it("maps a 422 to link_invalid", async () => {
    const h = makeHarness(() => errorResponse(422, "invalid_input"))
    await startAndSettle(h)
    expect(terminals(h)).toEqual([{ type: "terminal", reason: "link_invalid" }])
    expect(h.fetchSpy).toHaveBeenCalledTimes(1)
  })

  it("ends with unavailable for an uncoded 4xx without retrying", async () => {
    const h = makeHarness(() => jsonResponse(404, { detail: "Not Found" }))
    await startAndSettle(h)
    await vi.advanceTimersByTimeAsync(5 * MINUTE)
    expect(h.fetchSpy).toHaveBeenCalledTimes(1)
    expect(terminals(h)).toEqual([{ type: "terminal", reason: "unavailable" }])
  })

  it("ends with unavailable for a 4xx whose error code is not a known server code", async () => {
    const h = makeHarness(() => errorResponse(400, "unexpected_error"))
    await startAndSettle(h)
    await vi.advanceTimersByTimeAsync(5 * MINUTE)
    expect(h.fetchSpy).toHaveBeenCalledTimes(1)
    expect(terminals(h)).toEqual([{ type: "terminal", reason: "unavailable" }])
  })

  it("ends with unavailable for a 3xx without retrying", async () => {
    const h = makeHarness(() => jsonResponse(302, {}))
    await startAndSettle(h)
    await vi.advanceTimersByTimeAsync(5 * MINUTE)
    expect(h.fetchSpy).toHaveBeenCalledTimes(1)
    expect(terminals(h)).toEqual([{ type: "terminal", reason: "unavailable" }])
  })

  it("ends with unavailable when a 200 response lacks required fields", async () => {
    const h = makeHarness(() => jsonResponse(200, { session_token: "st", reconnect_token: "rt_1" }))
    await startAndSettle(h)
    expect(terminals(h)).toEqual([{ type: "terminal", reason: "unavailable" }])
    expect(h.fetchSpy).toHaveBeenCalledTimes(1)
  })
})

describe("link session Worker end reasons for server error codes", () => {
  const cases: Array<[string, number, string]> = [
    ["link_invalid", 401, "link_invalid"],
    ["session_expired", 401, "session_ended"],
    ["reconnect_invalid", 401, "session_ended"],
    ["identity_mismatch", 401, "session_ended"],
    ["agent_not_available", 403, "unavailable"],
    ["widget_disabled", 403, "unavailable"],
    ["invalid_input", 400, "unavailable"],
    ["grant_malformed", 400, "unavailable"],
    ["signature_invalid", 401, "unavailable"],
    ["grant_expired", 401, "unavailable"],
    ["grant_already_used", 401, "unavailable"],
    ["agent_not_granted", 403, "unavailable"],
    ["invalid_runtime_context", 400, "unavailable"],
    ["encryption_required", 400, "unavailable"],
    ["rate_limited", 429, "network"],
  ]

  it.each(cases)("redeem: %s (%i) ends with %s", async (code, status, reason) => {
    const h = makeHarness(() => errorResponse(status, code))
    await startAndSettle(h)
    expect(terminals(h)).toEqual([{ type: "terminal", reason }])
    expect(h.fetchSpy).toHaveBeenCalledTimes(1)
  })

  it.each(cases)(
    "renew: %s (%i) ends with %s",
    async (code, status, reason) => {
      const h = makeHarness(({ url }) => (url === REDEEM_URL ? jsonResponse(200, tokenResponse("rt_1")) : errorResponse(status, code)))
      await startAndSettle(h)
      h.core.handleMessage({ type: "renew" })
      await settle()
      expect(h.fetchSpy).toHaveBeenCalledTimes(2)
      expect(terminals(h)).toEqual([{ type: "terminal", reason }])
    },
  )

  it("renew: an uncoded 404 ends with unavailable after a single request", async () => {
    const h = makeHarness(({ url }) => (url === REDEEM_URL ? jsonResponse(200, tokenResponse("rt_1")) : jsonResponse(404, { detail: "Not Found" })))
    await startAndSettle(h)
    h.core.handleMessage({ type: "renew" })
    await vi.advanceTimersByTimeAsync(5 * MINUTE)
    expect(h.fetchSpy).toHaveBeenCalledTimes(2)
    expect(terminals(h)).toEqual([{ type: "terminal", reason: "unavailable" }])
  })

  it("renew: a 429 with Retry-After 60 is not retried and ends with network", async () => {
    const h = makeHarness(({ url }) => (
      url === REDEEM_URL ? jsonResponse(200, tokenResponse("rt_1")) : jsonResponse(429, "slow down", { "Retry-After": "60" })
    ))
    await startAndSettle(h)
    h.core.handleMessage({ type: "renew" })
    await vi.advanceTimersByTimeAsync(5 * MINUTE)
    expect(h.fetchSpy).toHaveBeenCalledTimes(2)
    expect(terminals(h)).toEqual([{ type: "terminal", reason: "network" }])
  })
})

describe("link session Worker renew", () => {
  async function withSession(handler: FetchHandler) {
    const h = makeHarness(({ url, body, init, n }) => (
      url === REDEEM_URL ? jsonResponse(200, tokenResponse("rt_1")) : handler({ url, body, init, n })
    ))
    await startAndSettle(h)
    h.posts.length = 0
    return h
  }

  it("presents the current reconnect token and replaces it with the rotated one", async () => {
    let issued = 1
    const h = await withSession(() => {
      issued += 1
      return jsonResponse(200, tokenResponse(`rt_${issued}`))
    })
    h.core.handleMessage({ type: "renew" })
    await settle()
    h.core.handleMessage({ type: "renew" })
    await settle()
    h.core.handleMessage({ type: "renew" })
    await settle()
    const renewBodies = h.fetchBodies.slice(1)
    expect(renewBodies).toEqual([
      { reconnect_token: "rt_1" },
      { reconnect_token: "rt_2" },
      { reconnect_token: "rt_3" },
    ])
    const [, init] = h.fetchSpy.mock.calls[1] as [string, RequestInit]
    expect(h.fetchSpy.mock.calls[1][0]).toBe(RENEW_URL)
    expect(init.credentials).toBe("omit")
    expect(h.posts.filter((p) => p.type === "session")).toHaveLength(3)
  })

  it("sends a single request for two simultaneous renews", async () => {
    const h = await withSession(({ init }) => hangingFetch(init))
    h.core.handleMessage({ type: "renew" })
    h.core.handleMessage({ type: "renew" })
    await settle()
    expect(h.fetchSpy).toHaveBeenCalledTimes(2)
  })

  it("does not send another request while a renew is waiting in backoff", async () => {
    const h = await withSession(() => Promise.reject(new TypeError("offline")))
    h.core.handleMessage({ type: "renew" })
    await settle()
    expect(h.fetchSpy).toHaveBeenCalledTimes(2)
    h.core.handleMessage({ type: "renew" })
    await settle()
    expect(h.fetchSpy).toHaveBeenCalledTimes(2)
  })

  it("retries without limit, capped at five minutes, and reports degraded each time", async () => {
    const h = await withSession(() => Promise.reject(new TypeError("offline")))
    h.core.handleMessage({ type: "renew" })
    await settle()
    await advanceUntilFetchCount(h, 30, 1000, 10 * HOUR)
    expect(h.fetchSpy.mock.calls.length).toBeGreaterThanOrEqual(30)
    expect(terminals(h)).toEqual([])
    const degraded = h.posts.filter((p) => p.type === "degraded")
    expect(degraded.length).toBeGreaterThanOrEqual(29)
    expect(degraded.every((p) => p.type === "degraded" && p.code === "network_unavailable")).toBe(true)
    const gaps = h.fetchTimes.slice(2).map((t, i) => t - h.fetchTimes[i + 1])
    expect(Math.max(...gaps)).toBe(300_000)
  })

  it("wakes a renew that is waiting in backoff", async () => {
    let attempts = 0
    const h = await withSession(() => {
      attempts += 1
      return attempts === 1 ? Promise.reject(new TypeError("offline")) : jsonResponse(200, tokenResponse("rt_2"))
    })
    h.core.handleMessage({ type: "renew" })
    await settle()
    expect(h.fetchSpy).toHaveBeenCalledTimes(2)
    h.core.handleMessage({ type: "wake" })
    await settle()
    expect(h.fetchSpy).toHaveBeenCalledTimes(3)
    expect(h.posts.some((p) => p.type === "session")).toBe(true)
  })

  it("times out a renew after five seconds", async () => {
    const signals: AbortSignal[] = []
    const h = await withSession(({ init }) => {
      signals.push(init.signal as AbortSignal)
      return hangingFetch(init)
    })
    h.core.handleMessage({ type: "renew" })
    await settle()
    await vi.advanceTimersByTimeAsync(4999)
    expect(signals[0].aborted).toBe(false)
    await vi.advanceTimersByTimeAsync(1)
    expect(signals[0].aborted).toBe(true)
  })

  it("ends with unavailable when the rotated session token is already about to expire", async () => {
    const h = await withSession(() => jsonResponse(200, tokenResponse("rt_2", {
      session_token_expires_at: new Date(Date.now() + 30_000).toISOString(),
    })))
    h.core.handleMessage({ type: "renew" })
    await settle()
    expect(terminals(h)).toEqual([{ type: "terminal", reason: "unavailable" }])
    expect(h.posts.some((p) => p.type === "session")).toBe(false)
  })
})

describe("link session Worker renew stays inside the server's 30 second grace", () => {
  // Drives the renew backoff up past 30 s with rejected fetches, then makes the
  // next attempt fail in the given way and measures when the attempt after that
  // is sent.
  async function backedOffRenew(lastAttempt: (init: RequestInit) => Promise<Response> | Response) {
    const rejections = 10
    const h = makeHarness(({ url, init, n }) => {
      if (url === REDEEM_URL) return jsonResponse(200, tokenResponse("rt_1"))
      // n === 1 is the redeem; renew attempts are n = 2 ... rejections + 1.
      if (n <= rejections + 1) return Promise.reject(new TypeError("offline"))
      if (n === rejections + 2) return lastAttempt(init)
      return jsonResponse(200, tokenResponse("rt_2"))
    })
    await startAndSettle(h)
    h.core.handleMessage({ type: "renew" })
    await settle()
    await advanceUntilFetchCount(h, rejections + 2, 500, 10 * HOUR)
    return h
  }

  it("sends the attempt after a timeout within ten seconds even when the backoff is longer", async () => {
    const h = await backedOffRenew((init) => hangingFetch(init))
    const rejectedGap = h.fetchTimes[11] - h.fetchTimes[10]
    expect(rejectedGap).toBeGreaterThan(30_000)
    const timeoutAt = h.fetchTimes[11] + 5000
    await vi.advanceTimersByTimeAsync(5000)
    await advanceUntilFetchCount(h, 13, 100, MINUTE)
    expect(h.fetchSpy.mock.calls.length).toBe(13)
    expect(h.fetchTimes[12] - timeoutAt).toBeLessThanOrEqual(10_000)
    expect(h.fetchTimes[12] - timeoutAt).toBeGreaterThan(0)
  })

  it("sends the attempt after a 504 within ten seconds", async () => {
    const h = await backedOffRenew(() => jsonResponse(504, "gateway timeout"))
    await advanceUntilFetchCount(h, 13, 100, MINUTE)
    expect(h.fetchSpy.mock.calls.length).toBe(13)
    expect(h.fetchTimes[12] - h.fetchTimes[11]).toBeLessThanOrEqual(10_000)
  })

  it("keeps the normal backoff after a rejected fetch", async () => {
    const h = await backedOffRenew(() => Promise.reject(new TypeError("offline")))
    const before = h.fetchTimes[11] - h.fetchTimes[10]
    expect(before).toBeGreaterThan(30_000)
    await vi.advanceTimersByTimeAsync(10_000)
    expect(h.fetchSpy.mock.calls.length).toBe(12)
    await advanceUntilFetchCount(h, 13, 500, 10 * MINUTE)
    expect(h.fetchTimes[12] - h.fetchTimes[11]).toBeGreaterThan(10_000)
  })

  it("keeps the normal backoff after an uncoded 5xx that is not a 504", async () => {
    const h = await backedOffRenew(() => jsonResponse(502, "bad gateway"))
    await vi.advanceTimersByTimeAsync(10_000)
    expect(h.fetchSpy.mock.calls.length).toBe(12)
  })
})

describe("link session Worker keeps the reconnect token inside the Worker", () => {
  const LEAK = "rt_SECRET_RECONNECT_VALUE"

  function assertNoLeak(h: Harness) {
    const serialized = JSON.stringify(h.posts)
    expect(serialized).not.toContain(LEAK)
    expect(serialized).not.toContain("rt_SECRET")
    expect(serialized).not.toContain("reconnect_token")
  }

  it("never posts the token across redeem, repeated renewals and terminals", async () => {
    let issued = 0
    const h = makeHarness(({ url }) => {
      if (url === REDEEM_URL) return jsonResponse(200, tokenResponse(LEAK))
      issued += 1
      if (issued <= 3) return jsonResponse(200, tokenResponse(`${LEAK}_${issued}`))
      return errorResponse(401, "reconnect_invalid")
    })
    h.core.announce()
    await startAndSettle(h)
    for (let i = 0; i < 4; i += 1) {
      h.core.handleMessage({ type: "renew" })
      await settle()
    }
    expect(h.posts.filter((p) => p.type === "session")).toHaveLength(4)
    expect(terminals(h)).toEqual([{ type: "terminal", reason: "session_ended" }])
    assertNoLeak(h)
  })

  it.each([
    ["a truncated JSON body", `{"session_token":"st","reconnect_token":"${LEAK}`],
    ["a non-JSON body that contains the token", `oops: ${LEAK} leaked`],
  ])("a 200 with %s produces only a terminal message", async (_label, body) => {
    const h = makeHarness(() => new Response(body, { status: 200 }))
    h.core.announce()
    await startAndSettle(h)
    expect(h.posts.filter((p) => p.type !== "hello" && p.type !== "redeeming")).toEqual([
      { type: "terminal", reason: "unavailable" },
    ])
    assertNoLeak(h)
  })

  it.each([
    ["a truncated JSON body", `{"session_token":"st","reconnect_token":"${LEAK}`],
    ["a non-JSON body that contains the token", `oops: ${LEAK} leaked`],
  ])("a renew 200 with %s produces only a terminal message", async (_label, body) => {
    const h = makeHarness(({ url }) => (url === REDEEM_URL ? jsonResponse(200, tokenResponse(LEAK)) : new Response(body, { status: 200 })))
    await startAndSettle(h)
    h.posts.length = 0
    h.core.handleMessage({ type: "renew" })
    await settle()
    expect(h.posts).toEqual([{ type: "terminal", reason: "unavailable" }])
    assertNoLeak(h)
  })

  it("catches an exception raised while handling a response and reports a fixed code", async () => {
    const poisoned = {
      get status(): number {
        throw new Error(`boom ${LEAK}`)
      },
    } as unknown as Response
    const h = makeHarness(() => poisoned)
    await startAndSettle(h)
    expect(h.posts.filter((p) => p.type !== "redeeming")).toEqual([{ type: "terminal", reason: "unavailable" }])
    assertNoLeak(h)
  })

  it("reports a fixed code when a message handler throws", async () => {
    const posts: WorkerOutboundMessage[] = []
    let failNext = true
    const core = createLinkSessionWorkerCore({
      fetch: vi.fn(),
      post: (message) => {
        if (failNext && message.type === "redeeming") {
          failNext = false
          throw new Error(`post failed ${LEAK}`)
        }
        posts.push(message)
      },
      now: () => Date.now(),
      random: () => 0.5,
      setTimer: (callback, ms) => setTimeout(callback, ms),
      clearTimer: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
    })
    core.handleMessage({ type: "start", code: CODE })
    await settle()
    expect(posts).toEqual([{ type: "terminal", reason: "unavailable" }])
    expect(JSON.stringify(posts)).not.toContain(LEAK)
  })
})

describe("link session Worker session fields", () => {
  it("rejects a session whose agent is not an object", async () => {
    const h = makeHarness(() => jsonResponse(200, tokenResponse("rt_1", {
      session: { absolute_expires_at: new Date(Date.now() + HOUR).toISOString(), agent: "x" },
    })))
    await startAndSettle(h)
    expect(terminals(h)).toEqual([{ type: "terminal", reason: "unavailable" }])
  })
})

describe("link session Worker browser storage", () => {
  // Every browser storage surface is replaced by a counting stub. The Worker
  // logic must not reach any of them on the redeem, renew or terminal paths.
  function installStorageSpies() {
    const spies = {
      localSet: vi.spyOn(Storage.prototype, "setItem"),
      localGet: vi.spyOn(Storage.prototype, "getItem"),
      cookieSet: vi.fn(),
      idbOpen: vi.fn(),
      cachesOpen: vi.fn(),
    }
    const cookieDescriptor = Object.getOwnPropertyDescriptor(Document.prototype, "cookie")
    Object.defineProperty(document, "cookie", { configurable: true, get: () => "", set: spies.cookieSet })
    vi.stubGlobal("indexedDB", { open: spies.idbOpen })
    vi.stubGlobal("caches", { open: spies.cachesOpen })
    return {
      spies,
      restore: () => {
        Reflect.deleteProperty(document, "cookie")
        if (cookieDescriptor) Object.defineProperty(Document.prototype, "cookie", cookieDescriptor)
        vi.unstubAllGlobals()
      },
    }
  }

  it.each([
    ["redeem and renew success", undefined, async (h: Harness) => { h.core.handleMessage({ type: "renew" }); await settle() }],
    ["a terminal", undefined, async (h: Harness) => { h.setHandler(() => errorResponse(401, "reconnect_invalid")); h.core.handleMessage({ type: "renew" }); await settle() }],
    ["a retry", ({ n }: { n: number }) => (n < 2 ? Promise.reject(new TypeError("offline")) : jsonResponse(200, tokenResponse("rt_1"))), async (h: Harness) => { await advanceUntilFetchCount(h, 2); await settle() }],
  ])("does not read or write storage on %s", async (_label, handler, act) => {
    const { spies, restore } = installStorageSpies()
    try {
      const h = makeHarness(handler ?? (() => jsonResponse(200, tokenResponse("rt_1"))))
      await startAndSettle(h)
      await act(h)
      expect(spies.localSet).not.toHaveBeenCalled()
      expect(spies.localGet).not.toHaveBeenCalled()
      expect(spies.cookieSet).not.toHaveBeenCalled()
      expect(spies.idbOpen).not.toHaveBeenCalled()
      expect(spies.cachesOpen).not.toHaveBeenCalled()
    } finally {
      restore()
    }
  })

})

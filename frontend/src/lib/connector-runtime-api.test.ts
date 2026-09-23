import { afterEach, describe, expect, it, vi } from "vitest"

import {
  CONNECTOR_RUNTIME_DIALOG_HOST_PATTERNS,
  CONNECTOR_RUNTIME_KNOWN_REASONS,
  CONNECTOR_RUNTIME_SECTIONS,
  CONNECTOR_RUNTIME_TYPES,
  DIALOG_OUTCOME_KINDS,
  KEY_NAME_REJECTED_REASON,
  buildSubmitItems,
  classifySubmitFailure,
  connectorRuntimeInputDraftKey,
  fetchTaskConnectorRuntimeRequirements,
  isConnectorRuntimeDialogHostPath,
  isSubmitEnabled,
  isTypeMismatchDispositionStale,
  readConnectorRuntimeReport,
  reconcileTypeMismatchDisposition,
  resolveDialogActions,
  resolveDialogOutcome,
  submitTaskConnectorRuntimeValues,
  type ConnectorRuntimeConnector,
  type ConnectorRuntimeErrorMessageKey,
  type ConnectorRuntimeInput,
  type ConnectorRuntimeReport,
  type ConnectorRuntimeSection,
  type ConnectorRuntimeType,
  type DialogOutcome,
  type DialogOutcomeKind,
  type SubmitTaskConnectorRuntimeValuesFailure,
} from "./connector-runtime-api"
import { AUTH_CACHE_KEY } from "@/lib/auth-cache"

const REF_A = { connector_type: "custom_api", connector_id: 1 }

function input(overrides: Partial<ConnectorRuntimeInput> & { section: ConnectorRuntimeSection; key: string; type: ConnectorRuntimeType }): ConnectorRuntimeInput {
  return { required: false, satisfied: false, expired: false, ...overrides }
}
function connector(ref: typeof REF_A, name: string, inputs: ConnectorRuntimeInput[]): ConnectorRuntimeConnector {
  return { connector_ref: ref, name, inputs }
}
function report(satisfied: boolean, connectors: ConnectorRuntimeConnector[]): ConnectorRuntimeReport {
  return { satisfied, secrets_expires_at: null, connectors }
}
function rawReport(overrides: Record<string, unknown> = {}) {
  return {
    satisfied: true,
    secrets_expires_at: null,
    connectors: [
      {
        connector_ref: { connector_type: "custom_api", connector_id: 1 },
        name: "Example",
        inputs: [
          { section: "context", key: "k", type: "string", required: false, satisfied: true, expired: false },
        ],
      },
    ],
    ...overrides,
  }
}

describe("readConnectorRuntimeReport", () => {
  it("accepts only the closed section and type sets", () => {
    expect(CONNECTOR_RUNTIME_SECTIONS).toHaveLength(3)
    expect(CONNECTOR_RUNTIME_TYPES).toHaveLength(2)

    for (const section of CONNECTOR_RUNTIME_SECTIONS) {
      for (const type of CONNECTOR_RUNTIME_TYPES) {
        const value = {
          satisfied: true,
          secrets_expires_at: null,
          connectors: [
            {
              connector_ref: { connector_type: "custom_api", connector_id: 1 },
              name: "Example",
              inputs: [{ section, key: "k", type, required: false, satisfied: true, expired: false }],
            },
          ],
        }
        expect(readConnectorRuntimeReport(value)).not.toBeNull()
      }
    }

    // Rejection rows: an unrecognized section, an unrecognized type, a
    // missing connectors array, a non-array connectors field, and a
    // connector_id of the wrong wire type.
    expect(readConnectorRuntimeReport(rawReport({
      connectors: [{ ...rawReport().connectors[0], inputs: [{ section: "password", key: "k", type: "string", required: false, satisfied: true, expired: false }] }],
    }))).toBeNull()
    expect(readConnectorRuntimeReport(rawReport({
      connectors: [{ ...rawReport().connectors[0], inputs: [{ section: "context", key: "k", type: "number", required: false, satisfied: true, expired: false }] }],
    }))).toBeNull()
    const withoutConnectors = { satisfied: rawReport().satisfied, secrets_expires_at: rawReport().secrets_expires_at }
    expect(readConnectorRuntimeReport(withoutConnectors)).toBeNull()
    expect(readConnectorRuntimeReport({ ...rawReport(), connectors: "nope" })).toBeNull()
    expect(readConnectorRuntimeReport({
      ...rawReport(),
      connectors: [{ ...rawReport().connectors[0], connector_ref: { connector_type: "custom_api", connector_id: "1" } }],
    })).toBeNull()

    expect(readConnectorRuntimeReport(null)).toBeNull()
    expect(readConnectorRuntimeReport(undefined)).toBeNull()
    expect(readConnectorRuntimeReport("nope")).toBeNull()
    expect(readConnectorRuntimeReport({ satisfied: true, secrets_expires_at: null, connectors: [] })).toEqual({
      satisfied: true,
      secrets_expires_at: null,
      connectors: [],
    })
  })
})

// The two functions that actually talk to the server, driven through a
// stubbed `fetch`. apiRequest falls straight through to `fetch` with no
// stored access token, which is what an empty localStorage gives every test
// here, so the stub sees the real request this client builds and the real
// response handling runs against it -- including readSubmitErrorEnvelope,
// which has no other way in.
describe("the connector-runtime HTTP calls", () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  /** Installs the stub and returns the request log it appends to. */
  function stubFetch(responder: () => Promise<Response> | Response) {
    const calls: Array<{ url: string; init: RequestInit | undefined }> = []
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, init })
      return responder()
    }))
    return calls
  }

  function jsonResponse(status: number, body: unknown): Response {
    return new Response(JSON.stringify(body), {
      status,
      headers: { "content-type": "application/json" },
    })
  }

  const validReportBody = rawReport()
  const expectedReport = {
    satisfied: true,
    secrets_expires_at: null,
    connectors: [
      {
        connector_ref: { connector_type: "custom_api", connector_id: 1 },
        name: "Example",
        inputs: [
          { section: "context", key: "k", type: "string", required: false, satisfied: true, expired: false },
        ],
      },
    ],
  }

  describe("fetchTaskConnectorRuntimeRequirements", () => {
    it("reads a report off a 200 and asks the per-task read endpoint for it", async () => {
      const calls = stubFetch(() => jsonResponse(200, validReportBody))
      await expect(fetchTaskConnectorRuntimeRequirements(7)).resolves.toEqual({
        ok: true,
        report: expectedReport,
      })
      expect(calls).toHaveLength(1)
      expect(calls[0].url).toBe("/api/chat/task/7/connector-runtime-requirements")
    })

    it("reports a transport failure when the request never completes", async () => {
      stubFetch(() => Promise.reject(new TypeError("Failed to fetch")))
      await expect(fetchTaskConnectorRuntimeRequirements(7)).resolves.toEqual({
        ok: false,
        kind: "transport",
      })
    })

    it("reports the status on any non-200, envelope or not", async () => {
      // The read endpoint downgrades every ConnectorRuntimeError into an
      // envelope-less HTTPException, so a body that happens to carry an
      // envelope must not be read as one here either: this path has exactly
      // one failure shape and the status is all of it.
      for (const [status, body] of [
        [503, { error: { code: "connector_runtime_unavailable", details: { reason: "team_scope_resolution_failed" } } }],
        [404, { detail: "Task not found" }],
        [403, { detail: "Forbidden" }],
      ] as const) {
        stubFetch(() => jsonResponse(status, body))
        await expect(fetchTaskConnectorRuntimeRequirements(7)).resolves.toEqual({
          ok: false,
          kind: "http",
          status,
        })
        vi.unstubAllGlobals()
      }
    })

    it("reports malformed for a 200 whose body is not a report", async () => {
      for (const body of [
        { satisfied: "yes", secrets_expires_at: null, connectors: [] },
        rawReport({ connectors: "not-an-array" }),
        rawReport({ connectors: [{ connector_ref: { connector_type: "custom_api" }, name: "x", inputs: [] }] }),
        [],
        null,
      ]) {
        stubFetch(() => jsonResponse(200, body))
        await expect(fetchTaskConnectorRuntimeRequirements(7)).resolves.toEqual({
          ok: false,
          kind: "malformed",
        })
        vi.unstubAllGlobals()
      }

      // A 200 carrying a body that is not JSON at all lands in the same
      // place rather than throwing: parseApiResponse hands back a null
      // `data`, which is not the closed report shape.
      stubFetch(() => new Response("<html>gateway</html>", {
        status: 200,
        headers: { "content-type": "text/html" },
      }))
      await expect(fetchTaskConnectorRuntimeRequirements(7)).resolves.toEqual({
        ok: false,
        kind: "malformed",
      })
    })
  })

  describe("submitTaskConnectorRuntimeValues", () => {
    const items = [{ connector_ref: REF_A, context: { token: "abc" } }]

    it("posts the items and reads the refreshed report off a 200", async () => {
      const calls = stubFetch(() => jsonResponse(200, validReportBody))
      await expect(submitTaskConnectorRuntimeValues(7, items)).resolves.toEqual({
        ok: true,
        report: expectedReport,
      })
      expect(calls).toHaveLength(1)
      expect(calls[0].url).toBe("/api/chat/task/7/connector-runtime-values")
      expect(calls[0].init?.method).toBe("POST")
      expect(JSON.parse(String(calls[0].init?.body))).toEqual({ items })
    })

    it("reports a transport failure when the request never completes", async () => {
      stubFetch(() => Promise.reject(new TypeError("Failed to fetch")))
      await expect(submitTaskConnectorRuntimeValues(7, items)).resolves.toEqual({
        ok: false,
        kind: "transport",
      })
    })

    it("reads the error envelope off a non-200 that carries one", async () => {
      stubFetch(() => jsonResponse(409, {
        error: {
          code: "runtime_context_immutable",
          details: { reason: "conflict.context.token", connector_ref: { connector_type: "custom_api", connector_id: 1 } },
        },
      }))
      await expect(submitTaskConnectorRuntimeValues(7, items)).resolves.toEqual({
        ok: false,
        kind: "coded",
        status: 409,
        code: "runtime_context_immutable",
        reason: "conflict.context.token",
        connectorRef: REF_A,
      })

      // The 503 shape carries no details at all. "Absent" has to survive as
      // absent: classifySubmitFailure tells the retryable 503 from the
      // unretryable one by whether `reason` is undefined, so an envelope
      // reader that normalized a missing reason to "" would flip that.
      stubFetch(() => jsonResponse(503, { error: { code: "connector_runtime_unavailable" } }))
      await expect(submitTaskConnectorRuntimeValues(7, items)).resolves.toEqual({
        ok: false,
        kind: "coded",
        status: 503,
        code: "connector_runtime_unavailable",
        reason: undefined,
        connectorRef: undefined,
      })

      // Present-and-empty is a different input from absent and must stay
      // one; a connector_ref that fails ref validation drops out without
      // taking the code with it.
      stubFetch(() => jsonResponse(400, {
        error: { code: "invalid_runtime_context", details: { reason: "", connector_ref: { connector_type: "custom_api" } } },
      }))
      await expect(submitTaskConnectorRuntimeValues(7, items)).resolves.toEqual({
        ok: false,
        kind: "coded",
        status: 400,
        code: "invalid_runtime_context",
        reason: "",
        connectorRef: undefined,
      })

      // A `details` that is not an object at all still leaves a usable code.
      stubFetch(() => jsonResponse(400, {
        error: { code: "invalid_runtime_context", details: "not-an-object" },
      }))
      await expect(submitTaskConnectorRuntimeValues(7, items)).resolves.toEqual({
        ok: false,
        kind: "coded",
        status: 400,
        code: "invalid_runtime_context",
        reason: undefined,
        connectorRef: undefined,
      })
    })

    it("falls back to the status on a non-200 with no readable envelope", async () => {
      // The task-not-found 404 is the production case: it is a plain
      // HTTPException and carries no envelope. The rest are the ways an
      // envelope can be present but unreadable -- every one of them has to
      // become a status, never a half-trusted `coded` with a guessed code.
      for (const body of [
        { detail: "Task not found" },
        { error: "not-an-object" },
        { error: { message: "no code field" } },
        { error: { code: 42 } },
        [],
        null,
      ]) {
        stubFetch(() => jsonResponse(404, body))
        await expect(submitTaskConnectorRuntimeValues(7, items)).resolves.toEqual({
          ok: false,
          kind: "http",
          status: 404,
        })
        vi.unstubAllGlobals()
      }

      // A non-200 whose body is not JSON at all -- the proxy HTML page --
      // has no envelope either.
      stubFetch(() => new Response("<html>502</html>", {
        status: 502,
        headers: { "content-type": "text/html" },
      }))
      await expect(submitTaskConnectorRuntimeValues(7, items)).resolves.toEqual({
        ok: false,
        kind: "http",
        status: 502,
      })
    })

    it("reports malformed for a 200 whose body is not a report", async () => {
      // Distinct from "http": a 200 that does not validate is a server bug,
      // and the dialog re-reads the report on it instead of retrying.
      stubFetch(() => jsonResponse(200, { satisfied: true, connectors: [] }))
      await expect(submitTaskConnectorRuntimeValues(7, items)).resolves.toEqual({
        ok: false,
        kind: "malformed",
      })

      stubFetch(() => new Response("", { status: 200 }))
      await expect(submitTaskConnectorRuntimeValues(7, items)).resolves.toEqual({
        ok: false,
        kind: "malformed",
      })
    })
  })

  describe("gives up on a request that never answers", () => {
    afterEach(() => {
      vi.useRealTimers()
      localStorage.removeItem(AUTH_CACHE_KEY)
    })

    /**
     * A fetch that answers only when its caller aborts it, which is what a
     * real one does: the request stays open until the signal fires and then
     * rejects. Returns the signals it was handed, so a test can check whether
     * the timer behind one was cleared.
     */
    function stubUnansweredFetch(): AbortSignal[] {
      const signals: AbortSignal[] = []
      vi.stubGlobal("fetch", vi.fn((_url: string, init?: RequestInit) => {
        const signal = init?.signal
        if (signal) signals.push(signal)
        return new Promise<Response>((_resolve, reject) => {
          // Real fetch rejects straight away when it is handed a signal that
          // has already fired, which is what every retry after the first
          // abort gets. Without this the retries here would hang instead.
          if (signal?.aborted) {
            reject(new Error("aborted"))
            return
          }
          signal?.addEventListener("abort", () => reject(new Error("aborted")))
        })
      }))
      return signals
    }

    it("abandons a requirements read that never answers", async () => {
      // Nothing else stops this request: the dialog's read effect ignores a
      // late answer but does not cancel it, so without the timeout the
      // promise below never settles and the dialog is left unable to save.
      vi.useFakeTimers()
      const signals = stubUnansweredFetch()
      const pending = fetchTaskConnectorRuntimeRequirements(7)

      await vi.advanceTimersByTimeAsync(20_000)

      await expect(pending).resolves.toEqual({ ok: false, kind: "transport" })
      expect(signals).toHaveLength(1)
      expect(signals[0].aborted).toBe(true)
    })

    it("abandons a save that never answers", async () => {
      // The save is the worse of the two: the dialog refuses to close while
      // one is in flight, so a request that never answers leaves no way out
      // of it at all.
      vi.useFakeTimers()
      const signals = stubUnansweredFetch()
      const pending = submitTaskConnectorRuntimeValues(7, [{ connector_ref: REF_A, context: { token: "abc" } }])

      await vi.advanceTimersByTimeAsync(20_000)

      await expect(pending).resolves.toEqual({ ok: false, kind: "transport" })
      expect(signals).toHaveLength(1)
      expect(signals[0].aborted).toBe(true)
    })

    it("stops the clock once the response has arrived", async () => {
      // The timer has to be cleared on the way out. Left running, it fires
      // against a request that already finished, and every call leaves one
      // more pending abort behind it.
      vi.useFakeTimers()
      const signals: AbortSignal[] = []
      vi.stubGlobal("fetch", vi.fn((_url: string, init?: RequestInit) => {
        if (init?.signal) signals.push(init.signal)
        return Promise.resolve(new Response(JSON.stringify(rawReport()), {
          status: 200,
          headers: { "content-type": "application/json" },
        }))
      }))

      await expect(fetchTaskConnectorRuntimeRequirements(7)).resolves.toEqual({
        ok: true,
        report: expectedReport,
      })
      await vi.advanceTimersByTimeAsync(30_000)

      expect(signals).toHaveLength(1)
      expect(signals[0].aborted).toBe(false)
    })

    /**
     * A fetch whose headers arrive at once and whose body then never does.
     * This is the shape the timeout used to miss entirely: fetch resolves on
     * the headers, so a body that stalls was read outside the window. The
     * body stream errors when the signal fires, the same way a real one does.
     */
    function stubHeadersOnlyFetch(): AbortSignal[] {
      const signals: AbortSignal[] = []
      vi.stubGlobal("fetch", vi.fn((_url: string, init?: RequestInit) => {
        const signal = init?.signal
        if (signal) signals.push(signal)
        const body = new ReadableStream({
          start(controller) {
            signal?.addEventListener("abort", () => controller.error(new Error("aborted")))
          },
        })
        return Promise.resolve(new Response(body, {
          status: 200,
          headers: { "content-type": "application/json" },
        }))
      }))
      return signals
    }

    it("abandons a requirements read whose headers arrive but whose body never does", async () => {
      // Not `malformed`: parseApiResponse turns the cut-short body read into
      // an empty body, which would otherwise be reported as a server bug --
      // no retry button, and the dialog re-reading a report that is never
      // going to arrive.
      vi.useFakeTimers()
      const signals = stubHeadersOnlyFetch()
      const pending = fetchTaskConnectorRuntimeRequirements(7)

      await vi.advanceTimersByTimeAsync(20_000)

      await expect(pending).resolves.toEqual({ ok: false, kind: "transport" })
      expect(signals).toHaveLength(1)
      expect(signals[0].aborted).toBe(true)
    })

    it("abandons a save whose headers arrive but whose body never does", async () => {
      // The save is the one the dialog refuses to close over, so a body that
      // never finishes arriving must end the same way a request that never
      // answered at all does.
      vi.useFakeTimers()
      const signals = stubHeadersOnlyFetch()
      const pending = submitTaskConnectorRuntimeValues(7, [{ connector_ref: REF_A, context: { token: "abc" } }])

      await vi.advanceTimersByTimeAsync(20_000)

      await expect(pending).resolves.toEqual({ ok: false, kind: "transport" })
      expect(signals).toHaveLength(1)
      expect(signals[0].aborted).toBe(true)
    })

    /**
     * A stored session, in the shape auth-cache validates (same fields
     * api-wrapper's own tests write). With one of these present apiRequest
     * stops calling fetch directly and goes through fetchWithRetry instead,
     * which is the path the three cases above never take.
     */
    function writeAuthCache() {
      const now = Date.now()
      localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify({
        schemaVersion: 2,
        sessionId: "timeout-session",
        credentialRevision: 0,
        profileRevision: 0,
        user: { id: "u1", username: "u1" },
        token: "access-token",
        refreshToken: "refresh-token",
        timestamp: now,
        expiresAt: now + 3_600_000,
        refreshExpiresAt: now + 7_200_000,
      }))
    }

    it("gives up once, not once per retry, when a signed-in request never answers", async () => {
      // A signed-in request goes through fetchWithRetry, which retries twice
      // more on a rejection. One controller covers all three attempts, so the
      // two retries are handed a signal that has already fired and reject at
      // once -- the whole call still ends 20 seconds in (plus fetchWithRetry's
      // own 100ms and 200ms backoffs), not 60. That is why no "do not retry a
      // timed-out request" rule was added to fetchWithRetry: there is nothing
      // for one to save.
      vi.useFakeTimers()
      writeAuthCache()
      const signals = stubUnansweredFetch()
      const pending = fetchTaskConnectorRuntimeRequirements(7)
      let settled = false
      void pending.then(() => { settled = true })

      await vi.advanceTimersByTimeAsync(19_999)
      expect(settled).toBe(false)

      await vi.advanceTimersByTimeAsync(1 + 300)

      await expect(pending).resolves.toEqual({ ok: false, kind: "transport" })
      expect(signals).toHaveLength(3)
      expect(signals.every(signal => signal.aborted)).toBe(true)
      expect(signals[1]).toBe(signals[0])
    })

    it("still reports a genuinely empty body as malformed", async () => {
      // The guard above keys on the signal, not on the body being empty, so
      // an empty body that arrived in time keeps its own distinct answer.
      vi.useFakeTimers()
      vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(new Response("", { status: 200 }))))

      await expect(fetchTaskConnectorRuntimeRequirements(7)).resolves.toEqual({
        ok: false,
        kind: "malformed",
      })
    })
  })
})

describe("buildSubmitItems", () => {
  it("submits only unsatisfied non-blank context drafts under the report's own ref", () => {
    const r = report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "satisfiedKey", type: "string", required: true, satisfied: true }),
        input({ section: "context", key: "unsatisfiedKey", type: "string", required: true, satisfied: false }),
        input({ section: "secrets", key: "secretSatisfied", type: "string", required: false, satisfied: false }),
        input({ section: "secrets", key: "secretUnsatisfied", type: "string", required: true, satisfied: false }),
        input({ section: "auth_selector", key: "authKey", type: "string", required: true, satisfied: false }),
        input({ section: "context", key: "objKey", type: "object", required: false, satisfied: false }),
      ]),
    ])

    const blankDrafts: Record<string, string> = {
      [connectorRuntimeInputDraftKey(REF_A, "context", "unsatisfiedKey", "string")]: "",
      [connectorRuntimeInputDraftKey(REF_A, "secrets", "secretUnsatisfied", "string")]: "value",
      [connectorRuntimeInputDraftKey(REF_A, "auth_selector", "authKey", "string")]: "value",
    }
    expect(buildSubmitItems(r, blankDrafts)).toEqual([])

    const whitespaceDrafts: Record<string, string> = {
      [connectorRuntimeInputDraftKey(REF_A, "context", "unsatisfiedKey", "string")]: "   ",
    }
    expect(buildSubmitItems(r, whitespaceDrafts)).toEqual([])

    const tabDrafts: Record<string, string> = {
      [connectorRuntimeInputDraftKey(REF_A, "context", "unsatisfiedKey", "string")]: "\t\n",
    }
    expect(buildSubmitItems(r, tabDrafts)).toEqual([])

    // An empty object is treated the same as a blank string: the server's
    // own blank check on an object-typed field is `not value`, and `{}`
    // fails it, so a submission that includes it 400s the whole batch.
    const emptyObjectDrafts: Record<string, string> = {
      [connectorRuntimeInputDraftKey(REF_A, "context", "objKey", "object")]: "{}",
    }
    expect(buildSubmitItems(r, emptyObjectDrafts)).toEqual([])

    const paddedEmptyObjectDrafts: Record<string, string> = {
      [connectorRuntimeInputDraftKey(REF_A, "context", "objKey", "object")]: "  {}  ",
    }
    expect(buildSubmitItems(r, paddedEmptyObjectDrafts)).toEqual([])

    const validDrafts: Record<string, string> = {
      [connectorRuntimeInputDraftKey(REF_A, "context", "unsatisfiedKey", "string")]: "a",
      [connectorRuntimeInputDraftKey(REF_A, "context", "objKey", "object")]: '{"k":1}',
      // These would 422 if they leaked into the request; proving they never
      // do is this test's whole point.
      [connectorRuntimeInputDraftKey(REF_A, "secrets", "secretUnsatisfied", "string")]: "value",
      [connectorRuntimeInputDraftKey(REF_A, "auth_selector", "authKey", "string")]: "value",
    }
    expect(buildSubmitItems(r, validDrafts)).toEqual([
      { connector_ref: REF_A, context: { unsatisfiedKey: "a", objKey: { k: 1 } } },
    ])
  })

  it("stores a string draft with its surrounding whitespace removed", () => {
    // The stored value is immutable once the server merges it, so submitting
    // the untrimmed text writes a pasted secret's stray whitespace
    // permanently: the same secret resubmitted without it comes back 409
    // runtime_context_immutable and the task cannot be recovered.
    const r = report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "token", type: "string", required: true }),
      ]),
    ])
    for (const raw of ["  abc  ", "\tabc\n", "abc ", " abc"]) {
      expect(buildSubmitItems(r, { [connectorRuntimeInputDraftKey(REF_A, "context", "token", "string")]: raw })).toEqual([
        { connector_ref: REF_A, context: { token: "abc" } },
      ])
    }
    // Interior whitespace is part of the value and is left alone.
    expect(buildSubmitItems(r, { [connectorRuntimeInputDraftKey(REF_A, "context", "token", "string")]: "  a b  " })).toEqual([
      { connector_ref: REF_A, context: { token: "a b" } },
    ])
  })
})

describe("classifySubmitFailure", () => {
  const baseReport = report(false, [
    connector(REF_A, "A", [
      input({ section: "context", key: "token", type: "string", required: true }),
      input({ section: "context", key: "config", type: "object", required: true }),
    ]),
  ])

  function coded(status: number, code: string, reason?: string, connectorRef?: typeof REF_A): SubmitTaskConnectorRuntimeValuesFailure {
    return { ok: false, kind: "coded", status, code, reason, connectorRef }
  }

  it("maps every write failure shape to exactly one disposition", () => {
    expect(CONNECTOR_RUNTIME_KNOWN_REASONS).toHaveLength(7)
    // Binds every member of the constant to the disposition it actually
    // produces -- a `Record` keyed by the constant's own member type, so
    // adding a reason to CONNECTOR_RUNTIME_KNOWN_REASONS without adding its
    // row here is a type error, not a silently-passing loop. A member mapped
    // to "contactAdmin" is one classifySubmitFailure has no specific branch
    // for and deliberately lets reach the generic fallthrough.
    const messageKeyByKnownReason: Record<(typeof CONNECTOR_RUNTIME_KNOWN_REASONS)[number], ConnectorRuntimeErrorMessageKey> = {
      empty_items: "contactAdmin",
      empty_item_payload: "contactAdmin",
      payload_too_large: "tooLarge",
      duplicate_ref: "contactAdmin",
      connector_not_selected: "notInSession",
      undeclared_context_key: "configChanged",
      auth_selector_not_supported: "contactAdmin",
    }
    for (const reason of CONNECTOR_RUNTIME_KNOWN_REASONS) {
      expect(classifySubmitFailure(coded(400, "invalid_runtime_context", reason, REF_A), baseReport).messageKey).toBe(
        messageKeyByKnownReason[reason],
      )
    }

    expect(classifySubmitFailure({ ok: false, kind: "transport" }, baseReport)).toEqual({
      messageKey: "network", retry: true, refresh: false, locate: {},
    })
    expect(classifySubmitFailure(coded(503, "connector_runtime_unavailable"), baseReport)).toEqual({
      messageKey: "busyRetry", retry: true, refresh: false, locate: {},
    })
    for (const reason of ["team_scope_resolution_failed", "", "unknown"]) {
      expect(classifySubmitFailure(coded(503, "connector_runtime_unavailable", reason), baseReport)).toEqual({
        messageKey: "contactAdmin", retry: false, refresh: false, locate: {},
      })
    }
    expect(classifySubmitFailure(coded(409, "runtime_context_immutable", "conflict.context.token", REF_A), baseReport)).toEqual({
      messageKey: "conflict", retry: false, refresh: true, locate: { connectorRef: REF_A, key: "token" },
    })
    // A type mismatch refreshes: the connector owner can change a key's
    // declared type between the report this dialog read and the write it
    // just submitted, and without a refresh every retry keeps failing the
    // same way against the stale declaration.
    expect(classifySubmitFailure(coded(400, "invalid_runtime_context", "type_mismatch.context.config", REF_A), baseReport)).toEqual({
      messageKey: "typeObject", retry: false, refresh: true, locate: { connectorRef: REF_A, key: "config" },
    })
    expect(classifySubmitFailure(coded(400, "invalid_runtime_context", "type_mismatch.context.token", REF_A), baseReport)).toEqual({
      messageKey: "typeString", retry: false, refresh: true, locate: { connectorRef: REF_A, key: "token" },
    })
    expect(classifySubmitFailure(coded(400, "invalid_runtime_context", "empty_value.context.token", REF_A), baseReport)).toEqual({
      messageKey: "emptyValue", retry: false, refresh: false, locate: { connectorRef: REF_A, key: "token" },
    })
    expect(classifySubmitFailure(coded(400, "invalid_runtime_context", KEY_NAME_REJECTED_REASON, REF_A), baseReport)).toEqual({
      messageKey: "keyNameRejected", retry: false, refresh: false, locate: { connectorRef: REF_A },
    })
    expect(classifySubmitFailure(coded(400, "invalid_runtime_context", "undeclared_context_key", REF_A), baseReport)).toEqual({
      messageKey: "configChanged", retry: false, refresh: true, locate: { connectorRef: REF_A },
    })
    expect(classifySubmitFailure(coded(400, "invalid_runtime_context", "connector_not_selected", REF_A), baseReport)).toEqual({
      messageKey: "notInSession", retry: false, refresh: true, locate: { connectorRef: REF_A },
    })
    expect(classifySubmitFailure(coded(404, "connector_not_found", undefined, REF_A), baseReport)).toEqual({
      messageKey: "connectorUnavailable", retry: false, refresh: true, locate: { connectorRef: REF_A },
    })
    expect(classifySubmitFailure(coded(404, "connector_not_found"), baseReport)).toEqual({
      messageKey: "connectorUnavailable", retry: false, refresh: true, locate: {},
    })
    expect(classifySubmitFailure(coded(400, "invalid_runtime_context", "payload_too_large", REF_A), baseReport)).toEqual({
      messageKey: "tooLarge", retry: false, refresh: false, locate: { connectorRef: REF_A },
    })
    expect(classifySubmitFailure(coded(400, "invalid_runtime_context", "payload_too_large"), baseReport)).toEqual({
      messageKey: "tooLarge", retry: false, refresh: false, locate: {},
    })
    for (const reason of ["empty_items", "empty_item_payload", "duplicate_ref"]) {
      expect(classifySubmitFailure(coded(400, "invalid_runtime_context", reason), baseReport)).toEqual({
        messageKey: "contactAdmin", retry: false, refresh: false, locate: {},
      })
    }
    for (const status of [401, 403, 404, 422, 500]) {
      expect(classifySubmitFailure({ ok: false, kind: "http", status }, baseReport)).toEqual({
        messageKey: "contactAdmin", retry: false, refresh: false, locate: {},
      })
    }
    expect(classifySubmitFailure({ ok: false, kind: "malformed" }, baseReport)).toEqual({
      messageKey: "contactAdmin", retry: false, refresh: true, locate: {},
    })

    // W13/W14: a corrupted stored selection also carries code
    // invalid_runtime_context and an English reason -- neither sentence
    // equals the key-name-rejected constant, so both must still fall
    // through to the generic disposition rather than being misread as a
    // fixable key name.
    for (const reason of [
      "stored selected refs must be a list",
      "connector ref must be an object",
      "connector ref has unknown field(s): ['x']",
    ]) {
      expect(classifySubmitFailure(coded(400, "invalid_runtime_context", reason), baseReport)).toEqual({
        messageKey: "contactAdmin", retry: false, refresh: false, locate: {},
      })
    }

    // A hook-installed code entirely outside the closed set.
    expect(classifySubmitFailure(coded(400, "some_future_code", "anything"), baseReport)).toEqual({
      messageKey: "contactAdmin", retry: false, refresh: false, locate: {},
    })
  })

  it("picks the context declaration's type, not a same-named secrets row that comes first in the report", () => {
    // A secrets-section row named "shared" is listed before the context row
    // sharing that name: findDeclaredInputType must not let the first
    // match-by-key-alone win, or a type_mismatch on the context row would
    // report the secrets row's type instead.
    const reportWithSecretsFirst = report(false, [
      connector(REF_A, "A", [
        input({ section: "secrets", key: "shared", type: "object", required: false }),
        input({ section: "context", key: "shared", type: "string", required: true }),
      ]),
    ])
    expect(classifySubmitFailure(
      coded(400, "invalid_runtime_context", "type_mismatch.context.shared", REF_A),
      reportWithSecretsFirst,
    )).toEqual({
      messageKey: "typeString", retry: false, refresh: true, locate: { connectorRef: REF_A, key: "shared" },
    })
  })

  it("reports an undeclared type as unknown instead of as text", () => {
    // "ghost" is not declared by any connector in baseReport, so there is
    // no declaration to read a type from at all -- this must not be
    // reported as "needs text" on the strength of a declaration nobody
    // read.
    expect(classifySubmitFailure(
      coded(400, "invalid_runtime_context", "type_mismatch.context.ghost", REF_A),
      baseReport,
    )).toEqual({
      messageKey: "typeUnknown", retry: false, refresh: true, locate: { connectorRef: REF_A, key: "ghost" },
    })
  })
})

describe("isTypeMismatchDispositionStale", () => {
  const disposition = (messageKey: ConnectorRuntimeErrorMessageKey, key = "token") => ({
    messageKey, retry: false as const, refresh: true as const, locate: { connectorRef: REF_A, key },
  })

  it("clears an unknown-type hint once a refreshed report declares a type", () => {
    // Both directions must be asserted: a mutation that drops the
    // typeUnknown-specific branch and falls through to the typeString
    // comparison would still read true for the "declared as object" case
    // by accident, and only the "declared as string" case catches it.
    const declaredObject = report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "object", required: true })]),
    ])
    const declaredString = report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])
    expect(isTypeMismatchDispositionStale(disposition("typeUnknown"), declaredObject)).toBe(true)
    expect(isTypeMismatchDispositionStale(disposition("typeUnknown"), declaredString)).toBe(true)
  })

  it("keeps an unknown-type hint while the row is still undeclared", () => {
    const stillUndeclared = report(false, [connector(REF_A, "A", [])])
    expect(isTypeMismatchDispositionStale(disposition("typeUnknown"), stillUndeclared)).toBe(false)
  })

  it("leaves the two named type hints behaving as before", () => {
    const declaredString = report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])
    const declaredObject = report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "object", required: true })]),
    ])
    const stillUndeclared = report(false, [connector(REF_A, "A", [])])
    expect(isTypeMismatchDispositionStale(disposition("typeObject"), declaredString)).toBe(true)
    expect(isTypeMismatchDispositionStale(disposition("typeObject"), declaredObject)).toBe(false)
    expect(isTypeMismatchDispositionStale(disposition("typeObject"), stillUndeclared)).toBe(false)
  })
})

describe("reconcileTypeMismatchDisposition", () => {
  const disposition = (messageKey: ConnectorRuntimeErrorMessageKey, key = "token") => ({
    messageKey, retry: false as const, refresh: true as const, locate: { connectorRef: REF_A, key },
  })
  const declaredString = report(false, [
    connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
  ])
  const declaredObject = report(false, [
    connector(REF_A, "A", [input({ section: "context", key: "token", type: "object", required: true })]),
  ])
  const stillUndeclared = report(false, [connector(REF_A, "A", [])])

  it("re-derives an unknown-type hint into the type the refreshed report declares", () => {
    // The refresh is what answers the question this hint said it could not,
    // so the answer replaces the hint. Dropping it instead would take a
    // rejection the user was shown off the screen while the draft that
    // provoked it is still in the box and still saveable.
    expect(reconcileTypeMismatchDisposition(disposition("typeUnknown"), declaredString))
      .toEqual({ ...disposition("typeUnknown"), messageKey: "typeString" })
    expect(reconcileTypeMismatchDisposition(disposition("typeUnknown"), declaredObject))
      .toEqual({ ...disposition("typeUnknown"), messageKey: "typeObject" })
  })

  it("drops a named type hint the refreshed report contradicts", () => {
    // Nothing in the refreshed report says what the server was enforcing
    // instead, so there is no re-derivation to make here.
    expect(reconcileTypeMismatchDisposition(disposition("typeObject"), declaredString)).toBeNull()
    expect(reconcileTypeMismatchDisposition(disposition("typeString"), declaredObject)).toBeNull()
  })

  it("hands back the same disposition when the refreshed report changes nothing", () => {
    // Identity, not equality: the dialog compares the result with what it
    // passed in to decide whether to touch its field-error state at all.
    const unknown = disposition("typeUnknown")
    expect(reconcileTypeMismatchDisposition(unknown, stillUndeclared)).toBe(unknown)
    const named = disposition("typeString")
    expect(reconcileTypeMismatchDisposition(named, declaredString)).toBe(named)
    const unrelated = disposition("conflict")
    expect(reconcileTypeMismatchDisposition(unrelated, declaredObject)).toBe(unrelated)
  })
})

describe("resolveDialogOutcome", () => {
  it("reads the top-level flag instead of recomputing it", () => {
    // Every required key satisfied, but a malformed optional key forces the
    // server's own aggregate to false (services/connector_runtime.py's
    // aggregation rule) -- recomputing from the per-key flags would read
    // this as "met" when the server does not.
    const badOptionalKeyReport = report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "token", type: "string", required: true, satisfied: true }),
        input({ section: "context", key: "bad key!", type: "string", required: false, satisfied: false }),
      ]),
    ])
    expect(resolveDialogOutcome(badOptionalKeyReport).kind).not.toBe("met")

    // The server's own top-level true, with an optional key still unfilled.
    const metWithOptionalGap = report(true, [
      connector(REF_A, "A", [
        input({ section: "context", key: "optional", type: "string", required: false, satisfied: false }),
      ]),
    ])
    expect(resolveDialogOutcome(metWithOptionalGap).kind).toBe("met")
  })
})

describe("resolves the dialog outcome from required context and unsupported inputs", () => {
  // The report shapes this rule has to reconcile, numbered so the standalone
  // cases below can refer back to a row here. Rows 4, 6 and 10 have two
  // shapes -- one at the moment the dialog opens, one after a save fills in
  // the required key -- and the rows below carry the post-save shape, which
  // is the stricter of the two. Row 10 is the standalone test further down.
  // A row for "top-level false with nothing unsatisfied" is left out: it is
  // not reachable in production, because a malformed key's own `satisfied`
  // is always false.
  const reconciliationRows: Array<[string, ConnectorRuntimeReport, DialogOutcome]> = [
    ["only a required context key missing (row 1)", report(false, [connector(REF_A, "A", [
      input({ section: "context", key: "k1", type: "string", required: true }),
    ])]), { kind: "fillable", blocking: [] }],
    ["one of two required context keys already satisfied (row 2)", report(false, [connector(REF_A, "A", [
      input({ section: "context", key: "k1", type: "string", required: true, satisfied: true }),
      input({ section: "context", key: "k2", type: "string", required: true }),
    ])]), { kind: "fillable", blocking: [] }],
    ["only a required secrets key missing (row 3a)", report(false, [connector(REF_A, "A", [
      input({ section: "secrets", key: "s1", type: "string", required: true }),
    ])]), { kind: "unsupported_only", blocking: [{ connectorRef: REF_A, key: "s1" }] }],
    ["only a required auth_selector key missing (row 3b)", report(false, [connector(REF_A, "A", [
      input({ section: "auth_selector", key: "a1", type: "string", required: true }),
    ])]), { kind: "unsupported_only", blocking: [{ connectorRef: REF_A, key: "a1" }] }],
    ["required context satisfied, required secrets still missing (row 4, post-save shape)", report(false, [connector(REF_A, "A", [
      input({ section: "context", key: "k1", type: "string", required: true, satisfied: true }),
      input({ section: "secrets", key: "s1", type: "string", required: true }),
    ])]), { kind: "unsupported_only", blocking: [{ connectorRef: REF_A, key: "s1" }] }],
    ["required context satisfied, an optional context key still open, required secrets missing (row 5)", report(false, [connector(REF_A, "A", [
      input({ section: "context", key: "k1", type: "string", required: true, satisfied: true }),
      input({ section: "context", key: "k2", type: "string", required: false }),
      input({ section: "secrets", key: "s1", type: "string", required: true }),
    ])]), { kind: "unsupported_only", blocking: [{ connectorRef: REF_A, key: "s1" }] }],
    ["required context satisfied, required secrets missing, an optional secret also unfilled (row 6, post-save shape)", report(false, [connector(REF_A, "A", [
      input({ section: "context", key: "k1", type: "string", required: true, satisfied: true }),
      input({ section: "secrets", key: "s1", type: "string", required: true }),
      input({ section: "secrets", key: "s2", type: "string", required: false }),
    ])]), { kind: "unsupported_only", blocking: [{ connectorRef: REF_A, key: "s1" }] }],
    ["a required context key with a malformed name (row 7)", report(false, [connector(REF_A, "A", [
      input({ section: "context", key: "bad key", type: "string", required: true }),
    ])]), { kind: "fillable", blocking: [] }],
    ["an optional malformed context key + a required secret (row 8)", report(false, [connector(REF_A, "A", [
      input({ section: "context", key: "bad key", type: "string", required: false }),
      input({ section: "secrets", key: "s1", type: "string", required: true }),
    ])]), { kind: "unsupported_only", blocking: [{ connectorRef: REF_A, key: "s1" }] }],
    ["a required secrets key with a malformed name (row 9)", report(false, [connector(REF_A, "A", [
      input({ section: "secrets", key: "bad key", type: "string", required: true }),
    ])]), { kind: "unsupported_only", blocking: [{ connectorRef: REF_A, key: "bad key" }] }],
    ["everything satisfied (row 12)", report(true, [connector(REF_A, "A", [
      input({ section: "context", key: "k1", type: "string", required: true, satisfied: true }),
    ])]), { kind: "met" }],
    ["satisfied with an unfilled optional context key (row 13)", report(true, [connector(REF_A, "A", [
      input({ section: "context", key: "optional", type: "string", required: false }),
    ])]), { kind: "met" }],
  ]
  it.each(reconciliationRows)("%s", (_name, r, expected) => {
    expect(resolveDialogOutcome(r)).toEqual(expected)
  })

  // Row 10, post-save shape: every required key satisfied, but a malformed
  // *optional* secrets key still forces the server's own aggregate false --
  // the one row where reading `report.satisfied` is stricter than
  // recomputing completeness from the per-key flags, which would close the
  // dialog here. A required context + a required secret would also hit this
  // if the required secret's own flag never trips true, but only a
  // malformed *optional* secrets key produces it in practice
  // (services/connector_runtime.py:1199).
  it("resolves nothing_fillable when a malformed optional secrets key blocks the aggregate with nothing left for the user to do (row 10, post-save shape)", () => {
    const r = report(false, [connector(REF_A, "A", [
      input({ section: "context", key: "k1", type: "string", required: true, satisfied: true }),
      input({ section: "secrets", key: "bad key", type: "string", required: false }),
    ])])
    expect(resolveDialogOutcome(r)).toEqual({ kind: "nothing_fillable" })
  })

  it("binds DIALOG_OUTCOME_KINDS to every kind the reconciliation rows above actually produce", () => {
    expect(DIALOG_OUTCOME_KINDS).toHaveLength(4)
    const producedKinds = new Set(reconciliationRows.map(([, , expected]) => expected.kind))
    producedKinds.add("nothing_fillable") // produced by the row-10 test above, not the table here
    expect(producedKinds).toEqual(new Set(DIALOG_OUTCOME_KINDS))
  })
})

describe("derives the action set from the outcome and the resend snapshot", () => {
  const fillable: DialogOutcome = { kind: "fillable", blocking: [] }
  const unsupportedOnly: DialogOutcome = { kind: "unsupported_only", blocking: [] }
  const nothingFillable: DialogOutcome = { kind: "nothing_fillable" }
  const met: DialogOutcome = { kind: "met" }

  it.each([
    ["fillable", fillable, true, ["saveAndResend", "saveOnly"]],
    ["fillable", fillable, false, ["saveOnly"]],
    ["unsupported_only", unsupportedOnly, true, ["acknowledge"]],
    ["unsupported_only", unsupportedOnly, false, ["acknowledge"]],
    ["nothing_fillable", nothingFillable, true, ["acknowledge"]],
    ["nothing_fillable", nothingFillable, false, ["acknowledge"]],
    // A met report is normally closed before it can render, but the refresh
    // a failed save triggers can install one into an open dialog. It must
    // still carry a button, or that dialog has an empty footer.
    ["met", met, true, ["acknowledge"]],
    ["met", met, false, ["acknowledge"]],
  ] as const)("derives the action set for %s with resend=%s", (_label, outcome, hasResendPayload, expected) => {
    expect(resolveDialogActions(outcome, hasResendPayload)).toEqual(expected)
  })

  it("gives every outcome kind at least one action", () => {
    const byKind: Record<DialogOutcomeKind, DialogOutcome> = {
      met,
      unsupported_only: unsupportedOnly,
      nothing_fillable: nothingFillable,
      fillable,
    }
    for (const kind of DIALOG_OUTCOME_KINDS) {
      for (const hasResendPayload of [true, false]) {
        expect(resolveDialogActions(byKind[kind], hasResendPayload).length).toBeGreaterThan(0)
      }
    }
  })
})

describe("isSubmitEnabled", () => {
  it("enables submit on one submittable key only", () => {
    expect(isSubmitEnabled([], false)).toBe(false)
    expect(isSubmitEnabled([{ connector_ref: REF_A, context: { k: "v" } }], false)).toBe(true)
    expect(isSubmitEnabled([{ connector_ref: REF_A, context: { k: "v" } }], true)).toBe(false)

    // Four missing context keys, only the malformed-name one filled in: a
    // malformed key name never blocks submission (it is a hint, not a
    // gate, 2.10), so this must still enable the button. Routed through
    // buildSubmitItems rather than a hand-written items array, so this row
    // actually exercises the two functions together instead of restating
    // isSubmitEnabled's own rule against itself.
    const fourMissingKeysReport = report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "k1", type: "string", required: true }),
        input({ section: "context", key: "k2", type: "string", required: true }),
        input({ section: "context", key: "k3", type: "string", required: true }),
        input({ section: "context", key: "bad key", type: "string", required: true }),
      ]),
    ])
    const onlyBadKeyFilled: Record<string, string> = {
      [connectorRuntimeInputDraftKey(REF_A, "context", "bad key", "string")]: "value",
    }
    expect(isSubmitEnabled(buildSubmitItems(fourMissingKeysReport, onlyBadKeyFilled), false)).toBe(true)
  })
})

describe("isConnectorRuntimeDialogHostPath", () => {
  it("matches only the three dialog host routes", () => {
    expect(CONNECTOR_RUNTIME_DIALOG_HOST_PATTERNS).toHaveLength(3)

    for (const path of ["/task/5", "/task/5/", "/workforces/7/run", "/workforces/7", "/workforces/new"]) {
      expect(isConnectorRuntimeDialogHostPath(path)).toBe(true)
    }
    for (const path of [
      "/task", "/workforces", "/settings", "/kb", "/build/3", "/agent/2", "/task/5/extra", null,
    ]) {
      expect(isConnectorRuntimeDialogHostPath(path)).toBe(false)
    }

    expect(CONNECTOR_RUNTIME_DIALOG_HOST_PATTERNS.every(pattern =>
      ["/task/5", "/workforces/7/run", "/workforces/7"].some(p => pattern.test(p)),
    )).toBe(true)
  })
})

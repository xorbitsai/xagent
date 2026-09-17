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
  readConnectorRuntimeReport,
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
  type SubmitTaskConnectorRuntimeValuesFailure,
} from "./connector-runtime-api"

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
      [connectorRuntimeInputDraftKey(REF_A, "unsatisfiedKey")]: "",
      [connectorRuntimeInputDraftKey(REF_A, "secretUnsatisfied")]: "value",
      [connectorRuntimeInputDraftKey(REF_A, "authKey")]: "value",
    }
    expect(buildSubmitItems(r, blankDrafts)).toEqual([])

    const whitespaceDrafts: Record<string, string> = {
      [connectorRuntimeInputDraftKey(REF_A, "unsatisfiedKey")]: "   ",
    }
    expect(buildSubmitItems(r, whitespaceDrafts)).toEqual([])

    const tabDrafts: Record<string, string> = {
      [connectorRuntimeInputDraftKey(REF_A, "unsatisfiedKey")]: "\t\n",
    }
    expect(buildSubmitItems(r, tabDrafts)).toEqual([])

    // An empty object is treated the same as a blank string: the server's
    // own blank check on an object-typed field is `not value`, and `{}`
    // fails it, so a submission that includes it 400s the whole batch.
    const emptyObjectDrafts: Record<string, string> = {
      [connectorRuntimeInputDraftKey(REF_A, "objKey")]: "{}",
    }
    expect(buildSubmitItems(r, emptyObjectDrafts)).toEqual([])

    const paddedEmptyObjectDrafts: Record<string, string> = {
      [connectorRuntimeInputDraftKey(REF_A, "objKey")]: "  {}  ",
    }
    expect(buildSubmitItems(r, paddedEmptyObjectDrafts)).toEqual([])

    const validDrafts: Record<string, string> = {
      [connectorRuntimeInputDraftKey(REF_A, "unsatisfiedKey")]: "a",
      [connectorRuntimeInputDraftKey(REF_A, "objKey")]: '{"k":1}',
      // These would 422 if they leaked into the request; proving they never
      // do is this test's whole point.
      [connectorRuntimeInputDraftKey(REF_A, "secretUnsatisfied")]: "value",
      [connectorRuntimeInputDraftKey(REF_A, "authKey")]: "value",
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
      expect(buildSubmitItems(r, { [connectorRuntimeInputDraftKey(REF_A, "token")]: raw })).toEqual([
        { connector_ref: REF_A, context: { token: "abc" } },
      ])
    }
    // Interior whitespace is part of the value and is left alone.
    expect(buildSubmitItems(r, { [connectorRuntimeInputDraftKey(REF_A, "token")]: "  a b  " })).toEqual([
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
    expect(classifySubmitFailure(coded(400, "invalid_runtime_context", "type_mismatch.context.config", REF_A), baseReport)).toEqual({
      messageKey: "typeObject", retry: false, refresh: false, locate: { connectorRef: REF_A, key: "config" },
    })
    expect(classifySubmitFailure(coded(400, "invalid_runtime_context", "type_mismatch.context.token", REF_A), baseReport)).toEqual({
      messageKey: "typeString", retry: false, refresh: false, locate: { connectorRef: REF_A, key: "token" },
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

  it.each([
    ["fillable", fillable, true, ["saveAndResend", "saveOnly"]],
    ["fillable", fillable, false, ["saveOnly"]],
    ["unsupported_only", unsupportedOnly, true, ["acknowledge"]],
    ["unsupported_only", unsupportedOnly, false, ["acknowledge"]],
    ["nothing_fillable", nothingFillable, true, ["acknowledge"]],
    ["nothing_fillable", nothingFillable, false, ["acknowledge"]],
  ] as const)("derives the action set for %s with resend=%s", (_label, outcome, hasResendPayload, expected) => {
    expect(resolveDialogActions(outcome, hasResendPayload)).toEqual(expected)
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
      [connectorRuntimeInputDraftKey(REF_A, "bad key")]: "value",
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

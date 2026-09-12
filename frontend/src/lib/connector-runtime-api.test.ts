import { describe, expect, it } from "vitest"

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
  isConnectorRuntimeDialogHostPath,
  isSubmitEnabled,
  readConnectorRuntimeReport,
  resolveDialogActions,
  resolveDialogOutcome,
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
    expect(CONNECTOR_RUNTIME_KNOWN_REASONS).toHaveLength(6)
    // Binds every member of the constant to the disposition it actually
    // produces -- a `Record` keyed by the constant's own member type, so
    // adding a reason to CONNECTOR_RUNTIME_KNOWN_REASONS without adding its
    // row here is a type error, not a silently-passing loop.
    const messageKeyByKnownReason: Record<(typeof CONNECTOR_RUNTIME_KNOWN_REASONS)[number], ConnectorRuntimeErrorMessageKey> = {
      empty_items: "contactAdmin",
      empty_item_payload: "contactAdmin",
      payload_too_large: "tooLarge",
      duplicate_ref: "contactAdmin",
      connector_not_selected: "notInSession",
      undeclared_context_key: "configChanged",
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

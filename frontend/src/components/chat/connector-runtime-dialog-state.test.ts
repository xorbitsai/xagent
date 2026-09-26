import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

import {
  canResendReport,
  deriveGates,
  gateFactsOf,
  INITIAL_DIALOG_STATE,
  isBusy,
  mergeSendFailureDisposition,
  reduceDialog,
  uniqueKeys,
  type DialogEvent,
  type DialogState,
  type FinishingEvent,
  type GateFacts,
  type InvalidObjectDraftReason,
  type MidEvent,
  type Phase,
  type SendFailureState,
  type View,
} from "./connector-runtime-dialog-state"
import type {
  ConnectorRuntimeConnector,
  ConnectorRuntimeFailureDisposition,
  ConnectorRuntimeInput,
  ConnectorRuntimeReport,
  ConnectorRuntimeSection,
  ConnectorRuntimeType,
} from "@/lib/connector-runtime-api"

const REF = { connector_type: "custom_api", connector_id: 1 }

function input(overrides: Partial<ConnectorRuntimeInput> & { section: ConnectorRuntimeSection; key: string; type: ConnectorRuntimeType }): ConnectorRuntimeInput {
  return { required: false, satisfied: false, expired: false, ...overrides }
}
function connector(inputs: ConnectorRuntimeInput[]): ConnectorRuntimeConnector {
  return { connector_ref: REF, name: "Example", inputs }
}
function report(satisfied: boolean, connectors: ConnectorRuntimeConnector[] = []): ConnectorRuntimeReport {
  return { satisfied, secrets_expires_at: null, connectors }
}

const MET_REPORT = report(true)
const NOTHING_FILLABLE_REPORT = report(false)
const UNSUPPORTED_ONLY_REPORT = report(false, [
  connector([input({ section: "secrets", key: "s", type: "string", required: true })]),
])
const FILLABLE_REPORT = report(false, [
  connector([input({ section: "context", key: "k", type: "string", required: true })]),
])

const DRAFT_KEY = `${REF.connector_type}:${REF.connector_id}:context:k:string`
const OBJECT_DRAFT_KEY = `${REF.connector_type}:${REF.connector_id}:context:obj:object`
const FILLED_ITEMS = [{ connector_ref: REF, context: { k: "value" } }]

function baseFacts(overrides: Partial<GateFacts> = {}): GateFacts {
  return {
    report: null,
    reportSeq: null,
    settledReadKey: null,
    readNonce: 0,
    busy: false,
    heldFailure: null,
    retrying: false,
    drafts: {},
    invalidDraftKeys: new Map<string, InvalidObjectDraftReason>(),
    request: { seq: 1, resendPayload: null },
    ...overrides,
  }
}

const FAILURE: SendFailureState = { snapshotId: "cid-1", disposition: "not_sent" }

describe("deriveGates", () => {
  it.each([
    {
      name: "liveSendFailure/sendFailed/needsSnapshotRecycle: no failure held",
      facts: baseFacts(),
      expect: { liveSendFailure: null, sendFailed: false, needsSnapshotRecycle: false },
    },
    {
      name: "liveSendFailure/sendFailed/needsSnapshotRecycle: held failure matches the request's resend payload",
      facts: baseFacts({ heldFailure: FAILURE, request: { seq: 1, resendPayload: { clientMessageId: "cid-1" } } }),
      expect: { liveSendFailure: FAILURE, sendFailed: true, needsSnapshotRecycle: false },
    },
    {
      name: "liveSendFailure/sendFailed/needsSnapshotRecycle: held failure no longer matches (snapshot moved on)",
      facts: baseFacts({ heldFailure: FAILURE, request: { seq: 1, resendPayload: { clientMessageId: "cid-2" } } }),
      expect: { liveSendFailure: null, sendFailed: false, needsSnapshotRecycle: true },
    },
    {
      name: "liveSendFailure/sendFailed/needsSnapshotRecycle: held failure but the request now carries no resend payload at all",
      facts: baseFacts({ heldFailure: FAILURE, request: { seq: 1, resendPayload: null } }),
      expect: { liveSendFailure: null, sendFailed: false, needsSnapshotRecycle: true },
    },
    {
      name: "readKey/reading: never settled",
      facts: baseFacts({ request: { seq: 3, resendPayload: null }, readNonce: 2 }),
      expect: { readKey: "3:2", reading: true },
    },
    {
      name: "readKey/reading: settled for this exact attempt",
      facts: baseFacts({ request: { seq: 3, resendPayload: null }, readNonce: 2, settledReadKey: "3:2" }),
      expect: { readKey: "3:2", reading: false },
    },
    {
      name: "readKey/reading: settled for an earlier nonce of the same request (a \"read again\" press is out)",
      facts: baseFacts({ request: { seq: 3, resendPayload: null }, readNonce: 2, settledReadKey: "3:1" }),
      expect: { readKey: "3:2", reading: true },
    },
    {
      name: "reportIsStale/readFailed: report matches the current request",
      facts: baseFacts({ report: MET_REPORT, reportSeq: 5, settledReadKey: "5:0", request: { seq: 5, resendPayload: null } }),
      expect: { reportIsStale: false, readFailed: false },
    },
    {
      name: "reportIsStale/readFailed: report is from an earlier request and the read has settled (the read failed)",
      facts: baseFacts({ report: MET_REPORT, reportSeq: 5, settledReadKey: "6:0", request: { seq: 6, resendPayload: null } }),
      expect: { reportIsStale: true, readFailed: true },
    },
    {
      name: "reportIsStale/readFailed: report is from an earlier request but a re-read for the new one is still out (not yet a failure)",
      facts: baseFacts({ report: MET_REPORT, reportSeq: 5, settledReadKey: null, request: { seq: 6, resendPayload: null } }),
      expect: { reportIsStale: true, readFailed: false },
    },
    {
      name: "outcome: no report yet",
      facts: baseFacts(),
      expect: { outcome: null },
    },
    {
      name: "outcome: met",
      facts: baseFacts({ report: MET_REPORT }),
      expect: { outcome: { kind: "met" } },
    },
    {
      name: "outcome: nothing_fillable",
      facts: baseFacts({ report: NOTHING_FILLABLE_REPORT }),
      expect: { outcome: { kind: "nothing_fillable" } },
    },
    {
      name: "outcome/actions/hasSaveEntryPoint: unsupported_only offers only acknowledge",
      facts: baseFacts({ report: UNSUPPORTED_ONLY_REPORT }),
      expect: {
        outcome: { kind: "unsupported_only", blocking: [{ connectorRef: REF, key: "s" }] },
        actions: ["acknowledge"],
        hasSaveEntryPoint: false,
      },
    },
    {
      name: "actions/hasSaveEntryPoint: fillable without a resend payload offers only saveOnly",
      facts: baseFacts({ report: FILLABLE_REPORT, request: { seq: 1, resendPayload: null } }),
      expect: { actions: ["saveOnly"], hasSaveEntryPoint: true, hasResendPayload: false },
    },
    {
      name: "actions/hasSaveEntryPoint/hasResendPayload: fillable with a resend payload offers both",
      facts: baseFacts({ report: FILLABLE_REPORT, request: { seq: 1, resendPayload: { clientMessageId: "cid-1" } } }),
      expect: { actions: ["saveAndResend", "saveOnly"], hasSaveEntryPoint: true, hasResendPayload: true },
    },
    {
      name: "submitItems/canSubmit/canSubmitNow: a fillable report with its required row still empty cannot submit",
      facts: baseFacts({ report: FILLABLE_REPORT, reportSeq: 1 }),
      expect: { submitItems: [], hasInvalidObjectDraft: false, canSubmit: false, canSubmitNow: false },
    },
    {
      name: "submitItems/canSubmit/canSubmitNow: filling the required row enables it",
      facts: baseFacts({ report: FILLABLE_REPORT, reportSeq: 1, drafts: { [DRAFT_KEY]: "value" } }),
      expect: { submitItems: FILLED_ITEMS, hasInvalidObjectDraft: false, canSubmit: true, canSubmitNow: true },
    },
    {
      name: "canSubmit/canSubmitNow: busy holds it closed even once the row is filled",
      facts: baseFacts({ report: FILLABLE_REPORT, reportSeq: 1, drafts: { [DRAFT_KEY]: "value" }, busy: true }),
      expect: { canSubmit: true, canSubmitNow: false },
    },
    {
      name: "canSubmit/canSubmitNow: a stale report holds it closed even once the row is filled",
      facts: baseFacts({ report: FILLABLE_REPORT, reportSeq: 1, drafts: { [DRAFT_KEY]: "value" }, request: { seq: 2, resendPayload: null } }),
      expect: { canSubmit: true, canSubmitNow: false },
    },
    {
      name: "hasInvalidObjectDraft/canSubmit: a live mark on an object row blocks submission even while another row is validly filled",
      facts: baseFacts({
        report: report(false, [connector([
          input({ section: "context", key: "k", type: "string", required: true }),
          input({ section: "context", key: "obj", type: "object" }),
        ])]),
        reportSeq: 1,
        drafts: { [DRAFT_KEY]: "value", [OBJECT_DRAFT_KEY]: "{" },
        invalidDraftKeys: new Map<string, InvalidObjectDraftReason>([[OBJECT_DRAFT_KEY, "invalid"]]),
      }),
      expect: { submitItems: FILLED_ITEMS, hasInvalidObjectDraft: true, canSubmit: false, canSubmitNow: false },
    },
    {
      name: "hasInvalidObjectDraft/canSubmit: a mark on an object row the report now reports satisfied is not live",
      facts: baseFacts({
        report: report(false, [connector([
          input({ section: "context", key: "k", type: "string", required: true }),
          input({ section: "context", key: "obj", type: "object", satisfied: true }),
        ])]),
        reportSeq: 1,
        drafts: { [DRAFT_KEY]: "value" },
        invalidDraftKeys: new Map<string, InvalidObjectDraftReason>([[OBJECT_DRAFT_KEY, "invalid"]]),
      }),
      expect: { submitItems: FILLED_ITEMS, hasInvalidObjectDraft: false, canSubmit: true, canSubmitNow: true },
    },
    {
      name: "hasInvalidObjectDraft/canSubmit: a mark on a row the report declares string-typed is not live",
      facts: baseFacts({
        report: FILLABLE_REPORT,
        reportSeq: 1,
        drafts: { [DRAFT_KEY]: "value" },
        invalidDraftKeys: new Map<string, InvalidObjectDraftReason>([[DRAFT_KEY, "invalid"]]),
      }),
      expect: { submitItems: FILLED_ITEMS, hasInvalidObjectDraft: false, canSubmit: true, canSubmitNow: true },
    },
    {
      name: "hasInvalidObjectDraft/canSubmitNow: a live invalid-object mark on the only required row blocks submission even though it is \"filled\"",
      facts: baseFacts({
        report: report(false, [connector([input({ section: "context", key: "k", type: "object", required: true })])]),
        reportSeq: 1,
        drafts: { [`${REF.connector_type}:${REF.connector_id}:context:k:object`]: "{}" },
        invalidDraftKeys: new Map([[`${REF.connector_type}:${REF.connector_id}:context:k:object`, "empty"]]),
      }),
      expect: { hasInvalidObjectDraft: true, canSubmitNow: false },
    },
    {
      name: "metHoldingSnapshot: met, holding a resend payload, nothing failed, not busy",
      facts: baseFacts({ report: MET_REPORT, request: { seq: 1, resendPayload: { clientMessageId: "cid-1" } } }),
      expect: { metHoldingSnapshot: true },
    },
    {
      name: "metHoldingSnapshot: false without a resend payload to hold",
      facts: baseFacts({ report: MET_REPORT, request: { seq: 1, resendPayload: null } }),
      expect: { metHoldingSnapshot: false },
    },
    {
      name: "metHoldingSnapshot: false while the send-failed panel is live for that same snapshot",
      facts: baseFacts({
        report: MET_REPORT,
        heldFailure: FAILURE,
        request: { seq: 1, resendPayload: { clientMessageId: "cid-1" } },
      }),
      expect: { metHoldingSnapshot: false },
    },
    {
      name: "metHoldingSnapshot: true when the held failure is for a snapshot the request no longer carries",
      facts: baseFacts({
        report: MET_REPORT,
        heldFailure: FAILURE,
        request: { seq: 1, resendPayload: { clientMessageId: "cid-2" } },
      }),
      expect: { sendFailed: false, needsSnapshotRecycle: true, metHoldingSnapshot: true },
    },
    {
      name: "metHoldingSnapshot: false while busy (a save-and-resend for this same message is still on the wire)",
      facts: baseFacts({ report: MET_REPORT, request: { seq: 1, resendPayload: { clientMessageId: "cid-1" } }, busy: true }),
      expect: { metHoldingSnapshot: false },
    },
    {
      name: "metHoldingSnapshot: false when the report is fillable rather than met, even holding a resend payload, nothing failed, not busy",
      facts: baseFacts({ report: FILLABLE_REPORT, request: { seq: 1, resendPayload: { clientMessageId: "cid-1" } } }),
      expect: { metHoldingSnapshot: false },
    },
    {
      name: "retryResendDisabled: neither retrying nor stale",
      facts: baseFacts({ reportSeq: 1, request: { seq: 1, resendPayload: null } }),
      expect: { retryResendDisabled: false },
    },
    {
      name: "retryResendDisabled: a retry is already out",
      facts: baseFacts({ reportSeq: 1, request: { seq: 1, resendPayload: null }, busy: true, retrying: true }),
      expect: { retryResendDisabled: true },
    },
    {
      name: "retryResendDisabled: the report on hand is stale",
      facts: baseFacts({ reportSeq: 1, request: { seq: 2, resendPayload: null } }),
      expect: { retryResendDisabled: true },
    },
  ])("$name", ({ facts, expect: partial }) => {
    const gates = deriveGates(facts)
    expect(gates).toMatchObject(partial)
  })

  // The invariant this exercises: the dialog's lifecycle (busy or not) and
  // the report's own outcome (met, fillable, ...) are independent
  // dimensions, not one flattened display phase. A same-task retarget's
  // re-read installs whatever report comes back regardless of what is in
  // flight, so a report that reads "still needs a value" can land while the
  // send-failed panel's retry resend for the earlier snapshot is still out.
  // The retarget swapped in a newer snapshot, so the held failure no longer
  // matches and the panel is gone; the retry is still in flight, so busy
  // holds every submit gate closed even with the row filled; and the
  // fillable outcome still drives the row and footer content underneath.
  it("holds submission closed while a retry for a superseded snapshot is out, even when a same-task re-read replaces the report with one still asking for a value", () => {
    const gates = deriveGates(baseFacts({
      report: FILLABLE_REPORT,
      reportSeq: 7,
      settledReadKey: "7:0",
      busy: true,
      retrying: true,
      heldFailure: FAILURE,
      drafts: { [DRAFT_KEY]: "value" },
      request: { seq: 7, resendPayload: { clientMessageId: "cid-9" } },
    }))
    expect(gates.reportIsStale).toBe(false)
    expect(gates.reading).toBe(false)
    expect(gates.liveSendFailure).toBeNull()
    expect(gates.sendFailed).toBe(false)
    expect(gates.needsSnapshotRecycle).toBe(true)
    expect(gates.canSubmit).toBe(true)
    // "blocking" on a fillable outcome lists the still-unsupported secrets
    // group, not the context row the report is fillable *because* of --
    // there is none here, so it is empty even though "k" is what makes this
    // report fillable at all (resolveDialogOutcome, connector-runtime-api.ts).
    expect(gates.outcome).toEqual({ kind: "fillable", blocking: [] })
    expect(gates.actions).toEqual(["saveAndResend", "saveOnly"])
    expect(gates.hasSaveEntryPoint).toBe(true)
    expect(gates.canSubmitNow).toBe(false)
    expect(gates.metHoldingSnapshot).toBe(false)
    expect(gates.retryResendDisabled).toBe(true)
  })
})

describe("mergeSendFailureDisposition", () => {
  it.each([
    { previous: null, attempt: null, merged: null },
    { previous: null, attempt: "not_sent", merged: "not_sent" },
    { previous: null, attempt: "rejected", merged: "rejected" },
    { previous: null, attempt: "outcome_unknown", merged: "outcome_unknown" },
    { previous: "not_sent", attempt: "rejected", merged: "rejected" },
    { previous: "rejected", attempt: "not_sent", merged: "not_sent" },
    { previous: "not_sent", attempt: "outcome_unknown", merged: "outcome_unknown" },
    // Uncertainty only accumulates: a later attempt that definitely did not
    // land does not make an earlier unknown one un-sent.
    { previous: "outcome_unknown", attempt: "not_sent", merged: "outcome_unknown" },
    { previous: "outcome_unknown", attempt: "rejected", merged: "outcome_unknown" },
    { previous: "outcome_unknown", attempt: null, merged: "outcome_unknown" },
  ] as const)("($previous, $attempt) -> $merged", ({ previous, attempt, merged }) => {
    expect(mergeSendFailureDisposition(previous, attempt)).toBe(merged)
  })
})

describe("canResendReport", () => {
  it.each([
    { name: "met", report: MET_REPORT, expected: true },
    { name: "fillable", report: FILLABLE_REPORT, expected: false },
    { name: "unsupported_only", report: UNSUPPORTED_ONLY_REPORT, expected: false },
    { name: "nothing_fillable", report: NOTHING_FILLABLE_REPORT, expected: false },
  ])("$name -> $expected", ({ report: r, expected }) => {
    expect(canResendReport(r)).toBe(expected)
  })
})

describe("uniqueKeys", () => {
  it.each([
    { locations: [], keys: [] },
    { locations: [{ key: "a" }], keys: ["a"] },
    { locations: [{ key: "a" }, { key: "b" }, { key: "a" }], keys: ["a", "b"] },
    { locations: [{ key: "b" }, { key: "a" }, { key: "b" }, { key: "a" }], keys: ["b", "a"] },
  ])("$locations.length locations -> $keys", ({ locations, keys }) => {
    expect(uniqueKeys(locations)).toEqual(keys)
  })
})

// ---------------------------------------------------------------------------
// reduceDialog
// ---------------------------------------------------------------------------

const VIEW: View = {
  report: FILLABLE_REPORT,
  reportSeq: 1,
  drafts: {},
  invalidDraftKeys: new Map(),
  fieldError: null,
  lastAlsoResend: false,
}

function shown(phase: Phase, view: Partial<View> = {}): DialogState {
  return { stage: "shown", read: { nonce: 0, settledKey: "1:0" }, view: { ...VIEW, ...view }, phase, verdicts: new Map() }
}

function busyOf(state: DialogState): boolean {
  return state.stage === "shown" && isBusy(state.phase)
}

const NETWORK: ConnectorRuntimeFailureDisposition = { messageKey: "network", retry: true, refresh: false, locate: {} }
const CONFLICT: ConnectorRuntimeFailureDisposition = {
  messageKey: "conflict", retry: false, refresh: true, locate: { connectorRef: REF, key: "k" },
}

const PHASES: Array<[string, Phase]> = [
  ["open", { kind: "open" }],
  ["saving/post", { kind: "saving", step: "post" }],
  ["saving/refresh", { kind: "saving", step: "refresh" }],
  ["sending", { kind: "sending" }],
  ["send-failed", { kind: "send-failed", failure: FAILURE }],
  ["retrying", { kind: "retrying", failure: FAILURE }],
  ["retrying, panel recycled", { kind: "retrying", failure: null }],
]
const BUSY_PHASES = PHASES.filter(([, phase]) => isBusy(phase))
const STATES: Array<[string, DialogState]> = [
  ["hidden", INITIAL_DIALOG_STATE],
  ...PHASES.map(([name, phase]): [string, DialogState] => [name, shown(phase, { fieldError: NETWORK })]),
]

const FINISHING: FinishingEvent[] = [
  { type: "superseded", retryAttempt: null },
  { type: "superseded", retryAttempt: { merged: "outcome_unknown" } },
  { type: "save-rejected", disposition: NETWORK },
  { type: "reject-refresh-settled", disposition: CONFLICT, refreshed: { report: MET_REPORT, seq: 1 } },
  { type: "reject-refresh-settled", disposition: CONFLICT, refreshed: null },
  { type: "save-landed", report: MET_REPORT, seq: 1 },
  { type: "resend-settled", failure: null },
  { type: "resend-settled", failure: FAILURE },
  { type: "retry-settled", result: { kind: "sent" } },
  { type: "retry-settled", result: { kind: "failed", merged: "outcome_unknown" } },
  { type: "retry-abandoned" },
]

const MID: MidEvent[] = [
  { type: "save-rejected-refreshing", disposition: CONFLICT },
  { type: "save-landed-resending", report: MET_REPORT, seq: 1 },
  { type: "read-settled", key: "2:0", result: { kind: "installed", report: MET_REPORT, seq: 2 } },
  { type: "read-settled", key: "2:0", result: { kind: "kept" } },
  { type: "resend-failed", snapshotId: "cid-1", verdict: "outcome_unknown" },
]

// The phase each finishing event ends. `superseded` ends whichever flow is out.
const ENDS: Record<Exclude<FinishingEvent["type"], "superseded">, Phase["kind"]> = {
  "save-rejected": "saving",
  "reject-refresh-settled": "saving",
  "save-landed": "saving",
  "resend-settled": "sending",
  "retry-settled": "retrying",
  "retry-abandoned": "send-failed",
}

function combos<A, B extends { type: string }>(as: Array<[string, A]>, bs: B[]): Array<[string, string, A, B]> {
  return as.flatMap(([name, a]) => bs.map((b): [string, string, A, B] => [name, b.type, a, b]))
}

describe("reduceDialog", () => {
  // A flow that has ended must never leave the dialog busy: a busy flag
  // nothing will lower again holds every save button and every way of
  // closing the dialog shut for good. Arriving in another flow's phase, it
  // must change nothing, or it would release that flow's hold.
  it.each(combos(STATES, FINISHING))("a finishing event ends only its own phase: %s + %s", (_name, _type, state, event) => {
    const next = reduceDialog(state, event)
    if (state.stage !== "shown" || (event.type !== "superseded" && ENDS[event.type] !== state.phase.kind)) {
      expect(next).toBe(state)
      return
    }
    expect(busyOf(next)).toBe(false)
    // The request moved on while the flow was out: nothing the flow learned
    // may be installed or pinned onto whatever the dialog now shows.
    if (event.type === "superseded" && state.stage === "shown" && next.stage === "shown") {
      expect(next.view.fieldError).toBe(state.view.fieldError)
      expect(next.view.report).toBe(state.view.report)
    }
    // A landed save installs a fresh report: a rejection shown before it must
    // go, not linger next to the report that reversed it. Only the POST step
    // is the save this event actually lands (see reduceDialog's own step
    // check); arriving during the refresh step is an unreached cell that
    // only clears the phase, not the field error, the same as the other
    // finishing events checked against a step they were not started from.
    if (event.type === "save-landed" && state.phase.kind === "saving" && state.phase.step === "post" && next.stage === "shown") {
      expect(next.view.fieldError).toBeNull()
    }
  })

  // The re-read a rejected save asks for, the resend a landed save runs, and
  // any read or failed attempt that settles in between all keep the flow
  // going: the save buttons must stay closed until the flow itself ends.
  it.each(combos(BUSY_PHASES, MID))("a mid-flow event keeps a busy phase busy: %s + %s", (_name, _type, phase, event) => {
    const next = reduceDialog(shown(phase), event)
    expect(busyOf(next)).toBe(true)
    const expected: Phase = phase.kind === "saving" && phase.step === "post"
      ? event.type === "save-rejected-refreshing"
        ? { kind: "saving", step: "refresh" }
        : event.type === "save-landed-resending" ? { kind: "sending" } : phase
      : phase
    expect(next.stage === "shown" && next.phase).toEqual(expected)
  })

  it("settles a superseded retry back onto its panel, carrying what the attempt established", () => {
    const next = reduceDialog(
      shown({ kind: "retrying", failure: FAILURE }),
      { type: "superseded", retryAttempt: { merged: "outcome_unknown" } },
    )
    expect(next.stage === "shown" && next.phase).toEqual({
      kind: "send-failed",
      failure: { snapshotId: "cid-1", disposition: "outcome_unknown" },
    })
  })

  // Two places install a report under a live rejection, and they reconcile
  // against different things: a read checks whatever is on screen, the
  // re-read a rejection asked for checks that rejection. Each has three
  // answers -- the hint stays, is re-derived, or goes.
  describe("reconciles a type hint when a report is installed", () => {
    const typed = (type: ConnectorRuntimeType, satisfied = false): ConnectorRuntimeReport => report(satisfied, [
      connector([input({ section: "context", key: "k", type, required: true, satisfied })]),
    ])
    const hint = (messageKey: "typeString" | "typeUnknown"): ConnectorRuntimeFailureDisposition => ({
      messageKey, retry: false, refresh: true, locate: { connectorRef: REF, key: "k" },
    })
    const STRING_HINT = hint("typeString")
    const UNKNOWN_HINT = hint("typeUnknown")

    it.each([
      ["kept when the row still declares the type the hint names", STRING_HINT, typed("string"), "same"],
      ["re-derived when the row finally declares a type", UNKNOWN_HINT, typed("string"), { ...UNKNOWN_HINT, messageKey: "typeString" }],
      ["cleared when the row now declares the other type", STRING_HINT, typed("object"), null],
      ["cleared by a met report too", STRING_HINT, typed("object", true), null],
    ] as const)("on a read: %s", (_name, fieldError, fresh, expected) => {
      const next = reduceDialog(
        shown({ kind: "sending" }, { fieldError }),
        { type: "read-settled", key: "2:0", result: { kind: "installed", report: fresh, seq: 2 } },
      )
      if (next.stage !== "shown") throw new Error("expected a shown dialog")
      expect(next.view.report).toBe(fresh)
      expect(next.view.reportSeq).toBe(2)
      if (expected === "same") expect(next.view.fieldError).toBe(fieldError)
      else expect(next.view.fieldError).toEqual(expected)
    })

    it.each([
      ["kept when the row still declares the type the hint names", STRING_HINT, typed("string"), "same"],
      ["re-derived when the row finally declares a type", UNKNOWN_HINT, typed("string"), { ...UNKNOWN_HINT, messageKey: "typeString" }],
      ["cleared when the row now declares the other type", STRING_HINT, typed("object"), null],
    ] as const)("on the refresh a rejection asked for: %s", (_name, rejection, fresh, expected) => {
      const next = reduceDialog(
        shown({ kind: "saving", step: "refresh" }, { fieldError: rejection }),
        { type: "reject-refresh-settled", disposition: rejection, refreshed: { report: fresh, seq: 1 } },
      )
      if (next.stage !== "shown") throw new Error("expected a shown dialog")
      expect(next.phase).toEqual({ kind: "open" })
      expect(next.view.report).toBe(fresh)
      if (expected === "same") expect(next.view.fieldError).toBe(rejection)
      else expect(next.view.fieldError).toEqual(expected)
    })

    it("on the refresh a rejection asked for: checks that rejection, not what is on screen", () => {
      const next = reduceDialog(
        shown({ kind: "saving", step: "refresh" }, { fieldError: null }),
        { type: "reject-refresh-settled", disposition: UNKNOWN_HINT, refreshed: { report: typed("string"), seq: 1 } },
      )
      expect(next.stage === "shown" && next.view.fieldError).toEqual({ ...UNKNOWN_HINT, messageKey: "typeString" })
    })
  })

  it("keeps the rejection and the old report when the refresh after it fails", () => {
    const state = shown({ kind: "saving", step: "refresh" }, { fieldError: CONFLICT })
    const next = reduceDialog(state, { type: "reject-refresh-settled", disposition: CONFLICT, refreshed: null })
    if (next.stage !== "shown" || state.stage !== "shown") throw new Error("expected a shown dialog")
    expect(next.phase).toEqual({ kind: "open" })
    expect(next.view.report).toBe(state.view.report)
    expect(next.view.fieldError).toBe(CONFLICT)
  })

  // Dispatched from render when the request stops carrying the held
  // failure's snapshot: one dispatch must make the condition false, or the
  // render loops, and a retry still out must stay busy.
  it.each([
    ["the panel", { kind: "send-failed", failure: FAILURE }, false],
    ["a retry off the panel", { kind: "retrying", failure: FAILURE }, true],
  ] as const)("recycles %s in one step when its snapshot is gone", (_name, phase, stillBusy) => {
    const request = { seq: 1, resendPayload: null }
    const state = shown(phase)
    expect(deriveGates(gateFactsOf(state, request)).needsSnapshotRecycle).toBe(true)
    const next = reduceDialog(state, { type: "snapshot-gone" })
    expect(deriveGates(gateFactsOf(next, request)).needsSnapshotRecycle).toBe(false)
    expect(busyOf(next)).toBe(stillBusy)
  })

  // deriveGates is written for fact sets where a retry out is also busy;
  // gateFactsOf is the only thing that builds them from the dialog's state,
  // so it must hold that for every state the dialog can be in.
  it.each(STATES)("reads a retry out as busy too: %s", (_name, state) => {
    const facts = gateFactsOf(state, { seq: 1, resendPayload: { clientMessageId: "cid-1" } })
    expect(facts.retrying).toBe(state.stage === "shown" && state.phase.kind === "retrying")
    if (facts.retrying) expect(facts.busy).toBe(true)
  })

  it("stays hidden, with the attempt recorded, when a read keeps nothing to show", () => {
    const next = reduceDialog(INITIAL_DIALOG_STATE, { type: "read-settled", key: "1:0", result: { kind: "kept" } })
    expect(next).toEqual({ stage: "hidden", read: { nonce: 0, settledKey: "1:0" } })
    expect(reduceDialog(next, { type: "read-again" }).read).toEqual({ nonce: 1, settledKey: "1:0" })
  })

  it.each<[string, DialogEvent]>([
    ["draft-changed", { type: "draft-changed", draftKey: DRAFT_KEY, value: "v" }],
    ["save-started", { type: "save-started", alsoResend: true }],
    ["snapshot-gone", { type: "snapshot-gone" }],
  ])("ignores %s while hidden", (_name, event) => {
    expect(reduceDialog(INITIAL_DIALOG_STATE, event)).toBe(INITIAL_DIALOG_STATE)
  })

  // The switches are exhaustive at compile time; a value outside the union
  // (only reachable by casting) fails loudly instead of being ignored.
  it("throws on an event or phase outside the union", () => {
    expect(() => reduceDialog(shown({ kind: "open" }), { type: "unknown" } as unknown as DialogEvent)).toThrow()
    expect(() => isBusy({ kind: "unknown" } as unknown as Phase)).toThrow()
    expect(() => reduceDialog(
      shown({ kind: "unknown" } as unknown as Phase),
      { type: "superseded", retryAttempt: null },
    )).toThrow()
  })

  it("keeps every failed attempt's verdict per snapshot", () => {
    const state = reduceDialog(shown({ kind: "sending" }), { type: "resend-failed", snapshotId: "cid-1", verdict: "outcome_unknown" })
    expect(state.stage === "shown" && state.verdicts.get("cid-1")).toBe("outcome_unknown")
  })

  it.each([
    ["save-started outside the rows", { kind: "sending" }, { type: "save-started", alsoResend: false }],
    ["retry-started with no panel", { kind: "open" }, { type: "retry-started" }],
    ["snapshot-gone with nothing held", { kind: "open" }, { type: "snapshot-gone" }],
    ["save-landed-resending outside a save POST", { kind: "sending" }, { type: "save-landed-resending", report: MET_REPORT, seq: 1 }],
    ["save-rejected-refreshing outside a save POST", { kind: "open" }, { type: "save-rejected-refreshing", disposition: CONFLICT }],
  ] as Array<[string, Phase, DialogEvent]>)("returns the state unchanged for %s", (_name, phase, event) => {
    const state = shown(phase)
    expect(reduceDialog(state, event)).toBe(state)
  })
})

// ---------------------------------------------------------------------------
// The dialog's endings go through its exits
// ---------------------------------------------------------------------------

describe("the dialog's endings", () => {
  // Comments stripped, so prose that mentions a name does not count as one.
  const source = readFileSync(path.resolve(__dirname, "./connector-runtime-dialog.tsx"), "utf8")
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/(^|[^:])\/\/.*$/gm, "$1")

  function within(src: string, index: number, ...names: string[]): boolean {
    return names.some((name) => {
      const start = src.indexOf(`  const ${name} = `)
      expect(start, name).toBeGreaterThanOrEqual(0)
      return index > start && index < src.indexOf("\n  }\n", start)
    })
  }
  function lineAt(src: string, index: number): string {
    return src.slice(src.lastIndexOf("\n", index) + 1, src.indexOf("\n", index)).trim()
  }

  const FINISHING_TYPES = [
    "superseded",
    "save-rejected",
    "reject-refresh-settled",
    "save-landed",
    "resend-settled",
    "retry-settled",
    "retry-abandoned",
  ] satisfies Array<FinishingEvent["type"]>

  // Every mention of the name counts, not only a call spelled `name(`, so a
  // method on it (`toast.error(`), a call on some other object (`ctl.close(`)
  // or an alias is caught too. Each rule lists the only places a mention may
  // stand and returns the lines of the ones that stand anywhere else.
  const RULES = {
    toast: (src: string) => strays(src, /\btoast\b/g, (i, line) => line.startsWith("import ") || within(src, i, "say")),
    close: (src: string) => strays(src, /\bclose\b/g, (i, line) => (
      line === "const { close } = useConnectorRuntimeDialog()"
      || line.endsWith(", close])")
      || src.startsWith("close: \"", i)
      || within(src, i, "finish")
    )),
    // Outside the exits, only a literal non-finishing event, or the read
    // effect's own "kept" read-settled event, may be dispatched.
    dispatch: (src: string) => strays(src, /\bdispatch\b/g, (i, line) => {
      if (line.startsWith("const [state, dispatch] = useReducer(") || within(src, i, "finish", "settle")) return true
      if (!src.startsWith("dispatch(", i)) return false
      const argument = src.slice(i + "dispatch(".length, i + "dispatch(".length + 80)
      const literal = /^\{\s*type:\s*"([a-z-]+)"/.exec(argument)
      return literal ? !(FINISHING_TYPES as string[]).includes(literal[1]) : argument.startsWith("kept)")
    }),
  }
  function strays(src: string, name: RegExp, allowed: (index: number, line: string) => boolean): string[] {
    return Array.from(src.matchAll(name), m => m.index ?? -1)
      .filter(index => !allowed(index, lineAt(src, index)))
      .map(index => lineAt(src, index))
  }

  it.each(["toast", "close", "dispatch"] as const)("names %s only where the exits allow", (rule) => {
    expect(RULES[rule](source)).toEqual([])
  })

  it.each([
    ["toast", "toast.error(\"x\")"],
    ["toast", "const shout = toast"],
    ["close", "ctl.close(\"dismissed\")"],
    ["close", "const ctl = { close }"],
    ["close", "const { close: shut } = useConnectorRuntimeDialog()"],
    ["dispatch", "dispatch({ type: \"retry-abandoned\" })"],
    ["dispatch", "const send = dispatch"],
  ] as const)("catches a stray %s: %s", (rule, stray) => {
    const at = source.indexOf("  const handleDismiss = ")
    expect(RULES[rule](`${source.slice(0, at)}  ${stray}\n${source.slice(at)}`)).toEqual([stray])
  })
})

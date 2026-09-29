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
  opensOnFirstRead,
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
    trigger: "turn_failure",
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
  lastResendFor: null,
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

const FINISHING: Array<[string, FinishingEvent]> = [
  ["superseded", { type: "superseded", retryAttempt: null }],
  ["superseded, retry failed", { type: "superseded", retryAttempt: { merged: "outcome_unknown" } }],
  ["save-rejected", { type: "save-rejected", disposition: NETWORK }],
  ["reject-refresh-settled", { type: "reject-refresh-settled", disposition: CONFLICT, refreshed: { report: MET_REPORT, seq: 1 } }],
  ["reject-refresh-settled, re-read failed", { type: "reject-refresh-settled", disposition: CONFLICT, refreshed: null }],
  ["save-landed", { type: "save-landed", report: MET_REPORT, seq: 1 }],
  ["resend-settled, sent", { type: "resend-settled", failure: null }],
  ["resend-settled, failed", { type: "resend-settled", failure: FAILURE }],
  ["retry-settled, sent", { type: "retry-settled", result: { kind: "sent" } }],
  ["retry-settled, failed", { type: "retry-settled", result: { kind: "failed", merged: "outcome_unknown" } }],
  ["retry-abandoned", { type: "retry-abandoned" }],
]

const MID: MidEvent[] = [
  { type: "save-rejected-refreshing", disposition: CONFLICT },
  { type: "save-landed-resending", report: MET_REPORT, seq: 1 },
  { type: "read-settled", key: "2:0", result: { kind: "installed", report: MET_REPORT, seq: 2 } },
  { type: "read-settled", key: "2:0", result: { kind: "kept" } },
  { type: "resend-failed", snapshotId: "cid-1", verdict: "outcome_unknown" },
]

const OPEN_PHASE: Phase = { kind: "open" }
const PANEL: Phase = { kind: "send-failed", failure: FAILURE }
const PANEL_RETRY_UNKNOWN: Phase = { kind: "send-failed", failure: { ...FAILURE, disposition: "outcome_unknown" } }

// Every cell where a finishing event applies, and the phase it must leave.
// Each event ends one stage -- the save's two steps count apart -- and
// `superseded` ends whichever flow is out. A cell missing here must leave
// the state untouched.
const APPLIES: Record<string, Phase> = {
  "open + superseded": OPEN_PHASE,
  "saving/post + superseded": OPEN_PHASE,
  "saving/refresh + superseded": OPEN_PHASE,
  "sending + superseded": OPEN_PHASE,
  "send-failed + superseded": PANEL,
  // A superseded retry that learned nothing new goes back to its panel
  // with the failure it was pressed on.
  "retrying + superseded": PANEL,
  "retrying, panel recycled + superseded": OPEN_PHASE,
  "open + superseded, retry failed": OPEN_PHASE,
  "saving/post + superseded, retry failed": OPEN_PHASE,
  "saving/refresh + superseded, retry failed": OPEN_PHASE,
  "sending + superseded, retry failed": OPEN_PHASE,
  "send-failed + superseded, retry failed": PANEL,
  // ...and one that failed carries what it established onto that panel.
  "retrying + superseded, retry failed": PANEL_RETRY_UNKNOWN,
  "retrying, panel recycled + superseded, retry failed": OPEN_PHASE,
  "saving/post + save-rejected": OPEN_PHASE,
  "saving/refresh + reject-refresh-settled": OPEN_PHASE,
  "saving/refresh + reject-refresh-settled, re-read failed": OPEN_PHASE,
  "saving/post + save-landed": OPEN_PHASE,
  "sending + resend-settled, sent": OPEN_PHASE,
  "sending + resend-settled, failed": PANEL,
  "retrying + retry-settled, sent": OPEN_PHASE,
  "retrying, panel recycled + retry-settled, sent": OPEN_PHASE,
  "retrying + retry-settled, failed": PANEL_RETRY_UNKNOWN,
  "retrying, panel recycled + retry-settled, failed": OPEN_PHASE,
  "send-failed + retry-abandoned": OPEN_PHASE,
}

function combos<A, B>(as: Array<[string, A]>, bs: Array<[string, B]>): Array<[string, string, A, B]> {
  return as.flatMap(([aName, a]) => bs.map(([bName, b]): [string, string, A, B] => [aName, bName, a, b]))
}

describe("reduceDialog", () => {
  // A flow that has ended must never leave the dialog busy: a busy flag
  // nothing will lower again holds every save button and every way of
  // closing the dialog shut for good. Arriving in another flow's phase, it
  // must change nothing, or it would release that flow's hold.
  it.each(combos(STATES, FINISHING))("a finishing event ends only its own stage: %s + %s", (name, eventName, state, event) => {
    const next = reduceDialog(state, event)
    const expected = APPLIES[`${name} + ${eventName}`]
    if (expected === undefined) {
      expect(next).toBe(state)
      return
    }
    if (state.stage !== "shown" || next.stage !== "shown") throw new Error("expected a shown dialog")
    expect(next.phase).toEqual(expected)
    expect(busyOf(next)).toBe(false)
    // The request moved on while the flow was out: nothing the flow learned
    // may be installed or pinned onto whatever the dialog now shows.
    if (event.type === "superseded") {
      expect(next.view.fieldError).toBe(state.view.fieldError)
      expect(next.view.report).toBe(state.view.report)
    }
    // A landed save installs a fresh report: a rejection shown before it must
    // go, not linger next to the report that reversed it.
    if (event.type === "save-landed") {
      expect(next.view.fieldError).toBeNull()
    }
  })

  it("lists only cells the table walks", () => {
    const walked = new Set(combos(STATES, FINISHING).map(([name, eventName]) => `${name} + ${eventName}`))
    expect(Object.keys(APPLIES).filter(cell => !walked.has(cell))).toEqual([])
  })

  // The re-read a rejected save asks for, the resend a landed save runs, and
  // any read or failed attempt that settles in between all keep the flow
  // going: the save buttons must stay closed until the flow itself ends.
  it.each(combos(BUSY_PHASES, MID.map((event): [string, MidEvent] => [event.type, event])))("a mid-flow event keeps a busy phase busy: %s + %s", (_name, _type, phase, event) => {
    const next = reduceDialog(shown(phase), event)
    expect(busyOf(next)).toBe(true)
    const expected: Phase = phase.kind === "saving" && phase.step === "post"
      ? event.type === "save-rejected-refreshing"
        ? { kind: "saving", step: "refresh" }
        : event.type === "save-landed-resending" ? { kind: "sending" } : phase
      : phase
    expect(next.stage === "shown" && next.phase).toEqual(expected)
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
    expect(deriveGates(gateFactsOf(state, request, "turn_failure")).needsSnapshotRecycle).toBe(true)
    const next = reduceDialog(state, { type: "snapshot-gone" })
    expect(deriveGates(gateFactsOf(next, request, "turn_failure")).needsSnapshotRecycle).toBe(false)
    expect(busyOf(next)).toBe(stillBusy)
  })

  // deriveGates is written for fact sets where a retry out is also busy;
  // gateFactsOf is the only thing that builds them from the dialog's state,
  // so it must hold that for every state the dialog can be in.
  it.each(STATES)("reads a retry out as busy too: %s", (_name, state) => {
    const facts = gateFactsOf(state, { seq: 1, resendPayload: { clientMessageId: "cid-1" } }, "turn_failure")
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
    ["save-started", { type: "save-started", resendFor: "orig-1" }],
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
    ["save-started outside the rows", { kind: "sending" }, { type: "save-started", resendFor: null }],
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
  // A text scan of the dialog's source, not a parse. What it relies on is
  // only what the rules below spell out: the exits are arrow functions bound
  // with `const <name> =`, and each allowed mention has the shape its rule
  // names. Whitespace, line breaks, trailing commas and the name of the
  // value the read effect dispatches do not matter, and the last test here
  // pins that. A new shape outside these fails the scan with no behaviour
  // change; the rule it fails is the thing to extend.
  //
  // Comments are stripped and string literals kept whole, so prose that
  // mentions a name does not count as one, and a "//" inside a string does
  // not swallow the rest of its line.
  function stripComments(src: string): string {
    return src.replace(
      /("(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`)|\/\*[\s\S]*?\*\/|\/\/[^\n]*/g,
      (match: string, literal: string | undefined) => literal ?? (match.startsWith("/*") ? " " : ""),
    )
  }
  const raw = readFileSync(path.resolve(__dirname, "./connector-runtime-dialog.tsx"), "utf8")
  const source = stripComments(raw)

  // Whether `index` falls inside the body of the arrow function bound to any
  // of `names`, found by balancing braces from the body's opening one.
  function within(src: string, index: number, ...names: string[]): boolean {
    return names.some((name) => {
      const head = new RegExp(`\\bconst\\s+${name}\\s*=[^]*?=>\\s*\\{`).exec(src)
      expect(head, name).not.toBeNull()
      if (!head) return false
      const open = head.index + head[0].length - 1
      let depth = 0
      for (let at = open; at < src.length; at++) {
        if (src[at] === "{") depth++
        else if (src[at] === "}" && --depth === 0) return index > open && index < at
      }
      return false
    })
  }
  // Whether the text right before `index` ends with `before` and the text
  // from `index` on starts with `after`.
  function around(src: string, index: number, before: RegExp, after: RegExp): boolean {
    return before.test(src.slice(Math.max(0, index - 200), index)) && after.test(src.slice(index))
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
  const isFinishing = (type: string) => (FINISHING_TYPES as string[]).includes(type)

  // Names bound to a literal event that does not end a flow, such as the
  // read effect's own "kept" read-settled event.
  function nonFinishingBindings(src: string): Set<string> {
    return new Set(Array.from(
      src.matchAll(/\bconst\s+([A-Za-z_$][\w$]*)\s*=\s*\{\s*type\s*:\s*"([a-z-]+)"/g),
      m => m[2] !== undefined && !isFinishing(m[2]) ? m[1] : "",
    ).filter(Boolean))
  }

  // Every mention of the name counts, not only a call spelled `name(`, so a
  // method on it (`toast.error(`), a call on some other object (`ctl.close(`)
  // or an alias is caught too. Each rule lists the only places a mention may
  // stand and returns the lines of the ones that stand anywhere else.
  const RULES = {
    toast: (src: string) => strays(src, /\btoast\b/g, i => (
      around(src, i, /\bimport\s*\{[^}]*$/, /^toast\s*[,}]/)
      || within(src, i, "say")
    )),
    close: (src: string) => strays(src, /\bclose\b/g, i => (
      around(src, i, /\bconst\s*\{\s*$/, /^close\s*,?\s*\}\s*=\s*useConnectorRuntimeDialog\s*\(\s*\)/)
      // The last dependency of an effect that reads it.
      || around(src, i, /[[,]\s*$/, /^close\s*,?\s*\]\s*\)/)
      || /^close\s*:\s*"/.test(src.slice(i))
      || within(src, i, "finish")
    )),
    // Outside the exits, only a literal non-finishing event, or a name bound
    // to one, may be dispatched.
    dispatch: (src: string) => strays(src, /\bdispatch\b/g, (i) => {
      if (around(src, i, /\bconst\s*\[\s*state\s*,\s*$/, /^dispatch\s*,?\s*\]\s*=\s*useReducer\s*\(/)) return true
      if (within(src, i, "finish", "settle")) return true
      const call = /^dispatch\s*\(\s*/.exec(src.slice(i))
      if (!call) return false
      const argument = src.slice(i + call[0].length)
      const literal = /^\{\s*type\s*:\s*"([a-z-]+)"/.exec(argument)
      if (literal) return literal[1] !== undefined && !isFinishing(literal[1])
      const name = /^([A-Za-z_$][\w$]*)\s*,?\s*\)/.exec(argument)
      return name !== null && name[1] !== undefined && nonFinishingBindings(src).has(name[1])
    }),
  }
  function strays(src: string, name: RegExp, allowed: (index: number) => boolean): string[] {
    return Array.from(src.matchAll(name), m => m.index ?? -1)
      .filter(index => !allowed(index))
      .map(index => lineAt(src, index))
  }

  it.each(["toast", "close", "dispatch"] as const)("names %s only where the exits allow", (rule) => {
    expect(RULES[rule](source)).toEqual([])
  })

  function insertBeforeDismiss(src: string, line: string): string {
    const at = src.indexOf("  const handleDismiss = ")
    expect(at).toBeGreaterThanOrEqual(0)
    return `${src.slice(0, at)}  ${line}\n${src.slice(at)}`
  }

  it.each([
    ["toast", "toast.error(\"x\")"],
    ["toast", "const shout = toast"],
    ["close", "ctl.close(\"dismissed\")"],
    ["close", "const ctl = { close }"],
    ["close", "const { close: shut } = useConnectorRuntimeDialog()"],
    ["dispatch", "dispatch({ type: \"retry-abandoned\" })"],
    ["dispatch", "const send = dispatch"],
    ["dispatch", "dispatch(someEvent)"],
  ] as const)("catches a stray %s: %s", (rule, stray) => {
    expect(RULES[rule](insertBeforeDismiss(source, stray))).toEqual([stray])
  })

  it("does not let a \"//\" inside a string hide what follows it", () => {
    const stray = "const note = \"a // b\"; toast(\"x\")"
    expect(RULES.toast(stripComments(insertBeforeDismiss(raw, stray)))).toEqual([stray])
  })

  // Layout the rules must not depend on: every line indented further, the
  // read effect's event under another name, and the destructurings, a
  // dependency list and a dispatch call each spread over several lines.
  // The same strays must still be caught in it, so a rule cannot pass the
  // new layout by allowing more than it did.
  it("passes the same source laid out differently, and still catches strays in it", () => {
    const edits: Array<[string | RegExp, string]> = [
      [/\bkept\b/g, "keptRead"],
      ["const { close } = useConnectorRuntimeDialog()", "const {\n    close,\n  } = useConnectorRuntimeDialog()"],
      ["const [state, dispatch] = useReducer(", "const [\n    state,\n    dispatch,\n  ] = useReducer("],
      ["[visible, pathname, close])", "[\n    visible,\n    pathname,\n    close,\n  ])"],
      ["dispatch({ type: \"snapshot-gone\" })", "dispatch(\n      { type: \"snapshot-gone\" },\n    )"],
      ["import { toast } from", "import {\n  toast,\n} from"],
      [/\n/g, "\n  "],
    ]
    const relaidOut = edits.reduce((src, [from, to]) => {
      const next = src.replace(from, to)
      expect(next, String(from)).not.toBe(src)
      return next
    }, source)
    for (const [rule, stray] of [
      ["toast", "toast.error(\"x\")"],
      ["close", "ctl.close(\"dismissed\")"],
      ["dispatch", "dispatch({ type: \"retry-abandoned\" })"],
    ] as const) {
      expect(RULES[rule](relaidOut), rule).toEqual([])
      expect(RULES[rule](insertBeforeDismiss(relaidOut, stray)), rule).toEqual([stray])
    }
  })
})

// Whether the read that would first show the dialog shows it, per trigger
// and per outcome: every one of the twelve pairs spelled out.
describe("opensOnFirstRead", () => {
  it.each([
    ["turn_failure", "met", false],
    ["turn_failure", "unsupported_only", true],
    ["turn_failure", "nothing_fillable", true],
    ["turn_failure", "fillable", true],
    ["session_open", "met", false],
    ["session_open", "unsupported_only", false],
    ["session_open", "nothing_fillable", false],
    ["session_open", "fillable", true],
    ["first_gate", "met", false],
    ["first_gate", "unsupported_only", false],
    ["first_gate", "nothing_fillable", false],
    ["first_gate", "fillable", true],
  ] as const)("%s + %s -> %s", (trigger, kind, opens) => {
    expect(opensOnFirstRead(kind, trigger)).toBe(opens)
  })
})

// A first gate's button set comes from its own function, and "save and send"
// counts as a way to save, so its rows are editable.
describe("deriveGates for a first gate", () => {
  it.each([
    ["fillable", ["saveAndSend"], true, FILLABLE_REPORT],
    ["met", ["sendHeld"], false, MET_REPORT],
    ["unsupported_only", ["sendHeld"], false, UNSUPPORTED_ONLY_REPORT],
    ["nothing_fillable", ["sendHeld"], false, NOTHING_FILLABLE_REPORT],
  ] as const)("%s -> %o, save entry point %s", (_kind, actions, canSave, r) => {
    const gates = deriveGates(baseFacts({ report: r, reportSeq: 1, trigger: "first_gate" }))
    expect(gates.actions).toEqual(actions)
    expect(gates.hasSaveEntryPoint).toBe(canSave)
  })

  it.each(["turn_failure", "session_open"] as const)("keeps %s on the other button sets", (trigger) => {
    expect(deriveGates(baseFacts({ report: FILLABLE_REPORT, reportSeq: 1, trigger })).actions).toEqual(["saveOnly"])
  })

  it("carries the trigger it is given", () => {
    expect(gateFactsOf(INITIAL_DIALOG_STATE, { seq: 1, resendPayload: null }, "first_gate").trigger).toBe("first_gate")
  })
})

// A first gate and a session check add no phase and no event: the dialog's
// state machine is the one the turn-failure dialog already had.
describe("keeps the dialog's phases and events as they were", () => {
  const stateSource = readFileSync(path.resolve(__dirname, "./connector-runtime-dialog-state.ts"), "utf8")
  const between = (from: string, to: string) => stateSource.slice(stateSource.indexOf(from), stateSource.indexOf(to, stateSource.indexOf(from)))
  it("names the same phases", () => {
    expect(Array.from(between("export type Phase =", "\n\n").matchAll(/kind: "([a-z-]+)"/g), m => m[1]))
      .toEqual(["open", "saving", "sending", "send-failed", "retrying"])
  })
  it("names the same events", () => {
    expect(Array.from(between("export type FinishingEvent =", "export type DialogEvent").matchAll(/\btype: "([a-z-]+)"/g), m => m[1])).toEqual([
      "superseded", "save-rejected", "reject-refresh-settled", "save-landed", "resend-settled", "retry-settled",
      "retry-abandoned", "save-rejected-refreshing", "save-landed-resending", "read-settled", "resend-failed",
      "read-again", "draft-changed", "object-blurred", "save-started", "retry-started", "snapshot-gone",
    ])
    expect(stateSource).toContain("\nexport type DialogEvent = FinishingEvent | MidEvent | LocalEvent\n")
  })
})

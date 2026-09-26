// The connector-runtime dialog's state, the one reducer that moves it, and
// the pure predicates that decide what it means, kept apart from
// connector-runtime-dialog.tsx so they can be data for a table-driven test
// instead of only reachable through rendering. No React, no i18n, no
// translation keys: nothing here may depend on how the dialog draws itself
// or what it says.

import { sendOutcomeMayHaveLanded } from "@/components/chat/clarification-delivery"
// Type-only: naming the close outcomes adds no runtime dependency on the
// dialog's React context.
import type { ConnectorRuntimeDialogCloseOutcome } from "@/contexts/connector-runtime-dialog-context"
// Type-only, like clarification-delivery's own import of it: naming the
// disposition union here adds no runtime dependency on the websocket hook.
import type { MessageDeliveryDisposition } from "@/hooks/use-websocket"
import {
  buildSubmitItems,
  connectorRuntimeInputDraftKey,
  isSubmitEnabled,
  reconcileTypeMismatchDisposition,
  resolveDialogActions,
  resolveDialogOutcome,
  type ConnectorRuntimeConnector,
  type ConnectorRuntimeDialogAction,
  type ConnectorRuntimeErrorMessageKey,
  type ConnectorRuntimeFailureDisposition,
  type ConnectorRuntimeInput,
  type ConnectorRuntimeReport,
  type ConnectorRuntimeSubmitItem,
  type DialogOutcome,
} from "@/lib/connector-runtime-api"

// Why a draft failed the object-field blur check: "invalid" for anything
// that is not JSON-object-shaped, "empty" for `{}`, which parses fine but
// isSubmittableObjectValue (connector-runtime-api.ts) rejects because the
// server treats it as a blank context value. The row's error message reads
// this to show the reason-specific hint instead of a generic one.
export type InvalidObjectDraftReason = "invalid" | "empty"

/**
 * Whether an invalid-object mark for this input is still live. Only a
 * `context` row the current report leaves unsatisfied and still declares
 * `object`-typed renders the textarea whose blur handler can clear such a
 * mark; against any other row the mark is unreachable. Both the submit gate
 * and the row's own error message read this one predicate, so the button can
 * never be disabled by an error the row does not show, and the row can never
 * show an error that leaves the button enabled.
 */
export function hasLiveInvalidObjectMark(
  connector: ConnectorRuntimeConnector,
  input: ConnectorRuntimeInput,
  invalidDraftKeys: ReadonlyMap<string, InvalidObjectDraftReason>,
): boolean {
  return (
    input.section === "context"
    && !input.satisfied
    && input.type === "object"
    && invalidDraftKeys.has(connectorRuntimeInputDraftKey(connector.connector_ref, input.section, input.key, input.type))
  )
}

export function uniqueKeys(locations: Array<{ key: string }>): string[] {
  return Array.from(new Set(locations.map(l => l.key)))
}

/**
 * The disposition the send-failed panel carries after one more attempt for
 * the same snapshot. Uncertainty only ever accumulates: once any attempt
 * ended with its outcome unknown, a later one the server definitely refused
 * does not make the earlier one un-sent, so neither the panel nor anything
 * standing in for it may fall back to saying the message never went out.
 *
 * One function rather than one rule in the panel and another wherever a
 * toast reports the same attempt: the two are read by the same user, seconds
 * apart, about one message.
 */
export function mergeSendFailureDisposition(
  previous: MessageDeliveryDisposition | null,
  attempt: MessageDeliveryDisposition | null,
): MessageDeliveryDisposition | null {
  if (sendOutcomeMayHaveLanded(previous) || sendOutcomeMayHaveLanded(attempt)) return "outcome_unknown"
  return attempt
}

/**
 * Whether the report currently in hand can carry a resend of the message
 * this dialog is holding. Only a met report can: `unsupported_only` still
 * lacks a required secret this dialog cannot collect, `nothing_fillable` is a
 * connector the server still reports unavailable with nothing left for the
 * user to fill, and `fillable` still has a required context value missing.
 * The backend rejects all three while it builds the turn's tool list, so a
 * resend would fail on the same gate and put a second failure in the
 * conversation.
 *
 * Both entry points into a resend ask this -- handleSave right after its own
 * save lands, and handleRetryResend against the report on screen -- so the
 * two cannot disagree about whether the same snapshot is sendable.
 */
export function canResendReport(report: ConnectorRuntimeReport): boolean {
  return resolveDialogOutcome(report).kind === "met"
}

/**
 * The failed send the "saved but not sent" panel is about: the
 * clientMessageId of the snapshot it names, and the disposition that failure
 * carried, held together so the panel can never word one send's outcome with
 * another's. See `sendFailed` in connector-runtime-dialog.tsx for how the id
 * half is read.
 */
export interface SendFailureState {
  snapshotId: string
  disposition: MessageDeliveryDisposition | null
}

/**
 * Everything deriveGates needs, read fresh off the dialog's own state and
 * the request it is currently showing. Flat facts rather than the dialog's
 * storage shape itself: a caller holding twelve independent pieces of state
 * today can feed this the same way a caller holding one combined state
 * value will tomorrow, and neither has to restate the formulas below --
 * only how to read its own storage into this shape.
 */
export interface GateFacts {
  report: ConnectorRuntimeReport | null
  reportSeq: number | null
  settledReadKey: string | null
  readNonce: number
  // Whether any submission-shaped action is in flight: an explicit save
  // (which may itself run a resend as part of "save and resend") or a
  // standalone retry resend from the send-failed panel. The caller keeps
  // this rather than deriveGates folding it from two flags of its own,
  // because what those flags are and how many there are is the caller's
  // storage question, not this module's.
  //
  // The caller must keep `retrying` implying `busy`: a retry resend is one
  // of the submission-shaped actions `busy` covers. A fact set with
  // `retrying` true and `busy` false describes no state the dialog can be
  // in, and nothing below is written to give it a meaning.
  busy: boolean
  // The send the "saved but not sent" panel is about, or null when no send
  // has failed. See `liveSendFailure` below for why this is joined against
  // the request's own resend payload on every call rather than trusted as
  // it stands.
  heldFailure: SendFailureState | null
  // Whether the send-failed panel's own retry resend is in flight. Implies
  // `busy`; see there.
  retrying: boolean
  drafts: Readonly<Record<string, string>>
  invalidDraftKeys: ReadonlyMap<string, InvalidObjectDraftReason>
  request: { seq: number; resendPayload: { clientMessageId: string } | null }
}

export interface Gates {
  liveSendFailure: SendFailureState | null
  sendFailed: boolean
  needsSnapshotRecycle: boolean
  readKey: string
  reading: boolean
  reportIsStale: boolean
  readFailed: boolean
  outcome: DialogOutcome | null
  submitItems: ConnectorRuntimeSubmitItem[]
  hasInvalidObjectDraft: boolean
  canSubmit: boolean
  canSubmitNow: boolean
  hasResendPayload: boolean
  actions: ConnectorRuntimeDialogAction[]
  hasSaveEntryPoint: boolean
  metHoldingSnapshot: boolean
  retryResendDisabled: boolean
}

/**
 * Every value the connector-runtime dialog derives from its own state and
 * the request it is currently showing, gathered in one place so the same
 * formulas cannot drift between whatever state produces them and whatever
 * renders off them. Everything here is computed fresh from `facts` on every
 * call -- nothing is cached across calls -- so it is the caller's choice how
 * often that happens (today, every render).
 */
export function deriveGates(facts: GateFacts): Gates {
  // Joined against the request's own resend payload on every call, rather
  // than trusted as it stands, because the panel and its retry button must
  // be about the same message on every call, including the first one after
  // a same-task retarget swaps in a newer resend candidate, or a
  // settlement frame takes the snapshot away without moving `seq` at all.
  // Today nothing brings a clientMessageId back once it has stopped
  // matching (a snapshot's own id is written once at send time), so a held
  // failure that stops matching is unreachable rather than wrong.
  // `needsSnapshotRecycle` below tells the caller, which drops it so it
  // cannot become wrong if an id ever does come back.
  const liveSendFailure = facts.heldFailure !== null
    && facts.heldFailure.snapshotId === facts.request.resendPayload?.clientMessageId
    ? facts.heldFailure
    : null
  const sendFailed = liveSendFailure !== null
  // True while `heldFailure` no longer matches the request's resend
  // payload. The caller must drop it -- from render, not an effect, since
  // an effect notices a frame late and never runs at all for a removal that
  // does not move `seq` -- so that the next call's facts no longer carry
  // it. How many calls this stays true for depends on the caller doing so.
  const needsSnapshotRecycle = facts.heldFailure !== null && liveSendFailure === null

  // The read attempt the dialog is currently on: the request it is for, and
  // which try for that request it is. Both halves are needed -- `seq` alone
  // cannot tell a fresh attempt for the same request from the one that just
  // failed, so a "read again" press would leave `reading` false and the
  // screen would say nothing while the retry was out.
  const readKey = `${facts.request.seq}:${facts.readNonce}`
  const reading = facts.settledReadKey !== readKey
  // Whether the report on hand belongs to some earlier request than the one
  // the dialog is now for. A same-task retarget bumps `seq` and starts a
  // fresh read while the previous report is still on screen; submitting or
  // resending against that report acts on rows the current one may no
  // longer declare, and a stored context value is immutable, so there is no
  // correcting it afterwards.
  //
  // This is the gate rather than `reading`, because the two come apart in
  // exactly the case that matters: a read that fails settles without
  // installing anything, so `reading` goes false while the report on hand
  // is still the previous request's. `reportSeq` can only catch up when the
  // attempt that installs a fresher report also records itself settled, so
  // this is true whenever `reading` is -- one gate covers the pending read
  // and the failed one both.
  const reportIsStale = facts.reportSeq !== facts.request.seq
  // The read for this request settled and left the previous request's
  // report on hand: it failed. Derived rather than stored, so it cannot
  // disagree with the two facts it is made of.
  const readFailed = !reading && reportIsStale

  const outcome: DialogOutcome | null = facts.report ? resolveDialogOutcome(facts.report) : null
  const submitItems = facts.report ? buildSubmitItems(facts.report, facts.drafts) : []
  // Only a mark on a row the current report still renders an editable
  // control for may gate submission. A key a refreshed report reports
  // satisfied loses its textarea, so its mark could never be cleared again
  // -- submit would stay disabled with no error anywhere on screen. Reads
  // the same `hasLiveInvalidObjectMark` predicate the row renderer reads
  // for its error message, so the two can never disagree. `invalidDraftKeys`
  // is read-only here the same way it is everywhere else this module reads
  // it: nothing in this module ever needs to write it back.
  const hasInvalidObjectDraft = facts.report !== null && facts.report.connectors.some(connector =>
    connector.inputs.some(input => hasLiveInvalidObjectMark(
      connector,
      input,
      facts.invalidDraftKeys,
    )),
  )
  const canSubmit = isSubmitEnabled(submitItems, hasInvalidObjectDraft)
  // The one value every entry point into a submission reads: both footer
  // save buttons, the retry button a retryable failure offers, and the save
  // handler itself. That matters because buildSubmitItems drops an
  // unparsable object draft instead of failing: such a batch writes every
  // other field, silently loses that one and closes the dialog -- and a
  // stored context value is immutable, so there is no correcting it
  // afterwards.
  //
  // `reportIsStale` is folded in here and not into `busy`, because the
  // caller also uses `busy` to gate dismissal: a read still out for a
  // retargeted request must not stand between the user and closing the
  // dialog. What `busy` itself holds open is the caller's business; see
  // where the dialog reads it for dismissal.
  const canSubmitNow = canSubmit && !facts.busy && !reportIsStale
  const hasResendPayload = facts.request.resendPayload !== null
  const actions = outcome ? resolveDialogActions(outcome, hasResendPayload) : []
  // Whether this shape offers any way to submit. The row renderer asks this
  // instead of listing the outcome kinds that offer none, because that list
  // was one kind short: a `met` report reaches the render whenever one is
  // installed into a dialog that stays open, and an unfilled *optional*
  // context key inside one was still drawn as an editable field with no
  // button able to send it. Derived from the action set, so the rows and
  // the footer cannot disagree about whether saving is possible.
  const hasSaveEntryPoint = actions.includes("saveOnly")
  // A met report that reached the render still carries the snapshot of the
  // message that failed, and this shape offers no way to send it: the
  // footer collapses to "Got it", and the save-and-resend button a
  // fillable report offers cannot be reused here because a met report
  // produces no submittable items, which leaves it permanently disabled.
  // Not while the send-failed panel is up, and not while a submission for
  // this same message is still in flight: saying it was not resent would
  // be false in both cases.
  const metHoldingSnapshot = outcome?.kind === "met" && hasResendPayload && !sendFailed && !facts.busy
  // Mirrors the guard the send-failed panel's own retry button carries: no
  // second attempt while one is already out, and none against a report the
  // current request no longer produced.
  const retryResendDisabled = facts.retrying || reportIsStale

  return {
    liveSendFailure,
    sendFailed,
    needsSnapshotRecycle,
    readKey,
    reading,
    reportIsStale,
    readFailed,
    outcome,
    submitItems,
    hasInvalidObjectDraft,
    canSubmit,
    canSubmitNow,
    hasResendPayload,
    actions,
    hasSaveEntryPoint,
    metHoldingSnapshot,
    retryResendDisabled,
  }
}

// ---------------------------------------------------------------------------
// The dialog's own state, as one value a reducer moves between.
// ---------------------------------------------------------------------------

/**
 * What the dialog is doing. Deliberately not what the report on screen says:
 * a same-task re-read can install a report of another tier under any phase,
 * so rendering reads the phase and `resolveDialogOutcome(view.report)` as two
 * separate things rather than one flattened list of combinations.
 *
 * `saving` covers the save POST (`post`) and the re-read a rejected save asks
 * for (`refresh`); `sending` is the resend a landed save promised. The two
 * phases carrying a failure name the send the "saved but not sent" panel
 * stands for; `retrying.failure` goes null when the request stops carrying
 * that snapshot mid-retry, and the phase stays busy until the retry settles.
 */
export type Phase =
  | { kind: "open" }
  | { kind: "saving"; step: "post" | "refresh" }
  | { kind: "sending" }
  | { kind: "send-failed"; failure: SendFailureState }
  | { kind: "retrying"; failure: SendFailureState | null }

export function assertNever(value: never): never {
  throw new Error(`unhandled value: ${JSON.stringify(value)}`)
}

/** Whether something the user started is in flight; submitting and closing wait on it. */
export function isBusy(phase: Phase): boolean {
  switch (phase.kind) {
    case "saving":
    case "sending":
    case "retrying":
      return true
    case "open":
    case "send-failed":
      return false
    default:
      return assertNever(phase)
  }
}

/** What is on screen. Exists only once the dialog has been shown. */
export interface View {
  report: ConnectorRuntimeReport
  // The `seq` of the request `report` was read for. Required rather than
  // nullable: a shown dialog always has a report, and a report is only ever
  // installed together with the request it answers. Kept apart from the read
  // cursor's `settledKey` for the reason deriveGates gives at `reportIsStale`.
  reportSeq: number
  drafts: Readonly<Record<string, string>>
  invalidDraftKeys: ReadonlyMap<string, InvalidObjectDraftReason>
  // Only the rejection itself, never where it attaches: the location is
  // re-derived on every render against whatever report is on screen then,
  // so a fresher report can never leave it pointing at a row that moved.
  fieldError: ConnectorRuntimeFailureDisposition | null
  // Which of the two save buttons was pressed last, for the retry button a
  // retryable rejection offers.
  lastAlsoResend: boolean
}

/**
 * Which read attempt the dialog is on for the current request (`nonce`,
 * bumped by "read again") and which attempt last settled (`settledKey`, in
 * the `${seq}:${nonce}` form deriveGates compares it against). A key rather
 * than a flag the read raises and lowers: a flag left raised on any of the
 * read's exits would disable saving for good.
 */
export interface ReadCursor {
  nonce: number
  settledKey: string | null
}

/**
 * What every failed resend so far has established about each snapshot's
 * message, keyed by its clientMessageId and merged with
 * mergeSendFailureDisposition. It outlives the panel on purpose: a retry the
 * re-read report turns down takes the panel away, and a superseded resend
 * never raises one, yet the next resend of that message must still not be
 * reported as "definitely not sent" after an earlier outcome was unknown.
 * It also decides whether a resend that never left the client may drop the
 * id it carried: once a message's verdict is "may have landed", that id is
 * kept so the next retry can still be coalesced with the earlier attempt.
 * Nothing renders from it.
 */
export type DeliveryVerdicts = ReadonlyMap<string, MessageDeliveryDisposition | null>

/**
 * The whole dialog. `hidden` until the first read shows it, and never back:
 * no event leads from `shown` to `hidden`, so a dialog the user has seen
 * cannot vanish from under a draft because something re-read the report.
 */
export type DialogState =
  | { stage: "hidden"; read: ReadCursor }
  | { stage: "shown"; read: ReadCursor; view: View; phase: Phase; verdicts: DeliveryVerdicts }

export const INITIAL_DIALOG_STATE: DialogState = {
  stage: "hidden",
  read: { nonce: 0, settledKey: null },
}

/** Events that end a flow, each in the phase it ends (see finishFrom). */
export type FinishingEvent =
  // The request moved on while the flow was out: never installs a report,
  // touches the field error or closes. `retryAttempt` is the merged verdict
  // of a panel retry that did not go out, for the panel it was pressed on.
  | { type: "superseded"; retryAttempt: { merged: MessageDeliveryDisposition | null } | null }
  | { type: "save-rejected"; disposition: ConnectorRuntimeFailureDisposition }
  | {
    type: "reject-refresh-settled"
    disposition: ConnectorRuntimeFailureDisposition
    // Null when the re-read failed and the report on screen stays.
    refreshed: { report: ConnectorRuntimeReport; seq: number } | null
  }
  | { type: "save-landed"; report: ConnectorRuntimeReport; seq: number }
  // Null when the message went out or there is no snapshot to name.
  | { type: "resend-settled"; failure: SendFailureState | null }
  | {
    type: "retry-settled"
    result: { kind: "sent" } | { kind: "failed"; merged: MessageDeliveryDisposition | null }
  }
  | { type: "retry-abandoned" }

/** Events in the middle of a flow: arriving in a busy phase, they leave it busy. */
export type MidEvent =
  | { type: "save-rejected-refreshing"; disposition: ConnectorRuntimeFailureDisposition }
  | { type: "save-landed-resending"; report: ConnectorRuntimeReport; seq: number }
  // Reads run under any phase, so this never changes the phase. `kept`
  // leaves whatever is on screen (the read failed, or had nothing to show).
  | {
    type: "read-settled"
    key: string
    result: { kind: "installed"; report: ConnectorRuntimeReport; seq: number } | { kind: "kept" }
  }
  // `verdict` is already merged with what earlier attempts established.
  | { type: "resend-failed"; snapshotId: string; verdict: MessageDeliveryDisposition | null }

/** Events the user or the render causes directly. */
export type LocalEvent =
  | { type: "read-again" }
  | { type: "draft-changed"; draftKey: string; value: string }
  | { type: "object-blurred"; draftKey: string; reason: InvalidObjectDraftReason | null }
  | { type: "save-started"; alsoResend: boolean }
  | { type: "retry-started" }
  // The request no longer carries the snapshot the held failure names.
  // Dispatched from render, so it must make needsSnapshotRecycle false in
  // one step for both phases that hold a failure.
  | { type: "snapshot-gone" }

export type DialogEvent = FinishingEvent | MidEvent | LocalEvent

/**
 * A field error after a report is installed under it, via
 * reconcileTypeMismatchDisposition. The two places a report is installed
 * reconcile against different things, and are kept that way: a read
 * (`"current"`) checks whatever rejection is on screen now; the re-read a
 * rejected save asked for (`{ rejection }`) checks that rejection, and when it
 * comes back unchanged leaves whatever is on screen as it is. An unchanged
 * hint comes back as the same object.
 */
export function reconcileFieldError(
  current: ConnectorRuntimeFailureDisposition | null,
  report: ConnectorRuntimeReport,
  basis: "current" | { rejection: ConnectorRuntimeFailureDisposition },
): ConnectorRuntimeFailureDisposition | null {
  if (basis === "current") return current && reconcileTypeMismatchDisposition(current, report)
  const reconciled = reconcileTypeMismatchDisposition(basis.rejection, report)
  return reconciled === basis.rejection ? current : reconciled
}

function installReport(
  view: View,
  report: ConnectorRuntimeReport,
  seq: number,
  basis: "current" | { rejection: ConnectorRuntimeFailureDisposition },
): View {
  return { ...view, report, reportSeq: seq, fieldError: reconcileFieldError(view.fieldError, report, basis) }
}

const OPEN: Phase = { kind: "open" }

/**
 * The idle phase a superseded flow leaves behind: a retry settles back onto
 * the panel it was pressed on while that panel still has a snapshot to stand
 * for, everything else onto the rows.
 */
function idle(phase: Phase): Phase {
  switch (phase.kind) {
    case "saving":
    case "sending":
      return OPEN
    case "retrying":
      return phase.failure ? { kind: "send-failed", failure: phase.failure } : OPEN
    case "open":
    case "send-failed":
      return phase
    default:
      return assertNever(phase)
  }
}

type Shown = Extract<DialogState, { stage: "shown" }>

/**
 * One finishing event, applied only in `expected`, the phase it ends, and
 * leaving the dialog idle there. Anywhere else it changes nothing: a flow's
 * own finishing event always arrives in that flow's own phase, so a busy
 * phase other than `expected` belongs to another flow still in flight, and
 * releasing it would reopen the save gate under that flow. That flow's own
 * finishing event releases it.
 */
function finishFrom(
  state: Shown,
  expected: Phase["kind"],
  apply: (state: Shown) => Shown,
): DialogState {
  return state.phase.kind === expected ? apply(state) : state
}

/**
 * The dialog's only state transition. Pure -- no requests, no toasts, no
 * translation -- because React may call it twice for one event. An event
 * that cannot arrive in the current phase returns the state unchanged.
 */
export function reduceDialog(state: DialogState, event: DialogEvent): DialogState {
  switch (event.type) {
    case "read-again":
      return { ...state, read: { ...state.read, nonce: state.read.nonce + 1 } }
    case "read-settled": {
      const read = { ...state.read, settledKey: event.key }
      if (event.result.kind === "kept") return { ...state, read }
      const { report, seq } = event.result
      if (state.stage === "hidden") {
        return {
          stage: "shown",
          read,
          view: {
            report,
            reportSeq: seq,
            drafts: {},
            invalidDraftKeys: new Map(),
            fieldError: null,
            lastAlsoResend: false,
          },
          phase: OPEN,
          verdicts: new Map(),
        }
      }
      return { ...state, read, view: installReport(state.view, report, seq, "current") }
    }
    default:
      break
  }
  // Every other event is about something on screen; a hidden dialog has none.
  if (state.stage === "hidden") return state
  const { phase, view } = state
  switch (event.type) {
    case "draft-changed":
      return { ...state, view: { ...view, drafts: { ...view.drafts, [event.draftKey]: event.value } } }
    case "object-blurred": {
      const invalidDraftKeys = new Map(view.invalidDraftKeys)
      if (event.reason) invalidDraftKeys.set(event.draftKey, event.reason)
      else invalidDraftKeys.delete(event.draftKey)
      return { ...state, view: { ...view, invalidDraftKeys } }
    }
    case "save-started":
      if (phase.kind !== "open") return state
      return { ...state, phase: { kind: "saving", step: "post" }, view: { ...view, lastAlsoResend: event.alsoResend } }
    case "retry-started":
      if (phase.kind !== "send-failed") return state
      return { ...state, phase: { kind: "retrying", failure: phase.failure } }
    case "snapshot-gone":
      if (phase.kind === "send-failed") return { ...state, phase: OPEN }
      if (phase.kind === "retrying" && phase.failure) return { ...state, phase: { kind: "retrying", failure: null } }
      return state
    case "save-rejected-refreshing":
      if (phase.kind !== "saving" || phase.step !== "post") return state
      return { ...state, phase: { kind: "saving", step: "refresh" }, view: { ...view, fieldError: event.disposition } }
    case "save-landed-resending":
      if (phase.kind !== "saving" || phase.step !== "post") return state
      return {
        ...state,
        phase: { kind: "sending" },
        view: { ...view, report: event.report, reportSeq: event.seq, fieldError: null },
      }
    case "resend-failed":
      return { ...state, verdicts: new Map(state.verdicts).set(event.snapshotId, event.verdict) }
    case "superseded": {
      const { retryAttempt } = event
      if (phase.kind === "retrying" && phase.failure && retryAttempt) {
        return { ...state, phase: { kind: "send-failed", failure: { ...phase.failure, disposition: retryAttempt.merged } } }
      }
      const next = idle(phase)
      return next === phase ? state : { ...state, phase: next }
    }
    case "save-rejected":
      return finishFrom(state, "saving", s => (
        s.phase.kind === "saving" && s.phase.step === "post"
          ? { ...s, phase: OPEN, view: { ...s.view, fieldError: event.disposition } }
          : { ...s, phase: OPEN }
      ))
    case "reject-refresh-settled":
      return finishFrom(state, "saving", (s) => {
        if (s.phase.kind !== "saving" || s.phase.step !== "refresh") return { ...s, phase: OPEN }
        if (!event.refreshed) return { ...s, phase: OPEN }
        const { report, seq } = event.refreshed
        return { ...s, phase: OPEN, view: installReport(s.view, report, seq, { rejection: event.disposition }) }
      })
    case "save-landed":
      return finishFrom(state, "saving", s => (
        s.phase.kind === "saving" && s.phase.step === "post"
          ? { ...s, phase: OPEN, view: { ...s.view, report: event.report, reportSeq: event.seq, fieldError: null } }
          : { ...s, phase: OPEN }
      ))
    case "resend-settled":
      return finishFrom(state, "sending", s => ({
        ...s,
        phase: event.failure ? { kind: "send-failed", failure: event.failure } : OPEN,
      }))
    case "retry-settled":
      return finishFrom(state, "retrying", (s) => {
        if (s.phase.kind !== "retrying" || !s.phase.failure || event.result.kind === "sent") return { ...s, phase: OPEN }
        return { ...s, phase: { kind: "send-failed", failure: { ...s.phase.failure, disposition: event.result.merged } } }
      })
    case "retry-abandoned":
      return finishFrom(state, "send-failed", s => ({ ...s, phase: OPEN }))
    default:
      return assertNever(event)
  }
}

// ---------------------------------------------------------------------------
// How a flow ends: what the dialog says, what it records, whether it closes.
// ---------------------------------------------------------------------------

/**
 * Why an ending says nothing. Every silent ending names one, so a silence
 * is a stated decision rather than a branch nobody wrote a toast for.
 */
export type SilenceReason =
  // Nothing was written and nothing was sent.
  | "nothing-irreversible"
  // The user pressed "save only": no message was promised.
  | "nothing-promised"
  // The message went out and appears in the conversation on its own.
  | "transcript-shows-it"
  // The send-failed panel stands in this same frame and says it.
  | "panel-carries-it"
  // The rows, or an error on them, say it in this same frame.
  | "rows-carry-it"
  // The user closed the dialog themselves.
  | "user-chose-it"
  // The dialog was never on screen.
  | "never-shown"
  // The user left the pages the dialog lives on.
  | "user-left"
  // A defensive exit a rendered frame has already ruled out.
  | "unreachable"

/**
 * What an ending tells the user. Only enum values and key names taken from
 * the server's report: no free text, so nothing the user typed and nothing
 * already translated can travel through one. Wording is the dialog's own
 * business, re-resolved when shown.
 */
export type Notice =
  | { kind: "saved-not-resent"; because: "unmounted" | "superseded" | "unavailable" | "incomplete" }
  | { kind: "saved-not-resent"; because: "unsupported"; keys: readonly string[] }
  | { kind: "only-unsupported-remaining"; keys: readonly string[] }
  | { kind: "save-rejected-elsewhere"; messageKey: ConnectorRuntimeErrorMessageKey }
  | { kind: "resend-not-sent"; disposition: MessageDeliveryDisposition | null }
  | { kind: "resend-already-sent" }

export type Tell = { notice: Notice } | { silent: SilenceReason }

/** How a flow ends while its request is still the current one. */
export interface Finish {
  tell: Tell
  event: FinishingEvent | null
  close?: ConnectorRuntimeDialogCloseOutcome
}

/**
 * The answers a flow gives at the moment an await returns, one per way it
 * can find the dialog. All three are required, so no await can answer only
 * for the case its author had in mind; the dialog, not the flow, decides
 * which one applies. `unmounted` can only say something. `superseded` can
 * also leave a panel retry's verdict on its panel, but has no event and no
 * close: closing would close the request the user is looking at now, which
 * this flow knows nothing about. `current` is how the flow ends, or
 * "continue" when it goes on to its next step.
 */
export interface Exit {
  unmounted: Tell
  superseded: { tell: Tell; retryAttempt?: { merged: MessageDeliveryDisposition | null } }
  current: Finish | "continue"
}

const NO_DRAFTS: Readonly<Record<string, string>> = Object.freeze({})
const NO_INVALID_DRAFT_KEYS: ReadonlyMap<string, InvalidObjectDraftReason> = new Map()

/**
 * The flat facts deriveGates reads, taken off the dialog's state. A hidden
 * dialog has no report, nothing in flight and no failure to stand for.
 */
export function gateFactsOf(state: DialogState, request: GateFacts["request"]): GateFacts {
  const read = { settledReadKey: state.read.settledKey, readNonce: state.read.nonce, request }
  if (state.stage === "hidden") {
    return {
      ...read,
      report: null,
      reportSeq: null,
      busy: false,
      heldFailure: null,
      retrying: false,
      drafts: NO_DRAFTS,
      invalidDraftKeys: NO_INVALID_DRAFT_KEYS,
    }
  }
  const { view, phase } = state
  return {
    ...read,
    report: view.report,
    reportSeq: view.reportSeq,
    // isBusy counts the retrying phase as busy, which keeps `retrying`
    // implying `busy` as GateFacts requires.
    busy: isBusy(phase),
    heldFailure: phase.kind === "send-failed" || phase.kind === "retrying" ? phase.failure : null,
    retrying: phase.kind === "retrying",
    drafts: view.drafts,
    invalidDraftKeys: view.invalidDraftKeys,
  }
}

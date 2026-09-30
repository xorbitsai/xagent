"use client"

import React, { useEffect, useReducer, useRef } from "react"
import { usePathname } from "next/navigation"

import {
  readRetryWithNewId,
  readSendDisposition,
  sendOutcomeMayHaveLanded,
} from "@/components/chat/clarification-delivery"
import {
  assertNever,
  canResendReport,
  deriveGates,
  gateFactsOf,
  hasLiveInvalidObjectMark,
  INITIAL_DIALOG_STATE,
  mergeSendFailureDisposition,
  opensOnFirstRead,
  reduceDialog,
  uniqueKeys,
  type Exit,
  type Finish,
  type InvalidObjectDraftReason,
  type Notice,
  type Tell,
} from "@/components/chat/connector-runtime-dialog-state"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { toast } from "@/components/ui/sonner"
import { Textarea } from "@/components/ui/textarea"
import { useApp } from "@/contexts/app-context-chat"
import {
  NOOP_ACTIONS,
  useConnectorRuntimeDialog,
  type ConnectorRuntimeDialogRequest,
} from "@/contexts/connector-runtime-dialog-context"
import { useI18n } from "@/contexts/i18n-context"
// Type-only, like clarification-delivery's own import of it: naming the
// disposition union here adds no runtime dependency on the websocket hook.
import type { MessageDeliveryDisposition } from "@/hooks/use-websocket"
import type { TranslationKey, TranslationVariables } from "@/i18n/translations"
import {
  buildSubmitItems,
  classifySubmitFailure,
  connectorRuntimeInputDraftKey,
  fetchTaskConnectorRuntimeRequirements,
  isAcceptedRuntimeKeyName,
  isDialogHostPathFor,
  isSubmittableObjectValue,
  resolveDialogOutcome,
  resolveOutcomeFor,
  submitTaskConnectorRuntimeValues,
  type ConnectorRuntimeConnector,
  type ConnectorRuntimeDialogTrigger,
  type ConnectorRuntimeErrorMessageKey,
  type ConnectorRuntimeFailureDisposition,
  type ConnectorRuntimeInput,
  type ConnectorRuntimeReport,
  type DialogOutcome,
} from "@/lib/connector-runtime-api"
import { isJsonRecord } from "@/lib/api-wrapper"
import { generateClientMessageId } from "@/lib/utils"

// An exhaustive lookup (not a template-literal key) so a source scan can
// verify every key this dialog renders exists in both locales without
// having to interpret string concatenation.
const ERROR_MESSAGE_TRANSLATION_KEYS: Record<ConnectorRuntimeErrorMessageKey, TranslationKey> = {
  network: "connectorRuntime.errors.network",
  busyRetry: "connectorRuntime.errors.busyRetry",
  contactAdmin: "connectorRuntime.errors.contactAdmin",
  conflict: "connectorRuntime.errors.conflict",
  typeString: "connectorRuntime.errors.typeString",
  typeObject: "connectorRuntime.errors.typeObject",
  typeUnknown: "connectorRuntime.errors.typeUnknown",
  emptyValue: "connectorRuntime.errors.emptyValue",
  keyNameRejected: "connectorRuntime.errors.keyNameRejected",
  configChanged: "connectorRuntime.errors.configChanged",
  notInSession: "connectorRuntime.errors.notInSession",
  connectorUnavailable: "connectorRuntime.errors.connectorUnavailable",
  tooLarge: "connectorRuntime.errors.tooLarge",
}

function translateFailure(
  t: (key: TranslationKey, vars?: TranslationVariables) => string,
  messageKey: ConnectorRuntimeErrorMessageKey,
  vars?: TranslationVariables,
): string {
  return t(ERROR_MESSAGE_TRANSLATION_KEYS[messageKey], vars)
}

/**
 * The same failure, worded for whole-dialog scope. That scope is where
 * locateFieldError falls back when the row a failure named is not one the
 * current report renders an editable control for -- the report dropped it,
 * or a refresh collapsed it into "already filled" -- so two kinds of text
 * stop being true there and get a variant instead of going through
 * translateFailure:
 *
 * - `conflict` is the only text carrying a `{key}` placeholder, and there
 *   is no key here to fill it with.
 * - `typeObject` and `typeString` each name the type one specific field
 *   needs. With no such field on screen that sentence points at nothing
 *   the user can act on, so the variant says only the part that stays
 *   true: the save was rejected over this value. The hint itself is not
 *   dropped -- a rejection the user saw must not vanish on a refresh they
 *   did not ask for -- only reworded, and a report that brings the row
 *   back brings the named text back with it.
 * - `emptyValue` says "this field", of a field that is not there. Its own
 *   disposition asks for no refresh, but the report under it still changes:
 *   a same-task re-read installs a fresher one, and the error's location is
 *   re-derived against whatever report is on screen at the time.
 *
 * `typeUnknown` needs no variant: it already names neither field nor type.
 * Every other messageKey is either about the connector or about the save as
 * a whole, and stays true with no field on screen.
 */
function translateDialogScopeFailure(
  t: (key: TranslationKey, vars?: TranslationVariables) => string,
  messageKey: ConnectorRuntimeErrorMessageKey,
): string {
  if (messageKey === "conflict") return t("connectorRuntime.errors.conflictNoKey")
  if (messageKey === "typeObject" || messageKey === "typeString") {
    return t("connectorRuntime.errors.typeNoField")
  }
  if (messageKey === "emptyValue") return t("connectorRuntime.errors.emptyValueNoField")
  return translateFailure(t, messageKey)
}

/**
 * The mount point for both halves of the connector-runtime dialog. Every
 * `AppProvider` in the tree renders this, including the two widget/share
 * ones -- there the dialog's own context read always returns the no-provider
 * default (no request, every action a no-op), so nothing below ever mounts.
 * Only two responsibilities live here: reading the current request, and the
 * two cleanup effects that narrow what the provider retains as the viewed
 * task changes or this tree unmounts. Everything about reading the report,
 * rendering rows, submitting and resending lives in the inner component
 * below, which only exists while there is a request to act on.
 */
export function ConnectorRuntimeDialog() {
  const { request, retainOnlyTask } = useConnectorRuntimeDialog()
  const { state } = useApp()
  const cleanupRef = useRef(retainOnlyTask)
  cleanupRef.current = retainOnlyTask

  // Widget and share pages mount this component with no
  // ConnectorRuntimeDialogProvider above it by design (see the docstring
  // above), so retainOnlyTask here is the shared no-op default -- reference-
  // equal to the module's own constant, since a real provider's action is a
  // distinct function from useMemo. Neither effect below calls it in that
  // case: the call would be harmless, but it would also trip the dev-only
  // "called outside provider" warning that exists to catch an actual wiring
  // mistake, not this expected shape.
  const hasProvider = retainOnlyTask !== NOOP_ACTIONS.retainOnlyTask
  const hasProviderRef = useRef(hasProvider)
  hasProviderRef.current = hasProvider

  // Task-switch cleanup: no cleanup function of its own. Combining this with
  // the unmount effect below into one effect would run the unmount cleanup
  // on every task change too, which would erase a same-render first-gate
  // request before anything could read it.
  useEffect(() => {
    if (!hasProvider) return
    retainOnlyTask(state.taskId)
  }, [state.taskId, retainOnlyTask, hasProvider])

  // Unmount cleanup: a separate effect with an empty dependency array, read
  // through a ref so it always calls the latest function without needing to
  // be in that array (matches the workforce pages' own unmount-cleanup shape).
  // hasProviderRef is read the same way for the same reason.
  useEffect(() => () => {
    if (hasProviderRef.current) cleanupRef.current(null)
  }, [])

  if (!request) return null
  // One body per task, and per held first message: a first gate is a message
  // of its own, so opening one mounts a fresh body even over a request for
  // the same task that is already on screen -- its first read then decides
  // whether it shows at all (opensOnFirstRead), and any flow the old body
  // had in flight ends as unmounted. A same-task retarget of the other two
  // triggers (gateId null) keeps the body, and the draft in it.
  return <ConnectorRuntimeDialogBody key={`${request.taskId}:${request.gateId ?? ""}`} request={request} />
}

function findConnector(
  report: ConnectorRuntimeReport,
  ref: { connector_type: string; connector_id: number } | undefined,
): ConnectorRuntimeConnector | null {
  if (!ref) return null
  return (
    report.connectors.find(
      c =>
        c.connector_ref.connector_type === ref.connector_type
        && c.connector_ref.connector_id === ref.connector_id,
    ) ?? null
  )
}

function connectorKeyOf(ref: { connector_type: string; connector_id: number }): string {
  return `${ref.connector_type}:${ref.connector_id}`
}

type FieldErrorLocation =
  | { scope: "dialog" }
  | { scope: "connector"; connectorKey: string }
  | { scope: "field"; connectorKey: string; draftKey: string }

/**
 * Where a failed save's error attaches, given the report the dialog is
 * currently showing. A location the current report can no longer find (the
 * connector's declaration changed, or a 409 refresh already collapsed the row
 * into "already filled") falls back to the whole dialog rather than being
 * silently dropped.
 *
 * Called fresh from render against whatever report the dialog currently
 * holds, never cached alongside the disposition that produced it. Every
 * disposition that asks for a refresh (`refresh: true`) is followed by
 * exactly one report install from one of three places -- the failed
 * save's own post-refresh install, the read effect's same-task re-request,
 * or that same effect's already-visible "met" branch -- and none of them
 * needs to also re-derive or clear a location: recomputing this on every
 * render against whatever report the dialog currently holds means all three
 * land on the right answer without any of them knowing this function
 * exists.
 */
function locateFieldError(
  report: ConnectorRuntimeReport,
  disposition: ConnectorRuntimeFailureDisposition,
): FieldErrorLocation {
  const { connectorRef, key } = disposition.locate
  if (!connectorRef) return { scope: "dialog" }
  const connector = findConnector(report, connectorRef)
  if (!connector) return { scope: "dialog" }
  const connectorKey = connectorKeyOf(connectorRef)
  if (key === undefined) return { scope: "connector", connectorKey }
  // Every key-bearing failure reason this locates is section-scoped to
  // "context" (type_mismatch.context., empty_value.context., conflict.context.
  // in connector-runtime-api.ts), and the draft key below is keyed by section.
  // Matching key alone would let a same-named row in another section (e.g.
  // "secrets") win the find, producing a draft key that no context row
  // holds — the error would then attach to nothing instead of falling back
  // to the whole-dialog scope this function otherwise guarantees.
  const input = connector.inputs.find(i => i.section === "context" && i.key === key)
  // A row the current report already reports satisfied renders the
  // "already filled" shortcut below instead of an editable control (or any
  // field-level error), most often because a refresh this same disposition
  // asked for just collapsed it: attaching here would pick a draft key
  // nothing on screen renders, so this falls back to the whole dialog
  // instead, same as a row it cannot find at all.
  if (!input || input.satisfied) return { scope: "dialog" }
  return {
    scope: "field",
    connectorKey,
    draftKey: connectorRuntimeInputDraftKey(connectorRef, input.section, key, input.type),
  }
}

/**
 * What this dialog says about a message it did not manage to send, given the
 * disposition the send path rejected with. Two texts, split by
 * sendOutcomeMayHaveLanded rather than by a second reading of the
 * disposition here, so this dialog and ClarificationForm cannot end up
 * describing the same failure differently.
 *
 * A rejection carrying no disposition at all takes the definite text. Every
 * rejection that leaves the wire carries one: use-websocket's sendChatMessage
 * wraps anything that is not already a MessageDeliveryError into a `not_sent`
 * one ("Pre-send failures never reached the server"), so the only
 * dispositionless rejections that reach here are AppContext.sendMessage's own
 * pre-flight refusals -- a closed session chat, files disabled for the
 * conversation, an attachment upload that failed -- all of which throw before
 * the send is attempted. doResend's id handling is deliberately more
 * conservative than this for the same case: reusing an id costs nothing if
 * the premise turns out to be too broad, while telling the user the turn did
 * not run costs a duplicate turn.
 */
function sendFailureTextKey(disposition: MessageDeliveryDisposition | null): TranslationKey {
  return sendOutcomeMayHaveLanded(disposition)
    ? "connectorRuntime.sendOutcomeUnknown"
    : "connectorRuntime.sendFailed"
}

// A settled resend attempt. The failed case carries the disposition the send
// path rejected with, already merged with what earlier failed attempts for
// the same snapshot established (doResend does the merge), because what this
// dialog then says about the message depends on it and every reader of this
// outcome -- the panel it raises and the three toasts that stand in for that
// panel where it cannot be rendered -- has to say the same thing. The
// nothing-to-send case carries that same verdict for the message the flow
// set out to resend: nothing was attempted, so it adds nothing, but a toast
// that words it as "not sent" would still take back an earlier unknown
// outcome.
type ResendOutcome =
  | { kind: "sent" }
  | { kind: "failed"; disposition: MessageDeliveryDisposition | null }
  | { kind: "nothing-to-send"; disposition: MessageDeliveryDisposition | null }

/**
 * The verdict a resend outcome leaves its message with: null once it went
 * out, otherwise the merged verdict either way of not sending carries. Every
 * reader of a resend outcome words itself off this one value, so no exit can
 * drop the carried verdict and take back an earlier unknown outcome, whether
 * or not that exit is reachable with a given kind today.
 */
function resendVerdict(outcome: ResendOutcome): MessageDeliveryDisposition | null {
  return outcome.kind === "sent" ? null : outcome.disposition
}

/**
 * Why the message was not resent, for a report canResendReport turns down.
 * Returns null for a met report, which has no such reason. Shared by the two
 * resend entry points so a user who reaches the same dead end from the
 * footer and from the retry button is told the same thing.
 */
function savedNotResentNotice(outcome: DialogOutcome): Notice | null {
  switch (outcome.kind) {
    case "unsupported_only":
      return { kind: "saved-not-resent", because: "unsupported", keys: uniqueKeys(outcome.blocking) }
    case "nothing_fillable":
      return { kind: "saved-not-resent", because: "unavailable" }
    case "fillable":
      return { kind: "saved-not-resent", because: "incomplete" }
    case "met":
      return null
    default:
      return assertNever(outcome)
  }
}

/** The only place a notice becomes text, resolved when it is shown. */
function noticeText(
  t: (key: TranslationKey, vars?: TranslationVariables) => string,
  notice: Notice,
): string {
  switch (notice.kind) {
    case "saved-not-resent":
      switch (notice.because) {
        case "unmounted": return t("connectorRuntime.savedNotResentUnmounted")
        case "superseded": return t("connectorRuntime.savedNotResentSuperseded")
        case "unsupported": return t("connectorRuntime.savedNotResentUnsupported", { keys: notice.keys.join(", ") })
        case "unavailable": return t("connectorRuntime.savedNotResentUnavailable")
        case "incomplete": return t("connectorRuntime.savedNotResentIncomplete")
        default:
          return assertNever(notice)
      }
    case "only-unsupported-remaining":
      return t("connectorRuntime.onlyUnsupportedRemaining", { keys: notice.keys.join(", ") })
    case "save-rejected-elsewhere":
      return translateDialogScopeFailure(t, notice.messageKey)
    case "resend-not-sent":
      return t(sendFailureTextKey(notice.disposition))
    case "resend-already-sent":
      return t("connectorRuntime.resendSupersededSent")
    case "saved-not-sent":
      return t("connectorRuntime.savedNotSent")
    default:
      return assertNever(notice)
  }
}

// The line under the title. Pointing the user back at the message box is
// only right while this dialog is holding a message it will not send and has
// nothing else in flight -- metHoldingSnapshot's own conditions. The neutral
// met line covers the rest: no snapshot at all, a send-failed panel that
// already carries the message and its own retry button, and a resend of this
// very message still on the wire. Short of met, the line says why the dialog
// is open; a session check asks the user to fill something in only while the
// report still has something this dialog can fill. A held first message was
// never sent, so its two lines say that whatever the report: fill in and send
// while something is fillable, otherwise send or leave it unsent.
// How a flow ends when its route gate finds the request off its pages: a
// dialog the user never saw closes as not-shown, which releases whatever it
// holds; one they saw was left by their own navigation, the same ending the
// leave effect gives it (left-host).
function offHostEnding(wasVisible: boolean): Finish {
  return wasVisible
    ? { tell: { silent: "user-left" }, event: null, close: "left-host" }
    : { tell: { silent: "never-shown" }, event: null, close: "not-shown" }
}

function descriptionKey(
  kind: DialogOutcome["kind"],
  trigger: ConnectorRuntimeDialogTrigger,
  metHoldingSnapshot: boolean,
): TranslationKey {
  if (trigger === "first_gate") {
    return kind === "fillable" ? "connectorRuntime.firstGateDescription" : "connectorRuntime.firstGateReadyDescription"
  }
  if (kind === "met") return metHoldingSnapshot ? "connectorRuntime.metNotResent" : "connectorRuntime.metNothingLeft"
  switch (trigger) {
    case "session_open":
      return kind === "fillable" ? "connectorRuntime.sessionOpenDescription" : "connectorRuntime.sessionOpenNotFillable"
    case "turn_failure": return "connectorRuntime.description"
    default: return assertNever(trigger)
  }
}

function ConnectorRuntimeDialogBody({ request }: { request: ConnectorRuntimeDialogRequest }) {
  const pathname = usePathname()
  const pathnameRef = useRef(pathname)
  pathnameRef.current = pathname

  const { close } = useConnectorRuntimeDialog()
  const { sendMessage } = useApp()
  const { t } = useI18n()

  const aliveRef = useRef(true)
  useEffect(() => {
    // The assignment is not redundant with useRef(true): React 18 StrictMode
    // double-invokes this effect in development (mount -> cleanup -> mount),
    // and the cleanup below runs in between. Without resetting here, every
    // guard that reads this ref would short-circuit for a component that is
    // genuinely still mounted, and the dialog would never become visible.
    aliveRef.current = true
    return () => { aliveRef.current = false }
  }, [])

  const requestRef = useRef(request)
  requestRef.current = request

  // Moved only by reduceDialog. Declared before anything reads it, because
  // the recycle check below dispatches during render.
  const [state, dispatch] = useReducer(reduceDialog, INITIAL_DIALOG_STATE)
  const visible = state.stage === "shown"
  const view = state.stage === "shown" ? state.view : null
  const report = view?.report ?? null
  // The client message id the most recent unresolved resend attempt used,
  // together with the clientMessageId of the snapshot it was sent for, so a
  // further retry can reuse the id only while it is still retrying that same
  // snapshot -- see doResend, which is the only reader and writer.
  const resendMessageIdRef = useRef<{ forSnapshotId: string, clientMessageId: string } | null>(null)

  const facts = gateFactsOf(state, { seq: request.seq, resendPayload: request.resendPayload }, request.trigger)
  // `canSubmitNow` already folds this in; what reads it here is dismissal,
  // which must keep the dialog open while any submission is in flight.
  //
  // That dismissal gate is not the whole picture today and this comment
  // must not pretend it is -- the saving and sending phases are part of
  // `busy`, so the save POST, the refresh GET a failed save runs, and both
  // sends do hold the dialog open while they are out. Each of those is now
  // bounded (the two connector-runtime calls time out after 20 seconds, and
  // a send settles or rejects), so none of them can hold it open
  // indefinitely any more, but taking those two phases out of `busy` would
  // change what closing does on four separate paths mid-write and is not
  // part of this change.
  const busy = facts.busy
  // Every value this dialog derives from its own state and the request it
  // is currently showing, gathered in one call placed after the reducer
  // above: the render-period recycle check right below needs
  // `needsSnapshotRecycle`, and the read effect further down needs
  // `readKey` -- both must see this render's state.
  // See connector-runtime-dialog-state.ts for what each of these means and
  // how it is computed.
  const {
    liveSendFailure,
    sendFailed,
    needsSnapshotRecycle,
    readKey,
    reading,
    reportIsStale,
    readFailed,
    outcome,
    canSubmitNow,
    actions,
    hasSaveEntryPoint,
    metHoldingSnapshot,
    retryResendDisabled,
  } = deriveGates(facts)
  // Dispatched during render, not from an effect. An effect noticing this
  // afterwards is the shape this dialog already replaced once: it lands a
  // frame late, and it never runs at all for a removal that does not move
  // `seq`. React re-runs this component with the reduced state before
  // committing anything, and snapshot-gone drops the held failure from
  // both phases that carry one, so the condition is false on the second
  // pass and no frame is painted from the state being dropped.
  if (needsSnapshotRecycle) {
    dispatch({ type: "snapshot-gone" })
  }

  // The only place this dialog raises a toast.
  const say = (tell: Tell): void => {
    if ("notice" in tell) toast(noticeText(t, tell.notice))
  }

  // How a flow ends while its request is current: say, record, close -- the
  // only place this dialog closes. Synchronous on purpose: an ending adds no
  // await of its own, so nothing can land between a flow's last await and
  // the state that ends it.
  const finish = (answer: Finish): void => {
    say(answer.tell)
    if (answer.event) dispatch(answer.event)
    if (answer.close) close(answer.close, request.taskId)
  }

  // Where every flow comes back after an await it started from `seqAtStart`,
  // and the one place that decides which of its three answers applies (see
  // Exit). Synchronous for the same reason as finish.
  const settle = (seqAtStart: number, exit: Exit): "continue" | "stopped" => {
    if (!aliveRef.current) {
      say(exit.unmounted)
      return "stopped"
    }
    if (requestRef.current.seq !== seqAtStart) {
      say(exit.superseded.tell)
      dispatch({ type: "superseded", retryAttempt: exit.superseded.retryAttempt ?? null })
      return "stopped"
    }
    if (exit.current === "continue") return "continue"
    finish(exit.current)
    return "stopped"
  }

  // Read on mount, on every subsequent request for this same task (the
  // dialog is already open and a new terminal frame retargeted it), and on
  // every "read again" press: the route gate runs before the request even
  // goes out, and again right before the dialog would become visible, since
  // the user is free to navigate away from a host page while this read is in
  // flight. Both gates check the pages the request's trigger may show on
  // (isDialogHostPathFor), and whether a first read shows at all depends on
  // the trigger too (opensOnFirstRead). Both gates end the flow through
  // offHostEnding: a first gate that leaves before it was ever visible closes
  // as not-shown, which releases its held message -- it goes out exactly as
  // it did before first gates existed -- and one that was visible closes as
  // left-host, which clears it, whether the leave effect below or one of
  // these gates sees the new path first.
  //
  // `visible` is read at the moment this effect starts, which is exactly
  // right here: nothing moves the dialog back from shown to hidden, so if
  // it was already showing something when this
  // request came in, it is still showing it by the time the fetch below
  // resolves (any path that would make it stop -- unmount, a task switch, a
  // host-page departure -- clears `request` and is caught by the seq/alive
  // checks first). A once-visible dialog must not vanish out from under a
  // user who is mid-draft. Neither kind of re-read is a decision about what
  // is on screen: a retarget is another terminal frame arriving, not
  // anything the user did, and a "read again" press asks for a fresher
  // report, not for the dialog to be emptied. The same reasoning covers a
  // still-live rejection message and a still-live "saved but not sent"
  // panel below: neither is cleared just because this re-read ran. A
  // rejection the server really made is not undone by reading the report
  // again, whoever asked for the read. Whether that panel still has a snapshot to be about is a
  // separate question this effect does not answer -- `sendFailed` above
  // derives it from the request on every render, including the retargets
  // that never reach this effect at all.
  useEffect(() => {
    if (!isDialogHostPathFor(requestRef.current.trigger, pathnameRef.current)) {
      finish(offHostEnding(visible))
      return
    }
    const wasVisible = visible
    const seqAtStart = request.seq
    const readKeyAtStart = readKey
    let cancelled = false
    fetchTaskConnectorRuntimeRequirements(request.taskId).then((result) => {
      if (cancelled || !aliveRef.current || requestRef.current.seq !== seqAtStart) return
      // This attempt has settled, whichever way the branches below go, and
      // each records it through one read-settled event. The three guards
      // above are exactly the cases where it must not be recorded (a newer
      // attempt owns the answer now).
      const kept = { type: "read-settled", key: readKeyAtStart, result: { kind: "kept" } } as const
      if (!result.ok) {
        console.warn(
          "[connector-runtime] requirements read failed",
          result.kind === "http" ? result.status : result.kind,
        )
        // Keep whatever the user is already looking at (report and draft)
        // rather than discarding it over a transient read failure. What the
        // user is looking at is then a report this request never produced,
        // which `reportIsStale` above keeps saving and resending closed over
        // until a later read installs the current one -- the dialog says so
        // and offers that read.
        dispatch(kept)
        finish(wasVisible
          ? { tell: { silent: "rows-carry-it" }, event: null }
          : { tell: { silent: "never-shown" }, event: null, close: "not-shown" })
        return
      }
      if (!isDialogHostPathFor(requestRef.current.trigger, pathnameRef.current)) {
        dispatch(kept)
        finish(offHostEnding(wasVisible))
        return
      }
      const outcome = resolveOutcomeFor(result.report, request.trigger)
      // A report the user has never seen and this request's trigger does
      // not open on (see opensOnFirstRead) is the one case where nothing is
      // installed at all: there is no dialog to keep open and nothing for it
      // to say. For a turn failure that is only a met report.
      if (!opensOnFirstRead(outcome.kind, request.trigger) && !wasVisible) {
        dispatch(kept)
        finish({ tell: { silent: "never-shown" }, event: null, close: "not-shown" })
        return
      }
      // This point is reached by the read that first shows the dialog, and
      // after that only because another terminal frame for the same task
      // retargeted it or the user asked to read again (see this effect's
      // opening comment) -- never because the user resolved anything -- so
      // a live rejection or send failure must survive it. A type-mismatch
      // hint is reconciled against this fresher report -- cleared when the
      // row's declared type changed under it, re-derived when the row
      // finally declares one at all, via reconcileFieldError (see there for
      // how handleSave's own post-failure refresh differs); a
      // 409 conflict hint has no such
      // report-derived staleness condition, so it is left alone here the
      // same way handleSave's refresh already leaves it alone. A met report
      // runs this too: "nothing is missing any more" says nothing about
      // whether a type hint still describes the row it names, and that row
      // can be declared with a different type in the very report that
      // reports the connector complete. The "saved but not sent" panel is
      // not about the report at all -- nothing a re-read can show would
      // make a send failure no longer have happened -- so nothing the read
      // returns clears it either. The only things that do are a resend that
      // actually completes (handleRetryResend), unmounting, and the
      // request no longer carrying the snapshot the panel is about, which
      // `sendFailed` derives during render rather than any effect here.
      //
      // An already-visible met report is installed the same way and stays
      // on screen with its footer collapsed to "Got it", rather than
      // silently discarding the user's in-progress draft.
      dispatch({
        type: "read-settled",
        key: readKeyAtStart,
        result: { kind: "installed", report: result.report, seq: seqAtStart },
      })
      finish({ tell: { silent: "rows-carry-it" }, event: null })
    })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [request.seq, state.read.nonce])

  // Leaving the host pages closes a dialog the user has already seen. A move
  // between two host pages is handled by the outer task-switch cleanup, not
  // by this effect noticing the path changed. A first gate's pages are wider
  // (isDialogHostPathFor), and leaving them after it was seen clears its
  // held message: the create path then says the message was not sent.
  useEffect(() => {
    if (visible && !isDialogHostPathFor(requestRef.current.trigger, pathname)) {
      finish({ tell: { silent: "user-left" }, event: null, close: "left-host" })
    }
    // finish is a fresh function every render. This call says nothing and
    // records nothing, so what it reads is `close`, which is listed, and the
    // trigger through requestRef: a first gate keeps its trigger for this
    // instance's life, and a session check upgraded to a turn failure keeps
    // the same host pages, so a stale trigger could not change the answer.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, pathname, close])

  // Re-derived every render against the report currently on screen, rather
  // than resolved once into the field error and cached there -- see
  // locateFieldError's own docstring for why a cached location goes stale
  // the moment any fresher report is installed.
  const activeFieldError = view?.fieldError
    ? { disposition: view.fieldError, location: locateFieldError(view.report, view.fieldError) }
    : null

  const handleDraftChange =(connector: ConnectorRuntimeConnector, input: ConnectorRuntimeInput, value: string) => {
    const draftKey = connectorRuntimeInputDraftKey(connector.connector_ref, input.section, input.key, input.type)
    dispatch({ type: "draft-changed", draftKey, value })
  }

  const handleObjectBlur = (connector: ConnectorRuntimeConnector, input: ConnectorRuntimeInput, value: string) => {
    const draftKey = connectorRuntimeInputDraftKey(connector.connector_ref, input.section, input.key, input.type)
    let reason: InvalidObjectDraftReason | null = null
    if (value.trim() !== "") {
      try {
        const parsed: unknown = JSON.parse(value)
        // Same predicate buildSubmitItems filters on, so a draft this marks
        // valid is never the one buildSubmitItems silently drops. A value
        // that fails it for being array/null/non-object is "invalid"; one
        // that is object-shaped but empty is "empty" -- the row's error
        // message tells the two apart. isSubmittableObjectValue is itself
        // built on this same isJsonRecord, so "object-shaped" cannot come to
        // mean one thing in the filter and another in this message.
        if (!isSubmittableObjectValue(parsed)) {
          reason = isJsonRecord(parsed) ? "empty" : "invalid"
        }
      } catch {
        reason = "invalid"
      }
    }
    dispatch({ type: "object-blurred", draftKey, reason })
  }

  // `resendFor` is the clientMessageId of the snapshot the caller set out to
  // resend. It is read only when the snapshot has gone by the time this runs;
  // otherwise the snapshot the request carries now is what goes out.
  const doResend = async (resendFor: string | null): Promise<ResendOutcome> => {
    const snapshot = requestRef.current.resendPayload
    if (!snapshot) {
      // Reachable, and covered by regression tests ("says the message did
      // not go out when the settlement lands mid save-and-resend", and the
      // footer-retry case in "words a settlement that takes the snapshot mid
      // save-and-resend off the message's verdict"). A settlement frame for
      // this task drops the snapshot in place (forgetDelivery), without
      // moving `seq`, so nothing re-runs the read effect and nothing else
      // notices. handleRetryResend cannot get here -- it compares the
      // panel's snapshot id against the one the request carries right now,
      // and awaits nothing between that comparison and the read above -- but
      // handleSave can, two ways: its save POST is awaited in between, and
      // the settlement can land during it; and its footer retry repeats an
      // earlier save-and-resend press, so it can start after the settlement
      // already took the snapshot. Kept distinct from "sent" and "failed" so
      // neither this case nor a future fourth caller is misreported as a
      // completed resend or as one the server refused.
      console.warn("[connector-runtime] resend attempted with no snapshot to send")
      // The snapshot that went is the one the caller named, so its verdict
      // is looked up under that id; the verdicts themselves are read off the
      // render this flow started from, for the reason given at the failed
      // branch below. This attempt itself is a definite not-sent, merged the
      // same way a refusal would be.
      const recorded = resendFor !== null && state.stage === "shown" ? state.verdicts.get(resendFor) : undefined
      return { kind: "nothing-to-send", disposition: mergeSendFailureDisposition(recorded ?? null, null) }
    }
    // Reuse the id the last unresolved attempt for this snapshot used,
    // unless that attempt's own outcome already proved it, or the snapshot
    // itself is not the one that id was minted for any more: the very first
    // attempt (ref starts null on a fresh dialog instance), any attempt an
    // earlier one proved not delivered, and any attempt whose carried id was
    // minted for a different snapshot, all mint fresh below. The comparison
    // is against the snapshot's own clientMessageId, not request.seq: a
    // same-task retarget that leaves this snapshot in place (openForTask's
    // "kept") bumps seq without invalidating this id, while one that swaps in
    // a different snapshot (a newer candidate staged while this dialog was
    // open, openForTask's "staged"/"stashed") must not let the old id carry
    // over onto different text. This is never the id of the original send
    // that opened this dialog -- the server has recorded that one as FAILED
    // and would bounce a same-id retry of it -- only ever an id this same
    // doResend minted.
    const carriedId = resendMessageIdRef.current
    const clientMessageId = carriedId && carriedId.forSnapshotId === snapshot.clientMessageId
      ? carriedId.clientMessageId
      : generateClientMessageId()
    try {
      // Matches every other programmatic resend call site in the app
      // (clarification-form.tsx, workforce-builder.tsx, agent-builder.tsx):
      // without force, a duplicate of this exact text still pending from an
      // earlier send on this same connection throws instead of sending. That
      // duplicate check only ever matches a *different* clientMessageId than
      // the one it is scanning for -- its own match condition excludes the
      // id under retry -- so a same-id retry never reaches it either way and
      // this flag changes nothing for it; force is what lets a fresh-id retry
      // (the original send still pending, unacknowledged) go out at all.
      // targetTaskId names the task this dialog is for rather than letting
      // sendMessage default to whichever task the page is currently
      // showing, the same way the app's other cross-task send sites
      // (workforce-builder.tsx, agent-builder.tsx) name theirs. The two
      // can differ: the effect that drops a request belonging to a task
      // the user has navigated away from is a plain useEffect, so it runs
      // after the browser has painted, and the frame where the new task is
      // already current while this dialog still holds the old one's
      // snapshot is a frame the user can click in. Naming the task turns
      // that click from a message delivered into the wrong conversation
      // into a send that fails and says so on the panel below.
      // request.taskId rather than snapshot.taskId: the two are always
      // equal (every candidate openForTask can hand over is filtered by
      // task id), and this body is mounted keyed on request.taskId, so it
      // cannot change for the life of this instance.
      await sendMessage(
        snapshot.text,
        { clientMessageId, force: true, targetTaskId: request.taskId },
        snapshot.files,
      )
      resendMessageIdRef.current = null
      return { kind: "sent" }
    } catch (error) {
      // Matches the read path's warn so a failing resend leaves the same
      // diagnostic signal. Carries the fixed prefix alone: unlike the read
      // path there is no closed-set status to report here, and the rejection
      // value is arbitrary, so logging it could carry message content.
      console.warn("[connector-runtime] resend failed")
      const disposition = readSendDisposition(error)
      // The verdicts are read off the render this attempt was started from:
      // both callers are held busy from their click until their own flow
      // ends, and every earlier attempt recorded its verdict before the flow
      // that ran it released that hold, so no verdict can be recorded
      // between that render and this read.
      const earlierVerdict = (state.stage === "shown" ? state.verdicts.get(snapshot.clientMessageId) : undefined) ?? null
      // The server refusing this id outright (`rejected`), or explicitly
      // demanding a new one, means this id is spent -- the next retry mints
      // fresh. An outcome_unknown one leaves open that the server already
      // durably accepted this attempt, so the next retry reuses this same id
      // rather than risking the same turn running twice under a second one.
      // A rejection carrying no disposition at all reuses it too, although
      // sendFailureTextKey establishes that such a rejection never left the
      // client: keeping an id is free, while minting one on a wrong guess is
      // not, so the cheap side is taken here and only the user-facing text
      // splits the two cases apart.
      //
      // `not_sent` is thrown by the websocket layer before anything reaches
      // the server, so it only speaks for this attempt: it says nothing about
      // an earlier attempt under the same id whose outcome was unknown. Once
      // this snapshot's verdict is already "may have landed", the id is kept
      // so the next retry can still be coalesced with that earlier attempt;
      // only a snapshot with no such attempt behind it mints fresh here.
      const mustMintNewId = (
        readRetryWithNewId(error)
        || disposition === "rejected"
        || (disposition === "not_sent" && !sendOutcomeMayHaveLanded(earlierVerdict))
      )
      resendMessageIdRef.current = mustMintNewId ? null : { forSnapshotId: snapshot.clientMessageId, clientMessageId }
      // What travels out is the message's verdict across every attempt so
      // far (see DeliveryVerdicts in connector-runtime-dialog-state.ts), not
      // this attempt's own disposition, so no caller can word a refusal of
      // this attempt as "not sent" after an earlier attempt's outcome was
      // unknown.
      const verdict = mergeSendFailureDisposition(earlierVerdict, disposition)
      dispatch({ type: "resend-failed", snapshotId: snapshot.clientMessageId, verdict })
      // The verdict travels out with the outcome rather than being turned
      // into text here: the caller decides whether this dialog is still on
      // screen to raise a panel or has to settle for a toast, and both have
      // to word the same failure the same way.
      return { kind: "failed", disposition: verdict }
    }
  }

  const handleSave = async (alsoResend: boolean) => {
    if (!view || !canSubmitNow) return
    const seqAtStart = request.seq
    const items = buildSubmitItems(view.report, view.drafts)
    // The message a save-and-resend sets out to resend: the snapshot the
    // request carries at this press. The footer retry repeats an earlier
    // save-and-resend, and a settlement frame (forgetDelivery) can have taken
    // that snapshot before the retry is pressed; the retry then stands for
    // the message the press it repeats named, so a resend that finds nothing
    // to send still words its toast off that message's verdict.
    const resendFor = alsoResend ? (request.resendPayload?.clientMessageId ?? view.lastResendFor) : null
    dispatch({ type: "save-started", resendFor })
    const result = await submitTaskConnectorRuntimeValues(request.taskId, items)

    if (!result.ok) {
      const disposition = classifySubmitFailure(result, view.report)
      const rejected = settle(seqAtStart, {
        unmounted: { silent: "nothing-irreversible" },
        // A silent end would show a save that simply stopped. A toast in the
        // whole-dialog wording rather than a field error: the report on
        // screen is the newer request's, and locating this rejection against
        // it would pin the server's reason to a row the rejected draft was
        // never built from.
        superseded: { tell: { notice: { kind: "save-rejected-elsewhere", messageKey: disposition.messageKey } } },
        current: disposition.refresh
          ? "continue"
          : { tell: { silent: "rows-carry-it" }, event: { type: "save-rejected", disposition } },
      })
      if (rejected === "stopped") return
      // The save buttons stay disabled across the refresh: re-enabling before
      // it settles lets a second submit go out built from the report this
      // refresh is about to replace, so the rejection is shown and the save
      // stays in flight until the refresh settles below.
      dispatch({ type: "save-rejected-refreshing", disposition })
      const refreshed = await fetchTaskConnectorRuntimeRequirements(request.taskId)
      settle(seqAtStart, {
        // The save failed, so nothing was written; a superseded refresh
        // leaves the rejection on screen.
        unmounted: { silent: "nothing-irreversible" },
        superseded: { tell: { silent: "nothing-irreversible" } },
        // Installed under the seq the save started with, which settle has
        // just proved current, and reconciled against this rejection (see
        // reconcileFieldError). A failed refresh keeps the report and the
        // rejection as they are.
        current: {
          tell: { silent: "rows-carry-it" },
          event: {
            type: "reject-refresh-settled",
            disposition,
            refreshed: refreshed.ok ? { report: refreshed.report, seq: seqAtStart } : null,
          },
        },
      })
      return
    }

    const newOutcome = resolveOutcomeFor(result.report, request.trigger)
    // Only a met report can carry the resend the primary button promised --
    // see canResendReport, which handleRetryResend asks too, so the two
    // entry points cannot disagree about the same snapshot. Because the
    // button promised a resend, a report that turns it down says so.
    const canResendNow = canResendReport(result.report)
    // The refreshed report's own reason for not resending, for a
    // save-and-resend that promised one. A "save only" press promised
    // nothing, so it stays quiet -- except for unsupported_only, which is
    // news either way: what this dialog cannot collect is not visible in
    // the rows it renders.
    const reason = (alsoResend ? savedNotResentNotice(newOutcome) : null)
      ?? (newOutcome.kind === "unsupported_only"
        ? { kind: "only-unsupported-remaining", keys: uniqueKeys(newOutcome.blocking) } as const
        : null)
    // A landed save installs its report and clears any rejection, even on
    // the met save-and-resend path that renders before it settles: a stale
    // rejection would land at whole-dialog scope, next to a send-failed
    // panel for a save that in fact succeeded.
    const landed = { report: result.report, seq: seqAtStart }
    if (request.trigger === "first_gate") {
      // The create path holding the message sends it once this lets it go:
      // stay while something is still fillable, release otherwise. Never a
      // resend, so none of the resend wording below applies.
      const notSent = { notice: { kind: "saved-not-sent" } } as const
      settle(seqAtStart, {
        unmounted: { silent: "sender-says-it" },
        // Unreachable: this body's request cannot move to a new `seq` while
        // it is a first gate. A failure frame or session check for its task
        // leaves the request as it is, and a second first gate for its task
        // has another gate id, so it mounts a new body (see the key on
        // ConnectorRuntimeDialogBody) and this save settles as unmounted.
        superseded: { tell: notSent },
        current: newOutcome.kind === "fillable"
          ? { tell: notSent, event: { type: "save-landed", ...landed } }
          : {
            tell: newOutcome.kind === "unsupported_only"
              ? { notice: { kind: "only-unsupported-remaining", keys: uniqueKeys(newOutcome.blocking) } }
              : { silent: "sender-says-it" },
            event: { type: "save-landed", ...landed },
            close: "resent",
          },
      })
      return
    }
    let current: Finish | "continue"
    switch (newOutcome.kind) {
      case "fillable":
      case "nothing_fillable":
        // Still blocked on something this dialog can collect (or, for
        // nothing_fillable, on nothing the user can act on beyond "Got it"):
        // stay open, re-render from the fresh report.
        current = { tell: reason ? { notice: reason } : { silent: "rows-carry-it" }, event: { type: "save-landed", ...landed } }
        break
      case "met":
      case "unsupported_only":
        current = (alsoResend && canResendNow)
          ? "continue"
          : {
            tell: reason ? { notice: reason } : { silent: "nothing-promised" },
            event: { type: "save-landed", ...landed },
            close: "dismissed",
          }
        break
      default:
        current = assertNever(newOutcome)
    }
    const saved = settle(seqAtStart, {
      // Nothing cancels the save when this tree goes, so a landed save
      // already wrote an immutable value and the promised resend will never
      // run; this toast is the only place left to say so.
      unmounted: alsoResend
        ? { notice: { kind: "saved-not-resent", because: "unmounted" } }
        : { silent: "nothing-promised" },
      superseded: {
        tell: alsoResend
          ? { notice: { kind: "saved-not-resent", because: "superseded" } }
          : { silent: "nothing-promised" },
      },
      current,
    })
    if (saved === "stopped") return

    dispatch({ type: "save-landed-resending", ...landed })
    const resendOutcome = await doResend(resendFor)
    // Both ways of not sending are worded off the verdict they carry -- the
    // toasts below and the panel alike -- so a snapshot taken mid-save cannot
    // un-say an earlier unknown outcome.
    const verdict = resendVerdict(resendOutcome)
    const notSent: Tell = { notice: { kind: "resend-not-sent", disposition: verdict } }
    // Read after the await: when settle finds the request still current,
    // the snapshot it carries is the one doResend read, which is what the
    // panel raised below is about and what its retry button would send.
    const failedSnapshotId = requestRef.current.resendPayload?.clientMessageId ?? null
    settle(seqAtStart, {
      // A failed resend would surface in the send-failed panel, which an
      // unmounted instance can never render, so say it once, globally.
      unmounted: resendOutcome.kind === "sent" ? { silent: "transcript-shows-it" } : notSent,
      // Nothing the fresher request renders says whether this message went
      // out, and its footer offers "Save and resend" again. A toast rather
      // than the panel: once the snapshot is replaced, the panel's retry
      // would go out under a new client message id rather than the one this
      // attempt used (xorbitsai/xagent#2502).
      superseded: {
        tell: resendOutcome.kind === "sent" ? { notice: { kind: "resend-already-sent" } } : notSent,
      },
      current: resendOutcome.kind === "sent"
        ? { tell: { silent: "transcript-shows-it" }, event: { type: "resend-settled", failure: null }, close: "resent" }
        : failedSnapshotId === null
          // A settlement frame for this task arrived while the save was in
          // flight and took the snapshot with it (forgetDelivery), without
          // reopening the dialog and so without moving `seq`. There is
          // nothing left to retry, and a panel here would draw a retry
          // button with nothing behind it -- so say the same thing the
          // panel says, once, and leave the dialog on its report.
          ? { tell: notSent, event: { type: "resend-settled", failure: null } }
          : {
            tell: { silent: "panel-carries-it" },
            event: {
              type: "resend-settled",
              failure: {
                snapshotId: failedSnapshotId,
                disposition: verdict,
              },
            },
          },
    })
  }

  const handleRetryResend = async () => {
    // A resend is one billed model call plus a possibly side-effecting tool
    // run; a double click here must not fire it twice.
    if (facts.retrying) return
    // The report this button's own gate reads has to be the current
    // request's. A same-task retarget starts a fresh read while the previous
    // report is still on screen, and a read that fails never replaces it at
    // all: across both windows canResendReport below would be answering
    // about a report this request did not produce, which is the same
    // question the save gate asks -- so both read the one predicate rather
    // than each deciding what "the latest report" means. The button is
    // disabled across both windows and the line above it says which one is
    // happening, so this is not a dead click; it re-checks rather than
    // trusting the render that drew it, the way handleSave re-checks
    // canSubmitNow.
    if (reportIsStale) return
    // The panel this button lives on is about one message, and doResend
    // below sends whichever snapshot the request carries when it runs: the
    // two must be the same message. Unreachable today -- `sendFailed`
    // derives the panel's visibility from exactly this comparison during
    // render, so a frame that draws this button has already proved them
    // equal -- but the handler re-checks rather than trusting the render
    // that drew it, the same way handleSave re-checks `canSubmitNow`. The
    // panel's own state is dropped here as well, for the same reason the
    // render above drops it: the send it was about can no longer be retried
    // from this dialog.
    const sendFailure = facts.heldFailure
    if (
      sendFailure === null
      || sendFailure.snapshotId !== requestRef.current.resendPayload?.clientMessageId
    ) {
      finish({ tell: { silent: "unreachable" }, event: { type: "retry-abandoned" } })
      return
    }
    // What the panel says right now, captured before the await below: a
    // superseded attempt whose panel is still up announces itself only when
    // it moves that wording, and by the time it settles the state behind the
    // wording may already have moved.
    const dispositionBefore = sendFailure.disposition
    // The report can change under a panel that stays up: this panel is about
    // a send that failed, not about the report, so a same-task re-read
    // leaves it alone while installing a report that no longer supports a
    // resend at all. handleSave asks canResendReport before its own resend
    // for exactly this reason -- the backend rejects the turn while it
    // builds the tool list, so sending anyway would put a second failure in
    // the conversation. This button asks the same question of the report on
    // screen, and answers a no the same way handleSave does: the panel goes
    // (there is nothing this dialog can still send) and the report's own
    // reason is said out loud, rather than leaving a disabled button with no
    // explanation next to it. A report that is not met always has such a
    // reason, and a shown dialog always has a report.
    if (!report || !canResendReport(report)) {
      const reason = report ? savedNotResentNotice(resolveDialogOutcome(report)) : null
      finish({ tell: reason ? { notice: reason } : { silent: "unreachable" }, event: { type: "retry-abandoned" } })
      return
    }
    const seqAtStart = request.seq
    dispatch({ type: "retry-started" })
    const resendOutcome = await doResend(sendFailure.snapshotId)
    // Every way this attempt can fail is worded off the verdict doResend
    // returns, which already folds in every earlier attempt for this message
    // (see DeliveryVerdicts in connector-runtime-dialog-state.ts), so a
    // refusal cannot un-say an earlier unknown outcome.
    const merged = resendVerdict(resendOutcome)
    const notSent: Tell = { notice: { kind: "resend-not-sent", disposition: merged } }
    settle(seqAtStart, {
      // A failed retry leaves no panel in an unmounted tree, so without this
      // nothing would tell the user the retry they pressed changed nothing.
      unmounted: resendOutcome.kind === "sent" ? { silent: "transcript-shows-it" } : notSent,
      // Sent: the panel may stay up with its button re-enabled, and without
      // a toast a second press would send this turn again. Not sent: said
      // according to whether the panel this retry was pressed on is still
      // up. The panel stands only while the request still carries the
      // snapshot it is about, so that is read live off the request here,
      // not off the render this click came from: a same-task retarget that
      // swapped in a different snapshot has taken the panel away (render
      // recycles it with snapshot-gone), and with the panel gone a toast is
      // the only report left, so every failed outcome gets one, as in the
      // unmounted case. A panel still up takes the merged wording, and a
      // toast fires only when that wording crosses from "definitely not
      // sent" to "may have landed", the one fact the panel does not already
      // say.
      superseded: resendOutcome.kind === "sent"
        ? { tell: { notice: { kind: "resend-already-sent" } } }
        : {
          tell: requestRef.current.resendPayload?.clientMessageId !== sendFailure.snapshotId
            || (sendOutcomeMayHaveLanded(merged) && !sendOutcomeMayHaveLanded(dispositionBefore))
            ? notSent
            : { silent: sendOutcomeMayHaveLanded(dispositionBefore) ? "panel-carries-it" : "nothing-irreversible" },
          // Only a failure the retry phase still holds takes this; once
          // snapshot-gone has dropped it the reducer leaves the phase to go
          // back to open, so the verdict can never land on a panel for a
          // different snapshot.
          retryAttempt: { merged },
        },
      // Failed again: where the panel's wording does not move, only the
      // toast tells "failed again" from "nothing happened"; both read `merged`.
      current: resendOutcome.kind === "sent"
        ? { tell: { silent: "transcript-shows-it" }, event: { type: "retry-settled", result: { kind: "sent" } }, close: "resent" }
        : { tell: notSent, event: { type: "retry-settled", result: { kind: "failed", merged } } },
    })
  }

  // A resend in flight holds the dialog open for the same reason a save does:
  // dismissing mid-send drops the request (and with it the snapshot the retry
  // button reads), leaving nothing to retry from if that send fails.
  const handleDismiss = () => {
    if (busy) return
    finish({ tell: { silent: "user-chose-it" }, event: null, close: "dismissed" })
  }

  const handleOpenChange = (open: boolean) => {
    if (open || busy) return
    handleDismiss()
  }

  // Lets a held first message go; its create path sends it. Not gated on the
  // report: nothing here is written from it, and the turn still meets the
  // per-turn check, whose own dialog offers a resend if it fails.
  const handleSendHeld = () => {
    if (busy) return
    finish({ tell: { silent: "sender-says-it" }, event: null, close: "resent" })
  }

  const dialogFieldError = activeFieldError?.location.scope === "dialog" ? activeFieldError.disposition : null

  if (state.stage !== "shown" || !outcome) return null
  const { drafts, invalidDraftKeys, lastResendFor } = state.view
  const { connectors } = state.view.report

  return (
    <Dialog open={visible} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          {/* The title follows the report and nothing else: a met report
              means nothing is missing, whatever else is going on, and the
              dialog must not go on claiming input is missing over a report
              that says it is not -- which it did whenever a met dialog
              stopped holding a snapshot, most directly when a settlement
              frame took one away in place. Only the line under it depends on
              the situation, and only to choose between two true things. */}
          <DialogTitle>
            {t(outcome.kind === "met" ? "connectorRuntime.metTitle" : "connectorRuntime.title")}
          </DialogTitle>
          <DialogDescription>
            {t(descriptionKey(outcome.kind, request.trigger, metHoldingSnapshot))}
          </DialogDescription>
        </DialogHeader>

        {outcome.kind === "unsupported_only" && (
          <p className="text-sm text-muted-foreground">{t("connectorRuntime.onlyUnsupportedNotice")}</p>
        )}

        {dialogFieldError && (
          <p className="text-sm text-destructive" role="alert">
            {/* This scope has no row identity in hand -- it is where
                locateFieldError falls back when the row a failure named is
                gone or unrecognized, most often a refresh that just
                dropped that row or collapsed it into "already filled".
                Which reasons that makes untrue, and what they say instead,
                is translateDialogScopeFailure's own business. */}
            {translateDialogScopeFailure(t, dialogFieldError.messageKey)}
          </p>
        )}

        {/* Both of these sit above the panel/rows split below, because both
            shapes need them: the save buttons and the panel's resend button
            are held closed by the same `reportIsStale`, and a disabled
            button with nothing next to it explaining why is the shape this
            dialog has been told twice not to leave on screen. */}
        {reading && (
          <p className="text-sm text-muted-foreground">{t("connectorRuntime.refreshing")}</p>
        )}

        {readFailed && (
          <div className="space-y-2">
            <p className="text-sm text-destructive" role="alert">{t("connectorRuntime.readFailed")}</p>
            {/* The only way back to a current report short of closing the
                dialog, which would drop the stashed message with it (a
                close counts as the user giving up) and take the one-click
                resend away for good. */}
            <Button variant="outline" onClick={() => dispatch({ type: "read-again" })}>
              {t("connectorRuntime.actions.readAgain")}
            </Button>
          </div>
        )}

        {liveSendFailure ? (
          <div className="space-y-3">
            {/* Two texts, one per outcome: a send the server definitely
                refused is reported as not sent, while one whose
                acknowledgement was lost may only warn. Re-resolved on every
                render rather than translated once at failure time, so a
                locale switch while this panel is up is not stuck in
                whatever language was active when the send failed. */}
            <p className="text-sm text-destructive">
              {t(sendFailureTextKey(liveSendFailure.disposition))}
            </p>
            <Button disabled={retryResendDisabled} onClick={handleRetryResend}>
              {t("connectorRuntime.actions.resend")}
            </Button>
          </div>
        ) : (
          <div className="space-y-4">
            {connectors.map((connector) => {
              const connectorKey = connectorKeyOf(connector.connector_ref)
              const connectorError =
                activeFieldError
                && activeFieldError.location.scope === "connector"
                && activeFieldError.location.connectorKey === connectorKey
                  ? activeFieldError.disposition
                  : null
              const stillMissing =
                outcome.kind === "fillable"
                  ? uniqueKeys(outcome.blocking.filter(b => connectorKeyOf(b.connectorRef) === connectorKey))
                  : []
              return (
                <div key={connectorKey} className="space-y-2 rounded-md border p-3">
                  <div className="text-sm font-medium">{connector.name}</div>
                  {connectorError && (
                    <p className="text-sm text-destructive" role="alert">
                      {translateFailure(t, connectorError.messageKey)}
                    </p>
                  )}
                  {connector.inputs.map((input) => {
                    // Doubles as this row's React key: it carries the
                    // input's full identity (connector, section, key name,
                    // declared type), so two rows that legitimately share a
                    // key name across sections never collide, and a row
                    // whose declared type changes across a refresh is
                    // treated as a new row rather than reusing the old
                    // one's draft and error state under a new meaning.
                    const draftKey = connectorRuntimeInputDraftKey(connector.connector_ref, input.section, input.key, input.type)
                    const acceptedKeyName = input.section !== "context" || isAcceptedRuntimeKeyName(input.key)
                    const markLive = hasLiveInvalidObjectMark(connector, input, invalidDraftKeys)
                    const fieldLevelError =
                      activeFieldError
                      && activeFieldError.location.scope === "field"
                      && activeFieldError.location.draftKey === draftKey
                        ? activeFieldError.disposition
                        : null

                    if (input.section === "context" && input.satisfied) {
                      return (
                        <div key={draftKey} className="text-sm">
                          <span className="font-medium">{input.key}</span>{" "}
                          <span className="text-muted-foreground">{t("connectorRuntime.filled")}</span>
                        </div>
                      )
                    }

                    if (input.section !== "context") {
                      return (
                        <div key={draftKey} className="text-sm">
                          <div className="font-medium">{input.key}</div>
                          <p className="text-muted-foreground">{t("connectorRuntime.unsupportedNote")}</p>
                        </div>
                      )
                    }

                    // Unfilled context row while only "Got it" is offered:
                    // no input control and no "saved, cannot be changed"
                    // hint, since there is no save entry point in this shape.
                    if (!hasSaveEntryPoint) {
                      return (
                        <div key={draftKey} className="text-sm">
                          <span className="font-medium">{input.key}</span>
                          {!acceptedKeyName && (
                            <p className="text-destructive">{t("connectorRuntime.keyNameWarning")}</p>
                          )}
                        </div>
                      )
                    }

                    return (
                      <div key={draftKey} className="space-y-1">
                        <Label htmlFor={`connector-runtime-${draftKey}`}>{input.key}</Label>
                        {input.type === "object" ? (
                          <Textarea
                            id={`connector-runtime-${draftKey}`}
                            value={drafts[draftKey] ?? ""}
                            aria-invalid={markLive}
                            onChange={e => handleDraftChange(connector, input, e.target.value)}
                            onBlur={e => handleObjectBlur(connector, input, e.target.value)}
                          />
                        ) : (
                          <Input
                            id={`connector-runtime-${draftKey}`}
                            type="text"
                            value={drafts[draftKey] ?? ""}
                            onChange={e => handleDraftChange(connector, input, e.target.value)}
                          />
                        )}
                        {markLive && (
                          <p className="text-sm text-destructive">
                            {t(
                              invalidDraftKeys.get(draftKey) === "empty"
                                ? "connectorRuntime.objectEmpty"
                                : "connectorRuntime.objectInvalid",
                            )}
                          </p>
                        )}
                        <p className="text-sm text-muted-foreground">{t("connectorRuntime.contextNote")}</p>
                        {!acceptedKeyName && (
                          <p className="text-sm text-destructive">{t("connectorRuntime.keyNameWarning")}</p>
                        )}
                        {fieldLevelError && (
                          <p className="text-sm text-destructive" role="alert">
                            {translateFailure(t, fieldLevelError.messageKey, { key: input.key })}
                          </p>
                        )}
                      </div>
                    )
                  })}
                  {stillMissing.length > 0 && (
                    <p className="text-sm text-muted-foreground">
                      {t("connectorRuntime.stillMissingAfterSave", { keys: stillMissing.join(", ") })}
                    </p>
                  )}
                </div>
              )
            })}
          </div>
        )}

        {!sendFailed && (
          <DialogFooter>
            {actions.includes("acknowledge") && (
              // handleDismiss already refuses while a submission is in
              // flight; without this the button still looks pressable and
              // does nothing when pressed. A met report renders this as the
              // only button, and a save-and-resend whose save came back met
              // is still busy for as long as its resend runs, so this
              // is a state the user can reach. The save buttons reach the
              // same guard through canSubmitNow, which folds busy in.
              <Button variant="outline" disabled={busy} onClick={handleDismiss}>
                {t("connectorRuntime.actions.acknowledge")}
              </Button>
            )}
            {actions.includes("saveOnly") && (
              <Button
                variant={actions.includes("saveAndResend") ? "outline" : "default"}
                disabled={!canSubmitNow}
                onClick={() => handleSave(false)}
              >
                {t("connectorRuntime.actions.saveOnly")}
              </Button>
            )}
            {actions.includes("saveAndResend") && (
              <Button disabled={!canSubmitNow} onClick={() => handleSave(true)}>
                {t("connectorRuntime.actions.saveAndResend")}
              </Button>
            )}
            {actions.includes("saveAndSend") && (
              <Button disabled={!canSubmitNow} onClick={() => handleSave(false)}>
                {t("connectorRuntime.actions.saveAndSend")}
              </Button>
            )}
            {actions.includes("sendHeld") && (
              <Button disabled={busy} onClick={handleSendHeld}>
                {t("connectorRuntime.actions.sendHeld")}
              </Button>
            )}
            {/* Both conditions, not just the retryable failure. A hint can
                outlive the report it was raised against: a transport failure
                raises a whole-dialog, retryable one that no refresh clears,
                and a same-task retarget can then install a report offering no
                way to save at all -- nothing left but what this dialog cannot
                collect, or nothing but "Got it". The rows render read-only in
                those shapes for that reason, so a retry button beside them
                was the one control still able to submit into a shape whose
                whole point is that submitting is not on offer. */}
            {hasSaveEntryPoint && dialogFieldError?.retry && (
              <Button variant="outline" disabled={!canSubmitNow} onClick={() => handleSave(lastResendFor !== null)}>
                {t("connectorRuntime.actions.retry")}
              </Button>
            )}
          </DialogFooter>
        )}
      </DialogContent>
    </Dialog>
  )
}

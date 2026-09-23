"use client"

import React, { useEffect, useRef, useState } from "react"
import { usePathname } from "next/navigation"

import {
  readRetryWithNewId,
  readSendDisposition,
  sendOutcomeMayHaveLanded,
} from "@/components/chat/clarification-delivery"
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
  isConnectorRuntimeDialogHostPath,
  isSubmitEnabled,
  isSubmittableObjectValue,
  reconcileTypeMismatchDisposition,
  resolveDialogActions,
  resolveDialogOutcome,
  submitTaskConnectorRuntimeValues,
  type ConnectorRuntimeConnector,
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
  // snapshot before anything could read it.
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
  return <ConnectorRuntimeDialogBody key={request.taskId} request={request} />
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

// Why a draft failed the object-field blur check: "invalid" for anything
// that is not JSON-object-shaped, "empty" for `{}`, which parses fine but
// isSubmittableObjectValue (connector-runtime-api.ts) rejects because the
// server treats it as a blank context value. The row's error message reads
// this to show the reason-specific hint instead of a generic one.
type InvalidObjectDraftReason = "invalid" | "empty"

/**
 * Whether an invalid-object mark for this input is still live. Only a
 * `context` row the current report leaves unsatisfied and still declares
 * `object`-typed renders the textarea whose blur handler can clear such a
 * mark; against any other row the mark is unreachable. Both the submit gate
 * and the row's own error message read this one predicate, so the button can
 * never be disabled by an error the row does not show, and the row can never
 * show an error that leaves the button enabled.
 */
function hasLiveInvalidObjectMark(
  connector: ConnectorRuntimeConnector,
  input: ConnectorRuntimeInput,
  invalidDraftKeys: Map<string, InvalidObjectDraftReason>,
): boolean {
  return (
    input.section === "context"
    && !input.satisfied
    && input.type === "object"
    && invalidDraftKeys.has(connectorRuntimeInputDraftKey(connector.connector_ref, input.section, input.key, input.type))
  )
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
 * exactly one `setReport` call from one of three places -- the failed
 * save's own post-refresh install, the read effect's same-task re-request,
 * or that same effect's already-visible "met" branch -- and none of them
 * needs to also re-derive or clear a location: recomputing this on every
 * render against whatever `report` state currently holds means all three
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

function uniqueKeys(locations: Array<{ key: string }>): string[] {
  return Array.from(new Set(locations.map(l => l.key)))
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
// path rejected with, because what this dialog then says about the message
// depends on it and every reader of this outcome -- the panel it raises and
// the three toasts that stand in for that panel where it cannot be rendered
// -- has to say the same thing.
type ResendOutcome =
  | { kind: "sent" }
  | { kind: "failed"; disposition: MessageDeliveryDisposition | null }
  | { kind: "nothing-to-send" }

/**
 * The same text, for a caller holding a settled attempt rather than a bare
 * disposition. "nothing-to-send" takes the definite text: nothing was handed
 * to the send path at all, so there is no uncertainty to preserve.
 */
function resendFailureTextKey(outcome: ResendOutcome): TranslationKey {
  return sendFailureTextKey(outcome.kind === "failed" ? outcome.disposition : null)
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
function mergeSendFailureDisposition(
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
function canResendReport(report: ConnectorRuntimeReport): boolean {
  return resolveDialogOutcome(report).kind === "met"
}

/**
 * Why the message was not resent, for a report canResendReport turns down.
 * Returns null for a met report, which has no such reason. Shared by the two
 * resend entry points so a user who reaches the same dead end from the
 * footer and from the retry button is told the same thing.
 */
function savedNotResentText(
  t: (key: TranslationKey, vars?: TranslationVariables) => string,
  outcome: DialogOutcome,
): string | null {
  if (outcome.kind === "unsupported_only") {
    return t("connectorRuntime.savedNotResentUnsupported", { keys: uniqueKeys(outcome.blocking).join(", ") })
  }
  if (outcome.kind === "nothing_fillable") return t("connectorRuntime.savedNotResentUnavailable")
  if (outcome.kind === "fillable") return t("connectorRuntime.savedNotResentIncomplete")
  return null
}

interface FieldErrorState {
  disposition: ConnectorRuntimeFailureDisposition
}

/**
 * The failed send the "saved but not sent" panel is about: the
 * clientMessageId of the snapshot it names, and the disposition that failure
 * carried, held together so the panel can never word one send's outcome with
 * another's. See `sendFailed` in the body below for how the id half is read.
 */
interface SendFailureState {
  snapshotId: string
  disposition: MessageDeliveryDisposition | null
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

  const [report, setReport] = useState<ConnectorRuntimeReport | null>(null)
  // Which read attempt has settled, or null before any has. Compared against
  // the attempt the dialog is currently on during render (see `readKey` and
  // `reading` below) rather than being a boolean the read effect raises and
  // lowers: that effect has six exits, and a flag left raised on any one of
  // them would disable saving for good, while a key that never catches up
  // cannot outlive the attempt it names.
  const [settledReadKey, setSettledReadKey] = useState<string | null>(null)
  // The `seq` of the request the report currently on screen was read for, or
  // null before any report is installed. Deliberately a second fact rather
  // than being folded into `settledReadKey`: a read settles whichever way it
  // goes, but only a read that succeeded installs a report, and a failed one
  // leaves the previous request's report on screen. Telling the two apart is
  // what lets saving stay closed over a report the current request never
  // produced, while dismissal and the re-read below stay available.
  const [reportSeq, setReportSeq] = useState<number | null>(null)
  // Bumped by the "read again" button a failed read offers. The read effect
  // depends on it, so a bump re-runs the read for the same request without
  // needing a new request to arrive.
  const [readNonce, setReadNonce] = useState(0)
  const [visible, setVisible] = useState(false)
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [invalidDraftKeys, setInvalidDraftKeys] = useState<Map<string, InvalidObjectDraftReason>>(new Map())
  const [submitting, setSubmitting] = useState(false)
  const [fieldError, setFieldError] = useState<FieldErrorState | null>(null)
  const [lastAlsoResend, setLastAlsoResend] = useState(false)
  // The send the "saved but not sent" panel is about, or null when no send
  // has failed. It carries the failed snapshot's clientMessageId rather than
  // being a bare flag because the panel names one message while its retry
  // button sends whichever snapshot the request currently holds, and the
  // request can stop carrying that snapshot underneath it: a same-task
  // retarget swaps in a newer candidate (openForTask), and a settlement
  // frame for this task takes it away without moving `seq` at all
  // (forgetDelivery). It carries that send's disposition alongside, because
  // the panel's text depends on it -- see sendFailureTextKey -- and the two
  // must never be able to come from different attempts.
  const [sendFailure, setSendFailure] = useState<SendFailureState | null>(null)
  // Derived every render against the snapshot the request currently
  // carries, the same way `activeFieldError` below is re-derived rather
  // than cached: the panel and its retry button must be about the same
  // message in every painted frame, including the first frame after a
  // retarget commits. An effect that noticed the two had come apart and
  // reset the panel afterwards left that first frame actionable, and never
  // ran at all for a removal that does not move `seq`.
  const liveSendFailure = sendFailure !== null
    && sendFailure.snapshotId === request.resendPayload?.clientMessageId
    ? sendFailure
    : null
  const sendFailed = liveSendFailure !== null
  // Deriving the panel from the request each render is what keeps every
  // painted frame honest, but it leaves the value itself behind once the
  // snapshot it names is gone: the three places that clear it all require
  // the panel to be rendering, and the two ways a snapshot disappears --
  // forgetDelivery dropping it in place, a retarget swapping in a newer
  // candidate -- are exactly the ways it stops rendering. Today nothing
  // brings a clientMessageId back once it has gone (doResend mints a fresh
  // id per attempt, and a snapshot's own id is written once at send time),
  // so that leftover is unreachable rather than wrong; it is dropped here so
  // it cannot become wrong if an id ever does come back.
  //
  // Set during render, not from an effect. An effect noticing this
  // afterwards is the shape this dialog already replaced once: it lands a
  // frame late, and it never runs at all for a removal that does not move
  // `seq`. React re-runs this component with the reset value before
  // committing anything, so the condition is false on the second pass and
  // no frame is painted from the state being dropped.
  if (sendFailure !== null && liveSendFailure === null) {
    setSendFailure(null)
  }
  const [resending, setResending] = useState(false)
  // The client message id the most recent unresolved resend attempt used,
  // together with the clientMessageId of the snapshot it was sent for, so a
  // further retry can reuse the id only while it is still retrying that same
  // snapshot -- see doResend, which is the only reader and writer.
  const resendMessageIdRef = useRef<{ forSnapshotId: string, clientMessageId: string } | null>(null)

  // The read attempt this dialog is currently on: the request it is for, and
  // which try for that request it is. Both halves are needed. `seq` alone
  // cannot tell a fresh attempt for the same request from the one that just
  // failed, so a "read again" press would leave `reading` false and the
  // screen would say nothing while the retry was out.
  const readKey = `${request.seq}:${readNonce}`

  // Read on mount, on every subsequent request for this same task (the
  // dialog is already open and a new terminal frame retargeted it), and on
  // every "read again" press: the route gate runs before the request even
  // goes out, and again right before the dialog would become visible, since
  // the user is free to navigate away from a host page while this read is in
  // flight.
  //
  // `visible` is read at the moment this effect starts, which is exactly
  // right here: setVisible is only ever called with `true` in this
  // component, so if the dialog was already showing something when this
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
    if (!isConnectorRuntimeDialogHostPath(pathnameRef.current)) {
      close("not-shown")
      return
    }
    const wasVisible = visible
    const seqAtStart = request.seq
    const readKeyAtStart = readKey
    let cancelled = false
    fetchTaskConnectorRuntimeRequirements(request.taskId).then((result) => {
      if (cancelled || !aliveRef.current || requestRef.current.seq !== seqAtStart) return
      // This attempt has settled, whichever way the branches below go.
      // Recorded once, here, rather than at each of those branches: the three
      // guards above are exactly the cases where it must not be recorded (a
      // newer attempt owns the answer now), and every branch past this point
      // either installs a report or deliberately keeps the one already on
      // screen. This is only half the story -- it says a read finished, not
      // that the report on screen is the current request's -- which is why
      // `reportSeq` is written separately, next to the one place a report is
      // actually installed.
      setSettledReadKey(readKeyAtStart)
      if (!result.ok) {
        console.warn(
          "[connector-runtime] requirements read failed",
          result.kind === "http" ? result.status : result.kind,
        )
        // Keep whatever the user is already looking at (report and draft)
        // rather than discarding it over a transient read failure. What the
        // user is looking at is then a report this request never produced,
        // which `reportIsStale` below keeps saving and resending closed over
        // until a later read installs the current one -- the dialog says so
        // and offers that read.
        if (!wasVisible) close("not-shown")
        return
      }
      if (!isConnectorRuntimeDialogHostPath(pathnameRef.current)) {
        close("not-shown")
        return
      }
      const outcome = resolveDialogOutcome(result.report)
      // A met report the user has never seen is the one case where nothing
      // is installed at all: there is no dialog to keep open and nothing
      // for it to say.
      if (outcome.kind === "met" && !wasVisible) {
        close("not-shown")
        return
      }
      setReport(result.report)
      setReportSeq(seqAtStart)
      // This point is only reached because another terminal frame for the
      // same task retargeted an already-open dialog (see this effect's
      // opening comment) -- never because the user resolved anything -- so
      // a live rejection or send failure must survive it. A type-mismatch
      // hint is reconciled against this fresher report -- cleared when the
      // row's declared type changed under it, re-derived when the row
      // finally declares one at all, via reconcileTypeMismatchDisposition,
      // the same call handleSave's own post-failure refresh makes below; a
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
      setFieldError((prev) => {
        if (!prev) return prev
        const reconciled = reconcileTypeMismatchDisposition(prev.disposition, result.report)
        // The same disposition back means this report changed nothing about
        // the hint, and returning `prev` keeps the field error referentially
        // stable rather than re-rendering over an equal value.
        if (reconciled === prev.disposition) return prev
        return reconciled ? { disposition: reconciled } : null
      })
      // An already-visible met report stays on screen with its footer
      // collapsed to "Got it" (the same path handleSave's post-save refresh
      // takes when a refresh finds nothing left to fill) rather than
      // silently discarding the user's in-progress draft. `visible` is
      // already true here, so there is nothing left to do for it.
      if (outcome.kind === "met") return
      setVisible(true)
    })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [request.seq, readNonce])

  // Leaving the host pages closes a dialog the user has already seen. A move
  // between two host pages is handled by the outer task-switch cleanup, not
  // by this effect noticing the path changed.
  useEffect(() => {
    if (visible && !isConnectorRuntimeDialogHostPath(pathname)) close("left-host")
  }, [visible, pathname, close])

  const outcome: DialogOutcome | null = report ? resolveDialogOutcome(report) : null
  // Re-derived every render against the report currently on screen, rather
  // than resolved once into `fieldError` and cached there -- see
  // locateFieldError's own docstring for why a cached location goes stale
  // the moment any of this dialog's three setReport call sites installs a
  // fresher report.
  const activeFieldError = fieldError && report
    ? { disposition: fieldError.disposition, location: locateFieldError(report, fieldError.disposition) }
    : null
  const submitItems = report ? buildSubmitItems(report, drafts) : []
  // Only a mark on a row the current report still renders an editable control
  // for may gate submission. A key a refreshed report reports satisfied loses
  // its textarea, so its mark could never be cleared again -- submit would
  // stay disabled with no error anywhere on screen. Derived from the report
  // during render rather than pruned at each point that installs one, because
  // there are three such points today and a fourth would silently reintroduce
  // this. Reads the same `hasLiveInvalidObjectMark` predicate the row
  // renderer reads for its error message, so the two can never disagree.
  const hasInvalidObjectDraft = report !== null && report.connectors.some(connector =>
    connector.inputs.some(input => hasLiveInvalidObjectMark(connector, input, invalidDraftKeys)),
  )
  const canSubmit = isSubmitEnabled(submitItems, hasInvalidObjectDraft)
  // Whether any submission-shaped action is in flight: an explicit save
  // (`submitting`, which may itself run a resend as part of "save and
  // resend") or a standalone retry resend from the send-failed panel
  // (`resending`). The footer `canSubmitNow` gates below is only ever
  // rendered while `sendFailed` is false, and `resending` only runs while
  // `sendFailed` is true -- its own button lives inside that panel -- so
  // folding `resending` into `busy` makes no difference to the footer
  // buttons today. What it does gate is `handleDismiss`/`handleOpenChange`
  // further down, which must keep the dialog open while either kind of
  // submission has not yet settled.
  const busy = submitting || resending
  // Whether a read for the attempt the dialog is currently on is still out.
  // Only used to say so on screen and to tell a pending read apart from a
  // failed one below; the gates read `reportIsStale`.
  const reading = settledReadKey !== readKey
  // Whether the report on screen belongs to some earlier request than the one
  // the dialog is now for. A same-task retarget bumps `seq` and starts a
  // fresh read while the previous report is still rendered; submitting or
  // resending against that report acts on rows the current one may no longer
  // declare, and a stored context value is immutable (see the note below), so
  // there is no correcting it afterwards.
  //
  // This is the gate rather than `reading`, because the two come apart in
  // exactly the case that matters: a read that fails settles without
  // installing anything, so `reading` goes false while the report on screen
  // is still the previous request's. `reportSeq` can only catch up in the
  // same `.then` that records the attempt as settled, so this is true
  // whenever `reading` is -- one gate covers the pending read and the failed
  // one both.
  const reportIsStale = reportSeq !== request.seq
  // The read for this request settled and left the previous request's report
  // on screen: it failed. Derived rather than stored, so it cannot disagree
  // with the two facts it is made of.
  const readFailed = !reading && reportIsStale
  // The one value every entry point into a submission reads: both footer
  // save buttons, the retry button a retryable failure offers, and
  // handleSave itself. The retry button used to be rendered off
  // `dialogFieldError.retry` alone, so a draft edited into an invalid object
  // after a failed save stayed submittable through it while the save buttons
  // were correctly disabled. That matters because buildSubmitItems drops an
  // unparsable object draft instead of failing: such a batch writes every
  // other field, silently loses that one and closes the dialog -- and a
  // stored context value is immutable, so there is no correcting it
  // afterwards. handleSave re-checks rather than trusting its callers, so a
  // fourth entry point cannot reintroduce the same bypass.
  //
  // `reportIsStale` is folded in here and not into `busy`, which also gates
  // dismissal: nothing in flight may stand between the user and closing this
  // dialog. That is not the whole picture today and the comment must not
  // pretend it is -- `submitting` is part of `busy`, so the save POST, the
  // refresh GET a failed save runs, and both sends do hold the dialog open
  // while they are out. Each of those is now bounded (the two
  // connector-runtime calls time out after 20 seconds, and a send settles or
  // rejects), so none of them can hold it open indefinitely any more, but
  // taking `submitting` out of `busy` would change what closing does on four
  // separate paths mid-write and is not part of this change.
  const canSubmitNow = canSubmit && !busy && !reportIsStale
  const hasResendPayload = request.resendPayload !== null
  const actions = outcome ? resolveDialogActions(outcome, hasResendPayload) : []
  // Whether this shape offers any way to submit. The row renderer asks this
  // instead of listing the outcome kinds that offer none, because that list
  // was one kind short: a `met` report reaches the render whenever one is
  // installed into a dialog that stays open -- the refresh a failed save
  // triggers, a same-task re-request's read that finds nothing missing, and
  // a save-and-resend's own successful save for as long as its resend is in
  // flight -- and an unfilled *optional* context key inside one was still
  // drawn as an editable field with no button able to send it. Derived from
  // the action set, so the rows and the footer cannot disagree about whether
  // saving is possible.
  const hasSaveEntryPoint = actions.includes("saveOnly")
  // A met report that reached the render still carries the snapshot of the
  // message that failed, and this shape offers no way to send it: the
  // footer collapses to "Got it", and the save-and-resend button a
  // fillable report offers cannot be reused here because a met report
  // produces no submittable items, which leaves it permanently disabled
  // (isSubmitEnabled in connector-runtime-api.ts). Nothing is missing any
  // more, so the header must stop saying one is and must say instead what
  // did not happen and what the user can still do. Not while the
  // send-failed panel is up: that panel is about a send this dialog
  // already attempted and carries its own retry button, so pointing the
  // user back at the message box there would contradict the button
  // directly under it. Nor while this dialog's own save-and-resend is
  // still sending that same message: saying it was not resent would be
  // false and would invite a second send while the first is still on the
  // wire.
  const metHoldingSnapshot = outcome?.kind === "met" && hasResendPayload && !sendFailed && !busy

  const handleDraftChange =(connector: ConnectorRuntimeConnector, input: ConnectorRuntimeInput, value: string) => {
    const draftKey = connectorRuntimeInputDraftKey(connector.connector_ref, input.section, input.key, input.type)
    setDrafts(prev => ({ ...prev, [draftKey]: value }))
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
    setInvalidDraftKeys((prev) => {
      const next = new Map(prev)
      if (reason) next.set(draftKey, reason)
      else next.delete(draftKey)
      return next
    })
  }

  const doResend = async (): Promise<ResendOutcome> => {
    const snapshot = requestRef.current.resendPayload
    if (!snapshot) {
      // Reachable, and covered by a regression test ("says the message did
      // not go out when the settlement lands mid save-and-resend"). Both
      // callers check a precondition that implies a snapshot exists, but a
      // precondition only holds until the next await: a settlement frame for
      // this task drops the snapshot in place (forgetDelivery), without
      // moving `seq`, so nothing re-runs the read effect and nothing else
      // notices. handleRetryResend cannot get here -- it compares the
      // panel's snapshot id against the one the request carries right now,
      // and awaits nothing between that comparison and the read above -- but
      // handleSave can: its save POST is awaited in between, and the
      // settlement can land during it. Kept distinct from "sent" and
      // "failed" so neither this case nor a future fourth caller is
      // misreported as a completed resend or as one the server refused.
      console.warn("[connector-runtime] resend attempted with no snapshot to send")
      return { kind: "nothing-to-send" }
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
      // A definite not_sent/rejected disposition, or the server explicitly
      // demanding a new id, means this id is spent -- the next retry mints
      // fresh. An outcome_unknown one leaves open that the server already
      // durably accepted this attempt, so the next retry reuses this same id
      // rather than risking the same turn running twice under a second one.
      // A rejection carrying no disposition at all reuses it too, although
      // sendFailureTextKey establishes that such a rejection never left the
      // client: keeping an id is free, while minting one on a wrong guess is
      // not, so the cheap side is taken here and only the user-facing text
      // splits the two cases apart.
      const disposition = readSendDisposition(error)
      const mustMintNewId = (
        readRetryWithNewId(error)
        || disposition === "not_sent"
        || disposition === "rejected"
      )
      resendMessageIdRef.current = mustMintNewId ? null : { forSnapshotId: snapshot.clientMessageId, clientMessageId }
      // The disposition travels out with the outcome rather than being
      // turned into text here: the caller decides whether this dialog is
      // still on screen to raise a panel or has to settle for a toast, and
      // both have to word the same failure the same way.
      return { kind: "failed", disposition }
    }
  }

  const handleSave = async (alsoResend: boolean) => {
    if (!report || !canSubmitNow) return
    const seqAtStart = request.seq
    const items = buildSubmitItems(report, drafts)
    setSubmitting(true)
    setLastAlsoResend(alsoResend)
    const result = await submitTaskConnectorRuntimeValues(request.taskId, items)
    if (!aliveRef.current) {
      // The dialog unmounted while the save was in flight (a task switch,
      // or leaving the host pages). Nothing cancels the save when this tree
      // goes -- its only abort is its own 20-second timeout -- so a
      // result.ok here already wrote an immutable value server-side, and the
      // resend the user asked for is never going to run. Nothing else in
      // this render tree still holds the
      // state to report that; this toast is the only place left to say
      // so, matching the sibling "superseded" branch right below.
      if (alsoResend && result.ok) toast(t("connectorRuntime.savedNotResentUnmounted"))
      return
    }
    if (requestRef.current.seq !== seqAtStart) {
      // A newer request retargeted this same dialog instance while the save
      // was in flight; the result is stale, but `submitting` must still
      // reset or the save buttons and close handlers stay stuck forever.
      // A successful "save and resend" whose resend never got to run needs
      // to say so, matching the other two "did not resend" paths below --
      // this one only fires when the save itself landed, since a rejected
      // save has nothing that was "saved but not resent" to report.
      if (alsoResend && result.ok) {
        toast(t("connectorRuntime.savedNotResentSuperseded"))
      }
      // A rejected save says so too. The dialog is still on screen and the
      // user's draft is still in it, so leaving silently would show a save
      // that simply stopped. A toast rather than the field error the
      // non-superseded path below sets: the report on screen is the newer
      // request's by now, and locating this rejection against it would pin
      // the server's reason to whatever row happens to hold that key in a
      // report the rejected draft was never built from. The whole-dialog
      // wording is used for the same reason -- there is no field here this
      // rejection can claim to be about.
      if (!result.ok) {
        toast(translateDialogScopeFailure(t, classifySubmitFailure(result, report).messageKey))
      }
      setSubmitting(false)
      return
    }

    if (!result.ok) {
      const disposition = classifySubmitFailure(result, report)
      setFieldError({ disposition })
      if (!disposition.refresh) {
        setSubmitting(false)
        return
      }
      // The save buttons stay disabled across the refresh: re-enabling before
      // it settles lets a second submit go out built from the report this
      // refresh is about to replace. Every path that settles the dialog below
      // still resets `submitting`, or the buttons and the close handlers stay
      // stuck forever; only an unmounted instance skips it, since it has no
      // buttons left to re-enable.
      const refreshed = await fetchTaskConnectorRuntimeRequirements(request.taskId)
      if (!aliveRef.current) {
        // This branch only runs after the save itself failed a few lines
        // above, so nothing was written server-side -- there is no "saved
        // but not resent" fact to report here, unlike the early return
        // right after the save POST above.
        return
      }
      if (requestRef.current.seq !== seqAtStart) {
        setSubmitting(false)
        return
      }
      if (refreshed.ok) {
        setReport(refreshed.report)
        // Same seq the save started under, proved unchanged by the check
        // above: this writes the value `reportSeq` already holds. Written
        // anyway so that every place a report is installed also says which
        // request it is for, and a fifth such place cannot be added without
        // the question being asked.
        setReportSeq(seqAtStart)
        // A type-mismatch hint names a specific declared type; once the
        // refreshed report shows this row now declares the other type, that
        // hint no longer describes the row it is attached to and must be
        // cleared outright rather than left to describe a type this row no
        // longer has. The one hint that names no type -- "this connector
        // does not say which type it expects" -- is re-derived rather than
        // cleared when the refresh finally declares one, since that refresh
        // is exactly what answers it.
        const reconciled = reconcileTypeMismatchDisposition(disposition, refreshed.report)
        if (reconciled !== disposition) setFieldError(reconciled ? { disposition: reconciled } : null)
      }
      setSubmitting(false)
      return
    }

    setReport(result.report)
    setReportSeq(seqAtStart)
    // A save that landed has no rejection left to show, even on the one path
    // below that renders before this dialog settles (a "save and resend"
    // whose report comes back met, which awaits the resend before closing):
    // without this, that rejection would re-derive against the fresh report
    // and land at whole-dialog scope, next to a send-failed panel for a save
    // that in fact succeeded.
    setFieldError(null)
    const newOutcome = resolveDialogOutcome(result.report)
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
    const notResentText = alsoResend ? savedNotResentText(t, newOutcome) : null
    if (notResentText) {
      toast(notResentText)
    } else if (newOutcome.kind === "unsupported_only") {
      toast(t("connectorRuntime.onlyUnsupportedRemaining", { keys: uniqueKeys(newOutcome.blocking).join(", ") }))
    }

    if (newOutcome.kind === "fillable" || newOutcome.kind === "nothing_fillable") {
      // Still blocked on something this dialog can collect (or, for
      // nothing_fillable, on nothing the user can act on beyond "Got it"):
      // stay open, re-render from the fresh report.
      setSubmitting(false)
      setFieldError(null)
      return
    }

    if (alsoResend && canResendNow) {
      const resendOutcome = await doResend()
      if (!aliveRef.current) {
        // The save has landed and the resend has run to completion. A sent
        // message shows up in the transcript on its own, so that outcome
        // stays silent. A failed one would normally surface in this dialog's
        // send-failed panel, which an unmounted instance can never render --
        // and doResend's console.warn reaches no user -- so say it once,
        // globally, without touching state or the provider.
        if (resendOutcome.kind !== "sent") toast(t(resendFailureTextKey(resendOutcome)))
        return
      }
      if (requestRef.current.seq !== seqAtStart) {
        // Same reason as the earlier seq check: a newer request retargeted
        // this dialog instance while the resend was in flight, so this
        // result is stale, but `submitting` must still reset. The resend's
        // own outcome is not stale: the values are stored either way, and
        // the message either went out or did not. Nothing the fresher
        // request renders carries that fact -- this branch leaves
        // `sendFailed` false, so no panel says it -- and the footer it
        // draws next offers "Save and resend this message" again, which a
        // user who was told nothing would press on a turn that already
        // went out. A toast rather than the send-failed panel: that
        // panel's retry button reads whichever snapshot the fresher
        // request now carries, and once that snapshot has been replaced
        // the retry goes out under a new client message id rather than the
        // one the attempt that just settled here used
        // (xorbitsai/xagent#2502).
        // "nothing-to-send" maps with "failed" on purpose -- both mean no
        // message went out -- and is unreachable from here anyway, since
        // this block only runs when doResend was called with a snapshot.
        setSubmitting(false)
        toast(resendOutcome.kind === "sent"
          ? t("connectorRuntime.resendSupersededSent")
          : t(resendFailureTextKey(resendOutcome)))
        return
      }
      if (resendOutcome.kind !== "sent") {
        setSubmitting(false)
        // The seq check just above proves no retarget landed while the
        // resend was in flight, so the snapshot the request carries here
        // is still the one doResend read -- which is what the panel this
        // raises is about, and what its retry button would send.
        const failedSnapshotId = requestRef.current.resendPayload?.clientMessageId ?? null
        if (failedSnapshotId === null) {
          // A settlement frame for this task arrived while the save was in
          // flight and took the snapshot with it (forgetDelivery), without
          // reopening the dialog and so without moving `seq`. There is
          // nothing left to retry, and a panel here would draw a retry
          // button with nothing behind it -- so say the same thing the
          // panel says, once, and leave the dialog on its report.
          toast(t(resendFailureTextKey(resendOutcome)))
          return
        }
        setSendFailure({
          snapshotId: failedSnapshotId,
          disposition: resendOutcome.kind === "failed" ? resendOutcome.disposition : null,
        })
        return
      }
    }
    setSubmitting(false)
    close(alsoResend && canResendNow ? "resent" : "dismissed")
  }

  const handleRetryResend = async () => {
    // A resend is one billed model call plus a possibly side-effecting tool
    // run; a double click here must not fire it twice.
    if (resending) return
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
    if (
      sendFailure === null
      || sendFailure.snapshotId !== requestRef.current.resendPayload?.clientMessageId
    ) {
      setSendFailure(null)
      return
    }
    // What the panel says right now, captured before the await below: a
    // failed attempt is reported against the wording the panel ends up
    // carrying, not against this attempt's own outcome, and by the time
    // this settles the state behind that wording may already have moved.
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
    // explanation next to it.
    if (!report || !canResendReport(report)) {
      setSendFailure(null)
      const reason = report ? savedNotResentText(t, resolveDialogOutcome(report)) : null
      if (reason) toast(reason)
      return
    }
    const seqAtStart = request.seq
    setResending(true)
    const resendOutcome = await doResend()
    if (!aliveRef.current) {
      // A retry that went out shows up in the transcript on its own, so that
      // outcome stays silent, matching handleSave's unmounted exit above. A
      // retry that failed leaves things exactly as the user last saw them --
      // saved, not sent -- with the panel that said so already gone, so
      // without this nothing would tell them the retry they pressed changed
      // nothing. Worded off the panel's own wording rather than this
      // attempt's, so a refusal cannot un-say an earlier unknown outcome.
      // Counting every exit of this handler in one place is still tracked in
      // xorbitsai/xagent#2478.
      if (resendOutcome.kind !== "sent") {
        toast(t(sendFailureTextKey(mergeSendFailureDisposition(
          dispositionBefore,
          resendOutcome.kind === "failed" ? resendOutcome.disposition : null,
        ))))
      }
      return
    }
    if (requestRef.current.seq !== seqAtStart) {
      // A newer request retargeted this same dialog instance while the
      // resend was in flight; the result is stale, but `resending` must
      // still reset or the retry button stays stuck forever. A resend that
      // did go out needs to say so: the send-failed panel this button
      // lives on is about to be replaced by whatever the fresher request
      // renders next, and without a toast the user has no way to tell
      // that clicking a resend button there would send this same turn a
      // second time.
      setResending(false)
      if (resendOutcome.kind === "sent") toast(t("connectorRuntime.resendSupersededSent"))
      return
    }
    setResending(false)
    if (resendOutcome.kind === "sent") {
      setSendFailure(null)
      close("resent")
      return
    }
    // A retry that failed again. The panel's wording only ever moves toward
    // uncertainty (see mergeSendFailureDisposition), which means that in the
    // two cases where it does not move at all the panel says exactly what it
    // said before the click: the user cannot tell "nothing happened" from
    // "it failed again". So the panel is updated and the fact is said once,
    // both off the same merged disposition -- the toast carries the new
    // event, the panel carries the standing state, and the two cannot word
    // the same message differently.
    const disposition = mergeSendFailureDisposition(
      dispositionBefore,
      resendOutcome.kind === "failed" ? resendOutcome.disposition : null,
    )
    setSendFailure(prev => (prev ? { ...prev, disposition } : prev))
    toast(t(sendFailureTextKey(disposition)))
  }

  // A resend in flight holds the dialog open for the same reason a save does:
  // dismissing mid-send drops the request (and with it the snapshot the retry
  // button reads), leaving nothing to retry from if that send fails.
  const handleDismiss = () => {
    if (busy) return
    close("dismissed")
  }

  const handleOpenChange = (open: boolean) => {
    if (open || busy) return
    handleDismiss()
  }

  const dialogFieldError = activeFieldError?.location.scope === "dialog" ? activeFieldError.disposition : null

  if (!visible || !report || !outcome) return null

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
            {t(
              outcome.kind !== "met"
                ? "connectorRuntime.description"
                // Pointing the user back at the message box is only right
                // while this dialog is holding a message it will not send
                // and has nothing else in flight -- metHoldingSnapshot's own
                // conditions. The neutral line covers the rest: no snapshot
                // at all, a send-failed panel that already carries the
                // message and its own retry button, and a resend of this
                // very message still on the wire.
                : metHoldingSnapshot
                  ? "connectorRuntime.metNotResent"
                  : "connectorRuntime.metNothingLeft",
            )}
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
            <Button variant="outline" onClick={() => setReadNonce(nonce => nonce + 1)}>
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
            <Button disabled={resending || reportIsStale} onClick={handleRetryResend}>
              {t("connectorRuntime.actions.resend")}
            </Button>
          </div>
        ) : (
          <div className="space-y-4">
            {report.connectors.map((connector) => {
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
              // is still submitting for as long as its resend runs, so this
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
              <Button variant="outline" disabled={!canSubmitNow} onClick={() => handleSave(lastAlsoResend)}>
                {t("connectorRuntime.actions.retry")}
              </Button>
            )}
          </DialogFooter>
        )}
      </DialogContent>
    </Dialog>
  )
}

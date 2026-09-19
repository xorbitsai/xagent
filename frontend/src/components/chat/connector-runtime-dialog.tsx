"use client"

import React, { useEffect, useRef, useState } from "react"
import { usePathname } from "next/navigation"

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
  isTypeMismatchDispositionStale,
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

interface FieldErrorState {
  disposition: ConnectorRuntimeFailureDisposition
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
  const [visible, setVisible] = useState(false)
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [invalidDraftKeys, setInvalidDraftKeys] = useState<Map<string, InvalidObjectDraftReason>>(new Map())
  const [submitting, setSubmitting] = useState(false)
  const [fieldError, setFieldError] = useState<FieldErrorState | null>(null)
  const [lastAlsoResend, setLastAlsoResend] = useState(false)
  const [sendFailed, setSendFailed] = useState(false)
  const [resending, setResending] = useState(false)

  // Read on mount and on every subsequent request for this same task (the
  // dialog is already open and a new terminal frame retargeted it): the
  // route gate runs before the request even goes out, and again right
  // before the dialog would become visible, since the user is free to
  // navigate away from a host page while this read is in flight.
  //
  // `visible` is read at the moment this effect starts, which is exactly
  // right here: setVisible is only ever called with `true` in this
  // component, so if the dialog was already showing something when this
  // request came in, it is still showing it by the time the fetch below
  // resolves (any path that would make it stop -- unmount, a task switch, a
  // host-page departure -- clears `request` and is caught by the seq/alive
  // checks first). A once-visible dialog must not vanish out from under a
  // user who is mid-draft: a re-read for the same task only happens because
  // another terminal frame retargeted this instance, not because the user
  // did anything, so it must never read as a decision the user made.
  useEffect(() => {
    if (!isConnectorRuntimeDialogHostPath(pathnameRef.current)) {
      close("not-shown")
      return
    }
    const wasVisible = visible
    const seqAtStart = request.seq
    let cancelled = false
    fetchTaskConnectorRuntimeRequirements(request.taskId).then((result) => {
      if (cancelled || !aliveRef.current || requestRef.current.seq !== seqAtStart) return
      if (!result.ok) {
        console.warn(
          "[connector-runtime] requirements read failed",
          result.kind === "http" ? result.status : result.kind,
        )
        // Keep whatever the user is already looking at (report and draft)
        // rather than discarding it over a transient read failure.
        if (!wasVisible) close("not-shown")
        return
      }
      if (!isConnectorRuntimeDialogHostPath(pathnameRef.current)) {
        close("not-shown")
        return
      }
      const outcome = resolveDialogOutcome(result.report)
      if (outcome.kind === "met") {
        if (!wasVisible) {
          close("not-shown")
          return
        }
        // Same path handleSave's post-save refresh already takes when a
        // refresh finds nothing left to fill (see "leaves a way out..."
        // below): install the report and let the footer collapse to "Got
        // it" instead of silently discarding the user's in-progress draft.
        setReport(result.report)
        return
      }
      setReport(result.report)
      setFieldError(null)
      setSendFailed(false)
      setVisible(true)
    })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [request.seq])

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
  // (`submitting`) or a resend retried directly off a failed send
  // (`resending`). A same-task terminal frame that arrives while a retry
  // resend is still awaiting `doResend` -- for example, a second tab's own
  // broadcast of the same failure -- runs the read effect above, which
  // clears `sendFailed` and brings the footer's save buttons back before
  // that resend settles. `canSubmitNow` must stay gated on this combined
  // value rather than `submitting` alone, or those buttons would let a
  // second send go out under a fresh message id while the first is still
  // unaccounted for.
  const busy = submitting || resending
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
  const canSubmitNow = canSubmit && !busy
  const hasResendPayload = request.resendPayload !== null
  const actions = outcome ? resolveDialogActions(outcome, hasResendPayload) : []
  // Whether this shape offers any way to submit. The row renderer asks this
  // instead of listing the outcome kinds that offer none, because that list
  // was one kind short: a `met` report reaches the render only through the
  // refresh a failed save triggers, and an unfilled *optional* context key
  // inside one was still drawn as an editable field with no button able to
  // send it. Derived from the action set, so the rows and the footer cannot
  // disagree about whether saving is possible.
  const hasSaveEntryPoint = actions.includes("saveOnly")

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
        // message tells the two apart.
        if (!isSubmittableObjectValue(parsed)) {
          reason = typeof parsed === "object" && parsed !== null && !Array.isArray(parsed) ? "empty" : "invalid"
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

  type ResendOutcome = "sent" | "failed" | "nothing-to-send"

  const doResend = async (): Promise<ResendOutcome> => {
    const snapshot = requestRef.current.resendPayload
    if (!snapshot) {
      // Unreachable today: both callers only reach this after a precondition
      // that implies a snapshot exists -- handleSave's canResendNow (which
      // itself requires the "met" outcome the saveAndResend button promised
      // to be resendable) and handleRetryResend's sendFailed precondition
      // (set only right after a resend that read a snapshot). Kept distinct
      // from "sent" and "failed" so a future caller that does reach it is
      // not misreported as either a completed resend or a failed one.
      console.warn("[connector-runtime] resend attempted with no snapshot to send")
      return "nothing-to-send"
    }
    try {
      // Matches every other programmatic resend call site in the app
      // (clarification-form.tsx, workforce-builder.tsx, agent-builder.tsx):
      // without force, a duplicate of this exact text still pending from an
      // earlier send on this same connection throws instead of sending.
      await sendMessage(
        snapshot.text,
        { clientMessageId: generateClientMessageId(), force: true },
        snapshot.files,
      )
      return "sent"
    } catch {
      // Matches the read path's warn so a failing resend leaves the same
      // diagnostic signal. Carries the fixed prefix alone: unlike the read
      // path there is no closed-set status to report here, and the rejection
      // value is arbitrary, so logging it could carry message content.
      console.warn("[connector-runtime] resend failed")
      return "failed"
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
      // or leaving the host pages) -- submitTaskConnectorRuntimeValues has
      // no abort signal, so a result.ok here already wrote an immutable
      // value server-side, and the resend the user asked for is never
      // going to run. Nothing else in this render tree still holds the
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
        // A type-mismatch hint names a specific declared type; once the
        // refreshed report shows this row now declares the other type, that
        // hint no longer describes the row it is attached to and must be
        // cleared outright rather than left to describe a type this row no
        // longer has.
        if (isTypeMismatchDispositionStale(disposition, refreshed.report)) setFieldError(null)
      }
      setSubmitting(false)
      return
    }

    setReport(result.report)
    // A save that landed has no rejection left to show, even on the one path
    // below that renders before this dialog settles (a "save and resend"
    // whose report comes back met, which awaits the resend before closing):
    // without this, that rejection would re-derive against the fresh report
    // and land at whole-dialog scope, next to a send-failed panel for a save
    // that in fact succeeded.
    setFieldError(null)
    const newOutcome = resolveDialogOutcome(result.report)
    // Only a met report can carry the resend the primary button promised.
    // `unsupported_only` still lacks a required secret this dialog cannot
    // collect, and `nothing_fillable` is a connector the server still reports
    // unavailable with nothing left for the user to fill; the backend rejects
    // either while it builds the turn's tool list, so a resend would fail on
    // the same gate and put a second failure in the conversation. Neither
    // resends, and because the button promised one, both say so.
    const canResendNow = newOutcome.kind === "met"

    if (newOutcome.kind === "unsupported_only") {
      const keys = uniqueKeys(newOutcome.blocking).join(", ")
      toast(alsoResend
        ? t("connectorRuntime.savedNotResentUnsupported", { keys })
        : t("connectorRuntime.onlyUnsupportedRemaining", { keys }))
    } else if (alsoResend && newOutcome.kind === "nothing_fillable") {
      toast(t("connectorRuntime.savedNotResentUnavailable"))
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
        if (resendOutcome !== "sent") toast(t("connectorRuntime.sendFailed"))
        return
      }
      if (requestRef.current.seq !== seqAtStart) {
        // Same reason as the earlier seq check: a newer request retargeted
        // this dialog instance while the resend was in flight, so this
        // result is stale, but `submitting` must still reset.
        setSubmitting(false)
        return
      }
      if (resendOutcome !== "sent") {
        setSubmitting(false)
        setSendFailed(true)
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
    const seqAtStart = request.seq
    setResending(true)
    const resendOutcome = await doResend()
    if (!aliveRef.current) {
      // Same reasoning as handleSave's post-doResend early return above:
      // the resend was already attempted by the time doResend resolves, so
      // there is nothing new for a toast to report here.
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
      if (resendOutcome === "sent") toast(t("connectorRuntime.resendSupersededUnknown"))
      return
    }
    setResending(false)
    if (resendOutcome === "sent") {
      setSendFailed(false)
      close("resent")
    }
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
          <DialogTitle>{t("connectorRuntime.title")}</DialogTitle>
          <DialogDescription>{t("connectorRuntime.description")}</DialogDescription>
        </DialogHeader>

        {outcome.kind === "unsupported_only" && (
          <p className="text-sm text-muted-foreground">{t("connectorRuntime.onlyUnsupportedNotice")}</p>
        )}

        {dialogFieldError && (
          <p className="text-sm text-destructive" role="alert">
            {/* This scope has no row identity in hand -- it is where
                locateFieldError falls back when the row a failure named is
                gone or unrecognized, most often a 409 conflict whose
                refresh just collapsed that row into "already filled". A
                conflict is the only messageKey whose text carries a
                {key} placeholder, and there is no key here to fill it
                with, so it gets a placeholder-free variant instead of
                going through translateFailure like every other reason. */}
            {dialogFieldError.messageKey === "conflict"
              ? t("connectorRuntime.errors.conflictNoKey")
              : translateFailure(t, dialogFieldError.messageKey)}
          </p>
        )}

        {sendFailed ? (
          <div className="space-y-3">
            <p className="text-sm text-destructive">{t("connectorRuntime.sendFailed")}</p>
            <Button disabled={resending} onClick={handleRetryResend}>{t("connectorRuntime.actions.resend")}</Button>
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
              <Button variant="outline" onClick={handleDismiss}>
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
            {dialogFieldError?.retry && (
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

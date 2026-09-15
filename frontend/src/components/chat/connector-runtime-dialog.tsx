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

  // Task-switch cleanup: no cleanup function of its own. Combining this with
  // the unmount effect below into one effect would run the unmount cleanup
  // on every task change too, which would erase a same-render first-gate
  // snapshot before anything could read it.
  useEffect(() => {
    retainOnlyTask(state.taskId)
  }, [state.taskId, retainOnlyTask])

  // Unmount cleanup: a separate effect with an empty dependency array, read
  // through a ref so it always calls the latest function without needing to
  // be in that array (matches the workforce pages' own unmount-cleanup shape).
  useEffect(() => () => cleanupRef.current(null), [])

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
  invalidDraftKeys: Set<string>,
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
  if (!input) return { scope: "dialog" }
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
  location: FieldErrorLocation
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
  const [invalidDraftKeys, setInvalidDraftKeys] = useState<Set<string>>(new Set())
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
  useEffect(() => {
    if (!isConnectorRuntimeDialogHostPath(pathnameRef.current)) {
      close("not-shown")
      return
    }
    const seqAtStart = request.seq
    let cancelled = false
    fetchTaskConnectorRuntimeRequirements(request.taskId).then((result) => {
      if (cancelled || !aliveRef.current || requestRef.current.seq !== seqAtStart) return
      if (!result.ok) {
        console.warn(
          "[connector-runtime] requirements read failed",
          result.kind === "http" ? result.status : result.kind,
        )
        close("not-shown")
        return
      }
      if (!isConnectorRuntimeDialogHostPath(pathnameRef.current)) {
        close("not-shown")
        return
      }
      const outcome = resolveDialogOutcome(result.report)
      if (outcome.kind === "met") {
        close("not-shown")
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
    let invalid = false
    if (value.trim() !== "") {
      try {
        const parsed: unknown = JSON.parse(value)
        invalid = typeof parsed !== "object" || parsed === null || Array.isArray(parsed)
      } catch {
        invalid = true
      }
    }
    setInvalidDraftKeys((prev) => {
      const next = new Set(prev)
      if (invalid) next.add(draftKey)
      else next.delete(draftKey)
      return next
    })
  }

  const doResend = async (): Promise<boolean> => {
    const snapshot = requestRef.current.resendPayload
    if (!snapshot) return true
    try {
      await sendMessage(snapshot.text, { clientMessageId: generateClientMessageId() }, snapshot.files)
      return true
    } catch {
      // Matches the read path's warn so a failing resend leaves the same
      // diagnostic signal. Carries the fixed prefix alone: unlike the read
      // path there is no closed-set status to report here, and the rejection
      // value is arbitrary, so logging it could carry message content.
      console.warn("[connector-runtime] resend failed")
      return false
    }
  }

  const handleSave = async (alsoResend: boolean) => {
    if (!report || !canSubmitNow) return
    const seqAtStart = request.seq
    const items = buildSubmitItems(report, drafts)
    setSubmitting(true)
    setLastAlsoResend(alsoResend)
    const result = await submitTaskConnectorRuntimeValues(request.taskId, items)
    if (!aliveRef.current) return
    if (requestRef.current.seq !== seqAtStart) {
      // A newer request retargeted this same dialog instance while the save
      // was in flight; the result is stale, but `submitting` must still
      // reset or the save buttons and close handlers stay stuck forever.
      setSubmitting(false)
      return
    }

    if (!result.ok) {
      const disposition = classifySubmitFailure(result, report)
      setFieldError({ disposition, location: locateFieldError(report, disposition) })
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
      if (!aliveRef.current) return
      if (requestRef.current.seq !== seqAtStart) {
        setSubmitting(false)
        return
      }
      if (refreshed.ok) setReport(refreshed.report)
      setSubmitting(false)
      return
    }

    setReport(result.report)
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
      const sent = await doResend()
      if (!aliveRef.current) return
      if (requestRef.current.seq !== seqAtStart) {
        // Same reason as the earlier seq check: a newer request retargeted
        // this dialog instance while the resend was in flight, so this
        // result is stale, but `submitting` must still reset.
        setSubmitting(false)
        return
      }
      if (!sent) {
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
    const sent = await doResend()
    if (!aliveRef.current) return
    if (requestRef.current.seq !== seqAtStart) {
      // A newer request retargeted this same dialog instance while the
      // resend was in flight; the result is stale, but `resending` must
      // still reset or the retry button stays stuck forever.
      setResending(false)
      return
    }
    setResending(false)
    if (sent) {
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

  const dialogFieldError = fieldError && fieldError.location.scope === "dialog" ? fieldError.disposition : null

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
            {translateFailure(t, dialogFieldError.messageKey)}
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
                fieldError
                && fieldError.location.scope === "connector"
                && fieldError.location.connectorKey === connectorKey
                  ? fieldError.disposition
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
                      fieldError
                      && fieldError.location.scope === "field"
                      && fieldError.location.draftKey === draftKey
                        ? fieldError.disposition
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
                          <p className="text-sm text-destructive">{t("connectorRuntime.objectInvalid")}</p>
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

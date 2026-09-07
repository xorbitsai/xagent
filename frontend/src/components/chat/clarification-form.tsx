import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import { Interaction } from "@/contexts/app-context-chat"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { MultiSelect } from "@/components/ui/multi-select"
import { Select } from "@/components/ui/select"
import { useApp } from "@/contexts/app-context-chat"
import { useI18n } from "@/contexts/i18n-context"
import { toast } from "@/components/ui/sonner"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { ChevronDown, ChevronRight, MessageSquare, Upload, File as FileIcon, X, Globe } from "lucide-react"
import { ConnectAppsField } from "./connect-apps-field"
import type { MessageDeliveryDisposition } from "@/hooks/use-websocket"
import { generateClientMessageId } from "@/lib/utils"
import { clientErrorTranslationKey, type ClientErrorCode } from "@/lib/client-errors"
import {
  readSendDisposition,
  readSendErrorCode,
  readSendReason,
  readSendRetryWithNewId,
  sendHintKey,
  type ClarificationOnSend,
  type ClarificationSendAttempt,
  type ClarificationSendMetadata,
} from "./clarification-delivery"

interface ClarificationFormProps {
  message?: string
  interactions: Interaction[]
  messageId?: string
  requestId?: string
  active?: boolean
  filesDisabled?: boolean
  /**
   * Injected send path (e.g. the agent builder chat). Held to the delivery
   * contract in clarification-delivery.ts: it receives the attempt identity
   * the internal path sends, and rejects with the same discriminated failure
   * shape (#1485).
   */
  onSend?: ClarificationOnSend
}

const FILE_ACTION_VALUE_RE = /(^|[-_\s])(upload|file)(?=$|[-_\s])/i

const isFileActionValue = (value: unknown): boolean =>
  typeof value === "string" && FILE_ACTION_VALUE_RE.test(value)

const isFileActionOption = (
  option: { value?: string; action_type?: string } | undefined,
): boolean => {
  const actionType = option?.action_type?.toLowerCase()
  if (actionType !== undefined) {
    return actionType === "upload"
  }
  return isFileActionValue(option?.value)
}

const isFileActionSelection = (
  option: { value?: string; action_type?: string } | undefined,
  value: unknown,
): boolean => option
  ? isFileActionOption(option)
  : isFileActionValue(value)

// Interaction types that are "live widgets" reflecting external state (e.g.
// useMcpApps()'s connection state), not a question with an answer to submit
// - see the comment on isConnectAppsOnly below for why that distinction
// matters. Named so a list mixing one of these with an ordinary field (not
// currently produced by any seeder, but nothing in the type system or
// backend schema rules it out) still renders correctly via renderField's
// switch below instead of falling through to its "unsupported type" case.
const LIVE_WIDGET_TYPES = new Set(["connect_apps"])

export function ClarificationForm({
  interactions,
  messageId,
  requestId,
  active = true,
  filesDisabled: filesDisabledOverride,
  onSend,
}: ClarificationFormProps) {
  // If onSend is provided, use it (e.g., from builder chat), otherwise use useApp
  let sendMessage: any, dispatch: any, contextFilesDisabled: boolean | undefined;
  try {
    const appCtx = useApp();
    sendMessage = appCtx.sendMessage;
    dispatch = appCtx.dispatch;
    contextFilesDisabled = appCtx.filesDisabled;
  } catch {
    // We might not be in the app context (e.g., agent builder chat)
  }
  const filesDisabled = filesDisabledOverride ?? contextFilesDisabled ?? true

  // "connect_apps" is a live widget (OAuth-popup buttons that reflect
  // useMcpApps()'s current connection state), not a question-and-submit
  // form field - it has no "answer" to gate behind `active`/waiting_for_user
  // the way every other interaction type does (see the type's doc comment
  // on Interaction.apps). A seed message attached at task-creation time
  // (the marketplace Hire flow) never transitions the task through
  // waiting_for_user at all, so gating this on `active` the normal way
  // would leave it permanently collapsed and inert.
  const isConnectAppsOnly =
    interactions.length > 0 && interactions.every((interaction) => LIVE_WIDGET_TYPES.has(interaction.type))

  const { t } = useI18n()

  // Ignore the persisted interaction.label for a live-widget type (it's only
  // ever a t()'d string frozen into the DB row at hire time - see
  // hire-agent.ts's buildConnectAppsInteraction/HireMessageStrings) and
  // re-resolve live instead, so a locale switch after hiring doesn't leave
  // it stuck in whatever language was active when the task was created. Used
  // by both the isConnectAppsOnly header below and the per-field label in
  // the mixed-interaction-list branch further down, since either render path
  // can be the one displaying a live-widget interaction's label.
  const fieldLabel = (interaction: Interaction): string =>
    LIVE_WIDGET_TYPES.has(interaction.type)
      ? t("chatPage.clarification.connectApps.title")
      : interaction.label || interaction.field

  const [formState, setFormState] = useState<Record<string, any>>({})
  const previousRequestIdRef = useRef(requestId)
  const latestRequestIdRef = useRef(requestId)
  // One delivery identity per unresolved submission, handed to both send
  // paths: an unchanged resubmit presents the same clientMessageId (so a
  // dedup-capable consumer can recognize the retry), while a changed draft,
  // a delivered send, or a `retryWithNewId` failure mints a fresh one. Same
  // semantics as ChatInput's delivery attempt. Submit and the connect-apps
  // skip are independent submissions a mixed interaction list renders side
  // by side, so each keeps its own slot - a skip between two submits of the
  // same draft must not burn the submit's identity.
  type DeliveryAttempt = { key: string } & ClarificationSendAttempt
  const submitAttemptRef = useRef<DeliveryAttempt | null>(null)
  const skipAttemptRef = useRef<DeliveryAttempt | null>(null)

  const resolveDeliveryAttempt = (
    ref: React.MutableRefObject<DeliveryAttempt | null>,
    deliveryKey: string,
  ): string => {
    const previous = ref.current
    const clientMessageId = previous?.key === deliveryKey
      ? previous.clientMessageId
      : generateClientMessageId()
    ref.current = { key: deliveryKey, clientMessageId }
    return clientMessageId
  }

  // Clears only the attempt this send owns: a concurrent send of the same
  // kind (e.g. a double-clicked skip) may have replaced the slot with its
  // own attempt, which must survive this send's resolution.
  const settleDeliveryAttempt = (
    ref: React.MutableRefObject<DeliveryAttempt | null>,
    clientMessageId: string,
  ) => {
    if (ref.current?.clientMessageId === clientMessageId) {
      ref.current = null
    }
  }
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isSubmitted, setIsSubmitted] = useState(!active && !isConnectAppsOnly)
  const [isOpen, setIsOpen] = useState(active || isConnectAppsOnly)
  // Raw evidence only. Translating at render (not at failure time) is what
  // lets a locale switch reach an alert that is already on screen.
  const [sendFailure, setSendFailure] = useState<
    {
      detail: string
      disposition: MessageDeliveryDisposition | null
      errorCode: ClientErrorCode | null
    } | null
  >(null)

  useLayoutEffect(() => {
    latestRequestIdRef.current = requestId
    if (previousRequestIdRef.current === requestId) return
    previousRequestIdRef.current = requestId
    setFormState({})
    setIsSubmitting(false)
    setIsSubmitted(!active && !isConnectAppsOnly)
    setIsOpen(active || isConnectAppsOnly)
    setSendFailure(null)
    // A new request is a new question; its answer must never present the
    // previous round's unresolved delivery identity.
    submitAttemptRef.current = null
    skipAttemptRef.current = null
  }, [active, isConnectAppsOnly, requestId])

  useEffect(() => {
    if (active) {
      // A new clarification round reuses this component instance on the live
      // turn render path, so a stale round-1 failure alert would sit on top
      // of round 2's question.
      setIsSubmitted(false)
      setIsOpen(true)
      setSendFailure(null)
    }
  }, [active])

  const normalizedInteractions = useMemo(() => {
    const seenFields = new Set<string>()
    return interactions.map((interaction: any, index) => {
      const rawType = interaction.type
      const type =
        rawType === "text" || rawType === "input" || rawType === "textarea" || rawType === "string"
          ? "text_input"
          : rawType === "file" || rawType === "upload"
            ? "file_upload"
            : rawType === "number" || rawType === "integer"
              ? "number_input"
              : rawType === "boolean"
                ? "confirm"
                : rawType
      const rawField = interaction.field || interaction.id || interaction.name || interaction.properties?.field || interaction.properties?.id || `response_${index}`
      const baseField = typeof rawField === "string" && rawField.trim() ? rawField.trim() : `response_${index}`
      const field = seenFields.has(baseField) ? `${baseField}_${index}` : baseField
      seenFields.add(field)
      const rawOptions = Array.isArray(interaction.options)
        ? interaction.options
        : Array.isArray(interaction.actions)
          ? interaction.actions
          : undefined
      const options = rawOptions
        ?.map((opt: any) => ({
          value: typeof opt?.value === "string" ? opt.value : String(opt?.label || ""),
          label: typeof opt?.label === "string" ? opt.label : String(opt?.value || ""),
          description: typeof opt?.description === "string" ? opt.description : undefined,
          action_type: typeof opt?.action_type === "string" ? opt.action_type : undefined,
        }))
        .filter((opt: { value: string; label: string }) => opt.value.trim() !== "" && opt.label.trim() !== "")
      return {
        ...interaction,
        type,
        field,
        ...(options ? { options } : {}),
      }
    }) as Interaction[]
  }, [interactions])

  useEffect(() => {
    if (!filesDisabled) return

    setFormState((previous) => {
      const next = { ...previous }
      let changed = false

      for (const interaction of normalizedInteractions) {
        if (interaction.type === "file_upload" && interaction.field in next) {
          delete next[interaction.field]
          changed = true
        }

        if (interaction.type === "action_cards") {
          const fileField = `${interaction.field}_files`
          if (fileField in next) {
            delete next[fileField]
            changed = true
          }
          const selectedOption = interaction.options?.find(
            (option) => option.value === next[interaction.field],
          )
          if (isFileActionSelection(
            selectedOption,
            next[interaction.field],
          )) {
            delete next[interaction.field]
            changed = true
          }
        }
      }

      return changed ? next : previous
    })
  }, [filesDisabled, normalizedInteractions])

  const handleInputChange = (field: string, value: any) => {
    setFormState((prev) => ({ ...prev, [field]: value }))
    setSendFailure(null)
  }

  const handleSubmit = async () => {
    const submittedRequestId = requestId
    // Construct the message
    const metadata: ClarificationSendMetadata = requestId ? { request_id: requestId } : {}
    const lines = normalizedInteractions.flatMap(interaction => {
      const value = formState[interaction.field]

      if (filesDisabled && interaction.type === "file_upload") {
        return []
      }
      if (
        filesDisabled
        && interaction.type === "action_cards"
        && isFileActionSelection(
          interaction.options?.find((option) => option.value === value),
          value,
        )
      ) {
        return []
      }

      // Skip empty values unless it's a boolean (confirm) which might be false
      if (value === undefined || value === null || (typeof value === "string" && value.trim() === "") || (Array.isArray(value) && value.length === 0)) {
        // If it's a confirm type, default to false if undefined? Or maybe it's required?
        if (interaction.type === "confirm" && value === undefined) {
          return [{ field: interaction.field, label: interaction.label || interaction.field, value: t("chatPage.clarification.no"), isFile: false }]
        }
        return []
      }

      let displayValue = value
      let isFile = false

      if (interaction.type === "select_multiple" && Array.isArray(value)) {
        const labels = value.map(v => interaction.options?.find(o => o.value === v)?.label || v)
        displayValue = labels.join(", ")
      } else if (interaction.type === "select_one" || interaction.type === "action_cards") {
        const label = interaction.options?.find(o => o.value === value)?.label || value
        displayValue = label
      } else if (interaction.type === "confirm") {
        displayValue = value ? t("chatPage.clarification.yes") : t("chatPage.clarification.no")
      } else if (interaction.type === "file_upload") {
        isFile = true
      }

      const results = [{ field: interaction.field, label: interaction.label || interaction.field, value: isFile ? value : displayValue, isFile }]

      // For action_cards, if files were uploaded alongside it, add them too
      if (interaction.type === "action_cards" && !filesDisabled) {
        const fileValue = formState[`${interaction.field}_files`]
        if (fileValue && ((fileValue instanceof FileList && fileValue.length > 0) || (Array.isArray(fileValue) && fileValue.length > 0))) {
          results.push({ field: `${interaction.field}_files`, label: t("chatPage.clarification.uploadedFiles"), value: fileValue, isFile: true })
        }

        const urlValue = formState[`${interaction.field}_url`]
        if (urlValue && typeof urlValue === "string" && urlValue.trim() !== "") {
          results.push({ field: `${interaction.field}_url`, label: t("chatPage.clarification.websiteUrl") || "Website URL", value: urlValue, isFile: false })
          metadata.url = urlValue
        }
      }

      return results
    }).filter(Boolean) as any[]

    if (lines.length === 0) {
      toast.error(t("chatPage.clarification.required"))
      return
    }

    // Separate files and text
    const textParts = lines.filter(l => !l.isFile).map(l => `${l.label}: ${l.value}`)
    const fileParts = lines.filter(l => l.isFile)

    const textMessage = textParts.join("\n")
    const files: File[] = []

    fileParts.forEach(part => {
      if (part.value instanceof FileList) {
        for (let i = 0; i < part.value.length; i++) {
          files.push(part.value[i])
        }
      } else if (Array.isArray(part.value)) {
        // Assuming array of Files
        part.value.forEach((f: any) => {
          if (f instanceof File) files.push(f)
        })
      }
    })

    // If textMessage is empty but we have files, send a generic message?
    const outboundFiles = filesDisabled ? [] : files
    const finalMessage = textMessage || (outboundFiles.length > 0 ? t("chatPage.clarification.uploadedFiles") : t("chatPage.clarification.confirmed"))

    const clientMessageId = resolveDeliveryAttempt(submitAttemptRef, JSON.stringify([
      submittedRequestId ?? null,
      finalMessage,
      outboundFiles.map((file) => [file.name, file.size, file.lastModified]),
    ]))

    try {
      setIsSubmitting(true)
      setSendFailure(null)

      if (onSend) {
        await onSend(finalMessage, outboundFiles, metadata, { clientMessageId })
      } else if (sendMessage) {
        await sendMessage(finalMessage, { force: true, metadata, clientMessageId }, outboundFiles)
      }
      // Delivered: the identity is spent whether or not this render is stale.
      settleDeliveryAttempt(submitAttemptRef, clientMessageId)

      if (latestRequestIdRef.current !== submittedRequestId) return
      setIsSubmitted(true)
      setIsOpen(false)
      if (!onSend && dispatch) {
        dispatch({ type: "UPDATE_TASK_STATUS", payload: { status: "running" } })
      }
    } catch (error) {
      if (readSendRetryWithNewId(error)) {
        settleDeliveryAttempt(submitAttemptRef, clientMessageId)
      }
      if (latestRequestIdRef.current !== submittedRequestId) return
      console.error("Failed to send clarification response", error)
      // The rejection reason ("a previous guidance message is still being
      // applied") is the only actionable part of the failure; the fixed
      // string is a last resort.
      const detail = readSendReason(error)
      const disposition = readSendDisposition(error)
      const errorCode = readSendErrorCode(error)
      setSendFailure({ detail, disposition, errorCode })
      // The toast is a snapshot - it keeps whatever language was active when
      // it fired. The alert below is not, and re-resolves on every render.
      // The draft is preserved and Submit stays enabled in every case, so the
      // copy may only warn, never promise: an unchanged resubmit re-presents
      // the same delivery identity, but only the internal path is known to
      // dedup on it - an onSend provider may still answer the question twice.
      const hintKey = sendHintKey(disposition)
      const hint = hintKey ? t(hintKey) : null
      const errorMessage = errorCode
        ? t(clientErrorTranslationKey(errorCode))
        : detail || t("chatPage.clarification.sendError")
      toast.error(
        errorMessage,
        hint ? { description: hint } : undefined,
      )
    } finally {
      if (latestRequestIdRef.current === submittedRequestId) {
        setIsSubmitting(false)
      }
    }
  }

  // "connect_apps" has no form fields to gather - skipping just logs a
  // plain acknowledgement message, matching every other interaction type's
  // "answer becomes a chat message" contract, but without the lines/
  // formState machinery handleSubmit above uses (there's nothing to gather).
  const handleSkipConnectApps = async () => {
    const message = t("chatPage.clarification.connectApps.skip")
    const metadata: ClarificationSendMetadata = requestId ? { request_id: requestId } : {}
    // A skip is a delivery attempt like any other. The button unmounts on
    // its first click, so the reuse only covers a double-click racing ahead
    // of that re-render - but a skip must never clobber the submit slot's
    // unresolved identity, which is why it gets its own.
    const clientMessageId = resolveDeliveryAttempt(skipAttemptRef, JSON.stringify([
      "connect_apps_skip",
      requestId ?? null,
      message,
    ]))
    try {
      if (onSend) {
        await onSend(message, [], metadata, { clientMessageId })
      } else if (sendMessage) {
        await sendMessage(message, { force: true, metadata, clientMessageId }, [])
      }
      settleDeliveryAttempt(skipAttemptRef, clientMessageId)
    } catch (error) {
      if (readSendRetryWithNewId(error)) {
        settleDeliveryAttempt(skipAttemptRef, clientMessageId)
      }
      console.error("Failed to send connect-apps skip response", error)
      toast.error(t("chatPage.clarification.sendError"))
    }
  }

  const renderField = (interaction: Interaction) => {
    const value = formState[interaction.field]

    switch (interaction.type) {
      case "text_input":
        return interaction.multiline ? (
          <Textarea
            placeholder={interaction.placeholder}
            value={value || ""}
            onChange={(e) => handleInputChange(interaction.field, e.target.value)}
          />
        ) : (
          <Input
            placeholder={interaction.placeholder}
            value={value || ""}
            onChange={(e) => handleInputChange(interaction.field, e.target.value)}
          />
        )

      case "number_input":
        return (
          <Input
            type="number"
            placeholder={interaction.placeholder}
            min={interaction.min}
            max={interaction.max}
            value={value || ""}
            onChange={(e) => handleInputChange(interaction.field, e.target.value)}
          />
        )

      case "select_one":
        return (
          <Select
            value={value}
            onValueChange={(v) => handleInputChange(interaction.field, v)}
            options={interaction.options || []}
            placeholder={t("chatPage.clarification.selectOption")}
          />
        )

      case "select_multiple":
        return (
          <MultiSelect
            values={value || []}
            onValuesChange={(v) => handleInputChange(interaction.field, v)}
            options={interaction.options || []}
            placeholder={interaction.placeholder || t("chatPage.clarification.selectOptions")}
          />
        )

      case "file_upload":
        if (filesDisabled) return null

        const fileValue = formState[interaction.field]
        const files: File[] = []
        if (fileValue instanceof FileList) {
          for (let i = 0; i < fileValue.length; i++) {
            files.push(fileValue[i])
          }
        } else if (Array.isArray(fileValue)) {
          files.push(...fileValue)
        }

        const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
          if (e.target.files && e.target.files.length > 0) {
            const newFiles = Array.from(e.target.files)
            // Always allow multiple files
            handleInputChange(interaction.field, [...files, ...newFiles])
          }
          // Reset input value to allow selecting same file again
          e.target.value = ''
        }

        const removeFile = (index: number) => {
          const newFiles = [...files]
          newFiles.splice(index, 1)
          handleInputChange(interaction.field, newFiles)
        }

        return (
          <div className="grid w-full gap-2">
            {files.length > 0 && (
              <div className="grid gap-2">
                {files.map((file, index) => (
                  <div key={index} className="flex items-center gap-2 rounded-md border p-2 text-sm bg-muted/50">
                    <FileIcon className="h-4 w-4 text-muted-foreground shrink-0" />
                    <span className="flex-1 truncate">{file.name}</span>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 shrink-0"
                      onClick={() => removeFile(index)}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            )}

            {/* Always show upload area to allow adding more files */}
            <div className="relative group cursor-pointer">
              <div className="flex h-24 w-full flex-col items-center justify-center gap-2 rounded-md border border-dashed bg-muted/30 hover:bg-muted/50 transition-colors">
                <Upload className="h-6 w-6 text-muted-foreground group-hover:scale-110 transition-transform" />
                <span className="text-xs text-muted-foreground">{t("chatPage.fileUpload.hintDragClick")}</span>
              </div>
              <Input
                type="file"
                className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
                accept={Array.isArray(interaction.accept) ? interaction.accept.join(",") : interaction.accept}
                multiple={interaction.multiple ?? true}
                onChange={handleFileChange}
              />
            </div>
            <div className="text-xs text-muted-foreground">
              {t("chatPage.clarification.acceptedFormats")}: {Array.isArray(interaction.accept) ? interaction.accept.join(", ") : interaction.accept || t("chatPage.clarification.any")}
            </div>
          </div>
        )

      case "confirm":
        return (
          <div className="flex items-center space-x-2">
            <Switch
              id={interaction.field}
              checked={!!value}
              onCheckedChange={(checked) => handleInputChange(interaction.field, checked)}
            />
            <Label htmlFor={interaction.field}>{t("chatPage.clarification.yes")}</Label>
          </div>
        )

      case "action_cards":
        const selectedOption = interaction.options?.find((opt) => opt.value === value)
        const visibleOptions = filesDisabled
          ? interaction.options?.filter((option) => !isFileActionOption(option))
          : interaction.options
        const selectedActionType = selectedOption?.action_type?.toLowerCase()
        const isUploadSelected = !filesDisabled && (
          selectedActionType === 'upload' || isFileActionValue(value)
        );
        const isWebsiteSelected = selectedActionType === 'input_url' || (typeof value === 'string' && (value.toLowerCase().includes('website') || value.toLowerCase().includes('url') || value.toLowerCase().includes('import')));

        return (
          <div className="flex flex-col gap-4 w-full">
            <div className="grid w-full grid-cols-1 sm:grid-cols-2 gap-4">
              {visibleOptions?.map((opt) => (
                <div
                  key={opt.value}
                  className={`flex flex-col items-center justify-center gap-2 rounded-lg border p-6 cursor-pointer transition-all ${value === opt.value ? 'border-primary bg-primary/5 shadow-sm ring-1 ring-primary' : 'bg-card hover:bg-muted/50 hover:border-muted-foreground/30'
                    }`}
                  onClick={() => handleInputChange(interaction.field, opt.value)}
                >
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
                    {opt.value.toLowerCase().includes('upload') || opt.value.toLowerCase().includes('file') ? (
                      <Upload className="h-5 w-5 text-primary" />
                    ) : opt.value.toLowerCase().includes('no') || opt.value.toLowerCase().includes('skip') ? (
                      <X className="h-5 w-5 text-primary" />
                    ) : (
                      <Globe className="h-5 w-5 text-primary" />
                    )}
                  </div>
                  <div className="flex flex-col items-center text-center gap-1">
                    <span className="font-medium text-sm text-foreground">{opt.label}</span>
                    {opt.description && <span className="text-xs text-muted-foreground">{opt.description}</span>}
                  </div>
                </div>
              ))}
            </div>
            {isUploadSelected && (
              <div className="mt-2 w-full animate-in fade-in slide-in-from-top-4">
                <div className="text-sm font-medium mb-2">{t("chatPage.fileUpload.hintDragClick")}</div>
                {renderField({ ...interaction, type: "file_upload", field: `${interaction.field}_files` })}
              </div>
            )}
            {isWebsiteSelected && (
              <div className="mt-2 w-full animate-in fade-in slide-in-from-top-4">
                <div className="text-sm font-medium mb-2">Enter website URL</div>
                {renderField({
                  ...interaction,
                  type: "text_input",
                  field: `${interaction.field}_url`,
                  placeholder: "https://...",
                  default: interaction.default_value
                })}
              </div>
            )}
          </div>
        )

      // Normally rendered via the isConnectAppsOnly branch below, which skips
      // renderField entirely - this case only matters if connect_apps is
      // ever mixed into a list with another interaction type (not producible
      // today, see LIVE_WIDGET_TYPES's comment), so that path still gets the
      // real widget instead of falling to the "unsupported type" case below.
      case "connect_apps":
        return <ConnectAppsField interaction={interaction} onSkip={handleSkipConnectApps} />

      default:
        return <div className="text-destructive text-sm">{t("chatPage.clarification.unsupportedType", { type: interaction.type })}</div>
    }
  }

  // Recomputed on every render, which is the point: a locale change re-renders
  // this component without remounting it.
  const sendFailureHintKey = sendFailure ? sendHintKey(sendFailure.disposition) : null
  const sendFailureMessage = sendFailure?.errorCode
    ? t(clientErrorTranslationKey(sendFailure.errorCode))
    : sendFailure?.detail || t("chatPage.clarification.sendError")

  return (
    <Collapsible
      open={isOpen}
      onOpenChange={setIsOpen}
      className="w-full space-y-2 rounded-lg border bg-card text-card-foreground shadow-sm my-2"
    >
      <CollapsibleTrigger asChild>
        <div className="flex items-center justify-between p-4 bg-muted/80 cursor-pointer hover:bg-muted/60 transition-colors">
          <div className="flex items-center gap-2 font-semibold">
            <MessageSquare className="h-4 w-4" />
            <span className="text-sm">
              {isConnectAppsOnly
                ? // Ignore the persisted interaction.label here (it's only ever a
                  // t()'d string frozen into the DB row at hire time - see
                  // hire-agent.ts's buildConnectAppsInteraction/HireMessageStrings)
                  // and re-resolve live instead, so a locale switch after hiring
                  // doesn't leave this header stuck in whatever language was
                  // active when the task was created.
                  t("chatPage.clarification.connectApps.title")
                : t("chatPage.clarification.title")}
            </span>
          </div>
          <div>
            {isOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          </div>
        </div>
      </CollapsibleTrigger>

      <CollapsibleContent className="space-y-4 p-4">
        {isConnectAppsOnly ? (
          <div className="space-y-4">
            {normalizedInteractions.map((interaction, index) => (
              <ConnectAppsField
                key={`${interaction.field}-${index}`}
                interaction={interaction}
                onSkip={handleSkipConnectApps}
              />
            ))}
          </div>
        ) : (
          <>
            <div className="space-y-4">
              {normalizedInteractions.map((interaction, index) => (
                filesDisabled && interaction.type === "file_upload" ? null : (
                <div key={`${interaction.field}-${index}`} className="space-y-2">
                  <Label className="text-sm font-medium">
                    {fieldLabel(interaction)}
                    {interaction.type === "confirm" || LIVE_WIDGET_TYPES.has(interaction.type) ? "" : ":"}
                  </Label>

                  {renderField(interaction)}
                </div>
                )
              ))}
            </div>

            {sendFailure && (
              <div role="alert" className="rounded-md border border-destructive/40 bg-destructive/5 p-3 text-sm text-destructive">
                <div>{sendFailureMessage}</div>
                {sendFailureHintKey && (
                  <div className="mt-1 text-xs text-destructive/80">{t(sendFailureHintKey)}</div>
                )}
              </div>
            )}

            <div className="pt-2 flex gap-2">
              <Button className="flex-1" size="sm" onClick={handleSubmit} disabled={!active || isSubmitting || isSubmitted}>
                {isSubmitting ? t("chatPage.clarification.submitting") : t("chatPage.clarification.submit")}
              </Button>
            </div>
          </>
        )}
      </CollapsibleContent>
    </Collapsible>
  )
}

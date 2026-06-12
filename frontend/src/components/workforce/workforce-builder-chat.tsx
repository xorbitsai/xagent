"use client"

import React, { useEffect, useMemo, useRef, useState } from "react"
import { AlertTriangle, Bot, CheckCircle2, Loader2, Sparkles, User } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Textarea } from "@/components/ui/textarea"
import { useI18n } from "@/contexts/i18n-context"
import { cn, formatDate } from "@/lib/utils"
import type { WorkforceBuilderMessage, WorkforceBuilderPatch } from "@/types/workforce"

interface WorkforceBuilderChatProps {
  messages: WorkforceBuilderMessage[]
  loading?: boolean
  submitting?: boolean
  readOnly?: boolean
  readOnlyReason?: string
  activeProposal?: WorkforceBuilderMessage | null
  applying?: boolean
  onSubmit: (message: string) => Promise<void> | void
  onApplyPatch?: (messageId: number, patch: WorkforceBuilderPatch) => Promise<void> | void
}

function roleIcon(role: string) {
  return role === "assistant" ? Bot : User
}

function formatOperationTitle(op: string) {
  return op
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ")
}

function operationTitle(op: string, t: (key: string) => string) {
  const key = `workforces.builder.operations.${op}`
  const value = t(key)
  return value === key ? formatOperationTitle(op) : value
}

function InlinePatchCard({
  patch,
  messageId,
  status,
  applying,
  readOnly,
  onApply,
  t,
}: {
  patch: WorkforceBuilderPatch
  messageId: number
  status?: string | null
  applying: boolean
  readOnly: boolean
  onApply: (messageId: number, patch: WorkforceBuilderPatch) => void
  t: (key: string) => string
}) {
  const alreadyApplied = status === "applied"
  const canApply =
    patch.operations.length > 0 && !applying && !alreadyApplied && !readOnly

  return (
    <div className="mt-3 space-y-3" onClick={(e) => e.stopPropagation()}>
      {/* Summary */}
      <div className="flex items-start gap-2.5 rounded-lg border bg-muted/30 p-3">
        <Sparkles className="mt-0.5 size-3.5 shrink-0 text-primary" />
        <div className="text-xs text-muted-foreground">{patch.summary}</div>
      </div>

      {/* Clarification */}
      {patch.clarification ? (
        <Alert className="p-3 text-xs">
          <AlertTriangle className="size-3.5" />
          <AlertTitle>{t("workforces.builder.clarificationNeeded")}</AlertTitle>
          <AlertDescription>{patch.clarification}</AlertDescription>
        </Alert>
      ) : null}

      {/* Warnings */}
      {patch.warnings.length > 0 ? (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-3">
          <div className="flex items-center gap-1.5 text-xs font-medium text-amber-800 mb-1.5">
            <AlertTriangle className="size-3.5" />
            {t("workforces.builder.warnings")}
          </div>
          <div className="space-y-1">
            {patch.warnings.map((warning, index) => (
              <div key={`${warning}-${index}`} className="text-xs text-amber-700">
                {warning}
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-1.5 text-xs text-emerald-600">
          <CheckCircle2 className="size-3.5" />
          {t("workforces.builder.noDestructiveWarning")}
        </div>
      )}

      {/* Operations */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium">{t("workforces.builder.operationsTitle")}</span>
          <Badge variant="secondary" className="text-[10px] h-4 px-1.5">
            {patch.operations.length}
          </Badge>
        </div>
        {patch.operations.map((operation, index) => (
          <div
            key={`${operation.op}-${index}`}
            className="rounded-lg border bg-background p-3"
          >
            <div className="flex items-center justify-between gap-2 mb-2">
              <span className="text-xs font-medium">
                {operationTitle(operation.op, t)}
              </span>
              <Badge variant="outline" className="text-[10px] h-4 px-1.5">
                #{index + 1}
              </Badge>
            </div>
            <pre className="overflow-x-auto whitespace-pre-wrap break-words text-[11px] leading-5 text-muted-foreground bg-muted/40 rounded-md p-2.5">
              {JSON.stringify(operation, null, 2)}
            </pre>
          </div>
        ))}
      </div>

      {/* Apply button */}
      <Button
        size="sm"
        className="w-full"
        variant={alreadyApplied ? "outline" : "default"}
        disabled={!canApply}
        onClick={(e) => {
          e.preventDefault()
          e.stopPropagation()
          onApply(messageId, patch)
        }}
      >
        {applying ? (
          <>
            <Loader2 className="size-3.5 animate-spin mr-1.5" />
            {t("workforces.loading.applyingChanges")}
          </>
        ) : alreadyApplied ? (
          t("workforces.actions.alreadyApplied")
        ) : readOnly ? (
          t("workforces.actions.readOnly")
        ) : (
          t("workforces.actions.applyChanges")
        )}
      </Button>
    </div>
  )
}

export function WorkforceBuilderChat({
  messages,
  loading = false,
  submitting = false,
  readOnly = false,
  readOnlyReason,
  activeProposal,
  applying = false,
  onSubmit,
  onApplyPatch,
}: WorkforceBuilderChatProps) {
  const { t } = useI18n()
  const [message, setMessage] = useState("")
  const containerRef = useRef<HTMLDivElement | null>(null)

  const canSubmit = useMemo(
    () => message.trim().length > 0 && !submitting && !readOnly,
    [message, readOnly, submitting],
  )

  useEffect(() => {
    const viewport = containerRef.current?.querySelector(
      "[data-radix-scroll-area-viewport]",
    ) as HTMLDivElement | null
    if (viewport) {
      viewport.scrollTop = viewport.scrollHeight
    }
  }, [messages, submitting])

  const handleSubmit = async () => {
    const value = message.trim()
    if (!value || submitting || readOnly) return
    setMessage("")
    await onSubmit(value)
  }

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b bg-background">
        <h3 className="font-semibold leading-none tracking-tight">{t("workforces.builder.chatTitle")}</h3>
        <p className="text-sm text-muted-foreground mt-1">{t("workforces.builder.chatDescription")}</p>
      </div>
      <div className="flex flex-1 flex-col min-h-0">
        <div ref={containerRef} className="flex-1 min-h-0">
          <ScrollArea className="h-full rounded-lg bg-muted/20">
            <div className="space-y-4 p-4">
              {loading ? (
                <div className="text-sm text-muted-foreground">
                  {t("workforces.loading.builderHistory")}
                </div>
              ) : messages.length === 0 ? (
                <div className="rounded-lg border border-dashed bg-background p-4 text-sm text-muted-foreground">
                  {t("workforces.builder.emptyPrompt")}
                </div>
              ) : (
                messages.map((item) => {
                  const Icon = roleIcon(item.role)
                  const isActiveProposal =
                    item.role === "assistant" &&
                    activeProposal?.id === item.id &&
                    item.proposed_patch

                  return (
                    <div
                      key={item.id}
                      className={cn(
                        "flex gap-3",
                        item.role === "user" ? "justify-end" : "justify-start",
                      )}
                    >
                      {item.role !== "user" ? (
                        <div className="mt-1 flex size-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
                          <Icon className="size-4" />
                        </div>
                      ) : null}
                      <div
                        className={cn(
                          "max-w-[85%] rounded-2xl border px-4 py-3",
                          item.role === "user"
                            ? "border-primary/20 bg-primary text-primary-foreground"
                            : "bg-background",
                        )}
                      >
                        <div
                          className={cn(
                            "mb-2 flex items-center gap-2 text-xs",
                            item.role === "user"
                              ? "text-primary-foreground/80"
                              : "text-muted-foreground",
                          )}
                        >
                          <span className="font-medium">
                            {item.role === "assistant"
                              ? t("workforces.builder.roleBuilder")
                              : t("workforces.builder.roleYou")}
                          </span>
                          {item.created_at ? <span>{formatDate(item.created_at)}</span> : null}
                        </div>
                        <div className="whitespace-pre-wrap text-sm leading-6">
                          {item.content}
                        </div>

                        {/* Inline patch card for the active proposal */}
                        {isActiveProposal && onApplyPatch ? (
                          <InlinePatchCard
                            patch={item.proposed_patch!}
                            messageId={item.id}
                            status={item.status}
                            applying={applying}
                            readOnly={readOnly}
                            onApply={onApplyPatch}
                            t={t}
                          />
                        ) : null}
                      </div>
                      {item.role === "user" ? (
                        <div className="mt-1 flex size-8 shrink-0 items-center justify-center rounded-full bg-secondary text-secondary-foreground">
                          <Icon className="size-4" />
                        </div>
                      ) : null}
                    </div>
                  )
                })
              )}
              {submitting ? (
                <div className="flex items-center gap-3 rounded-lg border bg-background px-4 py-3 text-sm text-muted-foreground">
                  <Loader2 className="size-4 animate-spin" />
                  <span>{t("workforces.builder.preparingPatch")}</span>
                </div>
              ) : null}
            </div>
          </ScrollArea>
        </div>

        <div className="space-y-3 shrink-0 p-4">
          <Textarea
            placeholder={t("workforces.builder.messagePlaceholder")}
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            rows={4}
            disabled={readOnly}
            onKeyDown={(event) => {
              if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
                event.preventDefault()
                void handleSubmit()
              }
            }}
          />
          <div className="flex items-center justify-between gap-3">
            <div className="text-xs text-muted-foreground">
              {readOnly && readOnlyReason ? readOnlyReason : t("workforces.builder.sendHint")}
            </div>
            <Button onClick={() => void handleSubmit()} disabled={!canSubmit}>
              {readOnly
                ? t("workforces.actions.readOnly")
                : submitting
                  ? t("workforces.loading.proposing")
                  : t("workforces.actions.proposeChanges")}
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

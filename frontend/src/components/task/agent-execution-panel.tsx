"use client"

import React from "react"
import { AlertCircle, Bot, Loader2, MessageSquare, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { MarkdownRenderer } from "@/components/ui/markdown-renderer"
import { TraceEventRenderer, type AgentExecutionSummary } from "@/components/chat/TraceEventRenderer"
import { useApp } from "@/contexts/app-context-chat"
import type { Translate } from "@/contexts/i18n-context"
import type { WorkforceAgentExecution, WorkforceAgentExecutionTraceEvent } from "@/types/workforce"
import { cn } from "@/lib/utils"

function normalizeTraceEventData(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {}
  return value as Record<string, unknown>
}

export function sanitizeAgentExecutionTraceEvents(
  value: unknown,
): WorkforceAgentExecutionTraceEvent[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((candidate) => {
    if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) {
      return []
    }
    const event = candidate as Record<string, unknown>
    return [{
      event_id: typeof event.event_id === "string" ? event.event_id : undefined,
      event_type: typeof event.event_type === "string" ? event.event_type : undefined,
      step_id: typeof event.step_id === "string" || event.step_id === null
        ? event.step_id
        : undefined,
      timestamp: typeof event.timestamp === "number" ||
        typeof event.timestamp === "string" ||
        event.timestamp === null
        ? event.timestamp
        : undefined,
      data: normalizeTraceEventData(event.data),
      parent_event_id: typeof event.parent_event_id === "string" ||
        event.parent_event_id === null
        ? event.parent_event_id
        : undefined,
    }]
  })
}

export function mergeAgentExecutionTraceEvents(
  historicalEvents: unknown,
  liveEvents: unknown,
  workerTaskId: string,
): WorkforceAgentExecutionTraceEvent[] {
  const events = sanitizeAgentExecutionTraceEvents(historicalEvents)
  const knownEventIds = new Set(events.map((event) => event.event_id).filter(Boolean))
  for (const event of sanitizeAgentExecutionTraceEvents(liveEvents)) {
    const eventData = event.data ?? {}
    if (
      eventData["source"] !== "xagent-agent-tool-child" ||
      String(eventData["worker_task_id"] ?? "") !== workerTaskId ||
      (event.event_id && knownEventIds.has(event.event_id))
    ) {
      continue
    }
    events.push({
      ...event,
      data: eventData,
    })
    if (event.event_id) knownEventIds.add(event.event_id)
  }
  return events
}

function formatAgentConclusion(value: unknown): string | null {
  if (typeof value === "string") return value.trim() || null
  if (!value || typeof value !== "object") return null
  try {
    return `\`\`\`json\n${JSON.stringify(value, null, 2)}\n\`\`\``
  } catch {
    return null
  }
}

export function getAgentExecutionConclusion(
  events: WorkforceAgentExecutionTraceEvent[],
): string | null {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]
    const data = event.data
    if (!data) continue

    if (event.event_type === "task_completion") {
      const result = data.result as Record<string, unknown> | string | undefined
      if (result && typeof result === "object") {
        const conclusion = formatAgentConclusion(
          result.chat_response ?? result.content ?? result.output ?? result.message,
        )
        if (conclusion) return conclusion
      }
      const conclusion = formatAgentConclusion(result ?? data.content ?? data.output)
      if (conclusion) return conclusion
    }

    if (event.event_type === "ai_message" || event.event_type === "agent_message") {
      const conclusion = formatAgentConclusion(data.content ?? data.message)
      if (conclusion) return conclusion
    }

    if (event.event_type === "react_task_end" || event.event_type === "task_end_react") {
      const result = data.result as Record<string, unknown> | string | undefined
      if (result && typeof result === "object") {
        const conclusion = formatAgentConclusion(
          result.output ?? result.content ?? result.message,
        )
        if (conclusion) return conclusion
      }
      const conclusion = formatAgentConclusion(result)
      if (conclusion) return conclusion
    }
  }
  return null
}

export function RunInspectorHeader({
  icon,
  title,
  subtitle,
  status,
  actions,
  onClose,
}: {
  icon: React.ReactNode
  title: React.ReactNode
  subtitle?: React.ReactNode
  status?: React.ReactNode
  actions?: React.ReactNode
  onClose: () => void
}) {
  return (
    <div className="flex h-14 shrink-0 items-center gap-3 border-b px-4">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
        {icon}
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex min-w-0 items-center gap-2">
          <div className="truncate text-sm font-semibold">{title}</div>
          {status}
        </div>
        {subtitle ? <div className="truncate text-xs text-muted-foreground">{subtitle}</div> : null}
      </div>
      {actions}
      <button
        type="button"
        onClick={onClose}
        className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  )
}

const WORKFORCE_STATUS_TRANSLATION_KEYS: Record<string, Parameters<Translate>[0]> = {
  completed: "workforces.status.completed",
  running: "workforces.status.running",
  failed: "workforces.status.failed",
  pending: "workforces.status.pending",
  paused: "workforces.status.paused",
  waiting_for_user: "workforces.status.waitingForUser",
  interrupted: "workforces.status.interrupted",
}

function runStatusLabel(status: string, t: Translate): string {
  const normalized = status.toLowerCase()
  const translationKey = WORKFORCE_STATUS_TRANSLATION_KEYS[normalized]
  return translationKey ? t(translationKey) : status
}

export function AgentExecutionPanel({
  selection,
  detail,
  events,
  status,
  loading,
  error,
  onClose,
  onRetry,
  t,
}: {
  selection: AgentExecutionSummary | null
  detail: WorkforceAgentExecution | null
  events: WorkforceAgentExecutionTraceEvent[]
  status?: string
  loading: boolean
  error: string | null
  onClose: () => void
  onRetry: () => void
  t: Translate
}) {
  const agentName = detail?.worker_alias || detail?.agent_name || selection?.agentName || t("traceEventRenderer.unknownWorker")
  const { openFilePreview } = useApp()
  const conclusion = React.useMemo(() => getAgentExecutionConclusion(events), [events])

  return (
    <div className="flex h-full min-h-0 flex-col bg-background">
      <RunInspectorHeader
        icon={<Bot className="h-4 w-4" />}
        title={agentName}
        subtitle={<span className="font-mono">{selection?.workerTaskId}</span>}
        status={status ? (
          <span className={cn(
            "rounded-full px-2 py-0.5 text-[11px] font-medium",
            status === "completed" && "bg-emerald-100 text-emerald-700",
            status === "running" && "bg-blue-100 text-blue-700",
            status === "failed" && "bg-red-100 text-red-700",
            status === "interrupted" && "bg-amber-100 text-amber-700",
          )}>
            {runStatusLabel(status, t)}
          </span>
        ) : null}
        onClose={onClose}
      />

      <div className="min-h-0 flex-1 overflow-y-auto bg-background px-6 py-5">
        {conclusion ? (
          <section className="mb-5 rounded-xl border bg-muted/20 p-4">
            <div className="mb-3 flex items-center gap-2 text-sm font-semibold">
              <MessageSquare className="h-4 w-4 text-primary" />
              {t("workforces.run.agentExecutionResult")}
            </div>
            <MarkdownRenderer
              content={conclusion}
              className="prose-sm leading-relaxed"
              onFileClick={openFilePreview}
            />
          </section>
        ) : null}
        {loading && events.length === 0 ? (
          <div className="flex h-full min-h-64 flex-col items-center justify-center gap-3 text-sm text-muted-foreground">
            <Loader2 className="h-6 w-6 animate-spin text-primary" />
            {t("workforces.run.loadingAgentExecution")}
          </div>
        ) : error ? (
          <div className="flex h-full min-h-64 flex-col items-center justify-center gap-3 text-center">
            <AlertCircle className="h-7 w-7 text-destructive" />
            <p className="max-w-sm text-sm text-muted-foreground">{error}</p>
            <Button type="button" variant="outline" size="sm" onClick={onRetry}>
              {t("workforces.run.retryAgentExecution")}
            </Button>
          </div>
        ) : events.length > 0 ? (
          <div className="mx-auto max-w-3xl">
            <TraceEventRenderer
              events={events}
              taskStatus={status}
              defaultExpandSteps
            />
          </div>
        ) : (
          <div className="flex h-full min-h-64 items-center justify-center text-sm text-muted-foreground">
            {t("workforces.run.emptyAgentExecution")}
          </div>
        )}
      </div>
    </div>
  )
}

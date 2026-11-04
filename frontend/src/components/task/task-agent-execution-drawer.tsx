"use client"

import React, { useEffect, useState } from "react"
import type { AgentExecutionSummary } from "@/components/chat/TraceEventRenderer"
import { AgentExecutionPanel, sanitizeAgentExecutionTraceEvents } from "@/components/task/agent-execution-panel"
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet"
import { useI18n } from "@/contexts/i18n-context"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import type { WorkforceAgentExecution } from "@/types/workforce"

export function TaskAgentExecutionDrawer({ taskId, selection, onClose }: {
  taskId: number
  selection: AgentExecutionSummary
  onClose: () => void
}) {
  const { t } = useI18n()
  const [detail, setDetail] = useState<WorkforceAgentExecution | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [retry, setRetry] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined
    setDetail(null)
    setLoading(true)
    setError(null)
    const load = async () => {
      try {
        const response = await apiRequest(
          `${getApiUrl()}/api/chat/task/${taskId}/agent-executions/${encodeURIComponent(selection.workerTaskId)}`,
          { signal: controller.signal },
        )
        if (!response.ok) throw new Error(t("workforces.run.agentExecutionLoadError"))
        const result: WorkforceAgentExecution = await response.json()
        if (controller.signal.aborted) return
        setDetail(result)
        setError(null)
        // Child facts are intentionally absent from the root stream. Refresh
        // the authorized scope while its inspector is open and still active.
        if (!["completed", "failed", "interrupted"].includes(result.status)) {
          timer = setTimeout(() => void load(), 5000)
        }
      } catch (cause) {
        if (!controller.signal.aborted) {
          setError(cause instanceof Error ? cause.message : t("workforces.run.agentExecutionLoadError"))
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false)
      }
    }
    void load()
    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [taskId, selection.workerTaskId, retry, t])

  return (
    <Sheet open onOpenChange={(open) => { if (!open) onClose() }}>
      <SheetContent className="w-full gap-0 p-0 sm:max-w-2xl [&>button]:hidden" aria-describedby={undefined}>
        <SheetTitle className="sr-only">{selection.agentName}</SheetTitle>
        <AgentExecutionPanel
          selection={selection}
          detail={detail}
          events={sanitizeAgentExecutionTraceEvents(detail?.trace_events)}
          status={detail?.status || selection.status}
          loading={loading}
          error={error}
          onClose={onClose}
          onRetry={() => setRetry((value) => value + 1)}
          t={t}
        />
      </SheetContent>
    </Sheet>
  )
}

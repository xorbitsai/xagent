"use client"

import React, { useCallback, useEffect, useState } from "react"
import { History, Loader2, RefreshCw } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { listWorkforceRuns } from "@/lib/workforces-api"
import { normalizeTaskStatus } from "@/lib/task-status"
import type { WorkforceRunHistoryItem, WorkforceRunHistoryResponse } from "@/types/workforce"
import { formatTime } from "@/lib/time-utils"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

const RUN_STATUS_CLASSES: Record<string, string> = {
  completed: "bg-green-100 text-green-700",
  running: "bg-blue-100 text-blue-700",
  pending: "bg-amber-100 text-amber-700",
  paused: "bg-amber-100 text-amber-700",
  waiting_for_user: "bg-amber-100 text-amber-700",
  failed: "bg-red-100 text-red-700",
}

function runDuration(run: WorkforceRunHistoryItem): string | null {
  if (!run.created_at || !run.completed_at) return null
  const start = new Date(run.created_at).getTime()
  const end = new Date(run.completed_at).getTime()
  if (Number.isNaN(start) || Number.isNaN(end)) return null
  const seconds = Math.round((end - start) / 1000)
  if (seconds < 0) return null
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ${seconds % 60}s`
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`
}

interface WorkforceRunsListProps {
  workforceId: number | string
  onSelectRun?: (run: WorkforceRunHistoryItem) => void
  className?: string
}

export function WorkforceRunsList({
  workforceId,
  onSelectRun,
  className,
}: WorkforceRunsListProps) {
  const { t } = useI18n()
  const size = 20

  const [data, setData] = useState<WorkforceRunHistoryResponse | null>(null)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(
    async (nextPage: number, options: { silent?: boolean } = {}) => {
      try {
        if (!options.silent) setLoading(true)
        setError(null)
        const response = await listWorkforceRuns(workforceId, { page: nextPage, size })
        setData(response)
        setPage(response.page)
      } catch (err) {
        setError(err instanceof Error ? err.message : t("workforces.runs.loadError"))
      } finally {
        if (!options.silent) setLoading(false)
      }
    },
    [workforceId, size, t],
  )

  useEffect(() => {
    // New workforce/size: drop the previous list so stale rows never flash.
    // Pagination keeps the current list visible while the next page loads.
    setData(null)
    void load(1)
  }, [load])

  if (loading && !data) {
    return (
      <div className={cn("flex items-center justify-center gap-2 py-10 text-sm text-muted-foreground", className)}>
        <Loader2 className="h-4 w-4 animate-spin" />
        {t("workforces.runs.loading")}
      </div>
    )
  }

  if (error && !data) {
    return (
      <div className={cn("flex flex-col items-center gap-3 py-10 text-sm text-red-500", className)}>
        <span>{error}</span>
        <Button variant="outline" size="sm" onClick={() => void load(page)}>
          {t("workforces.runs.retry")}
        </Button>
      </div>
    )
  }

  const items = data?.items ?? []

  if (items.length === 0) {
    return (
      <div className={cn("flex flex-col items-center justify-center gap-2 py-10 text-center", className)}>
        <History className="h-8 w-8 text-muted-foreground/50" />
        <p className="text-sm font-medium text-foreground">{t("workforces.runs.empty")}</p>
        <p className="max-w-xs text-xs text-muted-foreground">{t("workforces.runs.emptyHint")}</p>
      </div>
    )
  }

  return (
    <div className={cn("flex flex-col", className)}>
      <div className="flex items-center justify-between pb-3">
        <span className="text-xs text-muted-foreground">
          {t("workforces.pagination.showing", {
            start: (page - 1) * size + 1,
            end: (page - 1) * size + items.length,
            total: data?.total ?? items.length,
          })}
        </span>
        <Button
          variant="ghost"
          size="sm"
          className="gap-1.5 text-muted-foreground"
          onClick={() => void load(page, { silent: true })}
        >
          <RefreshCw className="h-3.5 w-3.5" />
          {t("workforces.runs.refresh")}
        </Button>
      </div>

      <ul className="flex flex-col gap-1.5">
        {items.map((run) => {
          const title =
            run.task_title || run.message || t("workforces.runs.untitled", { id: run.id })
          const duration = runDuration(run)
          const clickable = Boolean(onSelectRun && run.task_id)
          const statusKey = normalizeTaskStatus(run.status)
          return (
            <li key={run.id}>
              <button
                type="button"
                disabled={!clickable}
                onClick={() => {
                  if (clickable && onSelectRun) onSelectRun(run)
                }}
                className={cn(
                  "flex w-full flex-col gap-1 rounded-lg border bg-card px-4 py-3 text-left transition-colors",
                  clickable
                    ? "hover:border-primary/40 hover:bg-muted/50 cursor-pointer"
                    : "opacity-70 cursor-default",
                )}
              >
                <div className="flex items-center gap-2">
                  <span className="truncate text-sm font-medium">
                    {title}
                  </span>
                  {run.is_preview && (
                    <span className="shrink-0 rounded-full bg-muted px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
                      {t("workforces.runs.previewBadge")}
                    </span>
                  )}
                  <span
                    className={cn(
                      "ml-auto shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold",
                      RUN_STATUS_CLASSES[statusKey ?? run.status] ??
                        "bg-muted text-muted-foreground",
                    )}
                  >
                    {statusKey ? t(`workforces.runs.status.${statusKey}`) : run.status}
                  </span>
                </div>
                <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
                  {run.created_at && <span>{formatTime(run.created_at, "datetime")}</span>}
                  {duration && (
                    <>
                      <span>·</span>
                      <span>{duration}</span>
                    </>
                  )}
                  {!run.task_id && (
                    <>
                      <span>·</span>
                      <span>{t("workforces.runs.taskDeleted")}</span>
                    </>
                  )}
                </div>
              </button>
            </li>
          )
        })}
      </ul>

      {(data?.pages ?? 1) > 1 && (
        <div className="flex items-center justify-between pt-3">
          <Button
            variant="outline"
            size="sm"
            disabled={page <= 1 || loading}
            onClick={() => void load(page - 1)}
          >
            {t("workforces.pagination.prev")}
          </Button>
          <span className="text-xs text-muted-foreground">
            {t("workforces.pagination.page", { page, pages: data?.pages ?? 1 })}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={page >= (data?.pages ?? 1) || loading}
            onClick={() => void load(page + 1)}
          >
            {t("workforces.pagination.next")}
          </Button>
        </div>
      )}
    </div>
  )
}

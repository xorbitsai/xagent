"use client"

import { useEffect, useMemo, useState } from "react"
import { CheckCircle2, ChevronLeft, Clock, Loader2, RotateCcw, XCircle } from "lucide-react"
import { cn } from "@/lib/utils"
import { normalizeTimestampMs } from "@/lib/time-utils"
import { useI18n } from "@/contexts/i18n-context"

export interface ProgressStepView {
  id: string
  title: string
  description?: string
  status: "pending" | "running" | "completed" | "failed" | "skipped"
  startedAt?: string | number
  completedAt?: string | number
}

interface ProgressPanelProps {
  steps: ProgressStepView[]
  // true when the total step count is known upfront (DAG plans always know
  // it; a future dynamic ReAct panel would not, hence this flag).
  totalKnown: boolean
  startedAt: string | number | undefined
  // Set once the run has actually finished (completed/failed). While this is
  // undefined, the total elapsed keeps ticking; once set, it freezes at
  // endedAt - startedAt instead of continuing to count up against "now".
  endedAt?: string | number
  isPlanning?: boolean
  onCollapse: () => void
  onStepClick?: (stepId: string) => void
}

function formatElapsedCompact(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000))
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = totalSeconds % 60
  if (hours > 0) return `${hours}h ${String(minutes).padStart(2, "0")}m`
  if (minutes > 0) return `${minutes}m ${String(seconds).padStart(2, "0")}s`
  return `${seconds}s`
}

// Ticks once a second while `startedAt` is set, so callers get a live
// "elapsed since startedAt" string without each one re-implementing a timer.
// Checked for truthiness, not just `!== undefined`: upstream data can hand
// this an empty string for "no timestamp yet" (see stepsFromPlanData's
// `getString` fallback), and normalizeTimestampMs treats any falsy value as
// "now" - so a strict `!== undefined` check would let "" through and render
// a bogus "0s" instead of hiding the timer.
function useElapsed(startedAt: string | number | undefined): string | null {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    if (!startedAt) return
    // Re-seed immediately: `now` may still hold a value from long before this
    // particular `startedAt` became truthy (e.g. a step that sat pending for
    // minutes before going "running"), and the first tick is a full second
    // away - without this the first render briefly shows an elapsed time
    // computed against a stale `now`.
    setNow(Date.now())
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [startedAt])

  if (!startedAt) return null
  return formatElapsedCompact(now - normalizeTimestampMs(startedAt))
}

export function ProgressPanel({
  steps,
  totalKnown,
  startedAt,
  endedAt,
  isPlanning = false,
  onCollapse,
  onStepClick,
}: ProgressPanelProps) {
  const { t } = useI18n()
  const liveTotalElapsed = useElapsed(endedAt ? undefined : startedAt)
  const finishedTotalElapsed =
    endedAt && startedAt
      ? formatElapsedCompact(normalizeTimestampMs(endedAt) - normalizeTimestampMs(startedAt))
      : null
  const totalElapsed = liveTotalElapsed ?? finishedTotalElapsed
  // "Resolved" rather than strictly "succeeded" - a plan with conditional
  // branches can have steps that finish as "skipped" and will never become
  // "completed", so counting only completed would leave the header stuck
  // below the total (e.g. "4/6") even once the whole plan is done.
  const resolvedCount = useMemo(
    () => steps.filter((step) => (
      step.status === "completed" || step.status === "skipped" || step.status === "failed"
    )).length,
    [steps],
  )

  return (
    <div className="flex h-full flex-col bg-background/80">
      <div className="flex items-center justify-between border-b border-border bg-card/50 px-4 py-3 backdrop-blur-sm">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-foreground">
            {t("chatPage.progressPanel.title")}
          </h2>
          {totalKnown && (
            <span className="text-xs text-muted-foreground">
              {resolvedCount}/{steps.length}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={onCollapse}
            className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            aria-label={t("chatPage.progressPanel.collapse")}
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
        </div>
      </div>

      {totalElapsed && (
        <div className="flex items-center justify-between px-4 py-2 text-xs text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <Clock className="h-3.5 w-3.5" />
            {t("chatPage.progressPanel.elapsed")}
          </span>
          <span className="font-mono tabular-nums">{totalElapsed}</span>
        </div>
      )}

      <div className="flex-1 overflow-y-auto px-2 py-3">
        {steps.length === 0 && isPlanning ? (
          <div className="flex items-center gap-2 px-2 py-4 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            {t("chatPage.progressPanel.planning")}
          </div>
        ) : (
          <ol className="space-y-1">
            {steps.map((step, index) => (
              <ProgressStepRow key={step.id} step={step} index={index} onClick={onStepClick} />
            ))}
          </ol>
        )}
      </div>
    </div>
  )
}

function ProgressStepRow({
  step,
  index,
  onClick,
}: {
  step: ProgressStepView
  index: number
  onClick?: (stepId: string) => void
}) {
  const liveElapsed = useElapsed(step.status === "running" ? step.startedAt : undefined)
  // Completed/failed steps already have both endpoints - a static duration,
  // not another ticking timer, is all that's needed once a step is done.
  const finishedDuration =
    (step.status === "completed" || step.status === "failed")
    && step.startedAt
    && step.completedAt
      ? formatElapsedCompact(normalizeTimestampMs(step.completedAt) - normalizeTimestampMs(step.startedAt))
      : null
  const duration = liveElapsed ?? finishedDuration

  return (
    <li>
      <button
        type="button"
        onClick={() => onClick?.(step.id)}
        className={cn(
          "flex w-full items-start gap-3 rounded-lg px-2 py-2 text-left transition-colors",
          step.status === "running" ? "bg-primary/10" : "hover:bg-muted/50",
        )}
      >
        <span
          className={cn(
            "mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-medium",
            step.status === "completed" && "bg-green-500/10 text-green-600",
            step.status === "running" && "bg-primary text-primary-foreground",
            step.status === "failed" && "bg-red-500/10 text-red-600",
            (step.status === "pending" || step.status === "skipped") && "bg-muted text-muted-foreground",
          )}
        >
          {step.status === "completed" ? (
            <CheckCircle2 className="h-3.5 w-3.5" />
          ) : step.status === "failed" ? (
            <XCircle className="h-3.5 w-3.5" />
          ) : step.status === "running" ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : step.status === "skipped" ? (
            <RotateCcw className="h-3 w-3" />
          ) : (
            index + 1
          )}
        </span>
        <span className="min-w-0 flex-1">
          <span
            className={cn(
              "block truncate text-sm",
              step.status === "running" && "font-medium text-primary",
              (step.status === "completed" || step.status === "pending" || step.status === "skipped")
                && "text-muted-foreground",
              step.status === "failed" && "text-foreground",
            )}
          >
            {step.title}
          </span>
          {step.description && (
            <span className="mt-0.5 line-clamp-2 block text-xs text-muted-foreground/80">
              {step.description}
            </span>
          )}
        </span>
        {duration && (
          <span
            className={cn(
              "mt-0.5 shrink-0 text-xs tabular-nums",
              step.status === "running" ? "text-primary/80" : "text-muted-foreground",
            )}
          >
            {duration}
          </span>
        )}
      </button>
    </li>
  )
}

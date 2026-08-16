"use client"

import { useEffect, useState } from "react"
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
  startedAt: string | number | undefined
  // Set once the run has actually finished (completed/failed). While this is
  // undefined, the total elapsed keeps ticking; once set, it freezes at
  // endedAt - startedAt instead of continuing to count up against "now".
  endedAt?: string | number
  onCollapse: () => void
  onStepClick: (stepId: string) => void
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

// Distinguishes a genuinely missing timestamp (undefined/null, or the empty
// string upstream data uses for "not set yet" - see stepsFromPlanData's
// `getString` fallback) from a valid-but-falsy one: epoch 0 is a real instant
// in this type's `string | number` contract, and normalizeTimestampMs itself
// falls back to "now" for any falsy input, so a plain truthiness check would
// treat a genuine (if unlikely) zero timestamp as absent.
function hasTimestamp(value: string | number | undefined): value is string | number {
  return value !== undefined && value !== null && value !== ""
}

// Ticks once a second for as long as `active` is true, shared by the header
// and every running step row - a run with R running rows previously mounted
// R+1 independent one-second timers (one per `useElapsed` call) all doing
// the same job; one shared clock at the panel level does it once.
function useLiveNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    if (!active) return
    // Re-seed immediately rather than waiting for the first tick (a full
    // second away): `now` may still hold a stale value from before the panel
    // had anything to tick for.
    setNow(Date.now())
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [active])

  return now
}

export function ProgressPanel({
  steps,
  startedAt,
  endedAt,
  onCollapse,
  onStepClick,
}: ProgressPanelProps) {
  const { t } = useI18n()
  const runEnded = hasTimestamp(endedAt)
  const headerIsLive = !runEnded && hasTimestamp(startedAt)
  // The shared clock must stay active for as long as EITHER the header or any
  // individual row needs to tick - not just whenever the panel's own
  // `startedAt` is set. A row's own `step.startedAt` is what actually gates
  // its ticking (see ProgressStepRow below), and the two aren't guaranteed to
  // agree: a dagExecution constructed via a legacy/malformed event that never
  // got a `created_at` backfill would leave the panel's `startedAt` (and thus
  // `headerIsLive`) false even while a step with its own valid `startedAt` is
  // genuinely running - gating the whole clock on `headerIsLive` alone would
  // silently freeze that row's timer instead of just hiding the header's.
  const anyRowLive = !runEnded && steps.some((step) => step.status === "running" && hasTimestamp(step.startedAt))
  const now = useLiveNow(headerIsLive || anyRowLive)
  const liveTotalElapsed = headerIsLive
    ? formatElapsedCompact(now - normalizeTimestampMs(startedAt))
    : null
  const finishedTotalElapsed =
    runEnded && hasTimestamp(startedAt)
      ? formatElapsedCompact(normalizeTimestampMs(endedAt) - normalizeTimestampMs(startedAt))
      : null
  const totalElapsed = liveTotalElapsed ?? finishedTotalElapsed
  // "Resolved" rather than strictly "succeeded" - a plan with conditional
  // branches can have steps that finish as "skipped" and will never become
  // "completed", so counting only completed would leave the header stuck
  // below the total (e.g. "4/6") even once the whole plan is done. "failed"
  // is counted for the same reason: a failed step is also terminal (it will
  // never transition to "completed"), so excluding it would leave the same
  // "4/6" stuck header on a plan that has actually finished running.
  const resolvedCount = steps.filter((step) => (
    step.status === "completed" || step.status === "skipped" || step.status === "failed"
  )).length

  return (
    <div className="flex h-full flex-col bg-background/80">
      <div className="flex items-center justify-between border-b border-border bg-card/50 px-4 py-3 backdrop-blur-sm">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-foreground">
            {t("chatPage.progressPanel.title")}
          </h2>
          {steps.length > 0 && (
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
        {steps.length === 0 ? (
          // A run can end (a plan-time failure, for example) before any step
          // ever started, leaving steps permanently empty - `endedAt` (only
          // set once the run has actually finished) is what distinguishes
          // that from "still planning/about to execute", so this doesn't show
          // an indefinitely-spinning "generating plan" placeholder for a run
          // that's already over.
          !runEnded && (
            <div className="flex items-center gap-2 px-2 py-4 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              {t("chatPage.progressPanel.planning")}
            </div>
          )
        ) : (
          <ol className="space-y-1">
            {steps.map((step, index) => (
              <ProgressStepRow
                key={step.id}
                step={step}
                index={index}
                endedAt={endedAt}
                now={now}
                onClick={onStepClick}
              />
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
  endedAt,
  now,
  onClick,
}: {
  step: ProgressStepView
  index: number
  endedAt?: string | number
  now: number
  onClick: (stepId: string) => void
}) {
  const { t } = useI18n()
  const liveElapsed =
    !hasTimestamp(endedAt) && step.status === "running" && hasTimestamp(step.startedAt)
      ? formatElapsedCompact(now - normalizeTimestampMs(step.startedAt))
      : null
  // Completed/failed steps already have both endpoints - a static duration,
  // not another ticking timer, is all that's needed once a step is done. Once
  // the whole run has ended (`endedAt` set), no row should keep ticking
  // either: the backend cancels pending sibling steps without emitting their
  // own terminal event when a run finishes, so a step that's still "running"
  // at that point never gets its own completedAt - freeze it at the run's end
  // time instead, rather than letting it keep counting up against "now".
  const frozenEndpoint = hasTimestamp(step.completedAt)
    ? step.completedAt
    : (step.status === "running" ? endedAt : undefined)
  const finishedDuration =
    hasTimestamp(step.startedAt) && hasTimestamp(frozenEndpoint)
      ? formatElapsedCompact(normalizeTimestampMs(frozenEndpoint) - normalizeTimestampMs(step.startedAt))
      : null
  const duration = liveElapsed ?? finishedDuration
  // TraceEventRenderer only tags a step's trace group with data-step-id once
  // it's actually started (a ProcessedStep) - a step with no startedAt has no
  // corresponding group in the chat log, so clicking it would silently do
  // nothing. Keyed off startedAt itself (the actual precondition) rather than
  // re-deriving it from status: a step forced to "skipped" by branch-
  // activation logic can still have a startedAt preserved from before that
  // (see stepsFromPlanData's existingStep merge), and its trace group is
  // still there to scroll to even though its final status isn't "running".
  const hasTraceTarget = hasTimestamp(step.startedAt)

  return (
    <li>
      <button
        type="button"
        disabled={!hasTraceTarget}
        onClick={hasTraceTarget ? () => onClick(step.id) : undefined}
        className={cn(
          "flex w-full items-start gap-3 rounded-lg px-2 py-2 text-left transition-colors",
          step.status === "running" ? "bg-primary/10" : "hover:bg-muted/50",
          !hasTraceTarget && "cursor-default hover:bg-transparent",
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
            <span className="sr-only">{t(`agent.layout.status.${step.status}`)}: </span>
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

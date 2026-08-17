"use client"

import React, { Suspense, useCallback, useEffect, useRef, useState } from "react"
import { ArrowLeft, Loader2, PanelRight } from "lucide-react"
import { useParams, useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { TaskConversationPanel } from "@/components/task/task-conversation-panel"
import { ProgressPanel, type ProgressStepView } from "@/components/task/progress-panel"
import { isTerminalTaskStatus, useApp } from "@/contexts/app-context-chat"
import { useI18n } from "@/contexts/i18n-context"
import { cn, getApiUrl } from "@/lib/utils"

function TaskDetailContent() {
  const { state, setTaskId, closeFilePreview } = useApp()
  const { t } = useI18n()
  const params = useParams()
  const router = useRouter()
  const taskIdFromUrl = params.id
  const [progressPanelOpen, setProgressPanelOpen] = useState(false)
  const dismissedProgressRunKeyRef = useRef<string | null>(null)

  useEffect(() => {
    if (taskIdFromUrl && typeof taskIdFromUrl === "string") {
      const taskIdNum = parseInt(taskIdFromUrl, 10)
      if (!Number.isNaN(taskIdNum) && taskIdNum !== state.taskId) {
        setTaskId(taskIdNum)
      }
    }
  }, [taskIdFromUrl, setTaskId, state.taskId])

  useEffect(() => {
    return () => {
      closeFilePreview()
    }
  }, [closeFilePreview])

  // Prefer turn_id (a stable per-turn identity - see DAGExecution's own
  // comment) over created_at for identifying "this run" - created_at can get
  // rebuilt with a fresh value on a replan re-arrival while still being the
  // SAME run, which would incorrectly reopen a panel the user just dismissed
  // and restart its elapsed timer. Falls back to created_at only when
  // turn_id is absent (an older backend, or some other pattern that doesn't
  // set it).
  const progressRunKey = state.dagExecution?.turn_id ?? state.dagExecution?.created_at

  const handleCollapseProgressPanel = useCallback(() => {
    dismissedProgressRunKeyRef.current = progressRunKey !== undefined && progressRunKey !== null ? String(progressRunKey) : null
    setProgressPanelOpen(false)
  }, [progressRunKey])

  const handleProgressStepClick = useCallback((stepId: string) => {
    window.dispatchEvent(new CustomEvent("scrollToTraceStep", { detail: { stepId } }))
  }, [])

  // Auto-open the panel the moment this turn enters DAG/plan_execute mode -
  // the plan's step list is known as soon as planning starts, so there is no
  // reason to wait for the user to click the header toggle. A manual
  // collapse only suppresses re-opening for *this* run (tracked by
  // progressRunKey); the next DAG run gets a fresh key and opens again.
  useEffect(() => {
    if (progressRunKey === undefined || progressRunKey === null) return
    const key = String(progressRunKey)
    if (dismissedProgressRunKeyRef.current === key) return
    setProgressPanelOpen(true)
  }, [progressRunKey])

  // Scrolls the chat's trace log to a step when a Progress panel row is
  // clicked (TraceEventRenderer.tsx tags each DAG step group with
  // `data-step-id`).
  useEffect(() => {
    const handleScrollToTraceStep = (event: Event) => {
      const stepId = (event as CustomEvent<{ stepId?: string }>).detail?.stepId
      if (!stepId) return
      // Step ids are LLM-generated plan identifiers with no schema constraint
      // on their content (see plan_generator.py's tool schema), so they can
      // legitimately contain characters that aren't safe to interpolate into
      // a CSS attribute selector (quotes, backslashes, control characters
      // including NUL). Matching by actual attribute value instead of
      // building a selector string sidesteps escaping entirely - there's
      // nothing here for a pathological id to break. A replan reuses the same
      // step ids, and a multi-turn task's history can render more than one
      // trace group tagged with the same data-step-id - take the LAST match
      // (the current/most recent one) rather than the first.
      const matches = Array.from(document.querySelectorAll("[data-step-id]")).filter(
        (el) => el.getAttribute("data-step-id") === stepId
      )
      matches[matches.length - 1]?.scrollIntoView({ behavior: "smooth", block: "center" })
    }
    window.addEventListener("scrollToTraceStep", handleScrollToTraceStep as EventListener)
    return () => window.removeEventListener("scrollToTraceStep", handleScrollToTraceStep as EventListener)
  }, [])

  // Between `md` and `xl` the panel renders as a dismissible overlay (see the
  // render below) rather than a static column - Escape should close it the
  // same way clicking the backdrop does. Harmless above `xl`, where the panel
  // is a static sibling and there's no backdrop to match: closing it there
  // via Escape is just the same action the header toggle already offers.
  useEffect(() => {
    if (!progressPanelOpen) return
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") handleCollapseProgressPanel()
    }
    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [progressPanelOpen, handleCollapseProgressPanel])

  const progressSteps: ProgressStepView[] = state.steps.map((step) => ({
    id: step.id,
    title: step.name,
    description: step.description,
    status: step.status,
    startedAt: step.started_at,
    completedAt: step.completed_at,
  }))
  const progressPanelVisible = progressPanelOpen && Boolean(state.dagExecution)
  // Deliberately keyed off the TASK's own status, not dagExecution.phase:
  // a single step failing (dag_step_failed) sets the whole dagExecution.phase
  // to "failed" even when the DAG goes on to replan and keep running, and the
  // raw backend dag_execution payload never carries phase "completed"/"failed"
  // at all (only planning/replanning/executing/completion_assessment) nor an
  // updated_at - so freezing off dagExecution would freeze early on a
  // recoverable step failure, or never freeze, or un-freeze on a later
  // in-flight dag_execution event. The task's own status is the one signal
  // that's actually authoritative for "this run is truly over."
  const isDagFinished = isTerminalTaskStatus(state.currentTask?.status)
  // dagTerminatedAt, not updatedAt: the latter is general task metadata that
  // keeps changing after this run ends for unrelated reasons (a title edit,
  // another field update via a later task_info refresh), which would make
  // the frozen elapsed time silently drift on revisit. dagTerminatedAt is
  // stamped once by UPDATE_TASK_STATUS specifically for this and preserved
  // across those unrelated refreshes.
  const progressEndedAt = isDagFinished ? state.currentTask?.dagTerminatedAt : undefined

  return (
    <div className="h-full flex flex-col md:flex-row bg-background">
      <div className="flex-1 min-w-0 min-h-0 flex flex-col">
        {(state.currentTask?.agentId || state.dagExecution) && (
          <div className="flex-none flex items-center justify-between gap-3 px-4 py-3 bg-background/95 backdrop-blur z-50 sticky top-0">
            <div className="flex items-center gap-3 min-w-0">
              {state.currentTask?.agentId && (
                <Button
                  variant="ghost"
                  size="icon"
                  className="rounded-full hover:bg-muted flex-shrink-0"
                  onClick={() => {
                    const agentId = state.currentTask?.agentId
                    router.push(agentId ? `/agent/${agentId}` : "/task")
                  }}
                  title={t("common.back")}
                >
                  <ArrowLeft className="w-5 h-5" />
                </Button>
              )}
              {(state.currentTask?.agentName || state.currentTask?.agentLogoUrl) && (
                <div className="flex items-center gap-3 overflow-hidden">
                  {state.currentTask?.agentLogoUrl ? (
                    <img
                      src={state.currentTask.agentLogoUrl.startsWith('http') ? state.currentTask.agentLogoUrl : `${getApiUrl()}${state.currentTask.agentLogoUrl}`}
                      alt={state.currentTask.agentName || t("agent.logo")}
                      className="w-8 h-8 object-cover rounded-sm flex-shrink-0"
                    />
                  ) : null}
                  {state.currentTask?.agentName ? (
                    <span className="text-lg font-bold text-foreground truncate">{state.currentTask.agentName}</span>
                  ) : null}
                </div>
              )}
            </div>
            {state.dagExecution && (
              <Button
                variant="ghost"
                size="icon"
                className={cn("rounded-full flex-shrink-0", progressPanelOpen ? "bg-muted text-primary" : "hover:bg-muted")}
                onClick={() => (progressPanelOpen ? handleCollapseProgressPanel() : setProgressPanelOpen(true))}
                title={progressPanelOpen ? t("chatPage.progressPanel.collapse") : t("chatPage.progressPanel.toggleTooltip")}
                aria-pressed={progressPanelOpen}
              >
                <PanelRight className="w-5 h-5" />
              </Button>
            )}
          </div>
        )}
        <div className="flex-1 min-h-0 relative">
          <TaskConversationPanel mode="page" />
        </div>
      </div>

      {/* Three responsive treatments, since a fixed side column has nowhere
          good to go at every width once TaskConversationPanel's own
          chat/PreviewSheet split is also in play:
          - Below `md`: no room for a side-by-side column at all, so it
            stacks below the chat instead, capped to a fraction of the
            viewport height (shrink-0 so that cap holds) - unchanged from
            before.
          - `md` to below `xl` (tablet/narrow desktop): a fixed overlay
            sliding in from the right edge, with a dismissible backdrop -
            this is the "overlay/collapse mode at constrained widths" the PR
            review asked for, specifically because this is the range where
            TaskConversationPanel's own PreviewSheet split (when a file/graph
            preview is also open) leaves too little room for a permanent
            side-by-side rail; an overlay draws on top instead of taking flex
            space from that split, so the two no longer compete for width.
          - `xl` and up: reverts to a normal static sibling column (siblings
            the header+chat column above, not nested under it, so it visually
            pops out from the right edge of the entire page) - there's
            finally enough width for both panels to coexist without either
            being squeezed. */}
      {progressPanelVisible && (
        <>
          {/* z-[110]/z-[120], above every other fixed/high-z-index element on
              this page: the header sits at z-50, TaskConversationPanel's own
              PreviewSheet column-resize handle at z-[100] (its wide invisible
              hit area would otherwise still capture drag clicks meant for
              this panel), and the floating voice-input mic button at z-[70]
              (which would otherwise float visibly on top of the dimmed
              backdrop) - all need to render underneath this overlay, not just
              the header. */}
          <div
            className="hidden md:block xl:hidden fixed inset-0 z-[110] bg-black/20"
            onClick={handleCollapseProgressPanel}
            aria-hidden="true"
          />
          <div
            className="w-full shrink-0 h-[45vh] border-t border-border md:fixed md:inset-y-0 md:right-0 md:z-[120] md:h-full md:w-[360px] md:border-t-0 md:border-l md:shadow-2xl xl:static xl:z-auto xl:shadow-none"
          >
            <ProgressPanel
              steps={progressSteps}
              startedAt={state.dagExecution?.created_at}
              endedAt={progressEndedAt}
              onCollapse={handleCollapseProgressPanel}
              onStepClick={handleProgressStepClick}
            />
          </div>
        </>
      )}
    </div>
  )
}

export default function TaskDetailPage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center h-full"><Loader2 className="w-8 h-8 animate-spin" /></div>}>
      <TaskDetailContent />
    </Suspense>
  )
}

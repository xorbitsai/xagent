"use client"

import { Suspense, useCallback, useEffect, useRef, useState } from "react"
import { ArrowLeft, Loader2, PanelRight } from "lucide-react"
import { useParams, useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { TaskConversationPanel } from "@/components/task/task-conversation-panel"
import { ProgressPanel, type ProgressStepView } from "@/components/task/progress-panel"
import { useApp } from "@/contexts/app-context-chat"
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

  const handleCollapseProgressPanel = useCallback(() => {
    const key = state.dagExecution?.created_at
    dismissedProgressRunKeyRef.current = key !== undefined && key !== null ? String(key) : null
    setProgressPanelOpen(false)
  }, [state.dagExecution?.created_at])

  const handleProgressStepClick = useCallback((stepId: string) => {
    window.dispatchEvent(new CustomEvent("scrollToTraceStep", { detail: { stepId } }))
  }, [])

  // Auto-open the panel the moment this turn enters DAG/plan_execute mode -
  // the plan's step list is known as soon as planning starts, so there is no
  // reason to wait for the user to click the header toggle. A manual
  // collapse only suppresses re-opening for *this* run (tracked by
  // dagExecution.created_at); the next DAG run gets a fresh created_at and
  // opens again.
  useEffect(() => {
    const createdAt = state.dagExecution?.created_at
    if (createdAt === undefined || createdAt === null) return
    const key = String(createdAt)
    if (dismissedProgressRunKeyRef.current === key) return
    setProgressPanelOpen(true)
  }, [state.dagExecution?.created_at])

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
  // in-flight dag_execution event. The task's own status/updatedAt is the one
  // signal that's actually authoritative for "this run is truly over."
  const isDagFinished = state.currentTask?.status === "completed" || state.currentTask?.status === "failed"
  const progressEndedAt = isDagFinished ? state.currentTask?.updatedAt : undefined

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

      {/* Spans the whole page's height (siblings the header+chat column
          above, not nested under it) so it visually pops out from the right
          edge of the entire page rather than just the content area below
          the header - and stays independent of the file/graph preview's
          draggable PreviewSheet layout inside TaskConversationPanel. Below
          `md` there's no room for a fixed 360px side-by-side column without
          squeezing the chat unusably narrow, so it stacks below the chat
          instead - the same stack-on-mobile/side-column-on-desktop idea as
          resizable-three-column-layout.tsx, capped to a fraction of the
          viewport height (shrink-0 so that cap holds) rather than taking the
          full remaining width. */}
      {progressPanelVisible && (
        <div className="w-full md:w-[360px] shrink-0 h-[45vh] md:h-full border-t md:border-t-0 md:border-l border-border">
          <ProgressPanel
            steps={progressSteps}
            startedAt={state.dagExecution?.created_at}
            endedAt={progressEndedAt}
            onCollapse={handleCollapseProgressPanel}
            onStepClick={handleProgressStepClick}
          />
        </div>
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

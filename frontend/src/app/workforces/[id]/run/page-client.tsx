"use client"

import React, { Suspense, useCallback, useEffect, useRef, useState } from "react"
import Link from "next/link"
import { useParams, useRouter, useSearchParams } from "next/navigation"
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  Position,
  MarkerType,
  type Node,
  type Edge,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import { AlertCircle, ArrowLeft, Bot, Crown, FileText, GitBranch, History, Loader2, MessageSquare, Pencil, Users, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { useI18n, type Translate } from "@/contexts/i18n-context"
import { useApp } from "@/contexts/app-context-chat"
import type { Task } from "@/contexts/app-context-chat"
import { normalizeTaskStatus, type TaskStatus } from "@/lib/task-status"
import { getWorkforce, getWorkforceAgentExecution, getWorkforceRun, runWorkforce } from "@/lib/workforces-api"
import { WorkforceRunsList, WorkforceStatusBadge } from "@/components/workforce"
import { TaskConversationPanel } from "@/components/task/task-conversation-panel"
import { TraceEventRenderer, type AgentExecutionSummary } from "@/components/chat/TraceEventRenderer"
import { FilePreviewActionButtons } from "@/components/file/file-preview-action-buttons"
import { FilePreviewContent } from "@/components/file/file-preview-content"
import { ResizableSplitLayout } from "@/components/layout/resizable-split-layout"
import type {
  WorkforceDetail,
  WorkforceAgentExecution,
  WorkforceAgentExecutionTraceEvent,
  WorkforceRunHistoryItem,
  WorkforceRunResponse,
} from "@/types/workforce"
import { toast } from "sonner"
import { cn } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { MarkdownRenderer } from "@/components/ui/markdown-renderer"

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

// ─── Flow panel node components ───────────────────────────────────────────────

interface FlowNodeData extends Record<string, unknown> {
  name: string
  avatar: string
  description?: string
  isLead?: boolean
  status?: "idle" | "running" | "done" | "failed"
}

function FlowManagerNode({ data }: { data: FlowNodeData }) {
  const { t } = useI18n()
  return (
    <div className="flex w-52 flex-col items-center rounded-xl border-2 border-primary/40 bg-card p-4 shadow-sm">
      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/15 text-base font-bold text-primary">
        {data.avatar}
      </div>
      <div className="mt-2 text-sm font-semibold text-foreground text-center">{data.name}</div>
      <div className="mt-1.5 flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
        <Crown className="h-3 w-3" />
        {t("workforces.detail.leadBadge")}
      </div>
      <Handle type="source" position={Position.Bottom} className="!border-none !bg-transparent" />
    </div>
  )
}

function FlowWorkerNode({ data }: { data: FlowNodeData }) {
  const statusColors: Record<string, string> = {
    running: "text-blue-600",
    done: "text-green-600",
    failed: "text-red-500",
    idle: "text-muted-foreground",
  }
  const statusLabels: Record<string, string> = {
    running: "RUNNING",
    done: "DONE",
    failed: "FAILED",
    idle: "",
  }
  const status = data.status ?? "idle"
  return (
    <div className="flex w-52 flex-col rounded-xl border border-border bg-card p-3 shadow-sm">
      <Handle type="target" position={Position.Top} className="!border-none !bg-transparent" />
      <div className="flex items-center gap-2.5">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/15 text-sm font-bold text-primary">
          {data.avatar}
        </div>
        <div className="min-w-0 flex-1">
          <div className="truncate text-xs font-semibold text-foreground">{data.name}</div>
          {data.description && (
            <div className="line-clamp-2 text-[10px] text-muted-foreground mt-0.5">{data.description}</div>
          )}
        </div>
      </div>
      {status !== "idle" && (
        <div className={cn("mt-2 flex items-center gap-1 text-[10px] font-semibold", statusColors[status])}>
          <span className="h-1.5 w-1.5 rounded-full bg-current" />
          {statusLabels[status]}
        </div>
      )}
    </div>
  )
}

const flowNodeTypes = {
  flowManager: FlowManagerNode,
  flowWorker: FlowWorkerNode,
}

type RunInspectorMode = "flow" | "agent" | "file"

function RunInspectorHeader({
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

// ─── Workforce flow panel ──────────────────────────────────────────────────────

interface WorkforceFlowPanelProps {
  workforce: WorkforceDetail
  taskStatus?: TaskStatus | null
  onClose: () => void
}

function WorkforceFlowPanel({ workforce, taskStatus, onClose }: WorkforceFlowPanelProps) {
  const { t } = useI18n()

  const { nodes, edges } = React.useMemo(() => {
    const newNodes: Node[] = []
    const newEdges: Edge[] = []

    // Manager node
    newNodes.push({
      id: "manager",
      type: "flowManager",
      position: { x: 0, y: 0 },
      origin: [0.5, 0],
      data: {
        name: workforce.manager?.name || "Manager",
        avatar: workforce.manager?.name?.charAt(0).toUpperCase() || "M",
        isLead: true,
      },
    })

    // Worker nodes
    const workers = workforce.workers.filter((w) => w.enabled)
    const workerWidth = 208 // w-52
    const gap = 24
    const totalWidth = workers.length * workerWidth + (workers.length - 1) * gap
    const startX = -totalWidth / 2 + workerWidth / 2

    workers.forEach((worker, index) => {
      const name = worker.alias || worker.agent.name
      const workerId = `worker-${worker.id}`
      const globalTaskStatus: FlowNodeData["status"] =
        taskStatus === "completed" ? "done" :
        taskStatus === "failed" ? "failed" :
        taskStatus === "running" ? "running" :
        "idle"

      newNodes.push({
        id: workerId,
        type: "flowWorker",
        position: { x: startX + index * (workerWidth + gap), y: 220 },
        origin: [0.5, 0],
        data: {
          name,
          avatar: name.charAt(0).toUpperCase(),
          description: worker.agent.description || undefined,
          status: globalTaskStatus,
        },
      })

      const isCompleted = taskStatus === "completed"
      newEdges.push({
        id: `edge-${workerId}`,
        source: "manager",
        target: workerId,
        type: "smoothstep",
        animated: taskStatus === "running",
        style: {
          stroke: isCompleted ? "#22c55e" : "#cbd5e1",
          strokeWidth: 2,
          strokeDasharray: taskStatus === "running" ? "6 3" : undefined,
        },
        markerEnd: { type: MarkerType.ArrowClosed, color: isCompleted ? "#22c55e" : "#cbd5e1" },
      })
    })

    return { nodes: newNodes, edges: newEdges }
  }, [workforce, taskStatus])

  const statusBadge = () => {
    if (!taskStatus) return null
    if (taskStatus === "completed") return (
      <span className="flex items-center gap-1 rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700">
        <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
        {t("workforces.status.completed")}
      </span>
    )
    if (taskStatus === "running") return (
      <span className="flex items-center gap-1 rounded-full bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-700">
        <span className="h-1.5 w-1.5 rounded-full bg-blue-500 animate-pulse" />
        {t("workforces.status.running")}
      </span>
    )
    if (taskStatus === "failed") return (
      <span className="flex items-center gap-1 rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-700">
        <span className="h-1.5 w-1.5 rounded-full bg-red-500" />
        {t("workforces.status.failed")}
      </span>
    )
    return null
  }

  return (
    <div className="flex h-full w-full flex-col bg-background">
      <RunInspectorHeader
        icon={<GitBranch className="h-4 w-4" />}
        title={t("workforces.canvas.title")}
        status={statusBadge()}
        onClose={onClose}
      />

      {/* ReactFlow */}
      <div className="flex-1 min-h-0">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={flowNodeTypes}
          fitView
          fitViewOptions={{ padding: 0.3 }}
          minZoom={0.3}
          maxZoom={1.5}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
        >
          <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#e2e8f0" />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>

      {/* Legend */}
      <div className="shrink-0 border-t px-4 py-3">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[10px] text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <span className="h-px w-5 bg-border" />
            {t("workforces.canvas.legend.reportsTo")}
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-px w-5 border-t-2 border-dashed border-blue-400" />
            {t("workforces.canvas.legend.activeDelegation")}
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-px w-5 bg-green-500" />
            {t("workforces.canvas.legend.completed")}
          </span>
        </div>
      </div>
    </div>
  )
}

// ─── Main page ─────────────────────────────────────────────────────────────────

export default function WorkforceRunPage() {
  // useSearchParams must be inside a Suspense boundary for static export.
  return (
    <Suspense fallback={null}>
      <WorkforceRunPageInner />
    </Suspense>
  )
}

function WorkforceRunPageInner() {
  const { t } = useI18n()
  const params = useParams()
  const router = useRouter()
  const searchParams = useSearchParams()
  const { sendMessage, setTaskId, closeFilePreview, dispatch, state } = useApp()
  const id = Array.isArray(params.id) ? params.id[0] : params.id
  const runParam = searchParams.get("run")

  const [workforce, setWorkforce] = useState<WorkforceDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [inspectorMode, setInspectorMode] = useState<RunInspectorMode | null>(null)
  const [taskStarted, setTaskStarted] = useState(false)
  const [selectedTaskId, setSelectedTaskId] = useState<number | null>(null)
  const [historyOpen, setHistoryOpen] = useState(false)
  const [agentExecutionSelection, setAgentExecutionSelection] = useState<AgentExecutionSummary | null>(null)
  const [agentExecutionDetail, setAgentExecutionDetail] = useState<WorkforceAgentExecution | null>(null)
  const [agentExecutionLoading, setAgentExecutionLoading] = useState(false)
  const [agentExecutionError, setAgentExecutionError] = useState<string | null>(null)

  const previewTaskIdRef = useRef<number | null>(null)
  const openedRunParamRef = useRef<string | null>(null)
  const activeWorkerTaskIdRef = useRef<string | null>(null)
  const agentExecutionRequestIdRef = useRef(0)
  const taskStatus = state.currentTask?.status ?? null

  const openRun = useCallback((run: WorkforceRunHistoryItem) => {
    if (!run.task_id) {
      toast.error(t("workforces.runs.taskDeleted"))
      return
    }
    setHistoryOpen(false)
    if (previewTaskIdRef.current === run.task_id) return
    // Keep ?run= authoritative no matter which path opened the run, and mark
    // it as opened so the deep-link effect doesn't refetch it.
    openedRunParamRef.current = String(run.id)
    router.replace(`/workforces/${id}/run?run=${run.id}`, { scroll: false })
    previewTaskIdRef.current = run.task_id
    activeWorkerTaskIdRef.current = null
    agentExecutionRequestIdRef.current += 1
    setSelectedTaskId(run.task_id)
    setTaskStarted(true)
    setInspectorMode(null)
    closeFilePreview()
    dispatch({ type: "SET_DAG_EXECUTION", payload: null })
    dispatch({ type: "SET_CURRENT_TASK", payload: null })
    setTaskId(run.task_id, { navigate: false })
    const title = run.task_title || run.message || `Run #${run.id}`
    const taskPayload: Task = {
      id: String(run.task_id),
      title,
      description: run.message ?? "",
      status: normalizeTaskStatus(run.status) ?? "completed",
      createdAt: run.created_at ?? new Date().toISOString(),
      updatedAt: run.completed_at ?? run.created_at ?? new Date().toISOString(),
    }
    dispatch({ type: "SET_CURRENT_TASK", payload: taskPayload })
  }, [closeFilePreview, dispatch, setTaskId, t])

  const openAgentExecution = useCallback(async (execution: AgentExecutionSummary) => {
    if (!id || !selectedTaskId) return
    closeFilePreview()
    setAgentExecutionSelection(execution)
    setAgentExecutionDetail(null)
    setAgentExecutionError(null)
    setAgentExecutionLoading(true)
    setInspectorMode("agent")
    const workerTaskId = execution.workerTaskId
    const requestId = agentExecutionRequestIdRef.current + 1
    agentExecutionRequestIdRef.current = requestId
    activeWorkerTaskIdRef.current = workerTaskId
    const isActiveRequest = () => (
      activeWorkerTaskIdRef.current === workerTaskId &&
      agentExecutionRequestIdRef.current === requestId
    )
    try {
      const detail = await getWorkforceAgentExecution(
        id,
        selectedTaskId,
        workerTaskId,
      )
      if (isActiveRequest()) {
        setAgentExecutionDetail(detail)
      }
    } catch (error) {
      if (isActiveRequest()) {
        setAgentExecutionError(
          error instanceof Error ? error.message : t("workforces.run.agentExecutionLoadError"),
        )
      }
    } finally {
      if (isActiveRequest()) {
        setAgentExecutionLoading(false)
      }
    }
  }, [closeFilePreview, id, selectedTaskId, t])

  useEffect(() => {
    if (state.filePreview.isOpen) {
      setInspectorMode("file")
      return
    }
    setInspectorMode((current) => current === "file" ? null : current)
  }, [state.filePreview.isOpen])

  const liveTraceEvents = React.useMemo(
    () => sanitizeAgentExecutionTraceEvents(state.traceEvents),
    [state.traceEvents],
  )

  const agentExecutionEvents = React.useMemo(() => {
    if (!agentExecutionSelection) return []
    return mergeAgentExecutionTraceEvents(
      agentExecutionDetail?.trace_events ?? [],
      liveTraceEvents,
      agentExecutionSelection.workerTaskId,
    )
  }, [agentExecutionDetail?.trace_events, agentExecutionSelection, liveTraceEvents])

  const agentExecutionStatus = React.useMemo(() => {
    if (!agentExecutionSelection) return undefined
    let status = agentExecutionDetail?.status || agentExecutionSelection.status
    for (const event of liveTraceEvents) {
      const eventData = event.data ?? {}
      if (String(eventData["worker_task_id"] ?? "") !== agentExecutionSelection.workerTaskId) continue
      if (event.event_type === "workforce_delegation_end") status = "completed"
      if (event.event_type === "workforce_delegation_error") status = "failed"
    }
    if (
      status === "running" &&
      (taskStatus === "completed" || taskStatus === "failed")
    ) {
      status = "interrupted"
    }
    return status
  }, [
    agentExecutionDetail?.status,
    agentExecutionSelection,
    liveTraceEvents,
    taskStatus,
  ])

  useEffect(() => {
    const load = async () => {
      if (!id) return
      try {
        setWorkforce(await getWorkforce(id))
      } catch (error) {
        toast.error(error instanceof Error ? error.message : t("workforces.errors.load"))
      } finally {
        setLoading(false)
      }
    }
    void load()
  }, [id, t])

  useEffect(() => {
    if (!id) return
    if (!runParam) {
      if (openedRunParamRef.current !== null) {
        openedRunParamRef.current = null
        previewTaskIdRef.current = null
        activeWorkerTaskIdRef.current = null
        agentExecutionRequestIdRef.current += 1
        setSelectedTaskId(null)
        setTaskStarted(false)
        setInspectorMode(null)
        closeFilePreview()
        dispatch({ type: "CLEAR_MESSAGES" })
        dispatch({ type: "SET_TRACE_EVENTS", payload: [] })
        dispatch({ type: "SET_STEPS", payload: [] })
        dispatch({ type: "SET_DAG_EXECUTION", payload: null })
        dispatch({ type: "SET_CURRENT_TASK", payload: null })
        dispatch({ type: "SET_HISTORY_LOADING", payload: false })
        setTaskId(null, { navigate: false })
      }
      return
    }
    if (openedRunParamRef.current === runParam) return
    openedRunParamRef.current = runParam
    let active = true
    let settled = false
    void (async () => {
      try {
        const run = await getWorkforceRun(id, runParam)
        settled = true
        if (active) openRun(run)
      } catch (error) {
        settled = true
        if (active) {
          openedRunParamRef.current = null
          toast.error(
            error instanceof Error ? error.message : t("workforces.runs.loadError"),
          )
        }
      }
    })()
    return () => {
      active = false
      if (!settled && openedRunParamRef.current === runParam) {
        openedRunParamRef.current = null
      }
    }
  }, [closeFilePreview, dispatch, id, openRun, runParam, setTaskId, t])

  const cleanupRef = useRef({ closeFilePreview, dispatch, setTaskId })
  cleanupRef.current = { closeFilePreview, dispatch, setTaskId }

  useEffect(() => {
    return () => {
      const { closeFilePreview: close, dispatch: d, setTaskId: set } = cleanupRef.current
      previewTaskIdRef.current = null
      close()
      d({ type: "CLEAR_MESSAGES" })
      d({ type: "SET_TRACE_EVENTS", payload: [] })
      d({ type: "SET_STEPS", payload: [] })
      d({ type: "SET_DAG_EXECUTION", payload: null })
      d({ type: "SET_CURRENT_TASK", payload: null })
      d({ type: "SET_HISTORY_LOADING", payload: false })
      set(null, { navigate: false })
    }
  }, [])

  const handleSend = useCallback(async (
    content: string,
    _config?: unknown,
    files?: (File & { file_id?: string })[],
  ) => {
    if (!id) return
    let taskId = previewTaskIdRef.current
    if (taskId === -1) return

    try {
      if (!taskId) {
        previewTaskIdRef.current = -1
        const result: WorkforceRunResponse = await runWorkforce(id, {
          message: content,
          files: (files ?? []).map((f) => f.file_id).filter(Boolean) as string[],
          is_visible: false,
        })
        taskId = result.task_id
        if (!taskId) throw new Error("Invalid run response: missing task_id")
        previewTaskIdRef.current = taskId
        setSelectedTaskId(taskId)
        setTaskStarted(true)
        const now = new Date().toISOString()
        closeFilePreview()
        setTaskId(taskId, { navigate: false })
        const taskPayload: Task = {
          id: String(taskId),
          title: content.slice(0, 80),
          description: content,
          status: result.status as TaskStatus,
          createdAt: now,
          updatedAt: now,
        }
        dispatch({ type: "SET_CURRENT_TASK", payload: taskPayload })
        dispatch({ type: "TRIGGER_TASK_UPDATE" })
      } else {
        await sendMessage(content, { force: true, targetTaskId: taskId }, files)
      }
    } catch (err) {
      if (previewTaskIdRef.current === -1) previewTaskIdRef.current = null
      toast.error(err instanceof Error ? err.message : t("workforces.errors.run"))
    }
  }, [id, closeFilePreview, setTaskId, dispatch, sendMessage, t])

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-muted-foreground">{t("workforces.loading.runView")}</p>
      </div>
    )
  }

  if (!workforce) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-muted-foreground">{t("workforces.errors.notFound")}</p>
      </div>
    )
  }

  const managerName = workforce.manager?.name || t("workforces.canvas.nodeTypes.manager")

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Header */}
      <div className="flex h-14 shrink-0 items-center gap-2 border-b bg-card/30 px-4">
        <Link href="/workforces" className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors">
          <ArrowLeft className="h-3.5 w-3.5" />
          {t("workforces.list.title")}
        </Link>
        <span className="text-muted-foreground">/</span>
        <span className="max-w-[160px] truncate text-sm font-medium">{workforce.name}</span>
        <WorkforceStatusBadge status={workforce.status} />

        <div className="flex-1" />

        <Popover open={historyOpen} onOpenChange={setHistoryOpen}>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm" className="gap-1.5">
              <History className="h-3.5 w-3.5" />
              {t("workforces.runs.title")}
            </Button>
          </PopoverTrigger>
          <PopoverContent align="end" className="max-h-[70vh] w-96 overflow-y-auto p-3">
            {historyOpen && id ? (
              <WorkforceRunsList workforceId={id} compact onSelectRun={openRun} />
            ) : null}
          </PopoverContent>
        </Popover>
        <Button
          variant={inspectorMode === "flow" ? "secondary" : "outline"}
          size="sm"
          className="gap-1.5"
          onClick={() => {
            if (inspectorMode === "flow") {
              setInspectorMode(null)
              return
            }
            closeFilePreview()
            setInspectorMode("flow")
          }}
        >
          <GitBranch className="h-3.5 w-3.5" />
          {t("workforces.canvas.title")}
        </Button>
        <Button variant="outline" size="sm" className="gap-1.5" asChild>
          <Link href={`/workforces/${id}`}>
            <Pencil className="h-3.5 w-3.5" />
            {t("workforces.actions.edit")}
          </Link>
        </Button>
      </div>

      {/* Body */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        <div className="flex min-w-0 flex-1">
          <ResizableSplitLayout
            initialLeftWidth={inspectorMode === "file" ? 50 : 65}
            minLeftWidth={35}
            maxLeftWidth={80}
            leftPanel={<ChatArea taskStarted={taskStarted} managerName={managerName} handleSend={handleSend} onAgentExecutionClick={openAgentExecution} t={t} />}
            rightPanel={inspectorMode ? (
              <div
                className="h-full min-h-0 bg-background"
                data-testid="workforce-run-inspector"
                data-mode={inspectorMode}
              >
                {inspectorMode === "flow" ? (
                  <WorkforceFlowPanel
                    workforce={workforce}
                    taskStatus={taskStatus}
                    onClose={() => setInspectorMode(null)}
                  />
                ) : inspectorMode === "agent" ? (
                  <AgentExecutionPanel
                    selection={agentExecutionSelection}
                    detail={agentExecutionDetail}
                    events={agentExecutionEvents}
                    status={agentExecutionStatus}
                    loading={agentExecutionLoading}
                    error={agentExecutionError}
                    onClose={() => setInspectorMode(null)}
                    onRetry={() => agentExecutionSelection && void openAgentExecution(agentExecutionSelection)}
                    t={t}
                  />
                ) : (
                  <WorkforceFilePreviewPanel onClose={() => setInspectorMode(null)} />
                )}
              </div>
            ) : null}
          />
        </div>
      </div>
    </div>
  )
}

// ─── Run history ──────────────────────────────────────────────────────────────

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

function AgentExecutionPanel({
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

function WorkforceFilePreviewPanel({ onClose }: { onClose: () => void }) {
  const { state, dispatch, closeFilePreview, getFileDownloadUrl } = useApp()
  const { t } = useI18n()

  const handleClose = () => {
    closeFilePreview()
    onClose()
  }

  const handleDownload = async () => {
    try {
      if (!state.filePreview.fileId) return
      const response = await apiRequest(getFileDownloadUrl(state.filePreview.fileId))
      if (!response.ok) throw new Error(`Download failed: ${response.statusText}`)

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.href = url
      link.download = state.filePreview.fileName || "download"
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error("Failed to download file:", error)
      toast.error(t("workforces.errors.download"))
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-background">
      <RunInspectorHeader
        icon={<FileText className="h-4 w-4" />}
        title={state.filePreview.fileName}
        actions={(
          <FilePreviewActionButtons
            viewMode={state.filePreview.viewMode}
            onViewModeChange={(mode) => dispatch({ type: "SET_FILE_PREVIEW_MODE", payload: mode })}
            fileName={state.filePreview.fileName || ""}
            onDownload={() => void handleDownload()}
            showText={false}
          />
        )}
        onClose={handleClose}
      />
      <div className="min-h-0 flex-1 overflow-hidden">
        <FilePreviewContent open={state.filePreview.isOpen} />
      </div>
    </div>
  )
}

// ─── Chat area (shared between solo and split-layout render) ──────────────────

function ChatArea({
  taskStarted,
  managerName,
  handleSend,
  onAgentExecutionClick,
  t,
}: {
  taskStarted: boolean
  managerName: string
  handleSend: (content: string, config?: unknown, files?: (File & { file_id?: string })[]) => Promise<void>
  onAgentExecutionClick: (execution: AgentExecutionSummary) => void
  t: Translate
}) {
  if (!taskStarted) {
    return (
      <div className="flex h-full w-full flex-col">
        <div className="flex flex-1 flex-col items-center justify-center px-4 text-center">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-muted/60">
            <Users className="h-7 w-7 text-muted-foreground" />
          </div>
          <h2 className="mt-4 text-lg font-semibold">{t("workforces.run.readyTitle")}</h2>
          <p className="mt-1.5 max-w-sm text-sm text-muted-foreground">
            {t("workforces.run.readyDesc", { manager: managerName })}
          </p>
        </div>
        <div className="shrink-0 border-t bg-background">
          <div className="mx-auto max-w-3xl px-4 py-4">
            <RunInput
              placeholder={t("workforces.run.placeholder", { manager: managerName })}
              onSend={(content) => handleSend(content)}
              hint={t("workforces.run.inputHint")}
            />
          </div>
        </div>
      </div>
    )
  }

  return (
    <TaskConversationPanel
      mode="embedded-preview"
      showTaskActions
      showTokenUsage
      showDagPreview={false}
      showTaskFiles
      hideFileUpload={false}
      autoFocusInput={false}
      onSend={handleSend}
      onAgentExecutionClick={onAgentExecutionClick}
    />
  )
}

// ─── Simple input component for empty state ────────────────────────────────────

function RunInput({
  placeholder,
  hint,
  onSend,
}: {
  placeholder: string
  hint: string
  onSend: (content: string) => Promise<void>
}) {
  const [value, setValue] = useState("")
  const [loading, setLoading] = useState(false)

  const submit = async () => {
    const text = value.trim()
    if (!text || loading) return
    setLoading(true)
    try {
      await onSend(text)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-2">
      <div className="flex items-end gap-2 rounded-xl border bg-card px-4 py-3 shadow-sm focus-within:border-primary/50 transition-colors">
        <textarea
          className="flex-1 resize-none bg-transparent text-sm outline-none placeholder:text-muted-foreground/60 min-h-[44px] max-h-[160px]"
          placeholder={placeholder}
          value={value}
          rows={1}
          onChange={(e) => {
            setValue(e.target.value)
            e.target.style.height = "auto"
            e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault()
              void submit()
            }
          }}
        />
        <button
          disabled={!value.trim() || loading}
          onClick={submit}
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground transition-opacity disabled:opacity-40"
        >
          <svg className="h-4 w-4 rotate-90" fill="currentColor" viewBox="0 0 20 20">
            <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
          </svg>
        </button>
      </div>
      <p className="text-center text-xs text-muted-foreground">{hint}</p>
    </div>
  )
}

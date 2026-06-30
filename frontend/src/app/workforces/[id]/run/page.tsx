"use client"

import React, { useCallback, useEffect, useRef, useState } from "react"
import Link from "next/link"
import { useParams } from "next/navigation"
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
import { ArrowLeft, Crown, GitBranch, Pencil, Users, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useI18n } from "@/contexts/i18n-context"
import { useApp } from "@/contexts/app-context-chat"
import type { Task } from "@/contexts/app-context-chat"
import { type TaskStatus } from "@/lib/task-status"
import { getWorkforce, runWorkforce } from "@/lib/workforces-api"
import { WorkforceStatusBadge } from "@/components/workforce"
import { TaskConversationPanel } from "@/components/task/task-conversation-panel"
import { ResizableSplitLayout } from "@/components/layout/resizable-split-layout"
import type { WorkforceDetail, WorkforceRunResponse } from "@/types/workforce"
import { toast } from "sonner"
import { cn } from "@/lib/utils"

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
      {/* Header */}
      <div className="flex h-14 shrink-0 items-center justify-between border-b px-4">
        <div className="flex items-center gap-2">
          <GitBranch className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-semibold">{t("workforces.canvas.title")}</span>
          {statusBadge()}
        </div>
        <button
          onClick={onClose}
          className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

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
  const { t } = useI18n()
  const params = useParams()
  const { sendMessage, setTaskId, closeFilePreview, dispatch, state } = useApp()
  const id = Array.isArray(params.id) ? params.id[0] : params.id

  const [workforce, setWorkforce] = useState<WorkforceDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [showFlow, setShowFlow] = useState(false)
  const [taskStarted, setTaskStarted] = useState(false)

  const previewTaskIdRef = useRef<number | null>(null)
  const taskStatus = state.currentTask?.status ?? null

  useEffect(() => {
    const load = async () => {
      if (!id) return
      try {
        const data = await getWorkforce(id)
        setWorkforce(data)
      } catch (err) {
        toast.error(err instanceof Error ? err.message : t("workforces.errors.load"))
      } finally {
        setLoading(false)
      }
    }
    void load()
  }, [id, t])

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
        setTaskStarted(true)
        closeFilePreview()
        setTaskId(taskId, { navigate: false })
        const taskPayload: Task = {
          id: String(taskId),
          title: content.slice(0, 80),
          description: content,
          status: result.status as TaskStatus,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
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

        <Button
          variant={showFlow ? "secondary" : "outline"}
          size="sm"
          className="gap-1.5"
          onClick={() => setShowFlow((v) => !v)}
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
        {showFlow ? (
          <ResizableSplitLayout
            initialLeftWidth={65}
            minLeftWidth={35}
            maxLeftWidth={80}
            leftPanel={<ChatArea taskStarted={taskStarted} managerName={managerName} handleSend={handleSend} t={t} />}
            rightPanel={
              <WorkforceFlowPanel
                workforce={workforce}
                taskStatus={taskStatus}
                onClose={() => setShowFlow(false)}
              />
            }
          />
        ) : (
          <div className="flex flex-1 min-w-0 h-full">
            <ChatArea taskStarted={taskStarted} managerName={managerName} handleSend={handleSend} t={t} />
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Chat area (shared between solo and split-layout render) ──────────────────

function ChatArea({
  taskStarted,
  managerName,
  handleSend,
  t,
}: {
  taskStarted: boolean
  managerName: string
  handleSend: (content: string, config?: unknown, files?: (File & { file_id?: string })[]) => Promise<void>
  t: (key: string, params?: Record<string, string | number>) => string
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
      showTaskActions={false}
      showTokenUsage={false}
      showDagPreview={false}
      showTaskFiles={false}
      hideFileUpload={false}
      autoFocusInput={false}
      onSend={handleSend}
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

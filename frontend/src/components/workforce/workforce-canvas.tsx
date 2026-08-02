"use client"

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  Position,
  MarkerType,
  Node,
  Edge,
  FitViewOptions,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import { Crown, Plus, Settings } from "lucide-react"
import { toast } from "sonner"
import { useI18n } from "@/contexts/i18n-context"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import type { WorkforceAgentSummary, WorkforceWorker } from "@/types/workforce"
import type { WorkforceEditDialogsState } from "./workforce-edit-dialogs"
import { GetStartedChecklist, type GetStartedStep } from "./workforce-get-started"

interface WorkforceCanvasProps {
  name: string
  description: string
  onSaveDetails: (data: { name: string; description: string }) => Promise<void>
  manager: WorkforceAgentSummary | null
  workers: WorkforceWorker[]
  isArchived?: boolean
  dialogs: WorkforceEditDialogsState
  getStartedSteps: GetStartedStep[]
  getStartedCollapsed: boolean
  onToggleGetStarted: () => void
}

export interface NodeData {
  name: string
  avatar: string
  description?: string
  subtitle?: string
  worker?: WorkforceWorker
  clickable?: boolean
}

interface DetailsNodeData {
  name: string
  description: string
  onSave: (data: { name: string; description: string }) => Promise<void>
  // Every other interactive canvas node (manager, worker, add-worker) is
  // already gated on isArchived, and the equivalent Configure-panel editor
  // gates on it too -- this node was the one exception (PR review round 8,
  // F-NEW-2).
  isArchived: boolean
}

export function DetailsNode({ data }: { data: DetailsNodeData }) {
  const { t } = useI18n()
  const [name, setName] = useState(data.name)
  const [description, setDescription] = useState(data.description)
  const disabled = data.isArchived

  // Skip the prop resync while a field is focused: a save response landing
  // while the user is still actively editing that same field (see
  // savingRef/pendingRef below) would otherwise silently overwrite whatever
  // they've typed since.
  const nameFocusedRef = useRef(false)
  const descriptionFocusedRef = useRef(false)

  useEffect(() => {
    if (!nameFocusedRef.current) setName(data.name)
  }, [data.name])
  useEffect(() => {
    if (!descriptionFocusedRef.current) setDescription(data.description)
  }, [data.description])

  // Serializes this node's own saves instead of disabling the fields while
  // one is in flight. Disabling on blur used to force a synchronous
  // re-render mid-click, which un-focused whichever field the user was
  // clicking into next (e.g. blur name -> click description) before the
  // browser finished moving focus there -- a disabled element can't receive
  // focus (PR review round 9, NEW-F1). Queuing the latest values here keeps
  // both fields interactive at all times while still guaranteeing saves
  // land in order rather than racing each other.
  const savingRef = useRef(false)
  const pendingRef = useRef<{ name: string; description: string } | null>(null)

  const runSave = (payload: { name: string; description: string }) => {
    savingRef.current = true
    data.onSave(payload).finally(() => {
      savingRef.current = false
      const queued = pendingRef.current
      pendingRef.current = null
      if (queued) runSave(queued)
    })
  }

  const commit = () => {
    if (disabled) {
      // The workforce can be archived out from under an in-progress edit
      // here (the header's Archive button, unrelated to this node), which
      // forces a blur via the `disabled` attribute flipping true. There's
      // nothing to retry with -- an archived workforce's PATCH is rejected
      // either way -- but the edit used to simply vanish with no
      // indication it was never saved (PR review round 9, MINOR-3).
      if (name.trim() !== data.name || description.trim() !== data.description) {
        toast.error(t("workforces.errors.editDiscardedByArchive"))
      }
      return
    }
    // Mirror the Configure panel's save button, which is disabled while the
    // trimmed name is empty: don't fire an empty-name save (a 422 in edit
    // mode, or a silently-blanked draft name with no explanation in create
    // mode) -- revert to the last saved value instead.
    if (!name.trim()) {
      setName(data.name)
      return
    }
    const trimmedName = name.trim()
    const trimmedDescription = description.trim()
    if (trimmedName === data.name && trimmedDescription === data.description) return
    const payload = { name: trimmedName, description: trimmedDescription }
    if (savingRef.current) {
      pendingRef.current = payload
      return
    }
    runSave(payload)
  }

  return (
    <div className="w-80 cursor-default space-y-1.5 rounded-xl border bg-card p-4 shadow-sm">
      <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        {t("workforces.detail.detailsTitle")}
      </div>
      <Input
        value={name}
        onChange={(e) => setName(e.target.value)}
        onFocus={() => { nameFocusedRef.current = true }}
        onBlur={() => { nameFocusedRef.current = false; commit() }}
        disabled={disabled}
        placeholder={t("workforces.create.placeholders.name")}
        className="nodrag border-none bg-transparent px-0 text-base font-semibold shadow-none focus-visible:ring-0"
      />
      <Textarea
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        onFocus={() => { descriptionFocusedRef.current = true }}
        onBlur={() => { descriptionFocusedRef.current = false; commit() }}
        disabled={disabled}
        placeholder={t("workforces.create.placeholders.description")}
        rows={1}
        className="nodrag resize-none border-none bg-transparent px-0 text-sm shadow-none focus-visible:ring-0"
      />
    </div>
  )
}

export function ManagerNode({ data }: { data: NodeData }) {
  const { t } = useI18n()
  return (
    <div
      className={`flex w-72 flex-col items-center justify-center rounded-xl border-2 border-primary/30 bg-card p-6 shadow-sm transition-colors ${
        data.clickable ? "cursor-pointer hover:border-primary/60 hover:shadow-md" : ""
      }`}
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/15 text-xl font-bold text-primary">
        {data.avatar}
      </div>
      <div className="mt-3 text-base font-semibold text-foreground">{data.name}</div>
      <div className="mt-2 flex items-center gap-1 rounded-full bg-primary/10 px-2.5 py-0.5 text-xs font-medium text-primary">
        <Crown className="h-3.5 w-3.5" />
        {t("workforces.canvas.nodeTypes.manager")}
      </div>
      {data.description && (
        <div className="mt-3 text-center text-xs text-muted-foreground line-clamp-2">
          {data.description}
        </div>
      )}
      {data.clickable && (
        <div className="mt-3 flex items-center gap-1 text-[11px] font-medium text-muted-foreground">
          <Settings className="h-3 w-3" />
          {t("workforces.actions.change")}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} className="!border-none !bg-transparent" />
    </div>
  )
}

export function WorkerNode({ data }: { data: NodeData }) {
  const { t } = useI18n()
  return (
    <div
      className={`flex w-56 flex-col rounded-xl border border-border bg-card p-4 shadow-sm transition-colors ${
        data.clickable ? "cursor-pointer hover:border-primary/40 hover:shadow-md" : ""
      }`}
    >
      <Handle type="target" position={Position.Top} className="!border-none !bg-transparent" />
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/15 text-sm font-bold text-primary">
          {data.avatar}
        </div>
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-semibold text-foreground">{data.name}</div>
          {data.subtitle && (
            <div className="line-clamp-2 text-xs text-muted-foreground mt-0.5">{data.subtitle}</div>
          )}
        </div>
      </div>
      {data.clickable && (
        <div className="mt-3 flex items-center gap-1 border-t pt-2 text-[11px] font-medium text-muted-foreground">
          <Settings className="h-3 w-3" />
          {t("workforces.canvas.configure")}
        </div>
      )}
    </div>
  )
}

export function AddNode({ data }: { data: { label: string; subtitle: string } }) {
  return (
    <div className="flex w-56 cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-muted-foreground/30 bg-muted/20 p-4 text-center transition-colors hover:border-primary/50 hover:bg-muted/40">
      <Handle type="target" position={Position.Top} className="!border-none !bg-transparent" />
      <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary/10 text-primary">
        <Plus className="h-4 w-4" />
      </div>
      <div className="text-sm font-semibold text-foreground">{data.label}</div>
      <div className="text-xs text-muted-foreground line-clamp-2">{data.subtitle}</div>
    </div>
  )
}

export function ChooseLeadNode({ data }: { data: { label: string; subtitle: string } }) {
  return (
    <div className="flex w-72 cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-muted-foreground/30 bg-muted/20 p-6 text-center transition-colors hover:border-primary/50 hover:bg-muted/40">
      <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary/10 text-primary">
        <Crown className="h-4 w-4" />
      </div>
      <div className="text-sm font-semibold text-foreground">{data.label}</div>
      <div className="text-xs text-muted-foreground">{data.subtitle}</div>
      <Handle type="source" position={Position.Bottom} className="!border-none !bg-transparent" />
    </div>
  )
}

const nodeTypes = {
  details: DetailsNode,
  manager: ManagerNode,
  worker: WorkerNode,
  add: AddNode,
  "choose-lead": ChooseLeadNode,
}

const MANAGER_Y = 240
const WORKER_Y = 560

// Explicit widths matching each node's Tailwind width class — without these,
// @xyflow/react routes edges against an unmeasured (0×0) node on the very
// first render of a newly-added node, before its ResizeObserver callback
// fires, which is what made edges jump to the wrong spot right after adding
// an agent. Heights are intentionally left unset: they vary with content
// (a chosen manager's description, the empty-state hint copy, etc.), and
// forcing a guessed height either clips content or leaves a dead click zone
// below it — @xyflow/react measures the real height on mount and keeps
// edges anchored to it. WORKER_Y instead just leaves enough clearance below
// MANAGER_Y for the tallest of the manager/choose-lead variants.
const DETAILS_WIDTH = 320 // w-80
const MANAGER_WIDTH = 288 // w-72
const WORKER_WIDTH_PX = 224 // w-56

// Reserve room in the top-left corner so fitView doesn't center graph
// content underneath the fixed Get Started overlay (which floats above the
// canvas and isn't part of the flow itself).
const FIT_VIEW_OPTIONS: FitViewOptions = {
  padding: { top: "220px", left: "340px", right: "40px", bottom: "40px" },
}

function WorkforceCanvasInner({
  name,
  description,
  onSaveDetails,
  manager,
  workers,
  isArchived = false,
  dialogs,
  getStartedSteps,
  getStartedCollapsed,
  onToggleGetStarted,
}: WorkforceCanvasProps) {
  const { t } = useI18n()
  const { fitView } = useReactFlow()
  const isFirstRenderRef = useRef(true)

  // The `fitView` prop below only fits once, on mount. Expanding the Get
  // Started checklist after that grows the fixed overlay FIT_VIEW_OPTIONS
  // reserves room for, so without this, nodes laid out for the collapsed
  // (smaller) reservation can end up overlapping it.
  useEffect(() => {
    if (isFirstRenderRef.current) {
      isFirstRenderRef.current = false
      return
    }
    fitView(FIT_VIEW_OPTIONS)
  }, [getStartedCollapsed, fitView])

  const { nodes, edges } = useMemo(() => {
    const newNodes: Node[] = []
    const newEdges: Edge[] = []

    newNodes.push({
      id: "details",
      type: "details",
      position: { x: 0, y: 0 },
      origin: [0.5, 0],
      draggable: false,
      selectable: false,
      width: DETAILS_WIDTH,
      data: {
        name,
        description,
        onSave: (data: { name: string; description: string }) =>
          // handleSaveDetails already toasts and sets the error state on
          // failure, then re-throws so callers that need to react (the
          // config-panel's edit view stays open) can -- this call site
          // doesn't need to, so swallow it here rather than leaving an
          // unhandled promise rejection. Still returned (not fire-and-
          // forget) so DetailsNode's own save-serialization can await it.
          onSaveDetails(data).catch(() => {}),
        isArchived,
      } satisfies DetailsNodeData,
    })

    // Manager node — an empty dashed placeholder until a lead is chosen
    if (manager) {
      newNodes.push({
        id: "manager",
        type: "manager",
        position: { x: 0, y: MANAGER_Y },
        origin: [0.5, 0],
        // Layout is fully recomputed from array index/constants on every
        // render (no onNodesChange wired up to persist a drag, and
        // canvas_position isn't read here) -- dragging would just silently
        // snap back on the next unrelated re-render, so disable it rather
        // than ship a broken affordance.
        draggable: false,
        width: MANAGER_WIDTH,
        data: {
          name: manager.name,
          avatar: manager.name.charAt(0).toUpperCase(),
          description: manager.description || "",
          clickable: !isArchived,
        } satisfies NodeData,
      })
    } else {
      newNodes.push({
        id: "manager",
        type: "choose-lead",
        position: { x: 0, y: MANAGER_Y },
        origin: [0.5, 0],
        draggable: false,
        width: MANAGER_WIDTH,
        data: {
          label: t("workforces.canvas.chooseLead.title"),
          subtitle: t("workforces.canvas.chooseLead.hint"),
        },
      })
    }

    // Worker nodes
    const workerWidth = WORKER_WIDTH_PX
    const gap = 64
    const totalSlots = workers.length + (isArchived ? 0 : 1)
    const totalWidth = totalSlots * workerWidth + (totalSlots - 1) * gap
    const startX = -totalWidth / 2 + workerWidth / 2

    workers.forEach((worker, index) => {
      const workerId = `worker-${worker.id}`
      newNodes.push({
        id: workerId,
        type: "worker",
        position: { x: startX + index * (workerWidth + gap), y: WORKER_Y },
        origin: [0.5, 0],
        draggable: false,
        width: WORKER_WIDTH_PX,
        data: {
          name: worker.alias || worker.agent?.name || t("workforces.canvas.nodeTypes.worker"),
          avatar: (worker.alias || worker.agent?.name || "W").charAt(0).toUpperCase(),
          subtitle: worker.agent?.description || "",
          worker,
          clickable: !isArchived,
        } satisfies NodeData,
      })

      newEdges.push({
        id: `edge-manager-${workerId}`,
        source: "manager",
        target: workerId,
        type: "smoothstep",
        animated: true,
        style: { stroke: "#cbd5e1", strokeWidth: 2 },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: "#cbd5e1",
        },
      })
    })

    // "Add agent" node — always the last slot in the row, unless read-only
    if (!isArchived) {
      newNodes.push({
        id: "add-worker",
        type: "add",
        position: { x: startX + workers.length * (workerWidth + gap), y: WORKER_Y },
        origin: [0.5, 0],
        draggable: false,
        width: WORKER_WIDTH_PX,
        data:
          workers.length === 0
            ? {
                label: t("workforces.canvas.addFirstAgent.title"),
                subtitle: t("workforces.canvas.addFirstAgent.hint"),
              }
            : {
                label: t("workforces.actions.addAgent"),
                subtitle: t("workforces.detail.membersHint"),
              },
      })
      newEdges.push({
        id: "edge-manager-add-worker",
        source: "manager",
        target: "add-worker",
        type: "smoothstep",
        style: { stroke: "#cbd5e1", strokeWidth: 2, strokeDasharray: "4 4" },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: "#cbd5e1",
        },
      })
    }

    return { nodes: newNodes, edges: newEdges }
  }, [name, description, onSaveDetails, manager, workers, isArchived, t])

  const handleNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      if (isArchived) return
      if (node.type === "manager" || node.type === "choose-lead") {
        dialogs.setChangeLeadOpen(true)
      } else if (node.type === "worker") {
        const worker = (node.data as unknown as NodeData).worker
        if (worker) dialogs.openMemberDetail(worker)
      } else if (node.type === "add") {
        dialogs.setAddMemberOpen(true)
      }
    },
    [isArchived, dialogs],
  )

  return (
    <div className="relative h-full w-full rounded-xl border bg-gray-50/50">
      <div className="absolute top-4 left-4 z-10 w-80">
        <GetStartedChecklist
          steps={getStartedSteps}
          collapsed={getStartedCollapsed}
          onToggleCollapsed={onToggleGetStarted}
        />
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodeClick={handleNodeClick}
        fitView
        fitViewOptions={FIT_VIEW_OPTIONS}
        minZoom={0.1}
        maxZoom={1.5}
      >
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#cbd5e1" />
        <Controls />
      </ReactFlow>
    </div>
  )
}

export function WorkforceCanvas(props: WorkforceCanvasProps) {
  return (
    <ReactFlowProvider>
      <WorkforceCanvasInner {...props} />
    </ReactFlowProvider>
  )
}

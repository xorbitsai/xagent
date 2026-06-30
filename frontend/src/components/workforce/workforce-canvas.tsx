"use client"

import React, { useMemo } from "react"
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  Position,
  MarkerType,
  Node,
  Edge,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import { Crown } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import type { WorkforceDetail, WorkforceAgentOption, WorkforceWorkerDraft } from "@/types/workforce"

interface WorkforceCanvasProps {
  workforce: WorkforceDetail
}

interface NodeData {
  name: string
  avatar: string
  description: string
  subtitle?: string
}

function ManagerNode({ data }: { data: NodeData }) {
  const { t } = useI18n()
  return (
    <div className="flex w-72 flex-col items-center justify-center rounded-xl border-2 border-primary/30 bg-card p-6 shadow-sm">
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
      <Handle type="source" position={Position.Bottom} className="!border-none !bg-transparent" />
    </div>
  )
}

function WorkerNode({ data }: { data: NodeData }) {
  return (
    <div className="flex w-56 flex-col rounded-xl border border-border bg-card p-4 shadow-sm">
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
    </div>
  )
}

const nodeTypes = {
  manager: ManagerNode,
  worker: WorkerNode,
}

interface WorkforceDraftCanvasProps {
  managerAgent: WorkforceAgentOption | undefined
  managerInstructions: string
  workers: WorkforceWorkerDraft[]
  agents: WorkforceAgentOption[]
}

export function WorkforceDraftCanvas({
  managerAgent,
  managerInstructions,
  workers,
  agents,
}: WorkforceDraftCanvasProps) {
  const { t } = useI18n()

  const { nodes, edges } = useMemo(() => {
    const newNodes: Node[] = []
    const newEdges: Edge[] = []

    newNodes.push({
      id: "manager",
      type: "manager",
      position: { x: 0, y: 0 },
      origin: [0.5, 0],
      data: {
        name: managerAgent?.name || t("workforces.canvas.nodeTypes.manager"),
        avatar: managerAgent?.name ? managerAgent.name.charAt(0).toUpperCase() : "M",
        description: managerInstructions || managerAgent?.description || "",
      },
    })

    const workerWidth = 256
    const gap = 32
    const totalWidth = workers.length * workerWidth + (workers.length - 1) * gap
    const startX = -totalWidth / 2 + workerWidth / 2

    workers.forEach((worker, index) => {
      const agent = agents.find((a) => a.id === worker.agent_id)
      const displayName = worker.alias || agent?.name || t("workforces.canvas.nodeTypes.worker")
      const workerId = `worker-${index}`
      newNodes.push({
        id: workerId,
        type: "worker",
        position: { x: startX + index * (workerWidth + gap), y: 250 },
        origin: [0.5, 0],
        data: {
          name: displayName,
          avatar: displayName.charAt(0).toUpperCase(),
          subtitle: agent?.description || "",
          description: worker.assignment_instructions,
        },
      })
      newEdges.push({
        id: `edge-manager-${workerId}`,
        source: "manager",
        target: workerId,
        type: "smoothstep",
        animated: true,
        style: { stroke: "#cbd5e1", strokeWidth: 2 },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#cbd5e1" },
      })
    })

    return { nodes: newNodes, edges: newEdges }
  }, [managerAgent, managerInstructions, workers, agents, t])

  return (
    <div className="h-full w-full rounded-xl border bg-gray-50/50">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.1}
        maxZoom={1.5}
      >
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#cbd5e1" />
        <Controls />
      </ReactFlow>
    </div>
  )
}

export function WorkforceCanvas({ workforce }: WorkforceCanvasProps) {
  const { t } = useI18n()
  const { nodes, edges } = useMemo(() => {
    const newNodes: Node[] = []
    const newEdges: Edge[] = []

    // Manager Node
    const manager = workforce?.manager
    newNodes.push({
      id: "manager",
      type: "manager",
      position: { x: 0, y: 0 },
      origin: [0.5, 0],
      data: {
        name: manager?.name || t("workforces.canvas.nodeTypes.manager"),
        avatar: manager?.name ? manager.name.charAt(0).toUpperCase() : "M",
        description: workforce?.manager_instructions || manager?.description || "",
      },
    })

    // Worker Nodes
    const workers = workforce?.workers || []
    const workerWidth = 256 // w-64 = 16rem = 256px
    const gap = 32
    const totalWidth = workers.length * workerWidth + (workers.length - 1) * gap
    const startX = -totalWidth / 2 + workerWidth / 2

    workers.forEach((worker, index) => {
      const workerId = `worker-${worker.id}`
      newNodes.push({
        id: workerId,
        type: "worker",
        position: { x: startX + index * (workerWidth + gap), y: 250 },
        origin: [0.5, 0],
        data: {
          name: worker.agent?.name || t("workforces.canvas.nodeTypes.worker"),
          avatar: worker.agent?.name ? worker.agent.name.charAt(0).toUpperCase() : "W",
          subtitle: worker.agent?.description || "",
          description: worker.assignment_instructions,
        },
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

    return { nodes: newNodes, edges: newEdges }
  }, [workforce, t])

  return (
    <div className="h-full w-full rounded-xl border bg-gray-50/50">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.1}
        maxZoom={1.5}
      >
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#cbd5e1" />
        <Controls />
      </ReactFlow>
    </div>
  )
}

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
import type { WorkforceDetail } from "@/types/workforce"

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
    <div className="flex w-80 flex-col items-center justify-center rounded-xl border-2 border-blue-500 bg-white p-6 shadow-sm">
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-600 text-xl font-bold text-white">
        {data.avatar}
      </div>
      <div className="mt-3 text-lg font-bold text-gray-900">{data.name}</div>
      <div className="mt-2 flex items-center gap-1 rounded-full bg-blue-50 px-2.5 py-0.5 text-xs font-medium text-blue-700">
        <Crown className="h-3.5 w-3.5" />
        {t("workforces.canvas.nodeTypes.manager")}
      </div>
      {data.description && (
        <div className="mt-4 text-center text-sm text-gray-500 line-clamp-3">
          {data.description}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} className="!border-none !bg-transparent" />
    </div>
  )
}

function WorkerNode({ data }: { data: NodeData }) {
  return (
    <div className="flex w-64 flex-col rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
      <Handle type="target" position={Position.Top} className="!border-none !bg-transparent" />
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-lg font-bold text-blue-600">
          {data.avatar}
        </div>
        <div className="min-w-0 flex-1">
          <div className="truncate font-bold text-gray-900">{data.name}</div>
          {data.subtitle && (
            <div className="truncate text-xs text-gray-500">{data.subtitle}</div>
          )}
        </div>
      </div>
      {data.description && (
        <div className="mt-4 text-sm italic text-gray-500 line-clamp-3">
          &ldquo;{data.description}&rdquo;
        </div>
      )}
    </div>
  )
}

const nodeTypes = {
  manager: ManagerNode,
  worker: WorkerNode,
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

"use client"

import React from "react"
import Link from "next/link"
import { AlertTriangle } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { useI18n } from "@/contexts/i18n-context"
import { canEditAgent } from "@/lib/agent-ui-access"
import { WorkforceDraftCanvas } from "./workforce-canvas"
import type {
  WorkforceAgentOption,
  WorkforceWorkerDraft,
} from "@/types/workforce"

interface ReviewStepProps {
  name: string
  description: string
  managerAgentId: string
  managerInstructions: string
  workers: WorkforceWorkerDraft[]
  agents: WorkforceAgentOption[]
  getAgentHref?: (agentId: number) => string
}

export function ReviewStep({
  name,
  description,
  managerAgentId,
  managerInstructions,
  workers,
  agents,
  getAgentHref = (agentId) => `/build/${agentId}`,
}: ReviewStepProps) {
  const { t } = useI18n()
  const manager = agents.find((agent) => String(agent.id) === managerAgentId)

  const warnings: string[] = []
  if (manager && manager.status !== "published") {
    warnings.push(t("workforces.review.warnings.managerNotPublished"))
  }
  for (const worker of workers) {
    const agent = agents.find((item) => item.id === worker.agent_id)
    if (agent && agent.status !== "published") {
      warnings.push(
        t("workforces.review.warnings.workerNotPublished", {
          name: worker.alias || agent.name,
        }),
      )
    }
    if (!worker.assignment_instructions.trim()) {
      warnings.push(
        t("workforces.review.warnings.missingInstructions", {
          name: worker.alias || t("workforces.workers.aWorker"),
        }),
      )
    }
  }

  return (
    <div className="space-y-6">
      {/* Workforce name summary */}
      <div className="rounded-lg border bg-muted/30 px-4 py-3">
        <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">{t("workforces.list.badge")}</div>
        <div className="font-medium">{name || t("workforces.review.untitled")}</div>
        {description && <div className="text-sm text-muted-foreground mt-0.5">{description}</div>}
      </div>

      {/* Canvas preview — real ReactFlow canvas */}
      {manager && (
        <div>
          <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">
            {t("workforces.review.canvasPreview")}
          </div>
          <div className="h-72 rounded-xl overflow-hidden border">
            <WorkforceDraftCanvas
              managerAgent={manager}
              managerInstructions={managerInstructions}
              workers={workers}
              agents={agents}
            />
          </div>
        </div>
      )}

      {warnings.length > 0 ? (
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-amber-900">
          <div className="flex items-center gap-2 font-medium">
            <AlertTriangle className="size-4" />
            {t("workforces.review.potentialRisks")}
          </div>
          <div className="mt-2 space-y-1 text-sm">
            {warnings.map((warning, index) => (
              <p key={`${warning}-${index}`}>{warning}</p>
            ))}
          </div>
        </div>
      ) : null}

      <div className="space-y-2">
        <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          {t("workforces.fields.manager")}
        </div>
        <div className="rounded-lg border bg-card p-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-blue-600 text-white font-medium">
              {manager?.name?.charAt(0).toUpperCase() || "M"}
            </div>
            <div>
              <div className="flex items-center gap-2 font-medium">
                <span>{manager?.name || t("workforces.common.notSelected")}</span>
                {manager ? (
                  <Badge variant="outline">{t(`workforces.status.${manager.status}`)}</Badge>
                ) : null}
                {manager && !canEditAgent(manager) ? (
                  <Badge variant="secondary">{t("workforces.actions.readOnly")}</Badge>
                ) : null}
              </div>
              <div className="text-sm text-muted-foreground">
                {manager?.description || t("workforces.fields.manager")}
              </div>
            </div>
          </div>
          {manager && canEditAgent(manager) ? (
            <Link
              href={getAgentHref(manager.id)}
              target="_blank"
              className="mt-3 inline-block text-sm text-primary hover:underline"
            >
              {t("workforces.actions.openAgentEditor")}
            </Link>
          ) : null}
        </div>
      </div>

      <div className="space-y-2">
        <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          {t("workforces.fields.managerInstructions")}
        </div>
        <div className="rounded-lg border bg-card p-4 max-h-48 overflow-y-auto">
          <div className="whitespace-pre-wrap text-sm text-muted-foreground">
            {managerInstructions || t("workforces.review.noManagerInstructions")}
          </div>
        </div>
      </div>

      <div className="space-y-2">
        <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          {t("workforces.review.subAgentsDelegation")}
        </div>
        <div className="space-y-3">
          {workers.length === 0 ? (
            <div className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
              {t("workforces.workers.noneConfigured")}
            </div>
          ) : (
            workers.map((worker, index) => {
              const agent = agents.find((item) => item.id === worker.agent_id)
              const title = worker.alias
                || agent?.name
                || t("workforces.workers.fallbackName", { index: index + 1 })

              return (
                <div key={`${worker.source_type}-${index}`} className="rounded-xl border overflow-hidden">
                  <div className="flex items-center gap-3 bg-muted/50 p-4 border-b">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-blue-600 text-white font-medium">
                      {title.charAt(0).toUpperCase()}
                    </div>
                    <div>
                      <div className="flex flex-wrap items-center gap-2">
                        <div className="font-medium">{title}</div>
                        <Badge variant="outline">
                          {t(`workforces.sourceTypes.${worker.source_type}`)}
                        </Badge>
                        {agent ? (
                          <Badge variant="secondary">{t(`workforces.status.${agent.status}`)}</Badge>
                        ) : null}
                        {agent && !canEditAgent(agent) ? (
                          <Badge variant="secondary">{t("workforces.actions.readOnly")}</Badge>
                        ) : null}
                      </div>
                      <div className="text-sm text-muted-foreground">
                        {agent?.description || t("workforces.workers.publishedAgent")}
                      </div>
                    </div>
                  </div>
                  <div className="p-4">
                    <div className="text-sm font-medium mb-2">
                      {t("workforces.fields.assignmentInstructions")}
                    </div>
                    <div className="rounded-lg border p-3 text-sm text-muted-foreground max-h-48 overflow-y-auto whitespace-pre-wrap">
                      {worker.assignment_instructions}
                    </div>
                  </div>
                </div>
              )
            })
          )}
        </div>
      </div>
    </div>
  )
}

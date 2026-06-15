"use client"

import React, { useMemo, useState } from "react"
import { ArrowDown, ArrowUp, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { useI18n } from "@/contexts/i18n-context"
import type { WorkforceAgentOption, WorkforceWorkerDraft } from "@/types/workforce"
import { InfoTooltip } from "../ui/tooltip"

interface WorkersStepProps {
  managerAgentId: string
  agents: WorkforceAgentOption[]
  workers: WorkforceWorkerDraft[]
  onWorkersChange: (workers: WorkforceWorkerDraft[]) => void
}

export function WorkersStep({
  managerAgentId,
  agents,
  workers,
  onWorkersChange,
}: WorkersStepProps) {
  const { t } = useI18n()
  const [draftAgentId, setDraftAgentId] = useState("")
  const [draftAlias, setDraftAlias] = useState("")
  const [draftInstructions, setDraftInstructions] = useState("")

  const selectableAgents = useMemo(() => {
    const workerAgentIds = new Set(workers.map((worker) => worker.agent_id))
    return agents.filter(
      (agent) =>
        String(agent.id) !== managerAgentId
        && !workerAgentIds.has(agent.id),
    )
  }, [agents, managerAgentId, workers])

  const addWorker = () => {
    if (!draftAgentId || !draftInstructions.trim()) return
    onWorkersChange([
      ...workers,
      {
        source_type: "existing",
        agent_id: Number(draftAgentId),
        alias: draftAlias.trim(),
        assignment_instructions: draftInstructions.trim(),
        enabled: true,
        sort_order: workers.length + 1,
      },
    ])
    setDraftAgentId("")
    setDraftAlias("")
    setDraftInstructions("")
  }

  const updateWorker = (index: number, nextWorker: WorkforceWorkerDraft) => {
    const next = [...workers]
    next[index] = nextWorker
    onWorkersChange(next)
  }

  const moveWorker = (index: number, direction: -1 | 1) => {
    const targetIndex = index + direction
    if (targetIndex < 0 || targetIndex >= workers.length) return
    const next = [...workers]
    const [worker] = next.splice(index, 1)
    next.splice(targetIndex, 0, worker)
    onWorkersChange(
      next.map((item, currentIndex) => ({
        ...item,
        sort_order: currentIndex + 1,
      })),
    )
  }

  const removeWorker = (index: number) => {
    const next = workers.filter((_, currentIndex) => currentIndex !== index)
    onWorkersChange(
      next.map((worker, currentIndex) => ({
        ...worker,
        sort_order: currentIndex + 1,
      })),
    )
  }

  const canAdd = Boolean(draftAgentId && draftInstructions.trim())

  return (
    <div className="space-y-6">
      {
        workers.map((worker, index) => {
          const agent = agents.find((item) => item.id === worker.agent_id)
          const title = worker.alias
            || agent?.name
            || t("workforces.workers.fallbackName", { index: index + 1 })

          return (
            <div key={`${worker.agent_id}-${index}`} className="rounded-xl border overflow-hidden">
              <div className="flex items-start justify-between gap-4 bg-muted/50 px-4 py-2 border-b">
                <div className="flex items-start gap-3">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-blue-600 text-white font-medium">
                    {title.charAt(0).toUpperCase()}
                  </div>
                  <div>
                    <div className="font-medium">{title}</div>
                    <div className="text-sm text-muted-foreground">
                      {agent?.description || t("workforces.workers.defaultDescription")}
                    </div>
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => moveWorker(index, -1)}
                    disabled={index === 0}
                    aria-label={t("workforces.actions.up")}
                  >
                    <ArrowUp className="size-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => moveWorker(index, 1)}
                    disabled={index === workers.length - 1}
                    aria-label={t("workforces.actions.down")}
                  >
                    <ArrowDown className="size-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-muted-foreground hover:text-destructive"
                    onClick={() => removeWorker(index)}
                    aria-label={t("workforces.actions.remove")}
                  >
                    <X className="size-4" />
                  </Button>
                </div>
              </div>
              <div className="p-4 space-y-4">
                <div className="flex gap-4 items-end">
                  <div className="space-y-2 flex-1">
                    <Label>{t("workforces.fields.alias")}</Label>
                    <Input
                      value={worker.alias || ""}
                      onChange={(event) =>
                        updateWorker(index, { ...worker, alias: event.target.value })}
                    />
                  </div>
                  <div className="flex h-10 items-center justify-between rounded-lg border px-4 min-w-[140px]">
                    <div className="flex items-center gap-2">
                      <div className="font-medium">{t("workforces.fields.enabled")}</div>
                      <InfoTooltip content={t("workforces.workers.disabledHelp")} />
                    </div>
                    <Switch
                      checked={worker.enabled}
                      onCheckedChange={(checked) =>
                        updateWorker(index, { ...worker, enabled: checked })}
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>{t("workforces.fields.assignmentInstructions")}</Label>
                  <Textarea
                    value={worker.assignment_instructions}
                    onChange={(event) =>
                      updateWorker(index, {
                        ...worker,
                        assignment_instructions: event.target.value,
                      })}
                    rows={3}
                  />
                </div>
              </div>
            </div>
          )
        })
      }
      <Card>
        <CardHeader>
          <CardTitle>{t("workforces.workers.addTitle")}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>{t("workforces.fields.publishedAgent")}</Label>
            <Select
              value={draftAgentId}
              onValueChange={setDraftAgentId}
              placeholder={t("workforces.workers.chooseAgent")}
              options={selectableAgents.map((agent) => ({
                value: String(agent.id),
                label: agent.name,
                description: agent.description || undefined,
              }))}
            />
          </div>
          <div className="space-y-2">
            <Label>{t("workforces.fields.alias")}</Label>
            <Input
              value={draftAlias}
              onChange={(event) => setDraftAlias(event.target.value)}
              placeholder={t("workforces.workers.aliasPlaceholder")}
            />
          </div>
          <div className="space-y-2">
            <Label>{t("workforces.fields.assignmentInstructions")}</Label>
            <Textarea
              value={draftInstructions}
              onChange={(event) => setDraftInstructions(event.target.value)}
              placeholder={t("workforces.workers.instructionsPlaceholder")}
              rows={3}
            />
          </div>
          <Button onClick={addWorker} disabled={!canAdd}>
            {t("workforces.actions.addWorker")}
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}

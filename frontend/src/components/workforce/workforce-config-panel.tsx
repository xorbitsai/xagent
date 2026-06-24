"use client"

import Link from "next/link"
import React from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { useI18n } from "@/contexts/i18n-context"
import type { WorkforceDetail, WorkforceWorker } from "@/types/workforce"
import { canEditAgent } from "@/lib/agent-ui-access"

export interface WorkerEditState {
  alias: string
  assignment_instructions: string
  enabled: boolean
  sort_order: string
}

interface WorkforceConfigPanelProps {
  workforce: WorkforceDetail
  name: string
  description: string
  managerAgentId: string
  managerInstructions: string
  managerOptions: Array<{ value: string; label: string; description?: string }>
  workerOptions: Array<{ value: string; label: string; description?: string }>
  workerEdits: Record<number, WorkerEditState>
  newWorkerAgentId: string
  newWorkerAlias: string
  newWorkerInstructions: string
  isArchived: boolean
  saving: boolean
  onNameChange: (value: string) => void
  onDescriptionChange: (value: string) => void
  onManagerAgentIdChange: (value: string) => void
  onManagerInstructionsChange: (value: string) => void
  onSaveWorkforce: () => void
  onNewWorkerAgentIdChange: (value: string) => void
  onNewWorkerAliasChange: (value: string) => void
  onNewWorkerInstructionsChange: (value: string) => void
  onAddWorker: () => void
  onWorkerEditChange: (workerId: number, edit: Partial<WorkerEditState>) => void
  onSaveWorker: (worker: WorkforceWorker) => void
  onRemoveWorker: (workerId: number) => void
  onPublish: () => void
  onUnpublish: () => void
  onArchive: () => void
}

export function WorkforceConfigPanel({
  workforce,
  name,
  description,
  managerAgentId,
  managerInstructions,
  managerOptions,
  workerOptions,
  workerEdits,
  newWorkerAgentId,
  newWorkerAlias,
  newWorkerInstructions,
  isArchived,
  saving,
  onNameChange,
  onDescriptionChange,
  onManagerAgentIdChange,
  onManagerInstructionsChange,
  onSaveWorkforce,
  onNewWorkerAgentIdChange,
  onNewWorkerAliasChange,
  onNewWorkerInstructionsChange,
  onAddWorker,
  onWorkerEditChange,
  onSaveWorker,
  onRemoveWorker,
  onPublish,
  onUnpublish,
  onArchive,
}: WorkforceConfigPanelProps) {
  const { t } = useI18n()

  return (
    <div className="flex flex-col gap-6 p-6 h-full overflow-y-auto">
      {/* Header with actions */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold">{workforce.name}</h1>
          <p className="text-sm text-muted-foreground">
            {t("workforces.detail.description")}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {workforce.status === "draft" ? (
            <Button size="sm" onClick={onPublish} disabled={saving}>
              {saving ? t("workforces.loading.saving") : t("workforces.actions.publish")}
            </Button>
          ) : null}
          {workforce.status === "active" ? (
            <Button size="sm" variant="outline" onClick={onUnpublish} disabled={saving}>
              {saving ? t("workforces.loading.saving") : t("workforces.actions.unpublish")}
            </Button>
          ) : null}
          <Link href={`/workforces/${workforce.id}/canvas`}>
            <Button size="sm" variant="outline">{t("workforces.actions.canvas")}</Button>
          </Link>
          {!isArchived ? (
            <Button size="sm" variant="outline" onClick={onArchive} disabled={saving}>
              {t("workforces.actions.archive")}
            </Button>
          ) : null}
        </div>
      </div>

      {/* Basic Config */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">{t("workforces.detail.editTitle")}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>{t("workforces.fields.name")}</Label>
            <Input
              value={name}
              onChange={(event) => onNameChange(event.target.value)}
              disabled={isArchived}
            />
          </div>
          <div className="space-y-2">
            <Label>{t("workforces.fields.description")}</Label>
            <Textarea
              value={description}
              onChange={(event) => onDescriptionChange(event.target.value)}
              rows={2}
              disabled={isArchived}
            />
          </div>
          <div className="space-y-2">
            <Label>{t("workforces.fields.manager")}</Label>
            <Select
              value={managerAgentId}
              onValueChange={onManagerAgentIdChange}
              options={managerOptions}
              disabled={isArchived}
            />
          </div>
          <div className="space-y-2">
            <Label>{t("workforces.fields.managerInstructions")}</Label>
            <Textarea
              value={managerInstructions}
              onChange={(event) => onManagerInstructionsChange(event.target.value)}
              rows={4}
              disabled={isArchived}
            />
          </div>
          <Button
            onClick={onSaveWorkforce}
            disabled={saving || isArchived || !name.trim() || !managerAgentId}
            size="sm"
          >
            {saving ? t("workforces.loading.saving") : t("workforces.actions.saveWorkforce")}
          </Button>
        </CardContent>
      </Card>

      {/* Add Worker */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">{t("workforces.workers.addTitle")}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-2">
            <Label>{t("workforces.fields.publishedAgent")}</Label>
            <Select
              value={newWorkerAgentId}
              onValueChange={onNewWorkerAgentIdChange}
              placeholder={t("workforces.workers.chooseAgent")}
              options={workerOptions}
              disabled={isArchived}
            />
          </div>
          <div className="space-y-2">
            <Label>{t("workforces.fields.alias")}</Label>
            <Input
              value={newWorkerAlias}
              onChange={(event) => onNewWorkerAliasChange(event.target.value)}
              placeholder={t("workforces.workers.aliasPlaceholder")}
              disabled={isArchived}
            />
          </div>
          <div className="space-y-2">
            <Label>{t("workforces.fields.assignmentInstructions")}</Label>
            <Textarea
              value={newWorkerInstructions}
              onChange={(event) => onNewWorkerInstructionsChange(event.target.value)}
              rows={3}
              disabled={isArchived}
            />
          </div>
          <Button
            onClick={onAddWorker}
            disabled={saving || isArchived || !newWorkerAgentId || !newWorkerInstructions.trim()}
            size="sm"
          >
            {t("workforces.actions.addWorker")}
          </Button>
        </CardContent>
      </Card>

      {/* Manage Workers */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">{t("workforces.workers.manageTitle")}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {workforce.workers.length === 0 ? (
            <div className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
              {t("workforces.workers.noneConfigured")}
            </div>
          ) : (
            workforce.workers
              .slice()
              .sort((a, b) => (a.sort_order ?? 0) - (b.sort_order ?? 0))
              .map((worker) => {
                const edit = workerEdits[worker.id] || {
                  alias: worker.alias || "",
                  assignment_instructions: worker.assignment_instructions || "",
                  enabled: worker.enabled,
                  sort_order: String(worker.sort_order ?? 1),
                }
                return (
                  <div key={worker.id} className="rounded-xl border overflow-hidden">
                    <div className="flex items-center justify-between gap-4 bg-muted/50 p-4 border-b">
                      <div>
                        <div className="font-medium">
                          {worker.alias || worker.agent.name}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {worker.agent.name} · {t(`workforces.status.${worker.agent.status}`)}
                        </div>
                      </div>
                      <div className="flex gap-2">
                        {canEditAgent(worker.agent) ? (
                          <Link href={`/build/${worker.agent.id}`} target="_blank">
                            <Button variant="outline" size="sm">
                              {t("workforces.actions.openAgent")}
                            </Button>
                          </Link>
                        ) : null}
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => onRemoveWorker(worker.id)}
                          disabled={saving || isArchived}
                        >
                          {t("workforces.actions.remove")}
                        </Button>
                      </div>
                    </div>
                    <div className="p-4 space-y-4">
                      <div className="grid gap-4 md:grid-cols-[1fr_140px_140px]">
                        <div className="space-y-2">
                          <Label>{t("workforces.fields.alias")}</Label>
                          <Input
                            value={edit.alias}
                            onChange={(event) =>
                              onWorkerEditChange(worker.id, { alias: event.target.value })
                            }
                            disabled={isArchived}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label>{t("workforces.fields.order")}</Label>
                          <Input
                            type="number"
                            min={1}
                            step={1}
                            value={edit.sort_order}
                            onChange={(event) =>
                              onWorkerEditChange(worker.id, { sort_order: event.target.value })
                            }
                            disabled={isArchived}
                          />
                        </div>
                        <div className="flex items-center justify-between rounded-lg border px-3 py-2">
                          <div className="font-medium text-sm">{t("workforces.fields.enabled")}</div>
                          <Switch
                            checked={edit.enabled}
                            onCheckedChange={(checked) =>
                              onWorkerEditChange(worker.id, { enabled: checked })
                            }
                            disabled={isArchived}
                          />
                        </div>
                      </div>
                      <div className="space-y-2">
                        <Label>{t("workforces.fields.assignmentInstructions")}</Label>
                        <Textarea
                          value={edit.assignment_instructions}
                          onChange={(event) =>
                            onWorkerEditChange(worker.id, {
                              assignment_instructions: event.target.value,
                            })
                          }
                          rows={3}
                          disabled={isArchived}
                        />
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => onSaveWorker(worker)}
                        disabled={saving || isArchived || !edit.assignment_instructions.trim()}
                      >
                        {t("workforces.actions.saveWorker")}
                      </Button>
                    </div>
                  </div>
                )
              })
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function workerEditState(worker: WorkforceWorker): WorkerEditState {
  return {
    alias: worker.alias || "",
    assignment_instructions: worker.assignment_instructions || "",
    enabled: worker.enabled,
    sort_order: String(worker.sort_order ?? 1),
  }
}

function buildWorkerEditState(workers: WorkforceWorker[]): Record<number, WorkerEditState> {
  return workers.reduce<Record<number, WorkerEditState>>((accumulator, worker) => {
    accumulator[worker.id] = workerEditState(worker)
    return accumulator
  }, {})
}

function normalizeWorkerSortOrder(value: string, fallback: number | null | undefined): number {
  const normalized = value.trim()
  const parsed = /^\d+$/.test(normalized) ? Number.parseInt(normalized, 10) : NaN
  if (Number.isInteger(parsed) && parsed > 0) {
    return parsed
  }
  return fallback ?? 1
}

export { buildWorkerEditState, normalizeWorkerSortOrder, workerEditState }

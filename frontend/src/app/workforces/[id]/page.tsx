"use client"

import Link from "next/link"
import React, { useCallback, useEffect, useMemo, useState } from "react"
import { useParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { useI18n } from "@/contexts/i18n-context"
import {
  addWorkforceAgent,
  archiveWorkforce,
  getWorkforce,
  listAgentOptions,
  publishWorkforce,
  removeWorkforceAgent,
  unpublishWorkforce,
  updateWorkforce,
  updateWorkforceAgent,
} from "@/lib/workforces-api"
import type {
  WorkforceAgentOption,
  WorkforceDetail,
  WorkforceWorker,
} from "@/types/workforce"
import { WorkforceSummary } from "@/components/workforce"
import { canEditAgent } from "@/lib/agent-ui-access"
import { getRunDisabledReason } from "../workforce-ui-state"
import { toast } from "sonner"

interface WorkerEditState {
  alias: string
  assignment_instructions: string
  enabled: boolean
  sort_order: string
}

interface LoadOptions {
  silent?: boolean
}

interface SyncFormOptions {
  preserveEditableState?: boolean
}

function workerEditState(worker: WorkforceWorker): WorkerEditState {
  return {
    alias: worker.alias || "",
    assignment_instructions: worker.assignment_instructions,
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

export default function WorkforceDetailPage() {
  const { t } = useI18n()
  const params = useParams()
  const id = Array.isArray(params.id) ? params.id[0] : params.id
  const [workforce, setWorkforce] = useState<WorkforceDetail | null>(null)
  const [agents, setAgents] = useState<WorkforceAgentOption[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [managerAgentId, setManagerAgentId] = useState("")
  const [managerInstructions, setManagerInstructions] = useState("")
  const [workerEdits, setWorkerEdits] = useState<Record<number, WorkerEditState>>({})
  const [newWorkerAgentId, setNewWorkerAgentId] = useState("")
  const [newWorkerAlias, setNewWorkerAlias] = useState("")
  const [newWorkerInstructions, setNewWorkerInstructions] = useState("")

  const publishedAgents = useMemo(
    () => agents.filter((agent) => agent.status === "published"),
    [agents],
  )
  const isArchived = workforce?.status === "archived"

  const syncForm = useCallback((
    nextWorkforce: WorkforceDetail,
    options: SyncFormOptions = {},
  ) => {
    if (!options.preserveEditableState) {
      setName(nextWorkforce.name)
      setDescription(nextWorkforce.description || "")
      setManagerAgentId(String(nextWorkforce.manager.id))
      setManagerInstructions(nextWorkforce.manager_instructions || "")
      setWorkerEdits(buildWorkerEditState(nextWorkforce.workers))
      return
    }

    const serverWorkerEdits = buildWorkerEditState(nextWorkforce.workers)
    setWorkerEdits((current) =>
      nextWorkforce.workers.reduce<Record<number, WorkerEditState>>(
        (accumulator, worker) => {
          accumulator[worker.id] = current[worker.id] ?? serverWorkerEdits[worker.id]
          return accumulator
        },
        {},
      ),
    )
  }, [])

  const load = useCallback(async (options: LoadOptions = {}) => {
    if (!id) return
    const { silent = false } = options
    try {
      if (!silent) {
        setLoading(true)
      }
      setError(null)
      const [workforceData, agentData] = await Promise.all([
        getWorkforce(id),
        listAgentOptions(),
      ])
      setWorkforce(workforceData)
      setAgents(agentData)
      syncForm(workforceData, { preserveEditableState: silent })
    } catch (err) {
      const nextError = err instanceof Error ? err.message : t("workforces.errors.load")
      setError(nextError)
      toast.error(nextError)
    } finally {
      if (!silent) {
        setLoading(false)
      }
    }
  }, [id, syncForm, t])

  useEffect(() => {
    void load()
  }, [load])

  const managerOptions = useMemo(() => {
    const options = publishedAgents
      .filter((agent) => !workforce?.workers.some((worker) => worker.agent.id === agent.id))
      .map((agent) => ({
        value: String(agent.id),
        label: agent.name,
        description: agent.description || undefined,
      }))

    const currentManager = workforce?.manager
    if (
      currentManager?.status === "published" &&
      !options.some((option) => option.value === String(currentManager.id))
    ) {
      options.unshift({
        value: String(currentManager.id),
        label: currentManager.name,
        description: currentManager.description || undefined,
      })
    }

    return options
  }, [publishedAgents, workforce])

  const workerOptions = publishedAgents
    .filter(
      (agent) =>
        String(agent.id) !== managerAgentId &&
        !workforce?.workers.some((worker) => worker.agent.id === agent.id),
    )
    .map((agent) => ({
      value: String(agent.id),
      label: agent.name,
      description: agent.description || undefined,
    }))

  const beginMutation = () => {
    setSaving(true)
    setError(null)
    setMessage(null)
  }

  const saveWorkforce = async () => {
    if (!id || !name.trim() || !managerAgentId) return
    try {
      beginMutation()
      const next = await updateWorkforce(id, {
        name: name.trim(),
        description: description.trim() || null,
        manager_agent_id: Number(managerAgentId),
        manager_instructions: managerInstructions.trim() || null,
      })
      setWorkforce(next)
      syncForm(next)
      setMessage(t("workforces.messages.updated"))
    } catch (err) {
      const nextError = err instanceof Error ? err.message : t("workforces.errors.update")
      setError(nextError)
      toast.error(nextError)
    } finally {
      setSaving(false)
    }
  }

  const addWorker = async () => {
    if (!id || !newWorkerAgentId || !newWorkerInstructions.trim()) return
    try {
      beginMutation()
      await addWorkforceAgent(id, {
        source_type: "existing",
        agent_id: Number(newWorkerAgentId),
        alias: newWorkerAlias.trim() || undefined,
        assignment_instructions: newWorkerInstructions.trim(),
        enabled: true,
        sort_order: (workforce?.workers.length || 0) + 1,
      })
      setNewWorkerAgentId("")
      setNewWorkerAlias("")
      setNewWorkerInstructions("")
      await load({ silent: true })
      setMessage(t("workforces.messages.workerAdded"))
    } catch (err) {
      const nextError = err instanceof Error ? err.message : t("workforces.errors.addWorker")
      setError(nextError)
      toast.error(nextError)
    } finally {
      setSaving(false)
    }
  }

  const saveWorker = async (worker: WorkforceWorker) => {
    if (!id) return
    const edit = workerEdits[worker.id] ?? workerEditState(worker)
    if (!edit.assignment_instructions.trim()) return
    try {
      beginMutation()
      const updated = await updateWorkforceAgent(id, worker.id, {
        alias: edit.alias.trim() || null,
        assignment_instructions: edit.assignment_instructions.trim(),
        enabled: edit.enabled,
        sort_order: normalizeWorkerSortOrder(edit.sort_order, worker.sort_order),
      })
      setWorkforce((current) =>
        current
          ? {
            ...current,
            workers: current.workers.map((item) =>
              item.id === updated.id ? updated : item,
            ),
          }
          : current,
      )
      setWorkerEdits((current) => ({
        ...current,
        [updated.id]: workerEditState(updated),
      }))
      setMessage(t("workforces.messages.workerUpdated"))
    } catch (err) {
      const nextError = err instanceof Error ? err.message : t("workforces.errors.updateWorker")
      setError(nextError)
      toast.error(nextError)
    } finally {
      setSaving(false)
    }
  }

  const removeWorker = async (workerId: number) => {
    if (!id) return
    try {
      beginMutation()
      await removeWorkforceAgent(id, workerId)
      await load({ silent: true })
      setMessage(t("workforces.messages.workerRemoved"))
    } catch (err) {
      const nextError = err instanceof Error ? err.message : t("workforces.errors.removeWorker")
      setError(nextError)
      toast.error(nextError)
    } finally {
      setSaving(false)
    }
  }

  const publishCurrentWorkforce = async () => {
    if (!id) return
    try {
      beginMutation()
      const next = await publishWorkforce(id)
      setWorkforce(next)
      syncForm(next)
      setMessage(t("workforces.messages.published"))
    } catch (err) {
      const nextError = err instanceof Error ? err.message : t("workforces.errors.publish")
      setError(nextError)
      toast.error(nextError)
    } finally {
      setSaving(false)
    }
  }

  const unpublishCurrentWorkforce = async () => {
    if (!id) return
    try {
      beginMutation()
      const next = await unpublishWorkforce(id)
      setWorkforce(next)
      syncForm(next)
      setMessage(t("workforces.messages.unpublished"))
    } catch (err) {
      const nextError = err instanceof Error ? err.message : t("workforces.errors.unpublish")
      setError(nextError)
      toast.error(nextError)
    } finally {
      setSaving(false)
    }
  }

  const archiveCurrentWorkforce = async () => {
    if (!id) return
    try {
      beginMutation()
      await archiveWorkforce(id)
      const next = await getWorkforce(id)
      setWorkforce(next)
      syncForm(next)
      setMessage(t("workforces.messages.archived"))
    } catch (err) {
      const nextError = err instanceof Error ? err.message : t("workforces.errors.archive")
      setError(nextError)
      toast.error(nextError)
    } finally {
      setSaving(false)
    }
  }

  if (loading) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">{t("workforces.loading.detail")}</div>
  if (error && !workforce) return <div className="h-full overflow-y-auto p-4 text-red-500 sm:p-8">{error}</div>
  if (!workforce) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">{t("workforces.errors.notFound")}</div>

  const runDisabledReason = getRunDisabledReason(workforce.status, t)

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto flex w-full flex-col gap-6 p-4 sm:p-8">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold">{workforce.name}</h1>
            <p className="mt-2 text-muted-foreground">
              {t("workforces.detail.description")}
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            {workforce.status === "draft" ? (
              <Button onClick={() => void publishCurrentWorkforce()} disabled={saving}>
                {saving ? t("workforces.loading.saving") : t("workforces.actions.publish")}
              </Button>
            ) : null}
            {workforce.status === "active" ? (
              <Button
                variant="outline"
                onClick={() => void unpublishCurrentWorkforce()}
                disabled={saving}
              >
                {saving ? t("workforces.loading.saving") : t("workforces.actions.unpublish")}
              </Button>
            ) : null}
            <Link href={`/workforces/${workforce.id}/builder`}>
              <Button variant="outline">{t("workforces.actions.builder")}</Button>
            </Link>
            <Link href={`/workforces/${workforce.id}/canvas`}>
              <Button variant="outline">{t("workforces.actions.canvas")}</Button>
            </Link>
            <div className="flex flex-col gap-1">
              {runDisabledReason ? (
                <Button disabled>{t("workforces.actions.runWorkforce")}</Button>
              ) : (
                <Link href={`/workforces/${workforce.id}/run`}>
                  <Button>{t("workforces.actions.runWorkforce")}</Button>
                </Link>
              )}
              {runDisabledReason ? (
                <span className="max-w-48 text-xs text-muted-foreground">{runDisabledReason}</span>
              ) : null}
            </div>
            {!isArchived ? (
              <Button
                variant="outline"
                onClick={() => void archiveCurrentWorkforce()}
                disabled={saving}
              >
                {t("workforces.actions.archive")}
              </Button>
            ) : null}
          </div>
        </div>

        {error ? <div className="text-sm text-red-500">{error}</div> : null}
        {message ? <div className="text-sm text-emerald-600">{message}</div> : null}

        <div className="grid gap-6 lg:grid-cols-[1fr_0.9fr]">
          <Card>
            <CardHeader>
              <CardTitle>{t("workforces.detail.editTitle")}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>{t("workforces.fields.name")}</Label>
                <Input
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  disabled={isArchived}
                />
              </div>
              <div className="space-y-2">
                <Label>{t("workforces.fields.description")}</Label>
                <Textarea
                  value={description}
                  onChange={(event) => setDescription(event.target.value)}
                  rows={3}
                  disabled={isArchived}
                />
              </div>
              <div className="space-y-2">
                <Label>{t("workforces.fields.manager")}</Label>
                <Select
                  value={managerAgentId}
                  onValueChange={setManagerAgentId}
                  options={managerOptions}
                  disabled={isArchived}
                />
              </div>
              <div className="space-y-2">
                <Label>{t("workforces.fields.managerInstructions")}</Label>
                <Textarea
                  value={managerInstructions}
                  onChange={(event) => setManagerInstructions(event.target.value)}
                  rows={5}
                  disabled={isArchived}
                />
              </div>
              <Button
                onClick={saveWorkforce}
                disabled={saving || isArchived || !name.trim() || !managerAgentId}
              >
                {saving ? t("workforces.loading.saving") : t("workforces.actions.saveWorkforce")}
              </Button>
            </CardContent>
          </Card>

          <WorkforceSummary workforce={workforce} />
        </div>

        <Card>
          <CardHeader>
            <CardTitle>{t("workforces.workers.addTitle")}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>{t("workforces.fields.publishedAgent")}</Label>
              <Select
                value={newWorkerAgentId}
                onValueChange={setNewWorkerAgentId}
                placeholder={t("workforces.workers.chooseAgent")}
                options={workerOptions}
                disabled={isArchived}
              />
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label>{t("workforces.fields.alias")}</Label>
                <Input
                  value={newWorkerAlias}
                  onChange={(event) => setNewWorkerAlias(event.target.value)}
                  placeholder={t("workforces.workers.aliasPlaceholder")}
                  disabled={isArchived}
                />
              </div>
              <div className="space-y-2">
                <Label>{t("workforces.fields.assignmentInstructions")}</Label>
                <Textarea
                  value={newWorkerInstructions}
                  onChange={(event) => setNewWorkerInstructions(event.target.value)}
                  rows={3}
                  disabled={isArchived}
                />
              </div>
            </div>
            <Button
              onClick={addWorker}
              disabled={saving || isArchived || !newWorkerAgentId || !newWorkerInstructions.trim()}
            >
              {t("workforces.actions.addWorker")}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>{t("workforces.workers.manageTitle")}</CardTitle>
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
                  const edit = workerEdits[worker.id] || workerEditState(worker)
                  return (
                    <div key={worker.id} className="rounded-xl border p-4">
                      <div className="flex flex-wrap items-start justify-between gap-4">
                        <div>
                          <div className="font-medium">
                            {worker.alias || worker.agent.name}
                          </div>
                          <div className="text-sm text-muted-foreground">
                            {worker.agent.name} · {t(`workforces.status.${worker.agent.status}`)}
                          </div>
                        </div>
                        <div className="flex flex-wrap gap-2">
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
                            onClick={() => void removeWorker(worker.id)}
                            disabled={saving || isArchived}
                          >
                            {t("workforces.actions.remove")}
                          </Button>
                        </div>
                      </div>
                      <div className="mt-4 grid gap-4 md:grid-cols-[1fr_140px_140px]">
                        <div className="space-y-2">
                          <Label>{t("workforces.fields.alias")}</Label>
                          <Input
                            value={edit.alias}
                            onChange={(event) =>
                              setWorkerEdits((current) => ({
                                ...current,
                                [worker.id]: { ...edit, alias: event.target.value },
                              }))
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
                              setWorkerEdits((current) => ({
                                ...current,
                                [worker.id]: {
                                  ...edit,
                                  sort_order: event.target.value,
                                },
                              }))
                            }
                            disabled={isArchived}
                          />
                        </div>
                        <div className="flex items-center justify-between rounded-lg border px-3 py-2">
                          <div className="font-medium">{t("workforces.fields.enabled")}</div>
                          <Switch
                            checked={edit.enabled}
                            onCheckedChange={(checked) =>
                              setWorkerEdits((current) => ({
                                ...current,
                                [worker.id]: { ...edit, enabled: checked },
                              }))
                            }
                            disabled={isArchived}
                          />
                        </div>
                      </div>
                      <div className="mt-4 space-y-2">
                        <Label>{t("workforces.fields.assignmentInstructions")}</Label>
                        <Textarea
                          value={edit.assignment_instructions}
                          onChange={(event) =>
                            setWorkerEdits((current) => ({
                              ...current,
                              [worker.id]: {
                                ...edit,
                                assignment_instructions: event.target.value,
                              },
                            }))
                          }
                          rows={4}
                          disabled={isArchived}
                        />
                      </div>
                      <Button
                        className="mt-4"
                        variant="outline"
                        onClick={() => void saveWorker(worker)}
                        disabled={saving || isArchived || !edit.assignment_instructions.trim()}
                      >
                        {t("workforces.actions.saveWorker")}
                      </Button>
                    </div>
                  )
                })
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

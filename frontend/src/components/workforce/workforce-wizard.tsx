"use client"

import React, { useEffect, useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Stepper } from "@/components/ui/stepper"
import { useI18n } from "@/contexts/i18n-context"
import {
  createWorkforce,
  listAgentOptions,
} from "@/lib/workforces-api"
import type {
  WorkforceAgentOption,
  WorkforceDetail,
  WorkforceWorkerDraft,
} from "@/types/workforce"
import { toast } from "sonner"
import { ManagerStep } from "./manager-step"
import { ReviewStep } from "./review-step"
import { WorkersStep } from "./workers-step"


interface WorkforceWizardProps {
  onCreated: (workforce: WorkforceDetail) => void
  onBack?: () => void
}

export function WorkforceWizard({
  onCreated,
  onBack,
}: WorkforceWizardProps) {
  const { t } = useI18n()
  const [step, setStep] = useState(0)
  const [loadingAgents, setLoadingAgents] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [agents, setAgents] = useState<WorkforceAgentOption[]>([])

  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [managerAgentId, setManagerAgentId] = useState("")
  const [managerInstructions, setManagerInstructions] = useState("")
  const [workers, setWorkers] = useState<WorkforceWorkerDraft[]>([])

  const managerWorkerConflict = useMemo(() => {
    if (!managerAgentId) return null
    const managerId = Number(managerAgentId)
    return workers.find((worker) => worker.agent_id === managerId) ?? null
  }, [managerAgentId, workers])

  const managerWorkerConflictMessage = useMemo(() => {
    if (!managerWorkerConflict) return null
    const agent = agents.find((item) => item.id === managerWorkerConflict.agent_id)
    return t("workforces.review.warnings.managerCannotBeWorker", {
      name:
        managerWorkerConflict.alias
        || agent?.name
        || t("workforces.workers.aWorker"),
    })
  }, [agents, managerWorkerConflict, t])

  const workersAreValid = useMemo(
    () =>
      workers.length > 0
      && !managerWorkerConflict
      && workers.every((worker) => {
        if (!worker.assignment_instructions.trim()) return false
        return Boolean(worker.agent_id)
      }),
    [managerWorkerConflict, workers],
  )

  useEffect(() => {
    const loadAgents = async () => {
      try {
        setLoadingAgents(true)
        const agentData = await listAgentOptions()
        setAgents(agentData.filter((agent) => agent.status === "published"))
      } catch (err) {
        const nextError = err instanceof Error ? err.message : t("workforces.errors.loadAgents")
        toast.error(nextError)
      } finally {
        setLoadingAgents(false)
      }
    }
    void loadAgents()
  }, [t])

  const canMoveForward = useMemo(() => {
    if (step === 0) {
      return Boolean(name.trim() && managerAgentId)
    }
    if (step === 1) {
      return workersAreValid
    }
    return true
  }, [step, name, managerAgentId, workersAreValid])

  const handleCreate = async () => {
    if (!name.trim() || !managerAgentId || !workersAreValid) return
    setSubmitting(true)
    try {
      const workforce = await createWorkforce({
        name: name.trim(),
        description: description.trim() || undefined,
        manager_agent_id: Number(managerAgentId),
        manager_instructions: managerInstructions.trim() || undefined,
        workers: workers.map((worker) => ({
          source_type: worker.source_type,
          agent_id: worker.agent_id,
          alias: worker.alias.trim() || undefined,
          assignment_instructions: worker.assignment_instructions.trim(),
          enabled: worker.enabled,
          sort_order: worker.sort_order,
          canvas_position: worker.canvas_position,
        })),
      })
      onCreated(workforce)
    } catch (err) {
      const nextError = err instanceof Error ? err.message : t("workforces.errors.create")
      toast.error(nextError)
    } finally {
      setSubmitting(false)
    }
  }

  const handleBack = () => {
    setStep((current) => Math.max(0, current - 1))
  }

  return (
    <div className="flex w-full flex-col gap-6">
      <Stepper
        currentStep={step + 1}
        contentClassName={step === 0 ? "overflow-visible" : undefined}
        steps={[
          {
            label: t("workforces.create.steps.nameAndManager"),
            content: (
              <div className="flex flex-col gap-6">
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label>{t("workforces.create.fields.workforceName")} <span className="text-destructive">*</span></Label>
                    <Input
                      value={name}
                      onChange={(event) => setName(event.target.value)}
                      placeholder={t("workforces.create.placeholders.name")}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>{t("workforces.fields.description")} <span className="text-muted-foreground text-xs font-normal">{t("common.optional")}</span></Label>
                    <Textarea
                      value={description}
                      onChange={(event) => setDescription(event.target.value)}
                      placeholder={t("workforces.create.placeholders.description")}
                      rows={3}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>{t("workforces.create.fields.workforceInstructions")} <span className="text-destructive">*</span></Label>
                    <p className="text-xs text-muted-foreground">{t("workforces.create.fields.workforceInstructionsHint")}</p>
                    <Textarea
                      value={managerInstructions}
                      onChange={(event) => setManagerInstructions(event.target.value)}
                      placeholder={t("workforces.create.placeholders.managerInstructions")}
                      rows={4}
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>{t("workforces.create.fields.agentManager")} <span className="text-destructive">*</span></Label>
                  <p className="text-xs text-muted-foreground">{t("workforces.create.fields.agentManagerHint")}</p>
                  <ManagerStep
                    managerAgentId={managerAgentId}
                    onManagerAgentIdChange={setManagerAgentId}
                    agents={agents}
                    loadingAgents={loadingAgents}
                  />
                </div>
              </div>
            ),
          },
          {
            label: t("workforces.create.steps.subAgents"),
            content: (
              <WorkersStep
                managerAgentId={managerAgentId}
                agents={agents}
                workers={workers}
                onWorkersChange={setWorkers}
                onAgentCreated={(agent) => setAgents((prev) => [...prev, agent])}
              />
            ),
          },
          {
            label: t("workforces.create.steps.review"),
            content: (
              <ReviewStep
                name={name}
                description={description}
                managerAgentId={managerAgentId}
                managerInstructions={managerInstructions}
                workers={workers}
                agents={agents}
              />
            ),
          },
        ]}
      />

      {managerWorkerConflictMessage ? (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          {managerWorkerConflictMessage}
        </div>
      ) : null}

      <div className="flex items-center justify-between mt-4">
        {step > 0 || onBack ? (
          <Button
            variant="outline"
            onClick={() => {
              if (step === 0 && onBack) {
                onBack()
              } else {
                handleBack()
              }
            }}
            disabled={submitting}
          >
            {step === 0 ? t("common.cancel") : t("common.back")}
          </Button>
        ) : (
          <div />
        )}
        <div className="flex items-center gap-3">
          {step < 2 ? (
            <Button
              onClick={() => setStep((current) => current + 1)}
              disabled={!canMoveForward}
            >
              {t("common.next")}
            </Button>
          ) : (
            <Button
              onClick={handleCreate}
              disabled={submitting || !canMoveForward || !workersAreValid}
              className="bg-blue-600 hover:bg-blue-700 text-white"
            >
              {submitting
                ? t("workforces.loading.creating")
                : t("workforces.actions.createTeam")}
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}

"use client"

import React, { useEffect, useState } from "react"
import { Pencil, Plus, X, Edit } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import { useI18n } from "@/contexts/i18n-context"
import type { WorkforceAgentSummary, WorkforceWorker } from "@/types/workforce"
import { AgentAvatar, type WorkforceEditDialogsState } from "./workforce-edit-dialogs"
import { GetStartedChecklist, type GetStartedStep } from "./workforce-get-started"

export interface WorkerEditState {
  alias: string
  assignment_instructions: string
  enabled: boolean
  sort_order: string
}

interface WorkforceConfigPanelProps {
  name: string
  description: string
  manager: WorkforceAgentSummary | null
  workers: WorkforceWorker[]
  isArchived: boolean
  saving: boolean
  onSaveDetails: (data: { name: string; description: string }) => Promise<void>
  dialogs: WorkforceEditDialogsState
  getStartedSteps: GetStartedStep[]
  getStartedCollapsed: boolean
  onToggleGetStarted: () => void
  /** Reports whether the Workforce Details form is mid-edit, so a parent
   * that conditionally unmounts this panel (e.g. switching to the Canvas
   * tab) can warn before silently discarding an unsaved edit. */
  onEditingDetailsChange?: (editing: boolean) => void
}

// ─── Main panel ────────────────────────────────────────────────────────────
export function WorkforceConfigPanel({
  name,
  description,
  manager,
  workers,
  isArchived,
  saving,
  onSaveDetails,
  dialogs,
  getStartedSteps,
  getStartedCollapsed,
  onToggleGetStarted,
  onEditingDetailsChange,
}: WorkforceConfigPanelProps) {
  const { t } = useI18n()

  // Details section
  const [editingDetails, setEditingDetails] = useState(false)
  const [detailsName, setDetailsName] = useState(name)
  const [detailsDescription, setDetailsDescription] = useState(description)

  useEffect(() => {
    setDetailsName(name)
    setDetailsDescription(description)
  }, [name, description])

  useEffect(() => {
    onEditingDetailsChange?.(editingDetails)
    // Report "not editing" on unmount too -- a parent that force-switched
    // away mid-edit (after the user confirmed discarding it) shouldn't be
    // left thinking an edit is still in progress.
    return () => onEditingDetailsChange?.(false)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editingDetails])

  const handleSaveDetails = async () => {
    try {
      await onSaveDetails({
        name: detailsName.trim(),
        description: detailsDescription.trim(),
      })
      setEditingDetails(false)
    } catch {
      // onSaveDetails (workforce-builder.tsx) already toasts and sets the
      // error state on failure, then re-throws so the edit view can stay
      // open (setEditingDetails is skipped above) -- swallow it here
      // rather than leaving this onClick-bound async function's rejection
      // unhandled.
    }
  }

  const sortedWorkers = workers
    .slice()
    .sort((a, b) => (a.sort_order ?? 0) - (b.sort_order ?? 0))

  return (
    <div className="flex flex-col gap-8 p-6 h-full overflow-y-auto">
      <GetStartedChecklist
        steps={getStartedSteps}
        collapsed={getStartedCollapsed}
        onToggleCollapsed={onToggleGetStarted}
      />

      {/* Workforce Details */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-base font-semibold">{t("workforces.detail.detailsTitle")}</h2>
          {!isArchived && !editingDetails && (
            <button
              type="button"
              onClick={() => setEditingDetails(true)}
              className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              <Pencil className="h-3.5 w-3.5" />
              {t("common.edit")}
            </button>
          )}
        </div>

        {editingDetails ? (
          <div className="space-y-4 rounded-xl border p-4">
            <div className="space-y-1.5">
              <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                {t("workforces.fields.workforceName")} *
              </Label>
              <Input
                value={detailsName}
                onChange={(e) => setDetailsName(e.target.value)}
                disabled={saving}
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                {t("workforces.fields.description")}
              </Label>
              <Textarea
                value={detailsDescription}
                onChange={(e) => setDetailsDescription(e.target.value)}
                rows={2}
                disabled={saving}
              />
            </div>
            <div className="flex gap-2">
              <Button size="sm" onClick={handleSaveDetails} disabled={saving || !detailsName.trim()}>
                {saving ? t("workforces.loading.saving") : t("common.save")}
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  setDetailsName(name)
                  setDetailsDescription(description)
                  setEditingDetails(false)
                }}
                disabled={saving}
              >
                {t("common.cancel")}
              </Button>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="rounded-xl border p-4">
              <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                {t("workforces.fields.workforceName")} *
              </div>
              <div className="mt-1 text-sm">{name}</div>
            </div>
            <div className="rounded-xl border p-4">
              <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                {t("workforces.fields.description")}
              </div>
              <div className="mt-1 text-sm text-muted-foreground whitespace-pre-wrap">
                {description || (
                  <span className="italic">{t("workforces.detail.noDescription")}</span>
                )}
              </div>
            </div>
          </div>
        )}
      </section>

      {/* Workforce Lead */}
      <section>
        <div className="flex items-start justify-between mb-1">
          <div>
            <h2 className="text-base font-semibold">{t("workforces.detail.leadTitle")} *</h2>
            <p className="text-xs text-muted-foreground mt-0.5">{t("workforces.detail.leadHint")}</p>
          </div>
          {!isArchived && manager && (
            <button
              type="button"
              onClick={() => dialogs.setChangeLeadOpen(true)}
              className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors shrink-0 mt-0.5"
            >
              <Edit className="h-3.5 w-3.5" />
              {t("workforces.actions.change")}
            </button>
          )}
        </div>

        {manager ? (
          <div className="mt-3 rounded-xl border p-4 flex items-center gap-3">
            <AgentAvatar name={manager.name} size="lg" />
            <div className="flex-1 min-w-0">
              <div className="font-medium">{manager.name}</div>
              <div className="text-sm text-muted-foreground truncate">
                {manager.description || ""}
              </div>
            </div>
            <span className="shrink-0 rounded-full bg-primary/10 px-2.5 py-0.5 text-xs font-medium text-primary">
              {t("workforces.detail.leadBadge")}
            </span>
          </div>
        ) : (
          <button
            type="button"
            onClick={() => dialogs.setChangeLeadOpen(true)}
            disabled={isArchived}
            className="mt-3 w-full rounded-xl border border-dashed p-6 text-center text-sm text-muted-foreground hover:border-foreground/30 hover:text-foreground transition-colors disabled:pointer-events-none"
          >
            {t("workforces.canvas.chooseLead.title")}
          </button>
        )}
      </section>

      {/* Members */}
      <section>
        <div className="flex items-start justify-between mb-1">
          <div>
            <h2 className="text-base font-semibold">{t("workforces.detail.membersTitle")}</h2>
            <p className="text-xs text-muted-foreground mt-0.5">{t("workforces.detail.membersHint")}</p>
          </div>
          {!isArchived && (
            <Button
              size="sm"
              variant="outline"
              className="gap-1 shrink-0"
              onClick={() => dialogs.setAddMemberOpen(true)}
            >
              <Plus className="h-3.5 w-3.5" />
              {t("workforces.actions.addAgent")}
            </Button>
          )}
        </div>

        {sortedWorkers.length === 0 ? (
          <div className="mt-3 rounded-xl border border-dashed p-8 text-center text-sm text-muted-foreground">
            {t("workforces.workers.noneConfigured")}
          </div>
        ) : (
          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
            {sortedWorkers.map((worker) => {
              const displayName = worker.alias || worker.agent.name
              return (
                <div
                  key={worker.id}
                  role="button"
                  tabIndex={0}
                  onClick={() => dialogs.openMemberDetail(worker)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault()
                      dialogs.openMemberDetail(worker)
                    }
                  }}
                  className="group relative flex flex-col items-start gap-2 rounded-xl border p-4 text-left hover:border-foreground/30 hover:shadow-sm transition-all cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  {!isArchived && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        dialogs.handleRemoveMember(worker)
                      }}
                      className="absolute top-2 right-2 hidden group-hover:flex items-center justify-center h-5 w-5 rounded-full text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  )}
                  <AgentAvatar name={displayName} size="md" />
                  <div className="min-w-0 w-full">
                    <div className="font-medium text-sm truncate">{displayName}</div>
                    <div className="text-xs text-muted-foreground line-clamp-2 mt-0.5">
                      {worker.agent.description || ""}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </section>
    </div>
  )
}

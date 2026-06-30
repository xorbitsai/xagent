"use client"

import Link from "next/link"
import React, { useEffect, useMemo, useState } from "react"
import { Pencil, Trash2, ExternalLink, Plus, X, Search, Edit } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { useI18n } from "@/contexts/i18n-context"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import { toast } from "sonner"
import type { WorkforceDetail, WorkforceWorker, WorkforceAgentOption } from "@/types/workforce"
import { canEditAgent } from "@/lib/agent-ui-access"

export interface WorkerEditState {
  alias: string
  assignment_instructions: string
  enabled: boolean
  sort_order: string
}

interface WorkforceConfigPanelProps {
  workforce: WorkforceDetail
  agents: WorkforceAgentOption[]
  isArchived: boolean
  saving: boolean
  onSaveWorkforce: (data: {
    name: string
    description: string
    managerAgentId: string
    managerInstructions: string
  }) => Promise<void>
  onAddWorker: (agentId: number, instructions: string, alias?: string) => Promise<void>
  onSaveWorker: (worker: WorkforceWorker, edit: WorkerEditState) => Promise<void>
  onRemoveWorker: (workerId: number) => Promise<void>
}

interface Template {
  id: string
  name: string
  description?: string
}

function AgentAvatar({
  name,
  size = "md",
}: {
  name: string
  size?: "sm" | "md" | "lg"
}) {
  const sizeClass =
    size === "sm"
      ? "h-8 w-8 text-sm"
      : size === "lg"
        ? "h-12 w-12 text-lg"
        : "h-10 w-10 text-base"
  return (
    <div
      className={`${sizeClass} flex shrink-0 items-center justify-center rounded-lg font-semibold bg-primary/15 text-primary`}
    >
      {name.charAt(0).toUpperCase()}
    </div>
  )
}

// ─── Shared agent picker dialog ────────────────────────────────────────────
interface AgentPickerDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  agents: WorkforceAgentOption[]
  onSelectAgent: (agentId: number, description?: string) => Promise<void>
  /** Locale for template API */
  locale: string
  saving?: boolean
}

function AgentPickerDialog({
  open,
  onOpenChange,
  title,
  agents,
  onSelectAgent,
  locale,
  saving,
}: AgentPickerDialogProps) {
  const { t } = useI18n()
  const [tab, setTab] = useState<"my-agents" | "built-in">("my-agents")
  const [search, setSearch] = useState("")
  const [templates, setTemplates] = useState<Template[]>([])
  const [templatesLoading, setTemplatesLoading] = useState(false)
  const [creatingId, setCreatingId] = useState<string | null>(null)
  // template pending name confirmation
  const [pendingTemplate, setPendingTemplate] = useState<Template | null>(null)
  const [pendingName, setPendingName] = useState("")

  // Reset on open
  useEffect(() => {
    if (open) {
      setTab("my-agents")
      setSearch("")
      setPendingTemplate(null)
      setPendingName("")
    }
  }, [open])

  // Fetch templates when switching to built-in tab
  useEffect(() => {
    if (tab !== "built-in" || templates.length > 0) return
    setTemplatesLoading(true)
    apiRequest(`${getApiUrl()}/api/templates/?lang=${locale}`)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load templates")
        return res.json()
      })
      .then((data) => setTemplates(Array.isArray(data) ? data : (data.items ?? [])))
      .catch(() => toast.error(t("workforces.templates.loadError")))
      .finally(() => setTemplatesLoading(false))
  }, [tab, locale, templates.length, t])

  const filteredAgents = useMemo(() => {
    if (!search.trim()) return agents
    const q = search.toLowerCase()
    return agents.filter(
      (a) => a.name.toLowerCase().includes(q) || (a.description || "").toLowerCase().includes(q),
    )
  }, [agents, search])

  const filteredTemplates = useMemo(() => {
    if (!search.trim()) return templates
    const q = search.toLowerCase()
    return templates.filter(
      (t) => t.name.toLowerCase().includes(q) || (t.description || "").toLowerCase().includes(q),
    )
  }, [templates, search])

  const handleSelectTemplate = (template: Template) => {
    setPendingTemplate(template)
    setPendingName(template.name)
  }

  const handleCreateFromTemplate = async () => {
    if (!pendingTemplate || !pendingName.trim()) return
    setCreatingId(pendingTemplate.id)
    let newAgent: { id: number; name: string; description?: string } | null = null
    try {
      const res = await apiRequest(`${getApiUrl()}/api/agents/from-template`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ template_id: pendingTemplate.id, name: pendingName.trim() }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error((err as { detail?: string }).detail || "")
      }
      newAgent = await res.json()
      const publishRes = await apiRequest(`${getApiUrl()}/api/agents/${newAgent!.id}/publish`, { method: "POST" })
      if (!publishRes.ok) {
        const err = await publishRes.json().catch(() => ({}))
        throw new Error((err as { detail?: string }).detail || "")
      }
    } catch (e) {
      if (newAgent?.id) {
        await apiRequest(`${getApiUrl()}/api/agents/${newAgent.id}`, { method: "DELETE" }).catch(() => {})
      }
      toast.error(e instanceof Error && e.message ? e.message : t("workforces.templates.createError"))
      return
    } finally {
      setCreatingId(null)
    }
    toast.success(t("workforces.templates.createSuccess", { name: newAgent!.name }))
    setPendingTemplate(null)
    setPendingName("")
    await onSelectAgent(newAgent!.id, newAgent!.description || pendingTemplate.name)
  }

  const tabBtn = (id: "my-agents" | "built-in", label: string) => (
    <button
      type="button"
      onClick={() => setTab(id)}
      className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${tab === id
          ? "border-primary text-primary"
          : "border-transparent text-muted-foreground hover:text-foreground"
        }`}
    >
      {label}
    </button>
  )

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md p-0 gap-0 overflow-hidden">
        <DialogHeader className="px-6 pt-6 pb-4">
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>

        {/* Search */}
        <div className="px-6 pb-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder={t("workforces.workers.searchPlaceholder")}
              className="pl-9"
            />
          </div>
        </div>

        {/* Tabs */}
        <div className="flex border-b px-6">
          {tabBtn("my-agents", t("workforces.workers.tabMyAgents"))}
          {tabBtn("built-in", t("workforces.workers.tabTemplates"))}
        </div>

        {/* List */}
        <div className="max-h-80 overflow-y-auto">
          {tab === "my-agents" ? (
            filteredAgents.length === 0 ? (
              <p className="py-8 text-center text-sm text-muted-foreground">
                {search ? t("workforces.workers.noSearchResults") : t("workforces.workers.noAvailableAgents")}
              </p>
            ) : (
              <div className="p-2 space-y-0.5">
                {filteredAgents.map((agent) => (
                  <button
                    key={agent.id}
                    type="button"
                    onClick={() => onSelectAgent(agent.id, agent.description || undefined)}
                    disabled={saving}
                    className="w-full flex items-center gap-3 rounded-lg px-3 py-2.5 text-left hover:bg-muted/60 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
                  >
                    <AgentAvatar name={agent.name} size="sm" />
                    <div className="min-w-0 flex-1">
                      <div className="font-medium text-sm">{agent.name}</div>
                      <div className="text-xs text-muted-foreground line-clamp-2">
                        {agent.description || ""}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            )
          ) : templatesLoading ? (
            <p className="py-8 text-center text-sm text-muted-foreground">
              {t("workforces.templates.loading")}
            </p>
          ) : filteredTemplates.length === 0 ? (
            <p className="py-8 text-center text-sm text-muted-foreground">
              {t("workforces.templates.noTemplates")}
            </p>
          ) : (
            <div className="p-2 space-y-0.5">
              {filteredTemplates.map((tmpl) => (
                <div key={tmpl.id}>
                  <button
                    type="button"
                    onClick={() => handleSelectTemplate(tmpl)}
                    disabled={!!creatingId}
                    className={`w-full flex items-center gap-3 rounded-lg px-3 py-2.5 text-left transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50 ${pendingTemplate?.id === tmpl.id
                        ? "bg-muted/60"
                        : "hover:bg-muted/60"
                      }`}
                  >
                    <AgentAvatar name={tmpl.name} size="sm" />
                    <div className="min-w-0 flex-1">
                      <div className="font-medium text-sm">{tmpl.name}</div>
                      <div className="text-xs text-muted-foreground line-clamp-2">
                        {tmpl.description || ""}
                      </div>
                    </div>
                  </button>
                  {pendingTemplate?.id === tmpl.id && (
                    <div className="mx-3 mb-2 mt-1 space-y-2 rounded-lg border bg-muted/30 p-3">
                      <Label className="text-xs text-muted-foreground">
                        {t("workforces.templates.agentName")}
                      </Label>
                      <Input
                        value={pendingName}
                        onChange={(e) => setPendingName(e.target.value)}
                        placeholder={t("workforces.templates.agentNamePlaceholder")}
                        disabled={!!creatingId}
                        autoFocus
                        onKeyDown={(e) => {
                          if (e.key === "Enter") handleCreateFromTemplate()
                          if (e.key === "Escape") { setPendingTemplate(null); setPendingName("") }
                        }}
                      />
                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          onClick={handleCreateFromTemplate}
                          disabled={!!creatingId || !pendingName.trim()}
                          className="flex-1"
                        >
                          {creatingId ? t("workforces.templates.creating") : t("workforces.templates.createAndAdd")}
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => { setPendingTemplate(null); setPendingName("") }}
                          disabled={!!creatingId}
                        >
                          {t("common.cancel")}
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t">
          <Link href="/build/new" target="_blank">
            <Button variant="outline" className="w-full gap-1.5" size="sm">
              <Plus className="h-4 w-4" />
              {t("workforces.detail.createNewAgent")}
            </Button>
          </Link>
        </div>
      </DialogContent>
    </Dialog>
  )
}

// ─── Main panel ────────────────────────────────────────────────────────────
export function WorkforceConfigPanel({
  workforce,
  agents,
  isArchived,
  saving,
  onSaveWorkforce,
  onAddWorker,
  onSaveWorker,
  onRemoveWorker,
}: WorkforceConfigPanelProps) {
  const { t, locale } = useI18n()

  // Details section
  const [editingDetails, setEditingDetails] = useState(false)
  const [detailsName, setDetailsName] = useState(workforce.name)
  const [detailsDescription, setDetailsDescription] = useState(workforce.description || "")
  const [detailsInstructions, setDetailsInstructions] = useState(workforce.manager_instructions || "")

  // Lead section
  const [changeLeadOpen, setChangeLeadOpen] = useState(false)

  // Add member dialog
  const [addMemberOpen, setAddMemberOpen] = useState(false)

  // Member detail dialog
  const [selectedWorker, setSelectedWorker] = useState<WorkforceWorker | null>(null)
  const [memberAlias, setMemberAlias] = useState("")
  const [memberInstructions, setMemberInstructions] = useState("")

  useEffect(() => {
    setDetailsName(workforce.name)
    setDetailsDescription(workforce.description || "")
    setDetailsInstructions(workforce.manager_instructions || "")
  }, [workforce])

  const workerAgentIds = new Set(workforce.workers.map((w) => w.agent.id))

  const availableForLead = agents.filter(
    (a) => a.status === "published" && !workerAgentIds.has(a.id),
  )

  const availableForMember = agents.filter(
    (a) =>
      a.status === "published" &&
      String(a.id) !== String(workforce.manager.id) &&
      !workerAgentIds.has(a.id),
  )

  const handleSaveDetails = async () => {
    await onSaveWorkforce({
      name: detailsName.trim(),
      description: detailsDescription.trim(),
      managerAgentId: String(workforce.manager.id),
      managerInstructions: detailsInstructions.trim(),
    })
    setEditingDetails(false)
  }

  const handleChangeLead = async (agentId: number) => {
    await onSaveWorkforce({
      name: workforce.name,
      description: workforce.description || "",
      managerAgentId: String(agentId),
      managerInstructions: workforce.manager_instructions || "",
    })
    setChangeLeadOpen(false)
  }

  const openMemberDetail = (worker: WorkforceWorker) => {
    setSelectedWorker(worker)
    setMemberAlias(worker.alias || "")
    setMemberInstructions(worker.assignment_instructions || "")
  }

  const closeMemberDetail = () => setSelectedWorker(null)

  const handleSaveMember = async () => {
    if (!selectedWorker) return
    await onSaveWorker(selectedWorker, {
      alias: memberAlias,
      assignment_instructions: memberInstructions,
      enabled: selectedWorker.enabled,
      sort_order: String(selectedWorker.sort_order ?? 1),
    })
    closeMemberDetail()
  }

  const handleRemoveMember = async (worker?: WorkforceWorker) => {
    const target = worker ?? selectedWorker
    if (!target) return
    await onRemoveWorker(target.id)
    closeMemberDetail()
  }

  const handleAddMember = async (agentId: number, description?: string) => {
    const agent = agents.find((a) => a.id === agentId)
    const instructions = description || agent?.description || agent?.name || String(agentId)
    await onAddWorker(agentId, instructions, undefined)
    setAddMemberOpen(false)
  }

  const sortedWorkers = workforce.workers
    .slice()
    .sort((a, b) => (a.sort_order ?? 0) - (b.sort_order ?? 0))

  return (
    <div className="flex flex-col gap-8 p-6 h-full overflow-y-auto">
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
            <div className="space-y-1.5">
              <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                {t("workforces.fields.workforceInstructions")} *
              </Label>
              <Textarea
                value={detailsInstructions}
                onChange={(e) => setDetailsInstructions(e.target.value)}
                rows={4}
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
                  setDetailsName(workforce.name)
                  setDetailsDescription(workforce.description || "")
                  setDetailsInstructions(workforce.manager_instructions || "")
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
              <div className="mt-1 text-sm">{workforce.name}</div>
            </div>
            <div className="rounded-xl border p-4">
              <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                {t("workforces.fields.description")}
              </div>
              <div className="mt-1 text-sm text-muted-foreground whitespace-pre-wrap">
                {workforce.description || (
                  <span className="italic">{t("workforces.detail.noDescription")}</span>
                )}
              </div>
            </div>
            <div className="rounded-xl border p-4">
              <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                {t("workforces.fields.workforceInstructions")} *
              </div>
              <div className="mt-1 text-sm text-muted-foreground whitespace-pre-wrap">
                {workforce.manager_instructions || (
                  <span className="italic">{t("workforces.review.noManagerInstructions")}</span>
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
          {!isArchived && (
            <button
              type="button"
              onClick={() => setChangeLeadOpen(true)}
              className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors shrink-0 mt-0.5"
            >
              <Edit className="h-3.5 w-3.5" />
              {t("workforces.actions.change")}
            </button>
          )}
        </div>

        <div className="mt-3 rounded-xl border p-4 flex items-center gap-3">
          <AgentAvatar name={workforce.manager.name} size="lg" />
          <div className="flex-1 min-w-0">
            <div className="font-medium">{workforce.manager.name}</div>
            <div className="text-sm text-muted-foreground truncate">
              {workforce.manager.description || ""}
            </div>
          </div>
          <span className="shrink-0 rounded-full bg-primary/10 px-2.5 py-0.5 text-xs font-medium text-primary">
            {t("workforces.detail.leadBadge")}
          </span>
        </div>

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
              onClick={() => setAddMemberOpen(true)}
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
                  className="group relative flex flex-col items-start gap-2 rounded-xl border p-4 text-left hover:border-foreground/30 hover:shadow-sm transition-all cursor-pointer"
                  onClick={() => openMemberDetail(worker)}
                >
                  {!isArchived && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        handleRemoveMember(worker)
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

      {/* Change Lead Dialog */}
      <AgentPickerDialog
        open={changeLeadOpen}
        onOpenChange={setChangeLeadOpen}
        title={t("workforces.detail.changeLeadTitle")}
        agents={availableForLead}
        onSelectAgent={handleChangeLead}
        locale={locale}
        saving={saving}
      />

      {/* Add Member Dialog */}
      <AgentPickerDialog
        open={addMemberOpen}
        onOpenChange={setAddMemberOpen}
        title={t("workforces.detail.addMemberTitle")}
        agents={availableForMember}
        onSelectAgent={handleAddMember}
        locale={locale}
        saving={saving}
      />

      {/* Member Detail Dialog */}
      <Dialog open={!!selectedWorker} onOpenChange={(open) => { if (!open) closeMemberDetail() }}>
        <DialogContent className="sm:max-w-md flex flex-col max-h-[90vh]">
          {selectedWorker && (
            <>
              <DialogHeader className="shrink-0">
                <div className="flex items-center gap-3 pr-6 min-w-0">
                  <AgentAvatar
                    name={selectedWorker.alias || selectedWorker.agent.name}
                    size="md"
                  />
                  <div className="min-w-0 flex-1">
                    <DialogTitle className="text-base truncate">
                      {selectedWorker.alias || selectedWorker.agent.name}
                    </DialogTitle>
                    <p className="text-xs text-muted-foreground line-clamp-2 mt-0.5">
                      {selectedWorker.agent.description || ""}
                    </p>
                  </div>
                </div>
              </DialogHeader>

              <div className="space-y-4 mt-2 overflow-y-auto flex-1 min-h-0">
                <div className="space-y-1.5">
                  <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    {t("workforces.fields.alias")}
                  </Label>
                  <Input
                    value={memberAlias}
                    onChange={(e) => setMemberAlias(e.target.value)}
                    placeholder={t("workforces.workers.aliasPlaceholder")}
                    disabled={saving || isArchived}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    {t("workforces.fields.delegationPrompt")}
                  </Label>
                  <Textarea
                    value={memberInstructions}
                    onChange={(e) => setMemberInstructions(e.target.value)}
                    rows={4}
                    disabled={saving || isArchived}
                  />
                </div>
              </div>

              <div className="flex items-center justify-between mt-4 pt-4 border-t shrink-0">
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-destructive hover:text-destructive gap-1.5"
                  onClick={() => handleRemoveMember()}
                  disabled={saving || isArchived}
                >
                  <Trash2 className="h-4 w-4" />
                  {t("workforces.actions.remove")}
                </Button>
                <div className="flex items-center gap-2">
                  {canEditAgent(selectedWorker.agent) && (
                    <Link href={`/build/${selectedWorker.agent.id}`} target="_blank">
                      <Button variant="ghost" size="sm" className="gap-1.5">
                        <ExternalLink className="h-4 w-4" />
                        {t("workforces.actions.openAgent")}
                      </Button>
                    </Link>
                  )}
                  <Button
                    size="sm"
                    onClick={handleSaveMember}
                    disabled={saving || isArchived || !memberInstructions.trim()}
                  >
                    {saving ? t("workforces.loading.saving") : t("common.done")}
                  </Button>
                </div>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}

export function workerEditState(worker: WorkforceWorker): WorkerEditState {
  return {
    alias: worker.alias || "",
    assignment_instructions: worker.assignment_instructions || "",
    enabled: worker.enabled,
    sort_order: String(worker.sort_order ?? 1),
  }
}

export function buildWorkerEditState(workers: WorkforceWorker[]): Record<number, WorkerEditState> {
  return workers.reduce<Record<number, WorkerEditState>>((acc, w) => {
    acc[w.id] = workerEditState(w)
    return acc
  }, {})
}

export function normalizeWorkerSortOrder(
  value: string,
  fallback: number | null | undefined,
): number {
  const parsed = /^\d+$/.test(value.trim()) ? Number.parseInt(value.trim(), 10) : NaN
  return Number.isInteger(parsed) && parsed > 0 ? parsed : (fallback ?? 1)
}

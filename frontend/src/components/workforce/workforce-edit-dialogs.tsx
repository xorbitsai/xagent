"use client"

import Link from "next/link"
import React, { useEffect, useMemo, useState } from "react"
import { ExternalLink, Plus, Search, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { useI18n } from "@/contexts/i18n-context"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import { toast } from "sonner"
import type { WorkforceAgentSummary, WorkforceWorker, WorkforceAgentOption } from "@/types/workforce"
import { canEditAgent } from "@/lib/agent-ui-access"
import type { WorkerEditState } from "./workforce-config-panel"

interface Template {
  id: string
  name: string
  description?: string
}

export function AgentAvatar({
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
export interface AgentPickerDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  agents: WorkforceAgentOption[]
  onSelectAgent: (agentId: number, description?: string) => Promise<void>
  /** Locale for template API */
  locale: string
  saving?: boolean
  /** Reports a template-created agent to the parent so it can be added to
   * its own `agents` list immediately -- without this, create mode (whose
   * list is fetched once on mount and never refreshed) resolves the new
   * agent as a blank placeholder wherever it looks it up by id (PR review
   * round 9, NEW-F2). */
  onAgentCreated?: (agent: WorkforceAgentOption) => void
}

export function AgentPickerDialog({
  open,
  onOpenChange,
  title,
  agents,
  onSelectAgent,
  locale,
  saving,
  onAgentCreated,
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
  // Guards against a rapid double-click adding the same agent twice: the
  // create-mode draft path resolves onSelectAgent synchronously with no
  // network round trip, so the `saving` prop (only set by the edit-mode
  // API-backed path) never disables the row in time.
  const [selecting, setSelecting] = useState(false)

  // Reset on open
  useEffect(() => {
    if (open) {
      setTab("my-agents")
      setSearch("")
      setPendingTemplate(null)
      setPendingName("")
      setSelecting(false)
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

  const handleSelect = async (agentId: number, description?: string) => {
    if (selecting) return
    setSelecting(true)
    try {
      await onSelectAgent(agentId, description)
    } catch {
      // The caller (handleChangeLead/handleAddWorker) already toasts the
      // error and re-throws only so its own try/finally runs; this is a
      // fire-and-forget onClick with no caller to propagate to, so swallow
      // it here instead of leaving an unhandled promise rejection.
    } finally {
      setSelecting(false)
    }
  }

  const handleSelectTemplate = (template: Template) => {
    setPendingTemplate(template)
    setPendingName(template.name)
  }

  const handleCreateFromTemplate = async () => {
    if (!pendingTemplate || !pendingName.trim()) return
    setCreatingId(pendingTemplate.id)
    let newAgent: { id: number; name: string; description?: string; logo_url?: string | null } | null = null
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
    // Reported before onSelectAgent below so a create-mode parent's `agents`
    // list already contains this agent by the time onSelectAgent's own
    // state updates (draftWorkers/draftManagerAgentId) trigger the re-render
    // that looks it up.
    onAgentCreated?.({
      id: newAgent!.id,
      name: newAgent!.name,
      description: newAgent!.description ?? null,
      logo_url: newAgent!.logo_url ?? null,
      status: "published",
    })
    setPendingTemplate(null)
    setPendingName("")
    try {
      // Passes the freshly created agent's own name (never its description
      // -- see handleAddMember above) explicitly rather than relying on
      // handleAddMember's `agents.find(...)` fallback: `agents` is the
      // parent's already-fetched list, which this brand-new agent isn't in
      // yet, so that lookup would miss and fall all the way through to the
      // bare numeric agent id.
      await onSelectAgent(newAgent!.id, newAgent!.name)
    } catch {
      // Same fire-and-forget rationale as handleSelect: the caller already
      // toasted its own error, nothing here to propagate to.
    }
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
          <DialogDescription className="sr-only">
            {t("workforces.workers.searchPlaceholder")}
          </DialogDescription>
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
                    onClick={() => handleSelect(agent.id)}
                    disabled={saving || selecting}
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

// ─── Shared editing state, driving both the Configure tab and the Canvas ──
export interface UseWorkforceEditDialogsArgs {
  manager: WorkforceAgentSummary | null
  workers: WorkforceWorker[]
  agents: WorkforceAgentOption[]
  onChangeLead: (agentId: number) => Promise<void>
  onAddWorker: (agentId: number, instructions: string) => Promise<void>
  onSaveWorker: (worker: WorkforceWorker, edit: WorkerEditState) => Promise<void>
  onRemoveWorker: (worker: WorkforceWorker) => Promise<void>
}

export function useWorkforceEditDialogs({
  manager,
  workers,
  agents,
  onChangeLead,
  onAddWorker,
  onSaveWorker,
  onRemoveWorker,
}: UseWorkforceEditDialogsArgs) {
  const [changeLeadOpen, setChangeLeadOpen] = useState(false)
  const [addMemberOpen, setAddMemberOpen] = useState(false)
  const [selectedWorker, setSelectedWorker] = useState<WorkforceWorker | null>(null)
  const [memberAlias, setMemberAlias] = useState("")
  const [memberInstructions, setMemberInstructions] = useState("")

  const workerAgentIds = new Set(workers.map((w) => w.agent.id))

  const availableForLead = agents.filter(
    (a) => a.status === "published" && !workerAgentIds.has(a.id),
  )

  const availableForMember = agents.filter(
    (a) =>
      a.status === "published" &&
      String(a.id) !== String(manager?.id) &&
      !workerAgentIds.has(a.id),
  )

  const handleChangeLead = async (agentId: number) => {
    await onChangeLead(agentId)
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
    try {
      await onSaveWorker(selectedWorker, {
        alias: memberAlias,
        assignment_instructions: memberInstructions,
        enabled: selectedWorker.enabled,
        sort_order: String(selectedWorker.sort_order ?? 1),
      })
    } catch {
      // Same fire-and-forget rationale as handleSelect above: onSaveWorker
      // (workforce-builder.tsx) already toasts the error and re-throws only
      // so its own try/finally resets saving state; this onClick handler has
      // no caller to propagate to, so swallow it here instead of leaving an
      // unhandled promise rejection (PR review round 7, finding #8).
      return
    }
    closeMemberDetail()
  }

  const handleRemoveMember = async (worker?: WorkforceWorker) => {
    const target = worker ?? selectedWorker
    if (!target) return
    try {
      await onRemoveWorker(target)
    } catch {
      // Same rationale as handleSaveMember above.
      return
    }
    closeMemberDetail()
  }

  const handleAddMember = async (agentId: number, description?: string) => {
    const agent = agents.find((a) => a.id === agentId)
    // Never falls back to the agent's own description (PR review round 7,
    // finding #4): that's a public-facing blurb the user never wrote as an
    // instruction and never reviews before it becomes this member's live
    // system prompt -- one click on the picker used to copy it verbatim.
    // Callers no longer pass the agent's description through `description`
    // (kept in the shared onSelectAgent signature for API-contract
    // stability with handleChangeLead's implementation of the same prop).
    const instructions = description || agent?.name || String(agentId)
    await onAddWorker(agentId, instructions)
    setAddMemberOpen(false)
  }

  return {
    changeLeadOpen,
    setChangeLeadOpen,
    addMemberOpen,
    setAddMemberOpen,
    selectedWorker,
    openMemberDetail,
    closeMemberDetail,
    memberAlias,
    setMemberAlias,
    memberInstructions,
    setMemberInstructions,
    handleChangeLead,
    handleAddMember,
    handleSaveMember,
    handleRemoveMember,
    availableForLead,
    availableForMember,
  }
}

export type WorkforceEditDialogsState = ReturnType<typeof useWorkforceEditDialogs>

// ─── Dialogs themselves — rendered once, shared by Configure and Canvas ───
interface WorkforceEditDialogsProps {
  locale: string
  saving: boolean
  isArchived: boolean
  dialogs: WorkforceEditDialogsState
  onAgentCreated?: (agent: WorkforceAgentOption) => void
}

export function WorkforceEditDialogs({
  locale,
  saving,
  isArchived,
  dialogs,
  onAgentCreated,
}: WorkforceEditDialogsProps) {
  const { t } = useI18n()
  const {
    changeLeadOpen,
    setChangeLeadOpen,
    addMemberOpen,
    setAddMemberOpen,
    selectedWorker,
    closeMemberDetail,
    memberAlias,
    setMemberAlias,
    memberInstructions,
    setMemberInstructions,
    handleChangeLead,
    handleAddMember,
    handleSaveMember,
    handleRemoveMember,
    availableForLead,
    availableForMember,
  } = dialogs

  return (
    <>
      {/* Change Lead Dialog */}
      <AgentPickerDialog
        open={changeLeadOpen}
        onOpenChange={setChangeLeadOpen}
        title={t("workforces.detail.changeLeadTitle")}
        agents={availableForLead}
        onSelectAgent={handleChangeLead}
        locale={locale}
        saving={saving}
        onAgentCreated={onAgentCreated}
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
        onAgentCreated={onAgentCreated}
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
                    <DialogDescription className="text-xs text-muted-foreground line-clamp-2 mt-0.5">
                      {selectedWorker.agent.description || ""}
                    </DialogDescription>
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
    </>
  )
}

"use client"

import React, { useEffect, useMemo, useState } from "react"
import { ChevronDown, ChevronUp, Loader2, X, Plus, Search, Users, LayoutTemplate, Check } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { useI18n } from "@/contexts/i18n-context"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import { toast } from "sonner"
import type { WorkforceAgentOption, WorkforceWorkerDraft } from "@/types/workforce"
import type { Template } from "@/types/template"

interface WorkersStepProps {
  managerAgentId: string
  agents: WorkforceAgentOption[]
  workers: WorkforceWorkerDraft[]
  onWorkersChange: (workers: WorkforceWorkerDraft[]) => void
  onAgentCreated?: (agent: WorkforceAgentOption) => void
}

export function WorkersStep({
  managerAgentId,
  agents,
  workers,
  onWorkersChange,
  onAgentCreated,
}: WorkersStepProps) {
  const { t, locale } = useI18n()
  const [activeTab, setActiveTab] = useState<"my-agents" | "templates">("my-agents")
  const [addPanelOpen, setAddPanelOpen] = useState(true)
  const [searchQuery, setSearchQuery] = useState("")
  const [expandedIndices, setExpandedIndices] = useState<Set<number>>(new Set())

  // Templates state
  const [templates, setTemplates] = useState<Template[]>([])
  const [loadingTemplates, setLoadingTemplates] = useState(false)
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null)
  const [templateName, setTemplateName] = useState("")
  const [creatingFromTemplate, setCreatingFromTemplate] = useState(false)

  const workerAgentIds = useMemo(() => new Set(workers.map((w) => w.agent_id)), [workers])

  const selectableAgents = useMemo(() => {
    return agents.filter(
      (agent) =>
        String(agent.id) !== managerAgentId &&
        !workerAgentIds.has(agent.id) &&
        (searchQuery === "" ||
          agent.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          (agent.description ?? "").toLowerCase().includes(searchQuery.toLowerCase())),
    )
  }, [agents, managerAgentId, workerAgentIds, searchQuery])

  const filteredTemplates = useMemo(() => {
    if (!searchQuery) return templates
    return templates.filter(
      (tpl) =>
        tpl.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (tpl.description ?? "").toLowerCase().includes(searchQuery.toLowerCase()),
    )
  }, [templates, searchQuery])

  useEffect(() => {
    if (activeTab !== "templates" || templates.length > 0) return
    setLoadingTemplates(true)
    apiRequest(`${getApiUrl()}/api/templates/?lang=${locale}`)
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status}`)
        return res.json()
      })
      .then((data: Template[]) => setTemplates(data))
      .catch(() => toast.error(t("workforces.templates.loadError")))
      .finally(() => setLoadingTemplates(false))
  }, [activeTab, locale, templates.length, t])

  const addAgent = (agentId: number) => {
    const newIndex = workers.length
    onWorkersChange([
      ...workers,
      {
        source_type: "existing",
        agent_id: agentId,
        alias: "",
        assignment_instructions: "",
        enabled: true,
        sort_order: workers.length + 1,
      },
    ])
    setExpandedIndices((prev) => new Set([...prev, newIndex]))
  }

  const updateWorker = (index: number, next: WorkforceWorkerDraft) => {
    const updated = [...workers]
    updated[index] = next
    onWorkersChange(updated)
  }

  const removeWorker = (index: number) => {
    const next = workers.filter((_, i) => i !== index).map((w, i) => ({ ...w, sort_order: i + 1 }))
    onWorkersChange(next)
    setExpandedIndices((prev) => {
      const updated = new Set<number>()
      for (const idx of prev) {
        if (idx < index) updated.add(idx)
        else if (idx > index) updated.add(idx - 1)
      }
      return updated
    })
  }

  const toggleExpanded = (index: number) => {
    setExpandedIndices((prev) => {
      const next = new Set(prev)
      if (next.has(index)) next.delete(index)
      else next.add(index)
      return next
    })
  }

  const handleSelectTemplate = (templateId: string, defaultName: string) => {
    if (selectedTemplateId === templateId) {
      setSelectedTemplateId(null)
      setTemplateName("")
    } else {
      setSelectedTemplateId(templateId)
      setTemplateName(defaultName)
    }
  }

  const handleCreateFromTemplate = async () => {
    if (!selectedTemplateId || !templateName.trim() || creatingFromTemplate) return
    try {
      setCreatingFromTemplate(true)
      // Create agent from template
      const createRes = await apiRequest(`${getApiUrl()}/api/agents/from-template`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ template_id: selectedTemplateId, name: templateName.trim() }),
      })
      if (!createRes.ok) throw new Error(`${createRes.status}`)
      const agent = await createRes.json()

      // Publish the agent
      const publishRes = await apiRequest(`${getApiUrl()}/api/agents/${agent.id}/publish`, {
        method: "POST",
      })
      if (!publishRes.ok) throw new Error(`${publishRes.status}`)

      const newAgent: WorkforceAgentOption = {
        id: agent.id,
        name: agent.name,
        description: agent.description ?? null,
        logo_url: agent.logo_url ?? null,
        status: "published",
      }
      onAgentCreated?.(newAgent)
      addAgent(newAgent.id)
      setSelectedTemplateId(null)
      setTemplateName("")
      toast.success(t("workforces.templates.createSuccess", { name: newAgent.name }))
    } catch {
      toast.error(t("workforces.templates.createError"))
    } finally {
      setCreatingFromTemplate(false)
    }
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        {t("workforces.workers.addAgentsHint")}
      </p>

      {/* Add agents panel */}
      <div className="rounded-xl border overflow-hidden">
        <button
          onClick={() => setAddPanelOpen((v) => !v)}
          className="flex w-full items-center justify-between px-4 py-3 text-sm font-medium hover:bg-muted/30 transition-colors"
        >
          <div className="flex items-center gap-2">
            <Users className="h-4 w-4 text-muted-foreground" />
            {t("workforces.workers.addAgentsTitle")}
          </div>
          {addPanelOpen ? (
            <ChevronUp className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          )}
        </button>

        {addPanelOpen && (
          <div className="border-t">
            {/* Tab switcher */}
            <div className="flex border-b">
              <button
                onClick={() => setActiveTab("my-agents")}
                className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${activeTab === "my-agents"
                  ? "border-primary text-primary"
                  : "border-transparent text-muted-foreground hover:text-foreground"
                  }`}
              >
                <Users className="h-3.5 w-3.5" />
                {t("workforces.workers.tabMyAgents")}
              </button>
              <button
                onClick={() => setActiveTab("templates")}
                className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${activeTab === "templates"
                  ? "border-primary text-primary"
                  : "border-transparent text-muted-foreground hover:text-foreground"
                  }`}
              >
                <LayoutTemplate className="h-3.5 w-3.5" />
                {t("workforces.workers.tabTemplates")}
              </button>
            </div>

            {/* Search */}
            <div className="px-4 pt-3 pb-2">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder={
                    activeTab === "my-agents"
                      ? t("workforces.workers.searchPlaceholder")
                      : t("workforces.workers.searchTemplatesPlaceholder")
                  }
                  className="pl-9"
                />
              </div>
            </div>

            {/* My Agents list */}
            {activeTab === "my-agents" && (
              <div className="max-h-60 overflow-y-auto divide-y">
                {selectableAgents.length === 0 ? (
                  <div className="px-4 py-6 text-center text-sm text-muted-foreground">
                    {searchQuery
                      ? t("workforces.workers.noSearchResults")
                      : t("workforces.workers.noAvailableAgents")}
                  </div>
                ) : (
                  selectableAgents.map((agent) => (
                    <div key={agent.id} className="flex items-center gap-3 px-4 py-3 hover:bg-muted/20 transition-colors">
                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-blue-600 text-sm font-semibold text-white">
                        {agent.name.charAt(0).toUpperCase()}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm">{agent.name}</div>
                        {agent.description && (
                          <div className="text-xs text-muted-foreground truncate">{agent.description}</div>
                        )}
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 rounded-full hover:bg-blue-50 hover:text-blue-600"
                        onClick={() => addAgent(agent.id)}
                      >
                        <Plus className="h-4 w-4" />
                      </Button>
                    </div>
                  ))
                )}
              </div>
            )}

            {/* Templates list */}
            {activeTab === "templates" && (
              <div className="max-h-80 overflow-y-auto">
                {loadingTemplates ? (
                  <div className="flex items-center justify-center py-8 text-sm text-muted-foreground gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    {t("workforces.templates.loading")}
                  </div>
                ) : filteredTemplates.length === 0 ? (
                  <div className="px-4 py-6 text-center text-sm text-muted-foreground">
                    {searchQuery
                      ? t("workforces.workers.noSearchResults")
                      : t("workforces.templates.noTemplates")}
                  </div>
                ) : (
                  <div className="divide-y">
                    {filteredTemplates.map((tpl) => (
                      <div key={tpl.id}>
                        <div
                          className="flex items-center gap-3 px-4 py-3 hover:bg-muted/50 transition-colors cursor-pointer"
                          onClick={() => handleSelectTemplate(tpl.id, tpl.name)}
                        >
                          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-violet-100 text-sm font-semibold text-violet-700">
                            {tpl.name.charAt(0).toUpperCase()}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="font-medium text-sm">{tpl.name}</div>
                            {tpl.description && (
                              <div className="text-xs text-muted-foreground truncate">{tpl.description}</div>
                            )}
                          </div>
                          <Button
                            variant="ghost"
                            size="icon"
                            className={`h-7 w-7 rounded-full ${selectedTemplateId === tpl.id
                              ? "bg-violet-100 text-violet-600"
                              : "hover:bg-violet-50 hover:text-violet-600"
                              }`}
                          >
                            {selectedTemplateId === tpl.id ? (
                              <Check className="h-4 w-4" />
                            ) : (
                              <Plus className="h-4 w-4" />
                            )}
                          </Button>
                        </div>

                        {/* Inline name input when selected */}
                        {selectedTemplateId === tpl.id && (
                          <div className="border-t bg-muted/10 px-4 py-3 space-y-3">
                            <div className="space-y-1.5">
                              <Label className="text-xs">{t("workforces.templates.agentName")}</Label>
                              <Input
                                value={templateName}
                                onChange={(e) => setTemplateName(e.target.value)}
                                placeholder={t("workforces.templates.agentNamePlaceholder")}
                                autoFocus
                                onKeyDown={(e) => {
                                  if (e.key === "Enter") handleCreateFromTemplate()
                                  if (e.key === "Escape") { setSelectedTemplateId(null); setTemplateName("") }
                                }}
                              />
                            </div>
                            <div className="flex items-center gap-2 justify-end">
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => { setSelectedTemplateId(null); setTemplateName("") }}
                                disabled={creatingFromTemplate}
                              >
                                {t("common.cancel")}
                              </Button>
                              <Button
                                size="sm"
                                className="bg-violet-600 hover:bg-violet-700 text-white"
                                onClick={handleCreateFromTemplate}
                                disabled={!templateName.trim() || creatingFromTemplate}
                              >
                                {creatingFromTemplate ? (
                                  <>
                                    <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                                    {t("workforces.templates.creating")}
                                  </>
                                ) : (
                                  t("workforces.templates.createAndAdd")
                                )}
                              </Button>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Added agents */}
      {workers.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground px-0.5">
            {t("workforces.workers.addedAgentsTitle")}
          </div>
          {workers.map((worker, index) => {
            const agent = agents.find((a) => a.id === worker.agent_id)
            const title = worker.alias || agent?.name || t("workforces.workers.fallbackName", { index: index + 1 })
            const isExpanded = expandedIndices.has(index)

            return (
              <div key={`${worker.agent_id}-${index}`} className="rounded-xl border overflow-hidden">
                <div className="flex items-center gap-3 px-4 py-3">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-blue-600 text-sm font-semibold text-white">
                    {title.charAt(0).toUpperCase()}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-sm">{title}</div>
                    {agent?.description && (
                      <div className="text-xs text-muted-foreground truncate">{agent.description}</div>
                    )}
                  </div>
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-muted-foreground"
                      onClick={() => toggleExpanded(index)}
                    >
                      {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-muted-foreground hover:text-destructive"
                      onClick={() => removeWorker(index)}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>

                {isExpanded && (
                  <div className="border-t px-4 py-4 space-y-4 bg-muted/10">
                    <div className="space-y-2">
                      <Label className="text-xs">{t("workforces.fields.alias")}</Label>
                      <Input
                        value={worker.alias || ""}
                        onChange={(e) => updateWorker(index, { ...worker, alias: e.target.value })}
                        placeholder={t("workforces.workers.aliasPlaceholder")}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="text-xs">
                        {t("workforces.fields.assignmentInstructions")} <span className="text-destructive">*</span>
                      </Label>
                      <Textarea
                        value={worker.assignment_instructions}
                        onChange={(e) =>
                          updateWorker(index, { ...worker, assignment_instructions: e.target.value })
                        }
                        placeholder={t("workforces.workers.instructionsPlaceholder")}
                        rows={3}
                      />
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

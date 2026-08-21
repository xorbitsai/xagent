"use client"

import React, { useState, useEffect, useRef } from "react"
import { SearchInput } from "@/components/ui/search-input"
import { Button } from "@/components/ui/button"
import { PageHeader } from "@/components/ui/page-header"
import { Plus, Bot, Trash2, MessageSquare, Edit, MoreVertical, Globe, ArrowUpDown, Rocket, Sparkles, Settings2, ArrowRight, FileText, Wrench, Database, Plug, KeyRound, Webhook, Mic, Square, Loader2 } from "lucide-react"
import { Popover, PopoverTrigger, PopoverContent } from "@/components/ui/popover"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Textarea } from "@/components/ui/textarea"
import { DeployAgentDialog, Agent } from "@/components/build/deploy-agent-dialog"
import { AgentTriggersDialog } from "@/components/build/agent-triggers-dialog"
import {
  AgentDeleteDialog,
  type AgentDeletePendingAction,
} from "@/components/build/agent-delete-dialog"
import { FeatureEmptyState } from "@/components/ui/feature-empty-state"
import { SegmentedTabs, type SegmentedTabItem } from "@/components/ui/segmented-tabs"
import { PersonaAvatar } from "@/components/templates/persona-avatar"
import { pillClasses } from "@/components/templates/library-template-card"
import { categoryLabel } from "@/lib/template-categories"
import { useI18n } from "@/contexts/i18n-context"
import { useApp } from "@/contexts/app-context-chat"
import { useRouter, useSearchParams } from "next/navigation"
import { apiRequest, parseApiResponse } from "@/lib/api-wrapper"
import { getApiUrl, resolveAgentLogoUrl } from "@/lib/utils"
import { resolveTaskLlmSelection } from "@/lib/models"
import type { Template } from "@/types/template"
import { normalizeTaskPromptTitle, parseTaskCreateCore } from "@/lib/task-create"
import {
  canDeleteAgent,
  canEditAgent,
  canPublishAgent,
  canRunAgent,
  getAgentChatHref,
} from "@/lib/agent-ui-access"
import {
  requestAgentDeletion,
  type AgentDeleteConflictDetail,
  type AgentDeleteWorkforceReference,
} from "@/lib/agent-delete"
import {
  discardWorkforce,
  WorkforceDiscardError,
} from "@/lib/workforces-api"
import { toast } from "@/components/ui/sonner"
import { getBrandingFromEnv } from "@/lib/branding"
import { useVoiceInputControls } from "@/components/voice-input-controller"
import {
  BuildAgentCardExtension,
  BuildPageExtensionProvider,
} from "@/lib/build-page-extension"

function BuildsPageContent() {
  const { t, locale } = useI18n()
  const { dispatch, setTaskId, setPendingMessage } = useApp()
  const router = useRouter()
  const searchParams = useSearchParams()
  const hasAutoOpenedCreateRef = useRef(false)
  const createPromptRef = useRef<HTMLTextAreaElement | null>(null)
  const isMountedRef = useRef(false)
  const agentListRequestGenerationRef = useRef(0)
  const agentDeleteActionGenerationRef = useRef(0)
  const activeTaskCreateAttemptRef = useRef<number | null>(null)
  const taskCreateCounterRef = useRef(0)
  const draftRevisionRef = useRef(0)
  const [searchTerm, setSearchTerm] = useState("")
  const [agents, setAgents] = useState<Agent[]>([])
  const [loading, setLoading] = useState(true)
  const [statusTab, setStatusTab] = useState<"all" | "enabled" | "drafts">("all")
  const [sortMode, setSortMode] = useState<"updated" | "name">("updated")
  // Best-effort enrichment only: an agent traces back to the template it was
  // hired from via `template_id`, and this lookup supplies that template's
  // category/persona for the card badge and avatar. A custom, scratch-built
  // agent has no `template_id` and simply renders without them.
  const [templatesById, setTemplatesById] = useState<Record<string, Template>>({})
  const branding = getBrandingFromEnv();

  // Deploy Dialog State
  const [deployAgent, setDeployAgent] = useState<Agent | null>(null)
  const [triggersAgent, setTriggersAgent] = useState<Agent | null>(null)

  // Check for template parameter and redirect to create page
  useEffect(() => {
    const templateId = searchParams.get("template")
    if (templateId) {
      // Redirect to create page with template parameter
      router.replace(`/build/new?template=${templateId}`)
    }
  }, [searchParams, router])

  useEffect(() => {
    const shouldOpenCreate = searchParams.get("create") === "true"
    if (!shouldOpenCreate) {
      hasAutoOpenedCreateRef.current = false
      return
    }
    if (hasAutoOpenedCreateRef.current) return
    hasAutoOpenedCreateRef.current = true
    setIsCreateModalOpen(true)
    router.replace("/build")
  }, [searchParams, router])

  // Fetch agents on mount
  const fetchAgents = async () => {
    if (!isMountedRef.current) return
    const requestGeneration = agentListRequestGenerationRef.current + 1
    agentListRequestGenerationRef.current = requestGeneration

    try {
      setLoading(true)
      const response = await apiRequest(`${getApiUrl()}/api/agents`)
      if (response.ok) {
        const data = await response.json()
        if (
          !isMountedRef.current ||
          requestGeneration !== agentListRequestGenerationRef.current
        ) {
          return
        }
        setAgents(data)
      }
    } catch (error) {
      if (
        isMountedRef.current &&
        requestGeneration === agentListRequestGenerationRef.current
      ) {
        console.error("Failed to fetch agents:", error)
      }
    } finally {
      if (
        isMountedRef.current &&
        requestGeneration === agentListRequestGenerationRef.current
      ) {
        setLoading(false)
      }
    }
  }

  useEffect(() => {
    isMountedRef.current = true
    void fetchAgents()
    return () => {
      isMountedRef.current = false
      activeTaskCreateAttemptRef.current = null
      agentListRequestGenerationRef.current += 1
      agentDeleteActionGenerationRef.current += 1
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/templates/?lang=${locale}`)
        if (!response.ok || cancelled) return
        const data: Template[] = await response.json()
        if (cancelled) return
        const map: Record<string, Template> = {}
        for (const template of data) map[template.id] = template
        setTemplatesById(map)
      } catch {
        // Card badges/avatars just render without template enrichment.
      }
    })()
    return () => {
      cancelled = true
    }
  }, [locale])

  const publicationOperations = {
    publish: {
      suffix: "publish",
      errorKey: "builds.publication.publishFailed",
      diagnostic: "Failed to publish agent:",
    },
    unpublish: {
      suffix: "unpublish",
      errorKey: "builds.publication.unpublishFailed",
      diagnostic: "Failed to unpublish agent:",
    },
  } as const

  const performPublicationMutation = async (
    agentId: number,
    kind: keyof typeof publicationOperations,
  ): Promise<"success" | "failed" | "stale"> => {
    const operation = publicationOperations[kind]
    try {
      const response = await apiRequest(`${getApiUrl()}/api/agents/${agentId}/${operation.suffix}`, {
        method: "POST",
      })
      if (!isMountedRef.current) return "stale"
      if (!response.ok) {
        console.error(operation.diagnostic, response)
        toast.error(t(operation.errorKey))
        return "failed"
      }
      return "success"
    } catch (error) {
      if (!isMountedRef.current) return "stale"
      console.error(operation.diagnostic, error)
      toast.error(t(operation.errorKey))
      return "failed"
    }
  }

  const handlePublication = async (
    agentId: number,
    kind: keyof typeof publicationOperations,
  ) => {
    const outcome = await performPublicationMutation(agentId, kind)
    if (outcome === "success" && isMountedRef.current) {
      void fetchAgents()
    }
  }

  const [agentDeleteSession, setAgentDeleteSession] = useState<{
    target: { id: number; name: string }
    conflict: AgentDeleteConflictDetail | null
  } | null>(null)
  const [agentDeletePendingAction, setAgentDeletePendingAction] =
    useState<AgentDeletePendingAction>(null)

  const confirmDeleteAgent = async () => {
    if (!agentDeleteSession || agentDeletePendingAction) return
    const session = agentDeleteSession
    const actionGeneration = agentDeleteActionGenerationRef.current + 1
    agentDeleteActionGenerationRef.current = actionGeneration
    const isCurrentAction = () =>
      isMountedRef.current &&
      actionGeneration === agentDeleteActionGenerationRef.current
    setAgentDeletePendingAction({ kind: "delete" })

    try {
      const result = await requestAgentDeletion(
        session.target.id,
        t("common.deleteFailed"),
      )
      if (!isCurrentAction()) return
      if (result.kind === "blocked") {
        setAgentDeleteSession((current) =>
          current?.target.id === session.target.id
            ? { ...current, conflict: result.conflict }
            : current,
        )
        toast.error(
          t("builds.list.deleteDialog.blockedToast", {
            name: session.target.name,
          }),
        )
        return
      }

      setAgents((current) =>
        current.filter((agent) => agent.id !== session.target.id),
      )
      setAgentDeleteSession((current) =>
        current?.target.id === session.target.id ? null : current,
      )
      void fetchAgents()
    } catch (error) {
      if (!isCurrentAction()) return
      console.error("Failed to delete agent:", error)
      toast.error(
        error instanceof Error ? error.message : t("common.deleteFailed"),
      )
    } finally {
      if (isCurrentAction()) {
        setAgentDeletePendingAction(null)
      }
    }
  }

  const handleDiscardWorkforce = async (
    reference: AgentDeleteWorkforceReference,
  ) => {
    if (!agentDeleteSession || agentDeletePendingAction) return
    const session = agentDeleteSession
    const actionGeneration = agentDeleteActionGenerationRef.current + 1
    agentDeleteActionGenerationRef.current = actionGeneration
    const isCurrentAction = () =>
      isMountedRef.current &&
      actionGeneration === agentDeleteActionGenerationRef.current
    setAgentDeletePendingAction({
      kind: "discard",
      workforceId: reference.workforce_id,
    })

    try {
      await discardWorkforce(
        reference.workforce_id,
        t("builds.list.deleteDialog.discardFailed", {
          name: reference.name,
        }),
      )
      if (!isCurrentAction()) return
      setAgentDeleteSession((current) => {
        if (
          current?.target.id !== session.target.id ||
          current.conflict === null
        ) {
          return current
        }

        return {
          ...current,
          conflict: {
            ...current.conflict,
            references: current.conflict.references.filter(
              (item) => item.workforce_id !== reference.workforce_id,
            ),
          },
        }
      })
    } catch (error) {
      if (!isCurrentAction()) return
      if (error instanceof WorkforceDiscardError) {
        toast.error(t(
          error.code === "workforce_has_runs"
            ? "builds.list.deleteDialog.discardHasRuns"
            : "builds.list.deleteDialog.discardNotAllowed",
          { name: reference.name },
        ))
      } else {
        console.error("Failed to discard workforce:", error)
        toast.error(
          error instanceof Error
            ? error.message
            : t("builds.list.deleteDialog.discardFailed", {
                name: reference.name,
              }),
        )
      }
    } finally {
      if (isCurrentAction()) {
        setAgentDeletePendingAction(null)
      }
    }
  }

  const handleDelete = (agent: Agent) => {
    setAgentDeleteSession({
      target: { id: agent.id, name: agent.name },
      conflict: null,
    })
  }

  const matchesSearch = (agent: Agent) =>
    agent.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (agent.description && agent.description.toLowerCase().includes(searchTerm.toLowerCase()))

  const matchesStatusTab = (agent: Agent, tab: typeof statusTab) =>
    tab === "all" ? true : tab === "enabled" ? agent.status === "published" : agent.status === "draft"

  const statusTabs: SegmentedTabItem[] = (["all", "enabled", "drafts"] as const).map((tab) => ({
    id: tab,
    label: (
      <>
        {t(`builds.list.tabs.${tab}`)}
        <span className="ml-1.5 text-[10px] text-muted-foreground/70">
          {agents.filter((agent) => matchesStatusTab(agent, tab)).filter(matchesSearch).length}
        </span>
      </>
    ),
  }))

  const filteredAgents = agents
    .filter((agent) => matchesStatusTab(agent, statusTab))
    .filter(matchesSearch)
    .sort((a, b) =>
      sortMode === "name" ? a.name.localeCompare(b.name) : b.updated_at.localeCompare(a.updated_at)
    )

  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false)
  const [createPrompt, setCreatePrompt] = useState("")
  const [isStartingTask, setIsStartingTask] = useState(false)
  const createPromptVoiceInput = useVoiceInputControls()

  const handleCreate = () => {
    setIsCreateModalOpen(true)
  }

  const handleBuildWithPrompt = async () => {
    const rawPrompt = createPrompt
    const prompt = rawPrompt.trim()
    if (!prompt || activeTaskCreateAttemptRef.current !== null) return

    const attempt = ++taskCreateCounterRef.current
    activeTaskCreateAttemptRef.current = attempt
    const revision = draftRevisionRef.current
    const isCurrent = () => isMountedRef.current && activeTaskCreateAttemptRef.current === attempt

    setIsStartingTask(true)
    try {
      let task = null

      try {
        const selection = await resolveTaskLlmSelection()
        if (!isCurrent()) return

        if (selection.kind === "no_model") {
          toast.error(t("chatPage.input.noModelAlert"))
          return
        }
        if (selection.kind === "operational_error") throw selection.error

        const taskResponse = await apiRequest(`${getApiUrl()}/api/chat/task/create`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            title: normalizeTaskPromptTitle(rawPrompt),
            description: prompt,
            llm_ids: selection.llmIds,
          }),
        })

        if (!isCurrent()) return
        const parsed = await parseApiResponse(taskResponse)
        if (!isCurrent()) return

        task = taskResponse.ok ? parseTaskCreateCore(parsed.data) : null
        if (!task) throw new Error("Task create response is invalid")
      } catch (error) {
        if (isCurrent()) {
          console.error("Failed to start task from build modal:", error)
          toast.error(t("builds.list.createModal.startTaskFailed"))
        }
        return
      }

      if (!isCurrent() || !task) return

      try {
        if (!isCurrent()) return
        dispatch({ type: "RESET_STATE" })

        if (!isCurrent()) return
        setPendingMessage({
          message: prompt,
          files: [],
          targetTaskId: task.taskId,
        })

        if (!isCurrent()) return
        dispatch({ type: "TRIGGER_TASK_UPDATE" })

        if (!isCurrent()) return
        setTaskId(task.taskId)

        if (!isCurrent()) return
        setIsCreateModalOpen(false)

        if (!isCurrent()) return
        if (draftRevisionRef.current === revision) {
          draftRevisionRef.current += 1
          setCreatePrompt("")
        }
      } catch (error) {
        if (isCurrent()) console.error("Failed to commit task creation:", error)
      }
    } finally {
      if (isCurrent()) {
        activeTaskCreateAttemptRef.current = null
        setIsStartingTask(false)
      }
    }
  }

  const handleCreateModalOpenChange = (open: boolean) => {
    if (!open) {
      activeTaskCreateAttemptRef.current = null
      setIsStartingTask(false)
    }
    setIsCreateModalOpen(open)
  }

  const handleManualCreate = () => {
    handleCreateModalOpenChange(false)
    router.push("/build/new")
  }

  const createPromptVoiceInputLabel =
    createPromptVoiceInput.status === "recording"
      ? t("voiceInput.stop")
      : createPromptVoiceInput.status === "transcribing"
        ? t("voiceInput.transcribing")
        : t("voiceInput.start")
  const createPromptVoiceInputDisabled =
    createPromptVoiceInput.status === "transcribing" ||
    (createPromptVoiceInput.status === "idle" && isStartingTask)
  const handleCreatePromptVoiceInputClick = () => {
    if (createPromptVoiceInput.status === "recording") {
      createPromptVoiceInput.stopRecording()
      return
    }
    if (createPromptVoiceInput.status === "idle") {
      createPromptVoiceInput.startRecording(createPromptRef.current)
    }
  }

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <PageHeader
        title={t("builds.list.header.title")}
        description={t("builds.list.header.description")}
        actions={
          <>
            <SearchInput
              placeholder={t("builds.list.search.placeholder")}
              value={searchTerm}
              onChange={setSearchTerm}
              containerClassName="flex-1 sm:w-64"
            />
            <Button onClick={handleCreate} className="shrink-0 flex items-center gap-2 rounded-lg">
              <Plus className="h-4 w-4" />
              <span className="hidden sm:inline">{t("builds.list.header.create")}</span>
            </Button>
          </>
        }
      />

      {/* Main Content */}
      <div className="flex-1 px-4 md:px-6 py-6 space-y-6 overflow-auto">
        {/* Loading State */}
        {loading ? (
          <div className="flex items-center justify-center h-[400px]">
            <div className="text-muted-foreground">{t("common.loading")}</div>
          </div>
        ) : agents.length === 0 ? (
          <FeatureEmptyState
            icon={Bot}
            title={t("builds.emptyState.title")}
            description={t("builds.emptyState.description")}
            features={[
              {
                icon: FileText,
                title: t("builds.emptyState.features.instructions.title"),
                description: t("builds.emptyState.features.instructions.description")
              },
              {
                icon: Wrench,
                title: t("builds.emptyState.features.tools.title"),
                description: t("builds.emptyState.features.tools.description")
              },
              {
                icon: Database,
                title: t("builds.emptyState.features.knowledgeBase.title"),
                description: t("builds.emptyState.features.knowledgeBase.description")
              },
              {
                icon: Plug,
                title: t("builds.emptyState.features.connectors.title"),
                description: t("builds.emptyState.features.connectors.description")
              }
            ]}
            actionLabel={t("builds.emptyState.action")}
            onAction={handleCreate}
            className="h-full mt-4"
          />
        ) : (
          <>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <SegmentedTabs items={statusTabs} value={statusTab} onValueChange={(value) => setStatusTab(value as typeof statusTab)} />
              <button
                type="button"
                onClick={() => setSortMode((mode) => (mode === "updated" ? "name" : "updated"))}
                className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground"
              >
                {t(`builds.list.sort.${sortMode}`)}
                <ArrowUpDown className="h-3.5 w-3.5" />
              </button>
            </div>

            {/* List */}
            {filteredAgents.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-4">
                {filteredAgents.map((agent) => {
                  const resolvedLogoUrl = resolveAgentLogoUrl(agent.logo_url, getApiUrl())
                  const template = agent.template_id ? templatesById[agent.template_id] : undefined
                  const personaRole = template?.persona?.role
                  const category = template?.category
                  return (
                  <div
                    key={agent.id}
                    className="group relative flex flex-col rounded-[20px] border bg-card p-5 shadow-sm transition-all cursor-pointer hover:-translate-y-0.5 hover:shadow-md hover:border-primary/50"
                    onClick={() => router.push(`/build/${agent.id}`)}
                  >
                    <div className="flex flex-col items-start gap-3">
                      <PersonaAvatar
                        persona={{ name: agent.name, avatar: resolvedLogoUrl || template?.persona?.avatar }}
                        sizeClassName="h-20 w-20"
                        textClassName="text-2xl"
                        className="rounded-[22px] shadow-[0_0_0_4px_var(--card),0_0_0_6px_hsl(var(--primary)/0.15)]"
                      />
                      <div className="w-full min-w-0 pr-6">
                        <h3 className="font-bold text-xl leading-tight truncate" title={agent.name}>
                          {agent.name}
                        </h3>
                        {personaRole && (
                          <p className="text-[12.5px] text-muted-foreground truncate mt-0.5">{personaRole}</p>
                        )}
                        <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
                          <span className={`inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-full font-bold ${agent.status === 'published'
                            ? 'bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-400'
                            : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'
                            }`}>
                            <span className={`h-1.5 w-1.5 rounded-full ${agent.status === 'published' ? 'bg-green-500' : 'bg-gray-400'}`} />
                            {agent.status === 'published' ? t('builds.list.status.published') : t('builds.list.status.draft')}
                          </span>
                          {category && (
                            <span className={`inline-flex text-[11px] px-2 py-0.5 rounded-full font-medium ${pillClasses(category)}`}>
                              {categoryLabel(t, category)}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                        {(canPublishAgent(agent) || canDeleteAgent(agent) || canEditAgent(agent)) && (
                          <div className="absolute right-4 top-4" onClick={(e) => e.stopPropagation()}>
                            <Popover>
                              <PopoverTrigger asChild>
                                <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-foreground">
                                  <MoreVertical className="h-4 w-4" />
                                </Button>
                              </PopoverTrigger>
                              <PopoverContent align="end" className="w-40 p-1" onClick={(e) => e.stopPropagation()}>
                                <div className="flex flex-col">
                                  {canEditAgent(agent) && (
                                    <Button
                                      variant="ghost"
                                      className="justify-start px-2 py-1.5 h-auto font-normal text-sm"
                                      onClick={(e) => {
                                        e.stopPropagation()
                                        router.push(`/api-keys?agent=${agent.id}`)
                                      }}
                                    >
                                      <KeyRound className="mr-2 h-4 w-4" />
                                      {t('builds.list.actions.apiKey') || 'API Key'}
                                    </Button>
                                  )}
                                  {canEditAgent(agent) && (
                                    <Button
                                      variant="ghost"
                                      className="justify-start px-2 py-1.5 h-auto font-normal text-sm"
                                      onClick={(e) => {
                                        e.stopPropagation()
                                        setTriggersAgent(agent)
                                      }}
                                    >
                                      <Webhook className="mr-2 h-4 w-4" />
                                      {t('builds.list.actions.triggers') || 'Triggers'}
                                    </Button>
                                  )}
                                  {canEditAgent(agent) && (canPublishAgent(agent) || canDeleteAgent(agent)) && (
                                    <div className="h-px bg-border my-1 mx-1" />
                                  )}
                                  {canPublishAgent(agent) && (
                                    <Button
                                      variant="ghost"
                                      className="justify-start px-2 py-1.5 h-auto font-normal text-sm"
                                      onClick={(e) => {
                                        e.stopPropagation()
                                        if (agent.status === 'published') {
                                          void handlePublication(agent.id, "unpublish")
                                        } else {
                                          void handlePublication(agent.id, "publish")
                                        }
                                      }}
                                    >
                                      <Globe className="mr-2 h-4 w-4" />
                                      {agent.status === 'published' ? t('builds.list.actions.unpublish') : t('builds.list.actions.publish')}
                                    </Button>
                                  )}
                                  {canPublishAgent(agent) && canDeleteAgent(agent) && (
                                    <div className="h-px bg-border my-1 mx-1" />
                                  )}
                                  {canDeleteAgent(agent) && (
                                    <Button
                                      variant="ghost"
                                      className="justify-start px-2 py-1.5 h-auto font-normal text-sm text-destructive hover:text-destructive hover:bg-destructive/10"
                                      onClick={(e) => {
                                        e.stopPropagation()
                                        handleDelete(agent)
                                      }}
                                    >
                                      <Trash2 className="mr-2 h-4 w-4" />
                                      {t('builds.list.actions.delete')}
                                    </Button>
                                  )}
                                </div>
                              </PopoverContent>
                            </Popover>
                          </div>
                        )}

                      <p className="mt-4 flex-1 text-[13px] leading-relaxed text-muted-foreground line-clamp-3 min-h-[60px]">
                        {agent.description || t('builds.card.noDescription')}
                      </p>

                    <div className="mt-4 border-t pt-3.5" onClick={(e) => e.stopPropagation()}>
                      <div className="space-y-3.5">
                        <BuildAgentCardExtension agentId={agent.id} />
                        <div className="flex items-center gap-1.5">
                          {agent.status === 'published' ? (
                            <>
                              <Button
                                variant="default"
                                className="flex-1 rounded-full h-[34px] bg-blue-600 hover:bg-blue-700 text-white"
                                onClick={() => router.push(getAgentChatHref(agent))}
                              >
                                <MessageSquare className="mr-1.5 h-4 w-4" />
                                {t('builds.list.actions.chat')}
                              </Button>
                              {canEditAgent(agent) ? (
                                <>
                                  <Button
                                    variant="outline"
                                    className="rounded-full h-[34px] px-4"
                                    onClick={() => router.push(`/build/${agent.id}`)}
                                  >
                                    <Edit className="mr-1.5 h-4 w-4" />
                                    {t('builds.list.actions.edit')}
                                  </Button>
                                  <Button
                                    variant="outline"
                                    size="icon"
                                    className="rounded-full h-[34px] w-[34px] shrink-0"
                                    title={t('builds.list.actions.deploy')}
                                    onClick={() => setDeployAgent(agent)}
                                  >
                                    <Rocket className="h-4 w-4" />
                                  </Button>
                                </>
                              ) : (
                                <Button
                                  variant="outline"
                                  className={canRunAgent(agent) ? "rounded-full h-[34px] px-4" : "flex-1 w-full rounded-full h-[34px]"}
                                  onClick={() => router.push(`/build/${agent.id}`)}
                                >
                                  <Settings2 className="mr-1.5 h-4 w-4" />
                                  {t('builds.list.actions.viewConfig')}
                                </Button>
                              )}
                            </>
                          ) : (
                            canEditAgent(agent) ? (
                              <Button
                                variant="outline"
                                className="flex-1 w-full rounded-full h-[34px]"
                                onClick={() => router.push(`/build/${agent.id}`)}
                              >
                                <Edit className="mr-1.5 h-4 w-4" />
                                {t('builds.list.actions.edit')}
                              </Button>
                            ) : (
                              <Button
                                variant="outline"
                                className="flex-1 w-full rounded-full h-[34px]"
                                onClick={() => router.push(`/build/${agent.id}`)}
                              >
                                <Settings2 className="mr-1.5 h-4 w-4" />
                                {t('builds.list.actions.viewConfig')}
                              </Button>
                            )
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                  )
                })}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-[400px] text-center space-y-4 border rounded-lg bg-muted/10 border-dashed">
                <div className="h-12 w-12 rounded-full bg-muted flex items-center justify-center">
                  <Bot className="h-6 w-6 text-muted-foreground" />
                </div>
                <div className="space-y-2">
                  <h3 className="font-semibold text-lg">{t("builds.list.empty.title")}</h3>
                  <p className="text-muted-foreground max-w-sm mx-auto">
                    {t("builds.list.empty.description")}
                  </p>
                </div>
                <Button onClick={handleCreate} variant="outline">
                  <Plus className="mr-2 h-4 w-4" />
                  {t("builds.list.empty.create")}
                </Button>
              </div>
            )}
          </>
        )}
      </div>

      <AgentDeleteDialog
        target={agentDeleteSession?.target ?? null}
        conflict={agentDeleteSession?.conflict ?? null}
        pendingAction={agentDeletePendingAction}
        onOpenChange={(open) => {
          if (!open) setAgentDeleteSession(null)
        }}
        onConfirmDelete={confirmDeleteAgent}
        onDiscardWorkforce={handleDiscardWorkforce}
      />

      {/* Deploy Agent Dialog */}
      <DeployAgentDialog
        deployAgent={deployAgent}
        onClose={() => setDeployAgent(null)}
        onUpdate={(updatedAgent) => {
          setDeployAgent(updatedAgent)
          setAgents(agents.map(a => a.id === updatedAgent.id ? updatedAgent : a))
        }}
        onManageApiKey={() => { if (deployAgent) router.push(`/api-keys?agent=${deployAgent.id}`) }}
        onManageTriggers={() => { if (deployAgent) setTriggersAgent(deployAgent) }}
      />

      <AgentTriggersDialog
        agentId={triggersAgent?.id ?? null}
        agentName={triggersAgent?.name}
        open={triggersAgent !== null}
        onOpenChange={(open) => { if (!open) setTriggersAgent(null) }}
      />

      <Dialog open={isCreateModalOpen} onOpenChange={handleCreateModalOpenChange}>
        <DialogContent className="sm:max-w-[550px] gap-0 p-0 overflow-hidden bg-background shadow-lg rounded-xl">
          <DialogHeader className="px-6 py-5 border-b pr-10">
            <DialogTitle className="flex items-start sm:items-center gap-2 text-xl font-semibold">
              <Bot className="h-6 w-6 shrink-0 mt-0.5 sm:mt-0" />
              <span className="leading-tight text-left">{t("builds.list.createModal.title")}</span>
            </DialogTitle>
          </DialogHeader>

          <div className="p-6 space-y-6">
            {/* Option 1: By Describing It */}
            <div className="flex flex-col space-y-4 rounded-xl border border-border p-5 bg-card">
              <div className="flex items-start gap-4">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-indigo-50 text-indigo-500">
                  <Sparkles className="h-5 w-5" />
                </div>
                <div className="space-y-1">
                  <h3 className="font-semibold text-base">
                    {t("builds.list.createModal.describeTitle")}
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    {t("builds.list.createModal.describeDesc", { appName: branding.appName })}
                  </p>
                </div>
              </div>

              <div className="relative flex-1 w-full rounded-lg border border-input bg-background focus-within:ring-1 focus-within:ring-ring flex flex-col">
                <Textarea
                  ref={createPromptRef}
                  data-voice-input="false"
                  value={createPrompt}
                  onChange={(e) => { draftRevisionRef.current += 1; setCreatePrompt(e.target.value) }}
                  placeholder={t("builds.list.createModal.placeholder")}
                  className="min-h-[100px] flex-1 resize-none border-0 shadow-none focus-visible:ring-0"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey && !isStartingTask) {
                      e.preventDefault()
                      handleBuildWithPrompt()
                    }
                  }}
                />
                <div className="p-2 flex justify-end gap-2">
                  {createPromptVoiceInput.hasAsrModel && (
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      aria-label={createPromptVoiceInputLabel}
                      title={createPromptVoiceInputLabel}
                      className={
                        createPromptVoiceInput.status === "recording"
                          ? "h-9 w-9 rounded-full bg-red-500 text-white hover:bg-red-600 hover:text-white"
                          : "h-9 w-9 rounded-full text-muted-foreground hover:text-foreground"
                      }
                      disabled={createPromptVoiceInputDisabled}
                      onMouseDown={(event) => event.preventDefault()}
                      onClick={handleCreatePromptVoiceInputClick}
                    >
                      {createPromptVoiceInput.status === "recording" ? (
                        <Square className="h-3.5 w-3.5 fill-current" />
                      ) : createPromptVoiceInput.status === "transcribing" ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Mic className="h-4 w-4" />
                      )}
                    </Button>
                  )}
                  <Button
                    onClick={handleBuildWithPrompt}
                    disabled={!createPrompt.trim() || isStartingTask}
                    className="bg-indigo-400 hover:bg-indigo-500 text-white shadow-none shrink-0"
                  >
                    <Sparkles className="mr-2 h-4 w-4" />
                    {isStartingTask ? t("common.loading") : t("builds.list.createModal.buildBtn")}
                  </Button>
                </div>
              </div>
            </div>

            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t" />
              </div>
              <div className="relative flex justify-center text-xs">
                <span className="bg-background px-2 text-muted-foreground">
                  {t("common.or")}
                </span>
              </div>
            </div>

            {/* Option 2: Manually */}
            <div className="flex flex-col items-start gap-4 rounded-xl border border-border p-5 bg-card">
              <div className="flex items-start gap-4">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground">
                  <Settings2 className="h-5 w-5" />
                </div>
                <div className="space-y-1 flex-1">
                  <h3 className="font-semibold text-base">
                    {t("builds.list.createModal.manualTitle")}
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    {t("builds.list.createModal.manualDesc")}
                  </p>
                </div>
              </div>
              <Button
                variant="outline"
                onClick={handleManualCreate}
                className="gap-2"
              >
                {t("builds.list.createModal.manualBtn")}
                <ArrowRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default function BuildsPage() {
  return (
    <BuildPageExtensionProvider>
      <BuildsPageContent />
    </BuildPageExtensionProvider>
  )
}

"use client"

import React, { useCallback, useEffect, useRef, useState } from "react"
import Link from "next/link"
import { useRouter, useSearchParams } from "next/navigation"
import { ArrowLeft, LayoutDashboard, MessageSquare, Webhook } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { useApp } from "@/contexts/app-context-chat"
import type { Task } from "@/contexts/app-context-chat"
import { type TaskStatus } from "@/lib/task-status"
import {
    addWorkforceAgent,
    archiveWorkforce,
    createWorkforce,
    getWorkforce,
    listAgentOptions,
    publishWorkforce,
    removeWorkforceAgent,
    runWorkforce,
    runWorkforcePreview,
    unpublishWorkforce,
    updateWorkforce,
    updateWorkforceAgent,
} from "@/lib/workforces-api"
import type {
    WorkforceAgentOption,
    WorkforceAgentSummary,
    WorkforceDetail,
    WorkforceRunResponse,
    WorkforceWorker,
    WorkforceWorkerDraft,
} from "@/types/workforce"
import {
    normalizeWorkerSortOrder,
    WorkforceCanvas,
    WorkforceConfigPanel,
    WorkforceEditDialogs,
    WorkforceStatusBadge,
    useWorkforceEditDialogs,
    type WorkerEditState,
    type GetStartedStep,
} from "@/components/workforce"
import { AgentTriggersDialog } from "@/components/build/agent-triggers-dialog"
import { TaskConversationPanel } from "@/components/task/task-conversation-panel"
import { ResizableSplitLayout } from "@/components/layout/resizable-split-layout"
import { Button } from "@/components/ui/button"
import { toast } from "sonner"

type ActiveView = "configure" | "canvas"

interface LoadOptions {
    silent?: boolean
}

interface WorkforceBuilderProps {
    /** Omit to render the create flow (mirrors AgentBuilder's optional agentId). */
    workforceId?: string
}

function toFakeWorker(draft: WorkforceWorkerDraft, index: number, agents: WorkforceAgentOption[]): WorkforceWorker {
    const agent: WorkforceAgentSummary = agents.find((a) => a.id === draft.agent_id) ?? {
        id: draft.agent_id,
        name: "",
        description: null,
        logo_url: null,
        status: "published",
    }
    return {
        id: index,
        agent,
        alias: draft.alias || null,
        assignment_instructions: draft.assignment_instructions,
        source_type: draft.source_type,
        template_id: null,
        enabled: draft.enabled,
        sort_order: draft.sort_order,
        canvas_position: draft.canvas_position ?? null,
        created_at: null,
        updated_at: null,
    }
}

export function WorkforceBuilder({ workforceId }: WorkforceBuilderProps) {
    const { t, locale } = useI18n()
    const router = useRouter()
    const searchParams = useSearchParams()
    const { sendMessage, setTaskId, closeFilePreview, dispatch } = useApp()

    const [localId, setLocalId] = useState<string | undefined>(workforceId)
    const isEditMode = !!localId

    const [workforce, setWorkforce] = useState<WorkforceDetail | null>(null)
    const [agents, setAgents] = useState<WorkforceAgentOption[]>([])
    const [loading, setLoading] = useState(isEditMode)
    const [saving, setSaving] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [activeView, setActiveView] = useState<ActiveView>(workforceId ? "configure" : "canvas")
    const [triggersOpen, setTriggersOpen] = useState(false)
    const [getStartedCollapsed, setGetStartedCollapsed] = useState(true)
    const [hasSentTestMessage, setHasSentTestMessage] = useState(false)

    // Create-mode draft — local only, nothing hits the API until the first Create call.
    const [draftName, setDraftName] = useState("")
    const [draftDescription, setDraftDescription] = useState("")
    const [draftManagerAgentId, setDraftManagerAgentId] = useState("")
    const [draftWorkers, setDraftWorkers] = useState<WorkforceWorkerDraft[]>([])

    const previewTaskIdRef = useRef<number | null>(null)
    const isArchived = workforce?.status === "archived"

    const load = useCallback(async (options: LoadOptions = {}) => {
        if (!localId) return
        const { silent = false } = options
        try {
            if (!silent) setLoading(true)
            setError(null)
            const [workforceData, agentData] = await Promise.all([
                getWorkforce(localId),
                listAgentOptions(),
            ])
            setWorkforce(workforceData)
            setAgents(agentData)
        } catch (err) {
            const nextError = err instanceof Error ? err.message : t("workforces.errors.load")
            setError(nextError)
            toast.error(nextError)
        } finally {
            if (!silent) setLoading(false)
        }
    }, [localId, t])

    useEffect(() => {
        if (localId) {
            void load()
        } else {
            listAgentOptions()
                .then(setAgents)
                .catch((err) => {
                    const nextError = err instanceof Error ? err.message : t("workforces.errors.loadAgents")
                    toast.error(nextError)
                })
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [localId])

    useEffect(() => {
        if (searchParams.get("view") === "canvas") setActiveView("canvas")
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    const cleanupRef = useRef({ closeFilePreview, dispatch, setTaskId })
    cleanupRef.current = { closeFilePreview, dispatch, setTaskId }

    useEffect(() => {
        return () => {
            const { closeFilePreview: close, dispatch: d, setTaskId: set } = cleanupRef.current
            previewTaskIdRef.current = null
            close()
            d({ type: "CLEAR_MESSAGES" })
            d({ type: "SET_TRACE_EVENTS", payload: [] })
            d({ type: "SET_STEPS", payload: [] })
            d({ type: "SET_DAG_EXECUTION", payload: null })
            d({ type: "SET_CURRENT_TASK", payload: null })
            d({ type: "SET_HISTORY_LOADING", payload: false })
            set(null, { navigate: false })
        }
    }, [])

    const beginMutation = () => {
        setSaving(true)
        setError(null)
    }

    const handleSaveDetails = async (data: { name: string; description: string }) => {
        if (!isEditMode) {
            setDraftName(data.name)
            setDraftDescription(data.description)
            return
        }
        if (!localId) return
        try {
            beginMutation()
            const next = await updateWorkforce(localId, {
                name: data.name,
                description: data.description || null,
            })
            setWorkforce(next)
            toast.success(t("workforces.messages.updated"))
        } catch (err) {
            const nextError = err instanceof Error ? err.message : t("workforces.errors.update")
            setError(nextError)
            toast.error(nextError)
            throw err
        } finally {
            setSaving(false)
        }
    }

    const handleChangeLead = async (agentId: number) => {
        if (!isEditMode) {
            setDraftManagerAgentId(String(agentId))
            return
        }
        if (!localId) return
        try {
            beginMutation()
            const next = await updateWorkforce(localId, { manager_agent_id: agentId })
            setWorkforce(next)
            toast.success(t("workforces.messages.updated"))
        } catch (err) {
            const nextError = err instanceof Error ? err.message : t("workforces.errors.update")
            setError(nextError)
            toast.error(nextError)
            throw err
        } finally {
            setSaving(false)
        }
    }

    const handleAddWorker = async (agentId: number, instructions: string) => {
        if (!isEditMode) {
            setDraftWorkers((current) => [
                ...current,
                {
                    source_type: "existing",
                    agent_id: agentId,
                    alias: "",
                    assignment_instructions: instructions,
                    enabled: true,
                    sort_order: current.length + 1,
                },
            ])
            return
        }
        if (!localId) return
        try {
            beginMutation()
            await addWorkforceAgent(localId, {
                source_type: "existing",
                agent_id: agentId,
                assignment_instructions: instructions,
                enabled: true,
                sort_order: (workforce?.workers.length || 0) + 1,
            })
            await load({ silent: true })
            toast.success(t("workforces.messages.workerAdded"))
        } catch (err) {
            const nextError = err instanceof Error ? err.message : t("workforces.errors.addWorker")
            setError(nextError)
            toast.error(nextError)
            throw err
        } finally {
            setSaving(false)
        }
    }

    const handleSaveWorker = async (worker: WorkforceWorker, edit: WorkerEditState) => {
        if (!edit.assignment_instructions.trim()) return
        if (!isEditMode) {
            setDraftWorkers((current) =>
                current.map((draft, index) =>
                    index === worker.id
                        ? { ...draft, alias: edit.alias.trim(), assignment_instructions: edit.assignment_instructions.trim() }
                        : draft,
                ),
            )
            return
        }
        if (!localId) return
        try {
            beginMutation()
            const updated = await updateWorkforceAgent(localId, worker.id, {
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
            toast.success(t("workforces.messages.workerUpdated"))
        } catch (err) {
            const nextError = err instanceof Error ? err.message : t("workforces.errors.updateWorker")
            setError(nextError)
            toast.error(nextError)
            throw err
        } finally {
            setSaving(false)
        }
    }

    const handleRemoveWorker = async (worker: WorkforceWorker) => {
        if (!isEditMode) {
            setDraftWorkers((current) =>
                current
                    .filter((_, index) => index !== worker.id)
                    .map((draft, index) => ({ ...draft, sort_order: index + 1 })),
            )
            return
        }
        if (!localId) return
        try {
            beginMutation()
            await removeWorkforceAgent(localId, worker.id)
            await load({ silent: true })
            toast.success(t("workforces.messages.workerRemoved"))
        } catch (err) {
            const nextError = err instanceof Error ? err.message : t("workforces.errors.removeWorker")
            setError(nextError)
            toast.error(nextError)
            throw err
        } finally {
            setSaving(false)
        }
    }

    const publishCurrentWorkforce = async () => {
        if (!localId) return
        try {
            beginMutation()
            const next = await publishWorkforce(localId)
            setWorkforce(next)
            toast.success(t("workforces.messages.published"))
        } catch (err) {
            const nextError = err instanceof Error ? err.message : t("workforces.errors.publish")
            setError(nextError)
            toast.error(nextError)
        } finally {
            setSaving(false)
        }
    }

    const unpublishCurrentWorkforce = async () => {
        if (!localId) return
        try {
            beginMutation()
            const next = await unpublishWorkforce(localId)
            setWorkforce(next)
            toast.success(t("workforces.messages.unpublished"))
        } catch (err) {
            const nextError = err instanceof Error ? err.message : t("workforces.errors.unpublish")
            setError(nextError)
            toast.error(nextError)
        } finally {
            setSaving(false)
        }
    }

    const archiveCurrentWorkforce = async () => {
        if (!localId) return
        try {
            beginMutation()
            await archiveWorkforce(localId)
            const next = await getWorkforce(localId)
            setWorkforce(next)
            toast.success(t("workforces.messages.archived"))
        } catch (err) {
            const nextError = err instanceof Error ? err.message : t("workforces.errors.archive")
            setError(nextError)
            toast.error(nextError)
        } finally {
            setSaving(false)
        }
    }

    const canCreate = Boolean(draftName.trim() && draftManagerAgentId)
    const enabledDraftWorkers = draftWorkers.filter((worker) => worker.enabled)
    const canTestDraft = Boolean(draftManagerAgentId) && enabledDraftWorkers.length > 0

    const handleCreate = async () => {
        if (!canCreate || saving) return
        try {
            beginMutation()
            const created = await createWorkforce({
                name: draftName.trim(),
                description: draftDescription.trim() || undefined,
                manager_agent_id: Number(draftManagerAgentId),
                workers: draftWorkers,
            })
            setWorkforce(created)
            setLocalId(String(created.id))
            router.replace(`/workforces/${created.id}`)
        } catch (err) {
            const nextError = err instanceof Error ? err.message : t("workforces.errors.create")
            setError(nextError)
            toast.error(nextError)
        } finally {
            setSaving(false)
        }
    }

    const handleTestSendMessage = async (content: string, _config?: unknown, files?: (File & { file_id?: string })[]) => {
        if (isEditMode ? !localId : !canTestDraft) return

        let taskId = previewTaskIdRef.current
        if (taskId === -1) return

        try {
            if (!taskId) {
                previewTaskIdRef.current = -1
                const fileIds = (files || []).map(f => f.file_id).filter(Boolean) as string[]
                const result: WorkforceRunResponse = isEditMode && localId
                    ? await runWorkforce(localId, {
                        message: content,
                        files: fileIds,
                        is_preview: true,
                        is_visible: false,
                    })
                    : await runWorkforcePreview({
                        name: draftName.trim() || undefined,
                        description: draftDescription.trim() || undefined,
                        manager_agent_id: Number(draftManagerAgentId),
                        workers: enabledDraftWorkers.map((worker) => ({
                            agent_id: worker.agent_id,
                            alias: worker.alias || undefined,
                            assignment_instructions: worker.assignment_instructions,
                        })),
                        message: content,
                        files: fileIds,
                    })
                taskId = result.task_id
                if (!taskId) throw new Error("Invalid run response: missing task_id")
                previewTaskIdRef.current = taskId
                closeFilePreview()
                setTaskId(taskId, { navigate: false })
                const taskPayload: Task = {
                    id: String(taskId),
                    title: content.slice(0, 80),
                    description: content,
                    status: result.status as TaskStatus,
                    createdAt: new Date().toISOString(),
                    updatedAt: new Date().toISOString(),
                }
                dispatch({ type: "SET_CURRENT_TASK", payload: taskPayload })
                dispatch({ type: "TRIGGER_TASK_UPDATE" })
                setHasSentTestMessage(true)
            } else {
                await sendMessage(content, { force: true, targetTaskId: taskId }, files)
                setHasSentTestMessage(true)
            }
        } catch (err) {
            if (previewTaskIdRef.current === -1) previewTaskIdRef.current = null
            const nextError = err instanceof Error ? err.message : t("workforces.errors.run")
            toast.error(nextError)
        }
    }

    const manager: WorkforceAgentSummary | null = isEditMode
        ? workforce?.manager ?? null
        : agents.find((a) => String(a.id) === draftManagerAgentId) ?? null
    const workers: WorkforceWorker[] = isEditMode
        ? (workforce?.workers ?? [])
        : draftWorkers.map((draft, index) => toFakeWorker(draft, index, agents))
    const displayName = isEditMode ? (workforce?.name ?? "") : draftName
    const displayDescription = isEditMode ? (workforce?.description ?? "") : draftDescription

    const getStartedSteps: GetStartedStep[] = [
        { key: "name", label: t("workforces.getStarted.steps.name"), done: Boolean(displayName.trim()) },
        { key: "lead", label: t("workforces.getStarted.steps.lead"), done: !!manager },
        { key: "agents", label: t("workforces.getStarted.steps.agents"), done: workers.length > 0 },
        {
            key: "delegation",
            label: t("workforces.getStarted.steps.delegation"),
            done: workers.length > 0 && workers.every((worker) => !!worker.assignment_instructions?.trim()),
        },
        { key: "test", label: t("workforces.getStarted.steps.test"), done: hasSentTestMessage },
        {
            key: "publish",
            label: t("workforces.getStarted.steps.publish"),
            done: isEditMode && workforce?.status === "active",
        },
    ]

    const dialogs = useWorkforceEditDialogs({
        manager,
        workers,
        agents,
        onChangeLead: handleChangeLead,
        onAddWorker: handleAddWorker,
        onSaveWorker: handleSaveWorker,
        onRemoveWorker: handleRemoveWorker,
    })

    if (isEditMode && loading) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">{t("workforces.loading.detail")}</div>
    if (isEditMode && error && !workforce) return <div className="h-full overflow-y-auto p-4 text-red-500 sm:p-8">{error}</div>
    if (isEditMode && !workforce) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">{t("workforces.errors.notFound")}</div>

    return (
        <div className="flex h-full flex-col overflow-hidden">
            {/* Page header */}
            <div className="flex-none border-b bg-card/30 px-4 h-14 flex items-center gap-3">
                <Link href="/workforces" className="text-muted-foreground hover:text-foreground transition-colors">
                    <ArrowLeft className="h-4 w-4" />
                </Link>
                <span className="text-sm text-muted-foreground">{t("workforces.list.title")}</span>
                <span className="text-muted-foreground">/</span>
                <span className="font-medium text-sm truncate max-w-[200px]">{displayName || t("workforces.create.title")}</span>
                {isEditMode && workforce ? (
                    <WorkforceStatusBadge status={workforce.status} />
                ) : (
                    <span className="rounded-full bg-muted px-2.5 py-0.5 text-xs font-medium text-muted-foreground">
                        {t("workforces.create.unsavedBadge")}
                    </span>
                )}

                {/* Configure / Canvas toggle */}
                <div className="ml-4 flex items-center gap-1 rounded-lg border bg-muted/50 p-1">
                    <button
                        onClick={() => setActiveView("configure")}
                        className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                            activeView === "configure"
                                ? "bg-background text-foreground shadow-sm"
                                : "text-muted-foreground hover:text-foreground"
                        }`}
                    >
                        <LayoutDashboard className="h-3.5 w-3.5" />
                        {t("workforces.detail.configure")}
                    </button>
                    <button
                        onClick={() => setActiveView("canvas")}
                        className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                            activeView === "canvas"
                                ? "bg-background text-foreground shadow-sm"
                                : "text-muted-foreground hover:text-foreground"
                        }`}
                    >
                        <LayoutDashboard className="h-3.5 w-3.5 rotate-90" />
                        {t("workforces.canvas.title")}
                    </button>
                </div>

                <div className="flex-1" />

                {/* Action buttons */}
                <div className="flex items-center gap-2">
                    {error && <span className="text-xs text-red-500">{error}</span>}
                    {isEditMode && workforce ? (
                        <>
                            {!isArchived && (
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => setTriggersOpen(true)}
                                    disabled={saving}
                                >
                                    <Webhook className="mr-1.5 h-3.5 w-3.5" />
                                    {t("workforces.actions.triggers")}
                                </Button>
                            )}
                            {!isArchived && (
                                <Button variant="ghost" size="sm" onClick={archiveCurrentWorkforce} disabled={saving}>
                                    {t("workforces.actions.archive")}
                                </Button>
                            )}
                            {workforce.status === "active" ? (
                                <Button variant="outline" size="sm" onClick={unpublishCurrentWorkforce} disabled={saving || !!isArchived}>
                                    {t("workforces.actions.unpublish")}
                                </Button>
                            ) : (
                                <Button size="sm" onClick={publishCurrentWorkforce} disabled={saving || !!isArchived}>
                                    {saving ? t("workforces.loading.saving") : t("workforces.actions.publish")}
                                </Button>
                            )}
                        </>
                    ) : (
                        <Button size="sm" onClick={handleCreate} disabled={saving || !canCreate}>
                            {saving ? t("workforces.loading.creating") : t("workforces.actions.createTeam")}
                        </Button>
                    )}
                </div>
            </div>

            <WorkforceEditDialogs
                locale={locale}
                saving={saving}
                isArchived={!!isArchived}
                dialogs={dialogs}
            />

            {/* Body: main view + test panel */}
            <div className="flex-1 min-h-0 overflow-hidden">
                <ResizableSplitLayout
                    initialLeftWidth={65}
                    minLeftWidth={40}
                    maxLeftWidth={80}
                    leftPanel={
                        activeView === "configure" ? (
                            <div className="h-full overflow-y-auto">
                                <WorkforceConfigPanel
                                    name={displayName}
                                    description={displayDescription}
                                    manager={manager}
                                    workers={workers}
                                    isArchived={!!isArchived}
                                    saving={saving}
                                    onSaveDetails={handleSaveDetails}
                                    dialogs={dialogs}
                                    getStartedSteps={getStartedSteps}
                                    getStartedCollapsed={getStartedCollapsed}
                                    onToggleGetStarted={() => setGetStartedCollapsed((current) => !current)}
                                />
                            </div>
                        ) : (
                            <div className="h-full">
                                <WorkforceCanvas
                                    name={displayName}
                                    description={displayDescription}
                                    onSaveDetails={handleSaveDetails}
                                    manager={manager}
                                    workers={workers}
                                    isArchived={!!isArchived}
                                    dialogs={dialogs}
                                    getStartedSteps={getStartedSteps}
                                    getStartedCollapsed={getStartedCollapsed}
                                    onToggleGetStarted={() => setGetStartedCollapsed((current) => !current)}
                                />
                            </div>
                        )
                    }
                    rightPanel={
                        <div className="flex flex-col h-full bg-background border-l">
                            <div className="h-14 border-b flex items-center px-4 gap-2 bg-card/30 shrink-0">
                                <MessageSquare className="h-5 w-5 text-muted-foreground" />
                                <span className="font-medium text-sm">{t("workforces.run.testTitle")}</span>
                                {isEditMode && (
                                    <>
                                        <span className="ml-1 h-2 w-2 rounded-full bg-green-500" />
                                        <span className="text-xs text-green-600">{t("workforces.run.live")}</span>
                                    </>
                                )}
                            </div>
                            <div className="flex-1 min-h-0">
                                {isEditMode || canTestDraft ? (
                                    <TaskConversationPanel
                                        mode="embedded-preview"
                                        showTaskActions={false}
                                        showTokenUsage={false}
                                        showDagPreview={false}
                                        showTaskFiles={false}
                                        hideFileUpload={true}
                                        autoFocusInput={false}
                                        onSend={handleTestSendMessage}
                                    />
                                ) : (
                                    <div className="flex h-full items-center justify-center p-6 text-center text-sm text-muted-foreground">
                                        {t("workforces.run.createToTest")}
                                    </div>
                                )}
                            </div>
                        </div>
                    }
                />
            </div>

            {isEditMode && workforce && (
                <AgentTriggersDialog
                    agentId={null}
                    owner={{ kind: "workforce", id: localId ?? workforce.id }}
                    agentName={workforce.name}
                    open={triggersOpen}
                    onOpenChange={setTriggersOpen}
                />
            )}
        </div>
    )
}

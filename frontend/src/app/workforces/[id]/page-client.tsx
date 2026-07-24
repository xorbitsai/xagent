"use client"

import React, { useCallback, useEffect, useRef, useState } from "react"
import Link from "next/link"
import { useParams, useRouter } from "next/navigation"
import { ArrowLeft, History, LayoutDashboard, LayoutGrid, MessageSquare, Rocket, Share, Webhook } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { useApp } from "@/contexts/app-context-chat"
import type { Task } from "@/contexts/app-context-chat"
import { type TaskStatus } from "@/lib/task-status"
import {
    addWorkforceAgent,
    archiveWorkforce,
    getWorkforce,
    listAgentOptions,
    publishWorkforce,
    removeWorkforceAgent,
    runWorkforce,
    unpublishWorkforce,
    updateWorkforce,
    updateWorkforceAgent,
} from "@/lib/workforces-api"
import type {
    WorkforceAgentOption,
    WorkforceDetail,
    WorkforceRunResponse,
    WorkforceWorker,
} from "@/types/workforce"
import {
    normalizeWorkerSortOrder,
    DeployWorkforceDialog,
    WorkforceCanvas,
    WorkforceConfigPanel,
    WorkforceRunsList,
    WorkforceShareDialog,
    WorkforceWidgetDialog,
    WorkforceStatusBadge,
    type WorkerEditState,
} from "@/components/workforce"
import { AgentTriggersDialog } from "@/components/build/agent-triggers-dialog"
import { TaskConversationPanel } from "@/components/task/task-conversation-panel"
import { ResizableSplitLayout } from "@/components/layout/resizable-split-layout"
import { Button } from "@/components/ui/button"
import { toast } from "sonner"

type ActiveView = "configure" | "canvas" | "runs"

interface LoadOptions {
    silent?: boolean
}

export default function WorkforceDetailPage() {
    const { t } = useI18n()
    const params = useParams()
    const router = useRouter()
    const { sendMessage, setTaskId, closeFilePreview, dispatch } = useApp()
    const id = Array.isArray(params.id) ? params.id[0] : params.id

    const [workforce, setWorkforce] = useState<WorkforceDetail | null>(null)
    const [agents, setAgents] = useState<WorkforceAgentOption[]>([])
    const [loading, setLoading] = useState(true)
    const [saving, setSaving] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [activeView, setActiveView] = useState<ActiveView>("configure")
    const [deployOpen, setDeployOpen] = useState(false)
    const [triggersOpen, setTriggersOpen] = useState(false)
    const [isShareDialogOpen, setIsShareDialogOpen] = useState(false)
    const [isWidgetDialogOpen, setIsWidgetDialogOpen] = useState(false)

    const previewTaskIdRef = useRef<number | null>(null)
    const isArchived = workforce?.status === "archived"

    const load = useCallback(async (options: LoadOptions = {}) => {
        if (!id) return
        const { silent = false } = options
        try {
            if (!silent) setLoading(true)
            setError(null)
            const [workforceData, agentData] = await Promise.all([
                getWorkforce(id),
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
    }, [id, t])

    useEffect(() => {
        void load()
    }, [load])

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

    const handleSaveWorkforce = async (data: {
        name: string
        description: string
        managerAgentId: string
    }) => {
        if (!id) return
        try {
            beginMutation()
            const next = await updateWorkforce(id, {
                name: data.name,
                description: data.description || null,
                manager_agent_id: Number(data.managerAgentId),
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

    const handleAddWorker = async (agentId: number, instructions: string, alias?: string) => {
        if (!id) return
        try {
            beginMutation()
            await addWorkforceAgent(id, {
                source_type: "existing",
                agent_id: agentId,
                alias: alias || undefined,
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
        if (!id) return
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

    const handleRemoveWorker = async (workerId: number) => {
        if (!id) return
        try {
            beginMutation()
            await removeWorkforceAgent(id, workerId)
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
        if (!id) return
        try {
            beginMutation()
            const next = await publishWorkforce(id)
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
        if (!id) return
        try {
            beginMutation()
            const next = await unpublishWorkforce(id)
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
        if (!id) return
        try {
            beginMutation()
            await archiveWorkforce(id)
            const next = await getWorkforce(id)
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

    const handleTestSendMessage = async (content: string, _config?: unknown, files?: (File & { file_id?: string })[]) => {
        if (!id) return

        let taskId = previewTaskIdRef.current
        if (taskId === -1) return

        try {
            if (!taskId) {
                previewTaskIdRef.current = -1
                const result: WorkforceRunResponse = await runWorkforce(id, {
                    message: content,
                    files: (files || []).map(f => f.file_id).filter(Boolean) as string[],
                    is_preview: true,
                    is_visible: false,
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
            } else {
                await sendMessage(content, { force: true, targetTaskId: taskId }, files)
            }
        } catch (err) {
            if (previewTaskIdRef.current === -1) previewTaskIdRef.current = null
            const nextError = err instanceof Error ? err.message : t("workforces.errors.run")
            toast.error(nextError)
        }
    }

    if (loading) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">{t("workforces.loading.detail")}</div>
    if (error && !workforce) return <div className="h-full overflow-y-auto p-4 text-red-500 sm:p-8">{error}</div>
    if (!workforce) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">{t("workforces.errors.notFound")}</div>

    return (
        <div className="flex h-full flex-col overflow-hidden">
            {/* Page header */}
            <div className="flex-none border-b bg-card/30 px-4 h-14 flex items-center gap-3">
                <Link href="/workforces" className="text-muted-foreground hover:text-foreground transition-colors">
                    <ArrowLeft className="h-4 w-4" />
                </Link>
                <span className="text-sm text-muted-foreground">{t("workforces.list.title")}</span>
                <span className="text-muted-foreground">/</span>
                <span className="font-medium text-sm truncate max-w-[200px]">{workforce.name}</span>
                <WorkforceStatusBadge status={workforce.status} />

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
                    <button
                        onClick={() => setActiveView("runs")}
                        className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                            activeView === "runs"
                                ? "bg-background text-foreground shadow-sm"
                                : "text-muted-foreground hover:text-foreground"
                        }`}
                    >
                        <History className="h-3.5 w-3.5" />
                        {t("workforces.runs.title")}
                    </button>
                </div>

                <div className="flex-1" />

                {/* Action buttons */}
                <div className="flex items-center gap-2">
                    {error && <span className="text-xs text-red-500">{error}</span>}
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
                    {workforce.status === "active" && (
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setDeployOpen(true)}
                        >
                            <Rocket className="h-3.5 w-3.5 mr-1" />
                            {t("workforces.actions.deploy") || "Deploy"}
                        </Button>
                    )}
                    {!isArchived && (
                        <Button variant="outline" size="sm" onClick={() => setIsShareDialogOpen(true)} disabled={saving}>
                            <Share className="h-3.5 w-3.5 mr-1" />
                            {t("workforces.actions.share")}
                        </Button>
                    )}
                    {!isArchived && (
                        <Button variant="outline" size="sm" onClick={() => setIsWidgetDialogOpen(true)} disabled={saving}>
                            <LayoutGrid className="h-3.5 w-3.5 mr-1" />
                            {t("workforces.actions.embed")}
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
                </div>
            </div>

            {workforce && (
                <DeployWorkforceDialog
                    open={deployOpen}
                    workforceId={workforce.id}
                    workforceName={workforce.name}
                    onClose={() => setDeployOpen(false)}
                />
            )}
            <WorkforceShareDialog
                workforce={workforce}
                open={isShareDialogOpen}
                onClose={() => setIsShareDialogOpen(false)}
            />

            <WorkforceWidgetDialog
                workforce={workforce}
                open={isWidgetDialogOpen}
                onClose={() => setIsWidgetDialogOpen(false)}
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
                                    workforce={workforce}
                                    agents={agents}
                                    isArchived={!!isArchived}
                                    saving={saving}
                                    onSaveWorkforce={handleSaveWorkforce}
                                    onAddWorker={handleAddWorker}
                                    onSaveWorker={handleSaveWorker}
                                    onRemoveWorker={handleRemoveWorker}
                                />
                            </div>
                        ) : activeView === "canvas" ? (
                            <div className="h-full">
                                <WorkforceCanvas workforce={workforce} />
                            </div>
                        ) : (
                            <div className="h-full overflow-y-auto p-4 sm:p-6">
                                <div className="mx-auto max-w-3xl">
                                    <h2 className="text-sm font-semibold">{t("workforces.runs.historyTitle")}</h2>
                                    <p className="mt-1 text-xs text-muted-foreground">{t("workforces.runs.historyHint")}</p>
                                    <WorkforceRunsList
                                        className="mt-4"
                                        workforceId={id ?? workforce.id}
                                        onSelectRun={(run) => router.push(`/workforces/${id ?? workforce.id}/run?run=${run.id}`)}
                                    />
                                </div>
                            </div>
                        )
                    }
                    rightPanel={
                        <div className="flex flex-col h-full bg-background border-l">
                            <div className="h-14 border-b flex items-center px-4 gap-2 bg-card/30 shrink-0">
                                <MessageSquare className="h-5 w-5 text-muted-foreground" />
                                <span className="font-medium text-sm">{t("workforces.run.testTitle")}</span>
                                <span className="ml-1 h-2 w-2 rounded-full bg-green-500" />
                                <span className="text-xs text-green-600">{t("workforces.run.live")}</span>
                            </div>
                            <div className="flex-1 min-h-0">
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
                            </div>
                        </div>
                    }
                />
            </div>

            <AgentTriggersDialog
                agentId={null}
                owner={{ kind: "workforce", id: id ?? workforce.id }}
                agentName={workforce.name}
                open={triggersOpen}
                onOpenChange={setTriggersOpen}
            />
        </div>
    )
}

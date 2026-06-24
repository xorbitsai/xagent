"use client"

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useParams } from "next/navigation"
import { MessageSquare } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { useApp } from "@/contexts/app-context-chat"
import type { Task } from "@/contexts/app-context-chat"
import { type TaskStatus } from "@/lib/task-status"
import {
    addWorkforceAgent,
    applyWorkforceChanges,
    archiveWorkforce,
    getWorkforce,
    getWorkforceBuilderMessages,
    listAgentOptions,
    proposeWorkforceChanges,
    publishWorkforce,
    removeWorkforceAgent,
    runWorkforce,
    unpublishWorkforce,
    updateWorkforce,
    updateWorkforceAgent,
} from "@/lib/workforces-api"
import type {
    WorkforceAgentOption,
    WorkforceBuilderMessage,
    WorkforceBuilderPatch,
    WorkforceDetail,
    WorkforceRunResponse,
    WorkforceWorker,
} from "@/types/workforce"
import {
    buildWorkerEditState,
    normalizeWorkerSortOrder,
    workerEditState,
    WorkforceBuilderChat,
    WorkforceConfigPanel,
    type WorkerEditState,
} from "@/components/workforce"
import { TaskConversationPanel } from "@/components/task/task-conversation-panel"
import { ResizableThreeColumnLayout } from "@/components/layout/resizable-three-column-layout"
import { toast } from "sonner"

interface LoadOptions {
    silent?: boolean
}

interface SyncFormOptions {
    preserveEditableState?: boolean
}

function latestProposedAssistantMessage(
    messages: WorkforceBuilderMessage[],
): WorkforceBuilderMessage | null {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
        const item = messages[index]
        if (item.role === "assistant" && item.proposed_patch) {
            return item
        }
    }
    return null
}

export default function WorkforceDetailPage() {
    const { t } = useI18n()
    const params = useParams()
    const { sendMessage, setTaskId, closeFilePreview, dispatch } = useApp()
    const id = Array.isArray(params.id) ? params.id[0] : params.id
    const [workforce, setWorkforce] = useState<WorkforceDetail | null>(null)
    const [agents, setAgents] = useState<WorkforceAgentOption[]>([])
    const [loading, setLoading] = useState(true)
    const [saving, setSaving] = useState(false)
    const [error, setError] = useState<string | null>(null)

    // Config state
    const [name, setName] = useState("")
    const [description, setDescription] = useState("")
    const [managerAgentId, setManagerAgentId] = useState("")
    const [managerInstructions, setManagerInstructions] = useState("")
    const [workerEdits, setWorkerEdits] = useState<Record<number, WorkerEditState>>({})
    const [newWorkerAgentId, setNewWorkerAgentId] = useState("")
    const [newWorkerAlias, setNewWorkerAlias] = useState("")
    const [newWorkerInstructions, setNewWorkerInstructions] = useState("")

    // Builder state
    const [messages, setMessages] = useState<WorkforceBuilderMessage[]>([])
    const [submitting, setSubmitting] = useState(false)
    const [applying, setApplying] = useState(false)

    // Preview task ref
    const previewTaskIdRef = useRef<number | null>(null)

    const publishedAgents = useMemo(
        () => agents.filter((agent) => agent.status === "published"),
        [agents],
    )
    const isArchived = workforce?.status === "archived"

    const activeProposal = useMemo(() => latestProposedAssistantMessage(messages), [messages])

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
            const [workforceData, agentData, historyData] = await Promise.all([
                getWorkforce(id),
                listAgentOptions(),
                getWorkforceBuilderMessages(id).catch(() => ({ items: [] as WorkforceBuilderMessage[] })),
            ])
            setWorkforce(workforceData)
            setAgents(agentData)
            setMessages(historyData.items)
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

    const cleanupRef = useRef({ closeFilePreview, dispatch, setTaskId })
    cleanupRef.current = { closeFilePreview, dispatch, setTaskId }

    // Reset preview session on unmount
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
    }

    // ---- Workforce config actions ----

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
            toast.success(t("workforces.messages.updated"))
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
            toast.success(t("workforces.messages.workerAdded"))
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
            toast.success(t("workforces.messages.workerUpdated"))
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
            toast.success(t("workforces.messages.workerRemoved"))
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
            syncForm(next)
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
            syncForm(next)
            toast.success(t("workforces.messages.archived"))
        } catch (err) {
            const nextError = err instanceof Error ? err.message : t("workforces.errors.archive")
            setError(nextError)
            toast.error(nextError)
        } finally {
            setSaving(false)
        }
    }

    // ---- Builder actions ----

    const handleSubmit = async (message: string) => {
        if (!id) return
        try {
            setSubmitting(true)
            await proposeWorkforceChanges(id, { message })
            const history = await getWorkforceBuilderMessages(id)
            setMessages(history.items)
            toast.success(t("workforces.messages.proposalCreated"))
        } catch (err) {
            const nextError = err instanceof Error ? err.message : t("workforces.errors.proposeChanges")
            toast.error(nextError)
        } finally {
            setSubmitting(false)
        }
    }

    const handleApply = async (messageId: number, patch: WorkforceBuilderPatch) => {
        if (!id) return
        try {
            setApplying(true)
            const result = await applyWorkforceChanges(id, {
                message_id: messageId,
                proposed_patch: patch,
            })
            setWorkforce(result.workforce)
            syncForm(result.workforce, { preserveEditableState: false })
            const history = await getWorkforceBuilderMessages(id)
            setMessages(history.items)
            toast.success(t("workforces.messages.changesApplied"))
        } catch (err) {
            const nextError = err instanceof Error ? err.message : t("workforces.errors.applyChanges")
            toast.error(nextError)
        } finally {
            setApplying(false)
        }
    }

    // ---- Test preview actions ----

    const handleTestSendMessage = async (content: string, _config?: unknown, files?: (File & { file_id?: string })[]) => {
        if (!id) return

        let taskId = previewTaskIdRef.current
        // Prevent concurrent runWorkforce calls while the first one is pending
        if (taskId === -1) return

        try {
            if (!taskId) {
                previewTaskIdRef.current = -1

                // First message: create the workforce run
                const result: WorkforceRunResponse = await runWorkforce(id, {
                    message: content,
                    files: (files || []).map(f => f.file_id).filter(Boolean) as string[],
                    is_visible: false,
                })

                taskId = result.task_id
                if (!taskId) {
                    throw new Error("Invalid run response: missing task_id")
                }
                previewTaskIdRef.current = taskId

                // Connect to the task without navigating
                closeFilePreview()
                setTaskId(taskId, { navigate: false })

                // Set the current task info so the panel knows the status
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
                // Subsequent messages: send via websocket
                await sendMessage(content, { force: true, targetTaskId: taskId }, files)
            }
        } catch (err) {
            // Reset sentinel on failure so the user can retry
            if (previewTaskIdRef.current === -1) {
                previewTaskIdRef.current = null
            }
            const nextError = err instanceof Error ? err.message : t("workforces.errors.run")
            toast.error(nextError)
        }
    }

    if (loading) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">{t("workforces.loading.detail")}</div>
    if (error && !workforce) return <div className="h-full overflow-y-auto p-4 text-red-500 sm:p-8">{error}</div>
    if (!workforce) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">{t("workforces.errors.notFound")}</div>

    return (
        <div className="flex h-full flex-col overflow-hidden">
            {/* Status messages */}
            {error ? <div className="mx-auto w-full px-4 pt-2 text-sm text-red-500">{error}</div> : null}

            <div className="flex-1 min-h-0 w-full overflow-y-auto md:overflow-hidden">
                <ResizableThreeColumnLayout
                    showLeftPanel={true}
                    leftPanel={
                        <div className="h-full flex flex-col">
                            <WorkforceBuilderChat
                                messages={messages}
                                loading={loading}
                                submitting={submitting}
                                readOnly={isArchived}
                                readOnlyReason={isArchived ? t("workforces.builder.archivedReadOnly") : undefined}
                                activeProposal={activeProposal}
                                applying={applying}
                                onSubmit={handleSubmit}
                                onApplyPatch={handleApply}
                            />
                        </div>
                    }
                    middlePanel={
                        <div className="h-full overflow-y-auto">
                            <WorkforceConfigPanel
                                workforce={workforce}
                                name={name}
                                description={description}
                                managerAgentId={managerAgentId}
                                managerInstructions={managerInstructions}
                                managerOptions={managerOptions}
                                workerOptions={workerOptions}
                                workerEdits={workerEdits}
                                newWorkerAgentId={newWorkerAgentId}
                                newWorkerAlias={newWorkerAlias}
                                newWorkerInstructions={newWorkerInstructions}
                                isArchived={!!isArchived}
                                saving={saving}
                                onNameChange={setName}
                                onDescriptionChange={setDescription}
                                onManagerAgentIdChange={setManagerAgentId}
                                onManagerInstructionsChange={setManagerInstructions}
                                onSaveWorkforce={saveWorkforce}
                                onNewWorkerAgentIdChange={setNewWorkerAgentId}
                                onNewWorkerAliasChange={setNewWorkerAlias}
                                onNewWorkerInstructionsChange={setNewWorkerInstructions}
                                onAddWorker={addWorker}
                                onWorkerEditChange={(workerId, edit) => {
                                    setWorkerEdits((current) => {
                                        const worker = workforce?.workers.find((w) => w.id === workerId)
                                        const base = current[workerId] || {
                                            alias: worker?.alias || "",
                                            assignment_instructions: worker?.assignment_instructions || "",
                                            enabled: worker?.enabled ?? true,
                                            sort_order: String(worker?.sort_order ?? 1),
                                        }
                                        return {
                                            ...current,
                                            [workerId]: { ...base, ...edit },
                                        }
                                    })
                                }}
                                onSaveWorker={saveWorker}
                                onRemoveWorker={removeWorker}
                                onPublish={publishCurrentWorkforce}
                                onUnpublish={unpublishCurrentWorkforce}
                                onArchive={archiveCurrentWorkforce}
                            />
                        </div>
                    }
                    rightPanel={
                        <div className="flex flex-col flex-1 min-h-0 h-full bg-background border-l">
                            <div className="h-14 border-b flex items-center px-4 gap-2 bg-card/30">
                                <MessageSquare className="h-5 w-5 text-muted-foreground" />
                                <span className="font-medium">{t("workforces.run.testTitle")}</span>
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
                    initialLeftWidth={20}
                    initialMiddleWidth={50}
                    initialRightWidth={30}
                    minLeftWidth={15}
                    minMiddleWidth={45}
                    minRightWidth={20}
                />
            </div>
        </div>
    )
}

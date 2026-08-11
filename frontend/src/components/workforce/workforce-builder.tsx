"use client"

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import Link from "next/link"
import { useSearchParams } from "next/navigation"
import { ArrowLeft, LayoutDashboard, MessageSquare, Webhook } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { useApp } from "@/contexts/app-context-chat"
import type { Task } from "@/contexts/app-context-chat"
import { type TaskStatus } from "@/lib/task-status"
import {
    addWorkforceAgent,
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
    // Set by WorkforceConfigPanel while its Workforce Details form is
    // mid-edit; switching to Canvas unmounts the panel (and its local edit
    // state) with no autosave, so the switch is gated on this.
    const [isEditingDetails, setIsEditingDetails] = useState(false)

    // Shared by every action that would discard an in-progress Details edit
    // (switching views, navigating away, saving via Create) -- not just the
    // Configure/Canvas toggle.
    const confirmDiscardDetailsEdit = () =>
        !isEditingDetails || window.confirm(t("workforces.detail.discardEditConfirm"))

    const switchView = (view: ActiveView) => {
        if (view === activeView) return
        if (!confirmDiscardDetailsEdit()) return
        setActiveView(view)
    }

    const handleBackLinkClick = (e: React.MouseEvent) => {
        // At most one confirm() per click: an in-progress (uncommitted)
        // Details edit already implies leaving discards the whole draft, so
        // don't also ask about hasUnsavedDraft separately -- two native
        // confirm() popups for one navigation attempt reads as a glitch.
        if (isEditingDetails) {
            if (!confirmDiscardDetailsEdit()) e.preventDefault()
            return
        }
        if (!confirmDiscardDraft()) {
            e.preventDefault()
        }
    }

    // Create-mode draft — local only, nothing hits the API until the first Create call.
    const [draftName, setDraftName] = useState("")
    const [draftDescription, setDraftDescription] = useState("")
    const [draftManagerAgentId, setDraftManagerAgentId] = useState("")
    const [draftWorkers, setDraftWorkers] = useState<WorkforceWorkerDraft[]>([])

    // True once the user has put anything into an as-yet-uncreated draft.
    // Nothing here has hit the API, so an in-app nav-away, a browser
    // back/refresh, or a tab close all silently discard it with no recovery
    // path (PR review: create-mode draft loss on navigate-away).
    const hasUnsavedDraft =
        !isEditMode &&
        (draftName.trim() !== "" ||
            draftDescription.trim() !== "" ||
            draftManagerAgentId !== "" ||
            draftWorkers.length > 0)

    const confirmDiscardDraft = () =>
        !hasUnsavedDraft || window.confirm(t("workforces.create.discardDraftConfirm"))

    // Catches browser back/forward, refresh, and tab close — none of which
    // go through handleBackLinkClick's in-app Link guard. Also covers an
    // in-progress (uncommitted) Details edit, not just already-committed
    // draft state -- otherwise typing a name and hitting refresh before
    // clicking Save would silently lose it with no warning at all, even
    // though the in-app back-link guard already protects that exact case.
    useEffect(() => {
        if (!hasUnsavedDraft && !isEditingDetails) return
        const handleBeforeUnload = (e: BeforeUnloadEvent) => {
            e.preventDefault()
            e.returnValue = ""
        }
        window.addEventListener("beforeunload", handleBeforeUnload)
        return () => window.removeEventListener("beforeunload", handleBeforeUnload)
    }, [hasUnsavedDraft, isEditingDetails])

    const previewTaskIdRef = useRef<number | null>(null)
    // Bumped whenever handleCreate resets the preview state, so an in-flight
    // handleTestSendMessage call started before Create doesn't clobber the
    // reset with its now-orphaned (pre-save) result once its await resolves.
    const previewGenerationRef = useRef(0)
    // Independent of previewTaskIdRef's -1 sentinel: invalidatePreviewRun
    // resets that ref to null even while a preview-creation request is
    // in-flight (a config edit mid-send), which would otherwise let a second
    // handleTestSendMessage call slip past the "-1 means pending" guard and
    // fire a second, concurrent runWorkforcePreview/runWorkforce request.
    // This ref is only ever touched by handleTestSendMessage itself.
    const previewRequestInFlightRef = useRef(false)
    const isArchived = workforce?.status === "archived"

    // Set right before setLocalId on the create->save transition (handleCreate
    // already has fresh data via setWorkforce(created)): without this, the
    // [localId] mount effect below fires a non-silent load() on the very next
    // render, and the isEditMode && loading guard further down does a
    // top-level early return that unmounts the whole builder tree --
    // including the test-chat panel, losing an unsent draft message and
    // scroll position (PR review round 7, finding #1).
    const suppressNextLoadEffectRef = useRef(false)

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
            const silent = suppressNextLoadEffectRef.current
            suppressNextLoadEffectRef.current = false
            void load({ silent })
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

    // Drop any in-progress/completed test run so the next test message starts
    // a fresh one -- called after every persisted manager/worker/instructions
    // edit, since the running (or already-sent) preview task's snapshot was
    // frozen at send time and silently ignores later config changes.
    const invalidatePreviewRun = useCallback(() => {
        previewGenerationRef.current += 1
        previewTaskIdRef.current = null
        // The config just changed, so any prior test message was against a
        // now-stale snapshot -- the Get Started "test" step shouldn't stay
        // checked for a config nobody has actually tested yet.
        setHasSentTestMessage(false)
        closeFilePreview()
        dispatch({ type: "CLEAR_MESSAGES" })
        dispatch({ type: "SET_TRACE_EVENTS", payload: [] })
        dispatch({ type: "SET_STEPS", payload: [] })
        dispatch({ type: "SET_DAG_EXECUTION", payload: null })
        dispatch({ type: "SET_CURRENT_TASK", payload: null })
        dispatch({ type: "SET_HISTORY_LOADING", payload: false })
        setTaskId(null, { navigate: false })
    }, [closeFilePreview, dispatch, setTaskId])

    // useCallback: workforce-canvas.tsx's node-layout memo depends on this
    // reference, and a fresh one on every unrelated re-render (e.g. opening a
    // dialog) would otherwise rebuild the whole nodes array from scratch.
    const handleSaveDetails = useCallback(async (data: { name: string; description: string }) => {
        if (!isEditMode) {
            setDraftName(data.name)
            setDraftDescription(data.description)
            invalidatePreviewRun()
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
            invalidatePreviewRun()
            toast.success(t("workforces.messages.updated"))
        } catch (err) {
            const nextError = err instanceof Error ? err.message : t("workforces.errors.update")
            setError(nextError)
            toast.error(nextError)
            throw err
        } finally {
            setSaving(false)
        }
    }, [isEditMode, localId, invalidatePreviewRun, t])

    const handleChangeLead = async (agentId: number) => {
        if (!isEditMode) {
            setDraftManagerAgentId(String(agentId))
            invalidatePreviewRun()
            return
        }
        if (!localId) return
        try {
            beginMutation()
            const next = await updateWorkforce(localId, { manager_agent_id: agentId })
            setWorkforce(next)
            invalidatePreviewRun()
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
                    // Max-based, not length-based: collides with a survivor's
                    // sort_order once a middle draft worker has been removed.
                    sort_order: Math.max(0, ...current.map((w) => w.sort_order)) + 1,
                },
            ])
            invalidatePreviewRun()
            return
        }
        if (!localId) return
        try {
            beginMutation()
            // Length-based would collide with a survivor's sort_order once a
            // middle worker has been removed (e.g. 1,2,3 -> remove #2 leaves
            // 1,3, but length+1 computes 3 again) -- derive from the actual
            // max instead.
            const nextSortOrder =
                Math.max(0, ...(workforce?.workers.map((w) => w.sort_order ?? 0) ?? [])) + 1
            await addWorkforceAgent(localId, {
                source_type: "existing",
                agent_id: agentId,
                assignment_instructions: instructions,
                enabled: true,
                sort_order: nextSortOrder,
            })
            await load({ silent: true })
            invalidatePreviewRun()
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
            invalidatePreviewRun()
            return
        }
        if (!localId) return
        try {
            beginMutation()
            const updated = await updateWorkforceAgent(localId, worker.id, {
                alias: edit.alias.trim() || null,
                assignment_instructions: edit.assignment_instructions.trim(),
                enabled: edit.enabled,
                // normalizeWorkerSortOrder previously parsed edit.sort_order
                // back out of a form field, but that field was removed and
                // edit.sort_order (workforce-edit-dialogs.tsx) is always
                // seeded from worker.sort_order verbatim -- this send is
                // never actually a change, just a round-trip of the
                // existing value.
                sort_order: worker.sort_order ?? 1,
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
            invalidatePreviewRun()
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
            invalidatePreviewRun()
            return
        }
        if (!localId) return
        try {
            beginMutation()
            await removeWorkforceAgent(localId, worker.id)
            await load({ silent: true })
            invalidatePreviewRun()
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

    const canCreate = Boolean(draftName.trim() && draftManagerAgentId)
    const enabledDraftWorkers = draftWorkers.filter((worker) => worker.enabled)
    const canTestDraft = Boolean(draftManagerAgentId) && enabledDraftWorkers.length > 0

    // `saving` state alone never disables the button in time for a fast
    // double-click landing before the re-render commits (same hazard
    // AgentPickerDialog's `selecting` comment documents), which would create
    // a duplicate workforce row (PR review round 7, finding #7). Checked and
    // set synchronously, unlike state.
    const creatingRef = useRef(false)

    const handleCreate = async () => {
        if (!canCreate || saving || creatingRef.current) return
        if (!confirmDiscardDetailsEdit()) return
        creatingRef.current = true
        try {
            beginMutation()
            const created = await createWorkforce({
                name: draftName.trim(),
                description: draftDescription.trim() || undefined,
                manager_agent_id: Number(draftManagerAgentId),
                workers: draftWorkers,
            })
            // Clear the ephemeral pre-save preview run (workforce_id IS NULL,
            // frozen snapshot) so the next test message starts a fresh run
            // against the just-saved workforce instead of continuing to chat
            // into the stale draft's conversation, which would silently
            // ignore any config changes made after saving.
            invalidatePreviewRun()
            setWorkforce(created)
            suppressNextLoadEffectRef.current = true
            setLocalId(String(created.id))
            // router.replace/push would navigate across the /workforces/new
            // -> /workforces/[id] route-segment boundary -- separate pages
            // with no shared layout, so Next.js tears down this entire
            // component instance and mounts a fresh one under [id], taking
            // suppressNextLoadEffectRef and any in-flight test-chat state
            // with it (PR review round 8, finding #1 REOPENED: the ref-based
            // suppression above only works if this same instance survives).
            // The History API updates the address bar (so refresh/copy-link
            // still work) without invoking Next's router at all, keeping
            // this instance mounted -- the same fix AgentBuilder already
            // uses for the identical /build/new -> /build/[id] transition
            // (agent-builder-chat.tsx's replaceState call). replaceState
            // rather than pushState: this is a one-time create->save
            // transition, not a new history entry the user would ever want
            // to land back on -- pushState would leave /workforces/new on
            // the back-stack, so Back after a successful create would
            // return here instead of leaving the detail page.
            window.history.replaceState({}, "", `/workforces/${created.id}`)
        } catch (err) {
            const nextError = err instanceof Error ? err.message : t("workforces.errors.create")
            setError(nextError)
            toast.error(nextError)
        } finally {
            setSaving(false)
            creatingRef.current = false
        }
    }

    const handleTestSendMessage = async (content: string, _config?: unknown, files?: (File & { file_id?: string })[]) => {
        if (isEditMode ? !localId : !canTestDraft) return

        let taskId = previewTaskIdRef.current
        if (taskId === -1 || previewRequestInFlightRef.current) return

        const generationAtStart = previewGenerationRef.current

        previewRequestInFlightRef.current = true
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

                if (previewGenerationRef.current !== generationAtStart) {
                    // Create succeeded while this request was in flight: the
                    // run it started targets the now-discarded pre-save
                    // draft snapshot. Drop it instead of resurrecting it as
                    // the active conversation over handleCreate's reset.
                    return
                }
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
                if (previewGenerationRef.current !== generationAtStart) {
                    // Mirrors the !taskId branch's check above: an
                    // invalidatePreviewRun() (manager/worker/instructions
                    // edit) fired while this send was in flight, so the
                    // conversation this message just landed in is already
                    // discarded. Don't mark the Get Started "test" step done
                    // for a configuration that was invalidated out from
                    // under it (PR review round 7, finding #2).
                    return
                }
                setHasSentTestMessage(true)
            }
        } catch (err) {
            if (previewTaskIdRef.current === -1) previewTaskIdRef.current = null
            const nextError = err instanceof Error ? err.message : t("workforces.errors.run")
            toast.error(nextError)
        } finally {
            previewRequestInFlightRef.current = false
        }
    }

    const manager: WorkforceAgentSummary | null = isEditMode
        ? workforce?.manager ?? null
        : agents.find((a) => String(a.id) === draftManagerAgentId) ?? null
    // Memoized so the array reference is stable across unrelated re-renders
    // (e.g. opening a dialog) in create mode -- otherwise handleSaveDetails's
    // useCallback stabilization for the canvas node-layout memo is defeated
    // by a fresh `workers` value every time regardless.
    const createModeWorkers = useMemo(
        () => draftWorkers.map((draft, index) => toFakeWorker(draft, index, agents)),
        [draftWorkers, agents],
    )
    const workers: WorkforceWorker[] = isEditMode
        ? (workforce?.workers ?? [])
        : createModeWorkers
    const displayName = isEditMode ? (workforce?.name ?? "") : draftName
    const displayDescription = isEditMode ? (workforce?.description ?? "") : draftDescription

    const getStartedSteps: GetStartedStep[] = [
        { key: "name", label: t("workforces.getStarted.steps.name"), done: Boolean(displayName.trim()) },
        { key: "lead", label: t("workforces.getStarted.steps.lead"), done: !!manager },
        { key: "agents", label: t("workforces.getStarted.steps.agents"), done: workers.length > 0 },
        {
            key: "delegation",
            label: t("workforces.getStarted.steps.delegation"),
            // A worker added via one click auto-fills assignment_instructions
            // with the agent's own name (or id, as a last resort) when it has
            // no description to seed from -- that's a placeholder, not a
            // delegation rule the user actually wrote, so it shouldn't count.
            done:
                workers.length > 0 &&
                workers.every((worker) => {
                    const text = worker.assignment_instructions?.trim()
                    if (!text) return false
                    return text !== worker.agent.name && text !== String(worker.agent.id)
                }),
        },
        { key: "test", label: t("workforces.getStarted.steps.test"), done: hasSentTestMessage },
        {
            key: "publish",
            label: t("workforces.getStarted.steps.publish"),
            done: isEditMode && workforce?.status === "active",
        },
    ]

    // Lets AgentPickerDialog's create-from-template flow add the brand-new
    // agent straight into local state -- `agents` is otherwise only fetched
    // once (create mode has no [localId] to key a refetch off of), so any
    // by-id lookup against it (toFakeWorker, manager resolution below) would
    // miss a just-created agent and silently fall back to a blank/null
    // placeholder (PR review round 9, NEW-F2).
    const handleAgentCreated = useCallback((agent: WorkforceAgentOption) => {
        setAgents((current) => (current.some((a) => a.id === agent.id) ? current : [...current, agent]))
    }, [])

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
                <Link
                    href="/workforces"
                    onClick={handleBackLinkClick}
                    className="text-muted-foreground hover:text-foreground transition-colors"
                >
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
                        onClick={() => switchView("configure")}
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
                        onClick={() => switchView("canvas")}
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
                onAgentCreated={handleAgentCreated}
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
                                    onEditingDetailsChange={setIsEditingDetails}
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

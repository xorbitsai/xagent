"use client"

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { CheckCircle2, AlertCircle, FolderOpen, Loader2 } from "lucide-react"
import dagre from "dagre"
import { ChatInput, type ComposerAttachmentStatus } from "@/components/chat/ChatInput"
import { ChatMessage } from "@/components/chat/ChatMessage"
import type { AgentExecutionSummary } from "@/components/chat/TraceEventRenderer"
import { CompactionNotice } from "@/components/chat/CompactionNotice"
import { ContextUsageRing } from "@/components/chat/ContextUsageRing"
import { TokenUsageDisplay } from "@/components/chat/TokenUsageDisplay"
import { FilePreviewActionButtons } from "@/components/file/file-preview-action-buttons"
import { FilePreviewContent } from "@/components/file/file-preview-content"
import { TaskFileManager } from "@/components/file/task-file-manager"
import { CenterPanel } from "@/components/layout/center-panel"
import { PreviewSheet } from "@/components/preview-sheet"
import { Button } from "@/components/ui/button"
import { useApp, type SessionAttachmentsConfig, type TaskAutoResume } from "@/contexts/app-context-chat"
import { useFileAccess } from "@/contexts/file-access-context"
import { useI18n, type Translate } from "@/contexts/i18n-context"
import { isStreamingFinalAnswerMessage } from "@/lib/streaming-final-answer"
import { getProcessGroupIndex, getUserTimelineAnchors } from "@/lib/task-timeline"
import { resolveTraceProcessStatus } from "@/lib/trace-process-status"
import { cn } from "@/lib/utils"

export type TaskConversationPanelMode = "page" | "embedded-preview"

interface TaskConversationPanelProps {
  mode: TaskConversationPanelMode
  className?: string
  showTaskActions?: boolean
  showTokenUsage?: boolean
  showDagPreview?: boolean
  showTaskFiles?: boolean
  // Renders the execution trace (reasoning, tool arguments, tool output) above
  // each answer. Off for the embedded widget, where a visitor should see the
  // answer and a status line, not the run; share links keep it on.
  showProcessView?: boolean
  hideFileUpload?: boolean
  hideConfig?: boolean
  compactInput?: boolean
  autoFocusInput?: boolean
  uploadFile?: (file: File, params: { taskType: string }) => Promise<{ file_id: string }>
  deferFileUpload?: boolean
  onSend?: (message: string, config?: any, files?: File[]) => Promise<void> | void
  // Upload-only mode (see ChatInput): the composer collects files although
  // file reads stay disabled, and hands them to `onSend`.
  attachments?: SessionAttachmentsConfig
  attachmentStatuses?: ReadonlyMap<File, ComposerAttachmentStatus>
  onRetryAttachment?: (file: File) => void
  // Attachment work is in progress (an upload, or a send with files until it
  // has gone out and been accepted). It disables every file retry and counts
  // as a message in flight, which locks the composer only while the task is
  // pending or there is no task: while it is running, paused, waiting for an
  // answer, completed or failed, the composer stays open.
  attachmentsBusy?: boolean
  // The collected files, when the caller holds them so that they outlive this
  // panel. The panel then shows these and reports every change through
  // `onAttachmentFilesChange`.
  attachmentFiles?: File[]
  // Called with the collected files every time they change: when the visitor
  // adds or removes one, and once a send has completed.
  onAttachmentFilesChange?: (files: File[]) => void
  onAgentExecutionClick?: (execution: AgentExecutionSummary) => void
}

type CombinedItem = {
  id: string
  role: "user" | "assistant"
  content: string | React.ReactNode
  rawContent?: string
  timestamp: number
  status?: string
  isStreamingFinalAnswer?: boolean
  traceEvents?: any[]
  interactions?: any[]
  interactionRequestId?: string
  showEmptyStatus?: boolean
  processStatus?: string
  showInterruptedStatus?: boolean
  timelineOrder?: number
  isSystemNotice?: boolean
}

// Notice text for a task a system interruption paused, by its reason.
const INTERRUPTION_NOTICE_KEYS = {
  lease_expired: "taskInterruption.leaseExpired",
  persistence_failure: "taskInterruption.persistenceFailure",
  llm_unavailable: "taskInterruption.llmUnavailable",
} as const

const interruptionNoticeKey = (reason: string) =>
  INTERRUPTION_NOTICE_KEYS[reason as keyof typeof INTERRUPTION_NOTICE_KEYS]
  ?? "taskInterruption.paused"

// Notice text once automatic resume stopped, by its ``stop_reason``. The
// superseded copy deliberately does not offer "click resume": a later tick
// of the same scheduled trigger already ran, and resuming this one would
// duplicate its work.
const AUTO_RESUME_STOPPED_KEYS = {
  limit_reached: "taskInterruption.autoResumeStoppedLimit",
  expired: "taskInterruption.autoResumeStoppedExpired",
  superseded: "taskInterruption.autoResumeStoppedSuperseded",
  failed: "taskInterruption.autoResumeStoppedFailed",
  disabled: "taskInterruption.autoResumeStoppedDisabled",
} as const

const autoResumeNotice = (
  autoResume: TaskAutoResume,
  t: Translate,
  locale?: string,
): string => {
  if (autoResume.status === "resuming") return t("taskInterruption.autoResumeResuming")
  if (autoResume.status === "stopped") {
    return t(
      AUTO_RESUME_STOPPED_KEYS[autoResume.stopReason as keyof typeof AUTO_RESUME_STOPPED_KEYS]
      ?? "taskInterruption.autoResumeStopped",
    )
  }
  const at = autoResume.nextAttemptAt ? new Date(autoResume.nextAttemptAt) : undefined
  return at && !Number.isNaN(at.getTime())
    ? t("taskInterruption.autoResumeScheduledAt", {
      time: at.toLocaleTimeString(locale, { hour: "2-digit", minute: "2-digit" }),
    })
    : t("taskInterruption.autoResumeScheduled")
}

const toTimestampMs = (timestamp: unknown): number => {
  let time: number
  if (typeof timestamp === "number") {
    time = timestamp
  } else {
    const numeric = Number(timestamp)
    if (!Number.isNaN(numeric)) {
      time = numeric
    } else if (typeof timestamp === "string" || timestamp instanceof Date) {
      time = new Date(timestamp).getTime()
    } else {
      time = Number.NaN
    }
  }

  if (!Number.isFinite(time)) {
    return 0
  }

  return time < 100000000000 ? time * 1000 : time
}

const isDelegatedChildTraceEvent = (event: unknown): boolean => {
  if (!event || typeof event !== "object") return false
  const data = (event as { data?: unknown }).data
  return Boolean(
    data &&
    typeof data === "object" &&
    (data as Record<string, unknown>).source === "xagent-agent-tool-child",
  )
}

const isAgentDelegationToolName = (toolName: string): boolean => (
  /^agent_(?:[1-9]\d*|[a-z0-9_]+__a[1-9]\d*)$/.test(toolName) ||
  /^worker_[a-z0-9_]+__a[1-9]\d*$/.test(toolName) ||
  /^call_agent_[1-9]\d*$/.test(toolName)
)

const isWorkforceAgentToolTraceEvent = (event: unknown): boolean => {
  if (!event || typeof event !== "object") return false
  const traceEvent = event as { event_type?: unknown; data?: unknown }
  if (![
    "tool_execution_start",
    "tool_execution_end",
    "tool_execution_failed",
  ].includes(String(traceEvent.event_type || ""))) {
    return false
  }

  const data = traceEvent.data
  if (!data || typeof data !== "object") return false
  const record = data as Record<string, unknown>
  if (record.source === "xagent-agent-tool-child") return false
  const response = record.response && typeof record.response === "object"
    ? record.response as Record<string, unknown>
    : null
  const toolName = String(record.tool_name || response?.tool_name || "")
  return isAgentDelegationToolName(toolName)
}

const findWaitingPrompt = (currentTask: any, traceEvents: any[]) => {
  if (currentTask?.status !== "waiting_for_user") {
    return null
  }
  if (currentTask.waitingQuestion) {
    return currentTask.waitingQuestion
  }

  for (let i = traceEvents.length - 1; i >= 0; i--) {
    const event = traceEvents[i]
    if (event.event_type === "agent_message") {
      const expectsResponse = event.data?.expect_response === true
      const message = event.data?.message || event.data?.content
      if (expectsResponse && typeof message === "string" && message.trim()) {
        return message
      }
    }
    if (event.event_type === "react_task_end") {
      const result = event.data?.result
      if (result?.status === "waiting_for_user" && typeof result.message === "string" && result.message.trim()) {
        return result.message
      }
    }
  }

  return null
}

const findWaitingInteractions = (currentTask: any, traceEvents: any[]) => {
  if (currentTask?.status !== "waiting_for_user") {
    return undefined
  }
  if (currentTask.waitingRequestId) {
    return currentTask.waitingInteractions?.length
      ? currentTask.waitingInteractions
      : undefined
  }
  if (currentTask.waitingInteractions?.length) {
    return currentTask.waitingInteractions
  }

  for (let i = traceEvents.length - 1; i >= 0; i--) {
    const event = traceEvents[i]
    if (event.event_type === "agent_message") {
      const expectsResponse = event.data?.expect_response === true
      const interactions = event.data?.metadata?.interactions
      if (expectsResponse && Array.isArray(interactions) && interactions.length > 0) {
        return interactions
      }
    }
    if (event.event_type === "react_task_end") {
      const interactions = event.data?.result?.interactions
      if (Array.isArray(interactions) && interactions.length > 0) {
        return interactions
      }
    }
  }

  return undefined
}

export function TaskConversationPanel({
  mode,
  className,
  showTaskActions = mode === "page",
  showTokenUsage = mode === "page",
  showDagPreview = mode === "page",
  showTaskFiles = mode === "page",
  showProcessView = true,
  hideFileUpload = false,
  hideConfig = mode === "embedded-preview",
  compactInput = false,
  autoFocusInput = mode === "page",
  uploadFile,
  deferFileUpload = false,
  onSend,
  attachments,
  attachmentStatuses,
  onRetryAttachment,
  attachmentsBusy = false,
  attachmentFiles,
  onAttachmentFilesChange,
  onAgentExecutionClick,
}: TaskConversationPanelProps) {
  const {
    state,
    sendMessage,
    filesDisabled,
    voiceInputEnabled,
    taskControlsEnabled,
    pauseTask,
    resumeTask,
    canStopTask,
    stopTask,
    stopState,
    openFilePreview,
    closeFilePreview,
    requestStatus,
    dispatch,
    getFileDownloadUrl,
    isConversationResetPending,
    isMessageDeliveryPending,
    isSessionInteractionLocked,
    connectionNotice,
  } = useApp()
  const fileAccess = useFileAccess()
  const { t, locale } = useI18n()
  const [files, setFiles] = useState<File[]>([])
  const [dagPreviewOpen, setDagPreviewOpen] = useState(false)
  const [dagLayout, setDagLayout] = useState<"TB" | "LR">("TB")
  const [leftWidth, setLeftWidth] = useState(50)
  const [isDragging, setIsDragging] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const userMessageContextBadges = useMemo(() => {
    const hasBoundLocalBrowser = (
      (
        state.currentTask?.id === String(state.taskId)
        && state.currentTask.runtimeExtensionBindings?.includes("local_browser")
      )
      || Object.values(state.taskRuntimeExtensions || {}).some(
        (metadata) => metadata.kind === "local_browser",
      )
    )
    return hasBoundLocalBrowser
      ? [{
          kind: "computer_use" as const,
          label: t("chatPage.input.localBrowser.chipLabel"),
          detail: t("chatPage.input.localBrowser.label"),
        }]
      : []
  }, [
    state.currentTask?.id,
    state.currentTask?.runtimeExtensionBindings,
    state.taskId,
    state.taskRuntimeExtensions,
    t,
  ])
  const isFilePreviewOpen = !filesDisabled && state.filePreview.isOpen
  const anyPreviewOpen = mode === "page" && (isFilePreviewOpen || dagPreviewOpen)
  const hideDelegatedChildTraces = Boolean(onAgentExecutionClick)
  const shouldHideWorkforceInternalTrace = useCallback(
    (event: unknown) =>
      hideDelegatedChildTraces && (
        isDelegatedChildTraceEvent(event) ||
        isWorkforceAgentToolTraceEvent(event)
      ),
    [hideDelegatedChildTraces],
  )
  const managerTraceEvents = useMemo(
    () => Array.isArray(state.traceEvents)
      ? state.traceEvents.filter(
        (event: unknown) => !shouldHideWorkforceInternalTrace(event),
      )
      : [],
    [shouldHideWorkforceInternalTrace, state.traceEvents],
  )

  const openDagPreview = useCallback(() => {
    closeFilePreview()
    setDagPreviewOpen(true)
  }, [closeFilePreview])

  // Upload-only mode keeps collecting files although `filesDisabled` closes
  // every request that reads a file back.
  const filesBlocked = filesDisabled && !attachments

  const selectedFiles = attachmentFiles ?? files
  // The newest selection. A send reads it once it has completed, because the
  // visitor may have changed the files while it was in flight.
  const latestFilesRef = useRef(selectedFiles)
  latestFilesRef.current = selectedFiles

  const updateFiles = (next: File[]) => {
    latestFilesRef.current = next
    setFiles(next)
    onAttachmentFilesChange?.(next)
  }

  const handleSend = async (message: string, config?: any, filesToSend?: File[]) => {
    const outboundFiles = filesBlocked ? [] : (filesToSend || selectedFiles)
    await (onSend ?? sendMessage)(message, config, outboundFiles)
    // In session mode only the files this send carried leave the selection:
    // files added while it was in flight, and files kept hidden while uploads
    // are not offered, stay. Other modes clear the selection after a send.
    updateFiles(filesDisabled
      ? latestFilesRef.current.filter((file) => !outboundFiles.includes(file))
      : [])
  }

  const messageItems = useMemo<CombinedItem[]>(() => {
    const items = state.messages
      .filter((message: any) => message.role === "user" || message.isResult || message.isSystemNotice)
      .map((message: any) => {
        const id = message.id || `${message.role}-${toTimestampMs(message.timestamp)}`
        return {
          id,
          role: message.role,
          content: message.content,
          rawContent: message.rawContent,
          timestamp: toTimestampMs(message.timestamp),
          status: message.status,
          isStreamingFinalAnswer: isStreamingFinalAnswerMessage({
            id,
            role: message.role,
            isResult: message.isResult,
          }),
          traceEvents: message.traceEvents?.filter(
            (event: unknown) => !shouldHideWorkforceInternalTrace(event),
          ),
          interactions: message.interactions,
          interactionRequestId: message.interactionRequestId,
          isSystemNotice: message.isSystemNotice,
        }
      })

    items.sort((a, b) => a.timestamp - b.timestamp)
    return items
  }, [shouldHideWorkforceInternalTrace, state.messages])

  const lastMessageItem = [...messageItems]
    .reverse()
    .find((item) => !item.isSystemNotice)
  const hasFinalAssistantMessage =
    !!lastMessageItem &&
    lastMessageItem.role === "assistant" &&
    !(
      lastMessageItem.isStreamingFinalAnswer &&
      lastMessageItem.status === "failed"
    )

  const timelineItems = useMemo<CombinedItem[]>(() => {
    const sortedMessages = [...messageItems].sort((a, b) => a.timestamp - b.timestamp)
    const items: CombinedItem[] = sortedMessages.map((item, index) => ({
      ...item,
      traceEvents: undefined,
      timelineOrder: index * 2 + 1,
    }))

    type TimelineProcessEvent = {
      event_id?: string
      event_type?: string
      timestamp?: unknown
      [key: string]: unknown
    }

    const processEventsById = new Map<string, TimelineProcessEvent>()
    const addProcessEvent = (event: unknown, fallbackKey: string) => {
      if (!event || typeof event !== "object") {
        return
      }
      const processEvent = event as TimelineProcessEvent
      const eventKey =
        typeof processEvent.event_id === "string" && processEvent.event_id
          ? processEvent.event_id
          : fallbackKey
      if (!processEventsById.has(eventKey)) {
        processEventsById.set(eventKey, processEvent)
      }
    }

    if (managerTraceEvents.length > 0) {
      managerTraceEvents.forEach((event: unknown, index: number) => {
        addProcessEvent(event, `state-${index}`)
      })
    }
    messageItems.forEach((item) => {
      item.traceEvents?.forEach((event, index) => {
        addProcessEvent(event, `${item.id}-${index}`)
      })
    })

    const processEvents = Array.from(processEventsById.values())
    if (processEvents.length === 0) {
      return items
    }

    const userTurnAnchors = getUserTimelineAnchors(sortedMessages)
    const userTurnCount = userTurnAnchors.length

    const processGroups = new Map<number, TimelineProcessEvent[]>()
    processEvents.forEach((event) => {
      const eventTime = toTimestampMs(event.timestamp)
      // Events with a missing/invalid timestamp (eventTime === 0) must not be
      // grouped ahead of the first user turn; attach them to the current turn.
      const groupIndex =
        eventTime > 0
          ? getProcessGroupIndex(userTurnAnchors, eventTime)
          : userTurnCount
      const group = processGroups.get(groupIndex) || []
      group.push(event)
      processGroups.set(groupIndex, group)
    })

    const groupEntries = Array.from(processGroups.entries()).sort((a, b) => a[0] - b[0])
    const latestGroupIndex =
      groupEntries.length > 0 ? groupEntries[groupEntries.length - 1][0] : -1

    groupEntries.forEach(([groupIndex, events]) => {
      if (events.length === 0) {
        return
      }

      // Use only valid (>0) event timestamps; a missing/zero timestamp on any
      // event would otherwise drag the whole group to the top of the timeline.
      const validEventTimes = events
        .map((event) => toTimestampMs(event.timestamp))
        .filter((ts) => ts > 0)
      let groupTimestamp = validEventTimes.length
        ? Math.min(...validEventTimes)
        : 0
      // A process group belongs after the user turn it was anchored to; never
      // let it sort above that turn's message (fixes replay ordering).
      const anchor = userTurnAnchors[groupIndex - 1]
      if (anchor && groupTimestamp <= anchor.timestamp) {
        groupTimestamp = anchor.timestamp + 1
      }
      const firstEvent = events[0]
      const shouldShowEmptyStatus =
        !hasFinalAssistantMessage &&
        groupIndex === latestGroupIndex &&
        groupIndex >= userTurnCount
      const isCurrentTurnGroup =
        groupIndex >= userTurnCount && groupIndex === latestGroupIndex
      const inferredProcessStatus = resolveTraceProcessStatus({ traceEvents: events })
      // A turn the visitor stopped is recorded by the id of the user message
      // that opened it (the group's anchor). That record outranks both the
      // task status and the inferred status, for the current turn and for
      // earlier ones, so the group carries no "failed" status either and
      // never takes the turn's task-id failure reference. It is kept while
      // the same task stays bound and is cleared when the bound task changes
      // or the conversation or its messages are reset. When a bubble follows
      // the group, that bubble already says the turn was interrupted, so the
      // process does not repeat it; when nothing follows (the group still
      // shows the empty status), the process keeps its "Interrupted" line,
      // since nothing else on screen says so.
      const isVisitorStoppedGroup =
        !!anchor && state.stoppedTurnAnchorIds.includes(anchor.id)
      const processStatus = isVisitorStoppedGroup
        ? "interrupted"
        : isCurrentTurnGroup
        ? resolveTraceProcessStatus({
            processStatus: state.currentTask?.status,
            traceEvents: events,
          })
        // A later user turn closes this process group. If the old trace has no
        // terminal event (for example, a server restart interrupted a nested
        // Agent call), do not leave it spinning forever after replay.
        : inferredProcessStatus || "completed"

      items.push({
        id: `process-${groupIndex}-${firstEvent?.event_id || groupTimestamp}`,
        role: "assistant",
        content: null,
        timestamp: groupTimestamp,
        status:
          shouldShowEmptyStatus && !isVisitorStoppedGroup
            ? state.currentTask?.status
            : undefined,
        traceEvents: events,
        showEmptyStatus: shouldShowEmptyStatus,
        processStatus,
        showInterruptedStatus:
          isVisitorStoppedGroup && !shouldShowEmptyStatus ? false : undefined,
        timelineOrder: groupIndex * 2,
      })
    })

    items.sort(
      (a, b) =>
        a.timestamp - b.timestamp ||
        (a.timelineOrder ?? Number.MAX_SAFE_INTEGER) -
        (b.timelineOrder ?? Number.MAX_SAFE_INTEGER)
    )
    return items
  }, [
    hasFinalAssistantMessage,
    messageItems,
    state.currentTask?.status,
    state.stoppedTurnAnchorIds,
    managerTraceEvents,
  ])

  // One task-id reference per failed turn. With the trace hidden, a failed
  // turn can render both its failed process group and its failed result row,
  // so the result row takes the reference and the process group only gets it
  // when it is the turn's sole failure mark.
  const { failureReferenceItemIds, lastTurnHasFailureReference } = useMemo(() => {
    const ids = new Set<string>()
    let turn: CombinedItem[] = []
    let lastTurnHasReference = false
    const closeTurn = () => {
      const failed = turn.filter(
        (item) =>
          item.role === "assistant" &&
          (item.status === "failed" || item.processStatus === "failed")
      )
      const resultRows = failed.filter((item) => !Array.isArray(item.traceEvents))
      const chosen = resultRows.at(-1) ?? failed.at(-1)
      if (chosen) {
        ids.add(chosen.id)
      }
      lastTurnHasReference = Boolean(chosen)
      turn = []
    }
    timelineItems.forEach((item) => {
      if (item.role === "user") {
        closeTurn()
        return
      }
      turn.push(item)
    })
    closeTurn()
    return { failureReferenceItemIds: ids, lastTurnHasFailureReference: lastTurnHasReference }
  }, [timelineItems])

  const currentTurnTraceEvents = useMemo(() => {
    if (managerTraceEvents.length === 0) {
      return []
    }

    const sortedMessages = [...messageItems].sort((a, b) => a.timestamp - b.timestamp)
    const userTurnAnchors = getUserTimelineAnchors(sortedMessages)
    const userTurnCount = userTurnAnchors.length
    if (userTurnCount === 0) {
      return managerTraceEvents
    }

    return managerTraceEvents.filter((event: any) => {
      const eventTime = toTimestampMs(event?.timestamp)
      return getProcessGroupIndex(userTurnAnchors, eventTime) >= userTurnCount
    })
  }, [managerTraceEvents, messageItems])

  const waitingPrompt = useMemo(
    () => findWaitingPrompt(state.currentTask, managerTraceEvents as any[]),
    [managerTraceEvents, state.currentTask]
  )
  const waitingInteractions = useMemo(
    () => findWaitingInteractions(state.currentTask, managerTraceEvents as any[]),
    [managerTraceEvents, state.currentTask]
  )

  const activeWaitingMessageId = useMemo(() => {
    if (state.currentTask?.status !== "waiting_for_user") {
      return null
    }

    if (waitingPrompt) {
      const normalizedPrompt = waitingPrompt.trim()
      for (let i = messageItems.length - 1; i >= 0; i--) {
        const item = messageItems[i]
        if (item.role === "assistant" && typeof item.content === "string" && item.content.trim() === normalizedPrompt) {
          return item.id
        }
      }
    }

    for (let i = messageItems.length - 1; i >= 0; i--) {
      const item = messageItems[i]
      if (item.role === "assistant" && item.interactions && item.interactions.length > 0) {
        return item.id
      }
    }

    return null
  }, [messageItems, state.currentTask?.status, waitingPrompt])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView?.({ behavior: "smooth" })
  }, [state.messages, state.steps])

  useEffect(() => {
    if (isFilePreviewOpen) {
      setDagPreviewOpen(false)
    }
  }, [isFilePreviewOpen])

  useEffect(() => {
    if (filesDisabled) {
      setFiles([])
    }
  }, [filesDisabled])

  useEffect(() => {
    if (filesDisabled) return

    const handleFilePreviewEvent = (event: Event) => {
      const { filePath, fileName, allFiles, currentIndex } = (event as CustomEvent<any>).detail || {}
      if (!filePath) return
      if (Array.isArray(allFiles) && allFiles.length > 0) {
        openFilePreview(filePath, fileName, allFiles, typeof currentIndex === "number" ? currentIndex : 0)
      } else {
        openFilePreview(filePath, fileName)
      }
    }

    window.addEventListener("openFilePreview", handleFilePreviewEvent as EventListener)
    return () => window.removeEventListener("openFilePreview", handleFilePreviewEvent as EventListener)
  }, [filesDisabled, openFilePreview])

  const handleDownload = async () => {
    if (filesDisabled) return

    try {
      if (!state.filePreview.fileId) return
      const response = await fileAccess.request(getFileDownloadUrl(state.filePreview.fileId))
      if (!response.ok) {
        throw new Error(`Download failed: ${response.statusText}`)
      }

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.href = url
      link.download = state.filePreview.fileName || "download"
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error("Failed to download file:", error)
    }
  }

  const handleMouseDown = useCallback((event: React.MouseEvent) => {
    event.preventDefault()
    setIsDragging(true)
  }, [])

  const handleMouseMove = useCallback((event: MouseEvent) => {
    if (!isDragging || !containerRef.current) return
    const containerRect = containerRef.current.getBoundingClientRect()
    const nextWidth = Math.min(80, Math.max(20, ((event.clientX - containerRect.left) / containerRect.width) * 100))
    setLeftWidth(nextWidth)
  }, [isDragging])

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
  }, [])

  useEffect(() => {
    if (!isDragging) {
      document.body.style.cursor = ""
      document.body.style.userSelect = ""
      return
    }

    document.body.style.cursor = "col-resize"
    document.body.style.userSelect = "none"
    window.addEventListener("mousemove", handleMouseMove, { passive: true })
    window.addEventListener("mouseup", handleMouseUp)

    return () => {
      window.removeEventListener("mousemove", handleMouseMove)
      window.removeEventListener("mouseup", handleMouseUp)
      document.body.style.cursor = ""
      document.body.style.userSelect = ""
    }
  }, [isDragging, handleMouseMove, handleMouseUp])

  const dagreGraph = new dagre.graphlib.Graph()
  dagreGraph.setGraph({
    rankdir: dagLayout === "LR" ? "LR" : "TB",
    nodesep: 80,
    ranksep: 100,
    marginx: 20,
    marginy: 20,
  })
  dagreGraph.setDefaultEdgeLabel(() => "")

  const validSteps = state.steps.filter((step: any) => step && typeof step.id === "string" && step.id.trim() !== "")
  validSteps.forEach((step: any, index: number) => {
    try {
      dagreGraph.setNode(step.id, {
        label: step.name || `Step ${index + 1}`,
        width: 250,
        height: 200,
      })
    } catch (error) {
      console.error("Error adding node to dagre:", step, error)
    }
  })
  validSteps.forEach((step: any) => {
    if (!Array.isArray(step.dependencies)) return
    step.dependencies.forEach((depId: string) => {
      if (!depId || typeof depId !== "string" || depId.trim() === "") {
        return
      }
      const depStep = validSteps.find((candidate: any) => candidate.id === depId)
      if (depStep) {
        try {
          dagreGraph.setEdge(depId, step.id, {})
        } catch (error) {
          console.error("Error adding edge to dagre:", `${depId} -> ${step.id}`, error)
        }
      }
    })
  })

  let dagreLayoutSuccessful = true
  try {
    dagre.layout(dagreGraph)
  } catch (error) {
    dagreLayoutSuccessful = false
    console.error("Dagre layout failed:", error)
  }

  const dagNodes = state.steps.map((step: any, index: number) => {
    let node: any
    let safeNode: any
    if (!step.id || typeof step.id !== "string" || step.id.trim() === "") {
      safeNode = { x: (index % 3) * 300, y: Math.floor(index / 3) * 250 }
    } else if (dagreLayoutSuccessful) {
      try {
        node = dagreGraph.node(step.id)
        safeNode = typeof node === "object" && node !== null ? node : { x: (index % 3) * 300, y: Math.floor(index / 3) * 250 }
      } catch (error) {
        safeNode = { x: (index % 3) * 300, y: Math.floor(index / 3) * 250 }
      }
    } else {
      safeNode = { x: (index % 3) * 300, y: Math.floor(index / 3) * 250 }
    }
    return {
      id: step.id || `step-${index}`,
      type: "default",
      position: { x: (safeNode.x || 0) - 125, y: (safeNode.y || 0) - 100 },
      data: {
        label: step.name || `Step ${index + 1}`,
        status: step.status,
        description: step.description,
        tool_names: step.tool_names,
        started_at: step.started_at,
        completed_at: step.completed_at,
        result: step.result_data,
        conditional_branches: step.conditional_branches,
        required_branch: step.required_branch,
        is_conditional: step.is_conditional,
      },
    }
  })

  const validNodeIds = new Set(validSteps.map((step: any) => step.id))
  const dagEdges: any[] = []
  if (dagreLayoutSuccessful) {
    validSteps.forEach((step: any) => {
      if (!Array.isArray(step.dependencies)) {
        return
      }
      step.dependencies.forEach((depId: string) => {
        if (!depId || typeof depId !== "string" || depId.trim() === "") {
          return
        }
        if (validNodeIds.has(depId) && validNodeIds.has(step.id)) {
          dagEdges.push({
            id: `${depId}-${step.id}`,
            source: depId,
            target: step.id,
            data: {},
          })
        }
      })
    })
  }

  // "replanning" is a real, live phase the backend emits for a mid-run
  // replan (dag.py's on_dag_execution) - it's planning-like in exactly the
  // same way "planning" is (no NEW plan/graph to show yet), so it must show
  // the same in-progress indicator rather than falling through unrecognized.
  // Unlike "planning" (the very first plan, where dagNodes is naturally
  // still empty), a mid-run replan keeps the SAME turn_id as the run it's
  // revising, so app-context-chat's turn_id-based reset never clears the
  // prior plan's dagNodes - gating on dagNodes.length === 0 the same way
  // "planning" does would let the OLD, about-to-be-superseded graph keep
  // rendering as if it were still current for the whole replan. Show the
  // loading state unconditionally for "replanning" instead.
  const isPlanning =
    (dagNodes.length === 0 && state.dagExecution?.phase === "planning")
    || state.dagExecution?.phase === "replanning"
  const hasError = dagNodes.length === 0 && (state.dagExecution?.phase === "failed" || state.currentTask?.status === "failed")
  const shouldShowHistoryLoading =
    timelineItems.length === 0 &&
    state.isHistoryLoading
  // A user's own pause needs no explanation; a system interruption does.
  // Only for the viewed task: right after a switch currentTask can still be
  // the previous one.
  const interruptionReason = state.currentTask?.id === String(state.taskId)
    && state.currentTask.status === "paused"
    && state.currentTask.autoRecovery?.reason
    && state.currentTask.autoRecovery.reason !== "user_pause"
    ? state.currentTask.autoRecovery.reason
    : undefined
  // Automatic resume replaces the "click resume" advice while it applies to
  // the pause shown (the reducer drops it once the task leaves that fence).
  const autoResume = state.currentTask?.id === String(state.taskId)
    && state.currentTask.status === "paused"
    ? state.currentTask.autoResume
    : undefined
  // The virtual message belongs to the latest turn, so it follows that turn's
  // stopped record the same way the process groups above do, including taking
  // no task-id failure reference. It only appears while the turn has no answer
  // and no trace group of its own, so its "Interrupted" line is then the
  // turn's only sign of the interruption and stays on.
  const latestUserMessage = [...state.messages]
    .reverse()
    .find((message) => message.role === "user")
  const isLatestTurnVisitorStopped =
    !!latestUserMessage && state.stoppedTurnAnchorIds.includes(latestUserMessage.id)
  const shouldShowVirtualMessage =
    (state.isProcessing ||
      state.currentTask?.status === "paused" ||
      state.currentTask?.status === "waiting_for_user" ||
      state.currentTask?.status === "failed") &&
    !hasFinalAssistantMessage &&
    !timelineItems.some((item) => item.showEmptyStatus)

  return (
    <div
      ref={containerRef}
      className={cn(
        "h-full bg-background relative transition-all flex overflow-hidden",
        anyPreviewOpen ? "flex-row items-stretch" : "flex-col",
        mode === "embedded-preview" && "border-0",
        className
      )}
    >
      <div
        style={{ width: anyPreviewOpen ? `${leftWidth}%` : "100%" }}
        className={cn(anyPreviewOpen ? "" : "flex-1", "min-w-0 flex flex-col min-h-0 transition-[width] duration-0 relative")}
      >
        {connectionNotice === "lost" && (
          <div role="status" className="mx-4 mt-3 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:bg-amber-950 dark:text-amber-100">
            {t("clientErrors.connectionLost")}
          </div>
        )}
        {state.streamRecoveryTaskId === state.taskId && state.taskId !== null && (
          <div role="status" className="mx-4 mt-3 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:bg-amber-950 dark:text-amber-100">
            {t("sharedStream.interrupted")}
          </div>
        )}
        {(autoResume || interruptionReason) && (
          <div role="status" data-testid="task-interruption-notice" className="mx-4 mt-3 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:bg-amber-950 dark:text-amber-100">
            {autoResume
              ? autoResumeNotice(autoResume, t, locale)
              : t(interruptionNoticeKey(interruptionReason as string))}
          </div>
        )}
        <div className="flex-1 overflow-y-auto">
          <main className={cn("mx-auto px-4 relative z-0 transition-all", mode === "page" ? "container max-w-4xl py-8" : "max-w-3xl py-4")}>
            <div className={cn(mode === "page" ? "space-y-6 pb-4" : "space-y-4 pb-4")}>
              {shouldShowHistoryLoading ? (
                <div className={cn("flex flex-col items-center justify-center py-16 text-center", mode === "page" ? "min-h-[60vh]" : "min-h-[40vh]")}>
                  <div className="relative mb-6">
                    <div className="w-16 h-16 rounded-2xl bg-muted/30 flex items-center justify-center animate-pulse">
                      <Loader2 className="w-8 h-8 text-primary animate-spin" />
                    </div>
                  </div>
                  <h2 className="text-xl font-medium mb-2 text-foreground/80">
                    {t("common.loading")}
                  </h2>
                </div>
              ) : (
                <>
                  {timelineItems.map((item) => {
                    if (item.isSystemNotice) {
                      return <CompactionNotice key={item.id} text={item.content as string} />
                    }
                    // A failed final-answer stream always renders as failed; a
                    // verbatim terminal reason (#893) only needs the flag when
                    // the trace is hidden, so its raw text gets the generic
                    // replacement in ChatMessage.
                    const isFailedResultMessage =
                      item.status === "failed" &&
                      (item.isStreamingFinalAnswer || !showProcessView)
                    return (
                      <ChatMessage
                        key={item.id}
                        role={item.role}
                        content={item.content}
                        rawContent={item.rawContent}
                        traceEvents={item.traceEvents as any || []}
                        showProcessView={showProcessView}
                        showInterruptedStatus={item.showInterruptedStatus}
                        processStatus={item.status === "interrupted" ? "interrupted" : item.processStatus}
                        taskStatus={
                          isFailedResultMessage
                            ? "failed"
                            : item.showEmptyStatus
                              ? item.status
                              : undefined
                        }
                        timestamp={item.timestamp}
                        interactions={item.interactions}
                        interactionRequestId={item.interactionRequestId}
                        interactionsActive={item.id === activeWaitingMessageId}
                        showEmptyStatus={item.showEmptyStatus}
                        contextBadges={item.role === "user" ? userMessageContextBadges : undefined}
                        taskRuntimeExtensionMetadata={item.role === "user" ? {
                          bindings:
                            state.currentTask?.id === String(state.taskId)
                              ? state.currentTask.runtimeExtensionBindings || []
                              : [],
                          publicMetadata: state.taskRuntimeExtensions || {},
                        } : undefined}
                        onOpenExecutionPlan={showDagPreview ? openDagPreview : undefined}
                        onAgentExecutionClick={onAgentExecutionClick}
                        failureTaskId={
                          failureReferenceItemIds.has(item.id) ? state.taskId : undefined
                        }
                        connectionNotice={connectionNotice}
                      />
                    )
                  })}

                  {shouldShowVirtualMessage && (
                    <ChatMessage
                      role="assistant"
                      content={state.currentTask?.status === "waiting_for_user" ? waitingPrompt : null}
                      traceEvents={currentTurnTraceEvents as any || []}
                      showProcessView={showProcessView}
                      isVirtual
                      processStatus={isLatestTurnVisitorStopped ? "interrupted" : state.currentTask?.status}
                      taskStatus={state.currentTask?.status}
                      interactions={state.currentTask?.status === "waiting_for_user" ? waitingInteractions : undefined}
                      interactionRequestId={state.currentTask?.status === "waiting_for_user" ? state.currentTask.waitingRequestId : undefined}
                      interactionsActive={state.currentTask?.status === "waiting_for_user"}
                      onOpenExecutionPlan={showDagPreview ? openDagPreview : undefined}
                      onAgentExecutionClick={onAgentExecutionClick}
                      failureTaskId={
                        !isLatestTurnVisitorStopped &&
                        !lastTurnHasFailureReference &&
                        state.currentTask?.id === String(state.taskId) &&
                        state.currentTask.status === "failed"
                          ? state.taskId
                          : undefined
                      }
                      connectionNotice={connectionNotice}
                    />
                  )}
                </>
              )}
              {state.currentTask?.id === String(state.taskId)
                && state.currentTask.status === "completed"
                && !state.isProcessing
                && state.currentTask.completionOutcome && (
                <div role="status" className={cn("flex items-center gap-2 text-sm", state.currentTask.completionOutcome === "completed" ? "text-muted-foreground" : "text-amber-600 dark:text-amber-400")}>
                  {state.currentTask.completionOutcome === "completed" ? <CheckCircle2 className="h-4 w-4" /> : <AlertCircle className="h-4 w-4" />}
                  {t(`taskCompletion.${state.currentTask.completionOutcome}`)}
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </main>
        </div>

        <div className={cn("flex-shrink-0 z-10 glass", mode === "page" ? "pb-6" : "border-t bg-card/30 p-4")}>
          <div className={cn("mx-auto px-4", mode === "page" ? "container max-w-4xl" : "max-w-3xl px-0")}>
            {showTaskActions && (
              <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-3">
                <div className="flex flex-wrap items-center gap-2">
                  {showTaskFiles && !filesDisabled && (
                    <TaskFileManager taskId={state.taskId} onPreview={(fileId, fileName) => openFilePreview(fileId, fileName)}>
                      <Button type="button" variant="outline" className="h-auto rounded-xl bg-card/80 px-3 py-2 text-sm" title={t("files.header.title")}>
                        <FolderOpen className="w-3.5 h-3.5 mr-1" />
                        {t("files.header.title")}
                      </Button>
                    </TaskFileManager>
                  )}
                </div>

                {showTokenUsage && (
                  <div className="sm:ml-auto flex items-center gap-3">
                    {state.contextUsage && (
                      <div className="inline-flex items-center rounded-xl border bg-card/80 px-3 py-2 text-xs sm:text-sm">
                        <ContextUsageRing
                          tokens={state.contextUsage.tokens}
                          threshold={state.contextUsage.threshold}
                        />
                      </div>
                    )}
                    <TokenUsageDisplay taskId={state.taskId} isRunning={state.currentTask?.status === "running"} />
                  </div>
                )}
              </div>
            )}

            <ChatInput
              onSend={handleSend}
              currentInteractionRequestId={
                state.currentTask?.status === "waiting_for_user" &&
                state.currentTask.id === String(state.taskId)
                  ? state.currentTask.waitingRequestId
                  : undefined
              }
              isLoading={
                state.isProcessing
                || isConversationResetPending
                || isMessageDeliveryPending
                || isSessionInteractionLocked
                || attachmentsBusy
              }
              files={filesBlocked ? [] : selectedFiles}
              onFilesChange={filesBlocked ? undefined : updateFiles}
              showModeToggle={false}
              hideConfig={hideConfig}
              taskStatus={state.currentTask?.status}
              onPause={taskControlsEnabled ? pauseTask : undefined}
              onResume={taskControlsEnabled ? resumeTask : undefined}
              onStop={canStopTask ? stopTask : undefined}
              stopState={stopState}
              taskConfig={state.currentTask ? {
                model: state.currentTask.modelId || state.currentTask.modelName,
                smallFastModel: state.currentTask.smallFastModelId,
                visualModel: state.currentTask.visualModelId,
                compactModel: state.currentTask.compactModelId,
                executionMode: state.currentTask.executionMode,
              } : undefined}
              readOnlyConfig={true}
              filesDisabled={filesDisabled}
              attachments={attachments}
              attachmentStatuses={attachmentStatuses}
              onRetryAttachment={onRetryAttachment}
              attachmentsBusy={attachmentsBusy}
              voiceInputEnabled={voiceInputEnabled}
              hideFileUpload={hideFileUpload || filesBlocked}
              compact={compactInput}
              minHeightClass={compactInput ? "min-h-[44px]" : undefined}
              autoFocus={autoFocusInput}
              uploadFile={uploadFile}
              deferFileUpload={deferFileUpload}
            />
          </div>
        </div>
      </div>

      {anyPreviewOpen && (
        <div
          onMouseDown={handleMouseDown}
          className={cn("relative w-1 cursor-col-resize group z-[100] flex-shrink-0 hover:bg-primary/20 active:bg-primary/40 transition-colors", isDragging ? "bg-primary/40" : "bg-transparent")}
        >
          <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-[1px] bg-border group-hover:bg-primary group-hover:w-[2px] transition-all" />
          <div className="absolute inset-y-0 -left-2 -right-2" />
        </div>
      )}

      {anyPreviewOpen && (
        <div
          style={{ width: `${100 - leftWidth}%`, pointerEvents: isDragging ? "none" : "auto" }}
          className="flex-shrink-0 px-2 py-6 overflow-hidden relative"
        >
          <PreviewSheet
            open={isFilePreviewOpen || dagPreviewOpen}
            onOpenChange={(open) => {
              if (!open) {
                closeFilePreview()
                setDagPreviewOpen(false)
              }
            }}
            title={isFilePreviewOpen ? <>{state.filePreview.fileName}</> : t("chatPage.executionPlan.title")}
            actions={isFilePreviewOpen ? (
              <FilePreviewActionButtons
                viewMode={state.filePreview.viewMode}
                onViewModeChange={(mode) => dispatch({ type: "SET_FILE_PREVIEW_MODE", payload: mode })}
                fileName={state.filePreview.fileName || ""}
                onDownload={handleDownload}
                showText={true}
              />
            ) : null}
          >
            <div className="w-full h-full">
              {isFilePreviewOpen ? (
                <FilePreviewContent open={isFilePreviewOpen} />
              ) : (
                <CenterPanel
                  dagExecution={state.dagExecution}
                  dagNodes={dagNodes}
                  dagEdges={dagEdges as any}
                  dagLayout={dagLayout}
                  onLayoutChange={setDagLayout}
                  isPlanning={isPlanning}
                  hasError={hasError}
                  currentTaskStatus={state.currentTask?.status}
                  onRefresh={() => requestStatus()}
                  onFileClick={filesDisabled ? undefined : openFilePreview}
                />
              )}
            </div>
          </PreviewSheet>
        </div>
      )}

      {isDragging && <div className="fixed inset-0 z-[99] cursor-col-resize" />}
    </div>
  )
}

"use client"

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { FolderOpen, GitMerge, Loader2 } from "lucide-react"
import dagre from "dagre"
import { ChatInput } from "@/components/chat/ChatInput"
import { ChatMessage } from "@/components/chat/ChatMessage"
import { CompactionNotice } from "@/components/chat/CompactionNotice"
import { ContextUsageRing } from "@/components/chat/ContextUsageRing"
import { TokenUsageDisplay } from "@/components/chat/TokenUsageDisplay"
import { FilePreviewActionButtons } from "@/components/file/file-preview-action-buttons"
import { FilePreviewContent } from "@/components/file/file-preview-content"
import { TaskFileManager } from "@/components/file/task-file-manager"
import { CenterPanel } from "@/components/layout/center-panel"
import { PreviewSheet } from "@/components/preview-sheet"
import { Button } from "@/components/ui/button"
import { useApp } from "@/contexts/app-context-chat"
import { useI18n } from "@/contexts/i18n-context"
import { apiRequest } from "@/lib/api-wrapper"
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
  hideFileUpload?: boolean
  hideConfig?: boolean
  compactInput?: boolean
  autoFocusInput?: boolean
  uploadFile?: (file: File, params: { taskType: string }) => Promise<{ file_id: string }>
  deferFileUpload?: boolean
  onSend?: (message: string, config?: any, files?: File[]) => Promise<void> | void
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
  showEmptyStatus?: boolean
  processStatus?: string
  timelineOrder?: number
  isSystemNotice?: boolean
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
  hideFileUpload = false,
  hideConfig = mode === "embedded-preview",
  compactInput = false,
  autoFocusInput = mode === "page",
  uploadFile,
  deferFileUpload = false,
  onSend,
}: TaskConversationPanelProps) {
  const { state, sendMessage, pauseTask, resumeTask, openFilePreview, closeFilePreview, requestStatus, dispatch, getFileDownloadUrl } = useApp()
  const { t } = useI18n()
  const [files, setFiles] = useState<File[]>([])
  const [dagPreviewOpen, setDagPreviewOpen] = useState(false)
  const [dagLayout, setDagLayout] = useState<"TB" | "LR">("TB")
  const [leftWidth, setLeftWidth] = useState(50)
  const [isDragging, setIsDragging] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const anyPreviewOpen = mode === "page" && (state.filePreview.isOpen || dagPreviewOpen)

  const handleSend = async (message: string, config?: any, filesToSend?: File[]) => {
    await (onSend ?? sendMessage)(message, config, filesToSend || files)
    setFiles([])
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
          traceEvents: message.traceEvents,
          interactions: message.interactions,
          isSystemNotice: message.isSystemNotice,
        }
      })

    items.sort((a, b) => a.timestamp - b.timestamp)
    return items
  }, [state.messages])

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

    if (Array.isArray(state.traceEvents)) {
      state.traceEvents.forEach((event: unknown, index: number) => {
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
      const processStatus = isCurrentTurnGroup
        ? resolveTraceProcessStatus({
            processStatus: state.currentTask?.status,
            traceEvents: events,
          })
        : resolveTraceProcessStatus({ traceEvents: events })

      items.push({
        id: `process-${groupIndex}-${firstEvent?.event_id || groupTimestamp}`,
        role: "assistant",
        content: null,
        timestamp: groupTimestamp,
        status: shouldShowEmptyStatus ? state.currentTask?.status : undefined,
        traceEvents: events,
        showEmptyStatus: shouldShowEmptyStatus,
        processStatus,
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
    state.traceEvents,
  ])

  const currentTurnTraceEvents = useMemo(() => {
    if (!Array.isArray(state.traceEvents) || state.traceEvents.length === 0) {
      return []
    }

    const sortedMessages = [...messageItems].sort((a, b) => a.timestamp - b.timestamp)
    const userTurnAnchors = getUserTimelineAnchors(sortedMessages)
    const userTurnCount = userTurnAnchors.length
    if (userTurnCount === 0) {
      return state.traceEvents
    }

    return state.traceEvents.filter((event: any) => {
      const eventTime = toTimestampMs(event?.timestamp)
      return getProcessGroupIndex(userTurnAnchors, eventTime) >= userTurnCount
    })
  }, [messageItems, state.traceEvents])

  const waitingPrompt = useMemo(
    () => findWaitingPrompt(state.currentTask, state.traceEvents as any[]),
    [state.currentTask, state.traceEvents]
  )
  const waitingInteractions = useMemo(
    () => findWaitingInteractions(state.currentTask, state.traceEvents as any[]),
    [state.currentTask, state.traceEvents]
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
    if (state.filePreview.isOpen) {
      setDagPreviewOpen(false)
    }
  }, [state.filePreview.isOpen])

  useEffect(() => {
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
  }, [openFilePreview])

  const handleDownload = async () => {
    try {
      if (!state.filePreview.fileId) return
      const response = await apiRequest(getFileDownloadUrl(state.filePreview.fileId))
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

  const isPlanning = dagNodes.length === 0 && state.dagExecution?.phase === "planning"
  const hasError = dagNodes.length === 0 && (state.dagExecution?.phase === "failed" || state.currentTask?.status === "failed")
  const shouldShowHistoryLoading =
    timelineItems.length === 0 &&
    state.isHistoryLoading
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
                    const isFailedFinalAnswerStream =
                      item.isStreamingFinalAnswer && item.status === "failed"
                    return (
                      <ChatMessage
                        key={item.id}
                        role={item.role}
                        content={item.content}
                        rawContent={item.rawContent}
                        traceEvents={item.traceEvents as any || []}
                        showProcessView={true}
                        processStatus={item.processStatus}
                        taskStatus={
                          isFailedFinalAnswerStream
                            ? "failed"
                            : item.showEmptyStatus
                              ? item.status
                              : undefined
                        }
                        timestamp={item.timestamp}
                        interactions={item.interactions}
                        interactionsActive={item.id === activeWaitingMessageId}
                        showEmptyStatus={item.showEmptyStatus}
                      />
                    )
                  })}

                  {shouldShowVirtualMessage && (
                    <ChatMessage
                      role="assistant"
                      content={state.currentTask?.status === "waiting_for_user" ? waitingPrompt : null}
                      traceEvents={currentTurnTraceEvents as any || []}
                      showProcessView={true}
                      isVirtual
                      processStatus={state.currentTask?.status}
                      taskStatus={state.currentTask?.status}
                      interactions={state.currentTask?.status === "waiting_for_user" ? waitingInteractions : undefined}
                      interactionsActive={state.currentTask?.status === "waiting_for_user"}
                    />
                  )}
                </>
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
                  {showDagPreview && state.currentTask?.isDag !== false && (
                    <Button
                      type="button"
                      variant="outline"
                      className="h-auto rounded-xl bg-card/80 px-3 py-2 text-sm"
                      onClick={() => {
                        closeFilePreview()
                        setDagPreviewOpen(true)
                      }}
                      title={t("chatPage.executionPlan.title")}
                    >
                      <GitMerge className="w-3.5 h-3.5 mr-1" />
                      {t("chatPage.executionPlan.title")}
                    </Button>
                  )}

                  {showTaskFiles && (
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
              isLoading={state.isProcessing}
              files={files}
              onFilesChange={setFiles}
              showModeToggle={false}
              hideConfig={hideConfig}
              taskStatus={state.currentTask?.status}
              onPause={pauseTask}
              onResume={resumeTask}
              taskConfig={state.currentTask ? {
                model: state.currentTask.modelId || state.currentTask.modelName,
                smallFastModel: state.currentTask.smallFastModelId,
                visualModel: state.currentTask.visualModelId,
                compactModel: state.currentTask.compactModelId,
                executionMode: state.currentTask.executionMode,
              } : undefined}
              readOnlyConfig={true}
              hideFileUpload={hideFileUpload}
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
            open={state.filePreview.isOpen || dagPreviewOpen}
            onOpenChange={(open) => {
              if (!open) {
                closeFilePreview()
                setDagPreviewOpen(false)
              }
            }}
            title={state.filePreview.isOpen ? <>{state.filePreview.fileName}</> : t("chatPage.executionPlan.title")}
            actions={state.filePreview.isOpen ? (
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
              {state.filePreview.isOpen ? (
                <FilePreviewContent open={state.filePreview.isOpen} />
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
                  onFileClick={openFilePreview}
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

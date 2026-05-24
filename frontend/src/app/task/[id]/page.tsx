"use client";

import { useState, useRef, useEffect, Suspense, useCallback, useMemo } from "react";
import { GitMerge, Bot, ArrowLeft, Loader2, Sparkles, FolderOpen } from "lucide-react";
import { ChatMessage } from "@/components/chat/ChatMessage";
import { ChatInput } from "@/components/chat/ChatInput";
import { Button } from "@/components/ui/button";
import { useApp } from "@/contexts/app-context-chat";
import { useI18n } from "@/contexts/i18n-context";
import { useParams, useRouter } from "next/navigation"
import { PreviewSheet } from "@/components/preview-sheet";
import { FilePreviewContent } from "@/components/file/file-preview-content";
import { TokenUsageDisplay } from "@/components/chat/TokenUsageDisplay";
import { TaskFileManager } from "@/components/file/task-file-manager";
import { getApiUrl } from "@/lib/utils";
import { apiRequest } from "@/lib/api-wrapper";
import { isStreamingFinalAnswerMessage } from "@/lib/streaming-final-answer";
import { getProcessGroupIndex } from "@/lib/task-timeline";
import type React from "react";
import dagre from "dagre"
import { CenterPanel } from "@/components/layout/center-panel"
import { FilePreviewActionButtons } from "@/components/file/file-preview-action-buttons"

const toTimelineTime = (ts: unknown): number => {
  let time: number;
  if (typeof ts === 'number') {
    time = ts;
  } else {
    const n = Number(ts);
    if (!isNaN(n)) {
      time = n;
    } else if (typeof ts === 'string' || ts instanceof Date) {
      time = new Date(ts).getTime();
    } else {
      time = Number.NaN;
    }
  }

  if (!Number.isFinite(time)) {
    return 0;
  }

  if (time < 100000000000) {
    return time * 1000;
  }
  return time;
};

function TaskDetailContent() {
  const { state, sendMessage, setTaskId, openFilePreview, closeFilePreview, requestStatus, dispatch, pauseTask, resumeTask } = useApp();
  const { t } = useI18n();
  const [files, setFiles] = useState<File[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const params = useParams();
  const router = useRouter();
  const taskIdFromUrl = params.id;

  // DAG preview toggle and layout
  const [dagPreviewOpen, setDagPreviewOpen] = useState(false);
  const [dagLayout, setDagLayout] = useState<'TB' | 'LR'>('TB');
  const anyPreviewOpen = state.filePreview.isOpen || dagPreviewOpen;

  const [leftWidth, setLeftWidth] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const leftWidthRef = useRef(50);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging || !containerRef.current) return;
    const containerRect = containerRef.current.getBoundingClientRect();
    let newLeftWidth = ((e.clientX - containerRect.left) / containerRect.width) * 100;
    if (newLeftWidth < 20) newLeftWidth = 20;
    if (newLeftWidth > 80) newLeftWidth = 80;
    setLeftWidth(newLeftWidth);
    leftWidthRef.current = newLeftWidth;
  }, [isDragging]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  useEffect(() => {
    if (isDragging) {
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      window.addEventListener('mousemove', handleMouseMove, { passive: true });
      window.addEventListener('mouseup', handleMouseUp);
    } else {
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  useEffect(() => {
    if (taskIdFromUrl && typeof taskIdFromUrl === 'string') {
      const taskIdNum = parseInt(taskIdFromUrl, 10);
      if (!isNaN(taskIdNum) && taskIdNum !== state.taskId) {
        console.log('🔄 Setting taskId from URL:', taskIdNum);
        setTaskId(taskIdNum);
      }
    }
  }, [taskIdFromUrl, setTaskId, state.taskId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [state.messages, state.steps]);

  useEffect(() => {
    if (state.filePreview.isOpen) {
      setDagPreviewOpen(false);
    }
  }, [state.filePreview.isOpen]);

  useEffect(() => {
    const handleFilePreviewEvent = (event: Event) => {
      const { filePath, fileName, allFiles, currentIndex } = (event as CustomEvent<any>).detail || {};
      if (!filePath) return;
      if (Array.isArray(allFiles) && allFiles.length > 0) {
        openFilePreview(filePath, fileName, allFiles, typeof currentIndex === 'number' ? currentIndex : 0);
      } else {
        openFilePreview(filePath, fileName);
      }
    };
    window.addEventListener('openFilePreview', handleFilePreviewEvent as EventListener);
    return () => {
      window.removeEventListener('openFilePreview', handleFilePreviewEvent as EventListener);
    };
  }, [openFilePreview]);

  // Close file preview when leaving the task page
  useEffect(() => {
    return () => {
      closeFilePreview();
    };
  }, [closeFilePreview]);

  const handleDownload = async () => {
    try {
      if (!state.filePreview.fileId) return;

      const response = await apiRequest(`${getApiUrl()}/api/files/download/${state.filePreview.fileId}`);

      if (!response.ok) {
        throw new Error(`Download failed: ${response.statusText}`);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = state.filePreview.fileName || 'download';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to download file:', error);
    }
  };

  const handleSend = async (message: string, config?: any, filesToSend?: File[]) => {
    await sendMessage(message, config, filesToSend || files);
    setFiles([]);
  };

  // Only keep user messages and final assistant messages in timeline
  type CombinedItem = {
    id: string;
    role: "user" | "assistant";
    content: string | React.ReactNode;
    rawContent?: string;
    timestamp: number;
    status?: string;
    isStreamingFinalAnswer?: boolean;
    traceEvents?: any[];
    interactions?: any[];
    showEmptyStatus?: boolean;
    timelineOrder?: number;
  };
  const messageItems: CombinedItem[] = useMemo(() => {
    const items: CombinedItem[] = state.messages
      .filter((m) => m.role === 'user' || m.isResult)
      .map((m) => {
        const id = m.id || `${m.role}-${toTimelineTime(m.timestamp)}`;
        return {
          id,
          role: m.role,
          content: m.content,
          rawContent: m.rawContent,
          timestamp: toTimelineTime(m.timestamp),
          status: m.status,
          isStreamingFinalAnswer: isStreamingFinalAnswerMessage({
            id,
            role: m.role,
            isResult: m.isResult,
          }),
          traceEvents: m.traceEvents,
          interactions: m.interactions,
        };
      });

    items.sort((a, b) => a.timestamp - b.timestamp);
    return items;
  }, [state.messages]);

  const lastMessageItem = messageItems[messageItems.length - 1];
  const hasFinalAssistantMessage =
    !!lastMessageItem &&
    lastMessageItem.role === "assistant" &&
    !(
      lastMessageItem.isStreamingFinalAnswer &&
      lastMessageItem.status === "failed"
    );

  const timelineItems: CombinedItem[] = useMemo(() => {
    const sortedMessages = [...messageItems].sort((a, b) => a.timestamp - b.timestamp);
    const items: CombinedItem[] = sortedMessages.map((item, index) => {
      const timelineOrder = index * 2 + 1;
      return {
        ...item,
        traceEvents: undefined,
        timelineOrder,
      };
    });
    type TimelineProcessEvent = {
      event_id?: string;
      event_type?: string;
      timestamp?: unknown;
      [key: string]: unknown;
    };
    const processEventsById = new Map<string, TimelineProcessEvent>();
    const addProcessEvent = (event: unknown, fallbackKey: string) => {
      if (!event || typeof event !== 'object') {
        return;
      }
      const processEvent = event as TimelineProcessEvent;
      const eventKey =
        typeof processEvent.event_id === 'string' && processEvent.event_id
          ? processEvent.event_id
          : fallbackKey;
      if (!processEventsById.has(eventKey)) {
        processEventsById.set(eventKey, processEvent);
      }
    };

    if (Array.isArray(state.traceEvents)) {
      state.traceEvents.forEach((event, index) => {
        addProcessEvent(event, `state-${index}`);
      });
    }
    messageItems.forEach((item) => {
      item.traceEvents?.forEach((event, index) => {
        addProcessEvent(event, `${item.id}-${index}`);
      });
    });

    const processEvents = Array.from(processEventsById.values());
    if (processEvents.length === 0) {
      return items;
    }

    const processGroups = new Map<number, TimelineProcessEvent[]>();

    processEvents.forEach((event) => {
      const eventTime = toTimelineTime(event?.timestamp);
      const groupIndex = getProcessGroupIndex(sortedMessages, eventTime);

      const group = processGroups.get(groupIndex) || [];
      group.push(event);
      processGroups.set(groupIndex, group);
    });

    const groupEntries = Array.from(processGroups.entries()).sort((a, b) => a[0] - b[0]);
    const latestGroupIndex = groupEntries.length > 0
      ? groupEntries[groupEntries.length - 1][0]
      : -1;

    groupEntries.forEach(([groupIndex, events]) => {
      if (events.length === 0) {
        return;
      }

      const groupTimestamp = Math.min(
        ...events.map((event) => toTimelineTime(event?.timestamp))
      );
      const firstEvent = events[0];
      const shouldShowEmptyStatus =
        !hasFinalAssistantMessage &&
        groupIndex === latestGroupIndex &&
        groupIndex >= sortedMessages.length;

      items.push({
        id: `process-${groupIndex}-${firstEvent?.event_id || groupTimestamp}`,
        role: "assistant",
        content: null,
        timestamp: groupTimestamp,
        status: shouldShowEmptyStatus ? state.currentTask?.status : undefined,
        traceEvents: events,
        showEmptyStatus: shouldShowEmptyStatus,
        timelineOrder: groupIndex * 2,
      });
    });

    items.sort(
      (a, b) =>
        a.timestamp - b.timestamp ||
        (a.timelineOrder ?? Number.MAX_SAFE_INTEGER) -
          (b.timelineOrder ?? Number.MAX_SAFE_INTEGER)
    );
    return items;
  }, [
    hasFinalAssistantMessage,
    messageItems,
    state.currentTask?.status,
    state.isProcessing,
    state.traceEvents,
  ]);

  const waitingPrompt = useMemo(() => {
    if (state.currentTask?.status !== 'waiting_for_user') {
      return null;
    }
    if (state.currentTask.waitingQuestion) {
      return state.currentTask.waitingQuestion;
    }

    for (let i = state.traceEvents.length - 1; i >= 0; i--) {
      const event = state.traceEvents[i] as any;
      if (event.event_type === 'agent_message') {
        const expectsResponse = event.data?.expect_response === true || event.data?.message_type === 'question';
        if (!expectsResponse) {
          continue;
        }
        const message = event.data?.message || event.data?.content;
        if (typeof message === 'string' && message.trim()) {
          return message;
        }
      }
      if (event.event_type === 'react_task_end') {
        const result = event.data?.result;
        if (result?.status === 'waiting_for_user' && typeof result.message === 'string' && result.message.trim()) {
          return result.message;
        }
      }
    }

    return null;
  }, [state.currentTask?.status, state.currentTask?.waitingQuestion, state.traceEvents]);

  const waitingInteractions = useMemo(() => {
    if (state.currentTask?.status !== 'waiting_for_user') {
      return undefined;
    }
    if (state.currentTask.waitingInteractions?.length) {
      return state.currentTask.waitingInteractions;
    }

    for (let i = state.traceEvents.length - 1; i >= 0; i--) {
      const event = state.traceEvents[i] as any;
      if (event.event_type === 'agent_message') {
        const expectsResponse = event.data?.expect_response === true || event.data?.message_type === 'question';
        if (!expectsResponse) {
          continue;
        }
        const interactions = event.data?.metadata?.interactions;
        if (Array.isArray(interactions) && interactions.length > 0) {
          return interactions;
        }
      }
      if (event.event_type === 'react_task_end') {
        const interactions = event.data?.result?.interactions;
        if (Array.isArray(interactions) && interactions.length > 0) {
          return interactions;
        }
      }
    }

    return undefined;
  }, [state.currentTask?.status, state.currentTask?.waitingInteractions, state.traceEvents]);

  const activeWaitingMessageId = useMemo(() => {
    if (state.currentTask?.status !== 'waiting_for_user') {
      return null;
    }

    if (waitingPrompt) {
      const normalizedPrompt = waitingPrompt.trim();
      for (let i = messageItems.length - 1; i >= 0; i--) {
        const item = messageItems[i];
        if (
          item.role === 'assistant' &&
          typeof item.content === 'string' &&
          item.content.trim() === normalizedPrompt
        ) {
          return item.id;
        }
      }
    }

    for (let i = messageItems.length - 1; i >= 0; i--) {
      const item = messageItems[i];
      if (item.role === 'assistant' && item.interactions && item.interactions.length > 0) {
        return item.id;
      }
    }

    return null;
  }, [messageItems, state.currentTask?.status, waitingPrompt]);

  // DAG node and edge calculation
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setGraph({
    rankdir: dagLayout === 'LR' ? 'LR' : 'TB',
    nodesep: 80,
    ranksep: 100,
    marginx: 20,
    marginy: 20,
  });
  dagreGraph.setDefaultEdgeLabel(() => '');

  const validSteps = state.steps.filter(step => step && typeof step.id === 'string' && step.id.trim() !== '');

  // Set nodes
  validSteps.forEach((step, index) => {
    try {
      dagreGraph.setNode(step.id, {
        label: step.name || `Step ${index + 1}`,
        width: 250,
        height: 200,
      });
    } catch (error) {
      console.error('Error adding node to dagre:', step, error);
    }
  });

  // Set edges
  validSteps.forEach((step) => {
    if (!step.dependencies || !Array.isArray(step.dependencies)) {
      return;
    }
    step.dependencies.forEach(depId => {
      if (!depId || typeof depId !== 'string' || depId.trim() === '') {
        return;
      }
      const depStep = validSteps.find(s => s.id === depId);
      if (depStep) {
        try {
          dagreGraph.setEdge(depId, step.id, {});
        } catch (error) {
          console.error('Error adding edge to dagre:', `${depId} -> ${step.id}`, error);
        }
      }
    });
  });

  let dagreLayoutSuccessful = true;
  try {
    dagre.layout(dagreGraph);
  } catch (error) {
    console.error('Dagre layout failed:', error);
    dagreLayoutSuccessful = false;
  }

  const dagNodes = state.steps.map((step, index) => {
    let node: any, safeNode: any;
    if (!step.id || typeof step.id !== 'string' || step.id.trim() === '') {
      safeNode = { x: (index % 3) * 300, y: Math.floor(index / 3) * 250 };
    } else if (dagreLayoutSuccessful) {
      try {
        node = dagreGraph.node(step.id);
        safeNode = typeof node === 'object' && node !== null ? node : { x: (index % 3) * 300, y: Math.floor(index / 3) * 250 };
      } catch (error) {
        safeNode = { x: (index % 3) * 300, y: Math.floor(index / 3) * 250 };
      }
    } else {
      safeNode = { x: (index % 3) * 300, y: Math.floor(index / 3) * 250 };
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
    };
  });

  const dagEdges: any[] = [];
  const validNodeIds = new Set(validSteps.map(s => s.id));
  if (dagreLayoutSuccessful) {
    validSteps.forEach((step) => {
      if (!step.dependencies || !Array.isArray(step.dependencies)) {
        return;
      }
      step.dependencies.forEach(depId => {
        if (!depId || typeof depId !== 'string' || depId.trim() === '') {
          return;
        }
        if (validNodeIds.has(depId) && validNodeIds.has(step.id)) {
          const edge = {
            id: `${depId}-${step.id}`,
            source: depId,
            target: step.id,
            data: {}
          };
          dagEdges.push(edge);
        }
      });
    });
  }

  const isPlanning = dagNodes.length === 0 && state.dagExecution?.phase === "planning";
  const hasError = dagNodes.length === 0 && (state.dagExecution?.phase === "failed" || state.currentTask?.status === "failed");

  return (
    <div
      ref={containerRef}
      className={`h-full bg-background relative transition-all flex ${anyPreviewOpen ? 'flex-row items-stretch' : 'flex-col'} overflow-hidden`}
    >
      {/* Back Button - Only show if this task is from an agent */}
      {state.currentTask?.agentId && (
        <div className="absolute top-4 left-4 z-50">
          <Button
            variant="ghost"
            size="icon"
            className="rounded-full bg-background/50 hover:bg-background/80 backdrop-blur border shadow-sm"
            onClick={() => {
              const agentId = state.currentTask?.agentId;
              if (agentId) {
                router.push(`/agent/${agentId}`);
              } else {
                router.push("/task");
              }
            }}
            title={t("common.back")}
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
        </div>
      )}

      {/* Left Panel */}
      <div
        style={{ width: anyPreviewOpen ? `${leftWidth}%` : '100%' }}
        className={`${anyPreviewOpen ? '' : 'flex-1'} min-w-0 flex flex-col min-h-0 transition-[width] duration-0 relative`}
      >
        {/* Messages scroll area */}
        <div className="flex-1 overflow-y-auto">
          <main className={`container max-w-4xl mx-auto px-4 py-8 relative z-0 transition-all`}>
            <div className="space-y-6 pb-4">
              {state.isHistoryLoading && timelineItems.length === 0 ? (
                <div className="flex flex-col items-center justify-center min-h-[60vh] py-16 text-center">
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
                    const isFailedFinalAnswerStream =
                      item.isStreamingFinalAnswer && item.status === "failed";
                    return (
                      <ChatMessage
                        key={item.id}
                        role={item.role}
                        content={item.content}
                        rawContent={item.rawContent}
                        traceEvents={item.traceEvents as any || []}
                        showProcessView={true}
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
                    );
                  })}

                  {(state.isProcessing || state.currentTask?.status === 'paused' || state.currentTask?.status === 'waiting_for_user' || state.currentTask?.status === 'failed') && !hasFinalAssistantMessage && !timelineItems.some((item) => item.showEmptyStatus) && (
                    <ChatMessage
                      role="assistant"
                      content={
                        state.currentTask?.status === 'waiting_for_user'
                          ? waitingPrompt
                          : null
                      }
                      traceEvents={state.traceEvents as any || []}
                      showProcessView={true}
                      isVirtual
                      taskStatus={state.currentTask?.status}
                      interactions={
                        state.currentTask?.status === 'waiting_for_user'
                          ? waitingInteractions
                          : undefined
                      }
                      interactionsActive={state.currentTask?.status === 'waiting_for_user'}
                    />
                  )}
                </>
              )}
              <div ref={messagesEndRef} />
            </div>
          </main>
        </div>

        {/* Fixed input box at bottom */}
        <div className="flex-shrink-0 z-10 glass pb-6">
          <div className="container max-w-4xl mx-auto px-4">
            <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-3">
              <div className="flex flex-wrap items-center gap-2">
                {state.currentTask?.isDag !== false && (
                  <div
                    className="inline-flex items-center gap-1 rounded-xl border bg-card/80 backdrop-blur px-3 py-2 cursor-pointer hover:bg-muted/30 transition-colors text-sm shrink-0"
                    onClick={() => {
                      closeFilePreview();
                      setDagPreviewOpen(true);
                    }}
                    title={t("chatPage.executionPlan.title")}
                  >
                    <GitMerge className="w-3.5 h-3.5" />
                    {t("chatPage.executionPlan.title")}
                  </div>
                )}

                <TaskFileManager
                  taskId={state.taskId}
                  onPreview={(fileId, fileName) => openFilePreview(fileId, fileName)}
                >
                  <div
                    className="inline-flex items-center gap-1 rounded-xl border bg-card/80 backdrop-blur px-3 py-2 cursor-pointer hover:bg-muted/30 transition-colors text-sm shrink-0"
                    title={t("files.header.title")}
                  >
                    <FolderOpen className="w-3.5 h-3.5" />
                    {t("files.header.title")}
                  </div>
                </TaskFileManager>
              </div>

              <div className="sm:ml-auto">
                <TokenUsageDisplay
                  taskId={state.taskId}
                  isRunning={state.currentTask?.status === 'running'}
                />
              </div>
            </div>

            <ChatInput
              onSend={handleSend}
              isLoading={state.isProcessing}
              files={files}
              onFilesChange={setFiles}
              showModeToggle={false}
              taskStatus={state.currentTask?.status}
              onPause={pauseTask}
              onResume={resumeTask}
              taskConfig={state.currentTask ? {
                model: state.currentTask.modelId || state.currentTask.modelName,
                smallFastModel: state.currentTask.smallFastModelId,
                visualModel: state.currentTask.visualModelId,
                compactModel: state.currentTask.compactModelId,
                executionMode: state.currentTask.executionMode
              } : undefined}
              readOnlyConfig={true}
              autoFocus={true}
            />
          </div>
        </div>
      </div>

      {/* Divider */}
      {anyPreviewOpen && (
        <div
          onMouseDown={handleMouseDown}
          className={`relative w-1 cursor-col-resize group z-[100] flex-shrink-0 hover:bg-primary/20 active:bg-primary/40 transition-colors ${isDragging ? 'bg-primary/40' : 'bg-transparent'}`}
        >
          <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-[1px] bg-border group-hover:bg-primary group-hover:w-[2px] transition-all" />
          <div className="absolute inset-y-0 -left-2 -right-2" />
        </div>
      )}

      {/* Right Panel */}
      {anyPreviewOpen && (
        <div
          style={{ width: `${100 - leftWidth}%`, pointerEvents: isDragging ? 'none' : 'auto' }}
          className="flex-shrink-0 px-2 py-6 overflow-hidden relative"
        >
          <PreviewSheet
            open={state.filePreview.isOpen || dagPreviewOpen}
            onOpenChange={(open) => {
              if (!open) {
                closeFilePreview();
                setDagPreviewOpen(false);
              }
            }}
            title={
              state.filePreview.isOpen ? <>{state.filePreview.fileName}</> :
                t("chatPage.executionPlan.title")
            }
            actions={state.filePreview.isOpen ? (
              <FilePreviewActionButtons
                viewMode={state.filePreview.viewMode}
                onViewModeChange={(mode) => dispatch({ type: 'SET_FILE_PREVIEW_MODE', payload: mode })}
                fileName={state.filePreview.fileName || ''}
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

      {/* Drag overlay */}
      {isDragging && <div className="fixed inset-0 z-[99] cursor-col-resize" />}
    </div>
  );
}

export default function TaskDetailPage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center h-full"><Loader2 className="w-8 h-8 animate-spin" /></div>}>
      <TaskDetailContent />
    </Suspense>
  );
}

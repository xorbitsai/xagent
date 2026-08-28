import React, { useState, useRef, useEffect, useCallback } from "react";
import { ChevronDown, ChevronUp, Copy, Check, Laptop } from "lucide-react";
import { useRouter } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  TraceEventRenderer,
  getFriendlyToolName,
  getRawToolName,
  isAgentProgressEvent,
  getProgressNarrationText,
  type AgentExecutionSummary,
} from "./TraceEventRenderer";
import { useI18n } from "@/contexts/i18n-context";
import { useApp } from "@/contexts/app-context-chat";
import { MarkdownRenderer } from "@/components/ui/markdown-renderer";
import {
  sanitizeFilesDisabledPresentationText,
  serializeFilesDisabledPresentation,
} from "@/lib/files-disabled-presentation";
import { Button } from "@/components/ui/button";
import { normalizeTimestampMs } from "@/lib/time-utils";
import { FileChip } from "./FileChip";
import { ClarificationForm } from "./clarification-form";
import { isStoppedTraceProcessStatus, resolveTraceProcessStatus } from "@/lib/trace-process-status";
import {
  TaskRuntimeMessageMetadataExtension,
  type TaskRuntimeMessageMetadataExtensionProps,
} from "@/lib/task-runtime-ui-extension";

const MARKDOWN_FILE_REF_RE = /\[([^\]]+)\]\(file:(?:\/\/)?([^)]+)\)/g;
const BACKTICK_FILE_REF_RE = /`([^`]+)`/g;

interface ToolArgs {
  code?: string;
  file_path?: string;
  content?: string;
  [key: string]: unknown;
}

interface ToolResult {
  success?: boolean;
  output?: string;
  error?: string;
  message?: string;
}

interface TraceEvent {
  event_id?: string;
  event_type?: string;
  action_type?: string;
  step_id?: string;
  timestamp?: number;
  data?: {
    action?: string;
    step_name?: string;
    description?: string;
    tool_names?: string[];
    model_name?: string;
    tool_name?: string;
    tool_args?: ToolArgs;
    response?: {
      reasoning?: string;
      tool_name?: string;
      tool_args?: ToolArgs;
      answer?: string;
    };
    result?: ToolResult | string;
    tools?: Array<{
      function: {
        name: string;
        arguments?: string;
      };
    }>;
    success?: boolean;
    [key: string]: unknown;
  };
  tool_name?: string;
  result_type?: string;
}

export interface ChatMessageProps {
  role: "user" | "assistant" | "system";
  content: React.ReactNode;
  rawContent?: string;
  traceEvents?: TraceEvent[];
  showProcessView?: boolean;
  isVirtual?: boolean;
  taskStatus?: string;
  processStatus?: string;
  timestamp?: number | string;
  interactions?: any[];
  interactionRequestId?: string;
  interactionsActive?: boolean;
  showEmptyStatus?: boolean;
  onOpenExecutionPlan?: () => void;
  onAgentExecutionClick?: (execution: AgentExecutionSummary) => void;
  onSendInteraction?: (message: string, files?: File[], metadata?: any) => Promise<void> | void;
  contextBadges?: Array<{
    kind: "computer_use";
    label: string;
    detail: string;
  }>;
  taskRuntimeExtensionMetadata?: TaskRuntimeMessageMetadataExtensionProps;
}

function GeneratingIndicator({ latestTitle, taskStatus }: { latestTitle?: string, taskStatus?: string }) {
  const { t } = useI18n();

  const displayTitle = taskStatus === 'paused'
    ? t("common.taskPaused")
    : taskStatus === 'waiting_for_user'
      ? t("common.waitingForUser")
    : (latestTitle ? `${latestTitle} ` : t("common.planning"));

  return (
    <div className="py-3 text-sm leading-relaxed text-muted-foreground flex items-center">
      <span>{displayTitle}</span>
      {!["paused", "waiting_for_user", "completed"].includes(taskStatus || "") && (
        <span className="ml-1 inline-flex items-end gap-1">
          <span className="dot" />
          <span className="dot" />
          <span className="dot" />
        </span>
      )}
      {/* Wave animation style */}
      <style jsx>{`
        .dot {
          width: 4px;
          height: 4px;
          border-radius: 9999px;
          background-color: currentColor;
          display: inline-block;
          animation: dotWave 1s ease-in-out infinite;
          opacity: 0.6;
        }
        .dot:nth-child(2) {
          animation-delay: 0.15s;
        }
        .dot:nth-child(3) {
          animation-delay: 0.3s;
        }
        @keyframes dotWave {
          0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.5;
          }
          30% {
            transform: translateY(-4px);
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
}

function ExpandableMessage({
  content,
  filesDisabled,
}: {
  content: string;
  filesDisabled: boolean;
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isOverflowing, setIsOverflowing] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);
  const { t } = useI18n();
  const { openFilePreview } = useApp();

  const updateOverflowState = useCallback(() => {
    const el = contentRef.current;
    if (!el) return;
    setIsOverflowing(el.scrollHeight > el.clientHeight + 1);
  }, []);

  useEffect(() => {
    const el = contentRef.current;
    if (!el) return;

    const frameId = window.requestAnimationFrame(updateOverflowState);
    const observer = new ResizeObserver(() => updateOverflowState());
    observer.observe(el);

    return () => {
      window.cancelAnimationFrame(frameId);
      observer.disconnect();
    };
  }, [content, isExpanded, updateOverflowState]);

  if (!content) return null;

  if (filesDisabled) {
    const inertContent = sanitizeFilesDisabledPresentationText(content);
    return (
      <div className="relative max-w-full min-w-0">
        <div className="max-w-full min-w-0 text-sm leading-relaxed whitespace-pre-wrap break-words [overflow-wrap:anywhere] py-[2px]">
          {inertContent}
        </div>
      </div>
    );
  }

  const markdownRegex = new RegExp(MARKDOWN_FILE_REF_RE);
  const backtickRegex = new RegExp(BACKTICK_FILE_REF_RE);

  const segments: React.ReactNode[] = [];
  let lastIndex = 0;

  const processText = (text: string, startIndex: number) => {
    let textLastIndex = 0;
    let match;
    const regex = new RegExp(backtickRegex);

    while ((match = regex.exec(text)) !== null) {
      if (match.index > textLastIndex) {
        segments.push(text.substring(textLastIndex, match.index));
      }

      const path = match[1];
      const fileName = path.split('/').pop() || path;
      segments.push(
        <FileChip
          className="bg-[#F3F4F6]"
          key={`bt-${startIndex + match.index}`}
          path={path}
          onClick={() => openFilePreview?.(path, fileName, [{ fileName, fileId: path }])}
        />
      );

      textLastIndex = regex.lastIndex;
    }

    if (textLastIndex < text.length) {
      segments.push(text.substring(textLastIndex));
    }
  };

  let match;
  while ((match = markdownRegex.exec(content)) !== null) {
    if (match.index > lastIndex) {
      processText(content.substring(lastIndex, match.index), lastIndex);
    }

    const [_, filename, id] = match;

    segments.push(
      <FileChip
        className="bg-[#F3F4F6]"
        key={`md-${match.index}`}
        path={id}
        filename={filename}
        onClick={() => openFilePreview?.(id, filename, [{ fileName: filename, fileId: id }])}
      />
    );

    lastIndex = markdownRegex.lastIndex;
  }

  if (lastIndex < content.length) {
    processText(content.substring(lastIndex), lastIndex);
  }

  return (
    <div className="relative max-w-full min-w-0">
      <div
        ref={contentRef}
        className={cn(
          "max-w-full min-w-0 text-sm leading-relaxed whitespace-pre-wrap break-words [overflow-wrap:anywhere] transition-all duration-300 py-[2px]",
          !isExpanded && "max-h-[240px] overflow-hidden"
        )}
      >
        {segments}
      </div>
      {isOverflowing && !isExpanded && (
        <>
          <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-secondary to-transparent pointer-events-none" />
          <div className="absolute bottom-1 left-1/2 -translate-x-1/2">
            <Button
              variant="outline"
              size="sm"
              className="h-7 px-3 rounded-full shadow-sm bg-background hover:bg-accent text-xs text-foreground border"
              onClick={() => setIsExpanded(true)}
            >
              <ChevronDown className="w-3.5 h-3.5 mr-1" />
              {t("common.expand")}
            </Button>
          </div>
        </>
      )}
      {isOverflowing && isExpanded && (
        <div className="mt-3 flex justify-center">
          <Button
            variant="outline"
            size="sm"
            className="h-7 px-3 rounded-full shadow-sm bg-background hover:bg-accent text-xs text-foreground border"
            onClick={() => setIsExpanded(false)}
          >
            <ChevronUp className="w-3.5 h-3.5 mr-1" />
            {t("common.collapse")}
          </Button>
        </div>
      )}
    </div>
  );
}

export function ChatMessage({
  role,
  content,
  rawContent,
  traceEvents,
  // Default matches TaskConversationPanelProps: an unwired caller gets the
  // internal-page behavior; public surfaces opt out explicitly.
  showProcessView = true,
  taskStatus,
  processStatus,
  timestamp,
  interactions,
  interactionRequestId,
  interactionsActive = true,
  showEmptyStatus = true,
  onOpenExecutionPlan,
  onAgentExecutionClick,
  onSendInteraction,
  contextBadges,
  taskRuntimeExtensionMetadata,
}: ChatMessageProps) {
  const { t, tDynamic } = useI18n();
  const { filesDisabled, openFilePreview } = useApp();
  const router = useRouter();
  const isUser = role === "user";
  const [copied, setCopied] = useState(false);

  const handleAgentClick = (agentId: string, agentName: string) => {
    router.push(`/agent/${agentId}`);
  };

  const handleFileClick = (filePath: string, fileName: string) => {
    if (filesDisabled) return;
    openFilePreview?.(filePath, fileName, [{ fileName, fileId: filePath }]);
  };

  const formattedTime = timestamp
    ? new Date(normalizeTimestampMs(timestamp)).toLocaleString([], {
      month: 'numeric',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })
    : "";

  const hasTraceEvents = Array.isArray(traceEvents) && traceEvents.length > 0;
  const shouldShowProcess = !!showProcessView && hasTraceEvents;

  // traceEvents comes straight off an external WS/API payload, so validate
  // that each entry is actually an object here, once, rather than repeating
  // a null/type guard in every helper below that reads from the list.
  const sanitizedTraceEvents: TraceEvent[] = Array.isArray(traceEvents)
    ? traceEvents
        .filter(
          (event): event is TraceEvent =>
            !!event && typeof event === "object" && !Array.isArray(event),
        )
        // A non-string event_type (e.g. a stray number) would otherwise
        // render as a garbage status label further down — drop it here
        // rather than have every reader guard against it individually.
        .map((event) =>
          typeof event.event_type === "string"
            ? event
            : { ...event, event_type: undefined },
        )
    : [];

  // The model's own narration ("I'll look into pricing now...") is more
  // useful here than a generic action label. It keeps showing through one
  // later, un-narrated tool call — the narration is still accurate for
  // that long — but expires once a second one starts: without a cutoff a
  // narration from early in a long turn would keep naming a step the
  // trace has long since moved past.
  const latestProgressNarration = (): string => {
    let toolStartsSinceNarration = 0;
    for (let i = sanitizedTraceEvents.length - 1; i >= 0; i--) {
      const event = sanitizedTraceEvents[i];
      if (isAgentProgressEvent(event)) {
        const message = getProgressNarrationText(event);
        if (!message) continue;
        return toolStartsSinceNarration <= 1 ? message : "";
      }
      if (event.event_type === "tool_execution_start") {
        toolStartsSinceNarration += 1;
      }
    }
    return "";
  };

  // Map event/action to i18n key
  const getEventTitle = (e: TraceEvent | undefined) => {
    if (!e) return "";
    const type = e.event_type || "";
    // Tool events carry a specific tool name — prefer "Searching the web" over
    // the generic "Working on it" fallback whenever we know which tool it is.
    // Only for the start event: once the tool has finished, naming it again
    // here would read as still in progress, so tool_execution_end falls
    // through to the generic "Done" label below instead.
    if (type === "tool_execution_start") {
      const rawToolName = getRawToolName(e);
      if (rawToolName) {
        return getFriendlyToolName(rawToolName, tDynamic);
      }
    }
    const action = (typeof e.data?.action === "string" ? (e.data!.action as string) : "") || type;
    if (type) {
      const key = `agent.logs.event.actions.${type}`;
      return tDynamic(key, action || type);
    }
    return action || t("traceEventRenderer.taskExecution");
  };

  const resolvedProcessStatus = resolveTraceProcessStatus({
    processStatus,
    taskStatus,
    traceEvents,
  });

  // Once the turn has actually finished, the terminal event's own title
  // ("Done") is what belongs here — not a mid-run narration that's still
  // sitting in the trace. Without this gate, a completed turn with no
  // answer text (an empty final response) could keep showing "Searching
  // the web" indefinitely, since narration has no other way to know the
  // turn is over.
  const latestTitle =
    (resolvedProcessStatus === "completed" ? "" : latestProgressNarration()) ||
    getEventTitle(sanitizedTraceEvents.at(-1));

  // A turn that stopped without an answer — failed, paused, or waiting for
  // user input — must leave a visible mark once the trace is hidden: dropping
  // its bubble too would show the visitor's question followed by nothing.
  const isStoppedWithoutAnswer =
    isStoppedTraceProcessStatus(resolvedProcessStatus) &&
    resolvedProcessStatus !== "completed";

  // A trace-only turn carries no answer text and no status line of its own, so
  // its bubble would render as a bare avatar. Drop it whether the trace above
  // was rendered (internal pages) or suppressed (embedded chat) — keying this
  // off the events themselves rather than off shouldShowProcess. Stopped
  // unanswered turns are the exception once the trace is hidden (see above).
  const isProcessOnlyMessage =
    hasTraceEvents &&
    !isUser &&
    !content &&
    showEmptyStatus === false &&
    (showProcessView || !isStoppedWithoutAnswer);

  // The trace carries the backend's raw error string (a Python exception, more
  // often than not). With the trace hidden the failure line must not become its
  // replacement channel, so only mine the events when the process view is on.
  let errorMessage = "";
  if (showProcessView && resolvedProcessStatus === "failed") {
    for (let i = sanitizedTraceEvents.length - 1; i >= 0; i--) {
      const event = sanitizedTraceEvents[i];
      if (['trace_error', 'task_failed', 'react_task_failed', 'dag_step_failed', 'agent_error'].includes(event.event_type || '')) {
        errorMessage = (event.data?.error as string) || (event.data?.message as string) || (event.data?.error_message as string) || "";
        if (errorMessage) break;
      }
    }
  }
  const failureText =
    errorMessage
    || (showProcessView ? t("common.errors.unknown") : t("common.errors.taskFailed"));
  // A failed turn's content is failure text by construction (final_answer_error
  // streams str(exc); the terminal handler stores the reason verbatim), so with
  // the trace hidden it needs the same generic replacement as mined errors.
  const failedMessageText =
    showProcessView && typeof content === "string" && content.trim()
      ? content
      : failureText;

  // The copy button must not hand out what the bubble refuses to show: on a
  // failed turn, copy exactly the (possibly redacted) text that is displayed.
  const isAssistantFailure = !isUser && resolvedProcessStatus === "failed";
  const copyableContent = isAssistantFailure
    ? failedMessageText
    : typeof content === "string" ? content : rawContent;
  const displayCopyableContent = filesDisabled && copyableContent
    ? serializeFilesDisabledPresentation(copyableContent)
    : copyableContent;

  const handleCopy = () => {
    if (displayCopyableContent) {
      navigator.clipboard.writeText(displayCopyableContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  // latestTitle names the running step ("calling web_search"), which is part of
  // the trace: with the trace hidden it would leak the very detail the status
  // line replaces.
  const statusTitle = showProcessView
    ? latestTitle
    : resolvedProcessStatus === "completed"
      ? t("common.statusDone")
      : t("common.thinking");

  // Neither the trace nor the bubble is going to render, so there is nothing
  // left to wrap. Bail out rather than emit an empty div: the timeline
  // separates children with space-y-*, so a childless wrapper still takes its
  // gap, and a stray rawContent would leave the copy button floating alone.
  if (isProcessOnlyMessage && !shouldShowProcess) {
    return null;
  }

  return (
    <div className="w-full space-y-2 animate-fade-in group">
      {shouldShowProcess && !isUser && (
        <TraceEventRenderer
          events={traceEvents}
          taskStatus={resolvedProcessStatus}
          onOpenExecutionPlan={onOpenExecutionPlan}
          onAgentExecutionClick={onAgentExecutionClick}
        />
      )}

      {!isProcessOnlyMessage && (
        <div
          className={cn(
            "flex w-full",
            isUser ? "justify-end" : "justify-start"
          )}
        >
          <div
            className={cn(
              isUser
                ? "max-w-[85%] bg-secondary text-secondary-foreground p-3 rounded-2xl"
                : "bg-transparent p-0 w-full max-w-full"
            )}
          >
            {/* Message content */}
            <div>
              {isAssistantFailure ? (
                <div className="py-3 text-sm leading-relaxed text-red-500 break-words [overflow-wrap:anywhere]">
                  {displayCopyableContent}
                </div>
              ) : content ? (
                typeof content === "string" ? (
                  isUser ? (
                    <ExpandableMessage
                      content={content}
                      filesDisabled={filesDisabled}
                    />
                  ) : (
                    <MarkdownRenderer
                      content={content}
                      className="prose-sm pt-2 leading-relaxed break-words [overflow-wrap:anywhere]"
                      filesDisabled={filesDisabled}
                      onAgentClick={handleAgentClick}
                      onFileClick={filesDisabled ? undefined : handleFileClick}
                    />
                  )
                ) : (
                  <div className="text-sm leading-relaxed break-words [overflow-wrap:anywhere]">{content}</div>
                )
              ) : (
                // A past paused/waiting turn has showEmptyStatus=false, but with
                // the trace hidden its status line is all that marks the turn.
                !isUser && (showEmptyStatus || (!showProcessView && isStoppedWithoutAnswer)) && (
                  <GeneratingIndicator latestTitle={statusTitle} taskStatus={resolvedProcessStatus} />
                )
              )}
              {isUser && contextBadges && contextBadges.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {contextBadges.map((badge) => (
                    <div
                      key={`${badge.kind}:${badge.label}:${badge.detail}`}
                      role="note"
                      className="inline-flex h-7 min-w-0 items-center gap-1.5 rounded-lg border border-border/80 bg-background/55 px-2 text-xs text-muted-foreground"
                      aria-label={`${badge.label} · ${badge.detail}`}
                    >
                      <Laptop className="h-3.5 w-3.5 shrink-0" />
                      <span className="truncate">{badge.label}</span>
                      <span aria-hidden="true">·</span>
                      <span className="truncate">{badge.detail}</span>
                    </div>
                  ))}
                </div>
              )}
              {isUser && taskRuntimeExtensionMetadata && (
                <TaskRuntimeMessageMetadataExtension
                  {...taskRuntimeExtensionMetadata}
                />
              )}
              {!isUser && interactions && interactions.length > 0 && (
                <div className="mt-4 border-t pt-4">
                  <ClarificationForm
                    interactions={interactions}
                    requestId={interactionRequestId}
                    active={interactionsActive}
                    filesDisabled={filesDisabled}
                    onSend={onSendInteraction}
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Action Row */}
      {copyableContent && (
        <div
          className={cn(
            "flex items-center gap-1.5 text-xs text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity duration-200",
            isUser ? "justify-end mr-1" : "justify-start"
          )}
        >
          <button
            onClick={handleCopy}
            className="hover:text-foreground flex items-center justify-center p-1 rounded-md hover:bg-muted/50 transition-colors"
            title={t("common.copy") || "Copy"}
          >
            {copied ? (
              <Check className="w-3.5 h-3.5 text-green-500" />
            ) : (
              <Copy className="w-3.5 h-3.5" />
            )}
          </button>
          {formattedTime && <span>{formattedTime}</span>}
        </div>
      )}
    </div>
  );
}

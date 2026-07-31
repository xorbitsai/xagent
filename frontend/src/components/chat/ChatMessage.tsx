import React, { useState, useRef, useEffect, useCallback } from "react";
import { Bot, ChevronDown, ChevronUp, Copy, Check } from "lucide-react";
import { useRouter } from "next/navigation";
import { cn } from "@/lib/utils";
import { TraceEventRenderer, type AgentExecutionSummary } from "./TraceEventRenderer";
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
  interactionsActive?: boolean;
  showEmptyStatus?: boolean;
  onOpenExecutionPlan?: () => void;
  onAgentExecutionClick?: (execution: AgentExecutionSummary) => void;
  onSendInteraction?: (message: string, files?: File[], metadata?: any) => Promise<void> | void;
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
  interactionsActive = true,
  showEmptyStatus = true,
  onOpenExecutionPlan,
  onAgentExecutionClick,
  onSendInteraction,
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

  // Map event/action to i18n key
  const getEventTitle = (e: TraceEvent | undefined) => {
    if (!e) return "";
    const type = e.event_type || "";
    const action = (typeof e.data?.action === "string" ? (e.data!.action as string) : "") || type;
    if (type) {
      const key = `agent.logs.event.actions.${type}`;
      return tDynamic(key, action || type);
    }
    return action || t("traceEventRenderer.taskExecution");
  };

  const latestTitle = getEventTitle(
    Array.isArray(traceEvents) && traceEvents.length > 0
      ? traceEvents[traceEvents.length - 1]
      : undefined
  );
  const resolvedProcessStatus = resolveTraceProcessStatus({
    processStatus,
    taskStatus,
    traceEvents,
  });

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
  if (showProcessView && resolvedProcessStatus === "failed" && Array.isArray(traceEvents)) {
    for (let i = traceEvents.length - 1; i >= 0; i--) {
      const event = traceEvents[i];
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
        <div className={cn("pl-7")}>
          <TraceEventRenderer
            events={traceEvents}
            taskStatus={resolvedProcessStatus}
            onOpenExecutionPlan={onOpenExecutionPlan}
            onAgentExecutionClick={onAgentExecutionClick}
          />
        </div>
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
              "flex gap-4 transition-all duration-300",
              isUser
                ? "max-w-[85%] bg-secondary text-secondary-foreground p-3 rounded-2xl flex-row-reverse items-center"
                : "bg-transparent p-0 w-full max-w-full"
            )}
          >
            {/* Avatar */}
            {!isUser && (
              <div
                className={cn(
                  "flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center shadow-md bg-transparent"
                )}
              >
                <Bot className="w-5 h-5 text-muted-foreground" />
              </div>
            )}

            {/* Message content */}
            <div className={cn("flex-1 min-w-0")}>
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
              {!isUser && interactions && interactions.length > 0 && (
                <div className="mt-4 border-t pt-4">
                  <ClarificationForm
                    interactions={interactions}
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
            isUser ? "justify-end mr-1" : "justify-start ml-14"
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

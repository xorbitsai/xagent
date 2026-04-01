import React, { useState, useRef, useEffect } from "react";
import { Bot, ChevronDown, ChevronUp, Copy, Check } from "lucide-react";
import { useRouter } from "next/navigation";
import { cn } from "@/lib/utils";
import { TraceEventRenderer } from "./TraceEventRenderer";
import { useI18n } from "@/contexts/i18n-context";
import { useApp } from "@/contexts/app-context-chat";
import { MarkdownRenderer } from "@/components/ui/markdown-renderer";
import { Button } from "@/components/ui/button";
import { normalizeTimestampMs } from "@/lib/time-utils";
import { FileChip } from "./FileChip";

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
  timestamp?: number | string;
}

function GeneratingIndicator({latestTitle, taskStatus, errorMessage}: {latestTitle?: string, taskStatus?: string, errorMessage?: string}) {
  const { t } = useI18n();

  if (taskStatus === 'failed' && errorMessage) {
    return (
      <div className="py-3 text-sm leading-relaxed text-red-500">
        <span>{errorMessage}</span>
      </div>
    );
  }

  const displayTitle = taskStatus === 'paused'
    ? t("common.taskPaused")
    : (latestTitle ? `${latestTitle} ` : t("common.planning"));

  return (
    <div className="py-3 text-sm leading-relaxed text-muted-foreground flex items-center">
      <span>{displayTitle}</span>
      {!["failed", "paused"].includes(taskStatus || "") && (
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

function ExpandableMessage({ content }: { content: string }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isOverflowing, setIsOverflowing] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);
  const { t } = useI18n();
  const { openFilePreview } = useApp();

  useEffect(() => {
    if (contentRef.current) {
      setIsOverflowing(contentRef.current.scrollHeight > 240);
    }
  }, [content]);

  if (!content) return null;

  const markdownRegex = /\[([^\]]+)\]\(file:\/\/([^)]+)\)/g;
  const backtickRegex = /`([^`]+)`/g;

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
    <div className="relative">
      <div
        ref={contentRef}
        className={cn(
          "text-sm leading-relaxed whitespace-pre-wrap break-words transition-all duration-300 py-[2px]",
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
  showProcessView,
  taskStatus,
  timestamp,
}: ChatMessageProps) {
  const { t } = useI18n();
  const { openFilePreview } = useApp();
  const router = useRouter();
  const isUser = role === "user";
  const [copied, setCopied] = useState(false);

  const copyableContent = typeof content === "string" ? content : rawContent;

  const handleCopy = () => {
    if (copyableContent) {
      navigator.clipboard.writeText(copyableContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleAgentClick = (agentId: string, agentName: string) => {
    router.push(`/agent/${agentId}`);
  };

  const handleFileClick = (filePath: string, fileName: string) => {
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

  const shouldShowProcess =
    !!showProcessView &&
    Array.isArray(traceEvents) &&
    traceEvents.length > 0;

  // Map event/action to i18n key
  const getEventTitle = (e: TraceEvent | undefined) => {
    if (!e) return "";
    const type = e.event_type || "";
    const action = (typeof e.data?.action === "string" ? (e.data!.action as string) : "") || type;
    return t(`agent.logs.event.actions.${e.event_type}`) || action || t("common.executing");
  };

  const latestTitle = getEventTitle(
    Array.isArray(traceEvents) && traceEvents.length > 0
      ? traceEvents[traceEvents.length - 1]
      : undefined
  );

  let errorMessage = "";
  if (taskStatus === "failed" && Array.isArray(traceEvents)) {
    for (let i = traceEvents.length - 1; i >= 0; i--) {
      const event = traceEvents[i];
      if (['trace_error', 'task_failed', 'react_task_failed', 'dag_step_failed', 'agent_error'].includes(event.event_type || '')) {
         errorMessage = (event.data?.error as string) || (event.data?.message as string) || "";
         if (errorMessage) break;
      }
    }
  }

  return (
    <div className="w-full space-y-2 animate-fade-in group">
      {shouldShowProcess && !isUser && (
        <div className={cn("pl-7")}>
          <TraceEventRenderer events={traceEvents} />
        </div>
      )}

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
              ? "bg-secondary text-secondary-foreground p-3 rounded-2xl flex-row-reverse items-center"
              : "bg-transparent p-0 w-full"
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
            {content ? (
              typeof content === "string" ? (
                isUser ? (
                  <ExpandableMessage content={content} />
                ) : (
                  <MarkdownRenderer
                    content={content}
                    className="prose-sm pt-2 leading-relaxed"
                    onAgentClick={handleAgentClick}
                    onFileClick={handleFileClick}
                  />
                )
              ) : (
                <div className="text-sm leading-relaxed">{content}</div>
              )
            ) : (
              !isUser && <GeneratingIndicator latestTitle={latestTitle} taskStatus={taskStatus} errorMessage={errorMessage} />
            )}
          </div>
        </div>
      </div>

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

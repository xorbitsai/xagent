import { Bot } from "lucide-react";
import { cn } from "@/lib/utils";
import { TraceEventRenderer } from "./TraceEventRenderer";
import { useI18n } from "@/contexts/i18n-context";
import type React from "react";

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
  traceEvents?: TraceEvent[];
  showProcessView?: boolean;
  isVirtual?: boolean;
  taskStatus?: string;
  isPlanning?: boolean;
}

function GeneratingIndicator({latestTitle, taskStatus, isPlanning}: {latestTitle?: string, taskStatus?: string, isPlanning?: boolean}) {
  const { t } = useI18n();
  const displayTitle = taskStatus === 'paused'
    ? t("common.taskPaused")
    : (isPlanning ? t("common.planning") : (latestTitle ? `${latestTitle} ` : t("common.planning")));

  return (
    <div className="py-3 text-sm leading-relaxed text-muted-foreground flex items-center">
      <span>{displayTitle}</span>
      {taskStatus !== 'paused' && (
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

export function ChatMessage({
  role,
  content,
  traceEvents,
  showProcessView,
  taskStatus,
  isPlanning,
}: ChatMessageProps) {
  const isUser = role === "user";
  const { t } = useI18n();

  const shouldShowProcess =
    !!showProcessView &&
    Array.isArray(traceEvents) &&
    traceEvents.length > 0;

  // Map event/action to i18n key
  const getEventTitle = (e: TraceEvent | undefined) => {
    if (!e) return "";
    const type = e.event_type || "";
    const action = (typeof e.data?.action === "string" ? (e.data!.action as string) : "") || type;
    const map: Record<string, string> = {
      "dag_step_start": "agent.logs.event.actions.stepStart",
      "dag_step_end": "agent.logs.event.actions.stepCompleted",
      "dag_step_failed": "agent.logs.event.actions.stepFailed",
      "llm_call_start": "agent.logs.event.actions.llmStart",
      "llm_call_end": "agent.logs.event.actions.llmCompleted",
      "llm_call_failed": "agent.logs.event.actions.llmFailed",
      "tool_execution_start": "agent.logs.event.actions.toolStart",
      "tool_execution_end": "agent.logs.event.actions.toolCompleted",
      "tool_execution_failed": "agent.logs.event.actions.toolFailed",
      "task_start_memory_retrieve": "agent.logs.event.actions.memoryQuery",
      "task_end_memory_retrieve": "agent.logs.event.actions.memoryQueryCompleted",
      "task_start_memory_generate": "agent.logs.event.actions.memoryGenerateStart",
      "task_end_memory_generate": "agent.logs.event.actions.memoryGenerateCompleted",
      "task_start_memory_store": "agent.logs.event.actions.memoryStoreStart",
      "task_end_memory_store": "agent.logs.event.actions.memoryStoreCompleted",
      "action_start_compact": "agent.logs.event.actions.compactStart",
      "action_end_compact": "agent.logs.event.actions.compactCompleted",
    };
    const key = map[action] || map[type];
    return key ? t(key) : (action || t("common.executing"));
  };

  const latestTitle = getEventTitle(
    Array.isArray(traceEvents) && traceEvents.length > 0
      ? traceEvents[traceEvents.length - 1]
      : undefined
  );

  return (
    <div className="w-full space-y-2 animate-fade-in">
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
              ? "bg-slate-100 text-slate-700 p-3 rounded-2xl flex-row-reverse items-center"
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
                <p className={cn(
                  "text-sm leading-relaxed whitespace-pre-wrap",
                  isUser && "max-h-60 overflow-y-auto",
                  !isUser && "pt-2"
                )}>
                  {content}
                </p>
              ) : (
                <div className="text-sm leading-relaxed">{content}</div>
              )
            ) : (
              !isUser && <GeneratingIndicator latestTitle={latestTitle} taskStatus={taskStatus} isPlanning={isPlanning} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

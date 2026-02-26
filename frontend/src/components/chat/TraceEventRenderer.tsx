import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  CheckCircle2,
  Loader2,
  ChevronRight,
  ChevronDown,
  Wrench,
  Terminal,
  Cpu,
  Info,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useApp } from '@/contexts/app-context-chat';
import { useI18n } from '@/contexts/i18n-context';
import { MarkdownRenderer } from "@/components/ui/markdown-renderer";
import { ScrollArea } from "@/components/ui/scroll-area";

// Types
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
    selected?: boolean;
    skill_name?: string;
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

interface StepAction {
  id: string;
  type: 'llm' | 'tool' | 'info' | 'error';
  title: string;
  status: 'running' | 'completed' | 'failed';
  timestamp: number;
  data: {
    model?: string;
    tool?: string;
    args?: any;
    code?: string;
    output?: any;
    reasoning?: string;
    error?: any;
    tool_calls?: any;
  };
}

interface ProcessedStep {
  stepId: string;
  stepName: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  tools: Array<{ function: { name: string } }>;
  reasoning?: string;
  code: string;
  output: string;
  filePath?: string;
  actions: StepAction[];
}

interface TraceEventRendererProps {
  events: TraceEvent[];
}

// Process trace events into steps
function useProcessedSteps(events: TraceEvent[]): ProcessedStep[] {
  const { t } = useI18n();
  return useMemo(() => {
    const stepsMap = new Map<string, ProcessedStep>();
    let currentReactStepId: string | null = null;

    // Helper to find the last running action of a specific type
    const findLastRunningAction = (step: ProcessedStep, type: 'llm' | 'tool') => {
      for (let i = step.actions.length - 1; i >= 0; i--) {
        if (step.actions[i].type === type && step.actions[i].status === 'running') {
          return step.actions[i];
        }
      }
      return null;
    };

    events.forEach((event, index) => {
      if (event.event_type?.startsWith('skill_select')) {
        return;
      }

      let stepId = event.step_id || (event.data?.step_id as string) || 'default';

      if (event.event_type === 'react_task_start' || event.event_type === 'task_start_react') {
        currentReactStepId = stepId;
      }

      if ((event.event_type === 'react_task_end' || event.event_type === 'task_end_react' || event.event_type === 'task_completion' || event.event_type === 'react_task_failed' || event.event_type === 'task_failed_react') && stepId === 'default' && currentReactStepId) {
        stepId = currentReactStepId;
      }

      if (!stepsMap.has(stepId)) {
        stepsMap.set(stepId, {
          stepId,
          stepName: '',
          description: '',
          status: 'pending',
          tools: [],
          reasoning: '',
          code: '',
          output: '',
          filePath: '',
          actions: [],
        });
      }

      const step = stepsMap.get(stepId)!;
      const timestamp = event.timestamp || Date.now();
      const eventId = event.event_id || `event-${index}`;

      // Process different event types
      if (event.event_type === 'dag_step_start' || event.event_type === 'react_task_start') {
        step.stepName = (event.data?.step_name as string) || (event.event_type === 'react_task_start' ? t('traceEventRenderer.taskExecution') : '');
        step.description = (event.data?.description as string) || (event.data?.task as string) || '';
        step.status = 'running';

        const tools = event.data?.tool_names || event.data?.tools;
        if (tools && Array.isArray(tools)) {
          step.tools = tools.map((toolItem: any) => {
            if (typeof toolItem === 'string') return { function: { name: toolItem } };
            if (toolItem?.function?.name) return toolItem;
            return { function: { name: 'unknown' } };
          });
        }
      }

      if (event.event_type === 'llm_call_start') {
        step.actions.push({
          id: eventId,
          type: 'llm',
          title: t('traceEventRenderer.callLLM', { model: event.data?.model_name || t('traceEventRenderer.unknownModel') }),
          status: 'running',
          timestamp,
          data: { model: event.data?.model_name }
        });
      }

      if (event.event_type === 'llm_call_end' || event.event_type === 'llm_call_result') {
        if (event.data?.response?.reasoning) {
          step.reasoning = event.data.response.reasoning;
        }
        if (event.data?.tools) {
          step.tools = event.data.tools;
        }

        const action = findLastRunningAction(step, 'llm');
        if (action) {
          action.status = 'completed';
          action.data.reasoning = event.data?.response?.reasoning;
          action.data.tool_calls = event.data?.tools;
        } else {
           // Fallback if no start event found
           step.actions.push({
            id: eventId,
            type: 'llm',
            title: t('traceEventRenderer.llmResponse'),
            status: 'completed',
            timestamp,
            data: {
              reasoning: event.data?.response?.reasoning,
              tool_calls: event.data?.tools
            }
          });
        }
      }

      if (event.event_type === 'tool_execution_start') {
        // Support both data.response.tool_args.code and data.tool_args.code
        const toolArgs = event.data?.response?.tool_args || event.data?.tool_args;
        if (toolArgs?.code) {
          step.code = toolArgs.code as string;
        }
        // Support file operations as well (file_path, content, etc.)
        if (toolArgs?.file_path && toolArgs?.content) {
          step.code = toolArgs.content as string;
        }
        // Capture file path if provided
        if (toolArgs?.file_path) {
          step.filePath = String(toolArgs.file_path);
        }
        // Support both data.response.tool_name and data.tool_name
        const toolName = event.data?.response?.tool_name || event.data?.tool_name || t('traceEventRenderer.unknownTool');

        if (toolName) {
          // Merge with existing tools instead of replacing
          if (!step.tools.some(tItem => tItem.function.name === toolName)) {
            step.tools.push({ function: { name: toolName } });
          }
        }

        step.actions.push({
          id: eventId,
          type: 'tool',
          title: t('traceEventRenderer.executeTool', { tool: toolName }),
          status: 'running',
          timestamp,
          data: {
            tool: toolName,
            args: toolArgs,
            code: step.code
          }
        });
      }

      if (event.event_type === 'tool_execution_end') {
        const result = event.data?.result;
        let output = '';
        if (result) {
          if (typeof result === 'string') {
            output = result;
          } else if (typeof result === 'object') {
            if (result.output) {
              output = result.output;
            } else if (result.message) {
              output = result.message as string;
            }
          }
        }
        step.output = output;

        const action = findLastRunningAction(step, 'tool');
        if (action) {
          action.status = 'completed';
          action.data.output = output;
        } else {
          // Fallback
           step.actions.push({
            id: eventId,
            type: 'tool',
            title: t('traceEventRenderer.toolExecutionFinished'),
            status: 'completed',
            timestamp,
            data: { output }
          });
        }
      }

      if (event.event_type === 'dag_step_end' || event.event_type === 'step_completed' || event.event_type === 'react_task_end' || event.event_type === 'task_completion') {
        step.status = 'completed';
        // Ensure all actions are completed
        step.actions.forEach(a => {
            if (a.status === 'running') a.status = 'completed';
        });
      }

      if (['dag_step_failed', 'tool_execution_failed', 'llm_call_failed', 'react_task_failed', 'agent_error', 'trace_error'].includes(event.event_type as string)) {
         step.status = 'failed';

         // Extract error message with more fallback options
         const errorData = event.data || {};
         let errorMessage =
            errorData.error ||
            errorData.message;

         if (!errorMessage && errorData.result) {
            errorMessage = (errorData.result as any).error || (errorData.result as any).message;
         }

         if (!errorMessage && typeof errorData === 'string') {
             errorMessage = errorData;
         }

         if (!errorMessage) {
             errorMessage = t('traceEventRenderer.unknownError');
         }

         // Try to find specific action type based on event type
         let runningAction = step.actions.find(a => a.status === 'running');

         // If no running action found, or type mismatch, try to find the last action of corresponding type
         if (event.event_type === 'tool_execution_failed') {
             const lastTool = findLastRunningAction(step, 'tool');
             if (lastTool) runningAction = lastTool;
         } else if (event.event_type === 'llm_call_failed') {
             const lastLlm = findLastRunningAction(step, 'llm');
             if (lastLlm) runningAction = lastLlm;
         }

         if (runningAction) {
             runningAction.status = 'failed';
             runningAction.data.error = errorMessage;
         } else {
            step.actions.push({
               id: eventId,
               type: 'error',
               title: t('traceEventRenderer.executionFailed'),
               status: 'failed',
               timestamp,
               data: { error: errorMessage }
            });
         }
      }
    });

    return Array.from(stepsMap.values()).filter(step => step.stepName);
  }, [events, t]);
}

// Step Action Item Component
interface StepActionItemProps {
  action: StepAction;
  onViewDetail: (action: StepAction) => void;
  onOpenTerminal: (code: string, output: string, toolName: string, filePath?: string) => void;
  onFileClick?: (filePath: string, fileName: string) => void;
}

function StepActionItem({ action, onViewDetail, onOpenTerminal, onFileClick }: StepActionItemProps) {
  const { t } = useI18n();
  const scrollRef = useRef<HTMLDivElement>(null);
  const [isExpanded, setIsExpanded] = useState(false);
  const [userToggled, setUserToggled] = useState(false);

  // Auto-expand/collapse logic
  useEffect(() => {
    if (userToggled) return;

    if (action.status === 'running') {
      setIsExpanded(true);
    } else if (action.status === 'completed' || action.status === 'failed') {
      setIsExpanded(false);
    }
  }, [action.status, userToggled]);

  // Auto-scroll logic
  useEffect(() => {
    if (action.status === 'running' && isExpanded && scrollRef.current) {
      const scrollElement = scrollRef.current.querySelector('[data-radix-scroll-area-viewport]') ||
                            scrollRef.current.querySelector('[data-slot="scroll-area-viewport"]');
      if (scrollElement) {
        scrollElement.scrollTop = scrollElement.scrollHeight;
      }
    }
  }, [action.data, action.status, isExpanded]); // Re-run when data updates

  const handleToggle = () => {
    setIsExpanded(!isExpanded);
    setUserToggled(true);
  };

  const isRunning = action.status === 'running';
  const isFailed = action.status === 'failed';
  const isCompleted = action.status === 'completed';

  const summary = useMemo(() => {
    if (action.type === 'llm') {
      if (action.data.reasoning) {
        const clean = action.data.reasoning.replace(/[\n\r\s]+/g, ' ').trim();
        return clean.length > 70 ? clean.slice(0, 70) + '...' : clean;
      }
      return null;
    }
    if (action.type === 'tool') {
      const { args, code } = action.data;
      // Prefer showing file path for file operations
      if (args && typeof args === 'object') {
        if ('file_path' in args) return String(args.file_path);
        if ('query' in args) return String(args.query); // For search tools
        if ('path' in args) return String(args.path);
      }

      // Fallback to code snippet
      if (code) {
        const clean = code.replace(/[\n\r\s]+/g, ' ').trim();
        return clean.length > 70 ? clean.slice(0, 70) + '...' : clean;
      }

      // Fallback to generic args string
      if (args) {
         try {
           const str = JSON.stringify(args);
           return str.length > 70 ? str.slice(0, 70) + '...' : str;
         } catch (e) { return null; }
      }
    }
    return null;
  }, [action.type, action.data]);

  if (action.type === 'llm') {
    return (
        <div className="group transition-all duration-300">
          {action.data.reasoning && (
            <MarkdownRenderer
              content={action.data.reasoning}
              onFileClick={onFileClick}
              className="
                text-sm text-muted-foreground leading-relaxed pl-3
                prose-neutral dark:prose-invert max-w-none
                [&>p]:mb-2 [&>p:last-child]:mb-0
              "
            />
          )}
          {action.status === 'failed' && action.data.error && (
            <div className="text-red-400 text-sm pl-3 mt-1 whitespace-pre-wrap">
              {t('traceEventRenderer.errorLabel')}{String(action.data.error)}
            </div>
          )}
        </div>
    );
  }

  return (
    <div className="group transition-all duration-300">
      <button
        onClick={handleToggle}
        className={cn(
            "w-full flex items-center justify-between py-3 px-3 text-xs transition-colors rounded-md",
            isRunning ? "bg-primary/5 text-primary" : "hover:bg-muted/50 text-muted-foreground/80 hover:text-foreground"
        )}
      >
        <span className="flex items-center gap-2 overflow-hidden">
          <span className="flex-shrink-0 flex items-center">
            {action.type === 'tool' && <Wrench className="w-3.5 h-3.5" />}
            {action.type === 'error' && <Info className="w-3.5 h-3.5 text-red-500" />}
            {action.type === 'info' && <Info className="w-3.5 h-3.5" />}
          </span>

          <span className="font-medium whitespace-nowrap">{action.title}</span>

          {summary && (
              <span className="text-muted-foreground/50 font-normal truncate ml-1 hidden sm:block max-w-[600px]">
                - {summary}
              </span>
          )}

          {isRunning && <Loader2 className="w-3 h-3 animate-spin ml-1 flex-shrink-0" />}
        </span>
        <div className="flex items-center gap-2 flex-shrink-0">
            <span className="text-[10px] opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground/50">
                {new Date(action.timestamp).toLocaleTimeString()}
            </span>
            {isExpanded ? <ChevronDown className="w-3 h-3 opacity-50" /> : <ChevronRight className="w-3 h-3 opacity-50" />}
        </div>
      </button>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
             <ScrollArea ref={scrollRef} className="max-h-[200px] w-full mt-1 bg-muted/20 rounded-md overflow-auto">
               <div
                 className="p-3 space-y-2 font-mono text-xs cursor-pointer hover:bg-white/5 transition-colors"
                 onClick={() => onViewDetail(action)}
               >
                 {action.type === 'tool' && (
                     <div className="space-y-2">
                         {/* View Code & Output Button for Tool Actions */}
                         {(action.data.code || action.data.output) && (
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    onOpenTerminal(
                                        action.data.code || '',
                                        typeof action.data.output === 'string' ? action.data.output : JSON.stringify(action.data.output || ''),
                                        action.data.tool || 'tool',
                                        action.data.args?.file_path
                                    );
                                }}
                                className="group flex items-center gap-2 px-3 py-1.5 bg-[#1a1b26] hover:bg-[#1f2030] border border-[#2a2b3d] rounded-md transition-all duration-200 shadow-sm hover:shadow-md mb-2 w-fit"
                            >
                                <Terminal className="w-3.5 h-3.5 text-[#7aa2f7]" />
                                <span className="text-xs text-[#a9b1d6] font-mono">
                                  {t('traceEventRenderer.view')} {action.data.args?.file_path || t('traceEventRenderer.codeAndOutput')}
                                </span>
                                <ChevronRight className="w-3 h-3 text-[#565f89] group-hover:translate-x-0.5 transition-transform" />
                            </button>
                         )}
                         {action.data.args && (
                             <div className="text-muted-foreground">
                                 {t('traceEventRenderer.args')}: {JSON.stringify(action.data.args)}
                             </div>
                         )}
                         {action.data.output && (
                             <div className="pt-2 border-t border-border/30 text-muted-foreground/80 whitespace-pre-wrap">
                                 {t('traceEventRenderer.output')}: {typeof action.data.output === 'string' ? action.data.output : JSON.stringify(action.data.output, null, 2)}
                             </div>
                         )}
                         {!action.data.output && isRunning && (
                             <div className="text-muted-foreground italic">{t('traceEventRenderer.executing')}</div>
                         )}
                         {action.status === 'failed' && action.data.error && (
                             <div className="text-red-400 whitespace-pre-wrap pt-2 border-t border-red-500/30">
                                 {t('traceEventRenderer.errorLabel')} {String(action.data.error)}
                             </div>
                         )}
                     </div>
                 )}

                 {action.type === 'error' && (
                     <div className="text-red-400 whitespace-pre-wrap">
                         {String(action.data.error)}
                     </div>
                 )}
               </div>
             </ScrollArea>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Step Item Component
interface StepItemProps {
  step: ProcessedStep;
  index: number;
  onOpenTerminal: (code: string, output: string, toolName: string, filePath?: string) => void;
  onViewDetail: (action: StepAction) => void;
  onFileClick?: (filePath: string, fileName: string) => void;
}

function StepItem({ step, index, onOpenTerminal, onViewDetail, onFileClick }: StepItemProps) {
  const isCompleted = step.status === 'completed';
  const isFailed = step.status === 'failed';
  const [isExpanded, setIsExpanded] = useState(true); // Default to expanded

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 * (index + 1) }}
      className="space-y-3"
    >
      {/* Step Title */}
      <div
        className="flex items-start gap-2 cursor-pointer group/step"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className={cn(
          "flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center mt-0.5 transition-colors",
          isCompleted ? "bg-green-500/20" :
          isFailed ? "bg-red-500/20" :
          "bg-primary/20"
        )}>
          {isCompleted ? (
            <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />
          ) : isFailed ? (
            <Info className="w-3.5 h-3.5 text-red-500" />
          ) : (
            <Loader2 className="w-3.5 h-3.5 text-primary animate-spin" />
          )}
        </div>

        <div className="flex-1 min-w-0 flex items-center gap-2">
          <h3 className="text-sm font-medium text-foreground">
            {step.description || step.stepName}
          </h3>
          <div className="opacity-0 group-hover/step:opacity-100 transition-opacity">
            {isExpanded ? (
                <ChevronDown className="w-4 h-4 text-muted-foreground/50" />
            ) : (
                <ChevronRight className="w-4 h-4 text-muted-foreground/50" />
            )}
          </div>
        </div>
      </div>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
              {/* Actions List (replaces nested Execution Details) */}
              <div className="ml-2.5 pl-4 border-l-2 border-border/40 space-y-2">
                {step.actions.map((action) => (
                    <StepActionItem
                        key={action.id}
                        action={action}
                        onViewDetail={onViewDetail}
                        onOpenTerminal={onOpenTerminal}
                        onFileClick={onFileClick}
                    />
                ))}
              </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// Main TraceEventRenderer Component
export function TraceEventRenderer({ events }: TraceEventRendererProps) {
  const { t } = useI18n();
  const steps = useProcessedSteps(events);

  const { openFilePreview, dispatch } = useApp();

  const skillSelection = useMemo(() => {
    for (let i = events.length - 1; i >= 0; i--) {
      const event = events[i];
      if (event.event_type === 'skill_select_end') {
        if (event.data?.selected && event.data?.skill_name) {
          return event.data.skill_name as string;
        }
        return null;
      }
    }
    return null;
  }, [events]);

  const getFileNameFromPath = (path?: string) => {
    if (!path) return '';
    const parts = path.split('/');
    return parts[parts.length - 1] || path;
  };

  const handleOpenTerminal = useCallback((code: string, output: string, toolName: string, filePath?: string) => {
    if (filePath && filePath.trim()) {
      const fileName = getFileNameFromPath(filePath) || `${toolName || 'terminal'}-execution.txt`;
      openFilePreview(filePath, fileName);
      return;
    }

    const fileName = `${toolName || 'terminal'}-execution.txt`;
    openFilePreview('', fileName);
    const contentSections: string[] = [];
    if (code && code.trim()) {
      contentSections.push(`${t('traceEventRenderer.executionCode')}\n\n${code.trim()}`);
    }
    if (output && String(output).trim()) {
      contentSections.push(`\n\n${t('traceEventRenderer.outputResult')}\n\n${String(output).trim()}`);
    }
    dispatch({ type: "SET_FILE_PREVIEW_CONTENT", payload: { content: contentSections.join('\n'), error: null } });
  }, [openFilePreview, dispatch, t]);

  const handleViewActionDetail = useCallback((action: StepAction) => {
    const title = `${action.title.replace(/\s+/g, '_')}.json`;
    openFilePreview('', title);

    let content = '';
    // Better formatting for specific types
    if (action.type === 'tool') {
        content = `${t('traceEventRenderer.toolLabel')}${action.data.tool}\n\n${t('traceEventRenderer.argumentsLabel')}\n${JSON.stringify(action.data.args, null, 2)}`;
        if (action.data.code) {
          content += `\n\n${t('traceEventRenderer.codeLabel')}\n${action.data.code}`;
        }
        if (action.data.output) {
          content += `\n\n${t('traceEventRenderer.outputLabel')}\n${typeof action.data.output === 'string' ? action.data.output : JSON.stringify(action.data.output, null, 2)}`;
        }
    } else if (action.type === 'llm') {
        content = `${t('traceEventRenderer.modelLabel')}${action.data.model}\n\n${t('traceEventRenderer.reasoningLabel')}\n${action.data.reasoning || t('traceEventRenderer.noReasoning')}`;
        if (action.data.tool_calls) {
           content += `\n\n${t('traceEventRenderer.toolCallsLabel')}\n${JSON.stringify(action.data.tool_calls, null, 2)}`;
        }
    } else if (action.data.error) {
        content = `${t('traceEventRenderer.errorTitle')}\n${String(action.data.error)}`;
    } else {
        content = JSON.stringify(action.data, null, 2);
    }

    dispatch({ type: "SET_FILE_PREVIEW_CONTENT", payload: { content, error: null } });
  }, [openFilePreview, dispatch, t]);

  if (steps.length === 0 && !skillSelection) {
    return null;
  }

  return (
    <div className="space-y-4">
      {skillSelection && (
        <div className="bg-muted/30 border border-border/50 rounded-lg p-3 flex items-center gap-2">
          <Cpu className="w-4 h-4 text-primary" />
          <span className="text-sm">
            {t('traceEventRenderer.skillSelected')}: <span className="font-medium">{skillSelection}</span>
          </span>
        </div>
      )}
      <div className="flex gap-3">
        <div className="flex-1 space-y-4 overflow-hidden">
          {steps.map((step, index) => (
            <StepItem
              key={step.stepId}
              step={step}
              index={index}
              onOpenTerminal={handleOpenTerminal}
              onViewDetail={handleViewActionDetail}
              onFileClick={openFilePreview}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

export default TraceEventRenderer;

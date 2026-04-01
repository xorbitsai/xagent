import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import {
  CheckCircle2,
  Loader2,
  ChevronRight,
  ChevronDown,
  Wrench,
  Cpu,
  Info,
  Copy,
  Search,
  FileText,
  Check,
  Shield,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useApp } from '@/contexts/app-context-chat';
import { useI18n } from '@/contexts/i18n-context';
import { MarkdownRenderer } from "@/components/ui/markdown-renderer";
import { ScrollArea } from "@/components/ui/scroll-area";
import { normalizeTimestampMs } from '@/lib/time-utils';

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
    sandboxed?: boolean;
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
      const timestamp = normalizeTimestampMs(event.timestamp);
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
            code: step.code,
            sandboxed: !!event.data?.sandboxed
          }
        });
      }

      if (event.event_type === 'tool_execution_end') {
        const result = event.data?.result;
        let output: any = '';
        if (result !== undefined) {
          if (typeof result === 'string') {
            output = result;
          } else if (typeof result === 'object' && result !== null) {
            if ('output' in result) {
              output = result.output;
            } else if ('message' in result) {
              output = result.message;
            } else {
              output = result; // fallback to the entire result object
            }
          } else {
            output = String(result);
          }
        } else if (event.data?.output !== undefined) {
          output = event.data.output;
        } else if (event.data?.response !== undefined) {
          output = event.data.response;
        } else if (event.data !== undefined) {
          // If no specific result/output field is found, maybe data itself has it or we can dump data
          // But only dump if we are sure there's some outcome. We'll leave it empty if we can't find anything,
          // except if there's an 'error' field handled elsewhere.
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
            data: { output, sandboxed: !!event.data?.sandboxed }
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


// --- Specialized Tool Renderers ---

const ActionButton = ({ icon: Icon, onClick, title, className }: any) => (
  <button
    onClick={(e) => { e.stopPropagation(); onClick(e); }}
    className={cn("p-1 text-muted-foreground hover:text-foreground hover:bg-muted rounded transition-colors", className)}
    title={title}
  >
    <Icon className="w-3.5 h-3.5" />
  </button>
);

const CopyButton = ({ text, title }: { text: string, title?: string }) => {
  const { t } = useI18n();
  const [copied, setCopied] = useState(false);
  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <button
      onClick={handleCopy}
      className="p-1 text-muted-foreground hover:text-foreground hover:bg-muted rounded transition-colors"
      title={title || t('traceEventRenderer.copy')}
    >
      {copied ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Copy className="w-3.5 h-3.5" />}
    </button>
  );
};

const ToolOutputDisplay = ({ action, isRunning, t, onFileClick, onAgentClick }: { action: StepAction, isRunning: boolean, t: any, onFileClick?: (filePath: string, fileName: string) => void, onAgentClick?: (agentId: string, agentName: string) => void }) => (
  <>
    {action.data.output !== undefined && action.data.output !== '' && (
      <div className="mt-4 flex flex-col gap-1.5">
        <div className="text-xs text-muted-foreground px-1 flex justify-between items-center">
          <span>{t('traceEventRenderer.output')}</span>
          <CopyButton text={typeof action.data.output === 'string' ? action.data.output : JSON.stringify(action.data.output, null, 2)} />
        </div>
        <div className="p-3 bg-muted/30 border border-border/50 rounded-xl text-[10px] sm:text-xs overflow-x-auto">
          {typeof action.data.output === 'string' ? (
            <MarkdownRenderer
              content={action.data.output}
              onFileClick={onFileClick}
              onAgentClick={onAgentClick}
              className="prose-sm max-w-none"
            />
          ) : (
            <pre className="text-foreground/80 whitespace-pre-wrap break-all font-mono">
              {JSON.stringify(action.data.output, null, 2)}
            </pre>
          )}
        </div>
      </div>
    )}
    {(action.data.output === undefined || action.data.output === '') && isRunning && (
      <div className="mt-4 p-3 bg-muted/30 border border-border/50 rounded-xl text-muted-foreground italic flex items-center gap-2 text-xs">
        <Loader2 className="w-4 h-4 animate-spin" />
        {t('traceEventRenderer.executing')}
      </div>
    )}

  </>
);

const ToolErrorDisplay = ({ action, t }: { action: StepAction, t: any }) => {
  if (action.status === 'failed' && action.data.error) {
    return (
      <div className="mb-2 mt-2 p-3 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 whitespace-pre-wrap break-all text-xs">
        <span className="font-semibold">{t('traceEventRenderer.errorLabel')}</span> {String(action.data.error)}
      </div>
    );
  }
  return null;
};

const PythonToolRenderer = ({ action, onOpenTerminal, isRunning, t, onFileClick, onAgentClick }: any) => {
  const code = action.data.code;
  const filePath = action.data.args?.file_path;
  return (
    <div className="pt-2">
      {code !== undefined && (
        <div className="flex flex-col gap-1.5">
          {filePath && (
            <div className="mb-1 flex">
              <span className="inline-flex px-2 py-1 bg-blue-500/10 text-blue-600 dark:text-blue-400 rounded-md font-mono text-[11px] items-center gap-1.5 border border-blue-500/20">
                <FileText className="w-3.5 h-3.5" />
                {filePath}
              </span>
            </div>
          )}
          <div className="text-xs text-muted-foreground px-1 flex justify-between items-center">
            <div className="flex items-center gap-2">
              <span>{t('traceEventRenderer.code')}</span>
            </div>
            <CopyButton text={code} />
          </div>
          <div className="p-3 bg-muted/30 border border-border/50 rounded-xl font-mono text-[10px] sm:text-xs overflow-x-auto relative group">
            <span className="absolute right-3 top-3 text-[10px] font-bold text-muted-foreground/50 select-none">PYTHON</span>
            <pre className="text-foreground/80 whitespace-pre-wrap break-all">{code}</pre>
          </div>
        </div>
      )}
      <ToolOutputDisplay action={action} isRunning={isRunning} t={t} onFileClick={onFileClick} onAgentClick={onAgentClick} />
    </div>
  );
};

const BashToolRenderer = ({ action, onOpenTerminal, isRunning, t, onFileClick, onAgentClick }: any) => {
  const command = action.data.args?.command || JSON.stringify(action.data.args);
  return (
    <div className="pt-2">
      {command !== undefined && (
        <div className="flex flex-col gap-1.5">
          <div className="text-xs text-muted-foreground px-1 flex justify-between items-center">
            <span>{t('traceEventRenderer.command')}</span>
            <CopyButton text={command} />
          </div>
          <div className="p-3 bg-muted/30 border border-border/50 rounded-xl font-mono text-[10px] sm:text-xs overflow-x-auto text-foreground/80 whitespace-pre-wrap break-all">
            <span className="text-green-500/70 mr-2 select-none">$</span>
            {command}
          </div>
        </div>
      )}
      <ToolOutputDisplay action={action} isRunning={isRunning} t={t} onFileClick={onFileClick} onAgentClick={onAgentClick} />
    </div>
  );
};

const SearchToolRenderer = ({ action, isRunning, t, onFileClick, onAgentClick }: any) => {
  const query = action.data.args?.query || JSON.stringify(action.data.args);
  return (
    <div className="pt-2">
      <div className="flex flex-col gap-1.5">
        <div className="text-xs text-muted-foreground px-1 flex justify-between items-center">
          <span>{t('traceEventRenderer.searchQuery')}</span>
          <CopyButton text={query} />
        </div>
        <div className="p-3 bg-muted/30 border border-border/50 rounded-xl text-xs flex items-start gap-2">
          <Search className="w-4 h-4 text-muted-foreground mt-0.5 shrink-0" />
          <span className="italic text-foreground/80 whitespace-pre-wrap break-all">{query}</span>
        </div>
      </div>
      <ToolOutputDisplay action={action} isRunning={isRunning} t={t} onFileClick={onFileClick} onAgentClick={onAgentClick} />
    </div>
  );
};

const FileToolRenderer = ({ action, onOpenTerminal, isRunning, t, onFileClick, onAgentClick }: any) => {
  const { args, tool } = action.data;
  const filePath = args?.file_path || args?.path;
  const content = args?.content || args?.text || args?.code;
  const fallbackText = !content ? JSON.stringify(args, null, 2) : undefined;

  return (
    <div className="pt-2">
      <div className="flex flex-col gap-1.5">
        {filePath && (
          <div className="mb-1 flex">
            <span
              className="inline-flex px-2 py-1 bg-blue-500/10 text-blue-600 dark:text-blue-400 rounded-md font-mono text-[11px] items-center gap-1.5 border border-blue-500/20 cursor-pointer hover:bg-blue-500/20 transition-colors"
              onClick={(e) => {
                e.stopPropagation();
                onOpenTerminal(String(content || fallbackText || ''), typeof action.data.output === 'string' ? action.data.output : JSON.stringify(action.data.output ?? ''), tool || 'file_tool', filePath);
              }}
              title={t('traceEventRenderer.previewFile')}
            >
              <FileText className="w-3.5 h-3.5" />
              {filePath}
            </span>
          </div>
        )}
        <div className="text-xs text-muted-foreground px-1 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <span>{content ? (t('traceEventRenderer.content')) : (t('traceEventRenderer.args'))}</span>
          </div>
          <div className="flex items-center gap-1">
            {(content || fallbackText) && <CopyButton text={String(content || fallbackText)} />}
          </div>
        </div>
        <div className="p-3 bg-muted/30 border border-border/50 rounded-xl font-mono text-[10px] sm:text-xs overflow-x-auto text-foreground/80 whitespace-pre-wrap break-all">
          {content ? (
            <pre className="whitespace-pre-wrap break-all">{String(content)}</pre>
          ) : (
            <span className="whitespace-pre-wrap break-all">{fallbackText}</span>
          )}
        </div>
      </div>
      <ToolOutputDisplay action={action} isRunning={isRunning} t={t} onFileClick={onFileClick} onAgentClick={onAgentClick} />
    </div>
  );
};

const DefaultToolRenderer = ({ action, isRunning, t, onFileClick, onAgentClick }: any) => {
  const args = JSON.stringify(action.data.args, null, 2);
  return (
    <div className="pt-2">
      <div className="flex flex-col gap-1.5">
        <div className="text-xs text-muted-foreground px-1 flex justify-between items-center">
          <span>{t('traceEventRenderer.args')}</span>
          <CopyButton text={args} />
        </div>
        <div className="p-3 bg-muted/30 border border-border/50 rounded-xl font-mono text-[10px] sm:text-xs overflow-x-auto text-foreground/80 whitespace-pre-wrap break-all">
          <pre className="whitespace-pre-wrap break-all">{args}</pre>
        </div>
      </div>
      <ToolOutputDisplay action={action} isRunning={isRunning} t={t} onFileClick={onFileClick} onAgentClick={onAgentClick} />
    </div>
  );
};

const ToolDetailsRenderer = ({ action, onOpenTerminal, isRunning, t, onFileClick, onAgentClick }: any) => {
  const toolName = action.data.tool;
  let rendererContent = null;
  if (toolName === 'python_executor') {
    rendererContent = <PythonToolRenderer action={action} onOpenTerminal={onOpenTerminal} isRunning={isRunning} t={t} onFileClick={onFileClick} onAgentClick={onAgentClick} />;
  } else if (toolName === 'bash') {
    rendererContent = <BashToolRenderer action={action} onOpenTerminal={onOpenTerminal} isRunning={isRunning} t={t} onFileClick={onFileClick} onAgentClick={onAgentClick} />;
  } else if (toolName === 'web_search' || toolName === 'tavily_web_search') {
    rendererContent = <SearchToolRenderer action={action} isRunning={isRunning} t={t} onFileClick={onFileClick} onAgentClick={onAgentClick} />;
  } else if (toolName && (toolName.includes('file') || toolName === 'list_directory')) {
    rendererContent = <FileToolRenderer action={action} onOpenTerminal={onOpenTerminal} isRunning={isRunning} t={t} onFileClick={onFileClick} onAgentClick={onAgentClick} />;
  } else {
    rendererContent = <DefaultToolRenderer action={action} isRunning={isRunning} t={t} onFileClick={onFileClick} onAgentClick={onAgentClick} />;
  }

  return (
    <div className="flex flex-col">
      <ToolErrorDisplay action={action} t={t} />
      {rendererContent}
    </div>
  );
};

// --- End Specialized Tool Renderers ---

// Step Action Item Component
interface StepActionItemProps {
  action: StepAction;
  onViewDetail: (action: StepAction) => void;
  onOpenTerminal: (code: string, output: string, toolName: string, filePath?: string) => void;
  onFileClick?: (filePath: string, fileName: string) => void;
  onAgentClick?: (agentId: string, agentName: string) => void;
}

function StepActionItem({ action, onViewDetail, onOpenTerminal, onFileClick, onAgentClick }: StepActionItemProps) {
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
      const { tool, args, code } = action.data;

      if (tool === 'python_executor' && code) {
        return `Python: ${code.slice(0, 50).replace(/[\n\r\s]+/g, ' ').trim()}...`;
      }
      if (tool === 'bash' && args?.command) {
        return `${t('traceEventRenderer.bashPrefix')} ${String(args.command).slice(0, 50)}...`;
      }
      if ((tool === 'web_search' || tool === 'tavily_web_search') && args?.query) {
        return `${t('traceEventRenderer.searchPrefix')} ${args.query}`;
      }

      // Prefer showing file path for file operations
      if (args && typeof args === 'object') {
        if ('file_path' in args) return `${t('traceEventRenderer.filePrefix')} ${String(args.file_path)}`;
        if ('query' in args) return `${t('traceEventRenderer.queryPrefix')} ${String(args.query)}`;
        if ('path' in args) return `${t('traceEventRenderer.pathPrefix')} ${String(args.path)}`;
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
                text-sm text-muted-foreground leading-relaxed
                prose-neutral dark:prose-invert max-w-none
                [&>p]:mb-2 [&>p:last-child]:mb-0
              "
          />
        )}
        {action.status === 'failed' && action.data.error && (
          <div className="text-red-400 text-sm mt-1 whitespace-pre-wrap">
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
          "w-full flex items-center justify-between py-3 px-3 text-xs transition-colors rounded-md border",
          isRunning ? "bg-primary/10 border-primary/20 text-primary" :
            isExpanded ? "bg-muted/50 border-border text-foreground" :
              "bg-muted/50 border-transparent hover:bg-muted/60 text-muted-foreground/80 hover:text-foreground"
        )}
      >
        <span className="flex items-center gap-2 overflow-hidden">
          <span className="flex-shrink-0 flex items-center">
            {action.type === 'tool' && <Wrench className="w-3.5 h-3.5" />}
            {action.type === 'error' && <Info className="w-3.5 h-3.5 text-red-500" />}
            {action.type === 'info' && <Info className="w-3.5 h-3.5" />}
          </span>

          <span className="font-medium whitespace-nowrap">{action.title}</span>

          {action.data.sandboxed && (
            <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-green-500/10 text-green-600 dark:text-green-400 border border-green-500/20 whitespace-nowrap">
              <Shield className="w-3 h-3" />
              {t('traceEventRenderer.sandboxedExecution')}
            </span>
          )}

          {summary && (
            <span className="text-muted-foreground/50 font-normal truncate ml-1 hidden sm:block max-w-[600px]">
              - {summary}
            </span>
          )}

          {isRunning && <Loader2 className="w-3 h-3 animate-spin ml-1 flex-shrink-0" />}
        </span>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-[10px] opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground/50">
            {new Date(action.timestamp).toLocaleString([], {
              month: 'numeric',
              day: 'numeric',
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit'
            })}
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
            className="overflow-hidden"
          >
            <ScrollArea ref={scrollRef} className="max-h-[300px] w-full mt-1 bg-muted/30 border border-border/50 rounded-md overflow-auto">
              <div
                className="p-3 space-y-2 font-mono text-xs cursor-pointer hover:bg-muted/50 transition-colors"
                onClick={() => onViewDetail(action)}
              >
                {action.type === 'tool' && (
                  <ToolDetailsRenderer action={action} onOpenTerminal={onOpenTerminal} isRunning={isRunning} t={t} onFileClick={onFileClick} onAgentClick={onAgentClick} />
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
  onAgentClick?: (agentId: string, agentName: string) => void;
}

function StepItem({ step, index, onOpenTerminal, onViewDetail, onFileClick, onAgentClick }: StepItemProps) {
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
        {isCompleted ? (
          <CheckCircle2 className="w-5 h-5 text-green-500" />
        ) : isFailed ? (
          <Info className="w-5 h-5 text-red-500" />
        ) : (
          <Loader2 className="w-5 h-5 text-primary animate-spin" />
        )}

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
            className="overflow-hidden"
          >
            {/* Actions List (replaces nested Execution Details) */}
            <div className="ml-2.5 pl-6 border-l-2 border-border/40 space-y-2 pt-1 pb-2">
              {step.actions.map((action) => (
                <StepActionItem
                  key={action.id}
                  action={action}
                  onViewDetail={onViewDetail}
                  onOpenTerminal={onOpenTerminal}
                  onFileClick={onFileClick}
                  onAgentClick={onAgentClick}
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
  const router = useRouter();

  const { openFilePreview, dispatch } = useApp();

  const handleAgentClick = useCallback((agentId: string, agentName: string) => {
    router.push(`/agent/${agentId}`);
  }, [router]);

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
              onAgentClick={handleAgentClick}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

export default TraceEventRenderer;

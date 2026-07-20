import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
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
  GitMerge,
  Check,
  Shield,
  MessageSquare,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useApp } from '@/contexts/app-context-chat';
import { useI18n, type Translate } from '@/contexts/i18n-context';
import { MarkdownRenderer } from "@/components/ui/markdown-renderer";
import { ScrollArea } from "@/components/ui/scroll-area";
import { normalizeTimestampMs } from '@/lib/time-utils';
import { InlineFilePreview } from '@/components/file/inline-file-preview';
import {
  isStoppedTraceProcessStatus,
  resolveTraceProcessStatus,
} from '@/lib/trace-process-status';

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
  artifacts?: ToolArtifact[];
}

interface ToolArtifact {
  type?: string;
  file_id?: string;
  filename?: string;
  mime_type?: string;
  preview_url?: string;
  display?: string;
}

interface TraceEvent {
  event_id?: string;
  event_type?: string;
  action_type?: string;
  step_id?: string | null;
  timestamp?: number | string | null;
  data?: {
    action?: string;
    step_name?: string;
    description?: string;
    tool_names?: string[];
    model_name?: string;
    tool_name?: string;
    tool_args?: ToolArgs;
    tool_params?: ToolArgs;
    selected?: boolean;
    skill_name?: string;
    response?: {
      reasoning?: string;
      tool_name?: string;
      tool_args?: ToolArgs;
      tool_params?: ToolArgs;
      answer?: string;
      assistant_content?: string;
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

function sanitizeTraceEvents(value: unknown): TraceEvent[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((candidate) => {
    if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) {
      return [];
    }
    const source = candidate as Record<string, unknown>;
    const event = { ...source };
    const stringFields = [
      'event_id',
      'event_type',
      'action_type',
      'tool_name',
      'result_type',
    ];
    for (const field of stringFields) {
      if (field in event && typeof event[field] !== 'string') {
        delete event[field];
      }
    }
    if (
      'step_id' in event &&
      typeof event.step_id !== 'string' &&
      event.step_id !== null
    ) {
      delete event.step_id;
    }
    if (
      'timestamp' in event &&
      typeof event.timestamp !== 'number' &&
      typeof event.timestamp !== 'string' &&
      event.timestamp !== null
    ) {
      delete event.timestamp;
    }
    if (
      'data' in event &&
      (!event.data || typeof event.data !== 'object' || Array.isArray(event.data))
    ) {
      event.data = {};
    }
    return [event as TraceEvent];
  });
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
    artifacts?: ToolArtifact[];
    reasoning?: string;
    assistant_content?: string;
    error?: any;
    tool_calls?: any;
    tool_call_id?: string;
    sandboxed?: boolean;
    inline?: boolean;
    workforceSummary?: boolean;
  };
}

function formatActionContent(value: unknown): string {
  if (value === undefined || value === null) {
    return '';
  }
  if (typeof value === 'string') {
    return value;
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

interface ProcessedStep {
  stepId: string;
  stepName: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'paused' | 'waiting_for_user';
  tools: Array<{ function: { name: string } }>;
  reasoning?: string;
  code: string;
  output: string;
  filePath?: string;
  actions: StepAction[];
  agentExecution?: AgentExecutionSummary;
}

interface TraceEventRendererProps {
  events: TraceEvent[];
  taskStatus?: string;
  onOpenExecutionPlan?: () => void;
  onAgentExecutionClick?: (execution: AgentExecutionSummary) => void;
  defaultExpandSteps?: boolean;
}

export interface AgentExecutionSummary {
  workerTaskId: string;
  agentId?: number;
  agentName: string;
  workerAlias?: string;
  status: 'running' | 'completed' | 'failed';
}

function isDagExecutionEvent(event: TraceEvent): boolean {
  const eventType = event.event_type || '';
  return eventType === 'dag_execution'
    || eventType === 'dag_execute_start'
    || eventType === 'dag_execute_end'
    || eventType === 'dag_plan_start'
    || eventType === 'dag_plan_end'
    || eventType.startsWith('dag_step_');
}

function getTraceData(event: TraceEvent): NonNullable<TraceEvent['data']> | undefined {
  if (event.data && typeof event.data === 'object' && !Array.isArray(event.data)) {
    return event.data;
  }
  if ('data' in event) {
    return undefined;
  }
  return event as unknown as NonNullable<TraceEvent['data']>;
}

const getWaitingQuestionFromEvents = (events: TraceEvent[]): string | null => {
  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i];
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
      const result = event.data?.result as any;
      if (
        result?.status === 'waiting_for_user' &&
        typeof result.message === 'string' &&
        result.message.trim()
      ) {
        return result.message;
      }
    }
  }
  return null;
};

const isAgentProgressEvent = (event: TraceEvent): boolean => (
  event.event_type === 'agent_progress' ||
  (
    event.event_type === 'agent_message' &&
    event.data?.expect_response !== true &&
    event.data?.message_type !== 'question'
  )
);

// Process trace events into steps
// Pure reducer over trace events -> ordered steps. Exported for unit testing
// (e.g. tool_call_id attribution under in-turn tool concurrency).
export function processTraceEvents(
  events: TraceEvent[],
  t: Translate,
  taskStatus?: string,
): ProcessedStep[] {
    const stepsMap = new Map<string, ProcessedStep>();
    // Steps that ever had more than one tool in flight at once. Their step-level
    // output scalar is meaningless (whichever tool finishes last would clobber
    // it), so once a step is flagged concurrent we stop writing step.output and
    // rely on the per-action outputs instead.
    const concurrentSteps = new Set<string>();
    let currentReactStepId: string | null = null;
    let lastCompletedWorkforceStepId: string | null = null;
    const orderedEvents = sanitizeTraceEvents(events)
      .map((event, index) => {
        const timestamp = normalizeTimestampMs(event.timestamp)
        return {
          event,
          index,
          timestamp: Number.isFinite(timestamp) ? timestamp : 0,
        }
      })
      .sort((a, b) => a.timestamp - b.timestamp || a.index - b.index);

    // Helper to find the last running action of a specific type
    const findLastRunningAction = (step: ProcessedStep, type: 'llm' | 'tool') => {
      for (let i = step.actions.length - 1; i >= 0; i--) {
        if (step.actions[i].type === type && step.actions[i].status === 'running') {
          return step.actions[i];
        }
      }
      return null;
    };

    // Match a running tool action by its tool_call_id. With concurrent tool
    // execution several same-named tools can be in flight at once, so pairing
    // by "last running tool" mis-attributes results; the id makes it exact.
    // Returns null when the id is absent so callers can fall back to
    // findLastRunningAction (legacy / single-tool events).
    const findRunningToolByCallId = (step: ProcessedStep, toolCallId?: string) => {
      if (!toolCallId) return null;
      for (let i = step.actions.length - 1; i >= 0; i--) {
        const action = step.actions[i];
        if (
          action.type === 'tool' &&
          action.status === 'running' &&
          action.data?.tool_call_id === toolCallId
        ) {
          return action;
        }
      }
      return null;
    };

    orderedEvents.forEach(({ event, index, timestamp }) => {
      if (event.event_type?.startsWith('skill_select')) {
        return;
      }

      const eventData = getTraceData(event) || {};
      let stepId = event.step_id || (eventData.step_id as string) || 'default';
      const isProgressMessage = isAgentProgressEvent(event);
      const workerTaskId = typeof eventData.worker_task_id === 'string'
        || typeof eventData.worker_task_id === 'number'
        ? String(eventData.worker_task_id)
        : '';
      const isWorkforceDelegation = event.event_type?.startsWith('workforce_delegation_');
      const isDelegatedChildEvent =
        eventData.source === 'xagent-agent-tool-child' && Boolean(workerTaskId);
      const isWorkforceManagerSummary =
        isProgressMessage &&
        !isDelegatedChildEvent &&
        Boolean(lastCompletedWorkforceStepId);
      if (isWorkforceManagerSummary && lastCompletedWorkforceStepId) {
        stepId = lastCompletedWorkforceStepId;
      }
      if (isWorkforceDelegation || isDelegatedChildEvent) {
        stepId = String(
          workerTaskId ||
          `workforce-${eventData.workforce_run_id || 'run'}-${eventData.worker_member_id || eventData.agent_id || event.event_id || index}`
        );
      }

      if (event.event_type === 'react_task_start' || event.event_type === 'task_start_react') {
        currentReactStepId = stepId;
      }

      if ((event.event_type === 'react_task_end' || event.event_type === 'task_end_react' || event.event_type === 'task_completion' || event.event_type === 'react_task_failed' || event.event_type === 'task_failed_react') && stepId === 'default' && currentReactStepId) {
        stepId = currentReactStepId;
      }

      if (isProgressMessage && stepId === 'default' && currentReactStepId) {
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
      const eventId = event.event_id || `event-${index}`;

      if ((isWorkforceDelegation || isDelegatedChildEvent) && workerTaskId) {
        const previousExecution = step.agentExecution;
        const agentName = String(
          eventData.worker_alias ||
          eventData.agent_name ||
          eventData.tool_name ||
          previousExecution?.agentName ||
          t('traceEventRenderer.unknownWorker')
        );
        step.agentExecution = {
          workerTaskId,
          agentId: typeof eventData.agent_id === 'number' ? eventData.agent_id : previousExecution?.agentId,
          agentName,
          workerAlias: typeof eventData.worker_alias === 'string' ? eventData.worker_alias : previousExecution?.workerAlias,
          status: event.event_type === 'workforce_delegation_error'
            ? 'failed'
            : event.event_type === 'workforce_delegation_end'
              ? 'completed'
              : previousExecution?.status || 'running',
        };
      }

      // Process different event types
      if (event.event_type === 'dag_step_start' || event.event_type === 'react_task_start') {
        const delegatedAgentName = isDelegatedChildEvent && typeof eventData.agent_name === 'string'
          ? eventData.agent_name
          : '';
        step.stepName = delegatedAgentName
          || (eventData.step_name as string)
          || (event.event_type === 'react_task_start' ? t('traceEventRenderer.taskExecution') : '');
        step.description = (eventData.description as string) || (eventData.task as string) || '';
        step.status = 'running';

        const tools = eventData.tool_names || eventData.tools;
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
          title: t('traceEventRenderer.callLLM', { model: eventData.model_name || t('traceEventRenderer.unknownModel') }),
          status: 'running',
          timestamp,
          data: { model: eventData.model_name }
        });
      }

      if (event.event_type === 'llm_call_end' || event.event_type === 'llm_call_result') {
        if (eventData.response?.reasoning) {
          step.reasoning = eventData.response.reasoning;
        }
        if (eventData.tools) {
          step.tools = eventData.tools;
        }

        const action = findLastRunningAction(step, 'llm');
        if (action) {
          action.status = 'completed';
          action.data.reasoning = eventData.response?.reasoning;
          action.data.tool_calls = eventData.tools;
        } else {
          // Fallback if no start event found
          step.actions.push({
            id: eventId,
            type: 'llm',
            title: t('traceEventRenderer.llmResponse'),
            status: 'completed',
            timestamp,
            data: {
              reasoning: eventData.response?.reasoning,
              tool_calls: eventData.tools
            }
          });
        }
      }

      if (isProgressMessage) {
        const message = event.data?.message || event.data?.content;
        if (typeof message === 'string' && message.trim()) {
          if (!step.stepName) {
            step.stepName = t('traceEventRenderer.taskExecution');
          }
          if (step.status === 'pending') {
            step.status = 'running';
          }
          step.actions.push({
            id: eventId,
            type: 'info',
            title: t('traceEventRenderer.progressMessage'),
            status: 'completed',
            timestamp,
            data: {
              output: message.trim(),
              inline: true,
              workforceSummary: isWorkforceManagerSummary,
            }
          });
        }
      }

      if (event.event_type === 'workforce_delegation_start') {
        lastCompletedWorkforceStepId = null;
        const workerName = String(
          eventData.worker_alias ||
          eventData.agent_name ||
          eventData.tool_name ||
          t('traceEventRenderer.unknownWorker')
        );
        step.stepName = t('traceEventRenderer.workforceDelegation');
        step.description = t('traceEventRenderer.delegateToWorker', { worker: workerName });
        step.status = 'running';
        step.actions.push({
          id: eventId,
          type: 'info',
          title: t('traceEventRenderer.delegateToWorker', { worker: workerName }),
          status: 'running',
          timestamp,
          data: {
            tool: eventData.tool_name,
            output: eventData,
          }
        });
      }

      if (event.event_type === 'workforce_delegation_end') {
        const output = eventData.output || eventData.response || '';
        step.stepName = step.stepName || t('traceEventRenderer.workforceDelegation');
        step.description = step.description || t('traceEventRenderer.workforceDelegation');
        step.status = 'completed';
        const action = step.actions.find((item) => item.type === 'info' && item.status === 'running');
        if (action) {
          action.status = 'completed';
          action.data.output = output || eventData;
        } else {
          step.actions.push({
            id: eventId,
            type: 'info',
            title: t('traceEventRenderer.workerReturned'),
            status: 'completed',
            timestamp,
            data: { output: output || eventData }
          });
        }
        lastCompletedWorkforceStepId = stepId;
      }

      if (event.event_type === 'workforce_delegation_error') {
        const errorMessage = eventData.error || eventData.message || t('traceEventRenderer.unknownError');
        step.stepName = step.stepName || t('traceEventRenderer.workforceDelegation');
        step.description = step.description || t('traceEventRenderer.workforceDelegation');
        step.status = 'failed';
        const action = step.actions.find((item) => item.type === 'info' && item.status === 'running');
        if (action) {
          action.type = 'error';
          action.title = t('traceEventRenderer.workerFailed');
          action.status = 'failed';
          action.data.output = undefined;
          action.data.error = errorMessage;
        } else {
          step.actions.push({
            id: eventId,
            type: 'error',
            title: t('traceEventRenderer.workerFailed'),
            status: 'failed',
            timestamp,
            data: { error: errorMessage }
          });
        }
        lastCompletedWorkforceStepId = stepId;
      }

      if (event.event_type === 'tool_execution_start') {
        // Support v1 tool_args and v2 tool_params shapes.
        const toolArgs =
          event.data?.response?.tool_args ||
          event.data?.response?.tool_params ||
          event.data?.tool_args ||
          event.data?.tool_params;
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
        const toolCallId = event.data?.tool_call_id as string | undefined;
        const assistantContent = event.data?.response?.assistant_content || event.data?.assistant_content;

        if (typeof assistantContent === 'string' && assistantContent.trim()) {
          step.actions.push({
            id: `${eventId}-assistant-content`,
            type: 'info',
            title: t('traceEventRenderer.toolCallNote'),
            status: 'completed',
            timestamp,
            data: {
              output: assistantContent.trim(),
              inline: true,
            }
          });
        }

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
            tool_call_id: toolCallId,
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

        // Detect *actual* concurrency, not just multiple tools in the step: at
        // the moment a tool ends it is still marked 'running' (its status is set
        // below), so >1 running tool here means siblings overlapped it. Counting
        // total tool actions instead would wrongly suppress step.output for
        // tools that merely ran sequentially within the same step. Once a step
        // is concurrent the scalar is unreliable, so keep the per-action outputs
        // authoritative and stop writing step.output for it.
        const runningTools = step.actions.filter(
          a => a.type === 'tool' && a.status === 'running'
        );
        if (runningTools.length > 1) {
          concurrentSteps.add(stepId);
        }
        if (!concurrentSteps.has(stepId)) {
          step.output = output;
        }
        const artifacts =
          typeof result === 'object' &&
            result !== null &&
            Array.isArray(result.artifacts)
            ? result.artifacts
            : undefined;

        const endToolCallId = event.data?.tool_call_id as string | undefined;
        const action =
          findRunningToolByCallId(step, endToolCallId) ||
          findLastRunningAction(step, 'tool');
        if (action) {
          action.status = 'completed';
          action.data.output = output;
          if (artifacts) {
            action.data.artifacts = artifacts;
          }
        } else {
          // Fallback
          step.actions.push({
            id: eventId,
            type: 'tool',
            title: t('traceEventRenderer.toolExecutionFinished'),
            status: 'completed',
            timestamp,
            data: {
              output,
              artifacts,
              sandboxed: !!event.data?.sandboxed
            }
          });
        }
      }

      if (event.event_type === 'dag_step_end' || event.event_type === 'step_completed' || event.event_type === 'react_task_end' || event.event_type === 'task_completion') {
        step.status = 'completed';
        const shouldShowStepResult = event.event_type === 'dag_step_end' || event.event_type === 'step_completed';
        const result = shouldShowStepResult
          ? event.data?.result ?? event.data?.result_data ?? event.data?.response?.answer
          : undefined;
        const content = formatActionContent(result);
        if (shouldShowStepResult && content.trim()) {
          step.output = content;
          step.actions.push({
            id: `${eventId}-result`,
            type: 'info',
            title: t('traceEventRenderer.stepResult'),
            status: 'completed',
            timestamp,
            data: { output: content }
          });
        }
        // Ensure all actions are completed
        step.actions.forEach(a => {
          if (a.status === 'running') a.status = 'completed';
        });
      }

      if (['dag_step_failed', 'tool_execution_failed', 'llm_call_failed', 'react_task_failed', 'agent_error', 'trace_error'].includes(event.event_type as string)) {
        const isTerminalFailure =
          ['dag_step_failed', 'react_task_failed', 'agent_error', 'trace_error'].includes(event.event_type as string);
        if (isTerminalFailure) {
          step.status = 'failed';
        } else if (step.status === 'pending') {
          step.status = 'running';
        }

        // Extract error message with more fallback options
        const errorData = event.data || {};
        let errorMessage =
          errorData.error ||
          errorData.message ||
          errorData.error_message;

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
          const lastTool =
            findRunningToolByCallId(step, errorData.tool_call_id as string | undefined) ||
            findLastRunningAction(step, 'tool');
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

    const steps = Array.from(stepsMap.values()).filter(step => step.stepName);

    if (isStoppedTraceProcessStatus(taskStatus)) {
      steps.forEach((step) => {
        const runningActions = step.actions.filter(action => action.status === 'running');
        if (taskStatus === 'failed' && (step.status !== 'completed' || runningActions.length > 0)) {
          step.status = 'failed';
          runningActions.forEach((action) => {
            action.status = 'failed';
            action.data.error = action.data.error || t('traceEventRenderer.unknownError');
          });
          return;
        }

        if (taskStatus === 'completed' && step.status !== 'failed') {
          step.status = 'completed';
          runningActions.forEach((action) => {
            action.status = 'completed';
          });
          return;
        }

        if (
          (taskStatus === 'paused' || taskStatus === 'waiting_for_user') &&
          (step.status === 'pending' || step.status === 'running')
        ) {
          step.status = taskStatus;
          runningActions.forEach((action) => {
            action.status = 'completed';
          });
        }
      });
    }

    return steps;
}

function useProcessedSteps(events: TraceEvent[], taskStatus?: string): ProcessedStep[] {
  const { t } = useI18n();
  return useMemo(
    () => processTraceEvents(events, t, taskStatus),
    [events, taskStatus, t],
  );
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

const ToolArtifactsDisplay = ({ artifacts, onFileClick, t }: { artifacts?: ToolArtifact[]; onFileClick?: (filePath: string, fileName: string) => void; t: Translate }) => {
  const displayArtifacts = (artifacts || []).filter(
    artifact => artifact && (artifact.preview_url || artifact.file_id) && (artifact.display === undefined || artifact.display === 'inline')
  );

  if (displayArtifacts.length === 0) return null;

  return (
    <div className="mt-4 grid gap-3">
      {displayArtifacts.map((artifact, index) => (
        <InlineFilePreview
          key={`${artifact.file_id || artifact.preview_url || index}`}
          source={{
            fileId: artifact.file_id,
            previewUrl: artifact.preview_url,
            filename: artifact.filename,
            mimeType: artifact.mime_type,
            type: artifact.type,
          }}
          openLabel={t('files.previewDialog.buttons.open')}
          loadErrorText={t('files.previewDialog.errors.loadFailed')}
          onFileClick={onFileClick}
        />
      ))}
    </div>
  );
};

const ToolOutputDisplay = ({ action, isRunning, t, onFileClick, onAgentClick }: { action: StepAction, isRunning: boolean, t: any, onFileClick?: (filePath: string, fileName: string) => void, onAgentClick?: (agentId: string, agentName: string) => void }) => (
  <>
    <ToolArtifactsDisplay artifacts={action.data.artifacts} onFileClick={onFileClick} t={t} />
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
  if (toolName === 'python_executor' || toolName === 'execute_python_code') {
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
  const summaryMetaRef = useRef<HTMLDivElement>(null);
  const fixedMetaRef = useRef<HTMLDivElement>(null);
  const summaryMeasureRef = useRef<HTMLSpanElement>(null);
  const [hideToolSummary, setHideToolSummary] = useState(false);

  const summary = useMemo(() => {
    if (action.type === 'llm') {
      if (action.data.reasoning) {
        const clean = action.data.reasoning.replace(/[\n\r\s]+/g, ' ').trim();
        return clean.length > 50 ? clean.slice(0, 50) + '...' : clean;
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

      if (args && typeof args === 'object') {
        if ('file_path' in args) return `${t('traceEventRenderer.filePrefix')} ${String(args.file_path)}`;
        if ('query' in args) return `${t('traceEventRenderer.queryPrefix')} ${String(args.query)}`;
        if ('path' in args) return `${t('traceEventRenderer.pathPrefix')} ${String(args.path)}`;
      }

      if (code) {
        const clean = code.replace(/[\n\r\s]+/g, ' ').trim();
        return clean.length > 50 ? clean.slice(0, 50) + '...' : clean;
      }

      if (args) {
        try {
          const str = JSON.stringify(args);
          return str.length > 50 ? str.slice(0, 50) + '...' : str;
        } catch (e) { return null; }
      }
    }
    if (action.type === 'info' && action.data.output) {
      const clean = formatActionContent(action.data.output).replace(/[\n\r\s]+/g, ' ').trim();
      return clean.length > 50 ? clean.slice(0, 50) + '...' : clean;
    }
    return null;
  }, [action.type, action.data, t]);

  const updateToolSummaryVisibility = useCallback(() => {
    if (action.type !== 'tool' || !summary) {
      setHideToolSummary(false);
      return;
    }

    if (typeof window !== 'undefined' && window.innerWidth < 640) {
      setHideToolSummary(true);
      return;
    }

    const container = summaryMetaRef.current;
    const measure = summaryMeasureRef.current;
    const fixed = fixedMetaRef.current;

    if (!container || !measure) {
      setHideToolSummary(false);
      return;
    }

    const fixedWidth = fixed?.offsetWidth ?? 0;
    const availableWidth = container.clientWidth - fixedWidth - 8;
    setHideToolSummary(availableWidth <= 0 || measure.scrollWidth > availableWidth);
  }, [action.type, summary]);

  useEffect(() => {
    updateToolSummaryVisibility();

    const resizeObserver = new ResizeObserver(() => {
      updateToolSummaryVisibility();
    });

    if (summaryMetaRef.current) resizeObserver.observe(summaryMetaRef.current);
    if (fixedMetaRef.current) resizeObserver.observe(fixedMetaRef.current);
    if (summaryMeasureRef.current) resizeObserver.observe(summaryMeasureRef.current);

    return () => {
      resizeObserver.disconnect();
    };
  }, [updateToolSummaryVisibility]);

  if (action.type === 'info' && action.data.inline) {
    return (
      <div className="px-3 py-1.5">
        <MarkdownRenderer
          content={formatActionContent(action.data.output)}
          onFileClick={onFileClick}
          className="text-sm leading-relaxed text-foreground prose-neutral dark:prose-invert max-w-none [&>p]:mb-1.5 [&>p:last-child]:mb-0"
        />
      </div>
    );
  }

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
        <div className="flex flex-1 items-start gap-2 min-w-0">
          <div className="flex items-start gap-2 min-w-0">
            <span className="flex-shrink-0 flex items-center">
              {action.type === 'tool' && <Wrench className="w-3.5 h-3.5" />}
              {action.type === 'error' && <Info className="w-3.5 h-3.5 text-red-500" />}
              {action.type === 'info' && <MessageSquare className="w-3.5 h-3.5" />}
            </span>

            <span className="font-medium break-words [overflow-wrap:anywhere]">{action.title}</span>
          </div>

          <div ref={summaryMetaRef} className="relative flex items-center gap-1 min-w-0 flex-1 overflow-hidden">
            <div ref={fixedMetaRef} className="flex items-center gap-1 shrink-0">
              {action.data.sandboxed && (
                <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-green-500/10 text-green-600 dark:text-green-400 border border-green-500/20 whitespace-nowrap flex-shrink-0">
                  <Shield className="w-3 h-3" />
                  {t('traceEventRenderer.sandboxedExecution')}
                </span>
              )}

              {isRunning && <Loader2 className="w-3 h-3 animate-spin ml-1 flex-shrink-0" />}
            </div>

            {summary && (action.type !== 'tool' || !hideToolSummary) && (
              <span className="text-muted-foreground/50 font-normal ml-1 hidden sm:block min-w-0 truncate">
                - {summary}
              </span>
            )}
            {summary && action.type === 'tool' && (
              <span
                ref={summaryMeasureRef}
                aria-hidden="true"
                className="pointer-events-none absolute invisible whitespace-nowrap"
              >
                - {summary}
              </span>
            )}
          </div>
        </div>
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

                {action.type === 'info' && (
                  <MarkdownRenderer
                    content={formatActionContent(action.data.output)}
                    onFileClick={onFileClick}
                    className="text-sm leading-relaxed prose-neutral dark:prose-invert max-w-none"
                  />
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
  onAgentExecutionClick?: (execution: AgentExecutionSummary) => void;
  defaultExpanded?: boolean;
}

function StepItem({ step, index, onOpenTerminal, onViewDetail, onFileClick, onAgentClick, onAgentExecutionClick, defaultExpanded = false }: StepItemProps) {
  const { t } = useI18n();
  const isCompleted = step.status === 'completed';
  const isFailed = step.status === 'failed';
  const isPaused = step.status === 'paused' || step.status === 'waiting_for_user';
  const [isExpanded, setIsExpanded] = useState(() => defaultExpanded || !isCompleted);
  const wasCompletedRef = useRef(isCompleted);
  const rawTitle = step.description || step.stepName;
  const displayTitle =
    isCompleted && step.stepName === t('traceEventRenderer.taskExecution') && !step.description
      ? t('traceEventRenderer.thoughtProcess')
      : rawTitle;
  const workforceSummaries = step.actions.filter(
    (action) =>
      action.data.workforceSummary &&
      typeof action.data.output === 'string' &&
      action.data.output.trim(),
  );
  const processActions = step.actions.filter((action) => !action.data.workforceSummary);
  const usesAgentInspector = Boolean(step.agentExecution && onAgentExecutionClick);
  const canExpandProcess = processActions.length > 0 && !usesAgentInspector;
  const stepTitleContent = (
    <>
      {isCompleted ? (
        <CheckCircle2 className="w-5 h-5 text-green-500 mt-0.5" />
      ) : isFailed ? (
        <Info className="w-5 h-5 text-red-500 mt-0.5" />
      ) : isPaused ? (
        <Info className="w-5 h-5 text-yellow-500 mt-0.5" />
      ) : (
        <Loader2 className="w-5 h-5 text-primary animate-spin mt-0.5" />
      )}
      <h3 className="min-w-0 flex-1 text-sm font-medium text-foreground break-words [overflow-wrap:anywhere]">
        {displayTitle}
      </h3>
    </>
  );

  useEffect(() => {
    if (isCompleted && !wasCompletedRef.current && !defaultExpanded) {
      setIsExpanded(false);
    }
    wasCompletedRef.current = isCompleted;
  }, [defaultExpanded, isCompleted]);

  const handleToggle = () => {
    setIsExpanded((expanded) => !expanded);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 * (index + 1) }}
      className="space-y-3"
    >
      {/* Step Title */}
      <div className="flex w-full items-start gap-2 rounded-lg px-2 py-1 -ml-2 transition-colors hover:bg-muted/50 group/step">
        {usesAgentInspector ? (
          <div className="flex min-w-0 flex-1 items-start gap-2 text-left">
            {stepTitleContent}
          </div>
        ) : (
          <button
            type="button"
            className="flex min-w-0 flex-1 items-start gap-2 text-left"
            onClick={handleToggle}
            aria-expanded={isExpanded}
          >
            {stepTitleContent}
          </button>
        )}
        {step.agentExecution && onAgentExecutionClick && (
          <button
            type="button"
            className="mt-0.5 shrink-0 text-[11px] font-medium text-primary hover:underline"
            onClick={() => onAgentExecutionClick(step.agentExecution!)}
          >
            {t('traceEventRenderer.viewAgentExecution')}
          </button>
        )}
        {canExpandProcess && (
          <button
            type="button"
            className="mt-0.5 shrink-0 inline-flex items-center gap-1 rounded-full border border-border/60 bg-background/80 px-2 py-0.5 text-[11px] font-medium text-muted-foreground transition-colors group-hover/step:text-foreground"
            onClick={handleToggle}
            aria-expanded={isExpanded}
          >
            {isExpanded ? t('traceEventRenderer.hideProcess') : t('traceEventRenderer.showProcess')}
            {isExpanded ? (
              <ChevronDown className="w-3.5 h-3.5" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5" />
            )}
          </button>
        )}
      </div>

      {workforceSummaries.map((action) => (
        <div key={`${action.id}-summary`} className="ml-7 pr-2 text-sm">
          <MarkdownRenderer
            content={String(action.data.output)}
            className="prose-sm leading-relaxed"
            onFileClick={onFileClick}
            onAgentClick={onAgentClick}
          />
        </div>
      ))}

      <AnimatePresence>
        {isExpanded && canExpandProcess && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            {/* Actions List (replaces nested Execution Details) */}
            <div className="ml-2.5 pl-6 border-l-2 border-border/40 space-y-2 pt-1 pb-2">
              {processActions.map((action) => (
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
export function TraceEventRenderer({ events, taskStatus, onOpenExecutionPlan, onAgentExecutionClick, defaultExpandSteps = false }: TraceEventRendererProps) {
  const { t } = useI18n();
  const sanitizedEvents = useMemo(() => sanitizeTraceEvents(events), [events]);
  const processStatus = resolveTraceProcessStatus({
    taskStatus,
    traceEvents: sanitizedEvents,
  });
  const steps = useProcessedSteps(sanitizedEvents, processStatus);
  const router = useRouter();

  const { openFilePreview, dispatch } = useApp();

  const handleAgentClick = useCallback((agentId: string, agentName: string) => {
    router.push(`/agent/${agentId}`);
  }, [router]);

  const skillSelection = useMemo(() => {
    for (let i = sanitizedEvents.length - 1; i >= 0; i--) {
      const event = sanitizedEvents[i];
      if (event.event_type === 'skill_select_end') {
        if (event.data?.selected && event.data?.skill_name) {
          return event.data.skill_name as string;
        }
        return null;
      }
    }
    return null;
  }, [sanitizedEvents]);

  const hasExecutionPlan = useMemo(
    () => sanitizedEvents.some(isDagExecutionEvent),
    [sanitizedEvents],
  );

  const executionPlanStepCount = useMemo(() => {
    for (let index = sanitizedEvents.length - 1; index >= 0; index -= 1) {
      if (!isDagExecutionEvent(sanitizedEvents[index])) continue;
      const data = sanitizedEvents[index].data;
      if (!data || typeof data !== 'object') continue;

      const record = data as Record<string, unknown>;
      const planCandidates = [record, record.plan_data, record.current_plan, record.plan];
      for (const candidate of planCandidates) {
        if (!candidate || typeof candidate !== 'object') continue;
        const planSteps = (candidate as Record<string, unknown>).steps;
        if (Array.isArray(planSteps) && planSteps.length > 0) {
          return planSteps.length;
        }
      }
    }

    return steps.length > 0 ? steps.length : null;
  }, [sanitizedEvents, steps.length]);

  const executionPlanLabel = executionPlanStepCount === null
    ? t('chatPage.executionPlan.dagSection')
    : t(
      executionPlanStepCount === 1
        ? 'chatPage.executionPlan.dagSectionOne'
        : 'chatPage.executionPlan.dagSectionOther',
      { count: executionPlanStepCount },
    );

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
      if (action.data.assistant_content) {
        content += `\n\n${t('traceEventRenderer.toolCallNote')}\n${action.data.assistant_content}`;
      }
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

  return (
    <div className="space-y-4">
      {hasExecutionPlan && onOpenExecutionPlan && (
        <button
          type="button"
          className="flex w-full items-center justify-between gap-3 rounded-lg border border-border/50 bg-muted/20 px-3 py-2 text-left transition-colors hover:border-border hover:bg-muted/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          onClick={onOpenExecutionPlan}
          title={t('chatPage.executionPlan.tooltip')}
        >
          <div className="flex min-w-0 items-center gap-2">
            <GitMerge className="h-4 w-4 shrink-0 text-muted-foreground" />
            <span className="truncate text-sm font-medium text-foreground">
              {executionPlanLabel}
            </span>
          </div>
          <span className="flex shrink-0 items-center text-xs font-medium text-muted-foreground">
            {t('chatPage.executionPlan.view')}
            <ChevronRight className="ml-1 h-3.5 w-3.5" />
          </span>
        </button>
      )}
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
              onAgentExecutionClick={onAgentExecutionClick}
              defaultExpanded={defaultExpandSteps}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

export default TraceEventRenderer;

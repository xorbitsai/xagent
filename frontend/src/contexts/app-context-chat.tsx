"use client"

import React, { createContext, useContext, useReducer, useCallback, useEffect, useLayoutEffect, useState, useRef, useMemo } from "react"
import { useRouter } from "next/navigation"
import { FileText, Target, Zap, CheckCircle, XCircle, Wrench, Activity, Search, Lightbulb, AlertTriangle, Info, Brain, Bot } from "lucide-react"
import { JsonRenderer, MarkdownRenderer } from "../components/ui/markdown-renderer"
import { UserMessageContent } from "@/components/chat/user-message-content"
import { ReplayScheduler } from '@/lib/replay-scheduler'
import { CollapsibleSection } from "@/components/collapsible-section"
import { Badge } from "@/components/ui/badge"
import { ClarificationForm } from "@/components/chat/clarification-form"
import {
  AgentCardPresentationCapability,
  LinksOpenInNewTabCapability,
  resolveAgentCardPresentationCapability,
} from "@/contexts/presentation-capabilities"
import {
  FileAccessProvider,
  type FileAccessPolicy,
} from "@/contexts/file-access-context"

interface WebSocketMessage {
  type: string
  data?: unknown
  timestamp: string
  task_id?: number
  step_id?: string
  event_type?: string
  event_id?: string
  run_id?: string | null
  state_version?: number
  control_state?: TaskControlState
  status?: unknown
  task?: Record<string, unknown>
}

type TaskControlState =
  | "idle"
  | "running"
  | "pause_requested"
  | "paused"
  | "resume_requested"
  | "waiting_for_user"
  | "completed"
  | "failed"

type TaskControlEnvelope = {
  isStateEvent: boolean
  taskId?: number
  runId?: string | null
  stateVersion?: number
  controlState?: TaskControlState
  status?: TaskStatus
}

const VERSIONED_TASK_EVENT_TYPES = new Set([
  "agent_error",
  "error",
  "task_completed",
  "task_error",
  "task_pause_requested",
  "task_paused",
  "task_resumed",
  "task_started",
  "task_waiting_for_user",
])
// Action types that describe state scoped to one specific task's
// conversation (chat/trace/DAG/status). Used by handleMessage's dispatch
// wrapper to drop these when a message's task_id doesn't match the task
// actually being viewed - see the wrapper's own comment for why. Kept at
// module scope (built once) rather than inside handleMessage, matching
// VERSIONED_TASK_EVENT_TYPES above.
const TASK_SCOPED_ACTION_TYPES = new Set<AppAction["type"]>([
  "SET_DAG_EXECUTION",
  "SET_STEPS",
  "ADD_STEP",
  "UPDATE_STEP",
  "UPDATE_TASK_STATUS",
  "SET_PROCESSING",
  "ADD_MESSAGE",
  "UPSERT_STREAMING_FINAL_ANSWER",
  "ADD_TRACE_EVENT",
  "SET_CONTEXT_USAGE",
  "SET_PLAN_MEMORY_INFO",
])
const MAX_TRACKED_TASK_STATE_VERSIONS = 500
// A Session may reset repeatedly during its absolute lifetime. Retired ids are
// retained only to reject late frames, so bound this lineage guard separately
// from the unrelated task-state ordering cache.
const MAX_RETIRED_SESSION_TASK_IDS = 500
const SESSION_RESET_ACK_TIMEOUT_MS = 30_000
const SESSION_TASK_ADOPTION_TIMEOUT_MS = 30_000
const MAX_SESSION_PRE_ADOPTION_FRAMES = 64
const MAX_SESSION_PRE_ADOPTION_BYTES = 256 * 1024

type SessionConversationState =
  | { phase: "unbound"; connectionIdentity: string | null; taskId: null }
  | { phase: "bound"; connectionIdentity: string; taskId: number }
  | { phase: "reset_requested"; connectionIdentity: string; taskId: number }
  | { phase: "replacement_ready"; connectionIdentity: string; taskId: null }
  | { phase: "replacement_sending"; connectionIdentity: string; taskId: null }
  | { phase: "replacement_awaiting_task"; connectionIdentity: string; taskId: null }
  | { phase: "reload_required"; connectionIdentity: string | null; taskId: null }

type SessionConversationAction =
  | { type: "SESSION_TASK_INFO"; connectionIdentity: string; taskId: number }
  | { type: "SESSION_RESET_REQUESTED"; connectionIdentity: string; taskId: number }
  | { type: "SESSION_RESET_NOT_SENT"; connectionIdentity: string }
  | { type: "SESSION_RESET_ACKNOWLEDGED"; connectionIdentity: string }
  | { type: "SESSION_REPLACEMENT_SENDING"; connectionIdentity: string }
  | { type: "SESSION_REPLACEMENT_ACCEPTED"; connectionIdentity: string }
  | { type: "SESSION_REPLACEMENT_REJECTED"; connectionIdentity: string }
  | { type: "SESSION_BOUND_CONNECTION_REBOUND"; connectionIdentity: string }
  | { type: "SESSION_RELOAD_REQUIRED"; connectionIdentity: string | null }

const initialSessionConversationState: SessionConversationState = {
  phase: "unbound",
  connectionIdentity: null,
  taskId: null,
}

const reduceSessionConversation = (
  state: SessionConversationState,
  action: SessionConversationAction,
): SessionConversationState => {
  if (state.phase === "reload_required") return state
  switch (action.type) {
    case "SESSION_TASK_INFO":
      if (state.phase === "unbound") {
        return { phase: "bound", connectionIdentity: action.connectionIdentity, taskId: action.taskId }
      }
      if (
        state.phase === "replacement_awaiting_task"
        && state.connectionIdentity === action.connectionIdentity
      ) {
        return { phase: "bound", connectionIdentity: action.connectionIdentity, taskId: action.taskId }
      }
      return state
    case "SESSION_RESET_REQUESTED":
      return state.phase === "bound" && state.connectionIdentity === action.connectionIdentity
        ? { phase: "reset_requested", connectionIdentity: action.connectionIdentity, taskId: action.taskId }
        : state
    case "SESSION_RESET_NOT_SENT":
      return state.phase === "reset_requested" && state.connectionIdentity === action.connectionIdentity
        ? { phase: "bound", connectionIdentity: state.connectionIdentity, taskId: state.taskId }
        : state
    case "SESSION_RESET_ACKNOWLEDGED":
      return state.phase === "reset_requested" && state.connectionIdentity === action.connectionIdentity
        ? { phase: "replacement_ready", connectionIdentity: action.connectionIdentity, taskId: null }
        : state
    case "SESSION_REPLACEMENT_SENDING":
      return state.phase === "replacement_ready" && state.connectionIdentity === action.connectionIdentity
        ? { phase: "replacement_sending", connectionIdentity: action.connectionIdentity, taskId: null }
        : state
    case "SESSION_REPLACEMENT_ACCEPTED":
      return state.phase === "replacement_sending" && state.connectionIdentity === action.connectionIdentity
        ? { phase: "replacement_awaiting_task", connectionIdentity: action.connectionIdentity, taskId: null }
        : state
    case "SESSION_REPLACEMENT_REJECTED":
      return state.phase === "replacement_sending" && state.connectionIdentity === action.connectionIdentity
        ? { phase: "replacement_ready", connectionIdentity: action.connectionIdentity, taskId: null }
        : state
    case "SESSION_BOUND_CONNECTION_REBOUND":
      return state.phase === "bound"
        ? { ...state, connectionIdentity: action.connectionIdentity }
        : state
    case "SESSION_RELOAD_REQUIRED":
      return { phase: "reload_required", connectionIdentity: action.connectionIdentity, taskId: null }
  }
}

type SessionConversationTransition = {
  accepted: boolean
  changed: boolean
  next: SessionConversationState
}

const transitionSessionConversation = (
  current: SessionConversationState,
  action: SessionConversationAction,
): SessionConversationTransition => {
  const next = reduceSessionConversation(current, action)
  const changed = next !== current
  const accepted = changed || (
    action.type === "SESSION_TASK_INFO"
    && current.phase === "bound"
    && current.connectionIdentity === action.connectionIdentity
    && current.taskId === action.taskId
  )
  return { accepted, changed, next }
}

const asMessageRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" ? value as Record<string, unknown> : {}

const parseInteger = (value: unknown): number | undefined => {
  if (typeof value === "number") return Number.isInteger(value) ? value : undefined
  if (typeof value !== "string" || value.trim() === "") return undefined
  const parsed = Number(value)
  return Number.isInteger(parsed) ? parsed : undefined
}

export const extractTaskControlEnvelope = (message: WebSocketMessage): TaskControlEnvelope => {
  const root = message as unknown as Record<string, unknown>
  const data = asMessageRecord(root.data)
  const nestedData = asMessageRecord(data.data)
  const task = asMessageRecord(root.task)
  const eventType = String(root.event_type || data.event_type || "")
  const isStateEvent = VERSIONED_TASK_EVENT_TYPES.has(message.type)
    || (message.type === "trace_event" && eventType === "task_info")
  if (!isStateEvent) return { isStateEvent: false }

  const rawTaskId = root.task_id ?? task.id ?? data.id ?? nestedData.id
  const parsedTaskId = parseInteger(rawTaskId)
  const rawVersion =
    root.state_version
    ?? task.state_version
    ?? data.state_version
    ?? nestedData.state_version
  const parsedVersion = parseInteger(rawVersion)
  const rawRunId =
    root.run_id ?? task.run_id ?? data.run_id ?? nestedData.run_id
  const rawControlState =
    root.control_state
    ?? task.control_state
    ?? data.control_state
    ?? nestedData.control_state
  const rawStatus =
    root.status ?? task.status ?? data.status ?? nestedData.status

  return {
    isStateEvent: true,
    taskId: parsedTaskId !== undefined && parsedTaskId > 0 ? parsedTaskId : undefined,
    runId: typeof rawRunId === "string" || rawRunId === null ? rawRunId : undefined,
    stateVersion: parsedVersion !== undefined && parsedVersion >= 0 ? parsedVersion : undefined,
    controlState: typeof rawControlState === "string" ? rawControlState as TaskControlState : undefined,
    status: normalizeTaskStatus(rawStatus) || undefined,
  }
}

const taskEventMatchesControlState = (
  message: WebSocketMessage,
  envelope: TaskControlEnvelope,
): boolean => {
  const expected: Partial<Record<string, TaskControlState[]>> = {
    task_completed: ["completed", "failed"],
    task_pause_requested: ["pause_requested"],
    task_paused: ["paused"],
    task_resumed: ["running"],
    task_started: ["running"],
    task_waiting_for_user: ["waiting_for_user"],
  }
  const allowed = expected[message.type]
  return !allowed || !envelope.controlState || allowed.includes(envelope.controlState)
}

type TaskStateVersionEntry = {
  version: number
  runId?: string | null
}

const canAcceptTaskControlVersion = (
  message: WebSocketMessage,
  envelope: TaskControlEnvelope,
  versions: Map<number, TaskStateVersionEntry>,
): boolean => {
  if (!envelope.isStateEvent || envelope.taskId === undefined) return true

  const knownState = versions.get(envelope.taskId)
  if (envelope.stateVersion === undefined) {
    // Once the server establishes versioned state for a Task, an unversioned
    // replay cannot roll it back. Error frames remain informational.
    const isErrorEvent =
      message.type === "error" || message.type === "agent_error"
    return !knownState || isErrorEvent
  }

  const isOlderVersion =
    knownState && envelope.stateVersion < knownState.version
  const isDifferentRunAtSameVersion =
    knownState
    && envelope.stateVersion === knownState.version
    && knownState.runId !== undefined
    && envelope.runId !== undefined
    && knownState.runId !== envelope.runId
  if (isOlderVersion || isDifferentRunAtSameVersion) return false

  return true
}

const acceptTaskControlVersion = (
  message: WebSocketMessage,
  envelope: TaskControlEnvelope,
  versions: Map<number, TaskStateVersionEntry>,
): boolean => {
  if (!canAcceptTaskControlVersion(message, envelope, versions)) return false
  if (!envelope.isStateEvent || envelope.taskId === undefined) return true
  if (envelope.stateVersion === undefined) return true

  versions.delete(envelope.taskId)
  versions.set(envelope.taskId, {
    version: envelope.stateVersion,
    runId: envelope.runId,
  })
  if (versions.size > MAX_TRACKED_TASK_STATE_VERSIONS) {
    const oldestTaskId = versions.keys().next().value
    if (oldestTaskId !== undefined) versions.delete(oldestTaskId)
  }
  return true
}

export interface Interaction {
  type: "select_one" | "select_multiple" | "text_input" | "file_upload" | "confirm" | "number_input" | "action_cards";
  field: string;
  label: string;
  options?: Array<{ label: string; value: string; description?: string; action_type?: string }>;
  placeholder?: string;
  multiline?: boolean;
  min?: number;
  max?: number;
  default?: any;
  default_value?: string | number | boolean | null;
  accept?: string[] | string;
  multiple?: boolean;
}
import {
  useWebSocket,
  type WebSocketConnection,
  type WebSocketConnectionFailure,
} from "@/hooks/use-websocket"
import { generateClientMessageId, getApiUrl, getUploadApiUrl, shouldAutoOpenTaskPreview } from "@/lib/utils"
import { apiRequest, getApiErrorMessage, getUploadErrorMessage, isJsonRecord, parseApiResponse, UPLOAD_ERROR_MESSAGES } from "@/lib/api-wrapper"
import { useI18n } from "@/contexts/i18n-context"
import { normalizeTimestampMs } from "@/lib/time-utils"
import { unwrapFinalAnswerContent } from "@/lib/final-answer"
import { normalizeTaskCompletedMessage } from "@/lib/task-completion"
import { emitTaskError } from "@/lib/task-error-events"
import { isStoppedTaskStatus, normalizeTaskStatus, type TaskStatus } from "@/lib/task-status"
import {
  getFinalAnswerStreamActionPayload,
  getFinalAnswerStreamMessageId,
  getWebSocketEventType,
  isFinalAnswerStreamEventType,
  isStreamingFinalAnswerMessage,
  mergeTraceEventsById,
  shouldBufferMessageForHistoricalReplay,
} from "@/lib/streaming-final-answer"
import { extractSharedChatResponse } from "@/lib/chat-response"

// Unique ID generator for messages
let messageIdCounter = 0
const generateMessageId = (prefix: string) => {
  return `${prefix}-${++messageIdCounter}-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`
}

// Providers own their own dedupe cache. The window hook is retained solely for
// legacy hot-reload callers and clears registered provider-local caches.
const duplicateMessageCacheClearers = new Set<() => void>()

// Helper function to compare arrays
const arraysEqual = (a: string[], b: string[]): boolean => {
  if (a === b) return true
  if (a == null || b == null) return false
  if (a.length !== b.length) return false
  return a.every((val, index) => val === b[index])
}

type GeneratedPreviewFile = {
  fileId: string
  fileName: string
}

const normalizeGeneratedPreviewFiles = (files: Array<string | any> | undefined): GeneratedPreviewFile[] => {
  if (!Array.isArray(files)) return []

  return files.map((file) => {
    if (typeof file === 'object' && file !== null) {
      return {
        fileId: file.file_id || '',
        fileName: file.filename || 'unknown',
      }
    }

    return {
      fileId: '',
      fileName: 'unknown',
    }
  }).filter((file) => !!file.fileId)
}

const getAutoOpenGeneratedPreviewIndex = (files: GeneratedPreviewFile[]): number => {
  return files.findIndex((file) => shouldAutoOpenTaskPreview(file.fileName))
}

const dispatchAutoOpenPreview = (
  files: Array<string | any> | undefined,
  dispatch: React.Dispatch<any>,
) => {
  const previewFiles = normalizeGeneratedPreviewFiles(files)
  const autoOpenIndex = getAutoOpenGeneratedPreviewIndex(previewFiles)
  if (autoOpenIndex < 0) return

  const autoOpenFile = previewFiles[autoOpenIndex]
  dispatch({
    type: "OPEN_FILE_PREVIEW",
    payload: {
      fileId: autoOpenFile.fileId,
      fileName: autoOpenFile.fileName,
      files: previewFiles,
      index: autoOpenIndex,
    }
  })
}

const OPTIMISTIC_USER_MESSAGE_PREFIX = "msg-user-optimistic"
const USER_TURN_MESSAGE_PREFIX = "msg-user-turn"
const USER_EVENT_MESSAGE_PREFIX = "msg-user-event"
const USER_MESSAGE_REPLACE_WINDOW_MS = 30000

const userTurnMessageId = (turnId: string): string =>
  `${USER_TURN_MESSAGE_PREFIX}-${turnId}`

const stableUserMessageId = (
  eventData: { turn_id?: unknown } & Record<string, unknown>,
  eventId?: unknown,
): string | null => {
  const turnId = eventData.turn_id
  if (typeof turnId === "string" && turnId.trim()) {
    return userTurnMessageId(turnId.trim())
  }

  if (typeof eventId === "string" && eventId.trim()) {
    return `${USER_EVENT_MESSAGE_PREFIX}-${eventId.trim()}`
  }

  return null
}

const extractTextFromReactNode = (node: React.ReactNode): string => {
  if (typeof node === 'string') return node
  if (typeof node === 'number') return node.toString()
  if (Array.isArray(node)) return node.map(extractTextFromReactNode).join('')
  if (React.isValidElement(node) && node.props.children) {
    return extractTextFromReactNode(node.props.children)
  }
  return ''
}

const normalizeMessageContent = (content: string | React.ReactNode): string => {
  if (typeof content === 'string') {
    return content.trim()
  }
  if (typeof content === 'number') {
    return content.toString()
  }
  if (React.isValidElement(content) || Array.isArray(content)) {
    return extractTextFromReactNode(content).trim()
  }
  return ''
}

const findOptimisticUserMessageIndex = (
  messages: Message[],
  incomingMessage: Message,
): number => {
  if (incomingMessage.role !== "user") {
    return -1
  }

  const normalizedIncomingContent = normalizeMessageContent(incomingMessage.content)
  if (!normalizedIncomingContent) {
    return -1
  }

  const incomingTimestamp = normalizeTimestampMs(incomingMessage.timestamp)

  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const existingMessage = messages[index]
    if (
      existingMessage.role !== "user" ||
      typeof existingMessage.id !== "string" ||
      (
        !existingMessage.isOptimistic &&
        !existingMessage.id.startsWith(OPTIMISTIC_USER_MESSAGE_PREFIX)
      )
    ) {
      continue
    }

    if (normalizeMessageContent(existingMessage.content) !== normalizedIncomingContent) {
      continue
    }

    const existingTimestamp = normalizeTimestampMs(existingMessage.timestamp)
    if (
      Number.isFinite(existingTimestamp) &&
      Number.isFinite(incomingTimestamp) &&
      Math.abs(incomingTimestamp - existingTimestamp) > USER_MESSAGE_REPLACE_WINDOW_MS
    ) {
      continue
    }

    return index
  }

  return -1
}

// Expose to window for global access
if (typeof window !== 'undefined') {
  ; (window as any).clearDuplicateMessageCache = () => {
    duplicateMessageCacheClearers.forEach(clear => clear())
  }
}
const normalizeInteractions = (value: unknown): Interaction[] => {
  if (!Array.isArray(value)) {
    return []
  }

  const seenFields = new Set<string>()

  return value
    .map((item: any, index) => {
      if (!item || typeof item !== "object") {
        return null
      }

      const rawType = item.type
      const type =
        rawType === "input" || rawType === "text" || rawType === "textarea" || rawType === "string"
          ? "text_input"
          : rawType === "file" || rawType === "upload"
            ? "file_upload"
            : rawType === "number" || rawType === "integer"
              ? "number_input"
              : rawType === "boolean"
                ? "confirm"
                : rawType
      const rawField = item.field || item.id || item.name || item.properties?.field || item.properties?.id || `response_${index}`
      const baseField = typeof rawField === "string" && rawField.trim() ? rawField.trim() : `response_${index}`
      const field = seenFields.has(baseField) ? `${baseField}_${index}` : baseField
      seenFields.add(field)
      if (
        !["select_one", "select_multiple", "text_input", "file_upload", "confirm", "number_input", "action_cards"].includes(type) ||
        typeof field !== "string" ||
        !field.trim()
      ) {
        return null
      }

      const normalized: Interaction = {
        type,
        field,
        label: typeof item.label === "string" && item.label.trim() ? item.label : field,
      }

      const rawOptions = Array.isArray(item.options)
        ? item.options
        : Array.isArray(item.actions)
          ? item.actions
          : undefined

      if (Array.isArray(rawOptions)) {
        normalized.options = rawOptions
          .filter((opt: any) => opt && typeof opt.value === "string")
          .map((opt: any) => ({
            value: opt.value,
            label: typeof opt.label === "string" ? opt.label : opt.value,
            description: typeof opt.description === "string" ? opt.description : undefined,
            action_type: typeof opt.action_type === "string" ? opt.action_type : undefined,
          }))
      }

      if (typeof item.placeholder === "string") normalized.placeholder = item.placeholder
      if (typeof item.properties?.placeholder === "string") normalized.placeholder = item.properties.placeholder
      if (typeof item.multiline === "boolean") normalized.multiline = item.multiline
      if (typeof item.properties?.multiline === "boolean") normalized.multiline = item.properties.multiline
      if (typeof item.min === "number") normalized.min = item.min
      if (typeof item.max === "number") normalized.max = item.max
      if (typeof item.default !== "undefined") normalized.default = item.default
      if (Array.isArray(item.accept) || typeof item.accept === "string") normalized.accept = item.accept
      if (typeof item.multiple === "boolean") normalized.multiple = item.multiple

      return normalized
    })
    .filter(Boolean) as Interaction[]
}

const extractClarificationMessage = (raw: unknown): { message?: string; interactions: Interaction[] } | null => {
  const sharedResponse = extractSharedChatResponse(raw)
  const interactions = normalizeInteractions(sharedResponse?.interactions)

  if (interactions.length === 0) {
    return null
  }

  return {
    message: sharedResponse?.message,
    interactions,
  }
}


interface Message {
  id: string
  role: "user" | "assistant"
  content: string | React.ReactNode
  rawContent?: string
  timestamp: string
  status?: "pending" | "running" | "completed" | "failed"
  isResult?: boolean
  isFileOutput?: boolean
  streamMessageId?: string
  traceEvents?: TraceEvent[]
  interactions?: Interaction[]
  isSystemNotice?: boolean
  isOptimistic?: boolean
}

export type TaskRuntimeExtensions = Record<string, Record<string, unknown>>

const normalizeTaskRuntimeExtensions = (value: unknown): TaskRuntimeExtensions => {
  if (!isJsonRecord(value)) return {}
  return Object.fromEntries(
    Object.entries(value).filter((entry): entry is [string, Record<string, unknown>] => (
      isJsonRecord(entry[1])
    )),
  )
}

export interface Task {
  id: string
  title: string
  status: TaskStatus
  description: string
  createdAt: string | number
  updatedAt: string | number
  // Model configuration
  modelId?: string
  smallFastModelId?: string
  visualModelId?: string
  compactModelId?: string
  modelName?: string
  smallFastModelName?: string
  visualModelName?: string
  compactModelName?: string
  executionMode?: "auto" | "flash" | "balanced" | "think"
  isDag?: boolean
  agentId?: number
  agentName?: string
  agentLogoUrl?: string
  runtimeExtensionBindings?: string[]
  waitingQuestion?: string
  waitingInteractions?: Interaction[]
  runId?: string | null
  stateVersion?: number
  controlState?: TaskControlState
}

type SessionTaskBinding =
  | { present: false }
  | { present: true; valid: false }
  | { present: true; valid: true; taskId: number }

const hasOwn = (
  record: Record<string, unknown>,
  key: string,
): boolean => Object.prototype.hasOwnProperty.call(record, key)

const extractSessionTaskBinding = (
  message: WebSocketMessage,
): SessionTaskBinding => {
  const root = asMessageRecord(message)
  const data = asMessageRecord(root.data)
  const nestedData = asMessageRecord(data.data)
  const task = asMessageRecord(root.task)
  const dataTask = asMessageRecord(data.task)
  const nestedTask = asMessageRecord(nestedData.task)
  const candidates: unknown[] = []

  const collect = (record: Record<string, unknown>, key: string) => {
    if (hasOwn(record, key) && record[key] !== undefined) {
      candidates.push(record[key])
    }
  }
  collect(root, "task_id")
  collect(task, "id")
  collect(data, "task_id")
  collect(dataTask, "id")
  collect(nestedData, "task_id")
  collect(nestedTask, "id")

  if (candidates.length === 0) return { present: false }
  const parsed = candidates.map(parseInteger)
  if (parsed.some(taskId => taskId === undefined || taskId <= 0)) {
    return { present: true, valid: false }
  }
  const taskIds = new Set(parsed as number[])
  if (taskIds.size !== 1) return { present: true, valid: false }
  return { present: true, valid: true, taskId: parsed[0] as number }
}

const getSessionTaskInfoData = (
  message: WebSocketMessage,
): Record<string, unknown> | null => {
  if (message.type !== "trace_event") return null
  const data = asMessageRecord(message.data)
  const eventType = message.event_type ?? data.event_type
  if (eventType !== "task_info") return null
  const nestedData = asMessageRecord(data.data)
  return Object.keys(nestedData).length > 0 ? nestedData : data
}

interface StepExecution {
  id: string
  name: string
  description: string
  status: "pending" | "running" | "completed" | "failed" | "skipped"
  tool_names?: string[]
  dependencies: string[]
  started_at?: string | number
  completed_at?: string | number
  result_data?: unknown
  step_data?: unknown
  file_outputs?: string[]
  conditional_branches?: Record<string, string>
  required_branch?: string | null
  is_conditional?: boolean
}

interface TraceEvent {
  event_id: string
  event_type: string
  step_id?: string
  timestamp: string
  data: unknown
}

const normalizeStepStatus = (status: unknown): StepExecution["status"] => {
  return status === "running" || status === "completed" || status === "failed" || status === "skipped"
    ? status
    : "pending"
}

const getString = (value: unknown, fallback = ""): string => typeof value === "string" ? value : fallback
const getStringArray = (value: unknown): string[] => Array.isArray(value) ? value.map(item => String(item)) : []

const taskFromTaskInfoData = (
  taskData: Record<string, unknown>,
  taskId: number,
): Task => ({
  id: String(taskId),
  title: taskData.title as string,
  description: taskData.description as string,
  status: normalizeTaskStatus(taskData.status) || "pending",
  createdAt: taskData.created_at as string | number,
  updatedAt: taskData.updated_at as string | number,
  modelId: taskData.model_id as string | undefined,
  smallFastModelId: taskData.small_fast_model_id as string | undefined,
  visualModelId: taskData.visual_model_id as string | undefined,
  compactModelId: taskData.compact_model_id as string | undefined,
  modelName: taskData.model_name as string | undefined,
  smallFastModelName: taskData.small_fast_model_name as string | undefined,
  visualModelName: taskData.visual_model_name as string | undefined,
  compactModelName: taskData.compact_model_name as string | undefined,
  executionMode: taskData.execution_mode as Task["executionMode"],
  isDag: taskData.is_dag as boolean | undefined,
  agentId: taskData.agent_id as number | undefined,
  agentName: taskData.agent_name as string | undefined,
  agentLogoUrl: taskData.agent_logo_url as string | undefined,
  runtimeExtensionBindings: getStringArray(taskData.runtime_extension_bindings),
  waitingQuestion: taskData.waiting_question as string | undefined,
  waitingInteractions: normalizeInteractions(taskData.waiting_interactions),
  runId: taskData.run_id as string | null | undefined,
  stateVersion: parseInteger(taskData.state_version),
  controlState: taskData.control_state as TaskControlState | undefined,
})

const getWebSocketErrorMessage = (message: WebSocketMessage): string => {
  const root = message as unknown as Record<string, unknown>
  const data = isJsonRecord(message.data) ? message.data : null
  return getString(data?.message) || getString(data?.error) || getString(root.message) || getString(root.error) || "Unknown error"
}

const getWebSocketTaskStatus = (message: WebSocketMessage): Task["status"] | null => {
  const root = message as unknown as Record<string, unknown>
  const data = isJsonRecord(message.data) ? message.data : null
  const rootTask = isJsonRecord(root.task) ? root.task : null
  const dataTask = isJsonRecord(data?.task) ? data.task : null
  return normalizeTaskStatus(dataTask?.status) || normalizeTaskStatus(rootTask?.status) || normalizeTaskStatus(data?.status) || normalizeTaskStatus(root.status) || null
}

const shouldStopProcessingForTaskStatus = (status: unknown): boolean =>
  isStoppedTaskStatus(status)

const stepsFromPlanData = (planData: unknown, existingSteps: StepExecution[]): StepExecution[] | null => {
  const planRecord = planData && typeof planData === "object" ? planData as Record<string, unknown> : null
  const planSteps = Array.isArray(planRecord?.steps) ? planRecord.steps : null
  if (!planSteps) {
    return null
  }

  const existingStepsMap = new Map<string, StepExecution>()
  existingSteps.forEach(step => existingStepsMap.set(step.id, step))

  return planSteps.map((rawStep: unknown) => {
    const step = rawStep && typeof rawStep === "object" ? rawStep as Record<string, unknown> : {}
    const id = String(step.id)
    const existingStep = existingStepsMap.get(id)
    const task = getString(step.task)
    const name = getString(step.name, task || id)
    return {
      id,
      name,
      description: getString(step.description, task),
      status: existingStep?.status || normalizeStepStatus(step.status),
      tool_names: step.tool_name ? [String(step.tool_name)] : getStringArray(step.tool_names),
      dependencies: getStringArray(step.dependencies),
      started_at: existingStep?.started_at || getString(step.started_at),
      completed_at: existingStep?.completed_at || getString(step.completed_at),
      result_data: step.result_data ?? step.result,
      step_data: step.step_data,
      file_outputs: getStringArray(step.file_outputs),
      conditional_branches: step.conditional_branches && typeof step.conditional_branches === "object" ? step.conditional_branches as Record<string, string> : {},
      required_branch: typeof step.required_branch === "string" ? step.required_branch : null,
      is_conditional: step.is_conditional === true,
    }
  })
}

interface DAGExecution {
  phase: "planning" | "executing" | "completed" | "failed"
  current_plan: Record<string, unknown>
  created_at: string | number
  updated_at: string | number
}

interface AppState {
  messages: Message[]
  currentTask: Task | null
  taskRuntimeExtensions: TaskRuntimeExtensions
  dagExecution: DAGExecution | null
  steps: StepExecution[]
  traceEvents: TraceEvent[]
  selectedStepId: string | null
  isProcessing: boolean
  taskId: number | null
  filePreview: {
    isOpen: boolean
    fileId: string
    fileName: string
    content: string
    mimeType?: string
    isLoading: boolean
    error: string | null
    // Support switching between multiple file previews
    availableFiles: Array<{ fileId: string; fileName: string }>
    currentIndex: number
    viewMode: 'preview' | 'code'
  }
  isReplaying: boolean
  replaySpeed: number
  replayProgress: number
  replayEvents: TraceEvent[]
  replayTaskId: number | null
  replayScheduler: ReplayScheduler | null
  replayEventCache: WebSocketMessage[]
  planMemoryInfo: {
    memoriesFound: number
    memoriesUsed: number
    memoryCategory: string
    enhancedGoal?: string
    memories?: Array<{
      content: string
      category?: string
    }>
  } | null
  lastTaskUpdate?: number
  isHistoryLoading: boolean
  // Current context-window usage from the latest LLM call, for the usage gauge.
  contextUsage: { tokens: number; threshold: number } | null
  sessionConversation: SessionConversationState
}

type AppAction =
  | { type: "SESSION_CONVERSATION"; payload: SessionConversationAction }
  | { type: "SET_TASK_ID"; payload: number | null }
  | { type: "ADOPT_SESSION_TASK"; payload: { taskId: number; task: Task } }
  | { type: "RESET_SESSION_CONVERSATION" }
  | { type: "ADD_MESSAGE"; payload: Message }
  | { type: "UPSERT_STREAMING_FINAL_ANSWER"; payload: { messageId: string; delta?: string; content?: string; status?: Message["status"]; timestamp: string } }
  | { type: "SET_CURRENT_TASK"; payload: Task | null }
  | { type: "SET_TASK_RUNTIME_EXTENSIONS"; payload: { taskId: number; extensions: TaskRuntimeExtensions } }
  | { type: "UPDATE_TASK_STATUS"; payload: { status: Task["status"]; waitingQuestion?: string; waitingInteractions?: Interaction[]; runId?: string | null; stateVersion?: number; controlState?: TaskControlState } }
  | { type: "TRIGGER_TASK_UPDATE" }
  | { type: "SET_DAG_EXECUTION"; payload: DAGExecution | null }
  | { type: "SET_CONTEXT_USAGE"; payload: { tokens: number; threshold: number } | null }
  | { type: "ADD_STEP"; payload: StepExecution }
  | { type: "UPDATE_STEP"; payload: { stepId: string; updates: Partial<StepExecution> } }
  | { type: "SET_STEPS"; payload: StepExecution[] }
  | { type: "ADD_TRACE_EVENT"; payload: TraceEvent }
  | { type: "SET_TRACE_EVENTS"; payload: TraceEvent[] }
  | { type: "SELECT_STEP"; payload: string | null }
  | { type: "SET_PROCESSING"; payload: boolean }
  | {
    type: "CLEAR_MESSAGES";
    payload?: {
      preserveUserMessages?: boolean;
      preserveStreamingFinalAnswers?: boolean;
    };
  }
  | { type: "RESET_STATE" }
  | { type: "OPEN_FILE_PREVIEW"; payload: { fileId: string; fileName: string; files?: Array<{ fileId: string; fileName: string }>; index?: number } }
  | { type: "CLOSE_FILE_PREVIEW" }
  | { type: "SWITCH_FILE_PREVIEW"; payload: { fileId: string; fileName: string; index: number } }
  | { type: "SET_FILE_PREVIEW_CONTENT"; payload: { content: string; mimeType?: string; error: string | null } }
  | { type: "SET_FILE_PREVIEW_LOADING"; payload: boolean }
  | { type: "SET_FILE_PREVIEW_MODE"; payload: 'preview' | 'code' }
  | { type: "START_REPLAY"; payload: { taskId: number; events: TraceEvent[] } }
  | { type: "STOP_REPLAY" }
  | { type: "SET_PLAN_MEMORY_INFO"; payload: AppState["planMemoryInfo"] }
  | { type: "SET_REPLAY_TASK_ID"; payload: number | null }
  | { type: "SET_REPLAY_PLAYING"; payload: boolean }
  | { type: "SET_REPLAY_SPEED"; payload: number }
  | { type: "SET_REPLAY_PROGRESS"; payload: number }
  | { type: "SET_REPLAY_EVENTS"; payload: TraceEvent[] }
  | { type: "SET_REPLAY_SCHEDULER"; payload: ReplayScheduler | null }
  | { type: "ADD_TO_REPLAY_CACHE"; payload: WebSocketMessage }
  | { type: "CLEAR_REPLAY_CACHE" }
  | { type: "SET_HISTORY_LOADING"; payload: boolean }
  | { type: "SYNC_PROCESSING_STATUS" }

const createInitialState = (): AppState => ({
  messages: [],
  currentTask: null,
  taskRuntimeExtensions: {},
  dagExecution: null,
  steps: [],
  traceEvents: [],
  selectedStepId: null,
  isProcessing: false,
  taskId: null,
  filePreview: {
    isOpen: false,
    fileId: '',
    fileName: '',
    content: '',
    isLoading: false,
    error: null,
    availableFiles: [],
    currentIndex: 0,
    viewMode: 'preview',
  },
  isReplaying: false,
  replaySpeed: 1.0,
  replayProgress: 0, // 0-100
  replayEvents: [],
  replayTaskId: null,
  replayScheduler: null,
  replayEventCache: [],
  planMemoryInfo: null,
  lastTaskUpdate: Date.now(),
  isHistoryLoading: false,
  contextUsage: null,
  sessionConversation: { ...initialSessionConversationState },
})

function projectAppState(state: AppState, action: AppAction): AppState {
  console.log('🔍 Reducer called with action:', action.type, action)

  switch (action.type) {
    case "SESSION_CONVERSATION":
      return {
        ...state,
        sessionConversation: reduceSessionConversation(
          state.sessionConversation,
          action.payload,
        ),
      }
    case "SET_HISTORY_LOADING":
      return { ...state, isHistoryLoading: action.payload }

    case "SYNC_PROCESSING_STATUS":
      if (shouldStopProcessingForTaskStatus(state.currentTask?.status)) {
        return { ...state, isProcessing: false }
      }
      return state

    case "TRIGGER_TASK_UPDATE":
      return { ...state, lastTaskUpdate: Date.now() }

    case "SET_TASK_ID":
      console.log('🔄 Reducer SET_TASK_ID:', {
        currentTaskId: state.taskId,
        newTaskId: action.payload,
        payloadType: typeof action.payload
      })
      // Clear messages if task ID changes
      const taskChanged = state.taskId !== action.payload
      const messages = taskChanged ? [] : state.messages
      const contextUsage = taskChanged ? null : state.contextUsage
      const taskRuntimeExtensions = taskChanged ? {} : state.taskRuntimeExtensions
      const newState = {
        ...state,
        taskId: action.payload,
        messages,
        contextUsage,
        taskRuntimeExtensions,
      }
      console.log('🔄 Reducer returning new state:', newState)
      return newState

    case "ADOPT_SESSION_TASK":
      return {
        ...state,
        taskId: action.payload.taskId,
        currentTask: action.payload.task,
        taskRuntimeExtensions: {},
      }

    case "RESET_SESSION_CONVERSATION":
      return {
        ...createInitialState(),
        lastTaskUpdate: Date.now(),
        sessionConversation: state.sessionConversation,
      }

    case "ADD_MESSAGE": {
      const newMessage = action.payload
      let messageToAdd = newMessage
      let newTraceEvents = state.traceEvents

      if (newMessage.role === "assistant" && newMessage.isResult) {
        messageToAdd = {
          ...newMessage,
          traceEvents: mergeTraceEventsById(newMessage.traceEvents, state.traceEvents)
        }
        newTraceEvents = []
      }

      if (newMessage.role === "assistant" && newMessage.isResult) {
        const replaceMessageAt = (targetIndex: number) => {
          const updatedMessages = state.messages.map((message, index) =>
            index === targetIndex
              ? {
                ...message,
                ...messageToAdd,
                id: message.id,
                status: newMessage.status || "completed",
                traceEvents: mergeTraceEventsById(message.traceEvents, messageToAdd.traceEvents),
              }
              : message
          )
          return { ...state, messages: updatedMessages, traceEvents: newTraceEvents }
        }
        if (newMessage.streamMessageId) {
          const streamingIndex = state.messages.findIndex(
            message =>
              message.id === newMessage.streamMessageId &&
              isStreamingFinalAnswerMessage(message)
          )
          if (streamingIndex >= 0) {
            return replaceMessageAt(streamingIndex)
          }
        }
      }

      const optimisticUserMessageIndex = findOptimisticUserMessageIndex(
        state.messages,
        messageToAdd,
      )

      if (messageToAdd.role === "user") {
        const existingUserMessageIndex = state.messages.findIndex(
          message => message.role === "user" && message.id === messageToAdd.id,
        )
        if (existingUserMessageIndex >= 0) {
          const existingMessage = state.messages[existingUserMessageIndex]
          if (messageToAdd.isOptimistic && !existingMessage.isOptimistic) {
            return state
          }

          const updatedMessages = state.messages.map((message, index) =>
            index === existingUserMessageIndex
              ? {
                ...message,
                ...messageToAdd,
                isOptimistic: messageToAdd.isOptimistic ?? false,
              }
              : message
          )
          updatedMessages.sort((a, b) => {
            return normalizeTimestampMs(a.timestamp) - normalizeTimestampMs(b.timestamp)
          })
          return { ...state, messages: updatedMessages, traceEvents: newTraceEvents }
        }
      }

      if (optimisticUserMessageIndex >= 0) {
        const updatedMessages = state.messages.map((message, index) =>
          index === optimisticUserMessageIndex
            ? {
              ...message,
              ...messageToAdd,
              id: messageToAdd.id,
              isOptimistic: messageToAdd.isOptimistic ?? false,
            }
            : message
        )
        updatedMessages.sort((a, b) => {
          return normalizeTimestampMs(a.timestamp) - normalizeTimestampMs(b.timestamp)
        })
        return { ...state, messages: updatedMessages, traceEvents: newTraceEvents }
      }

      const updatedMessages = [...state.messages, messageToAdd]
      updatedMessages.sort((a, b) => {
        return normalizeTimestampMs(a.timestamp) - normalizeTimestampMs(b.timestamp)
      })
      return { ...state, messages: updatedMessages, traceEvents: newTraceEvents }
    }

    case "UPSERT_STREAMING_FINAL_ANSWER": {
      const { messageId, delta, content, status, timestamp } = action.payload
      const existing = state.messages.find(message => message.id === messageId)
      if (!existing) {
        const message: Message = {
          id: messageId,
          role: "assistant",
          content: content || delta || "",
          rawContent: content || delta || "",
          timestamp,
          status: status || "running",
          isResult: true,
          traceEvents: [...state.traceEvents],
        }
        return { ...state, messages: [...state.messages, message], traceEvents: [] }
      }
      const updatedMessages = state.messages.map(message => {
        if (message.id !== messageId) {
          return message
        }
        const currentContent =
          typeof message.content === "string" ? message.content : ""
        const nextContent = content !== undefined ? content : currentContent + (delta || "")
        return {
          ...message,
          content: nextContent,
          rawContent: nextContent,
          status: status || message.status || "running",
        }
      })
      return { ...state, messages: updatedMessages }
    }

    case "SET_CURRENT_TASK": {
      const incomingTask = action.payload
        ? {
          ...action.payload,
          status:
            normalizeTaskStatus(action.payload.status) ||
            (state.currentTask?.id === action.payload.id
              ? state.currentTask.status
              : "pending"),
        }
        : null

      const currentTask = (state.currentTask && incomingTask && state.currentTask.id === incomingTask.id)
        ? { ...state.currentTask, ...incomingTask, agentName: incomingTask.agentName || state.currentTask.agentName, agentLogoUrl: incomingTask.agentLogoUrl || state.currentTask.agentLogoUrl }
        : incomingTask

      return {
        ...state,
        currentTask,
        isProcessing: currentTask && shouldStopProcessingForTaskStatus(currentTask.status)
          ? false
          : state.isProcessing,
      }
    }

    case "SET_TASK_RUNTIME_EXTENSIONS":
      if (state.taskId !== action.payload.taskId) return state
      return {
        ...state,
        taskRuntimeExtensions: action.payload.extensions,
      }

    case "UPDATE_TASK_STATUS": {
      if (!state.currentTask) {
        return state
      }

      const nextStatus = normalizeTaskStatus(action.payload.status)
      if (!nextStatus) {
        return state
      }

      const isWaitingForUser = nextStatus === "waiting_for_user"
      return {
        ...state,
        isProcessing: shouldStopProcessingForTaskStatus(nextStatus)
          ? false
          : state.isProcessing,
        currentTask: {
          ...state.currentTask,
          status: nextStatus,
          updatedAt: new Date().toISOString(),
          waitingQuestion: isWaitingForUser
            ? action.payload.waitingQuestion ?? state.currentTask.waitingQuestion
            : undefined,
          waitingInteractions: isWaitingForUser
            ? action.payload.waitingInteractions ?? state.currentTask.waitingInteractions
            : undefined,
          runId: action.payload.runId ?? state.currentTask.runId,
          stateVersion: action.payload.stateVersion ?? state.currentTask.stateVersion,
          controlState: action.payload.controlState ?? state.currentTask.controlState,
        },
      }
    }

    case "SET_DAG_EXECUTION":
      return { ...state, dagExecution: action.payload }

    case "SET_CONTEXT_USAGE":
      return { ...state, contextUsage: action.payload }

    case "ADD_STEP":
      const newStep = action.payload
      const existingStepIndex = state.steps.findIndex(s => s.id === newStep.id)
      if (existingStepIndex >= 0) {
        // Update existing step - merge data intelligently to preserve existing information
        const existingStep = state.steps[existingStepIndex]
        const shouldUpdate = newStep.name !== existingStep.name ||
          newStep.description !== existingStep.description ||
          !arraysEqual(newStep.tool_names || [], existingStep.tool_names || []) ||
          newStep.status !== existingStep.status

        if (shouldUpdate) {
          const mergedStep = {
            ...existingStep,
            ...newStep,
            // Preserve existing started_at if new one is not provided
            started_at: newStep.started_at || existingStep.started_at,
            // Preserve existing tool_names if new one is not provided
            tool_names: newStep.tool_names || existingStep.tool_names,
            // Preserve existing description if new one is not provided
            description: newStep.description || existingStep.description,
            // Preserve dependencies if new step doesn't have them
            dependencies: newStep.dependencies && newStep.dependencies.length > 0 ? newStep.dependencies : existingStep.dependencies || [],
            // Preserve conditional branch fields if new step doesn't have them
            conditional_branches: newStep.conditional_branches && Object.keys(newStep.conditional_branches).length > 0 ? newStep.conditional_branches : existingStep.conditional_branches || {},
            required_branch: newStep.required_branch ?? existingStep.required_branch ?? null,
            is_conditional: newStep.is_conditional ?? existingStep.is_conditional ?? false,
          }
          return {
            ...state,
            steps: state.steps.map((step, index) =>
              index === existingStepIndex ? mergedStep : step
            )
          }
        } else {
          return state // No update needed
        }
      } else {
        // Add new step
        return { ...state, steps: [...state.steps, action.payload] }
      }

    case "UPDATE_STEP":
      return {
        ...state,
        steps: state.steps.map(step =>
          step.id === action.payload.stepId
            ? { ...step, ...action.payload.updates }
            : step
        ),
      }

    case "SET_STEPS":
      return { ...state, steps: action.payload }

    case "ADD_TRACE_EVENT":
      // If the last message is a result message from assistant, append the trace event to that message directly.
      // This ensures that events arriving after the result message (like react_task_end) are correctly displayed.
      const lastMsg = state.messages.length > 0 ? state.messages[state.messages.length - 1] : null
      if (lastMsg && lastMsg.role === "assistant" && lastMsg.isResult) {
        const updatedLastMsg = {
          ...lastMsg,
          traceEvents: [...(lastMsg.traceEvents || []), action.payload]
        }
        return {
          ...state,
          messages: [...state.messages.slice(0, -1), updatedLastMsg]
        }
      }
      return { ...state, traceEvents: [...state.traceEvents, action.payload] }

    case "SET_TRACE_EVENTS":
      return { ...state, traceEvents: action.payload }

    case "SELECT_STEP":
      return { ...state, selectedStepId: action.payload }

    case "SET_PROCESSING":
      return { ...state, isProcessing: action.payload }

    case "CLEAR_MESSAGES":
      if (action.payload) {
        const {
          preserveUserMessages,
          preserveStreamingFinalAnswers,
        } = action.payload
        const messagesToKeep = state.messages.filter(message => {
          if (preserveUserMessages && message.role === "user") {
            return true
          }
          return (
            preserveStreamingFinalAnswers &&
            isStreamingFinalAnswerMessage(message)
          )
        })
        return { ...state, messages: messagesToKeep }
      }
      return { ...state, messages: [] }

    case "RESET_STATE":
      return {
        ...createInitialState(),
        sessionConversation: state.sessionConversation,
      }

    case "OPEN_FILE_PREVIEW":
      // Support passing single file or multiple file list
      const files = action.payload.files || [{ fileId: action.payload.fileId, fileName: action.payload.fileName }]
      const currentIndex = action.payload.index || 0

      return {
        ...state,
        filePreview: {
          ...state.filePreview,
          isOpen: true,
          fileId: files[currentIndex]?.fileId || action.payload.fileId,
          fileName: files[currentIndex]?.fileName || action.payload.fileName,
          content: '',
          isLoading: true,
          error: null,
          availableFiles: files,
          currentIndex: currentIndex,
          viewMode: 'preview',
        }
      }

    case "CLOSE_FILE_PREVIEW":
      return {
        ...state,
        filePreview: {
          ...state.filePreview,
          isOpen: false,
          isLoading: false,
        }
      }

    case "SWITCH_FILE_PREVIEW":
      return {
        ...state,
        filePreview: {
          ...state.filePreview,
          fileId: action.payload.fileId,
          fileName: action.payload.fileName,
          content: '',
          isLoading: true,
          error: null,
          currentIndex: action.payload.index,
          viewMode: 'preview',
        }
      }

    case "SET_FILE_PREVIEW_MODE":
      return {
        ...state,
        filePreview: {
          ...state.filePreview,
          viewMode: action.payload,
        }
      }

    case "SET_FILE_PREVIEW_CONTENT":
      return {
        ...state,
        filePreview: {
          ...state.filePreview,
          content: action.payload.content,
          mimeType: action.payload.mimeType,
          error: action.payload.error,
          isLoading: false,
        }
      }

    case "SET_FILE_PREVIEW_LOADING":
      return {
        ...state,
        filePreview: {
          ...state.filePreview,
          isLoading: action.payload,
        }
      }

    case "START_REPLAY":
      return {
        ...state,
        isReplaying: true, // We start replaying immediately
        replayEvents: action.payload.events,
        replayTaskId: action.payload.taskId,
        replayProgress: 0,
        replaySpeed: state.replaySpeed,
        replayScheduler: null, // Will be initialized when actually starting playback
      }

    case "STOP_REPLAY":
      // Clean up scheduler if it exists
      if (state.replayScheduler) {
        state.replayScheduler.stop()
      }
      return {
        ...state,
        isReplaying: false,
        replayEvents: [],
        replayTaskId: null,
        replayProgress: 0,
        replayScheduler: null,
        replayEventCache: [], // Also clear the event cache
      }

    case "SET_REPLAY_TASK_ID":
      return {
        ...state,
        replayTaskId: action.payload,
      }

    case "SET_REPLAY_PLAYING":
      if (action.payload && state.replayScheduler) {
        // Start playing
        state.replayScheduler.play()
      } else if (!action.payload && state.replayScheduler) {
        // Pause playing
        state.replayScheduler.pause()
      }
      return {
        ...state,
        isReplaying: action.payload,
      }

    case "SET_REPLAY_SPEED":
      if (state.replayScheduler) {
        state.replayScheduler.setPlaybackSpeed(action.payload)
      }
      return {
        ...state,
        replaySpeed: action.payload,
      }

    case "SET_REPLAY_PROGRESS":
      return {
        ...state,
        replayProgress: action.payload,
      }

    case "SET_REPLAY_EVENTS":
      return {
        ...state,
        replayEvents: action.payload,
      }

    case "SET_REPLAY_SCHEDULER":
      return {
        ...state,
        replayScheduler: action.payload,
      }

    case "ADD_TO_REPLAY_CACHE":
      return {
        ...state,
        replayEventCache: [...state.replayEventCache, action.payload],
      }

    case "CLEAR_REPLAY_CACHE":
      return {
        ...state,
        replayEventCache: [],
      }

    case "SET_PLAN_MEMORY_INFO":
      return {
        ...state,
        planMemoryInfo: action.payload,
      }

    default:
      return state
  }
}

type AppStateCommit = { state: AppState }

const commitProjectedAppState = (
  _: AppState,
  commit: AppStateCommit,
): AppState => commit.state

interface PendingMessage {
  message: string
  files?: File[]
  targetTaskId?: number
  force?: boolean
  clientMessageId?: string
  resolve?: () => void
  reject?: (error: Error) => void
}

interface AppContextType {
  state: AppState
  dispatch: React.Dispatch<AppAction>
  filesDisabled: boolean
  agentCardsEnabled: boolean
  voiceInputEnabled: boolean
  taskControlsEnabled: boolean
  sendMessage: (message: string, config?: any, files?: File[]) => Promise<void>
  executeTask: (description: string) => void
  pauseTask: () => void
  resumeTask: () => void
  selectStep: (stepId: string | null) => void
  clearMessages: () => void
  isConnected: boolean
  connectionError: Error | null
  startNewConversation: () => Promise<void>
  isConversationResetPending: boolean
  isMessageDeliveryPending: boolean
  isSessionInteractionLocked: boolean
  sessionConversationState: SessionConversationState["phase"]
  setTaskId: (taskId: number | null, options?: { navigate?: boolean }) => void
  requestStatus: () => void
  getFilePreviewUrl: (fileId: string) => string
  getFileDownloadUrl: (fileId: string) => string
  openFilePreview: (fileId: string, fileName: string, files?: Array<{ fileId: string; fileName: string }>, index?: number) => void
  switchFilePreview: (index: number) => void
  closeFilePreview: () => void
  startReplay: (taskId: number, events: TraceEvent[]) => void
  stopReplay: () => void
  setReplayPlaying: (isPlaying: boolean) => void
  setReplaySpeed: (speed: number) => void
  setReplayProgress: (progress: number) => void
  setPendingMessage: React.Dispatch<React.SetStateAction<PendingMessage | null>>
}

const AppContext = createContext<AppContextType | undefined>(undefined)

type TransportCapabilityState = "enabled" | "disabled"

export interface AppProviderTransportCapabilities {
  files?: TransportCapabilityState
  agentCards?: TransportCapabilityState
  voice?: TransportCapabilityState
  taskControls?: TransportCapabilityState
  // Unlike the capabilities above, this defaults closed for every transport
  // (including "page"): only the embedded Chat Widget opts in, since that is
  // the only surface where an in-tab navigation abandons the visitor's iframe.
  linksOpenInNewTab?: TransportCapabilityState
}

export interface AppProviderTransportConfig {
  buildWebSocketUrl?: (params: { baseUrl: string; taskId: number; token?: string }) => string
  /**
   * Owns every file URL and request made below this provider. Public
   * transports use this to keep guest credentials instance-scoped.
   */
  fileAccess?: FileAccessPolicy
  uploadFiles?: (files: File[], params: { taskId?: number | null; taskType: string }) => Promise<Array<{ file_id: string; name?: string; size?: number; type?: string }>>
  capabilities?: AppProviderTransportCapabilities
  session?: {
    connection: WebSocketConnection | null
    onConnectionClose: (
      event: CloseEvent,
      connectionIdentity?: string,
    ) => "handled"
    onConnectionFailure: (
      failure: WebSocketConnectionFailure,
      connectionIdentity?: string,
    ) => void
    onConnectionOpen?: (connectionIdentity: string) => void
    allowTasklessChat: true
    supportsConversationReset: true
    history: "none"
    files: "disabled"
    agentCards: "disabled"
    voice: "disabled"
    taskControls: "disabled"
  }
}

interface SessionResetFlight {
  connectionIdentity: string
  deliveryGeneration: number
  promise: Promise<void>
  resolve: () => void
  reject: (error: Error) => void
  timeout: ReturnType<typeof setTimeout>
}

interface SessionMessageOwner {
  connectionIdentity: string
}

interface SessionPreAdoptionBuffer {
  connectionIdentity: string
  candidate: { message: WebSocketMessage; taskId: number } | null
  frames: WebSocketMessage[]
  bytes: number
  timeout: ReturnType<typeof setTimeout> | null
}

const serializedWebSocketMessageBytes = (message: WebSocketMessage): number =>
  new TextEncoder().encode(JSON.stringify(message)).byteLength

function resolveTransportCapability(
  sessionTransport: AppProviderTransportConfig["session"],
  sessionState: TransportCapabilityState | undefined,
  transportState: TransportCapabilityState | undefined,
): boolean {
  // Session transports are external credential domains and therefore default
  // closed when a required runtime descriptor is malformed. Other transports
  // preserve the existing enabled behavior unless they opt out explicitly.
  return (
    sessionTransport
      ? sessionState ?? "disabled"
      : transportState ?? "enabled"
  ) === "enabled"
}

export function AppProvider({
  children,
  token,
  transport,
}: {
  children: React.ReactNode
  token?: string
  transport?: AppProviderTransportConfig
}) {
  const [state, privateCommit] = useReducer(
    commitProjectedAppState,
    undefined,
    createInitialState,
  )
  const pendingTaskToExecuteRef = useRef<{ description: string } | null>(null)
  const startDelayedPlaybackRef = useRef<() => void>(() => {})
  const isHistoricalDataLoadingRef = useRef(false)
  const historicalDataRequestMapRef = useRef(new Map<number, boolean>())
  const recentMessagesRef = useRef(new Set<string>())
  const isDuplicateMessage = useCallback((
    content: string | React.ReactNode,
    type = "general",
    force = false,
    shouldCache = true,
  ) => {
    const contentStr = normalizeMessageContent(content)
    const key = `${type}:${contentStr}`
    const cache = recentMessagesRef.current
    if (!force && cache.has(key)) return true
    if (shouldCache) {
      cache.add(key)
      setTimeout(() => cache.delete(key), 30_000)
    }
    return false
  }, [])
  const isDuplicateResult = useCallback(
    (content: string) => isDuplicateMessage(content, "result"),
    [isDuplicateMessage],
  )
  // All actions are projected synchronously so each action observes the state
  // produced by the preceding action before React commits the batch.
  const stateRef = useRef(state)
  const dispatch = useCallback((action: AppAction) => {
    const next = projectAppState(stateRef.current, action)
    stateRef.current = next
    privateCommit({ state: next })
  }, [])
  const sessionConversationRef = useRef<SessionConversationState>(initialSessionConversationState)
  const dispatchSessionConversation = useCallback(
    (action: SessionConversationAction) => {
      const current = sessionConversationRef.current
      const transition = transitionSessionConversation(current, action)
      if (transition.changed) {
        sessionConversationRef.current = transition.next
        dispatch({ type: "SESSION_CONVERSATION", payload: action })
      }
      return transition
    },
    [dispatch],
  )
  useLayoutEffect(() => {
    sessionConversationRef.current = state.sessionConversation
  }, [state.sessionConversation])
  const [pendingMessage, setPendingMessage] = useState<PendingMessage | null>(null)
  const pendingMessageRef = useRef(pendingMessage)
  pendingMessageRef.current = pendingMessage
  useEffect(() => {
    const clear = () => recentMessagesRef.current.clear()
    duplicateMessageCacheClearers.add(clear)
    return () => {
      duplicateMessageCacheClearers.delete(clear)
    }
  }, [])
  const sessionTransport = transport?.session
  const filesDisabled = !resolveTransportCapability(
    sessionTransport,
    sessionTransport?.files,
    transport?.capabilities?.files,
  )
  const rejectDisabledFileUpload = useCallback(async () => {
    throw new Error("Files are disabled for this conversation.")
  }, [])
  const requestedAgentCardsEnabled = resolveTransportCapability(
    sessionTransport,
    sessionTransport?.agentCards,
    transport?.capabilities?.agentCards,
  )
  const agentCardsEnabled = resolveAgentCardPresentationCapability(
    filesDisabled,
    requestedAgentCardsEnabled,
  )
  // Deliberately not resolveTransportCapability: that helper defaults every
  // other capability to "enabled" for non-session transports, which is the
  // wrong default here — this must stay off everywhere except the transports
  // that explicitly request it.
  const linksOpenInNewTab = transport?.capabilities?.linksOpenInNewTab === "enabled"
  const voiceInputEnabled = resolveTransportCapability(
    sessionTransport,
    sessionTransport?.voice,
    transport?.capabilities?.voice,
  )
  const taskControlsEnabled = resolveTransportCapability(
    sessionTransport,
    sessionTransport?.taskControls,
    transport?.capabilities?.taskControls,
  )
  const sessionConnectionIdentity =
    sessionTransport?.connection?.identity ?? null
  const [deliveryGeneration, setDeliveryGeneration] = useState(0)
  const deliveryGenerationRef = useRef(deliveryGeneration)
  deliveryGenerationRef.current = deliveryGeneration
  const [messageDeliveryCount, setMessageDeliveryCount] = useState(0)
  const messageDeliveryCountRef = useRef(0)
  const sessionConnectionIdentityRef = useRef(sessionConnectionIdentity)
  sessionConnectionIdentityRef.current = sessionConnectionIdentity
  const previousSessionConnectionIdentityRef = useRef(
    sessionConnectionIdentity
  )
  const sessionResetFlightRef = useRef<SessionResetFlight | null>(null)
  const sessionPreAdoptionBufferRef = useRef<SessionPreAdoptionBuffer | null>(null)
  const flushingSessionPreAdoptionCandidateRef = useRef<WebSocketMessage | null>(null)
  const retiredSessionTaskIdsRef = useRef(new Set<number>())
  const sessionMessageHandlerRef = useRef<
    (message: WebSocketMessage, owner: SessionMessageOwner) => void
  >(() => {})
  const mountedRef = useRef(false)
  const { t } = useI18n()
  const router = useRouter()
  const lastConnectedTaskId = useRef<number | null>(null)
  const taskStateVersionsRef = useRef(
    new Map<number, TaskStateVersionEntry>()
  )

  // Session task ownership is established only by a current-socket task_info;
  // never import a legacy route/storage Task id into this protocol state.
  const sessionTaskIdRef = useRef<number | null>(null)

  const rejectSessionResetFlight = useCallback(
    (
      resetFlight: SessionResetFlight | null,
      error: Error,
    ): boolean => {
      if (
        !resetFlight
        || sessionResetFlightRef.current !== resetFlight
      ) {
        return false
      }

      sessionResetFlightRef.current = null
      clearTimeout(resetFlight.timeout)
      resetFlight.reject(error)
      return true
    },
    [],
  )

  const discardSessionPreAdoptionBuffer = useCallback(() => {
    const buffer = sessionPreAdoptionBufferRef.current
    if (!buffer) return
    if (buffer.timeout !== null) clearTimeout(buffer.timeout)
    sessionPreAdoptionBufferRef.current = null
  }, [])

  const requireSessionReload = useCallback((error: Error) => {
    discardSessionPreAdoptionBuffer()
    rejectSessionResetFlight(sessionResetFlightRef.current, error)
    dispatchSessionConversation({
      type: "SESSION_RELOAD_REQUIRED",
      connectionIdentity: sessionConnectionIdentityRef.current,
    })
  }, [discardSessionPreAdoptionBuffer, dispatchSessionConversation, rejectSessionResetFlight])

  const beginSessionPreAdoptionBuffer = useCallback((connectionIdentity: string) => {
    discardSessionPreAdoptionBuffer()
    sessionPreAdoptionBufferRef.current = {
      connectionIdentity,
      candidate: null,
      frames: [],
      bytes: 0,
      timeout: null,
    }
  }, [discardSessionPreAdoptionBuffer])

  const activateSessionPreAdoptionBuffer = useCallback((connectionIdentity: string) => {
    const buffer = sessionPreAdoptionBufferRef.current
    if (!buffer || buffer.connectionIdentity !== connectionIdentity) {
      requireSessionReload(
        new Error("Replacement conversation transaction is missing; reload required.")
      )
      return
    }
    if (buffer.timeout !== null) return
    buffer.timeout = setTimeout(() => {
      const buffer = sessionPreAdoptionBufferRef.current
      if (buffer?.connectionIdentity !== connectionIdentity) return
      requireSessionReload(
        new Error("Replacement conversation did not publish task_info before its deadline; reload required.")
      )
    }, SESSION_TASK_ADOPTION_TIMEOUT_MS)
    if (buffer.candidate) {
      flushingSessionPreAdoptionCandidateRef.current = buffer.candidate.message
      try {
        sessionMessageHandlerRef.current(buffer.candidate.message, { connectionIdentity })
      } finally {
        flushingSessionPreAdoptionCandidateRef.current = null
      }
    }
  }, [requireSessionReload])

  const bufferSessionPreAdoptionFrame = useCallback((
    message: WebSocketMessage,
    owner: SessionMessageOwner,
    candidateTaskId?: number,
  ): boolean => {
    const buffer = sessionPreAdoptionBufferRef.current
    if (!buffer || buffer.connectionIdentity !== owner.connectionIdentity) {
      requireSessionReload(
        new Error("Replacement frame ownership is unknown; reload required.")
      )
      return false
    }
    const bytes = serializedWebSocketMessageBytes(message)
    if (candidateTaskId !== undefined && buffer.candidate) {
      if (buffer.candidate.taskId === candidateTaskId) return true
      requireSessionReload(
        new Error("Replacement conversation published conflicting task_info; reload required.")
      )
      return false
    }
    if (
      buffer.frames.length + (buffer.candidate ? 1 : 0) >= MAX_SESSION_PRE_ADOPTION_FRAMES
      || buffer.bytes + bytes > MAX_SESSION_PRE_ADOPTION_BYTES
    ) {
      requireSessionReload(
        new Error("Replacement frame buffer exceeded its safe bound; reload required.")
      )
      return false
    }
    if (candidateTaskId !== undefined) {
      buffer.candidate = { message, taskId: candidateTaskId }
    } else {
      buffer.frames.push(message)
    }
    buffer.bytes += bytes
    return true
  }, [requireSessionReload])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      messageDeliveryCountRef.current = 0
      discardSessionPreAdoptionBuffer()
      rejectSessionResetFlight(
        sessionResetFlightRef.current,
        new Error("Conversation reset was cancelled because chat was closed.")
      )
    }
  }, [discardSessionPreAdoptionBuffer, rejectSessionResetFlight])

  useEffect(() => {
    const taskId = state.taskId
    const hasRuntimeExtensionBinding = (
      state.currentTask?.id === String(taskId)
      && (state.currentTask.runtimeExtensionBindings?.length || 0) > 0
    )
    if (taskId === null || sessionTransport || !hasRuntimeExtensionBinding) return

    let cancelled = false
    const loadRuntimeExtensions = async () => {
      try {
        const response = await apiRequest(
          `${getApiUrl()}/api/chat/task/${taskId}/runtime-extensions`,
          { cache: "no-store" },
        )
        if (!response || !response.ok) return
        const payload = await response.json()
        if (cancelled) return
        dispatch({
          type: "SET_TASK_RUNTIME_EXTENSIONS",
          payload: {
            taskId,
            extensions: normalizeTaskRuntimeExtensions(payload?.runtime_extensions),
          },
        })
      } catch (error) {
        // Runtime metadata only decorates the transcript. A temporary metadata
        // failure must not block loading or running the task itself.
        console.warn("Failed to load task runtime metadata", error)
      }
    }
    void loadRuntimeExtensions()

    return () => {
      cancelled = true
    }
  }, [
    dispatch,
    sessionTransport,
    state.currentTask?.id,
    state.currentTask?.runtimeExtensionBindings,
    state.taskId,
  ])

  useEffect(() => {
    const previousIdentity = previousSessionConnectionIdentityRef.current
    previousSessionConnectionIdentityRef.current = sessionConnectionIdentity
    if (previousIdentity === sessionConnectionIdentity) return

    if (sessionConversationRef.current.phase === "bound") {
      if (sessionConnectionIdentity) {
        dispatchSessionConversation({
          type: "SESSION_BOUND_CONNECTION_REBOUND",
          connectionIdentity: sessionConnectionIdentity,
        })
      }
      return
    }
    if (sessionConversationRef.current.phase !== "unbound") {
      requireSessionReload(
        new Error("Conversation outcome is unknown after a connection refresh; reload required.")
      )
    }
  }, [dispatchSessionConversation, requireSessionReload, sessionConnectionIdentity])

  const onConnect = useCallback(() => {
    if (sessionTransport?.history === "none") {
      const connectionIdentity = sessionConnectionIdentityRef.current
      if (connectionIdentity && connectionIdentity === sessionConnectionIdentity) {
        sessionTransport.onConnectionOpen?.(connectionIdentity)
      }
      isHistoricalDataLoadingRef.current = false
      dispatch({ type: "SET_HISTORY_LOADING", payload: false })
      return
    }

    // Fix: If we should be in replay mode but got disconnected, restore replay state
    if (stateRef.current.replayTaskId && stateRef.current.taskId === stateRef.current.replayTaskId && !stateRef.current.isReplaying) {
      dispatch({ type: "SET_REPLAY_PLAYING", payload: true })
    }

    // Handle clearing messages on reconnection
    // This prevents stale data issues and fixes race conditions
    if (lastConnectedTaskId.current === stateRef.current.taskId) {
      // Reconnection to SAME task -> Clear messages
      dispatch({
        type: "CLEAR_MESSAGES",
        payload: {
          preserveUserMessages: true,
          preserveStreamingFinalAnswers: true,
        },
      })
      dispatch({ type: "SET_TRACE_EVENTS", payload: [] })
      dispatch({ type: "SET_STEPS", payload: [] })
      // Clear dagExecution alongside steps - otherwise the Progress panel
      // stays open (it only needs dagExecution to be non-null) rendering an
      // empty step list for the gap between reconnecting and history replay
      // repopulating both together.
      dispatch({ type: "SET_DAG_EXECUTION", payload: null })
    } else {
      // New task connection -> Update tracker, Don't clear (handled by setTaskId)
      lastConnectedTaskId.current = stateRef.current.taskId
    }

    // Set history loading state
    isHistoricalDataLoadingRef.current = true
    dispatch({ type: "SET_HISTORY_LOADING", payload: true })

    // Safety timeout: if no history arrives within 2 seconds, assume empty or done
    setTimeout(() => {
      dispatch({ type: "SET_HISTORY_LOADING", payload: false })
    }, 2000)

    // Auto-execute PENDING tasks from Agent Builder
    setTimeout(() => {
      if (pendingTaskToExecuteRef.current) {
        const hasUserMessages = stateRef.current.messages.some(m => m.role === 'user')
        console.log('🔍 onConnect - checking auto-execute:', {
          hasPendingTask: !!pendingTaskToExecuteRef.current,
          pendingDescription: pendingTaskToExecuteRef.current.description,
          hasUserMessages,
        })

        if (!hasUserMessages) {
          console.log('🚀 Auto-executing PENDING task from Agent Builder (onConnect):', pendingTaskToExecuteRef.current.description)
          // sendChatMessage(pendingTaskToExecute.description, []) // Cannot access sendChatMessage
          pendingTaskToExecuteRef.current = null
        } else {
          console.log('⏭️ Skipping auto-execute, already has user messages')
          pendingTaskToExecuteRef.current = null
        }
      }
    }, 1000)
  }, [
    sessionConnectionIdentity,
    sessionTransport?.history,
    sessionTransport?.onConnectionOpen,
  ])

  const sessionMessageOwner = sessionConnectionIdentity
    ? {
      connectionIdentity: sessionConnectionIdentity,
    }
    : null

  const {
    isConnected,
    connectionError,
    sendMessage: sendRawMessage,
    sendChatMessage,
    executeTask: wsExecuteTask,
    pauseTask: wsPauseTask,
    resumeTask: wsResumeTask,
    requestStatus,
    connect,
  } = useWebSocket({
    taskId: state.taskId || undefined,
    token,
    buildWebSocketUrl: transport?.buildWebSocketUrl,
    uploadFiles: filesDisabled
      ? rejectDisabledFileUpload
      : transport?.uploadFiles,
    connection:
      sessionTransport === undefined
        ? undefined
        : sessionTransport.connection,
    deliveryGeneration:
      sessionTransport === undefined
        ? undefined
        : deliveryGeneration,
    onSessionConnectionClose: sessionTransport?.onConnectionClose,
    onSessionConnectionFailure: sessionTransport?.onConnectionFailure,
    onMessage: (message) => {
      if (sessionTransport) {
        if (
          !mountedRef.current
          || !sessionMessageOwner
          || sessionMessageOwner.connectionIdentity
            !== sessionConnectionIdentityRef.current
        ) {
          return
        }
        sessionMessageHandlerRef.current(message, sessionMessageOwner)
        return
      }
      handleMessage(message, dispatch, stateRef.current, { filesDisabled })
    },
    onConnect: onConnect, // Pass the callback
    autoConnect: true,
  })

  useEffect(() => {
    if (!sessionTransport || isConnected) return
    if (
      sessionConversationRef.current.phase !== "reset_requested"
      && sessionConversationRef.current.phase !== "replacement_ready"
      && sessionConversationRef.current.phase !== "replacement_sending"
      && sessionConversationRef.current.phase !== "replacement_awaiting_task"
    ) {
      return
    }
    requireSessionReload(
      new Error("Conversation outcome is unknown because the Session disconnected; reload required.")
    )
  }, [isConnected, requireSessionReload, sessionTransport])

  // Handle pending messages separately since we need sendChatMessage
  useEffect(() => {
    if (isConnected && pendingMessage) {
      // Ensure we are sending to the correct task
      // If targetTaskId is set, it must match the current connected task
      if (pendingMessage.targetTaskId) {
        // We use lastConnectedTaskId.current because state.taskId might be updated before the socket is connected
        // But sendChatMessage sends to the currently connected socket.
        // We need to make sure the CURRENT socket corresponds to the targetTaskId.
        // lastConnectedTaskId is updated in onConnect, so it reflects the current socket's task ID.
        if (lastConnectedTaskId.current !== pendingMessage.targetTaskId) {
          console.log('⏳ Pending message target task mismatch, waiting...', {
            target: pendingMessage.targetTaskId,
            current: lastConnectedTaskId.current
          })
          return
        }
      }

      console.log('📤 Sending pending message:', {
        message: pendingMessage.message,
        hasFiles: pendingMessage.files && pendingMessage.files.length > 0,
        targetTaskId: pendingMessage.targetTaskId
      })
      void Promise.resolve(
        sendChatMessage(
          pendingMessage.message,
          pendingMessage.files,
          pendingMessage.force,
          pendingMessage.clientMessageId,
        )
      ).then(() => {
        pendingMessage.resolve?.()
        setPendingMessage(current => current === pendingMessage ? null : current)
      }).catch((error) => {
        const deliveryError = error instanceof Error ? error : new Error(String(error))
        pendingMessage.reject?.(deliveryError)
        setPendingMessage(current => current === pendingMessage ? null : current)
      })
    }
  }, [isConnected, pendingMessage, sendChatMessage])

  // A queued message can only ever flush to its target task. If the task is
  // switched away before that socket connects (e.g. the widget's "New
  // conversation" reset), fail it immediately — left queued it would sit out
  // the 30s timeout and vanish as an unhandled rejection with no visible error.
  useEffect(() => {
    if (!pendingMessage?.targetTaskId) return
    if (state.taskId === pendingMessage.targetTaskId) return
    pendingMessage.reject?.(
      new Error('Message not sent: the conversation was reset before it could be delivered.')
    )
    setPendingMessage(current => (current === pendingMessage ? null : current))
  }, [pendingMessage, state.taskId])

  const queuePendingMessage = useCallback((message: Omit<PendingMessage, 'resolve' | 'reject'>) => {
    return new Promise<void>((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        setPendingMessage(current => (
          current?.clientMessageId === message.clientMessageId ? null : current
        ))
        reject(new Error('Message not sent: timed out waiting for the task connection.'))
      }, 30000)
      setPendingMessage({
        ...message,
        resolve: () => {
          window.clearTimeout(timeout)
          resolve()
        },
        reject: (error) => {
          window.clearTimeout(timeout)
          reject(error)
        },
      })
    })
  }, [])

  // Handle auto-execute pending task separately
  useEffect(() => {
    if (isConnected && pendingTaskToExecuteRef.current) {
      // Logic moved to effect
      // But wait, pendingTaskToExecute is not state, it's a let variable.
      // Effect won't run when it changes.
      // But it runs when isConnected changes.

      const timer = setTimeout(() => {
        if (pendingTaskToExecuteRef.current) {
          const hasUserMessages = stateRef.current.messages.some(m => m.role === 'user')
          if (!hasUserMessages) {
            sendChatMessage(pendingTaskToExecuteRef.current.description, [])
            pendingTaskToExecuteRef.current = null
          }
        }
      }, 1000)
      return () => clearTimeout(timer)
    }
  }, [isConnected, sendChatMessage])

  // Debug: Log when taskId is passed to useWebSocket
  useEffect(() => {
    console.log('🔧 useWebSocket taskId prop:', {
      taskId: state.taskId,
      taskIdType: typeof state.taskId
    })
  }, [state.taskId])

  // Track connection state changes
  useEffect(() => {
    console.log('🔄 AppContext - WebSocket connection state changed:', {
      isConnected,
      taskId: state.taskId,
      hasConnectionError: !!connectionError,
      connectionErrorMessage: connectionError?.message,
      timestamp: new Date().toISOString()
    })
  }, [isConnected, state.taskId, connectionError])

  const handleMessage = useCallback((
    message: WebSocketMessage,
    rawDispatch: React.Dispatch<AppAction>,
    currentState: AppState,
    options?: {
      skipHistory?: boolean
      filesDisabled?: boolean
    },
  ) => {
    // A task's WebSocket connection is per-task (or, for session transports,
    // multiplexes a sequence of tasks over one socket) - either way, a
    // just-superseded socket/task can still have events in flight after the
    // app has moved on to a different task (e.g. the user starts a new chat
    // while a previous DAG run is still executing in the background). Those
    // stray events must not repaint TASK_SCOPED_ACTION_TYPES's state for
    // whatever task is now on screen - not just DAG/step state, but the chat
    // transcript, trace log, context/memory display, and processing/status
    // flags too, since a background task's reply or tool trace has no
    // business appearing in a different task's conversation. Deliberately
    // keyed ONLY off currentState.taskId (not currentTask.id): switching
    // tasks via setTaskId leaves a real window where taskId has already
    // advanced to the new task but currentTask still describes the old one
    // (until that task's own task_info arrives) - matching against
    // currentTask.id in that window would let the old task's stray events
    // right back through, defeating the whole guard.
    const messageTaskId = (message as unknown as { task_id?: unknown }).task_id
    // No task is even being viewed (currentState.taskId null/undefined) but
    // this message names a specific task - that can only be a stray event
    // for a task other than "the current view", so it counts as "for another
    // task" too (not just the "viewing a *different* task" case).
    const isMessageForOtherTask =
      messageTaskId !== undefined && messageTaskId !== null && messageTaskId !== ""
      && String(messageTaskId) !== String(currentState.taskId)
    const dispatch: React.Dispatch<AppAction> = (action) => {
      if (TASK_SCOPED_ACTION_TYPES.has(action.type) && isMessageForOtherTask) return
      rawDispatch(action)
    }
    // The 30s dedup cache below is keyed on message content/type only, not
    // task id - several dedupKeys (e.g. dag-execute-end's "task end, this
    // iteration") are templated purely on generic fields like iteration
    // count, so a background task and the currently-viewed task can produce
    // the exact same key. A stale event for another task never reaches
    // `dispatch` (filtered above), but if it were still allowed to *insert*
    // into the cache, it would silently swallow the viewed task's own
    // legitimate ADD_MESSAGE as a "duplicate" moments later. Skip caching
    // (not skip the check itself) whenever the message belongs elsewhere.
    const isDuplicateMessageForViewedTask = (
      content: string | React.ReactNode,
      type = "general",
      force = false,
      shouldCache = true,
    ) => isDuplicateMessage(content, type, force, shouldCache && !isMessageForOtherTask)
    const isDuplicateResultForViewedTask = (content: string) =>
      isDuplicateMessage(content, "result", false, !isMessageForOtherTask)
    // If we're in replay mode, don't process immediately - collect for delayed playback
    if (
      !options?.skipHistory
      && shouldBufferMessageForHistoricalReplay({
        isReplaying: currentState.isReplaying,
        isHistoryLoading: currentState.isHistoryLoading || isHistoricalDataLoadingRef.current,
        message,
      })
    ) {
      // Add to replay cache
      dispatch({ type: "ADD_TO_REPLAY_CACHE", payload: message })

      // If this is historical_data_complete, start the delayed playback
      const isHistoricalComplete =
        getWebSocketEventType(message) === "historical_data_complete"

      if (isHistoricalComplete) {
        isHistoricalDataLoadingRef.current = false
        dispatch({ type: "SET_HISTORY_LOADING", payload: false })
        dispatch({ type: "SYNC_PROCESSING_STATUS" })
        // Add a small delay to ensure all events are collected before starting playback
        setTimeout(() => {
          startDelayedPlaybackRef.current()
        }, 500) // 500ms delay to collect remaining events
      }

      return
    }

    const controlEnvelope = extractTaskControlEnvelope(message)
    if (controlEnvelope.isStateEvent && controlEnvelope.taskId !== undefined) {
      if (
        !acceptTaskControlVersion(
          message,
          controlEnvelope,
          taskStateVersionsRef.current,
        )
      ) return

      // A late event may have an old semantic type (for example
      // ``task_paused``) after a newer run is already RUNNING. The backend
      // rewrites its state tuple to the canonical row; apply only that tuple
      // and skip the stale event-specific side effects.
      if (!taskEventMatchesControlState(message, controlEnvelope)) {
        if (controlEnvelope.status) {
          dispatch({
            type: "UPDATE_TASK_STATUS",
            payload: {
              status: controlEnvelope.status,
              runId: controlEnvelope.runId,
              stateVersion: controlEnvelope.stateVersion,
              controlState: controlEnvelope.controlState,
            },
          })
        }
        return
      }
    }

    // Normal message processing when not in replay mode
    if (isFinalAnswerStreamEventType(message.type)) {
      const payload = getFinalAnswerStreamActionPayload({
        eventType: message.type,
        eventData: message,
        eventId: message.event_id,
        timestamp: message.timestamp,
        fallbackMessageId: generateMessageId("msg-final-answer"),
      })
      if (payload) {
        dispatch({ type: "UPSERT_STREAMING_FINAL_ANSWER", payload })
      }
      return
    }

    switch (message.type) {
      case "chat":
        const chatData = message as any
        const messageContent = chatData.message || ""

        if (!isDuplicateMessageForViewedTask(messageContent, 'user-message')) {
          dispatch({
            type: "ADD_MESSAGE",
            payload: {
              id: generateMessageId("msg-user"),
              role: "user",
              content: messageContent,
              timestamp: message.timestamp?.toString() || Date.now().toString(),
            }
          })
        }
        break

      case "trace_event":
        const traceEventData = (message.data ?? {}) as any

        // Check if this has the expected structure with event_type
        // event_type can be in message.event_type (new format) or traceEventData.event_type (old format)
        const eventType = message.event_type || traceEventData.event_type

        if (eventType) {
          // eventData should be the data field from traceEventData, but also include top-level fields
          const eventData = {
            ...(traceEventData.data || traceEventData || {}),
            step_id: message.step_id || traceEventData.step_id || (traceEventData.data || {}).step_id,
            task_id: message.task_id || traceEventData.task_id || (traceEventData.data || {}).task_id,
          }
          const isDelegatedChildEvent =
            eventData.source === "xagent-agent-tool-child"
          const addDelegatedChildTraceEvent = () => {
            dispatch({
              type: "ADD_TRACE_EVENT",
              payload: {
                event_id: message.event_id || eventData.event_id || generateMessageId("trace-agent-child"),
                event_type: eventType,
                step_id: message.step_id || eventData.step_id,
                timestamp: message.timestamp?.toString() || Date.now().toString(),
                data: eventData,
              }
            })
          }

          // Handle structured trace events
          if (eventType === "task_info") {
            const taskData = eventData as Record<string, unknown>
            const taskId = parseInteger(taskData.id)
            if (taskId === undefined || taskId <= 0) return
            const task = taskFromTaskInfoData(taskData, taskId)
            const taskStatus = task.status
            console.log('📥 Received task_info event:', {
              taskData,
              status: taskData.status,
              statusType: typeof taskData.status
            })

            // Store pending task for auto-execution
            if (taskStatus === 'pending' && task.description) {
              pendingTaskToExecuteRef.current = { description: task.description }
              console.log('💾 Stored pending task for auto-execution:', taskData.description)
            }

            // Check if status changed and trigger update if so
            if (currentState.currentTask?.id === task.id && currentState.currentTask?.status !== taskStatus) {
              dispatch({ type: "TRIGGER_TASK_UPDATE" })
            }

            dispatch({
              type: "SET_CURRENT_TASK",
              payload: task,
            })
            // Check if this is a new task (created within last 5 seconds)
            // If so, we don't expect historical messages, so stop loading
            // We do NOT stop loading here for new tasks anymore.
            // We wait for the user_message event or the timeout to handle it.
            // This prevents the empty state flash when task_info arrives before user_message.
          } else if (eventType === "dag_execution") {
            dispatch({ type: "SET_HISTORY_LOADING", payload: false })
            const steps = stepsFromPlanData(eventData, currentState.steps)
            if (steps) {
              dispatch({ type: "SET_STEPS", payload: steps })
            }
            // The backend's dag_execution payload (dag.py's on_dag_execution)
            // never actually carries a created_at - only completed_step_count/
            // plan_step_count/steps. Without a fallback here, the DAG run
            // never gets a stable "started at" timestamp, so the Progress
            // panel's total elapsed time never renders. Keep whatever
            // created_at this run already established (planning's first event
            // sets it; later phase updates for the same run must not reset
            // it), falling back to this event's own timestamp only the first
            // time.
            const dagCreatedAt =
              currentState.dagExecution?.created_at
              ?? (eventData as { created_at?: string | number }).created_at
              ?? message.timestamp
            dispatch({
              type: "SET_DAG_EXECUTION",
              payload: { ...eventData, created_at: dagCreatedAt } as DAGExecution,
            })
          } else if (eventType === "dag_step_info") {
            dispatch({ type: "SET_HISTORY_LOADING", payload: false })
            const stepInfo = eventData
            const step: StepExecution = {
              id: stepInfo.id,
              name: stepInfo.name || stepInfo.id,
              description: stepInfo.description || "",
              status: stepInfo.status,
              tool_names: stepInfo.tool_name ? [stepInfo.tool_name] : stepInfo.tool_names || [],
              dependencies: stepInfo.dependencies || [],
              started_at: stepInfo.started_at,
              completed_at: stepInfo.completed_at,
              result_data: stepInfo.result_data,
              step_data: stepInfo.step_data,
              file_outputs: stepInfo.file_outputs || [],
              conditional_branches: stepInfo.conditional_branches || {},
              required_branch: stepInfo.required_branch || null,
              is_conditional: stepInfo.is_conditional || false,
            }
            dispatch({ type: "ADD_STEP", payload: step })
          }

          // User Message Events
          else if (eventType === "user_message") {
            // A delegated Agent's task prompt is execution detail, not another
            // user turn in the parent Workforce conversation. Keep it in the
            // trace store so the Agent inspector can still render it live.
            if (isDelegatedChildEvent) {
              addDelegatedChildTraceEvent()
              return
            }
            dispatch({ type: "SET_HISTORY_LOADING", payload: false })
            const messageContent = eventData.message || eventData.content || ""
            const userMessageId = stableUserMessageId(
              eventData,
              message.event_id || traceEventData.event_id,
            )

            // Debug log
            console.log('🔍 User message debug:', {
              eventData,
              messageContent,
              hasMessage: !!eventData.message,
              hasContent: !!eventData.content,
              eventType,
              fullEvent: message,
              messageId: message.event_id,
              timestamp: message.timestamp
            })

            // Modern user-message events carry a stable turn_id (or at least
            // an event_id). The reducer reconciles those identities across
            // live delivery, optimistic rendering, and history replay. Only
            // legacy events without either identity fall back to short-lived
            // content-based deduplication.
            const isDuplicate = userMessageId === null
              ? isDuplicateMessageForViewedTask(messageContent, 'user-message', false, true)
              : false
            console.log('🔍 Duplicate check:', {
              messageContent,
              isDuplicate,
              recentMessages: Array.from(recentMessagesRef.current)
            })

            if (isDuplicate) {
              console.log('⚠️ User message filtered as duplicate:', messageContent)
              return
            }

            // Extract files from context.state.file_info (based on the actual WS event structure)
            let files = eventData.files || []
            if (eventData.context && eventData.context.state && eventData.context.state.file_info) {
              files = eventData.context.state.file_info
            }

            console.log('📁 Files extracted:', files)
            console.log('🔍 Context structure:', eventData.context)
            console.log('🔍 State structure:', eventData.context?.state)

            // Create message content with file attachments
            let content: React.ReactNode = messageContent

            if (files.length > 0) {
              content = (
                <UserMessageContent
                  message={messageContent}
                  files={files}
                  onPreview={options?.filesDisabled ? undefined : (file, previewFiles) => {
                    const currentFileId = file.file_id || ""
                    const normalizedFiles = previewFiles.map((previewFile) => ({
                      fileId: previewFile.file_id || "",
                      fileName: previewFile.name,
                    })).filter((item) => !!item.fileId)

                    if (!currentFileId) {
                      return
                    }
                    dispatch({
                      type: "OPEN_FILE_PREVIEW",
                      payload: {
                        fileId: currentFileId,
                        fileName: file.name,
                        files: normalizedFiles,
                        index: normalizedFiles.findIndex((previewFile) => previewFile.fileId === currentFileId)
                      }
                    })
                  }}
                />
              )
            }

            console.log('📤 Dispatching user message:', {
              content,
              filesCount: files.length,
              timestamp: message.timestamp,
              messageId: generateMessageId("msg-user")
            })

            const messagePayload = {
              id: userMessageId || generateMessageId("msg-user"),
              role: "user" as const,
              content: content,
              timestamp: message.timestamp,
              isOptimistic: false,
            }

            console.log('📤 Message payload:', messagePayload)

            dispatch({
              type: "ADD_MESSAGE",
              payload: messagePayload
            })

            console.log('✅ User message dispatched successfully')
          }

          else if (isFinalAnswerStreamEventType(eventType)) {
            const payload = getFinalAnswerStreamActionPayload({
              eventType,
              eventData,
              eventId: message.event_id,
              timestamp: message.timestamp,
              fallbackMessageId: generateMessageId("msg-final-answer"),
            })
            if (!payload) {
              return
            }
            dispatch({ type: "UPSERT_STREAMING_FINAL_ANSWER", payload })
          }

          // Agent progress messages belong in the execution timeline, not the chat transcript.
          else if (eventType === "agent_progress") {
            dispatch({
              type: "ADD_TRACE_EVENT",
              payload: {
                event_id: message.event_id || eventData.event_id || generateMessageId("trace-agent-progress"),
                event_type: eventType,
                step_id: message.step_id || eventData.step_id,
                timestamp: message.timestamp?.toString() || Date.now().toString(),
                data: eventData,
              }
            })
          }

          // Workforce delegation events stay in the manager timeline as compact
          // summaries. Their child traces are fetched only after the user opens
          // the agent execution drawer.
          else if (
            eventType === "workforce_delegation_start" ||
            eventType === "workforce_delegation_end" ||
            eventType === "workforce_delegation_error"
          ) {
            dispatch({
              type: "ADD_TRACE_EVENT",
              payload: {
                event_id: message.event_id || eventData.event_id || generateMessageId("trace-workforce-delegation"),
                event_type: eventType,
                step_id: message.step_id || eventData.step_id,
                timestamp: message.timestamp?.toString() || Date.now().toString(),
                data: eventData,
              }
            })
          }

          // Agent-to-user messages, including ask_user_question prompts.
          else if (eventType === "agent_message" || eventType === "ai_message") {
            // Child Agent output belongs exclusively to the on-demand Agent
            // inspector. Rendering it as parent chat content makes live state
            // disagree with historical replay and can expose child clarifiers
            // as if the Workforce manager asked them.
            if (isDelegatedChildEvent) {
              addDelegatedChildTraceEvent()
              return
            }
            const rawMessageContent = eventData.message || eventData.content || ""
            const messageContent =
              typeof rawMessageContent === "string"
                ? unwrapFinalAnswerContent(rawMessageContent)
                : rawMessageContent
            if (!messageContent) {
              return
            }
            const interactions = normalizeInteractions(eventData.metadata?.interactions)
            const isAgentMessage = eventType === "agent_message"
            const isAiMessage = eventType === "ai_message"
            const expectsUserResponse =
              isAgentMessage &&
              eventData.expect_response === true
            const agentMessageDisplay = eventData.display || eventData.metadata?.display
            const isExplicitTranscriptMessage =
              agentMessageDisplay === "chat" ||
              eventData.source === "chat_history" ||
              eventData.role === "assistant"
            const isTimelineAgentMessage =
              eventType === "agent_message" &&
              !isExplicitTranscriptMessage &&
              agentMessageDisplay === "timeline"
            if (isTimelineAgentMessage) {
              dispatch({
                type: "ADD_TRACE_EVENT",
                payload: {
                  event_id: message.event_id || eventData.event_id || generateMessageId("trace-agent-progress"),
                  event_type: "agent_progress",
                  step_id: message.step_id || eventData.step_id,
                  timestamp: message.timestamp?.toString() || Date.now().toString(),
                  data: eventData,
                }
              })
              return
            }
            const shouldHideAgentMessage =
              isAgentMessage &&
              eventData.visible === false
            if (expectsUserResponse) {
              dispatch({
                type: "UPDATE_TASK_STATUS",
                payload: {
                  status: "waiting_for_user",
                  waitingQuestion: messageContent,
                  waitingInteractions: interactions.length > 0 ? interactions : undefined,
                }
              })
            }
            const streamMessageId =
              isAiMessage
                ? getFinalAnswerStreamMessageId(eventData)
                : undefined
            if (shouldHideAgentMessage) {
              return
            }
            if (!streamMessageId && isDuplicateMessageForViewedTask(messageContent, 'agent-message')) {
              return
            }
            const msgId = generateMessageId("msg-agent")
            dispatch({
              type: "ADD_MESSAGE",
              payload: {
                id: msgId,
                role: "assistant",
                content: messageContent,
                rawContent: messageContent,
                timestamp: message.timestamp,
                status: eventData.status === "completed" ? "completed" : "running",
                isResult: true,
                streamMessageId,
                interactions: interactions.length > 0 ? interactions : undefined,
              }
            })
            if (eventData.status === "completed") {
              dispatch({ type: "UPDATE_TASK_STATUS", payload: { status: "completed" } })
              dispatch({ type: "SET_PROCESSING", payload: false })
            }
          }

          // DAG Plan Events
          else if (eventType === "dag_plan_start") {
            dispatch({ type: "SET_HISTORY_LOADING", payload: false })
            const phase = eventData.phase || "planning"
            const iteration = eventData.iteration || 1
            const content = (
              <>
                <FileText className="h-4 w-4 inline mr-2" />
                {t('agent.logs.event.messages.planStart', { phase })}
              </>
            )

            // Set DAG execution state to planning phase (only if not already executing or completed)
            if (!currentState.dagExecution || currentState.dagExecution.phase === "planning") {
              const dagExecution: DAGExecution = {
                phase: phase as "planning" | "executing" | "completed" | "failed",
                current_plan: {},
                created_at: message.timestamp,
                updated_at: message.timestamp,
              }

              // Use consistent string format for deduplication
              const dedupKey = `plan-start:${phase}`
              if (!isDuplicateMessageForViewedTask(dedupKey, 'dag-plan-start')) {
                dispatch({
                  type: "ADD_MESSAGE",
                  payload: {
                    id: generateMessageId("msg-plan-start"),
                    role: "assistant",
                    content,
                    timestamp: message.timestamp,
                    status: "completed",
                  }
                })

                // Set DAG execution state to show loading state
                dispatch({ type: "SET_DAG_EXECUTION", payload: dagExecution })
              }
            }
          } else if (eventType === "dag_plan_end") {
            const stepsCount = eventData.steps_count || 0
            const planId = eventData.plan_id || "unknown"
            const planData = eventData.plan_data || {}

            const content = (
              <>
                <CheckCircle className="h-4 w-4 inline mr-2 text-green-500" />
                {t('agent.logs.event.messages.planEnd', { planId, stepsCount })}
                {currentState.planMemoryInfo && (
                  <div className="mt-2">
                    <CollapsibleSection
                      title={t('agent.planDetails.memory.title')}
                      icon={<Brain className="h-4 w-4" />}
                      badge={t('agent.planDetails.badge.memory')}
                    >
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div className="flex items-center gap-1 p-2 bg-muted/30 rounded">
                          <Search className="h-3 w-3" />
                          <span>{t('agent.planDetails.memory.stats.found', { count: currentState.planMemoryInfo.memoriesFound })}</span>
                        </div>
                        <div className="flex items-center gap-1 p-2 bg-muted/30 rounded">
                          <Target className="h-3 w-3" />
                          <span>{t('agent.planDetails.memory.stats.used', { count: currentState.planMemoryInfo.memoriesUsed })}</span>
                        </div>
                      </div>
                      {currentState.planMemoryInfo.enhancedGoal && (
                        <div className="mt-2">
                          <div className="text-xs font-medium text-muted-foreground mb-1">{t('agent.planDetails.memory.enhancedGoalTitle')}</div>
                          <div className="text-xs bg-blue-500/10 p-2 rounded border border-blue-500/20">
                            {currentState.planMemoryInfo.enhancedGoal}
                          </div>
                        </div>
                      )}
                      {currentState.planMemoryInfo.memories && currentState.planMemoryInfo.memories.length > 0 && (
                        <div className="mt-2">
                          <div className="text-xs font-medium text-muted-foreground mb-1">{t('agent.planDetails.memory.relatedTitle')}</div>
                          <div className="space-y-1">
                            {currentState.planMemoryInfo.memories.map((memory, index) => (
                              <div
                                key={index}
                                className="text-xs p-2 bg-muted/20 rounded border border-border/50"
                              >
                                <div className="flex items-start gap-1">
                                  <Info className="h-3 w-3 mt-0.5 text-blue-400 flex-shrink-0" />
                                  <span className="whitespace-pre-wrap">{memory.content}</span>
                                </div>
                                {memory.category && (
                                  <Badge variant="outline" className="text-xs mt-1">
                                    {memory.category}
                                  </Badge>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </CollapsibleSection>
                  </div>
                )}
              </>
            )

            // Process step data in the plan, including dependencies
            const steps = stepsFromPlanData(planData, currentState.steps)
            if (steps) {
              dispatch({ type: "SET_STEPS", payload: steps })
            }

            const dedupKey = t('agent.logs.event.messages.planEnd', { planId, stepsCount })
            if (!isDuplicateMessageForViewedTask(dedupKey, 'plan-end')) {
              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId("msg-plan-end"),
                  role: "assistant",
                  content,
                  timestamp: message.timestamp,
                  status: "completed",
                }
              })

              // Update DAG execution state to executing phase (only if not already completed or failed)
              if (currentState.dagExecution && currentState.dagExecution.phase !== "completed" && currentState.dagExecution.phase !== "failed") {
                const updatedDAGExecution = {
                  ...currentState.dagExecution,
                  phase: "executing" as const,
                  current_plan: planData,
                  updated_at: message.timestamp,
                }
                dispatch({ type: "SET_DAG_EXECUTION", payload: updatedDAGExecution })
              }
            }
          }

          // DAG Execution Events
          else if (eventType === "dag_execute_start") {
            const iteration = eventData.iteration || 1
            const taskPreview = eventData.task_preview || t('agent.header.badge.task')

            // Set processing state to true when task execution starts
            dispatch({ type: "UPDATE_TASK_STATUS", payload: { status: "running" } })
            dispatch({ type: "SET_PROCESSING", payload: true })

            // Update DAG execution state to executing phase
            if (currentState.dagExecution) {
              const updatedDAGExecution = {
                ...currentState.dagExecution,
                phase: "executing" as const,
                updated_at: message.timestamp,
              }
              dispatch({ type: "SET_DAG_EXECUTION", payload: updatedDAGExecution })
            } else {
              const dagExecution: DAGExecution = {
                phase: "executing" as const,
                current_plan: {},
                created_at: message.timestamp,
                updated_at: message.timestamp,
              }
              dispatch({ type: "SET_DAG_EXECUTION", payload: dagExecution })
            }

            // Use consistent string format for deduplication
            const dedupKey = t('agent.logs.event.messages.taskStart', { iteration })
            if (!isDuplicateMessageForViewedTask(dedupKey, 'dag-execute-start')) {
              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId("msg-exec-start"),
                  role: "assistant",
                  content: (
                    <>
                      <Zap className="h-4 w-4 inline mr-2 text-yellow-500" />
                      {t('agent.logs.event.messages.taskStart', { iteration })}
                      <br />
                      <FileText className="h-4 w-4 inline mr-2 mt-1 text-cyan-500" />
                      {t('agent.logs.event.messages.taskDesc', { taskPreview })}
                    </>
                  ),
                  timestamp: message.timestamp,
                  status: "completed",
                }
              })
            }
          } else if (eventType === "dag_execute_end") {
            console.log("DEBUG: Received dag_execute_end event:", eventData)
            const iteration = eventData.iteration || 1
            const taskPreview = eventData.task_preview || t('agent.header.badge.task')
            console.log(`DEBUG: Processing dag_execute_end - GLOBAL iteration: ${iteration}, taskPreview: ${taskPreview}`)

            // Clear processing state when task completes
            dispatch({ type: "SET_PROCESSING", payload: false })

            // Update DAG execution state to completed phase
            if (currentState.dagExecution) {
              const updatedDAGExecution = {
                ...currentState.dagExecution,
                phase: "completed" as const,
                updated_at: message.timestamp,
              }
              dispatch({ type: "SET_DAG_EXECUTION", payload: updatedDAGExecution })
            }

            // Use consistent string format for deduplication
            const dedupKey = t('agent.logs.event.messages.taskEnd', { iteration })
            if (!isDuplicateMessageForViewedTask(dedupKey, 'dag-execute-end')) {
              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId("msg-exec-end"),
                  role: "assistant",
                  content: (
                    <>
                      <CheckCircle className="h-4 w-4 inline mr-2 text-green-500" />
                      {t('agent.logs.event.messages.taskEnd', { iteration })}
                      <br />
                      <FileText className="h-4 w-4 inline mr-2 mt-1 text-cyan-500" />
                      {t('agent.logs.event.messages.taskDesc', { taskPreview })}
                    </>
                  ),
                  timestamp: message.timestamp,
                  status: "completed",
                }
              })
            }
          }
          // Compact Events - All occur within a step, displayed in the corresponding step in the right panel
          else if (eventType === "action_start_compact") {
            const stepId = eventData.step_id
            if (stepId) {
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`compact-start-${stepId}`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.action_start_compact'),
                  message: t('agent.logs.event.messages.compactStart'),
                  compact_type: eventData.compact_type,
                  original_tokens: eventData.original_tokens,
                  threshold: eventData.threshold,
                  compact_model: eventData.compact_model,
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            }
          } else if (eventType === "action_end_compact") {
            const stepId = eventData.step_id
            if (stepId) {
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`compact-end-${stepId}`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.action_end_compact'),
                  message: t('agent.logs.event.messages.compactCompleted'),
                  compact_type: eventData.compact_type,
                  original_tokens: eventData.original_tokens,
                  compacted_tokens: eventData.compacted_tokens,
                  compression_ratio: eventData.compression_ratio,
                  compact_model: eventData.compact_model,
                  error: eventData.error,
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })

              // Surface a standalone system notice in the conversation so the
              // user sees that context was compacted without expanding the process.
              const noticeText = t('agent.logs.event.messages.compactNotice', {
                original: eventData.original_tokens ?? '?',
                compacted: eventData.compacted_tokens ?? '?',
              })
              // Key by step + event timestamp so distinct compactions (even in
              // the same step, or with identical token counts) each show, while
              // a re-dispatched same event is still deduped.
              const noticeKey = `compact-notice-${stepId}-${message.timestamp}`
              if (!isDuplicateMessageForViewedTask(noticeText, noticeKey)) {
                dispatch({
                  type: "ADD_MESSAGE",
                  payload: {
                    id: generateMessageId(noticeKey),
                    role: "assistant",
                    content: noticeText,
                    timestamp: message.timestamp,
                    status: "completed",
                    isSystemNotice: true,
                  }
                })
              }
            }
          }

          // DAG Step Events
          else if (eventType === "dag_step_start") {
            const stepName = eventData.step_name || eventData.name || eventData.title || `${t('agent.logs.event.messages.execStepPrefix')}${eventData.step_id || t('common.errors.unknown')}`

            // dag_step_start has step_id, should update the right-side step data, do not display message on the left
            // Find existing step first, preserve dependencies
            const existingStep = currentState.steps.find(s => s.id === (message.step_id || eventData.step_id || stepName))
            const step: StepExecution = {
              id: message.step_id || eventData.step_id || stepName,
              name: stepName,
              description: eventData.description || "",
              status: "running",
              tool_names: eventData.tool_name ? [eventData.tool_name] : eventData.tool_names || [],
              dependencies: eventData.dependencies || existingStep?.dependencies || [],
              started_at: eventData.started_at || message.timestamp,
              completed_at: eventData.completed_at,
              result_data: eventData.result_data,
              step_data: eventData.step_data,
              file_outputs: eventData.file_outputs || [],
              conditional_branches: eventData.conditional_branches || existingStep?.conditional_branches || {},
              required_branch: eventData.required_branch ?? existingStep?.required_branch ?? null,
              is_conditional: eventData.is_conditional ?? existingStep?.is_conditional ?? false,
            }
            dispatch({ type: "ADD_STEP", payload: step })

            // Also add to traceEvents for displaying execution logs
            const traceEvent: TraceEvent = {
              event_id: generateMessageId(`trace-step-start`),
              event_type: eventType,
              step_id: message.step_id || eventData.step_id || stepName,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.dag_step_start'),
                step_name: stepName,
                description: eventData.description,
                tool_names: eventData.tool_name ? [eventData.tool_name] : eventData.tool_names || [],
                started_at: eventData.started_at || message.timestamp,
              }
            }
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          } else if (eventType === "dag_step_end") {
            const stepName = eventData.step_name || eventData.name || eventData.title || `${t('agent.logs.event.messages.execStepPrefix')}${eventData.step_id || t('common.errors.unknown')}`
            console.log('✅ dag_step_end:', stepName, JSON.stringify(message))
            const resultData = eventData.result_data ?? eventData.result

            // dag_step_end has step_id, should update right-side step data, do not display message on the left
            const stepId = message.step_id || eventData.step_id || stepName
            const existingStep = currentState.steps.find(s => s.id === stepId)
            const step: StepExecution = {
              id: stepId,
              name: stepName,
              description: eventData.description || "",
              status: eventData.status || "completed",
              tool_names: eventData.tool_name ? [eventData.tool_name] : eventData.tool_names || [],
              dependencies: eventData.dependencies || existingStep?.dependencies || [],
              // Don't override started_at from end event to preserve the original start time
              started_at: undefined, // Let the reducer handle preserving existing started_at
              completed_at: eventData.completed_at || message.timestamp,
              result_data: resultData,
              step_data: eventData.step_data,
              file_outputs: eventData.file_outputs || [],
              conditional_branches: eventData.conditional_branches || {},
              required_branch: eventData.required_branch || null,
              is_conditional: eventData.is_conditional || false,
            }
            dispatch({ type: "ADD_STEP", payload: step })

            // Also add to traceEvents for displaying execution logs
            const traceEvent: TraceEvent = {
              event_id: generateMessageId(`trace-step-end`),
              event_type: eventType,
              step_id: message.step_id || eventData.step_id || stepName,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.dag_step_end'),
                step_name: stepName,
                description: eventData.description,
                tool_names: eventData.tool_name ? [eventData.tool_name] : eventData.tool_names || [],
                completed_at: eventData.completed_at || message.timestamp,
                result: resultData,
                result_data: resultData,
                step_data: eventData.step_data,
                file_outputs: eventData.file_outputs || [],
              }
            }
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          } else if (eventType === "dag_step_failed") {
            const stepName = eventData.step_name || eventData.name || eventData.title || `${t('agent.logs.event.messages.execStepPrefix')}${eventData.step_id || t('common.errors.unknown')}`
            const stepId = message.step_id || eventData.step_id || stepName
            const existingStep = currentState.steps.find(s => s.id === stepId)

            // Update DAG execution state to failed
            if (currentState.dagExecution) {
              const updatedDAGExecution = {
                ...currentState.dagExecution,
                phase: "failed" as const,
                updated_at: message.timestamp,
              }
              dispatch({ type: "SET_DAG_EXECUTION", payload: updatedDAGExecution })
            }

            // Update step status
            const step: StepExecution = {
              id: stepId,
              name: stepName,
              description: eventData.description || "",
              status: "failed",
              tool_names: eventData.tool_name ? [eventData.tool_name] : eventData.tool_names || [],
              dependencies: existingStep?.dependencies || [],
              started_at: eventData.started_at || existingStep?.started_at,
              completed_at: eventData.completed_at || message.timestamp,
              result_data: eventData.result_data,
              step_data: eventData.step_data,
              file_outputs: eventData.file_outputs || [],
              conditional_branches: eventData.conditional_branches || existingStep?.conditional_branches || {},
              required_branch: eventData.required_branch ?? existingStep?.required_branch ?? null,
              is_conditional: eventData.is_conditional ?? existingStep?.is_conditional ?? false,
            }
            dispatch({ type: "ADD_STEP", payload: step })

            // Add to left panel messages
            dispatch({
              type: "ADD_MESSAGE",
              payload: {
                id: generateMessageId("msg-step-failed"),
                role: "assistant",
                content: (
                  <>
                    <XCircle className="h-4 w-4 inline mr-2 text-red-500" />
                    {t('agent.logs.event.messages.stepFailed', { stepName })}
                  </>
                ),
                timestamp: message.timestamp,
                status: "failed",
              }
            })

            // Also add to traceEvents for displaying execution logs
            const traceEvent: TraceEvent = {
              event_id: generateMessageId(`trace-step-failed`),
              event_type: eventType,
              step_id: stepId,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.dag_step_failed'),
                step_name: stepName,
                description: eventData.description,
                tool_names: eventData.tool_name ? [eventData.tool_name] : eventData.tool_names || [],
                error: eventData.error,
                completed_at: eventData.completed_at || message.timestamp,
              }
            }
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          } else if (eventType === "dag_step_skipped") {
            const stepName = eventData.step_name || eventData.name || eventData.title || `${t('agent.logs.event.messages.execStepPrefix')}${eventData.step_id || t('common.errors.unknown')}`
            dispatch({
              type: "ADD_MESSAGE",
              payload: {
                id: generateMessageId("msg-step-skipped"),
                role: "assistant",
                content: `${t('agent.logs.event.messages.stepSkipped', { stepName })}`,
                timestamp: message.timestamp,
                status: "completed",
              }
            })
          }

          // Task-level LLM Call Events - show as messages (these don't have step_id)
          else if (eventType === "task_start_llm") {
            const modelName = eventData.model_name || "LLM"
            const taskType = eventData.task_type || "LLM Call"

            // Special handling for final answer generation
            if (eventData.task_type === "final_answer_generation") {
              // Check for duplicate final_answer_generation start events
              const content = t('agent.logs.event.messages.finalAnswerGenerating')
              if (!isDuplicateMessageForViewedTask(content, 'final_answer_start')) {
                dispatch({
                  type: "ADD_MESSAGE",
                  payload: {
                    id: generateMessageId("msg-final-answer-start"),
                    role: "assistant",
                    content: (
                      <>
                        <Lightbulb className="h-4 w-4 inline mr-2 text-yellow-500" />
                        {content}
                      </>
                    ),
                    timestamp: message.timestamp,
                    status: "completed",
                  }
                })
              }
            } else if (eventData.task_type === "comprehensive_goal_check") {
              // Show goal check start message
              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId("msg-goal-check-start"),
                  role: "assistant",
                  content: (
                    <div className="flex items-center gap-2">
                      <Target className="h-4 w-4 text-blue-500" />
                      <span className="font-medium">{t('agent.logs.event.messages.goalCheckStart')}</span>
                    </div>
                  ),
                  timestamp: message.timestamp,
                  status: "completed",
                }
              })
            } else {
              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId("msg-task-llm-start"),
                  role: "assistant",
                  content: (
                    <>
                      <Bot className="h-4 w-4 inline mr-2" />
                      {t('agent.logs.event.messages.taskLLMStart', { taskType })}
                    </>
                  ),
                  timestamp: message.timestamp,
                  status: "completed",
                }
              })
            }
            // Task-level LLM Call End Events
          } else if (eventType === "task_end_llm") {
            const modelName = eventData.model_name || "LLM"
            const taskType = eventData.task_type || "LLM Call"

            // Special handling for final answer generation completion
            if (eventData.task_type === "final_answer_generation") {
              // Check for duplicate final_answer_generation end events
              const content = t('agent.logs.event.messages.finalAnswerCompleted')
              if (!isDuplicateMessageForViewedTask(content, 'final_answer_end')) {
                dispatch({
                  type: "ADD_MESSAGE",
                  payload: {
                    id: generateMessageId("msg-final-answer-end"),
                    role: "assistant",
                    content: (
                      <>
                        <CheckCircle className="h-4 w-4 inline mr-2 text-green-500" />
                        {content}
                      </>
                    ),
                    timestamp: message.timestamp,
                    status: "completed",
                  }
                })
              }
            } else if (eventData.task_type === "comprehensive_goal_check") {
              // Display comprehensive goal check results (only in end events)
              const goalAchieved = eventData.goal_achieved || false
              const goalReason = eventData.goal_reason || "No reason provided"
              const goalConfidence = eventData.goal_confidence || 0
              const memoryShouldStore = eventData.memory_should_store || false
              const memoryReason = eventData.memory_reason || "No memory reason provided"

              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId("msg-goal-check-result"),
                  role: "assistant",
                  content: (
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        {goalAchieved ? (
                          <CheckCircle className="h-4 w-4 text-green-500" />
                        ) : (
                          <XCircle className="h-4 w-4 text-red-500" />
                        )}
                        <span className="font-medium">
                          {t('agent.logs.event.messages.goalCheck')}: {goalAchieved ? t('agent.logs.event.messages.goalAchieved') : t('agent.logs.event.messages.goalNotAchieved')}
                        </span>
                        {goalConfidence > 0 && (
                          <span className="text-sm text-gray-500">
                            ({t('agent.logs.event.messages.confidence', { percent: (goalConfidence * 100).toFixed(0) })})
                          </span>
                        )}
                      </div>
                      {goalReason && (
                        <div className="text-sm text-gray-600 bg-gray-50 p-2 rounded">
                          {t('agent.logs.event.messages.reasonLabel', { goalReason })}
                        </div>
                      )}
                      {memoryShouldStore && (
                        <div className="text-sm text-blue-600 bg-blue-50 p-2 rounded">
                          <Brain className="h-3 w-3 inline mr-1" />
                          {t('agent.logs.event.messages.memoryWillStore', { memoryReason })}
                        </div>
                      )}
                    </div>
                  ),
                  timestamp: message.timestamp,
                  status: "completed",
                }
              })
            } else {
              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId("msg-task-llm-end"),
                  role: "assistant",
                  content: (
                    <>
                      <CheckCircle className="h-4 w-4 inline mr-2 text-green-500" />
                      {t('agent.logs.event.messages.taskLLMCompleted', { taskType })}
                    </>
                  ),
                  timestamp: message.timestamp,
                  status: "completed",
                }
              })
            }
          }

          // Step-level LLM Call Events - add to traceEvents for step execution logs
          else if (eventType === "llm_call_start") {
            dispatch({ type: "UPDATE_TASK_STATUS", payload: { status: "running" } })
            dispatch({ type: "SET_PROCESSING", payload: true })
            if (Number.isFinite(eventData.context_tokens) && Number.isFinite(eventData.context_threshold) && eventData.context_threshold > 0) {
              dispatch({
                type: "SET_CONTEXT_USAGE",
                payload: { tokens: eventData.context_tokens, threshold: eventData.context_threshold },
              })
            }
            if (message.step_id) {
              const modelName = eventData.model_name || "LLM"
              const taskType = eventData.task_type || "LLM Call"

              // Add to traceEvents for step execution logs
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`trace-llm-start`),
                event_type: eventType,
                step_id: message.step_id,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.llm_call_start'),
                  model_name: modelName,
                  task_type: taskType,
                  ...eventData
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            }
          } else if (eventType === "llm_call_end") {
            if (message.step_id) {
              const modelName = eventData.model_name || "LLM"
              const taskType = eventData.task_type || "LLM Call"

              // Add to traceEvents for step execution logs
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`trace-llm-end`),
                event_type: eventType,
                step_id: message.step_id,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.llm_call_end'),
                  model_name: modelName,
                  task_type: taskType,
                  ...eventData
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            }
          }

          // LLM Call Info Events - these are step-level events
          else if (eventType === "llm_call_info") {
            const modelName = eventData.model_name || "LLM"
            const taskType = eventData.task_type || "LLM Call"

            if (!message.step_id) {
              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId("msg-llm-info"),
                  role: "assistant",
                  content: t('agent.logs.event.messages.planLLMSending', { modelName }),
                  timestamp: message.timestamp,
                  status: "completed",
                }
              })
            } else {
              // Add to traceEvents for step execution logs
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`trace-llm-info`),
                event_type: eventType,
                step_id: message.step_id,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.llm_call_info'),
                  model_name: modelName,
                  task_type: taskType,
                  ...eventData
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            }
          }

          // LLM Call Result Events - these are step-level events
          else if (eventType === "llm_call_result") {
            const modelName = eventData.model_name || "LLM"

            if (!message.step_id) {
              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId("msg-llm-result"),
                  role: "assistant",
                  content: (
                    <>
                      <Lightbulb className="h-4 w-4 inline mr-2 text-yellow-500" />
                      {t('agent.logs.event.messages.planLLMResponseCompleted', { modelName })}
                    </>
                  ),
                  timestamp: message.timestamp,
                  status: "completed",
                }
              })
            } else {
              // Add to traceEvents for step execution logs
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`trace-llm-result`),
                event_type: eventType,
                step_id: message.step_id,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.llm_call_result'),
                  model_name: modelName,
                  ...eventData
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            }
          }

          // Tool Execution Events - show as messages if no step_id, otherwise add to traceEvents
          else if (eventType === "tool_execution_start") {
            const toolName = eventData.tool_name || t('nav.tools')
            const stepId = message.step_id || eventData.step_id

            if (!stepId) {
              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId("msg-tool-start"),
                  role: "assistant",
                  content: (
                    <>
                      <Wrench className="h-4 w-4 inline mr-2 text-orange-500" />
                      {t('agent.logs.event.actions.tool_execution_start')}: {toolName}
                    </>
                  ),
                  timestamp: message.timestamp,
                  status: "completed",
                }
              })
            } else {
              // Add to traceEvents for step execution logs
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`trace-tool-start`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.tool_execution_start'),
                  tool_names: [toolName],
                  ...eventData
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            }
          } else if (eventType === "tool_execution_end") {
            const toolName = eventData.tool_name || t('nav.tools')
            const stepId = message.step_id || eventData.step_id

            if (!stepId) {
              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId("msg-tool-end"),
                  role: "assistant",
                  content: (
                    <>
                      <CheckCircle className="h-4 w-4 inline mr-2 text-green-500" />
                      {t('agent.logs.event.actions.tool_execution_end')}: {toolName}
                    </>
                  ),
                  timestamp: message.timestamp,
                  status: "completed",
                }
              })
            } else {
              // Add to traceEvents for step execution logs
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`trace-tool-end`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.tool_execution_end'),
                  tool_names: [toolName],
                  ...eventData
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            }
          } else if (eventType === "tool_execution_failed") {
            const toolName = eventData.tool_name || "Tool"
            const stepId = message.step_id || eventData.step_id

            if (!stepId) {
              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId("msg-tool-failed"),
                  role: "assistant",
                  content: (
                    <>
                      <XCircle className="h-4 w-4 inline mr-2 text-red-500" />
                      {t('agent.logs.event.actions.tool_execution_failed')}: {toolName}
                    </>
                  ),
                  timestamp: message.timestamp,
                  status: "failed",
                }
              })
            } else {
              // Add to traceEvents for step execution logs
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`trace-tool-failed`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.tool_execution_failed'),
                  tool_names: [toolName],
                  ...eventData
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            }
          } else if (eventType === "tool_using") {
            const toolName = eventData.tool_name || t('nav.tools')
            const stepId = message.step_id || eventData.step_id

            if (!stepId) {
              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId("msg-tool-using"),
                  role: "assistant",
                  content: t('agent.logs.event.messages.useTool', { toolName }),
                  timestamp: message.timestamp,
                  status: "completed",
                }
              })
            } else {
              // Add to traceEvents for step execution logs
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`trace-tool-using`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.tool_using'),
                  tool_names: [toolName],
                  ...eventData
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            }
          }

          // Task Completion Events
          else if (eventType === "task_completion") {
            const { result, success } = eventData
            // Check for clarification request in task completion
            const clarification = extractClarificationMessage(eventData)
            if (clarification) {
              const clarificationMessage =
                clarification.message
                || (typeof result?.content === "string" ? result.content : "")
              const msgId = generateMessageId("msg-clarification")
              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: msgId,
                  role: "assistant",
                  content: <div className="space-y-2">
                    <MarkdownRenderer
                      content={clarificationMessage}
                      filesDisabled={options?.filesDisabled}
                      agentCardsEnabled={agentCardsEnabled}
                    />
                    <ClarificationForm
                      interactions={clarification.interactions}
                      messageId={msgId}
                    />
                  </div>,
                  timestamp: message.timestamp,
                  status: "completed",
                  isResult: true,
                  interactions: clarification.interactions,
                }
              })
              return
            }

            // Parse result string into object
            let resultData = {}
            const resultContent = (
              result
              && typeof result === 'object'
              && !Array.isArray(result)
              && ((result as any).file_outputs || (result as any).output || (result as any).content)
            ) ? result : (result?.content || result)
            if (typeof resultContent === 'string') {
              try {
                resultData = JSON.parse(resultContent)
              } catch (e) {
                console.log('Result is not JSON, treating as plain text output:', result.content)
                resultData = { output: resultContent }
              }
            } else if (typeof resultContent === 'object' && resultContent !== null) {
              resultData = resultContent
            } else {
              resultData = { output: resultContent }
            }
            if (
              resultData
              && typeof resultData === 'object'
              && (resultData as any).content
              && !(resultData as any).output
            ) {
              const content = (resultData as any).content
              resultData = {
                ...resultData,
                output: typeof content === "string" ? unwrapFinalAnswerContent(content) : content,
              }
            }

            // 1. Output meta info (excluding output, file_outputs, and history)
            const metaInfo = { ...resultData }
            delete (metaInfo as any).output
            delete (metaInfo as any).content
            delete (metaInfo as any).file_outputs
            delete (metaInfo as any).history
            delete (metaInfo as any).stream_message_id
            delete (metaInfo as any).streamMessageId
            const hasMetaInfo = Object.keys(metaInfo).length > 0 && metaInfo !== null && metaInfo !== undefined

            // 1.5. Extract step data from history and update state.steps
            const history = (resultData as any).history
            if (history && Array.isArray(history) && history.length > 0) {
              const latestIteration = history[history.length - 1] // Latest iteration (the last one)
              if (latestIteration.plan && latestIteration.plan.steps && Array.isArray(latestIteration.plan.steps)) {
                // Create results map for quick lookup
                const resultsMap = new Map<string, any>()
                if (latestIteration.results && Array.isArray(latestIteration.results)) {
                  latestIteration.results.forEach((result: any) => {
                    resultsMap.set(result.step_id, result)
                  })
                }

                // Get active_branches to determine which steps are skipped
                const activeBranches = latestIteration.plan?.active_branches || {}

                // Get existing steps to preserve timing information
                const existingSteps = currentState.steps
                const existingStepsMap = new Map<string, StepExecution>()
                existingSteps.forEach(step => existingStepsMap.set(step.id, step))

                const steps: StepExecution[] = latestIteration.plan.steps.map((step: any) => {
                  // Find corresponding execution result from results
                  const stepResult = resultsMap.get(step.id)
                  // Find existing step
                  const existingStep = existingStepsMap.get(step.id)

                  // If execution result exists, use its status; otherwise use status from plan
                  let finalStatus = step.status || "pending"
                  let startedAt = step.started_at
                  let completedAt = step.completed_at
                  let resultData = step.result

                  if (stepResult) {
                    // Determine status based on result field
                    if (stepResult.result !== undefined && stepResult.result !== null) {
                      finalStatus = "completed"
                    }
                    // Use timing info from stepResult regardless of result existence (if present)
                    if (stepResult.started_at) startedAt = stepResult.started_at
                    if (stepResult.completed_at) completedAt = stepResult.completed_at
                    // If stepResult has result field, use it
                    if (stepResult.result !== undefined && stepResult.result !== null) {
                      resultData = stepResult.result
                    }
                  }

                  // Check if should be skipped: if step requires specific branch but it is not activated
                  if (step.required_branch) {
                    // Find condition node this step depends on
                    const dependencyNodeId = step.dependencies && step.dependencies.length > 0 ? step.dependencies[0] : null
                    if (dependencyNodeId) {
                      const activeBranch = activeBranches[dependencyNodeId]
                      if (activeBranch && activeBranch !== step.required_branch) {
                        // Branch not activated, so this step is skipped
                        finalStatus = "skipped"
                      }
                    }
                  }

                  // Prioritize existing step info (if no explicit info in new data)
                  if (existingStep) {
                    // Prioritize existing step timing info
                    if (!startedAt && existingStep.started_at) startedAt = existingStep.started_at
                    if (!completedAt && existingStep.completed_at) completedAt = existingStep.completed_at

                    // Prioritize existing step status (if new step status is pending or running)
                    // This ensures status from dag_step_end event is not overwritten by plan data
                    if (finalStatus === "pending" || finalStatus === "running") {
                      if (existingStep.status && existingStep.status !== "pending" && existingStep.status !== "running") {
                        finalStatus = existingStep.status
                      }
                    }
                  }

                  return {
                    id: step.id,
                    name: step.name || step.id,
                    description: step.description || "",
                    status: finalStatus,
                    tool_names: step.tool_name ? [step.tool_name] : step.tool_names || [],
                    dependencies: step.dependencies || [],
                    started_at: startedAt,
                    completed_at: completedAt,
                    result_data: resultData,
                    step_data: step.step_data,
                    file_outputs: step.file_outputs || [],
                    conditional_branches: step.conditional_branches || {},
                    required_branch: step.required_branch || null,
                    is_conditional: step.is_conditional || false,
                  }
                })
                dispatch({ type: "SET_STEPS", payload: steps })
              }
            }

            if (hasMetaInfo) {
              const metaContent = (
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-sm text-purple-400">
                    <Target className="h-4 w-4" />
                    <span>{t('agent.logs.event.messages.metaTitle')}</span>
                  </div>
                  <div className="ml-6">
                    <JsonRenderer
                      data={metaInfo}
                      filesDisabled={options?.filesDisabled}
                      agentCardsEnabled={agentCardsEnabled}
                      onFileClick={options?.filesDisabled ? undefined : openFilePreview}
                      onAgentClick={(agentId) => router.push(`/agent/${agentId}`)}
                    />
                  </div>
                </div>
              )
              if (!isDuplicateResultForViewedTask(`📋 ${t('agent.logs.event.messages.metaTitle')}: ${JSON.stringify(metaInfo)}`)) {
                dispatch({
                  type: "ADD_MESSAGE",
                  payload: {
                    id: generateMessageId("msg-meta-info"),
                    role: "assistant",
                    content: metaContent,
                    timestamp: message.timestamp,
                    status: success ? "completed" : "failed",
                    // @ts-ignore
                    isMetaInfo: true,
                  }
                })
              }
            }

            // 2. Output file outputs
            const fileOutputsData = (resultData as any).file_outputs
            if (fileOutputsData && fileOutputsData.length > 0) {
              const fileCount = fileOutputsData.length
              const fileContent = (
                <>
                  <FileText className="h-4 w-4 inline mr-2 text-green-500" />
                  {t('agent.logs.event.messages.fileOutputsGenerated', { count: fileCount })}:
                  <div className="mt-2 space-y-1">
                    {fileOutputsData.map((file: string | any, index: number) => {
                      let fileName, filePath
                      if (typeof file === 'object' && file !== null) {
                        fileName = file.filename || 'unknown'
                        filePath = file.file_id || ''
                      } else {
                        fileName = 'unknown'
                        filePath = ''
                      }

                      return (
                        <div key={index} className="flex items-center justify-between bg-muted/30 rounded p-2">
                          <span className="text-sm font-mono">{fileName}</span>
                          <button
                            onClick={() => {
                              if (options?.filesDisabled) return
                              // Dispatch custom event to open file preview with all files
                              const allFiles = normalizeGeneratedPreviewFiles(fileOutputsData)

                              if (!filePath) {
                                return
                              }

                              window.dispatchEvent(new CustomEvent('openFilePreview', {
                                detail: {
                                  filePath,
                                  fileName,
                                  allFiles,
                                  currentIndex: index
                                }
                              }))
                            }}
                            disabled={options?.filesDisabled || !filePath}
                            className="text-xs bg-primary/10 hover:bg-primary/20 text-primary px-2 py-1 rounded transition-colors"
                          >
                            {t('agent.logs.event.messages.previewLabel')}
                          </button>
                        </div>
                      )
                    })}
                  </div>
                </>
              )

              if (!isDuplicateResultForViewedTask(`📁 ${t('agent.logs.event.messages.fileOutputsGenerated', { count: fileCount })}`)) {
                dispatch({
                  type: "ADD_MESSAGE",
                  payload: {
                    id: generateMessageId("msg-file-outputs"),
                    role: "assistant",
                    content: fileContent,
                    timestamp: message.timestamp,
                    status: "completed",
                    isFileOutput: true,
                  }
                })
              }

              if (!options?.filesDisabled) {
                dispatchAutoOpenPreview(fileOutputsData, dispatch)
              }
            }

            // Update task status and trigger sidebar update
            dispatch({
              type: "UPDATE_TASK_STATUS",
              payload: { status: success ? "completed" : "failed" }
            })
          }

          // Execution Log Events
          else if (eventType === "execution_log") {
            const { level, message: logMessage, step_id, step_name } = eventData
            let displayMessage = logMessage
            if (step_name) {
              displayMessage = `[${step_name}] ${logMessage}`
            }

            const getIcon = () => {
              switch (level) {
                case 'info': return <Info className="h-4 w-4 inline mr-2 text-blue-500" />
                case 'warning': return <AlertTriangle className="h-4 w-4 inline mr-2 text-yellow-500" />
                case 'error': return <XCircle className="h-4 w-4 inline mr-2 text-red-500" />
                case 'debug': return <Search className="h-4 w-4 inline mr-2 text-purple-500" />
                case 'success': return <CheckCircle className="h-4 w-4 inline mr-2 text-green-500" />
                default: return <FileText className="h-4 w-4 inline mr-2 text-gray-500" />
              }
            }

            dispatch({
              type: "ADD_MESSAGE",
              payload: {
                id: generateMessageId("msg-exec-log"),
                role: "assistant",
                content: (
                  <>
                    {getIcon()}
                    {displayMessage}
                  </>
                ),
                timestamp: message.timestamp,
                status: level === 'error' ? 'failed' : 'completed',
              }
            })
          }

          // Error Events
          else if (eventType === "trace_error") {
            // Prioritize error_message, if not present use error, finally use default message
            const errorMessage = eventData.error_message || eventData.error || 'Trace error occurred'
            const stepName = eventData.step_name || eventData.name || `${t('agent.logs.event.messages.execStepPrefix')}${eventData.step_id || t('common.errors.unknown')}`
            const stepId = message.step_id || eventData.step_id

            // Debug information
            console.trace('trace_error debug:', {
              message_step_id: message.step_id,
              eventData_step_id: eventData.step_id,
              stepName: stepName,
              stepId: stepId,
              hasStepId: !!stepId,
              eventData: eventData,
              errorMessage: errorMessage
            })

            // Only add to trace events for displaying execution logs, do not mark step as failed
            const traceEvent: TraceEvent = {
              event_id: generateMessageId(`trace-error-${stepId || 'global'}`),
              event_type: eventType,
              step_id: stepId,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.trace_error'),
                step_name: stepName,
                error: errorMessage,
                error_type: eventData.error_type,
                tool_names: eventData.tool_name ? [eventData.tool_name] : eventData.tool_names || [],
                ...(eventData.execution_time && { execution_time: eventData.execution_time }),
              }
            }
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })

            // For step-related errors, do not display in left panel, only in right panel
            // Only display non-step-related global errors in left panel
            if (!stepId || stepId === 'unknown') {
              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId("msg-trace-error"),
                  role: "assistant",
                  content: (
                    <>
                      <XCircle className="h-4 w-4 inline mr-2 text-red-500" />
                      {t('agent.logs.event.messages.errorPrefix')} {errorMessage}
                    </>
                  ),
                  timestamp: message.timestamp,
                  status: "failed",
                }
              })
            }
          }

          // Visualization Events
          else if (eventType === "visualization_update") {
            dispatch({
              type: "ADD_MESSAGE",
              payload: {
                id: generateMessageId("msg-viz"),
                role: "assistant",
                content: (
                  <>
                    <Activity className="h-4 w-4 inline mr-2 text-blue-500" />
                    {t('agent.logs.event.messages.visualUpdate', { type: eventData.type || 'unknown' })}
                  </>
                ),
                timestamp: message.timestamp,
                status: "completed",
              }
            })
          }

          // ReAct Pattern Events - these should be displayed in the right panel
          else if (eventType === "react_task_start" || eventType === "task_start_react") {
            dispatch({ type: "UPDATE_TASK_STATUS", payload: { status: "running" } })
            dispatch({ type: "SET_PROCESSING", payload: true })

            // Add to trace events for displaying execution logs
            const traceEvent: TraceEvent = {
              event_id: generateMessageId("react-task-start"),
              event_type: eventType,
              step_id: eventData.step_id,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.react_task_start'),
                message: t('agent.logs.event.messages.reactTaskStart'),
                ...eventData
              }
            }
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          } else if (eventType === "react_task_end" || eventType === "task_end_react") {
            const traceEvent: TraceEvent = {
              event_id: generateMessageId("react-task-end"),
              event_type: eventType,
              step_id: eventData.step_id,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.react_task_end'),
                message: t('agent.logs.event.messages.reactTaskCompleted'),
                ...eventData
              }
            }
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })

            const result = eventData.result
            const messageContent =
              result && typeof result === "object" && result.status === "waiting_for_user"
                ? result.message || ""
                : ""
            if (messageContent && !isDuplicateMessageForViewedTask(messageContent, 'agent-message')) {
              const interactions = normalizeInteractions(result.interactions)
              const msgId = generateMessageId("msg-agent")
              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: msgId,
                  role: "assistant",
                  content: messageContent,
                  rawContent: messageContent,
                  timestamp: message.timestamp,
                  status: "running",
                  isResult: true,
                  interactions: interactions.length > 0 ? interactions : undefined,
                }
              })
            }
          } else if (eventType === "react_task_failed" || eventType === "task_failed_react") {
            const traceEvent: TraceEvent = {
              event_id: generateMessageId("react-task-failed"),
              event_type: eventType,
              step_id: eventData.step_id,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.react_task_failed'),
                message: t('agent.logs.event.messages.reactTaskFailed'),
                error: eventData.error || eventData.message,
                ...eventData
              }
            }
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          } else if (eventType === "react_action_start") {
            const stepId = message.step_id || traceEventData.step_id
            const traceEvent: TraceEvent = {
              event_id: generateMessageId("react-action-start"),
              event_type: eventType,
              step_id: stepId,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.react_action_start') || 'Action Start',
                ...eventData
              }
            }
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          } else if (eventType === "llm_call_start") {
            dispatch({ type: "UPDATE_TASK_STATUS", payload: { status: "running" } })
            dispatch({ type: "SET_PROCESSING", payload: true })
            const stepId = message.step_id || traceEventData.step_id
            const traceEvent: TraceEvent = {
              event_id: generateMessageId("llm-call-start"),
              event_type: eventType,
              step_id: stepId,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.llm_call_start'),
                ...eventData
              }
            }
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          } else if (eventType === "llm_call_end") {
            const stepId = message.step_id || traceEventData.step_id
            const traceEvent: TraceEvent = {
              event_id: generateMessageId("llm-call-end"),
              event_type: eventType,
              step_id: stepId,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.llm_call_end'),
                ...eventData
              }
            }
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          } else if (eventType === "llm_call_failed") {
            const stepId = message.step_id || traceEventData.step_id
            const traceEvent: TraceEvent = {
              event_id: generateMessageId("llm-call-failed"),
              event_type: eventType,
              step_id: stepId,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.llm_call_failed'),
                ...eventData
              }
            }
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          } else if (eventType === "react_action_end") {
            const stepId = message.step_id || traceEventData.step_id
            const traceEvent: TraceEvent = {
              event_id: generateMessageId("react-action-end"),
              event_type: eventType,
              step_id: stepId,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.react_action_end') || 'Action End',
                ...eventData
              }
            }
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          } else if (eventType === "task_completion") {
            // Trace task completion event
            const traceEvent: TraceEvent = {
              event_id: generateMessageId("task-completion"),
              event_type: eventType,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.task_completion'),
                message: t('agent.logs.event.messages.taskCompleted'),
                result: eventData.result,
                success: eventData.success
              }
            }
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          } else if (eventType === "react_task_end" || eventType === "task_end_react") {
            // Add to trace events for execution log display
            const traceEvent: TraceEvent = {
              event_id: generateMessageId("react-task-end"),
              event_type: eventType,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.react_task_end'),
                message: t('agent.logs.event.messages.reactTaskCompleted'),
                output: eventData.output
              }
            }
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          } else if (eventType === "step_start_react") {
            const stepName = eventData.step_name || 'unknown'
            const stepId = `react-${stepName}`

            // Create or update step
            const step: StepExecution = {
              id: stepId,
              name: stepName,
              description: `ReAct Step: ${stepName}`,
              status: "running",
              tool_names: eventData.tool_name ? [eventData.tool_name] : eventData.tool_names || [],
              dependencies: [],
              started_at: message.timestamp,
              completed_at: undefined,
              result_data: null,
              step_data: eventData,
              file_outputs: [],
            }
            dispatch({ type: "ADD_STEP", payload: step })

            // Add to trace events for displaying execution logs
            const traceEvent: TraceEvent = {
              event_id: generateMessageId(`react-step-start-${stepId}`),
              event_type: eventType,
              step_id: stepId,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.step_start_react'),
                step_name: stepName,
                tool_names: eventData.tool_name ? [eventData.tool_name] : eventData.tool_names || [],
                message: t('agent.logs.event.messages.reactStepStart', { stepName }),
              }
            }
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          } else if (eventType === "step_end_react") {
            const stepName = eventData.step_name || 'unknown'
            const stepId = `react-${stepName}`

            // Update step status
            const step: StepExecution = {
              id: stepId,
              name: stepName,
              description: `ReAct Step: ${stepName}`,
              status: "completed",
              tool_names: eventData.tool_name ? [eventData.tool_name] : eventData.tool_names || [],
              dependencies: [],
              started_at: undefined, // Preserve original start time
              completed_at: message.timestamp,
              result_data: eventData.result_data,
              step_data: eventData,
              file_outputs: eventData.file_outputs || [],
            }
            dispatch({ type: "ADD_STEP", payload: step })

            // Add to trace events for execution log display
            const traceEvent: TraceEvent = {
              event_id: generateMessageId(`react-step-end-${stepId}`),
              event_type: eventType,
              step_id: stepId,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.step_end_react'),
                step_name: stepName,
                tool_names: eventData.tool_name ? [eventData.tool_name] : eventData.tool_names || [],
                result_data: eventData.result_data,
                message: t('agent.logs.event.messages.reactStepCompleted', { stepName }),
              }
            }
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          }
          // Skill Selection Events
          else if (eventType === "skill_select_start") {
            dispatch({ type: "UPDATE_TASK_STATUS", payload: { status: "running" } })
            dispatch({ type: "SET_PROCESSING", payload: true })

            const traceEvent: TraceEvent = {
              event_id: generateMessageId("skill-select-start"),
              event_type: eventType,
              step_id: eventData.step_id,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.skill_select_start'),
                ...eventData
              }
            }
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          } else if (eventType === "skill_select_end") {
            const traceEvent: TraceEvent = {
              event_id: generateMessageId("skill-select-end"),
              event_type: eventType,
              step_id: eventData.step_id,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.skill_select_end'),
                ...eventData
              }
            }
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          }
          // Memory Events - Determine display location based on step_id
          else if (eventType === "task_start_memory_generate") {
            const stepId = eventData.step_id

            // If step_id exists, add to corresponding step; otherwise do not display (skip useless start events)
            if (stepId) {
              // ReAct pattern - Display in corresponding step in right panel
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`memory-generate-start-${stepId}`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.task_start_memory_generate'),
                  message: '🧠 ' + t('agent.logs.event.actions.task_start_memory_generate'),
                  task: eventData.task,
                  iterations: eventData.iterations,
                  result_length: eventData.result_length,
                  messages_count: eventData.messages_count,
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            }
            // Skip if no step_id, do not display start event
          } else if (eventType === "task_end_memory_generate") {
            const taskId = eventData.task_id || "unknown"
            const stepId = eventData.step_id

            // If step_id exists, add to corresponding step; otherwise display in left panel
            if (stepId) {
              // ReAct pattern - Display in corresponding step in right panel
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`memory-generate-end-${stepId}`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.task_end_memory_generate'),
                  message: '🧠 ' + t('agent.logs.event.actions.task_end_memory_generate'),
                  insights_generated: eventData.insights_generated,
                  should_store: eventData.should_store,
                  reason: eventData.reason,
                  source: eventData.source,
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            } else {
              // DAG plan-execute pattern - Display in left panel
              const shouldStore = eventData.should_store || false
              const reason = eventData.reason || ""
              const source = eventData.source || "unknown"

              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId("msg-memory-generate-end"),
                  role: "assistant",
                  content: (
                    <>
                      <span>
                        <Brain className="h-4 w-4 inline mr-2" />
                        {t('agent.logs.event.actions.task_end_memory_generate')}
                      </span>
                      <div className="mt-2">
                        <CollapsibleSection
                          title={t('agent.logs.event.messages.detailsTitle')}
                          badge={t('agent.logs.event.messages.memoryBadge')}
                        >
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <span className="font-medium text-sm">{t('agent.logs.event.messages.insightsLabel')}</span>
                              {eventData.insights_generated ? (
                                <Badge className="bg-green-100 text-green-800 text-xs">{t('agent.logs.event.labels.success')}</Badge>
                              ) : (
                                <Badge variant="destructive" className="text-xs">{t('agent.logs.event.labels.failed')}</Badge>
                              )}
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="font-medium text-sm">{t('agent.logs.event.messages.storeSuggestion')}</span>
                              {shouldStore ? (
                                <Badge className="bg-green-100 text-green-800 text-xs">{t('agent.logs.event.messages.worthStoring')}</Badge>
                              ) : (
                                <Badge variant="secondary" className="text-xs">{t('agent.logs.event.messages.notWorthStoring')}</Badge>
                              )}
                            </div>
                            {reason && (
                              <div className="text-sm">
                                <span className="font-medium">{t('agent.logs.event.messages.reason')}</span> {reason}
                              </div>
                            )}
                          </div>
                        </CollapsibleSection>
                      </div>
                    </>
                  ),
                  timestamp: message.timestamp,
                  status: "completed",
                }
              })
            }
          } else if (eventType === "task_start_memory_store") {
            const taskId = eventData.task_id || "unknown"
            const stepId = eventData.step_id

            // If step_id exists, add to corresponding step; otherwise display in left panel
            if (stepId) {
              // ReAct pattern - Display in corresponding step in right panel
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`memory-store-start-${stepId}`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.task_start_memory_store'),
                  message: '🧠 ' + t('agent.logs.event.actions.task_start_memory_store'),
                  task: eventData.task,
                  memory_category: eventData.memory_category,
                  classification: eventData.classification,
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            } else {
              // DAG plan-execute pattern - Display in left panel
              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId("msg-memory-store-start"),
                  role: "assistant",
                  content: (
                    <>
                      <Brain className="h-4 w-4 inline mr-2" />
                      {t('agent.logs.event.actions.task_start_memory_store')}
                      {eventData.task && (
                        <div className="text-sm text-gray-600 mt-1">
                          {t('agent.logs.event.messages.taskLabel')} {eventData.task.length > 100 ? eventData.task.substring(0, 100) + '...' : eventData.task}
                        </div>
                      )}
                      {eventData.memory_category && (
                        <div className="text-sm text-gray-600 mt-1">
                          {t('agent.logs.event.memory.category')}: {eventData.memory_category}
                        </div>
                      )}
                    </>
                  ),
                  timestamp: message.timestamp,
                  status: "running",
                }
              })
            }
          } else if (eventType === "task_end_memory_store") {
            const taskId = eventData.task_id || "unknown"
            const stepId = eventData.step_id

            // If step_id exists, add to corresponding step; otherwise display in left panel
            if (stepId) {
              // ReAct pattern - Display in corresponding step in right panel
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`memory-store-end-${stepId}`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.task_end_memory_store'),
                  message: '🧠 ' + t('agent.logs.event.actions.task_end_memory_store'),
                  storage_success: eventData.storage_success,
                  reason: eventData.reason,
                  decision: eventData.decision,
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            } else {
              // DAG plan-execute pattern - Display in left panel
              const storageSuccess = eventData.storage_success || false
              const reason = eventData.reason || ""
              const decision = eventData.decision || "unknown"

              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId("msg-memory-store-end"),
                  role: "assistant",
                  content: (
                    <>
                      <span>
                        <Brain className="h-4 w-4 inline mr-2" />
                        {t('agent.logs.event.actions.task_end_memory_store')}
                      </span>
                      <div className="mt-2">
                        <CollapsibleSection
                          title={t('agent.logs.event.messages.detailsTitle')}
                          badge={t('agent.logs.event.messages.memoryBadge')}
                        >
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <span className="font-medium text-sm">{t('agent.logs.event.messages.storageStatusLabel')}</span>
                              {storageSuccess ? (
                                <Badge className="bg-green-100 text-green-800 text-xs">{t('agent.logs.event.labels.success')}</Badge>
                              ) : (
                                <Badge variant="secondary" className="text-xs">{t('agent.logs.event.messages.notStored')}</Badge>
                              )}
                            </div>
                            {reason && (
                              <div className="text-sm">
                                <span className="font-medium">{t('agent.logs.event.messages.reason')}</span> {reason}
                              </div>
                            )}
                            {decision && decision !== 'unknown' && (
                              <div className="text-sm">
                                <span className="font-medium">{t('agent.logs.event.messages.decisionLabel')}</span> {decision === 'not_worth_storing' ? t('agent.logs.event.messages.notWorthStoring') : decision}
                              </div>
                            )}
                          </div>
                        </CollapsibleSection>
                      </div>
                    </>
                  ),
                  timestamp: message.timestamp,
                  status: "completed",
                }
              })
            }
          } else if (eventType === "task_start_memory_retrieve") {
            const taskId = eventData.task_id || "unknown"
            const stepId = eventData.step_id

            // If step_id exists, add to corresponding step; otherwise display in left panel
            if (stepId) {
              // ReAct pattern - Display in corresponding step in right panel
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`memory-retrieve-start-${stepId}`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.task_start_memory_retrieve'),
                  message: '🔍 ' + t('agent.logs.event.actions.task_start_memory_retrieve'),
                  // Display full data
                  rawData: eventData,
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            } else {
              // DAG plan-execute pattern - Display in left panel
              const stepId = eventData.step_id || "unknown"

              // Memory retrieval start event
              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId(`memory-retrieve-start-${stepId}`),
                  role: "assistant",
                  content: (
                    <>
                      <Search className="h-4 w-4 inline mr-2" />
                      {t('agent.logs.event.actions.task_start_memory_retrieve')}
                      <div className="mt-1">
                        <CollapsibleSection
                          title={t('agent.logs.event.common.fullData')}
                          badge={t('agent.logs.event.messages.memoryBadge')}
                        >
                          <div className="text-xs bg-muted/80 p-2 rounded font-mono text-foreground">
                            {JSON.stringify(eventData, null, 2)}
                          </div>
                        </CollapsibleSection>
                      </div>
                    </>
                  ),
                  timestamp: message.timestamp,
                  status: "running",
                }
              })
            }
          } else if (eventType === "task_end_memory_retrieve") {
            const taskId = eventData.task_id || "unknown"
            const stepId = eventData.step_id

            // If step_id exists, add to corresponding step; otherwise display in left panel
            if (stepId) {
              // ReAct pattern - Display in corresponding step in right panel
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`memory-retrieve-end-${stepId}`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.task_end_memory_retrieve'),
                  message: '🔍 ' + t('agent.logs.event.actions.task_end_memory_retrieve'),
                  // Display full data
                  rawData: eventData,
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            } else {
              // DAG plan-execute pattern - Display in left panel
              const stepId = eventData.step_id || "unknown"
              const memoriesFound = eventData.memories_found || 0
              const memoriesUsed = eventData.memories_used || 0
              const memoryCategory = eventData.memory_category || t('agent.logs.event.messages.categoryUnknown')
              const enhancedGoal = eventData.enhanced_goal
              const memories = eventData.memories || []

              // Store plan memory information for display
              console.log("Setting planMemoryInfo:", { memoriesFound, memoriesUsed, memoryCategory, enhancedGoal, memories })
              dispatch({
                type: "SET_PLAN_MEMORY_INFO",
                payload: {
                  memoriesFound,
                  memoriesUsed,
                  memoryCategory,
                  enhancedGoal,
                  memories: memories.map((mem: any) => ({
                    content: mem.content || mem,
                    category: mem.category
                  }))
                }
              })

              // Memory retrieval end event
              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId(`memory-retrieve-end-${stepId}`),
                  role: "assistant",
                  content: (
                    <>
                      <Search className="h-4 w-4 inline mr-2" />
                      {t('agent.logs.event.actions.task_end_memory_retrieve')}
                      <div className="mt-2">
                        <CollapsibleSection
                          title={t('agent.logs.event.messages.detailsTitle')}
                          badge={t('agent.logs.event.messages.memoryBadge')}
                        >
                          <div className="grid grid-cols-2 gap-2 text-xs">
                            <div className="flex items-center gap-1 p-2 bg-muted/30 rounded">
                              <Search className="h-3 w-3" />
                              <span>{t('agent.logs.event.memory.found')}: {memoriesFound} {t('agent.logs.event.common.itemsSuffix')}</span>
                            </div>
                            <div className="flex items-center gap-1 p-2 bg-muted/30 rounded">
                              <Target className="h-3 w-3" />
                              <span>{t('agent.logs.event.memory.used')}: {memoriesUsed} {t('agent.logs.event.common.itemsSuffix')}</span>
                            </div>
                          </div>
                          {enhancedGoal && (
                            <div className="mt-2">
                              <div className="text-xs font-medium text-muted-foreground mb-1">{t('agent.planDetails.memory.enhancedGoalTitle')}</div>
                              <div className="text-xs bg-blue-500/10 p-2 rounded border border-blue-500/20">
                                {enhancedGoal}
                              </div>
                            </div>
                          )}
                          {memories && memories.length > 0 && (
                            <div className="mt-2">
                              <div className="text-xs font-medium text-muted-foreground mb-1">{t('agent.logs.event.memory.relatedTitle')}:</div>
                              <div className="space-y-1">
                                {memories.map((memory: any, index: number) => (
                                  <div
                                    key={index}
                                    className="text-xs p-2 bg-muted/20 rounded border border-border/50"
                                  >
                                    <div className="flex items-start gap-1">
                                      <Info className="h-3 w-3 mt-0.5 text-blue-400 flex-shrink-0" />
                                      <span className="whitespace-pre-wrap">{memory.content}</span>
                                    </div>
                                    {memory.category && (
                                      <Badge variant="outline" className="text-xs mt-1">
                                        {memory.category}
                                      </Badge>
                                    )}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </CollapsibleSection>
                      </div>
                    </>
                  ),
                  timestamp: message.timestamp,
                  status: "completed",
                }
              })
            }
          }

          // Legacy Events
          else if (eventType === "task-info") {
            dispatch({
              type: "ADD_MESSAGE",
              payload: {
                id: generateMessageId("msg-task-info"),
                role: "assistant",
                content: (
                  <>
                    <FileText className="h-4 w-4 inline mr-2" />
                    {t('agent.logs.event.messages.taskInfoLabel')} {eventData.title || 'unknown'}
                  </>
                ),
                timestamp: message.timestamp,
                status: "completed",
              }
            })
          }
          // final-result event type removed - use task_completion instead
          // file-output event type removed - handled in task_completion instead

          // Historical Data Events - handled by the main message handler below
          else if (eventType === "historical_data_complete") {
            isHistoricalDataLoadingRef.current = false
            dispatch({ type: "SET_HISTORY_LOADING", payload: false })
            dispatch({ type: "SYNC_PROCESSING_STATUS" })

            // If we're in replay mode, initialize the replay scheduler
            if (currentState.isReplaying && currentState.replayTaskId && currentState.replayEventCache.length > 0) {
              initializeReplayScheduler()
            } else {
              // Fix: If we have cache but replay mode is not set, force start replay
              if (currentState.replayEventCache.length > 0 && currentState.replayTaskId && !currentState.isReplaying) {
                dispatch({ type: "SET_REPLAY_PLAYING", payload: true })
                setTimeout(() => {
                  initializeReplayScheduler()
                }, 50)
              }
            }
          }

          // Default: add as trace event
          else {
            console.trace('Original message:', JSON.stringify(message), 'Handler: handleMessage (unhandled event_type:', eventType, ')')
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEventData })
          }
        } else {
          console.trace('Original message:', JSON.stringify(message), 'Handler: handleMessage (no event_type, direct trace event)')
          // Handle direct trace events (without event_type wrapper) - infer type from content
          // Check if this is DAG execution data
          if (traceEventData.phase && (traceEventData.current_plan !== undefined)) {
            dispatch({ type: "SET_DAG_EXECUTION", payload: traceEventData })
          }
          // Check if this is step data (has id and status)
          else if (traceEventData.id && traceEventData.status) {
            // More strict criteria for step identification
            const hasStepProperties = traceEventData.name || traceEventData.tool_name || traceEventData.tool_names || traceEventData.description
            const hasValidStepId = typeof traceEventData.id === 'string' && traceEventData.id.length > 2
            const isNotNumericId = isNaN(traceEventData.id)

            if (hasStepProperties && hasValidStepId && isNotNumericId) {
              const step: StepExecution = {
                id: traceEventData.id,
                name: traceEventData.name || traceEventData.id,
                description: traceEventData.description || "",
                status: traceEventData.status,
                tool_names: traceEventData.tool_name ? [traceEventData.tool_name] : traceEventData.tool_names || [],
                dependencies: traceEventData.dependencies || [],
                started_at: traceEventData.started_at,
                completed_at: traceEventData.completed_at,
                result_data: traceEventData.result_data,
                step_data: traceEventData.step_data,
                file_outputs: traceEventData.file_outputs || [],
              }
              dispatch({ type: "ADD_STEP", payload: step })
            } else {
              // Add as trace event instead
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEventData })
            }
          }
          // Check if this is task info (has goal)
          else if (traceEventData.goal) {
            // For now, create a basic task structure
            const task = {
              id: currentState.taskId?.toString() || "unknown",
              title: traceEventData.task_preview || traceEventData.goal,
              description: traceEventData.goal,
              status: "completed" as const,
              createdAt: new Date().toISOString(),
              updatedAt: new Date().toISOString(),
            }
            dispatch({ type: "SET_CURRENT_TASK", payload: task })
          }
          // Check if this is a plan start event (has plan_data or current_plan)
          else if (traceEventData.plan_data || traceEventData.current_plan) {
            const planData = traceEventData
            const phase = planData.phase || "planning"
            const planInfo = planData.plan_data || planData.current_plan

            if (planInfo && planInfo.goal && planInfo.steps) {
              // Detailed plan information
              const stepsCount = planInfo.steps.length || planData.steps_count || 0
              const goal = planInfo.goal
              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId("msg-plan-start"),
                  role: "assistant",
                  content: (
                    <>
                      <FileText className="h-4 w-4 inline mr-2" />
                      {t('agent.logs.event.messages.planStart', { phase })}
                      <br />
                      <Target className="h-4 w-4 inline mr-2 mt-1 text-red-500" />
                      {t('agent.logs.event.messages.goalTitle')}: {goal}
                      <br />
                      <Activity className="h-4 w-4 inline mr-2 mt-1 text-blue-500" />
                      {t('agent.logs.event.messages.stepsCount', { count: stepsCount })}
                    </>
                  ),
                  timestamp: message.timestamp,
                  status: "completed",
                }
              })

              // Add individual step messages
              planInfo.steps.forEach((step: any, index: number) => {
                dispatch({
                  type: "ADD_MESSAGE",
                  payload: {
                    id: generateMessageId(`msg-plan-step-${index}`),
                    role: "assistant",
                    content: (
                      <>
                        <Target className="h-4 w-4 inline mr-2 text-red-500" />
                        {t('agent.logs.event.messages.execStepPrefix')}{index + 1}: {step.name || step.id}
                        <br />
                        <span className="ml-6">{step.description || ''}</span>
                      </>
                    ),
                    timestamp: message.timestamp,
                    status: "completed",
                  }
                })
              })
            } else {
              // Basic plan information
              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId("msg-plan-start"),
                  role: "assistant",
                  content: (
                    <>
                      <FileText className="h-4 w-4 inline mr-2" />
                      {t('agent.logs.event.messages.planStart', { phase })}
                    </>
                  ),
                  timestamp: message.timestamp,
                  status: "completed",
                }
              })
            }
          }
          else {
            // Add to trace events for other types
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEventData })
          }
        }
        break

      case "chat_message":
        console.trace('Original message:', JSON.stringify(message), 'Handler: handleMessage (chat_message)')
        const messageData = message.data as any
        dispatch({
          type: "ADD_MESSAGE",
          payload: {
            id: `msg-${messageData.id}`,
            role: messageData.role,
            content: messageData.content,
            timestamp: messageData.timestamp,
          },
        })
        break

      case "task_completed":
        const taskData = normalizeTaskCompletedMessage(message)
        dispatch({
          type: "UPDATE_TASK_STATUS",
          payload: {
            status: taskData.status,
            runId: controlEnvelope.runId,
            stateVersion: controlEnvelope.stateVersion,
            controlState: controlEnvelope.controlState,
          }
        })
        dispatch({ type: "TRIGGER_TASK_UPDATE" })
        dispatch({ type: "SET_PROCESSING", payload: false })  // Stop processing on task completion

        if (!taskData.success) {
          // error_details carries the structured reason ({code, ..., message});
          // fall back to its message when the terminal event omits output. The
          // broadcast always sets `result` === `output`, so `result` is not a
          // distinct source and is intentionally not consulted here.
          const failureReason =
            getString(taskData.output) ||
            getString(taskData.errorDetails?.message)
          // Coded failures (e.g. a quota-gate refusal) also notify the app
          // layer so it can surface them richly (see task-error-events). Stock
          // xagent has a no-op controller and only an app layer ever sets a code.
          // Not gated on failureReason: the coded dialog carries its own copy
          // and must still fire when the reason is empty. Tag it with the
          // event's own task id (not the currently-viewed one) so a dialog
          // cannot be attributed to the wrong task under a task-switch race.
          if (taskData.errorCode) {
            emitTaskError({
              code: taskData.errorCode,
              details: taskData.errorDetails,
              message: failureReason,
              taskId: controlEnvelope.taskId ?? currentState.taskId ?? null,
            })
          }
          // Surface the reason live as a failed assistant message so the
          // conversation matches what a page reload shows. Guard only against an
          // adjacent duplicate (same reason already the last message) rather than
          // a TTL cache — a re-executed turn re-emits the same terminal event and
          // a content-TTL dedup would drop the bubble, leaving an empty turn.
          const lastMessage = currentState.messages[currentState.messages.length - 1]
          const lastContent =
            typeof lastMessage?.content === "string" ? lastMessage.content : ""
          if (failureReason && lastContent !== failureReason) {
            dispatch({
              type: "ADD_MESSAGE",
              payload: {
                id: generateMessageId("msg-task-failed"),
                role: "assistant",
                // The reason verbatim, no prefix: a reload replays the
                // persisted transcript row (this same text), so any live-only
                // decoration would make the two views diverge.
                content: failureReason,
                timestamp: message.timestamp,
                status: "failed",
                // Terminal failure IS this turn's result. Without the flag the
                // conversation panel (which only shows user / isResult /
                // system-notice messages) filters the bubble out and falls back
                // to a virtual "unknown error" placeholder until reload.
                isResult: true,
              },
            })
          }
        }

        // Sync DAG execution status to completed/failed - but only for a
        // task that actually had a DAG plan running (currentState.dagExecution
        // already set by an earlier dag_execution/dag_step_* event this turn).
        // Fabricating a fresh DAGExecution for every completed task regardless
        // of pattern (the previous `else` branch here) made every flash/ReAct
        // task look like a completed zero-step DAG run to any DAG-only UI
        // (e.g. the Progress panel), popping it open for plain replies.
        if (currentState.dagExecution) {
          const updatedDAGExecution = {
            ...currentState.dagExecution,
            phase: taskData.status,
            updated_at: new Date().toISOString()
          }
          dispatch({ type: "SET_DAG_EXECUTION", payload: updatedDAGExecution })
        }

        // Mark that historical data should not be requested again for completed/failed tasks
        if (currentState.taskId) {
          historicalDataRequestMapRef.current.set(currentState.taskId, true)
        }

        // Handle file outputs
        if (taskData.fileOutputs.length > 0) {
          const fileCount = taskData.fileOutputs.length
          const fileContent = (
            <>
              <FileText className="h-4 w-4 inline mr-2 text-green-500" />
              {t('agent.logs.event.messages.fileOutputsGenerated', { count: fileCount })}:
              <div className="mt-2 space-y-1">
                {taskData.fileOutputs.map((file: string | any, index: number) => {
                  let fileName, filePath
                  if (typeof file === 'object' && file !== null) {
                    fileName = file.filename || 'unknown'
                    filePath = file.file_id || ''
                  } else {
                    fileName = 'unknown'
                    filePath = ''
                  }

                  return (
                    <div key={index} className="flex items-center justify-between bg-muted/30 rounded p-2">
                      <span className="text-sm font-mono">{fileName}</span>
                      <button
                        onClick={() => {
                          if (options?.filesDisabled) return
                          // Dispatch custom event to open file preview with all files
                          const allFiles = normalizeGeneratedPreviewFiles(taskData.fileOutputs)

                          if (!filePath) {
                            return
                          }

                          window.dispatchEvent(new CustomEvent('openFilePreview', {
                            detail: {
                              filePath,
                              fileName,
                              allFiles,
                              currentIndex: index
                            }
                          }))
                        }}
                        disabled={options?.filesDisabled || !filePath}
                        className="text-xs bg-primary/10 hover:bg-primary/20 text-primary px-2 py-1 rounded transition-colors"
                      >
                        {t('agent.logs.event.messages.previewLabel')}
                      </button>
                    </div>
                  )
                })}
              </div>
            </>
          )

          if (!isDuplicateResultForViewedTask(`📁 ${t('agent.logs.event.messages.fileOutputsGenerated', { count: fileCount })}`)) {
            dispatch({
              type: "ADD_MESSAGE",
              payload: {
                id: generateMessageId("msg-file-outputs"),
                role: "assistant",
                content: fileContent,
                timestamp: message.timestamp,
                status: "completed",
                isFileOutput: true,
              }
            })
          }

          if (!options?.filesDisabled) {
            dispatchAutoOpenPreview(taskData.fileOutputs, dispatch)
          }
        }

        dispatch({ type: "SET_PROCESSING", payload: false })
        break

      case "dag_step_info":
        const stepInfo = message.data as {
          id: string
          name?: string
          description?: string
          status: StepExecution["status"]
          tool_name?: string
          tool_names?: string[]
          dependencies?: string[]
          started_at?: string | number
          completed_at?: string | number
          result_data?: unknown
          step_data?: unknown
          file_outputs?: string[]
        }
        const step: StepExecution = {
          id: stepInfo.id,
          name: stepInfo.name || stepInfo.id,
          description: stepInfo.description || "",
          status: stepInfo.status,
          tool_names: stepInfo.tool_name ? [stepInfo.tool_name] : stepInfo.tool_names || [],
          dependencies: stepInfo.dependencies || [],
          started_at: stepInfo.started_at,
          completed_at: stepInfo.completed_at,
          result_data: stepInfo.result_data,
          step_data: stepInfo.step_data,
          file_outputs: stepInfo.file_outputs || [],
        }
        dispatch({ type: "ADD_STEP", payload: step })

        // Update DAG execution status
        // Update overall DAG status based on step status
        if (currentState.dagExecution) {
          const updatedDAGExecution = { ...currentState.dagExecution }

          // Update DAG phase based on step status
          if (stepInfo.status === "running") {
            updatedDAGExecution.phase = "executing" as const
          } else if (stepInfo.status === "completed") {
            // Check if all steps are completed
            const allStepsCompleted = currentState.steps.every(step =>
              step.id === stepInfo.id ? stepInfo.status === "completed" : step.status === "completed"
            )
            if (allStepsCompleted) {
              updatedDAGExecution.phase = "completed" as const
            } else {
              updatedDAGExecution.phase = "executing" as const
            }
          } else if (stepInfo.status === "failed") {
            updatedDAGExecution.phase = "failed" as const
          }

          // Update timestamp
          updatedDAGExecution.updated_at = new Date().toISOString()

          dispatch({ type: "SET_DAG_EXECUTION", payload: updatedDAGExecution })
        }
        break

      case "dag_execution":
        const dagSteps = stepsFromPlanData(message.data, currentState.steps)
        if (dagSteps) {
          dispatch({ type: "SET_STEPS", payload: dagSteps })
        }
        {
          const legacyDagData = (message.data ?? {}) as { created_at?: string | number }
          const legacyDagCreatedAt =
            currentState.dagExecution?.created_at ?? legacyDagData.created_at ?? message.timestamp
          dispatch({
            type: "SET_DAG_EXECUTION",
            payload: { ...(message.data as DAGExecution), created_at: legacyDagCreatedAt },
          })
        }
        break


      case "task_paused":
        console.trace('Original message:', JSON.stringify(message), 'Handler: handleMessage (task_paused)')
        dispatch({
          type: "UPDATE_TASK_STATUS",
          payload: {
            status: controlEnvelope.status || "paused",
            runId: controlEnvelope.runId,
            stateVersion: controlEnvelope.stateVersion,
            controlState: controlEnvelope.controlState || "paused",
          },
        })
        dispatch({ type: "SET_PROCESSING", payload: false })
        break

      case "task_pause_requested":
        if (controlEnvelope.status) {
          dispatch({
            type: "UPDATE_TASK_STATUS",
            payload: {
              status: controlEnvelope.status,
              runId: controlEnvelope.runId,
              stateVersion: controlEnvelope.stateVersion,
              controlState: controlEnvelope.controlState || "pause_requested",
            },
          })
        }
        break

      case "task_waiting_for_user":
        console.trace('Original message:', JSON.stringify(message), 'Handler: handleMessage (task_waiting_for_user)')
        const waitingData = message.data as any
        const waitingMessage = waitingData?.question || waitingData?.message || ""
        const interactions = normalizeInteractions(waitingData?.interactions)
        dispatch({
          type: "UPDATE_TASK_STATUS",
          payload: {
            status: controlEnvelope.status || "waiting_for_user",
            waitingQuestion: waitingMessage && waitingMessage !== "Task waiting for user response" ? waitingMessage : undefined,
            waitingInteractions: interactions.length > 0 ? interactions : undefined,
            runId: controlEnvelope.runId,
            stateVersion: controlEnvelope.stateVersion,
            controlState: controlEnvelope.controlState || "waiting_for_user",
          }
        })
        dispatch({ type: "SET_PROCESSING", payload: false })
        if (
          waitingMessage &&
          waitingMessage !== "Task waiting for user response" &&
          !isDuplicateMessageForViewedTask(waitingMessage, 'agent-message')
        ) {
          dispatch({
            type: "ADD_MESSAGE",
            payload: {
              id: generateMessageId("msg-agent"),
              role: "assistant",
              content: waitingMessage,
              rawContent: waitingMessage,
              timestamp: message.timestamp,
              status: "running",
              isResult: true,
              interactions: interactions.length > 0 ? interactions : undefined,
            }
          })
        }
        break

      case "task_resumed":
        console.trace('Original message:', JSON.stringify(message), 'Handler: handleMessage (task_resumed)')
        dispatch({
          type: "UPDATE_TASK_STATUS",
          payload: {
            status: controlEnvelope.status || "running",
            runId: controlEnvelope.runId,
            stateVersion: controlEnvelope.stateVersion,
            controlState: controlEnvelope.controlState || "running",
          },
        })
        break

      case "agent_error":
        console.trace('Original message:', JSON.stringify(message), 'Handler: handleMessage (agent_error)')
        const agentErrorMessage = getWebSocketErrorMessage(message)
        const agentErrorTaskStatus = getWebSocketTaskStatus(message)

        if (agentErrorTaskStatus) {
          dispatch({
            type: "UPDATE_TASK_STATUS",
            payload: {
              status: agentErrorTaskStatus,
              runId: controlEnvelope.runId,
              stateVersion: controlEnvelope.stateVersion,
              controlState: controlEnvelope.controlState,
            },
          })
          dispatch({ type: "TRIGGER_TASK_UPDATE" })
        }

        if (agentErrorTaskStatus === "failed" && currentState.dagExecution) {
          const updatedDAGExecution = {
            ...currentState.dagExecution,
            phase: "failed" as const,
            updated_at: message.timestamp,
          }
          dispatch({ type: "SET_DAG_EXECUTION", payload: updatedDAGExecution })
        }

        if (shouldStopProcessingForTaskStatus(agentErrorTaskStatus)) {
          dispatch({ type: "SET_PROCESSING", payload: false })
        }

        dispatch({
          type: "ADD_MESSAGE",
          payload: {
            id: generateMessageId("msg"),
            role: "assistant",
            content: `${t('agent.logs.event.messages.errorPrefix')} ${agentErrorMessage || t('common.errors.unknown')}`,
            timestamp: message.timestamp,
            status: "failed",
          },
        })
        break

      case "error":
      case "task_error":
        console.trace('Original message:', JSON.stringify(message), 'Handler: handleMessage (error)')
        const websocketErrorMessage = getWebSocketErrorMessage(message)
        const websocketTaskStatus = getWebSocketTaskStatus(message)

        if (websocketTaskStatus) {
          dispatch({ type: "UPDATE_TASK_STATUS", payload: { status: websocketTaskStatus } })
          dispatch({ type: "TRIGGER_TASK_UPDATE" })
        }
        if (shouldStopProcessingForTaskStatus(websocketTaskStatus)) {
          dispatch({ type: "SET_PROCESSING", payload: false })
        }

        if (!isDuplicateMessageForViewedTask(websocketErrorMessage, "agent-error")) {
          dispatch({
            type: "ADD_MESSAGE",
            payload: {
              id: generateMessageId("msg-error"),
              role: "assistant",
              content: `${t('agent.logs.event.messages.errorPrefix')} ${websocketErrorMessage}`,
              timestamp: message.timestamp,
              status: "failed",
            },
          })
        }
        break

      case "message_received":
        console.trace('Original message:', JSON.stringify(message), 'Handler: handleMessage (message_received)')
        // User message confirmation
        dispatch({ type: "SET_PROCESSING", payload: true })
        break

      case "historical_data_complete":
        // Historical data loading complete
        isHistoricalDataLoadingRef.current = false
        dispatch({ type: "SET_HISTORY_LOADING", payload: false })
        dispatch({ type: "SYNC_PROCESSING_STATUS" })

        // If we're in replay mode, initialize the replay scheduler
        if (currentState.isReplaying && currentState.replayTaskId && currentState.replayEventCache.length > 0) {
          initializeReplayScheduler()
        }
        break
    }
  }, [])

  const handleSessionMessage = (
    message: WebSocketMessage,
    owner: SessionMessageOwner,
  ) => {
    if (!mountedRef.current) return
    if (
      owner.connectionIdentity !== sessionConnectionIdentityRef.current
    ) {
      return
    }

    if (message.type === "conversation_reset") {
      const resetFlight = sessionResetFlightRef.current
      if (
        !resetFlight
        || resetFlight.connectionIdentity !== owner.connectionIdentity
        || resetFlight.deliveryGeneration !== deliveryGenerationRef.current
      ) {
        return
      }

      const transition = dispatchSessionConversation({
        type: "SESSION_RESET_ACKNOWLEDGED",
        connectionIdentity: owner.connectionIdentity,
      })
      if (!transition.accepted) return

      const retiredTaskId = sessionTaskIdRef.current
      if (retiredTaskId !== null) {
        retiredSessionTaskIdsRef.current.add(retiredTaskId)
        if (
          retiredSessionTaskIdsRef.current.size
          > MAX_RETIRED_SESSION_TASK_IDS
        ) {
          const oldestTaskId =
            retiredSessionTaskIdsRef.current.values().next().value
          if (oldestTaskId !== undefined) {
            retiredSessionTaskIdsRef.current.delete(oldestTaskId)
          }
        }
      }

      sessionResetFlightRef.current = null
      clearTimeout(resetFlight.timeout)
      sessionTaskIdRef.current = null
      taskStateVersionsRef.current.clear()
      stateRef.current.replayScheduler?.stop()
      pendingTaskToExecuteRef.current = null
      lastConnectedTaskId.current = null
      recentMessagesRef.current.clear()
      const pending = pendingMessageRef.current
      pendingMessageRef.current = null
      setPendingMessage(null)
      pending?.reject?.(
        new Error("Message delivery was cancelled by conversation reset.")
      )

      const nextDeliveryGeneration = deliveryGenerationRef.current + 1
      deliveryGenerationRef.current = nextDeliveryGeneration
      setDeliveryGeneration(nextDeliveryGeneration)
      dispatch({ type: "RESET_SESSION_CONVERSATION" })
      resetFlight.resolve()
      return
    }

    const binding = extractSessionTaskBinding(message)
    if (binding.present && !binding.valid) return

    const taskInfoData = getSessionTaskInfoData(message)
    if (taskInfoData) {
      const taskId = parseInteger(taskInfoData.id)
      if (
        taskId === undefined
        || taskId <= 0
        || retiredSessionTaskIdsRef.current.has(taskId)
        || (
          binding.present
          && binding.valid
          && binding.taskId !== taskId
        )
      ) {
        return
      }
      const lifecycle = sessionConversationRef.current
      const isFlushingCandidate =
        flushingSessionPreAdoptionCandidateRef.current === message
      if (
        !isFlushingCandidate
        && (
          lifecycle.phase === "replacement_sending"
          || lifecycle.phase === "replacement_awaiting_task"
        )
      ) {
        if (!bufferSessionPreAdoptionFrame(message, owner, taskId)) return
        const buffer = sessionPreAdoptionBufferRef.current
        if (
          lifecycle.phase === "replacement_awaiting_task"
          && buffer?.timeout != null
        ) {
          flushingSessionPreAdoptionCandidateRef.current = message
          try {
            handleSessionMessage(message, owner)
          } finally {
            flushingSessionPreAdoptionCandidateRef.current = null
          }
        }
        return
      }
      if (
        lifecycle.phase === "reset_requested"
        || lifecycle.phase === "replacement_ready"
        || lifecycle.phase === "reload_required"
      ) {
        return
      }

      const currentTaskId = sessionTaskIdRef.current
      if (
        currentTaskId !== null
        && currentTaskId !== taskId
      ) {
        requireSessionReload(
          new Error("Session task lineage changed without a reset; reload required.")
        )
        return
      }
      const taskInfoEnvelope = extractTaskControlEnvelope(message)
      if (
        taskInfoEnvelope.taskId !== taskId
        || !canAcceptTaskControlVersion(
          message,
          taskInfoEnvelope,
          taskStateVersionsRef.current,
        )
      ) {
        return
      }

      const bufferedFrames = lifecycle.phase === "replacement_awaiting_task"
        ? sessionPreAdoptionBufferRef.current?.frames.slice() ?? []
        : []
      discardSessionPreAdoptionBuffer()
      const transition = dispatchSessionConversation({
        type: "SESSION_TASK_INFO",
        connectionIdentity: owner.connectionIdentity,
        taskId,
      })
      if (!transition.accepted || transition.next.phase !== "bound") return
      if (!acceptTaskControlVersion(
        message,
        taskInfoEnvelope,
        taskStateVersionsRef.current,
      )) return
      sessionTaskIdRef.current = taskId
      dispatch({
        type: "ADOPT_SESSION_TASK",
        payload: {
          taskId,
          task: taskFromTaskInfoData(taskInfoData, taskId),
        },
      })
      for (const bufferedFrame of bufferedFrames) {
        handleSessionMessage(bufferedFrame, owner)
      }
      return
    }

    if (
      binding.present
      && binding.valid
      && retiredSessionTaskIdsRef.current.has(binding.taskId)
    ) {
      return
    }
    if (
      sessionConversationRef.current.phase === "replacement_sending"
      || sessionConversationRef.current.phase === "replacement_awaiting_task"
    ) {
      bufferSessionPreAdoptionFrame(message, owner)
      return
    }
    if (
      sessionConversationRef.current.phase === "reset_requested"
      || sessionConversationRef.current.phase === "replacement_ready"
      || sessionConversationRef.current.phase === "reload_required"
    ) return
    if (binding.present && binding.valid) {
      if (
        sessionTaskIdRef.current === null
      ) {
        return
      }
      if (sessionTaskIdRef.current !== binding.taskId) {
        requireSessionReload(
          new Error("Session task lineage changed without a reset; reload required.")
        )
        return
      }
    }

    handleMessage(message, dispatch, stateRef.current, {
      skipHistory: true,
      filesDisabled,
    })
  }

  useLayoutEffect(() => {
    sessionMessageHandlerRef.current = handleSessionMessage
  }, [handleSessionMessage])

  const getLLMIdsFromConfig = (config?: any) => {
    if (!config || !config.model) {
      return null
    }

    // Debug log to see what config is being passed
    console.log('getLLMIdsFromConfig called with:', config)

    // Always return exactly 4 elements in fixed order: [default, fast_small, vision, compact]
    // Use null for unconfigured models
    const llmIds = [
      config.model,                           // Default model (required)
      config.smallFastModel || null,         // Fast small model (optional)
      config.visualModel || null,            // Vision model (optional)
      config.compactModel || null            // Compact model (optional)
    ]

    return llmIds
  }

  const beginSessionMessageDelivery = useCallback(() => {
    if (!mountedRef.current) return
    const nextCount = messageDeliveryCountRef.current + 1
    messageDeliveryCountRef.current = nextCount
    setMessageDeliveryCount(nextCount)
  }, [])

  const endSessionMessageDelivery = useCallback(() => {
    const nextCount = Math.max(0, messageDeliveryCountRef.current - 1)
    messageDeliveryCountRef.current = nextCount
    if (mountedRef.current) setMessageDeliveryCount(nextCount)
  }, [])

  const sendMessage = useCallback(async (message: string, config?: any, files?: File[]) => {
    console.log('🚀 sendMessage called:', { message, files: files?.map(f => f.name), taskId: state.taskId })

    // A prior turn's DAG plan/steps must not linger into this turn - otherwise
    // the Progress panel would auto-open (or stay open) showing stale steps
    // from a different execution mode/run before this turn's own dag_execution
    // event (if any) arrives. But sending into a run that's still actively
    // going - answering a mid-run clarification, or the "live guidance" input
    // ChatInput allows while running/paused/waiting_for_user (see
    // ChatInput.tsx's `allowsLiveGuidanceInput`; the task page never passes
    // onSend/onSendInteraction, so all of these fall back to this same
    // sendMessage) - is a CONTINUATION of that run, not a new turn. Clearing
    // here would wipe out the in-progress DAG plan/steps the Progress panel
    // is actively showing.
    const isContinuingActiveRun =
      state.currentTask?.status === "running"
      || state.currentTask?.status === "paused"
      || state.currentTask?.status === "waiting_for_user"
    if (!isContinuingActiveRun) {
      dispatch({ type: "SET_DAG_EXECUTION", payload: null })
      dispatch({ type: "SET_STEPS", payload: [] })
    }

    if (sessionTransport && !mountedRef.current) {
      throw new Error("Message not sent: the Session chat is closed.")
    }
    if (
      filesDisabled
      && files
      && files.length > 0
    ) {
      throw new Error("Files are disabled for this conversation.")
    }
    if (sessionTransport && sessionConversationRef.current.phase === "reset_requested") {
      throw new Error(
        "Message delivery is blocked while conversation reset is pending."
      )
    }

    const clientMessageId = typeof config?.clientMessageId === 'string'
      ? config.clientMessageId
      : generateClientMessageId()
    const sessionDeliveryOwner =
      sessionTransport && sessionConnectionIdentityRef.current
        ? {
          connectionIdentity: sessionConnectionIdentityRef.current,
          deliveryGeneration: deliveryGenerationRef.current,
        }
        : null

    // Optimistic copy of the sender's own bubble, added once delivery is
    // acknowledged. The live user_message trace event only exists after the
    // agent run actually starts — a run refused before that (e.g. the quota
    // gate) never emits it, so without this copy the sender's message stays
    // invisible until a reload replays the persisted transcript row. The
    // reducer reconciles by turn id / content, keeping the persisted event
    // when it already arrived and replacing this copy when it arrives later.
    const addOptimisticUserMessage = (intendedTaskId: number | null) => {
      // Delivery acknowledgement can arrive after the user has navigated to a
      // different task. Never append this turn to whichever task happens to be
      // active when the async send resumes.
      if (sessionTransport) {
        if (
          !mountedRef.current
          || !sessionDeliveryOwner
          || sessionDeliveryOwner.connectionIdentity
            !== sessionConnectionIdentityRef.current
          || sessionDeliveryOwner.deliveryGeneration
            !== deliveryGenerationRef.current
        ) {
          return
        }
      } else if (
        intendedTaskId === null
        || stateRef.current.taskId !== intendedTaskId
      ) {
        return
      }
      let content: React.ReactNode = message
      if (files && files.length > 0) {
        content = (
          <UserMessageContent
            message={message}
            files={files.map(f => ({
              name: f.name,
              type: f.type,
              size: f.size,
            }))}
          />
        )
      }
      dispatch({
        type: "ADD_MESSAGE",
        payload: {
          id: userTurnMessageId(clientMessageId),
          role: "user",
          content: content,
          timestamp: Date.now().toString(),
          isOptimistic: true,
        }
      })
    }

    const targetTaskId = typeof config?.targetTaskId === 'number' ? config.targetTaskId : null
    if (
      sessionTransport
      && targetTaskId !== null
      && state.taskId !== targetTaskId
    ) {
      throw new Error(
        "The Session connection does not own the requested task."
      )
    }

    if (sessionTransport?.allowTasklessChat) {
      if (!sessionDeliveryOwner) {
        throw new Error(
          "Message not sent: the Session connection is not ready."
        )
      }

      if (sessionConversationRef.current.phase === "reload_required") {
        throw new Error("Conversation outcome is unknown; reload required.")
      }
      if (
        sessionConversationRef.current.phase === "replacement_sending"
        || sessionConversationRef.current.phase === "replacement_awaiting_task"
      ) {
        throw new Error("The replacement conversation is already starting.")
      }

      const startsReplacementConversation =
        sessionConversationRef.current.phase === "replacement_ready"
      const replacementSendStillOwned = () => {
        const lifecycle: SessionConversationState = sessionConversationRef.current
        return (
          mountedRef.current
          && sessionConnectionIdentityRef.current === sessionDeliveryOwner.connectionIdentity
          && deliveryGenerationRef.current === sessionDeliveryOwner.deliveryGeneration
          && lifecycle.phase === "replacement_sending"
        )
      }
      if (startsReplacementConversation) {
        const transition = dispatchSessionConversation({
          type: "SESSION_REPLACEMENT_SENDING",
          connectionIdentity: sessionDeliveryOwner.connectionIdentity,
        })
        if (!transition.accepted) {
          throw new Error("The replacement conversation is no longer available.")
        }
        beginSessionPreAdoptionBuffer(sessionDeliveryOwner.connectionIdentity)
      }
      beginSessionMessageDelivery()
      try {
        await sendChatMessage(
          message,
          undefined,
          config?.force,
          clientMessageId,
        )
        if (startsReplacementConversation) {
          if (!replacementSendStillOwned()) {
            return
          }
          const transition = dispatchSessionConversation({
            type: "SESSION_REPLACEMENT_ACCEPTED",
            connectionIdentity: sessionDeliveryOwner.connectionIdentity,
          })
          if (!transition.accepted) return
          activateSessionPreAdoptionBuffer(sessionDeliveryOwner.connectionIdentity)
        }
        addOptimisticUserMessage(state.taskId)
      } catch (error) {
        if (startsReplacementConversation) {
          if (!replacementSendStillOwned()) {
            throw error
          }
          const candidatePresent = sessionPreAdoptionBufferRef.current?.candidate != null
          const disposition = (
            error
            && typeof error === "object"
            && "disposition" in error
            && (
              error.disposition === "not_sent"
              || error.disposition === "rejected"
              || error.disposition === "outcome_unknown"
            )
          )
            ? error.disposition
            : "outcome_unknown"
          if (disposition === "outcome_unknown" || candidatePresent) {
            requireSessionReload(
              error instanceof Error
                ? error
                : new Error("Conversation outcome is unknown; reload required."),
            )
          } else {
            discardSessionPreAdoptionBuffer()
            dispatchSessionConversation({
              type: "SESSION_REPLACEMENT_REJECTED",
              connectionIdentity: sessionDeliveryOwner.connectionIdentity,
            })
          }
        }
        throw error
      } finally {
        endSessionMessageDelivery()
      }
      return
    }

    if (targetTaskId !== null && state.taskId !== targetTaskId) {
      await queuePendingMessage({
        message,
        files,
        targetTaskId,
        force: config?.force,
        clientMessageId,
      })
      addOptimisticUserMessage(targetTaskId)
      return
    }

    if (!state.taskId) {
      // Create a new task via API
      try {
        const apiUrl = getApiUrl()

        // Build internal model identifiers from config
        const llmIds = getLLMIdsFromConfig(config)

        // Note: Files will be uploaded via WebSocket after task creation
        // The backend TaskCreateRequest expects JSON with 'files' as a list of filenames (strings)
        // Since we haven't uploaded files yet, we don't include them in the task creation request
        const executionMode = config?.executionMode?.mode || (config?.agentId ? "balanced" : undefined)

        const requestBody: any = {
          title: message,
          description: message,
          memory_similarity_threshold: config?.memorySimilarityThreshold ?? 1.5,
        }
        if (executionMode) {
          requestBody.execution_mode = executionMode
        }

        // Upload files first if present
        if (files && files.length > 0) {
          const filesToUpload = files.filter(f => !(f as any).file_id)
          const uploadedFileIds = files.filter(f => (f as any).file_id).map(f => (f as any).file_id)

          if (filesToUpload.length > 0) {
            const formData = new FormData()
            filesToUpload.forEach(f => formData.append('files', f))
            formData.append('task_type', executionMode ?? 'general')

            try {
              const uploadResponse = await apiRequest(`${getUploadApiUrl()}/api/files/upload`, {
                method: 'POST',
                body: formData
              })

              const parsed = await parseApiResponse(uploadResponse)

              if (uploadResponse.ok && isJsonRecord(parsed.data)) {
                const uploadData = parsed.data
                if (uploadData.success && Array.isArray(uploadData.files)) {
                  uploadData.files
                    .filter((f): f is { file_id: string } => isJsonRecord(f) && typeof f.file_id === 'string')
                    .forEach(f => uploadedFileIds.push(f.file_id))
                }
              } else {
                throw new Error(getUploadErrorMessage(uploadResponse, parsed, {
                  generic: t("files.uploadFailed") || "Upload failed",
                  ...UPLOAD_ERROR_MESSAGES,
                }))
              }
            } catch (e) {
              console.error('Error uploading files before task creation:', e)
              throw e
            }
          }

          if (uploadedFileIds.length > 0) {
            requestBody.files = uploadedFileIds
          }
        }

        // Agent chats should use the published agent's own model configuration.
        if (llmIds && !config?.agentId) {
          requestBody.llm_ids = llmIds
        }

        if (config?.executionMode?.processDescription) {
          requestBody.process_description = config.executionMode.processDescription
        }
        if (config?.executionMode?.examples) {
          requestBody.examples = config.executionMode.examples
        }
        if (config?.agentId) {
          requestBody.agent_id = config.agentId
        }
        if (config?.agentType) {
          requestBody.agent_type = config.agentType
        }
        if (config?.agentConfig) {
          requestBody.agent_config = config.agentConfig
        }
        if (config?.runtimeExtensions) {
          requestBody.runtime_extensions = config.runtimeExtensions
        }

        const response = await apiRequest(`${apiUrl}/api/chat/task/create`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
        })

        if (response.ok) {
          const taskData = await response.json()
          const newTaskId = taskData.task_id

          console.log('✅ Task created successfully:', {
            taskId: newTaskId,
            taskIdType: typeof newTaskId,
            taskData: taskData,
            status: taskData.status
          })

          console.log('🎯 About to call setTaskId with payload:', newTaskId)
          setTaskId(newTaskId)
          console.log('🎯 setTaskId completed')

          dispatch({
            type: "SET_TASK_RUNTIME_EXTENSIONS",
            payload: {
              taskId: newTaskId,
              extensions: normalizeTaskRuntimeExtensions(taskData.runtime_extensions),
            },
          })

          // Create a new task from response
          const newTask: Task = {
            id: newTaskId.toString(),
            title: taskData.title,
            status: normalizeTaskStatus(taskData.status) || "pending",
            description: taskData.description || message,
            createdAt: taskData.created_at,
            updatedAt: taskData.updated_at,
            modelId: taskData.model_id,
            smallFastModelId: taskData.small_fast_model_id,
            visualModelId: taskData.visual_model_id,
            compactModelId: taskData.compact_model_id,
            modelName: taskData.model_name || taskData.modelName, // API response field
            smallFastModelName: taskData.small_fast_model_name || taskData.smallFastModelName, // API response field
            visualModelName: taskData.visual_model_name || taskData.visual_model_name,
            compactModelName: taskData.compact_model_name || taskData.compact_model_name,
            executionMode: taskData.execution_mode,
            isDag: taskData.is_dag,
            agentId: taskData.agent_id,
            agentName: taskData.agent_name,
            agentLogoUrl: taskData.agent_logo_url,
            waitingQuestion: taskData.waiting_question,
            waitingInteractions: normalizeInteractions(taskData.waiting_interactions),
            runId: taskData.run_id,
            stateVersion: taskData.state_version,
            controlState: taskData.control_state,
          }
          dispatch({ type: "SET_CURRENT_TASK", payload: newTask })
          dispatch({ type: "TRIGGER_TASK_UPDATE" })

          // User message will be handled by backend via trace event

          // For new tasks, always send chat message to support file uploads
          console.log('💬 Queuing chat message for new task:', {
            taskId: newTaskId,
            taskStatus: taskData.status,
            hasFiles: files && files.length > 0
          })

          // Do not clear the composer until the newly connected task socket
          // confirms that the message was durably accepted.
          await queuePendingMessage({
            message,
            files,
            targetTaskId: newTaskId,
            force: config?.force,
            clientMessageId,
          })
          addOptimisticUserMessage(newTaskId)
        } else {
          const parsed = await parseApiResponse(response)
          const errorMessage = getApiErrorMessage(
            response,
            parsed,
            t("builds.list.chat.sendFailed") || "Failed to create task",
          )
          console.error('Failed to create task:', errorMessage)
          throw new Error(errorMessage)
        }
      } catch (error) {
        console.error('Error creating task:', error)
        throw error
      }
    }

    // For existing tasks (when task already exists)
    if (state.taskId) {
      console.log('🚀 AppContext sendMessage - sending chat message:', {
        message,
        files: files?.map(f => f.name) || [],
        hasFiles: files && files.length > 0,
        taskId: state.taskId
      })

      // Wait for the server's durable-delivery acknowledgement. If the socket
      // is disconnected or the backend rejects the turn, this throws and the
      // composer keeps both its text and attached files.
      await sendChatMessage(message, files, config?.force, clientMessageId)

      if (state.currentTask?.status === 'completed' || state.currentTask?.status === 'failed') {
        dispatch({
          type: "UPDATE_TASK_STATUS",
          payload: { status: 'running' }
        })
        dispatch({ type: "TRIGGER_TASK_UPDATE" })
      }

      addOptimisticUserMessage(state.taskId)
    }
  }, [
    beginSessionMessageDelivery,
    beginSessionPreAdoptionBuffer,
    activateSessionPreAdoptionBuffer,
    discardSessionPreAdoptionBuffer,
    dispatchSessionConversation,
    endSessionMessageDelivery,
    filesDisabled,
    queuePendingMessage,
    sendChatMessage,
    sessionTransport,
    state.currentTask?.status,
    state.taskId,
  ])

  const startNewConversation = useCallback((): Promise<void> => {
    const existingReset = sessionResetFlightRef.current
    if (existingReset) return existingReset.promise
    if (sessionConversationRef.current.phase === "reload_required") {
      return Promise.reject(
        new Error("Conversation outcome is unknown; reload required.")
      )
    }
    if (!sessionTransport?.supportsConversationReset) {
      return Promise.reject(
        new Error("Conversation reset is not supported by this transport.")
      )
    }
    if (!mountedRef.current) {
      return Promise.reject(
        new Error("Conversation reset is unavailable after chat is closed.")
      )
    }
    if (messageDeliveryCountRef.current > 0) {
      return Promise.reject(
        new Error(
          "Conversation reset is blocked while message delivery is pending."
        )
      )
    }
    const establishedSessionTaskId = sessionTaskIdRef.current
    if (establishedSessionTaskId === null) {
      return Promise.reject(
        new Error("Conversation reset requires an established Session task.")
      )
    }
    if (sessionConversationRef.current.phase !== "bound") {
      return Promise.reject(
        new Error(
          "Start the replacement conversation before resetting again."
        )
      )
    }

    const connectionIdentity = sessionConnectionIdentityRef.current
    if (!connectionIdentity || !isConnected) {
      return Promise.reject(
        new Error("Conversation reset requires a connected Session.")
      )
    }

    let resolveReset!: () => void
    let rejectReset!: (error: Error) => void
    const promise = new Promise<void>((resolve, reject) => {
      resolveReset = resolve
      rejectReset = reject
    })
    const resetFlight: SessionResetFlight = {
      connectionIdentity,
      deliveryGeneration: deliveryGenerationRef.current,
      promise,
      resolve: resolveReset,
      reject: rejectReset,
      timeout: setTimeout(() => {
        if (sessionResetFlightRef.current !== resetFlight) return
        requireSessionReload(
          new Error("Conversation reset acknowledgement timed out; reload required.")
        )
      }, SESSION_RESET_ACK_TIMEOUT_MS),
    }
    sessionResetFlightRef.current = resetFlight
    const transition = dispatchSessionConversation({
      type: "SESSION_RESET_REQUESTED",
      connectionIdentity,
      taskId: establishedSessionTaskId,
    })
    if (!transition.accepted) {
      rejectSessionResetFlight(
        resetFlight,
        new Error("Conversation reset is no longer available."),
      )
      return promise
    }

    try {
      const sent = sendRawMessage({ type: "new_conversation" })
      if (sent !== "sent") {
        dispatchSessionConversation({
          type: "SESSION_RESET_NOT_SENT",
          connectionIdentity,
        })
        rejectSessionResetFlight(
          resetFlight,
          new Error("Conversation reset was not sent; retry the request."),
        )
      }
    } catch (error) {
      const resetError =
        error instanceof Error ? error : new Error(String(error))
      requireSessionReload(resetError)
    }
    return promise
  }, [
    isConnected,
    dispatchSessionConversation,
    rejectSessionResetFlight,
    requireSessionReload,
    sendRawMessage,
    sessionTransport?.supportsConversationReset,
  ])

  // Initialize the replay scheduler function
  const initializeReplayScheduler = useCallback(() => {
    // Get cached events
    const cachedEvents = state.replayEventCache

    if (cachedEvents.length === 0) {
      return
    }

    // Convert WebSocket messages to replay events
    const replayEvents = cachedEvents.map((wsMessage, index) => ({
      type: 'ws_message' as const,
      data: wsMessage,
      timestamp: wsMessage.timestamp,
      originalIndex: index
    }))

    // Create and configure the replay scheduler
    const scheduler = new ReplayScheduler(
      (event) => {
        // Process the original message using the existing message handling logic
        // but with isReplaying: false to ensure it gets processed for display
        const message = event.data as WebSocketMessage
        const tempState = { ...stateRef.current, isReplaying: false }
        handleMessage(message, dispatch, tempState, { filesDisabled })
      },
      () => {
        // Replay completed
        dispatch({ type: "STOP_REPLAY" })
      },
      true // Skip user message delays by default
    )

    // Set the events and configure the scheduler
    scheduler.setEvents(replayEvents)
    scheduler.setPlaybackSpeed(state.replaySpeed)

    // Store the scheduler in state
    dispatch({ type: "SET_REPLAY_SCHEDULER", payload: scheduler })

    // Always start the scheduler since this function is called when we want to replay
    scheduler.play()
  }, [state.isReplaying, state.replayTaskId, state.replayEventCache, state.replaySpeed, dispatch, filesDisabled])

  const executeTask = useCallback((description: string) => {
    if (!state.taskId) return
    wsExecuteTask(description)
  }, [state.taskId, wsExecuteTask])

  const pauseTask = useCallback(() => {
    if (!state.taskId) return
    wsPauseTask()
  }, [state.taskId, wsPauseTask])

  const resumeTask = useCallback(() => {
    if (!state.taskId) return
    wsResumeTask()
  }, [state.taskId, wsResumeTask])

  const selectStep = useCallback((stepId: string | null) => {
    dispatch({ type: "SELECT_STEP", payload: stepId })
  }, [dispatch])

  const clearMessages = useCallback(() => {
    dispatch({ type: "CLEAR_MESSAGES" })
  }, [dispatch])

  const setTaskId = useCallback((taskId: number | null, options?: { navigate?: boolean }) => {
    // Only reset historical data request flag when changing to a different task
    if (taskId !== stateRef.current.taskId) {
      if (taskId) {
        historicalDataRequestMapRef.current.set(taskId, false)
      }
      // Clear recentMessages cache when switching tasks to prevent false duplicates
      recentMessagesRef.current.clear()
      isHistoricalDataLoadingRef.current = false

      // Clear existing data immediately when switching tasks to prevent stale data display
      // This fixes the issue where messages from the previous task might be cleared
      // by the connection effect AFTER the new task's history has already arrived.
      dispatch({ type: "CLEAR_MESSAGES" })
      dispatch({ type: "SET_TRACE_EVENTS", payload: [] })
      dispatch({ type: "SET_STEPS", payload: [] })
      // Also clear the previous task's DAG plan/phase - otherwise switching
      // tasks via the sidebar (no page reload) leaves the Progress panel
      // showing the OLD task's steps and elapsed time until the new task's
      // own dag_execution history replay arrives (or the page is refreshed,
      // which happens to force a clean re-init).
      dispatch({ type: "SET_DAG_EXECUTION", payload: null })
    }

    if (options?.navigate !== false) {
      // Update URL to use dynamic route for task detail page
      if (taskId) {
        router.push(`/task/${taskId}`)
      } else {
        router.push('/task')
      }
    }

    dispatch({ type: "SET_TASK_ID", payload: taskId })
    // Set history loading state immediately when switching tasks to prevent empty state flash
    if (taskId) {
      isHistoricalDataLoadingRef.current = true
      dispatch({ type: "SET_HISTORY_LOADING", payload: true })
    }
  }, [dispatch, router])

  const openFilePreview = useCallback((fileId: string, fileName: string, files?: Array<{ fileId: string; fileName: string }>, index?: number) => {
    if (filesDisabled) return
    console.log('🎯 openFilePreview called:', {
      fileId,
      fileName,
      files: files,
      filesLength: files?.length,
      index
    })
    dispatch({ type: "OPEN_FILE_PREVIEW", payload: { fileId, fileName, files, index } })
  }, [dispatch, filesDisabled])

  const switchFilePreview = useCallback((index: number) => {
    if (filesDisabled) return
    const { availableFiles } = state.filePreview
    if (index >= 0 && index < availableFiles.length) {
      const file = availableFiles[index]
      dispatch({ type: "SWITCH_FILE_PREVIEW", payload: { fileId: file.fileId, fileName: file.fileName, index } })
    }
  }, [filesDisabled, state.filePreview.availableFiles])

  const closeFilePreview = useCallback(() => {
    dispatch({ type: "CLOSE_FILE_PREVIEW" })
  }, [dispatch])

  const getFilePreviewUrl = useCallback((fileId: string) => {
    if (filesDisabled) {
      throw new Error("Files are disabled for this conversation.")
    }
    if (transport?.fileAccess) {
      return transport.fileAccess.previewUrl(fileId)
    }
    return `${getApiUrl()}/api/files/preview/${encodeURIComponent(fileId)}`
  }, [filesDisabled, transport])

  const getFileDownloadUrl = useCallback((fileId: string) => {
    if (filesDisabled) {
      throw new Error("Files are disabled for this conversation.")
    }
    if (transport?.fileAccess) {
      return transport.fileAccess.downloadUrl(fileId)
    }
    return `${getApiUrl()}/api/files/download/${encodeURIComponent(fileId)}`
  }, [filesDisabled, transport])


  // Replay control methods
  const startReplay = useCallback((taskId: number, events: TraceEvent[]) => {
    dispatch({ type: "START_REPLAY", payload: { taskId, events } })
  }, [dispatch])

  const stopReplay = useCallback(() => {
    dispatch({ type: "STOP_REPLAY" })
  }, [dispatch])

  const setReplayPlaying = useCallback((isPlaying: boolean) => {
    dispatch({ type: "SET_REPLAY_PLAYING", payload: isPlaying })
  }, [dispatch])

  const setReplaySpeed = useCallback((speed: number) => {
    dispatch({ type: "SET_REPLAY_SPEED", payload: speed })
  }, [dispatch])

  const setReplayProgress = useCallback((progress: number) => {
    dispatch({ type: "SET_REPLAY_PROGRESS", payload: progress })
  }, [dispatch])

  useLayoutEffect(() => {
    startDelayedPlaybackRef.current = initializeReplayScheduler
  }, [initializeReplayScheduler])

  return (
    <AgentCardPresentationCapability.Provider value={agentCardsEnabled}>
      <LinksOpenInNewTabCapability.Provider value={linksOpenInNewTab}>
        <FileAccessProvider policy={transport?.fileAccess}>
          <AppContext.Provider
          value={{
          state,
          dispatch,
          filesDisabled,
          agentCardsEnabled,
          voiceInputEnabled,
          taskControlsEnabled,
          sendMessage,
          executeTask,
          pauseTask,
          resumeTask,
          selectStep,
          clearMessages,
          isConnected,
          connectionError,
          startNewConversation,
          isConversationResetPending:
            state.sessionConversation.phase === "reset_requested",
          isMessageDeliveryPending: messageDeliveryCount > 0,
          isSessionInteractionLocked:
            state.sessionConversation.phase === "reload_required",
          sessionConversationState: state.sessionConversation.phase,
          setTaskId,
          requestStatus,
          getFilePreviewUrl,
          getFileDownloadUrl,
          openFilePreview,
          switchFilePreview,
          closeFilePreview,
          startReplay,
          stopReplay,
          setReplayPlaying,
          setReplaySpeed,
          setReplayProgress,
          setPendingMessage,
          }}
        >
          {children}
          </AppContext.Provider>
        </FileAccessProvider>
      </LinksOpenInNewTabCapability.Provider>
    </AgentCardPresentationCapability.Provider>
  )
}

export function useApp() {
  const context = useContext(AppContext)
  if (context === undefined) {
    throw new Error("useApp must be used within an AppProvider")
  }
  return context
}

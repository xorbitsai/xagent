"use client"

import React, { createContext, useContext, useReducer, useCallback, useEffect, useState, useRef, useMemo } from "react"
import { useRouter } from "next/navigation"
import { FileText, Target, Zap, CheckCircle, XCircle, Wrench, Activity, Search, Lightbulb, AlertTriangle, Info, Brain, Bot } from "lucide-react"
import { JsonRenderer } from "../components/ui/markdown-renderer"
import { FileAttachment } from "../components/file-attachment"
import { ReplayScheduler } from '@/lib/replay-scheduler'
import { CollapsibleSection } from "../components/collapsible-section"
import { Badge } from "@/components/ui/badge"

interface WebSocketMessage {
  type: string
  data: unknown
  timestamp: string
  task_id?: number
  step_id?: string
  event_type?: string
  event_id?: string
}
import { useWebSocket } from "@/hooks/use-websocket"
import { useAuth } from "@/contexts/auth-context"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useI18n } from "@/contexts/i18n-context"

// Unique ID generator for messages
let messageIdCounter = 0
const generateMessageId = (prefix: string) => {
  return `${prefix}-${++messageIdCounter}-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`
}

// Simple deduplication for all messages
const recentMessages = new Set<string>()

// Helper function to compare arrays
const arraysEqual = (a: string[], b: string[]): boolean => {
  if (a === b) return true
  if (a == null || b == null) return false
  if (a.length !== b.length) return false
  return a.every((val, index) => val === b[index])
}

// Function to clear duplicate message cache
const clearDuplicateMessageCache = () => {
  recentMessages.clear()
}

// Function to start delayed playback
let startDelayedPlayback = () => {
  // Will be initialized later
}

// Expose to window for global access
if (typeof window !== 'undefined') {
  ;(window as any).clearDuplicateMessageCache = clearDuplicateMessageCache
}
// Flag to track if we're loading historical data
let isHistoricalDataLoading = false
// Store pending task info for auto-execution after historical data loads
let pendingTaskToExecute: { description: string } | null = null
const isDuplicateMessage = (content: string | React.ReactNode, type: string = 'general', force: boolean = false) => {
  // Convert React element to string representation for comparison
  let contentStr: string
  if (typeof content === 'string') {
    contentStr = content.trim()
  } else if (React.isValidElement(content)) {
    // For React elements, extract text content more comprehensively
    const extractTextFromReactNode = (node: React.ReactNode): string => {
      if (typeof node === 'string') return node
      if (typeof node === 'number') return node.toString()
      if (Array.isArray(node)) return node.map(extractTextFromReactNode).join('')
      if (React.isValidElement(node) && node.props.children) {
        return extractTextFromReactNode(node.props.children)
      }
      return ''
    }
    contentStr = extractTextFromReactNode(content).trim()
  } else {
    contentStr = ''
  }

  const key = `${type}:${contentStr}`
  if (recentMessages.has(key)) {
    return true
  }
  recentMessages.add(key)
  // Clean up old messages after 30 seconds
  setTimeout(() => {
    recentMessages.delete(key)
  }, 30000)
  return false
}

// Backward compatibility for result messages
const isDuplicateResult = (content: string) => {
  return isDuplicateMessage(content, 'result')
}


interface Message {
  id: string
  role: "user" | "assistant"
  content: string | React.ReactNode
  timestamp: string
  status?: "pending" | "running" | "completed" | "failed"
  isResult?: boolean
  isFileOutput?: boolean
  traceEvents?: TraceEvent[]
}

interface Task {
  id: string
  title: string
  status: "pending" | "running" | "completed" | "failed" | "paused"
  description: string
  createdAt: string | number
  updatedAt: string | number
  // Model configuration
  modelName?: string
  smallFastModelName?: string
  visualModelName?: string
  compactModelName?: string
  vibeMode?: "task" | "process"
  isDag?: boolean
  agentId?: number
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

interface DAGExecution {
  phase: "planning" | "executing" | "completed" | "failed"
  current_plan: Record<string, unknown>
  created_at: string | number
  updated_at: string | number
}

interface AppState {
  messages: Message[]
  currentTask: Task | null
  dagExecution: DAGExecution | null
  steps: StepExecution[]
  traceEvents: TraceEvent[]
  selectedStepId: string | null
  isProcessing: boolean
  taskId: number | null
  filePreview: {
    isOpen: boolean
    filePath: string
    fileName: string
    content: string
    isLoading: boolean
    error: string | null
    // 支持多个文件预览切换
    availableFiles: Array<{ filePath: string; fileName: string }>
    currentIndex: number
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
}

type AppAction =
  | { type: "SET_TASK_ID"; payload: number | null }
  | { type: "ADD_MESSAGE"; payload: Message }
  | { type: "SET_CURRENT_TASK"; payload: Task }
  | { type: "UPDATE_TASK_STATUS"; payload: { status: Task["status"] } }
  | { type: "TRIGGER_TASK_UPDATE" }
  | { type: "SET_DAG_EXECUTION"; payload: DAGExecution | null }
  | { type: "ADD_STEP"; payload: StepExecution }
  | { type: "UPDATE_STEP"; payload: { stepId: string; updates: Partial<StepExecution> } }
  | { type: "SET_STEPS"; payload: StepExecution[] }
  | { type: "ADD_TRACE_EVENT"; payload: TraceEvent }
  | { type: "SET_TRACE_EVENTS"; payload: TraceEvent[] }
  | { type: "SELECT_STEP"; payload: string | null }
  | { type: "SET_PROCESSING"; payload: boolean }
  | { type: "CLEAR_MESSAGES"; payload?: { keepMessageId?: string | null } }
  | { type: "RESET_STATE" }
  | { type: "OPEN_FILE_PREVIEW"; payload: { filePath: string; fileName: string; files?: Array<{ filePath: string; fileName: string }>; index?: number } }
  | { type: "CLOSE_FILE_PREVIEW" }
  | { type: "SWITCH_FILE_PREVIEW"; payload: { filePath: string; fileName: string; index: number } }
  | { type: "SET_FILE_PREVIEW_CONTENT"; payload: { content: string; error: string | null } }
  | { type: "SET_FILE_PREVIEW_LOADING"; payload: boolean }
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

const initialState: AppState = {
  messages: [],
  currentTask: null,
  dagExecution: null,
  steps: [],
  traceEvents: [],
  selectedStepId: null,
  isProcessing: false,
  taskId: null,
  filePreview: {
    isOpen: false,
    filePath: '',
    fileName: '',
    content: '',
    isLoading: false,
    error: null,
    availableFiles: [],
    currentIndex: 0,
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
}

function appReducer(state: AppState, action: AppAction): AppState {
  console.log('🔍 Reducer called with action:', action.type, action)

  switch (action.type) {
    case "SET_HISTORY_LOADING":
      return { ...state, isHistoryLoading: action.payload }

    case "TRIGGER_TASK_UPDATE":
      return { ...state, lastTaskUpdate: Date.now() }

    case "SET_TASK_ID":
      console.log('🔄 Reducer SET_TASK_ID:', {
        currentTaskId: state.taskId,
        newTaskId: action.payload,
        payloadType: typeof action.payload
      })
      const newState = { ...state, taskId: action.payload }
      console.log('🔄 Reducer returning new state:', newState)
      return newState

    case "ADD_MESSAGE": {
      const newMessage = action.payload
      let messageToAdd = newMessage
      let newTraceEvents = state.traceEvents

      if (newMessage.role === "assistant" && newMessage.isResult) {
        messageToAdd = {
          ...newMessage,
          traceEvents: [...state.traceEvents]
        }
        newTraceEvents = []
      }

      const updatedMessages = [...state.messages, messageToAdd]
      updatedMessages.sort((a, b) => {
        const timeA = typeof a.timestamp === 'number' ? a.timestamp : new Date(a.timestamp).getTime()
        const timeB = typeof b.timestamp === 'number' ? b.timestamp : new Date(b.timestamp).getTime()
        return timeA - timeB
      })
      return { ...state, messages: updatedMessages, traceEvents: newTraceEvents }
    }

    case "SET_CURRENT_TASK":
      return { ...state, currentTask: action.payload }

    case "UPDATE_TASK_STATUS":
      return state.currentTask
        ? {
            ...state,
            currentTask: {
              ...state.currentTask,
              status: action.payload.status,
              updatedAt: new Date().toISOString(),
            },
          }
        : state

    case "SET_DAG_EXECUTION":
      return { ...state, dagExecution: action.payload }

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
      if (action.payload?.keepMessageId) {
        return {
          ...state,
          messages: state.messages.filter(m => m.id === action.payload?.keepMessageId)
        }
      }
      return { ...state, messages: [] }

    case "RESET_STATE":
      return initialState

    case "OPEN_FILE_PREVIEW":
      // 支持传入单个文件或多个文件列表
      const files = action.payload.files || [{ filePath: action.payload.filePath, fileName: action.payload.fileName }]
      const currentIndex = action.payload.index || 0

      return {
        ...state,
        filePreview: {
          ...state.filePreview,
          isOpen: true,
          filePath: files[currentIndex]?.filePath || action.payload.filePath,
          fileName: files[currentIndex]?.fileName || action.payload.fileName,
          content: '',
          isLoading: true,
          error: null,
          availableFiles: files,
          currentIndex: currentIndex,
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
          filePath: action.payload.filePath,
          fileName: action.payload.fileName,
          content: '',
          isLoading: true,
          error: null,
          currentIndex: action.payload.index,
        }
      }

    case "SET_FILE_PREVIEW_CONTENT":
      return {
        ...state,
        filePreview: {
          ...state.filePreview,
          content: action.payload.content,
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

interface AppContextType {
  state: AppState
  dispatch: React.Dispatch<AppAction>
  sendMessage: (message: string, config?: any, files?: File[]) => void
  executeTask: (description: string) => void
  pauseTask: () => void
  resumeTask: () => void
  selectStep: (stepId: string | null) => void
  clearMessages: () => void
  isConnected: boolean
  connectionError: Error | null
  setTaskId: (taskId: number | null) => void
  requestStatus: () => void
  openFilePreview: (filePath: string, fileName: string, files?: Array<{ filePath: string; fileName: string }>, index?: number) => void
  switchFilePreview: (index: number) => void
  closeFilePreview: () => void
  startReplay: (taskId: number, events: TraceEvent[]) => void
  stopReplay: () => void
  setReplayPlaying: (isPlaying: boolean) => void
  setReplaySpeed: (speed: number) => void
  setReplayProgress: (progress: number) => void
}

const AppContext = createContext<AppContextType | undefined>(undefined)

// Global ref to track historical data requests per task ID
const historicalDataRequestMap = new Map<number, boolean>()

export function AppProvider({ children, token }: { children: React.ReactNode; token?: string }) {
  const [state, dispatch] = useReducer(appReducer, initialState)
  const [pendingMessage, setPendingMessage] = useState<{ message: string; files?: File[] } | null>(null)
  const { token: authToken } = useAuth() // Get auth token from context
  const { t } = useI18n()
  const router = useRouter()
  const pendingOptimisticMessageId = useRef<string | null>(null)

  // Ref to track current state for WebSocket message handler
  const stateRef = useRef(state)
  stateRef.current = state

  const {
    isConnected,
    connectionError,
    sendChatMessage,
    executeTask: wsExecuteTask,
    pauseTask: wsPauseTask,
    resumeTask: wsResumeTask,
    requestStatus,
    connect,
  } = useWebSocket({
    taskId: state.taskId || undefined,
    token,
    onMessage: (message) => {
      handleMessage(message, dispatch, stateRef.current)
    },
    onConnect: () => {
      // Fix: If we should be in replay mode but got disconnected, restore replay state
      if (state.replayTaskId && state.taskId === state.replayTaskId && !state.isReplaying) {
        dispatch({ type: "SET_REPLAY_PLAYING", payload: true })
      }

      if (pendingMessage) {
        console.log('📤 Sending pending message:', {
          message: pendingMessage.message,
          hasFiles: pendingMessage.files && pendingMessage.files.length > 0
        })
        sendChatMessage(pendingMessage.message, pendingMessage.files)
        setPendingMessage(null)
      }

      // Auto-execute PENDING tasks from Agent Builder
      // Wait a bit for state to sync after receiving task_info event
      setTimeout(() => {
        if (pendingTaskToExecute) {
          const hasUserMessages = stateRef.current.messages.some(m => m.role === 'user')
          console.log('🔍 onConnect - checking auto-execute:', {
            hasPendingTask: !!pendingTaskToExecute,
            pendingDescription: pendingTaskToExecute.description,
            hasUserMessages,
          })

          if (!hasUserMessages) {
            console.log('🚀 Auto-executing PENDING task from Agent Builder (onConnect):', pendingTaskToExecute.description)
            sendChatMessage(pendingTaskToExecute.description, [])
            pendingTaskToExecute = null
          } else {
            console.log('⏭️ Skipping auto-execute, already has user messages')
            pendingTaskToExecute = null
          }
        }
      }, 1000)
    },
    autoConnect: true,
  })

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

  // Debug taskId value
  // useEffect(() => {
  //   console.log('🎯 Debug taskId:', {
  //     stateTaskId: state.taskId,
  //     stateTaskIdType: typeof state.taskId,
  //     stateTaskIdIsNull: state.taskId === null,
  //     stateTaskIdIsUndefined: state.taskId === undefined,
  //     finalTaskId: state.taskId !== null && state.taskId !== undefined ? state.taskId : undefined
  //   })
  // }, [state.taskId])

  // Manually trigger connection when taskId is set
  useEffect(() => {
    if (state.taskId && !isConnected && !connectionError) {
      console.log('🔧 Manually triggering WebSocket connection for task:', state.taskId)
      connect()
    }
  }, [state.taskId, isConnected, connectionError, connect])

  const handleMessage = useCallback((message: WebSocketMessage, dispatch: React.Dispatch<AppAction>, currentState: AppState) => {
    // If we're in replay mode, don't process immediately - collect for delayed playback
    if (currentState.isReplaying) {
      // Add to replay cache
      dispatch({ type: "ADD_TO_REPLAY_CACHE", payload: message })

      // If this is historical_data_complete, start the delayed playback
      const isHistoricalComplete = message.type === "historical_data_complete" ||
          (message.type === "trace_event" && (message as any).event_type === "historical_data_complete")

      if (isHistoricalComplete) {
        // Add a small delay to ensure all events are collected before starting playback
        setTimeout(() => {
          startDelayedPlayback()
        }, 500) // 500ms delay to collect remaining events
      }

      return
    }

    // Normal message processing when not in replay mode
    switch (message.type) {
      case "chat":
        const chatData = message as any
        const messageContent = chatData.message || ""

        if (!isDuplicateMessage(messageContent, 'user-message')) {
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
        const traceEventData = message.data as any

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

          // Handle structured trace events
          if (eventType === "task_info") {
            const taskData = eventData
            console.log('📥 Received task_info event:', {
              taskData,
              status: taskData.status,
              statusType: typeof taskData.status
            })

            // Store pending task for auto-execution
            if (taskData.status === 'pending' && taskData.description) {
              pendingTaskToExecute = { description: taskData.description }
              console.log('💾 Stored pending task for auto-execution:', taskData.description)
            }

            dispatch({
              type: "SET_CURRENT_TASK",
              payload: {
                id: taskData.id.toString(),
                title: taskData.title,
                description: taskData.description,
                status: taskData.status,
                createdAt: taskData.created_at,
                updatedAt: taskData.updated_at,
                modelName: taskData.model_name,
                smallFastModelName: taskData.small_fast_model_name,
                visualModelName: taskData.visual_model_name,
                compactModelName: taskData.compact_model_name,
                vibeMode: taskData.vibe_mode,
                isDag: taskData.is_dag,
                agentId: taskData.agent_id,
              }
            })

            // Check if this is a new task (created within last 5 seconds)
            // If so, we don't expect historical messages, so stop loading
            const createdAt = typeof taskData.created_at === 'number'
              ? (taskData.created_at > 10000000000 ? taskData.created_at : taskData.created_at * 1000) // Handle ms vs s
              : new Date(taskData.created_at).getTime()

            // We do NOT stop loading here for new tasks anymore.
            // We wait for the user_message event or the timeout to handle it.
            // This prevents the empty state flash when task_info arrives before user_message.
          } else if (eventType === "dag_execution") {
            dispatch({ type: "SET_HISTORY_LOADING", payload: false })
            dispatch({ type: "SET_DAG_EXECUTION", payload: eventData })
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
            dispatch({ type: "SET_HISTORY_LOADING", payload: false })
            const messageContent = eventData.message || eventData.content || ""

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

            // Check if this is a duplicate message
            const isDuplicate = isDuplicateMessage(messageContent, 'user-message')
            console.log('🔍 Duplicate check:', {
              messageContent,
              isDuplicate,
              recentMessages: Array.from(recentMessages)
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
                <div className="space-y-2">
                  <div>{messageContent}</div>
                  <FileAttachment
                    files={files}
                    variant="user-message"
                    onPreview={(file) => {
                      dispatch({
                        type: "OPEN_FILE_PREVIEW",
                        payload: {
                          filePath: file.path || "",
                          fileName: file.name,
                          files: files.map((f: any) => ({ filePath: f.path || "", fileName: f.name })),
                          index: files.findIndex((f: any) => f.path === file.path)
                        }
                      })
                    }}
                  />
                </div>
              )
            }

            console.log('📤 Dispatching user message:', {
              content,
              filesCount: files.length,
              timestamp: message.timestamp,
              messageId: generateMessageId("msg-user")
            })

            const messagePayload = {
              id: generateMessageId("msg-user"),
              role: "user" as const,
              content: content,
              timestamp: message.timestamp,
            }

            console.log('📤 Message payload:', messagePayload)

            dispatch({
              type: "ADD_MESSAGE",
              payload: messagePayload
            })

            console.log('✅ User message dispatched successfully')
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
            if (!state.dagExecution || state.dagExecution.phase === "planning") {
              const dagExecution: DAGExecution = {
                phase: phase as "planning" | "executing" | "completed" | "failed",
                current_plan: {},
                created_at: message.timestamp,
                updated_at: message.timestamp,
              }

              // Use consistent string format for deduplication
              const dedupKey = `plan-start:${phase}`
              if (!isDuplicateMessage(dedupKey, 'dag-plan-start')) {
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
                {state.planMemoryInfo && (
                  <div className="mt-2">
                    <CollapsibleSection
                      title={t('agent.planDetails.memory.title')}
                      icon={<Brain className="h-4 w-4" />}
                      badge={t('agent.planDetails.badge.memory')}
                    >
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div className="flex items-center gap-1 p-2 bg-muted/30 rounded">
                          <Search className="h-3 w-3" />
                              <span>{t('agent.planDetails.memory.stats.found', { count: state.planMemoryInfo.memoriesFound })}</span>
                        </div>
                        <div className="flex items-center gap-1 p-2 bg-muted/30 rounded">
                          <Target className="h-3 w-3" />
                              <span>{t('agent.planDetails.memory.stats.used', { count: state.planMemoryInfo.memoriesUsed })}</span>
                        </div>
                      </div>
                      {state.planMemoryInfo.enhancedGoal && (
                        <div className="mt-2">
                          <div className="text-xs font-medium text-muted-foreground mb-1">{t('agent.planDetails.memory.enhancedGoalTitle')}</div>
                          <div className="text-xs bg-blue-500/10 p-2 rounded border border-blue-500/20">
                            {state.planMemoryInfo.enhancedGoal}
                          </div>
                        </div>
                      )}
                      {state.planMemoryInfo.memories && state.planMemoryInfo.memories.length > 0 && (
                        <div className="mt-2">
                          <div className="text-xs font-medium text-muted-foreground mb-1">{t('agent.planDetails.memory.relatedTitle')}</div>
                          <div className="space-y-1">
                            {state.planMemoryInfo.memories.map((memory, index) => (
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

            // 处理计划中的步骤数据，包含依赖关系
            if (planData.steps && Array.isArray(planData.steps)) {
              // 获取现有步骤，以便保留时间信息
              const existingSteps = currentState.steps
              const existingStepsMap = new Map<string, StepExecution>()
              existingSteps.forEach(step => existingStepsMap.set(step.id, step))

              const steps: StepExecution[] = planData.steps.map((step: any) => {
                const existingStep = existingStepsMap.get(step.id)
                return {
                  id: step.id,
                  name: step.name || step.id,
                  description: step.description || "",
                  // 优先使用现有步骤的状态，如果没有则使用 plan 中的状态
                  status: existingStep?.status || step.status || "pending",
                  tool_names: step.tool_name ? [step.tool_name] : step.tool_names || [],
                  dependencies: step.dependencies || [],
                  // 优先使用现有步骤的时间信息
                  started_at: existingStep?.started_at || step.started_at,
                  completed_at: existingStep?.completed_at || step.completed_at,
                  result_data: step.result_data,
                  step_data: step.step_data,
                  file_outputs: step.file_outputs || [],
                  conditional_branches: step.conditional_branches || {},
                  required_branch: step.required_branch || null,
                  is_conditional: step.is_conditional || false,
                }
              })
              dispatch({ type: "SET_STEPS", payload: steps })
            }

            const dedupKey = t('agent.logs.event.messages.planEnd', { planId, stepsCount })
            if (!isDuplicateMessage(dedupKey, 'plan-end')) {
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
              if (state.dagExecution && state.dagExecution.phase !== "completed" && state.dagExecution.phase !== "failed") {
                const updatedDAGExecution = {
                  ...state.dagExecution,
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
            dispatch({ type: "SET_PROCESSING", payload: true })

            // Update DAG execution state to executing phase
            if (state.dagExecution) {
              const updatedDAGExecution = {
                ...state.dagExecution,
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
            if (!isDuplicateMessage(dedupKey, 'dag-execute-start')) {
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
            if (state.dagExecution) {
              const updatedDAGExecution = {
                ...state.dagExecution,
                phase: "completed" as const,
                updated_at: message.timestamp,
              }
              dispatch({ type: "SET_DAG_EXECUTION", payload: updatedDAGExecution })
            }

            // Use consistent string format for deduplication
            const dedupKey = t('agent.logs.event.messages.taskEnd', { iteration })
            if (!isDuplicateMessage(dedupKey, 'dag-execute-end')) {
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
          // Compact Events - 都发生在step内部，显示在右侧面板的对应步骤中
          else if (eventType === "action_start_compact") {
            const stepId = eventData.step_id
            if (stepId) {
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`compact-start-${stepId}`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.compactStart'),
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
                  action: t('agent.logs.event.actions.compactCompleted'),
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
            }
          }

          // DAG Step Events
          else if (eventType === "dag_step_start") {
            const stepName = eventData.step_name || eventData.name || eventData.title || `${t('agent.logs.event.messages.execStepPrefix')}${eventData.step_id || t('common.errors.unknown')}`

            // dag_step_start有step_id，应该更新右侧步骤数据，不在左侧显示消息
            // 先查找现有步骤，保留依赖关系
            const existingStep = state.steps.find(s => s.id === (message.step_id || eventData.step_id || stepName))
            const step: StepExecution = {
              id: message.step_id || eventData.step_id || stepName,
              name: stepName,
              description: eventData.description || "",
              status: "running",
              tool_names: eventData.tool_name ? [eventData.tool_name] : eventData.tool_names || [],
              dependencies: existingStep?.dependencies || [],
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

            // 同时添加到 traceEvents 用于显示执行日志
            const traceEvent: TraceEvent = {
              event_id: generateMessageId(`trace-step-start`),
              event_type: eventType,
              step_id: message.step_id || eventData.step_id || stepName,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.stepStart'),
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

            // dag_step_end有step_id，应该更新右侧步骤数据，不在左侧显示消息
            const step: StepExecution = {
              id: message.step_id || eventData.step_id || stepName,
              name: stepName,
              description: eventData.description || "",
              status: eventData.status || "completed",
              tool_names: eventData.tool_name ? [eventData.tool_name] : eventData.tool_names || [],
              dependencies: [],
              // Don't override started_at from end event to preserve the original start time
              started_at: undefined, // Let the reducer handle preserving existing started_at
              completed_at: eventData.completed_at || message.timestamp,
              result_data: eventData.result_data,
              step_data: eventData.step_data,
              file_outputs: eventData.file_outputs || [],
              conditional_branches: eventData.conditional_branches || {},
              required_branch: eventData.required_branch || null,
              is_conditional: eventData.is_conditional || false,
            }
            dispatch({ type: "ADD_STEP", payload: step })

            // 同时添加到 traceEvents 用于显示执行日志
            const traceEvent: TraceEvent = {
              event_id: generateMessageId(`trace-step-end`),
              event_type: eventType,
              step_id: message.step_id || eventData.step_id || stepName,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.stepCompleted'),
                step_name: stepName,
                description: eventData.description,
                tool_names: eventData.tool_name ? [eventData.tool_name] : eventData.tool_names || [],
                completed_at: eventData.completed_at || message.timestamp,
                result_data: eventData.result_data,
                step_data: eventData.step_data,
                file_outputs: eventData.file_outputs || [],
              }
            }
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          } else if (eventType === "dag_step_failed") {
            const stepName = eventData.step_name || eventData.name || eventData.title || `${t('agent.logs.event.messages.execStepPrefix')}${eventData.step_id || t('common.errors.unknown')}`
            const stepId = message.step_id || eventData.step_id || stepName
            const existingStep = state.steps.find(s => s.id === stepId)

            // 更新 DAG 执行状态为失败
            if (state.dagExecution) {
              const updatedDAGExecution = {
                ...state.dagExecution,
                phase: "failed" as const,
                updated_at: message.timestamp,
              }
              dispatch({ type: "SET_DAG_EXECUTION", payload: updatedDAGExecution })
            }

            // 更新步骤状态
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

            // 添加到左侧消息
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

            // 同时添加到 traceEvents 用于显示执行日志
            const traceEvent: TraceEvent = {
              event_id: generateMessageId(`trace-step-failed`),
              event_type: eventType,
              step_id: stepId,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.stepFailed'),
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
            const taskType = eventData.task_type || "LLM调用"

            // Special handling for final answer generation
            if (eventData.task_type === "final_answer_generation") {
              // Check for duplicate final_answer_generation start events
              const content = t('agent.logs.event.messages.finalAnswerGenerating')
              if (!isDuplicateMessage(content, 'final_answer_start')) {
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
            const taskType = eventData.task_type || "LLM调用"

            // Special handling for final answer generation completion
            if (eventData.task_type === "final_answer_generation") {
              // Check for duplicate final_answer_generation end events
              const content = t('agent.logs.event.messages.finalAnswerCompleted')
              if (!isDuplicateMessage(content, 'final_answer_end')) {
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
            if (message.step_id) {
              const modelName = eventData.model_name || "LLM"
              const taskType = eventData.task_type || "LLM调用"

              // 添加到 traceEvents 用于步骤执行日志
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`trace-llm-start`),
                event_type: eventType,
                step_id: message.step_id,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.llmStart'),
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
              const taskType = eventData.task_type || "LLM调用"

              // 添加到 traceEvents 用于步骤执行日志
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`trace-llm-end`),
                event_type: eventType,
                step_id: message.step_id,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.llmCompleted'),
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
            const taskType = eventData.task_type || "LLM调用"

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
              // 添加到 traceEvents 用于步骤执行日志
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`trace-llm-info`),
                event_type: eventType,
                step_id: message.step_id,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.llmInfo'),
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
              // 添加到 traceEvents 用于步骤执行日志
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`trace-llm-result`),
                event_type: eventType,
                step_id: message.step_id,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.llmResult'),
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
                    {t('agent.logs.event.actions.toolStart')}: {toolName}
                  </>
                ),
                  timestamp: message.timestamp,
                  status: "completed",
                }
              })
            } else {
              // 添加到 traceEvents 用于步骤执行日志
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`trace-tool-start`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.toolStart'),
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
                    {t('agent.logs.event.actions.toolCompleted')}: {toolName}
                  </>
                ),
                  timestamp: message.timestamp,
                  status: "completed",
                }
              })
            } else {
              // 添加到 traceEvents 用于步骤执行日志
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`trace-tool-end`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.toolCompleted'),
                  tool_names: [toolName],
                  ...eventData
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            }
          } else if (eventType === "tool_execution_failed") {
            const toolName = eventData.tool_name || "工具"
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
                    {t('agent.logs.event.actions.toolFailed')}: {toolName}
                  </>
                ),
                  timestamp: message.timestamp,
                  status: "failed",
                }
              })
            } else {
              // 添加到 traceEvents 用于步骤执行日志
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`trace-tool-failed`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.toolFailed'),
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
              // 添加到 traceEvents 用于步骤执行日志
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`trace-tool-using`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.useTool'),
                  tool_names: [toolName],
                  ...eventData
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            }
          }

          // Task Completion Events
          else if (eventType === "task_completion") {
            const { result, success, metadata } = eventData
            console.log('🔍 task_completion event:', { result, success, metadata, hasResult: !!(result && result.trim() !== '') })

            // 解析 result 字符串为对象
            let resultData = {}
            if (typeof result === 'string') {
              try {
                resultData = JSON.parse(result)
              } catch (e) {
                console.log('Result is not JSON, treating as plain text output:', result)
                resultData = { output: result }
              }
            } else if (typeof result === 'object' && result !== null) {
              resultData = result
            } else {
              resultData = { output: result }
            }

            // 1. 输出元信息（排除 output 和 file_outputs 和 history）
            const metaInfo = { ...resultData }
            delete (metaInfo as any).output
            delete (metaInfo as any).file_outputs
            delete (metaInfo as any).history
            const hasMetaInfo = Object.keys(metaInfo).length > 0 && metaInfo !== null && metaInfo !== undefined

            // 1.5. 从 history 中提取步骤数据并更新 state.steps
            const history = (resultData as any).history
            if (history && Array.isArray(history) && history.length > 0) {
              const latestIteration = history[history.length - 1] // 最新的迭代（最后一个）
              if (latestIteration.plan && latestIteration.plan.steps && Array.isArray(latestIteration.plan.steps)) {
                // 创建 results 映射，方便快速查找
                const resultsMap = new Map<string, any>()
                if (latestIteration.results && Array.isArray(latestIteration.results)) {
                  latestIteration.results.forEach((result: any) => {
                    resultsMap.set(result.step_id, result)
                  })
                }

                // 获取 active_branches，用于判断哪些步骤被跳过
                const activeBranches = latestIteration.plan?.active_branches || {}

                // 获取现有步骤，以便保留时间信息
                const existingSteps = currentState.steps
                const existingStepsMap = new Map<string, StepExecution>()
                existingSteps.forEach(step => existingStepsMap.set(step.id, step))

                const steps: StepExecution[] = latestIteration.plan.steps.map((step: any) => {
                  // 从 results 中查找对应的执行结果
                  const stepResult = resultsMap.get(step.id)
                  // 查找现有步骤
                  const existingStep = existingStepsMap.get(step.id)

                  // 如果有执行结果，使用执行结果的状态；否则使用计划中的状态
                  let finalStatus = step.status || "pending"
                  let startedAt = step.started_at
                  let completedAt = step.completed_at
                  let resultData = step.result

                  if (stepResult) {
                    // 根据 result 字段判断状态
                    if (stepResult.result !== undefined && stepResult.result !== null) {
                      finalStatus = "completed"
                    }
                    // 无论 result 是否存在，都使用 stepResult 中的时间信息（如果存在）
                    if (stepResult.started_at) startedAt = stepResult.started_at
                    if (stepResult.completed_at) completedAt = stepResult.completed_at
                    // 如果 stepResult 有 result 字段，使用它
                    if (stepResult.result !== undefined && stepResult.result !== null) {
                      resultData = stepResult.result
                    }
                  }

                  // 检查是否应该被跳过：如果步骤需要特定分支，但该分支没有被激活
                  if (step.required_branch) {
                    // 找到这个步骤依赖的条件节点
                    const dependencyNodeId = step.dependencies && step.dependencies.length > 0 ? step.dependencies[0] : null
                    if (dependencyNodeId) {
                      const activeBranch = activeBranches[dependencyNodeId]
                      if (activeBranch && activeBranch !== step.required_branch) {
                        // 该分支没有被激活，所以这个步骤被跳过
                        finalStatus = "skipped"
                      }
                    }
                  }

                  // 优先使用现有步骤的信息（如果新数据中没有明确的信息）
                  if (existingStep) {
                    // 优先使用现有步骤的时间信息
                    if (!startedAt && existingStep.started_at) startedAt = existingStep.started_at
                    if (!completedAt && existingStep.completed_at) completedAt = existingStep.completed_at

                    // 优先使用现有步骤的状态（如果新步骤状态是 pending 或 running）
                    // 这确保了从 dag_step_end 事件获得的状态不会被 plan 数据覆盖
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
                    <JsonRenderer data={metaInfo} onFileClick={openFilePreview} />
                  </div>
                </div>
              )
              if (!isDuplicateResult(`📋 ${t('agent.logs.event.messages.metaTitle')}: ${JSON.stringify(metaInfo)}`)) {
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

            // 2. 输出文件输出
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
                      if (typeof file === 'string') {
                        fileName = file.split('/').pop() || file
                        filePath = file
                      } else if (typeof file === 'object' && file !== null) {
                        fileName = file.filename || file.file_path?.split('/').pop() || 'unknown'
                        filePath = file.download_path || file.download_url || file.file_path || file.relative_path || fileName
                      } else {
                        fileName = 'unknown'
                        filePath = 'unknown'
                      }

                      return (
                        <div key={index} className="flex items-center justify-between bg-muted/30 rounded p-2">
                          <span className="text-sm font-mono">{fileName}</span>
                          <button
                            onClick={() => {
                              // Dispatch custom event to open file preview with all files
                              const allFiles = fileOutputsData.map((file: string | any) => {
                                let fFileName, fFilePath
                                if (typeof file === 'string') {
                                  fFileName = file.split('/').pop() || file
                                  fFilePath = file
                                } else if (typeof file === 'object' && file !== null) {
                                  fFileName = file.filename || file.file_path?.split('/').pop() || 'unknown'
                                  fFilePath = file.download_path || file.download_url || file.file_path || file.relative_path || fFileName
                                } else {
                                  fFileName = 'unknown'
                                  fFilePath = 'unknown'
                                }
                                return { fileName: fFileName, filePath: fFilePath }
                              })

                                window.dispatchEvent(new CustomEvent('openFilePreview', {
                                detail: {
                                  filePath,
                                  fileName,
                                  allFiles,
                                  currentIndex: index
                                }
                              }))
                            }}
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

              if (!isDuplicateResult(`📁 ${t('agent.logs.event.messages.fileOutputsGenerated', { count: fileCount })}`)) {
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
            }

            // 3. 输出执行结果
            const finalOutput = (resultData as any).output
            if (finalOutput && finalOutput.trim() !== '') {
              const resultContent = (
                <div>
                  <JsonRenderer data={finalOutput} onFileClick={openFilePreview} />
                </div>
              )
              if (!isDuplicateResult(`📊 ${t('agent.logs.event.messages.executionResultPrefix')} ${finalOutput}`)) {
                dispatch({
                  type: "ADD_MESSAGE",
                  payload: {
                    id: generateMessageId("msg-task-result"),
                    role: "assistant",
                    content: resultContent,
                    timestamp: message.timestamp,
                    status: success ? "completed" : "failed",
                    isResult: true,
                  }
                })
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
            // 优先使用 error_message，如果没有则使用 error，最后使用默认消息
            const errorMessage = eventData.error_message || eventData.error || 'Trace error occurred'
            const stepName = eventData.step_name || eventData.name || `${t('agent.logs.event.messages.execStepPrefix')}${eventData.step_id || t('common.errors.unknown')}`
            const stepId = message.step_id || eventData.step_id

            // 调试信息
            console.trace('trace_error debug:', {
              message_step_id: message.step_id,
              eventData_step_id: eventData.step_id,
              stepName: stepName,
              stepId: stepId,
              hasStepId: !!stepId,
              eventData: eventData,
              errorMessage: errorMessage
            })

            // 只添加到 trace events 用于显示执行日志，不将步骤标记为失败
            if (stepId && stepId !== 'unknown') {
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`trace-error-${stepId}`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.compactStepErrorLog'),
                  step_name: stepName,
                  error: errorMessage,
                  error_type: eventData.error_type,
                  tool_names: eventData.tool_name ? [eventData.tool_name] : eventData.tool_names || [],
                  ...(eventData.execution_time && { execution_time: eventData.execution_time }),
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            }

            // 对于步骤相关的错误，不在左侧面板显示，只在右侧面板显示
            // 只在左侧显示非步骤相关的全局错误
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

          // AI Message Events
          else if (eventType === "ai_message") {
            dispatch({
              type: "ADD_MESSAGE",
              payload: {
                id: generateMessageId("msg-ai"),
                role: "assistant",
                content: eventData.content || "",
                timestamp: message.timestamp,
                status: "completed",
              }
            })
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

          // React Pattern Events - 这些应该显示在右侧面板
          else if (eventType === "react_task_start" || eventType === "task_start_react") {
            // ReAct pattern often doesn't emit user_message event, so we extract it from task data
            const taskContent = eventData.task || eventData.description || ""
            if (taskContent && !isDuplicateMessage(taskContent, 'user-message')) {
               dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId("msg-user-react-start"),
                  role: "user",
                  content: taskContent,
                  timestamp: message.timestamp,
                }
              })
            }

            // 添加到 trace events 用于显示执行日志
            const traceEvent: TraceEvent = {
              event_id: generateMessageId("react-task-start"),
              event_type: eventType,
              step_id: eventData.step_id,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.reactTaskStart'),
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
                  action: t('agent.logs.event.actions.reactTaskEnd') || 'Task Completed',
                  message: t('agent.logs.event.messages.reactTaskEnd') || 'ReAct Task Completed',
                  ...eventData
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          } else if (eventType === "react_task_failed" || eventType === "task_failed_react") {
             const traceEvent: TraceEvent = {
                event_id: generateMessageId("react-task-failed"),
                event_type: eventType,
                step_id: eventData.step_id,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.reactTaskFailed') || 'Task Failed',
                  message: t('agent.logs.event.messages.reactTaskFailed') || 'ReAct Task Failed',
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
                  action: t('agent.logs.event.actions.reactActionStart') || 'Action Start',
                  ...eventData
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          } else if (eventType === "llm_call_start") {
             const stepId = message.step_id || traceEventData.step_id
             const traceEvent: TraceEvent = {
                event_id: generateMessageId("llm-call-start"),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.llmCallStart') || 'LLM Call Start',
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
                  action: t('agent.logs.event.actions.llmCallEnd') || 'LLM Call End',
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
                  action: t('agent.logs.event.actions.llmCallFailed') || 'LLM Call Failed',
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
                  action: t('agent.logs.event.actions.reactActionEnd') || 'Action End',
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
                action: t('agent.logs.event.actions.taskCompleted'),
                message: t('agent.logs.event.messages.taskCompleted'),
                result: eventData.result,
                success: eventData.success
              }
            }
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          } else if (eventType === "react_task_end" || eventType === "task_end_react") {
            // 添加到 trace events 用于显示执行日志
            const traceEvent: TraceEvent = {
              event_id: generateMessageId("react-task-end"),
              event_type: eventType,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.reactTaskCompleted'),
                message: t('agent.logs.event.messages.reactTaskCompleted'),
                output: eventData.output
              }
            }
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          } else if (eventType === "step_start_react") {
            const stepName = eventData.step_name || 'unknown'
            const stepId = `react-${stepName}`

            // 创建或更新步骤
            const step: StepExecution = {
              id: stepId,
              name: stepName,
              description: `ReAct步骤: ${stepName}`,
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

            // 添加到 trace events 用于显示执行日志
            const traceEvent: TraceEvent = {
              event_id: generateMessageId(`react-step-start-${stepId}`),
              event_type: eventType,
              step_id: stepId,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.reactStepStart'),
                step_name: stepName,
                tool_names: eventData.tool_name ? [eventData.tool_name] : eventData.tool_names || [],
                message: t('agent.logs.event.messages.reactStepStart', { stepName }),
              }
            }
            dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          } else if (eventType === "step_end_react") {
            const stepName = eventData.step_name || 'unknown'
            const stepId = `react-${stepName}`

            // 更新步骤状态
            const step: StepExecution = {
              id: stepId,
              name: stepName,
              description: `ReAct步骤: ${stepName}`,
              status: "completed",
              tool_names: eventData.tool_name ? [eventData.tool_name] : eventData.tool_names || [],
              dependencies: [],
              started_at: undefined, // 保持原有的开始时间
              completed_at: message.timestamp,
              result_data: eventData.result_data,
              step_data: eventData,
              file_outputs: eventData.file_outputs || [],
            }
            dispatch({ type: "ADD_STEP", payload: step })

            // 添加到 trace events 用于显示执行日志
            const traceEvent: TraceEvent = {
              event_id: generateMessageId(`react-step-end-${stepId}`),
              event_type: eventType,
              step_id: stepId,
              timestamp: message.timestamp,
              data: {
                action: t('agent.logs.event.actions.reactStepCompleted'),
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
             const traceEvent: TraceEvent = {
                event_id: generateMessageId("skill-select-start"),
                event_type: eventType,
                step_id: eventData.step_id,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.skillSelectStart'),
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
                  action: t('agent.logs.event.actions.skillSelectEnd'),
                  ...eventData
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
          }
          // Memory Events - 根据是否有step_id决定显示位置
          else if (eventType === "task_start_memory_generate") {
            const stepId = eventData.step_id

            // 如果有step_id，添加到对应的步骤；否则不显示（跳过无用的开始事件）
            if (stepId) {
              // ReAct pattern - 显示在右侧面板的对应步骤中
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`memory-generate-start-${stepId}`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.memoryGenerateStart'),
                  message: '🧠 ' + t('agent.logs.event.actions.memoryGenerateStart'),
                  task: eventData.task,
                  iterations: eventData.iterations,
                  result_length: eventData.result_length,
                  messages_count: eventData.messages_count,
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            }
            // 没有step_id的情况直接跳过，不显示开始事件
          } else if (eventType === "task_end_memory_generate") {
            const taskId = eventData.task_id || "unknown"
            const stepId = eventData.step_id

            // 如果有step_id，添加到对应的步骤；否则显示在左侧面板
            if (stepId) {
              // ReAct pattern - 显示在右侧面板的对应步骤中
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`memory-generate-end-${stepId}`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.memoryGenerateCompleted'),
                  message: '🧠 ' + t('agent.logs.event.actions.memoryGenerateCompleted'),
                  insights_generated: eventData.insights_generated,
                  should_store: eventData.should_store,
                  reason: eventData.reason,
                  source: eventData.source,
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            } else {
              // DAG plan-execute pattern - 显示在左侧面板
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
                        {t('agent.logs.event.actions.memoryGenerateCompleted')}
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

            // 如果有step_id，添加到对应的步骤；否则显示在左侧面板
            if (stepId) {
              // ReAct pattern - 显示在右侧面板的对应步骤中
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`memory-store-start-${stepId}`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.memoryStoreStart'),
                  message: '🧠 ' + t('agent.logs.event.actions.memoryStoreStart'),
                  task: eventData.task,
                  memory_category: eventData.memory_category,
                  classification: eventData.classification,
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            } else {
              // DAG plan-execute pattern - 显示在左侧面板
              dispatch({
                type: "ADD_MESSAGE",
                payload: {
                  id: generateMessageId("msg-memory-store-start"),
                  role: "assistant",
                  content: (
                    <>
                      <Brain className="h-4 w-4 inline mr-2" />
                      {t('agent.logs.event.actions.memoryStoreStart')}
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

            // 如果有step_id，添加到对应的步骤；否则显示在左侧面板
            if (stepId) {
              // ReAct pattern - 显示在右侧面板的对应步骤中
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`memory-store-end-${stepId}`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.memoryStoreCompleted'),
                  message: '🧠 ' + t('agent.logs.event.actions.memoryStoreCompleted'),
                  storage_success: eventData.storage_success,
                  reason: eventData.reason,
                  decision: eventData.decision,
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            } else {
              // DAG plan-execute pattern - 显示在左侧面板
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
                        {t('agent.logs.event.actions.memoryStoreCompleted')}
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

            // 如果有step_id，添加到对应的步骤；否则显示在左侧面板
            if (stepId) {
              // ReAct pattern - 显示在右侧面板的对应步骤中
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`memory-retrieve-start-${stepId}`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.memoryQuery'),
                  message: '🔍 ' + t('agent.logs.event.actions.memoryQuery'),
                  // 显示完整数据
                  rawData: eventData,
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            } else {
              // DAG plan-execute pattern - 显示在左侧面板
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
                      {t('agent.logs.event.actions.memoryQueryStart')}
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

            // 如果有step_id，添加到对应的步骤；否则显示在左侧面板
            if (stepId) {
              // ReAct pattern - 显示在右侧面板的对应步骤中
              const traceEvent: TraceEvent = {
                event_id: generateMessageId(`memory-retrieve-end-${stepId}`),
                event_type: eventType,
                step_id: stepId,
                timestamp: message.timestamp,
                data: {
                  action: t('agent.logs.event.actions.memoryQueryCompleted'),
                  message: '🔍 ' + t('agent.logs.event.actions.memoryQueryCompleted'),
                  // 显示完整数据
                  rawData: eventData,
                }
              }
              dispatch({ type: "ADD_TRACE_EVENT", payload: traceEvent })
            } else {
              // DAG plan-execute pattern - 显示在左侧面板
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
                      {t('agent.logs.event.actions.memoryQueryCompleted')}
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
            isHistoricalDataLoading = false
            dispatch({ type: "SET_HISTORY_LOADING", payload: false })

            // If we're in replay mode, initialize the replay scheduler
            if (state.isReplaying && state.replayTaskId && state.replayEventCache.length > 0) {
              initializeReplayScheduler()
            } else {
              // Fix: If we have cache but replay mode is not set, force start replay
              if (state.replayEventCache.length > 0 && state.replayTaskId && !state.isReplaying) {
                dispatch({ type: "SET_REPLAY_PLAYING", payload: true })
                setTimeout(() => {
                  initializeReplayScheduler()
                }, 50)
              }
            }
          }

          // Default: add as trace event
          else {
            console.trace('原始 message:', JSON.stringify(message), '处理函数: handleMessage (unhandled event_type:', eventType, ')')
                        dispatch({ type: "ADD_TRACE_EVENT", payload: traceEventData })
          }
        } else {
          console.trace('原始 message:', JSON.stringify(message), '处理函数: handleMessage (no event_type, direct trace event)')
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
              id: state.taskId?.toString() || "unknown",
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
        console.trace('原始 message:', JSON.stringify(message), '处理函数: handleMessage (chat_message)')
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
        const taskData = message.data as { success?: boolean; result?: string; file_outputs?: string[] }
        dispatch({
          type: "UPDATE_TASK_STATUS",
          payload: { status: taskData.success ? "completed" : "failed" }
        })
        dispatch({ type: "TRIGGER_TASK_UPDATE" })

        // 更新 DAG 执行状态为完成
        if (state.dagExecution) {
          const updatedDAGExecution = {
            ...state.dagExecution,
            phase: (taskData.success ? "completed" : "failed") as "completed" | "failed",
            updated_at: new Date().toISOString()
          }
          dispatch({ type: "SET_DAG_EXECUTION", payload: updatedDAGExecution })
        } else {
          const dagExecution: DAGExecution = {
            phase: (taskData.success ? "completed" : "failed") as "completed" | "failed",
            current_plan: {},
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString()
          }
          dispatch({ type: "SET_DAG_EXECUTION", payload: dagExecution })
        }

        // Mark that historical data should not be requested again for completed/failed tasks
        if (state.taskId) {
          historicalDataRequestMap.set(state.taskId, true)
        }

        // Note: Result is now handled by trace events, not included in task_completed event

        // Handle file outputs
        if (taskData.file_outputs && taskData.file_outputs.length > 0) {
          const fileCount = taskData.file_outputs.length
          const fileContent = (
            <>
              <FileText className="h-4 w-4 inline mr-2 text-green-500" />
                    {t('agent.logs.event.messages.fileOutputsGenerated', { count: fileCount })}:
                    <div className="mt-2 space-y-1">
                {taskData.file_outputs.map((file: string | any, index: number) => {
                  let fileName, filePath
                  if (typeof file === 'string') {
                    fileName = file.split('/').pop() || file
                    filePath = file
                  } else if (typeof file === 'object' && file !== null) {
                    fileName = file.filename || file.file_path?.split('/').pop() || 'unknown'
                    filePath = file.download_path || file.download_url || file.file_path || file.relative_path || fileName
                  } else {
                    fileName = 'unknown'
                    filePath = 'unknown'
                  }

                  return (
                    <div key={index} className="flex items-center justify-between bg-muted/30 rounded p-2">
                      <span className="text-sm font-mono">{fileName}</span>
                      <button
                        onClick={() => {
                          // Dispatch custom event to open file preview with all files
                          const allFiles = (taskData.file_outputs || []).map((file: string | any) => {
                            let fFileName, fFilePath
                            if (typeof file === 'string') {
                              fFileName = file.split('/').pop() || file
                              fFilePath = file
                            } else if (typeof file === 'object' && file !== null) {
                              fFileName = file.filename || file.file_path?.split('/').pop() || 'unknown'
                              fFilePath = file.download_path || file.download_url || file.file_path || file.relative_path || fFileName
                            } else {
                              fFileName = 'unknown'
                              fFilePath = 'unknown'
                            }
                            return { fileName: fFileName, filePath: fFilePath }
                          })

                            window.dispatchEvent(new CustomEvent('openFilePreview', {
                            detail: {
                              filePath,
                              fileName,
                              allFiles,
                              currentIndex: index
                            }
                          }))
                        }}
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

          if (!isDuplicateResult(`📁 ${t('agent.logs.event.messages.fileOutputsGenerated', { count: fileCount })}`)) {
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

        // 更新 DAG 执行状态
        // 根据步骤状态更新整体 DAG 状态
        if (state.dagExecution) {
          const updatedDAGExecution = { ...state.dagExecution }

          // 根据步骤状态更新 DAG phase
          if (stepInfo.status === "running") {
            updatedDAGExecution.phase = "executing" as const
          } else if (stepInfo.status === "completed") {
            // 检查是否所有步骤都完成了
            const allStepsCompleted = state.steps.every(step =>
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

          // 更新时间戳
          updatedDAGExecution.updated_at = new Date().toISOString()

          dispatch({ type: "SET_DAG_EXECUTION", payload: updatedDAGExecution })
        }
        break

      case "dag_execution":
        dispatch({ type: "SET_DAG_EXECUTION", payload: message.data as DAGExecution })
        break


      case "task_paused":
        console.trace('原始 message:', JSON.stringify(message), '处理函数: handleMessage (task_paused)')
        dispatch({ type: "UPDATE_TASK_STATUS", payload: { status: "paused" } })
        break

      case "task_resumed":
        console.trace('原始 message:', JSON.stringify(message), '处理函数: handleMessage (task_resumed)')
        dispatch({ type: "UPDATE_TASK_STATUS", payload: { status: "running" } })
        break

      case "agent_error":
        console.trace('原始 message:', JSON.stringify(message), '处理函数: handleMessage (agent_error)')
        const errorData = message.data as { message?: string }

        // 更新 DAG 执行状态为失败
        if (state.dagExecution) {
          const updatedDAGExecution = {
            ...state.dagExecution,
            phase: "failed" as const,
            updated_at: message.timestamp,
          }
          dispatch({ type: "SET_DAG_EXECUTION", payload: updatedDAGExecution })
        }

        dispatch({ type: "SET_PROCESSING", payload: false })
        dispatch({
          type: "ADD_MESSAGE",
          payload: {
            id: generateMessageId("msg"),
            role: "assistant",
            content: `${t('agent.logs.event.messages.errorPrefix')} ${errorData.message || t('common.errors.unknownError')}`,
            timestamp: message.timestamp,
            status: "failed",
          },
        })
        break

      case "message_received":
        console.trace('原始 message:', JSON.stringify(message), '处理函数: handleMessage (message_received)')
        // User message confirmation
        dispatch({ type: "SET_PROCESSING", payload: true })
        break

      case "historical_data_complete":
        // Historical data loading complete
        isHistoricalDataLoading = false
        dispatch({ type: "SET_HISTORY_LOADING", payload: false })

        // If we're in replay mode, initialize the replay scheduler
        if (state.isReplaying && state.replayTaskId && state.replayEventCache.length > 0) {
          initializeReplayScheduler()
        }
        break
    }
  }, [])

  const getLLMNamesFromConfig = (config?: any) => {
    if (!config || !config.model) {
      return null
    }

    // Debug log to see what config is being passed
    console.log('getLLMNamesFromConfig called with:', config)

    // Always return exactly 4 elements in fixed order: [default, fast_small, vision, compact]
    // Use null for unconfigured models
    const llmNames = [
      config.model,                           // Default model (required)
      config.smallFastModel || null,         // Fast small model (optional)
      config.visualModel || null,            // Vision model (optional)
      config.compactModel || null            // Compact model (optional)
    ]

    return llmNames
  }

  const sendMessage = useCallback(async (message: string, config?: any, files?: File[]) => {
    console.log('🚀 sendMessage called:', { message, files: files?.map(f => f.name), taskId: state.taskId })

    if (!state.taskId) {
      // Create a new task via API
      try {
        const apiUrl = getApiUrl()

        // Build LLM names from config
        const llmNames = getLLMNamesFromConfig(config)

        // For process mode, message is already processDescription (from handleSendMessage)
        // For task mode, message is user input
        let taskDescription = message
        const taskTitle = message.length > 50 ? `${message.substring(0, 50)}...` : message

        // Note: Files will be uploaded via WebSocket after task creation
        // The backend TaskCreateRequest expects JSON with 'files' as a list of filenames (strings)
        // Since we haven't uploaded files yet, we don't include them in the task creation request

        const requestBody: any = {
          title: taskTitle,
          description: taskDescription,
          vibe_mode: config?.vibeMode?.mode || "task",
          memory_similarity_threshold: config?.memorySimilarityThreshold ?? 1.5,
        }

        // Add LLM configuration
        if (llmNames) {
          requestBody.llm_names = llmNames
        }

        if (config?.vibeMode?.processDescription) {
          requestBody.process_description = config.vibeMode.processDescription
        }
        if (config?.vibeMode?.examples) {
          requestBody.examples = config.vibeMode.examples
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

          // Create a new task from response
          const newTask: Task = {
            id: newTaskId.toString(),
            title: taskData.title,
            status: taskData.status,
            description: taskData.description || message,
            createdAt: taskData.created_at,
            updatedAt: taskData.updated_at,
            modelName: taskData.model_name || taskData.modelName, // API response field
            smallFastModelName: taskData.small_fast_model_name || taskData.smallFastModelName, // API response field
            visualModelName: taskData.visual_model_name || taskData.visual_model_name,
            compactModelName: taskData.compact_model_name || taskData.compact_model_name,
            vibeMode: taskData.vibe_mode,
            isDag: taskData.is_dag,
            agentId: taskData.agent_id,
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

          // Store the message to be sent after WebSocket connects
          setPendingMessage({ message, files })

          // Optimistically add the user message to the UI
          if (!isDuplicateMessage(message, 'user-message')) {
            let content: React.ReactNode = message
            if (files && files.length > 0) {
              content = (
                <div className="space-y-2">
                  <div>{message}</div>
                  <FileAttachment
                    files={files.map(f => ({ name: f.name, type: f.type, size: f.size, path: '' }))} // Basic info for optimistic render
                    variant="user-message"
                  />
                </div>
              )
            }

            const optimisticId = generateMessageId("msg-user-optimistic")
            // Store ID to prevent clearing it when task loads
            pendingOptimisticMessageId.current = optimisticId

            dispatch({
              type: "ADD_MESSAGE",
              payload: {
                id: optimisticId,
                role: "user",
                content: content,
                timestamp: Date.now().toString(),
              }
            })
          }
        } else {
          console.error('Failed to create task:', response.statusText)
          return
        }
      } catch (error) {
        console.error('Error creating task:', error)
        return
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

      // Optimistically add the user message to the UI
      if (!isDuplicateMessage(message, 'user-message')) {
        let content: React.ReactNode = message
        if (files && files.length > 0) {
          content = (
            <div className="space-y-2">
              <div>{message}</div>
              <FileAttachment
                files={files.map(f => ({ name: f.name, type: f.type, size: f.size, path: '' }))} // Basic info for optimistic render
                variant="user-message"
              />
            </div>
          )
        }

        dispatch({
          type: "ADD_MESSAGE",
          payload: {
            id: generateMessageId("msg-user-optimistic"),
            role: "user",
            content: content,
            timestamp: Date.now().toString(),
          }
        })

        // Send chat message - backend will handle user message via trace event
        // Only send if not a duplicate
        sendChatMessage(message, files)
      } else {
        console.log('⚠️ Duplicate message blocked from sending:', message)
      }
    }
  }, [state.taskId, sendChatMessage, wsExecuteTask, state.currentTask?.status])

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
        handleMessage(message, dispatch, tempState)
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
  }, [state.isReplaying, state.replayTaskId, state.replayEventCache, state.replaySpeed, dispatch])

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
  }, [])

  const clearMessages = useCallback(() => {
    dispatch({ type: "CLEAR_MESSAGES" })
  }, [])

  const setTaskId = useCallback((taskId: number | null) => {
    // Only reset historical data request flag when changing to a different task
    if (taskId !== stateRef.current.taskId) {
      if (taskId) {
        historicalDataRequestMap.set(taskId, false)
      }
      // Clear recentMessages cache when switching tasks to prevent false duplicates
      recentMessages.clear()
      isHistoricalDataLoading = false
    }

    // Update URL to use dynamic route for task detail page
    if (taskId) {
      router.push(`/task/${taskId}`)
    } else {
      router.push('/task')
    }

    dispatch({ type: "SET_TASK_ID", payload: taskId })
    // Set history loading state immediately when switching tasks to prevent empty state flash
    if (taskId) {
      dispatch({ type: "SET_HISTORY_LOADING", payload: true })
    }
  }, [router])

  const openFilePreview = useCallback((filePath: string, fileName: string, files?: Array<{ filePath: string; fileName: string }>, index?: number) => {
    console.log('🎯 openFilePreview called:', {
      filePath,
      fileName,
      files: files,
      filesLength: files?.length,
      index
    })
    dispatch({ type: "OPEN_FILE_PREVIEW", payload: { filePath, fileName, files, index } })
  }, [])

  const switchFilePreview = useCallback((index: number) => {
    const { availableFiles } = state.filePreview
    if (index >= 0 && index < availableFiles.length) {
      const file = availableFiles[index]
      dispatch({ type: "SWITCH_FILE_PREVIEW", payload: { filePath: file.filePath, fileName: file.fileName, index } })
    }
  }, [state.filePreview.availableFiles])

  const closeFilePreview = useCallback(() => {
    dispatch({ type: "CLOSE_FILE_PREVIEW" })
  }, [])

  // Historical data is automatically sent by backend when WebSocket connects
  useEffect(() => {
    if (isConnected && state.taskId) {
      // Clear existing data to prepare for incoming historical data
      // Keep pending optimistic message if exists
      const keepMessageId = pendingOptimisticMessageId.current
      pendingOptimisticMessageId.current = null

      dispatch({ type: "CLEAR_MESSAGES", payload: { keepMessageId } })
      dispatch({ type: "SET_TRACE_EVENTS", payload: [] })
      dispatch({ type: "SET_STEPS", payload: [] })

      // Set history loading state
      dispatch({ type: "SET_HISTORY_LOADING", payload: true })

      // Safety timeout: if no history arrives within 2 seconds, assume empty or done
      const timer = setTimeout(() => {
        dispatch({ type: "SET_HISTORY_LOADING", payload: false })
      }, 2000)

      return () => clearTimeout(timer)
    }
  }, [isConnected, state.taskId])

  // Replay control methods
  const startReplay = useCallback((taskId: number, events: TraceEvent[]) => {
    dispatch({ type: "START_REPLAY", payload: { taskId, events } })
  }, [])

  const stopReplay = useCallback(() => {
    dispatch({ type: "STOP_REPLAY" })
  }, [])

  const setReplayPlaying = useCallback((isPlaying: boolean) => {
    dispatch({ type: "SET_REPLAY_PLAYING", payload: isPlaying })
  }, [])

  const setReplaySpeed = useCallback((speed: number) => {
    dispatch({ type: "SET_REPLAY_SPEED", payload: speed })
  }, [])

  const setReplayProgress = useCallback((progress: number) => {
    dispatch({ type: "SET_REPLAY_PROGRESS", payload: progress })
  }, [])

  // Initialize the delayed playback function
  startDelayedPlayback = useCallback(() => {
    // Use the replay scheduler to play all events with proper time intervals
    initializeReplayScheduler()
  }, [state.replayEventCache, initializeReplayScheduler])

  return (
    <AppContext.Provider
      value={{
        state,
        dispatch,
        sendMessage,
        executeTask,
        pauseTask,
        resumeTask,
        selectStep,
        clearMessages,
        isConnected,
        connectionError,
        setTaskId,
        requestStatus,
        openFilePreview,
        switchFilePreview,
        closeFilePreview,
        startReplay,
        stopReplay,
        setReplayPlaying,
        setReplaySpeed,
        setReplayProgress,
      }}
    >
      {children}
    </AppContext.Provider>
  )
}

export function useApp() {
  const context = useContext(AppContext)
  if (context === undefined) {
    throw new Error("useApp must be used within an AppProvider")
  }
  return context
}

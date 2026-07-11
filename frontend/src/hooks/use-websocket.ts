"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import { useAuth } from "@/contexts/auth-context"
import { apiRequest, getUploadErrorMessage, isJsonRecord, parseApiResponse, UPLOAD_ERROR_MESSAGES } from "@/lib/api-wrapper"
import { generateClientMessageId, getWsUrl, getUploadApiUrl } from "@/lib/utils"
import { isFinalAnswerStreamEventType } from "@/lib/streaming-final-answer"

// Duplicate message detection: record recently sent messages
const recentMessages: Array<{
  message: string
  timestamp: number
  taskId: number
  clientMessageId: string
}> = []
const MESSAGE_DUPLICATE_THRESHOLD = 2000 // Same message within 2 seconds is considered duplicate

interface WebSocketMessage {
  type: string
  data: unknown
  timestamp: string
  task_id?: number
  step_id?: string
  event_id?: string
  event_type?: string
  message_id?: string
  delta?: string
  content?: string
  run_id?: string | null
  state_version?: number
  control_state?: "idle" | "running" | "pause_requested" | "paused" | "resume_requested" | "waiting_for_user" | "completed" | "failed"
  status?: unknown
  task?: Record<string, unknown>
}

interface MessageDeliveryAck {
  client_message_id: string
  turn_id: string
}

interface UseWebSocketOptions {
  url?: string
  taskId?: number
  token?: string
  buildWebSocketUrl?: (params: { baseUrl: string; taskId: number; token?: string }) => string
  uploadFiles?: (files: File[], params: { taskId?: number | null; taskType: string }) => Promise<Array<{ file_id: string; name?: string; size?: number; type?: string }>>
  autoConnect?: boolean
  onMessage?: (message: WebSocketMessage) => void
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Error) => void
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    url = getWsUrl(),
    taskId,
    token,
    buildWebSocketUrl,
    uploadFiles,
    autoConnect = true,
    onMessage,
    onConnect,
    onDisconnect,
    onError,
  } = options

  const { token: authToken, refreshToken: authRefreshToken } = useAuth()


  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const [connectionError, setConnectionError] = useState<Error | null>(null)
  const isConnectingRef = useRef(false)

  const socketRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const taskIdRef = useRef(taskId)
  const tokenRef = useRef(token || authToken) // Prioritize passed token, otherwise use auth token
  const pendingDeliveriesRef = useRef(new Map<string, {
    resolve: (ack: MessageDeliveryAck) => void
    reject: (error: Error) => void
    timeout: ReturnType<typeof setTimeout>
  }>())
  const maxReconnectAttempts = 3

  const rejectPendingDeliveries = useCallback((error: Error) => {
    for (const pending of pendingDeliveriesRef.current.values()) {
      clearTimeout(pending.timeout)
      pending.reject(error)
    }
    pendingDeliveriesRef.current.clear()
  }, [])

  // Update token ref when token changes
  useEffect(() => {
    tokenRef.current = token || authToken
  }, [token, authToken])

  // Update token ref when auth token changes (for refresh token support)
  useEffect(() => {
    if (!token && authToken) {
      tokenRef.current = authToken

      // If WebSocket is connected and we got a new token, reconnect with new token
      if (socketRef.current?.readyState === WebSocket.OPEN && taskId) {
        disconnect()
        setTimeout(() => {
          connect()
        }, 1000)
      }
    }
  }, [authToken, token, taskId])

  const connect = useCallback(() => {
    if (socketRef.current?.readyState === WebSocket.OPEN || isConnectingRef.current) return
    isConnectingRef.current = true

    try {
      // Don't try to connect if there's no task ID
      if (!taskId) {
        isConnectingRef.current = false
        return
      }

      const wsUrl = buildWebSocketUrl
        ? buildWebSocketUrl({
          baseUrl: url,
          taskId,
          token: tokenRef.current || undefined,
        })
        : `${url}/ws/chat/${taskId}${tokenRef.current ? `?token=${tokenRef.current}` : ''}`

      // Test if the URL is valid before creating WebSocket
      if (!wsUrl.startsWith('ws://') && !wsUrl.startsWith('wss://')) {
        throw new Error("Invalid WebSocket URL configuration")
      }

      const socket = new WebSocket(wsUrl)
      socketRef.current = socket

      socket.onopen = () => {
        setIsConnected(true)
        setConnectionError(null)
        reconnectAttemptsRef.current = 0
        isConnectingRef.current = false
        onConnect?.()
      }

      socket.onclose = (event) => {
        rejectPendingDeliveries(new Error('Connection closed before the message was accepted.'))
        setIsConnected(false)
        isConnectingRef.current = false
        onDisconnect?.()

        // Handle authentication errors (4001 = Authentication required)
        if (event.code === 4001) {
          if (authRefreshToken && typeof authRefreshToken === 'function') {
            try {
              const refreshTokenFunc = authRefreshToken as () => Promise<boolean>
              refreshTokenFunc().then(refreshSuccess => {
                if (refreshSuccess) {
                  setTimeout(() => {
                    if (taskIdRef.current) {
                      connect()
                    }
                  }, 1000)
                } else {
                  onError?.(new Error('Authentication failed and token refresh failed'))
                }
              }).catch(error => {
                console.error('Error refreshing auth token for WebSocket', error)
                onError?.(new Error('Authentication failed and token refresh error'))
              })
            } catch (error) {
              console.error('Error refreshing auth token for WebSocket', error)
              onError?.(new Error('Authentication failed and token refresh error'))
            }
          } else {
            onError?.(new Error('Authentication failed and no refresh token available'))
          }
          return
        }

        if (event.code === 4003) {
          const accessError = new Error(event.reason || 'Access denied')
          setConnectionError(accessError)
          onError?.(accessError)
          return
        }

        // Don't reconnect if it's a 404 error or abnormal closure (1006)
        if (event.code === 1006) {
          return
        }

        // Don't reconnect if it's a clean close (might be intentional)
        if (event.code === 1000) {
          return
        }

        // Don't reconnect if the reason is component unmounting
        if (event.reason === 'Component unmounting') {
          return
        }

        // Only attempt to reconnect if under max attempts and taskId exists
        if (reconnectAttemptsRef.current < maxReconnectAttempts && taskId) {
          reconnectAttemptsRef.current++
          const delay = Math.min(1000 * reconnectAttemptsRef.current, 5000)
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, delay)
        }
      }

      socket.onerror = (error) => {
        console.error('WebSocket error', error)
        const connectionError = new Error("WebSocket connection failed. The backend WebSocket endpoint may not be available.")
        setConnectionError(connectionError)
        setIsConnected(false)
        isConnectingRef.current = false
        onError?.(connectionError)

        // Don't attempt to reconnect if there's an immediate error (like 404)
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current)
          reconnectTimeoutRef.current = null
        }

        // Reset reconnect attempts to prevent immediate reconnection when backend is not available
        reconnectAttemptsRef.current = maxReconnectAttempts
      }

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)

          if (data.type === 'message_accepted' || data.type === 'message_rejected') {
            const clientMessageId = data.client_message_id
            const pending = typeof clientMessageId === 'string'
              ? pendingDeliveriesRef.current.get(clientMessageId)
              : undefined
            if (pending) {
              clearTimeout(pending.timeout)
              pendingDeliveriesRef.current.delete(clientMessageId)
              if (data.type === 'message_accepted') {
                pending.resolve({
                  client_message_id: clientMessageId,
                  turn_id: typeof data.turn_id === 'string' ? data.turn_id : clientMessageId,
                })
              } else {
                const error = new Error(data.message || 'Message was rejected.')
                Object.assign(error, {
                  retryWithNewId: data.retry_with_new_id === true,
                })
                pending.reject(error)
              }
            }
            return
          }

          // Handle different message types from the backend
          let message: WebSocketMessage

          if (isFinalAnswerStreamEventType(data.type)) {
            message = {
              type: data.type,
              data,
              timestamp: data.timestamp || new Date().toISOString(),
              task_id: data.task_id,
              event_id: data.event_id,
              message_id: data.message_id,
              delta: data.delta,
              content: data.content,
            }
          } else if (data.type === "trace_event") {
            // Ensure data.data is not an empty string
            const safeData = typeof data.data === 'string' && data.data === ''
              ? {}
              : data.data;

            message = {
              type: "trace_event",
              data: safeData,
              timestamp: data.timestamp,
              task_id: data.task_id,
              step_id: data.step_id,
              event_id: data.event_id,
              event_type: data.event_type,  // Keep event_type field!
            }
          } else if (data.type === "task_completed") {
            message = {
              type: "task_completed",
              data: data,
              timestamp: data.timestamp,
              task_id: data.task?.id || data.task_id,
            }
          } else if (data.type === "dag_execution") {
            // Ensure data.data is not an empty string
            const safeData = typeof data.data === 'string' && data.data === ''
              ? {}
              : data.data;

            message = {
              type: "dag_execution",
              data: safeData,
              timestamp: data.timestamp,
              task_id: data.task_id,
            }
          } else if (data.type === "dag_step_info") {
            // Ensure data.data is not an empty string
            const safeData = typeof data.data === 'string' && data.data === ''
              ? {}
              : data.data;

            message = {
              type: "dag_step_info",
              data: safeData,
              timestamp: data.timestamp,
              task_id: data.task_id,
              step_id: safeData?.id,
            }
          } else if (data.type === "task_paused") {
            message = {
              type: "task_paused",
              data: data,
              timestamp: data.timestamp,
              task_id: data.task_id,
            }
          } else if (data.type === "task_waiting_for_user") {
            message = {
              type: "task_waiting_for_user",
              data: data,
              timestamp: data.timestamp,
              task_id: data.task_id,
            }
          } else if (data.type === "task_resumed") {
            message = {
              type: "task_resumed",
              data: data,
              timestamp: data.timestamp,
              task_id: data.task_id,
            }
          } else if (data.type === "agent_error") {
            message = {
              type: "agent_error",
              data: data,
              timestamp: data.timestamp,
              task_id: data.task_id,
            }
          } else if (data.type === "historical_data_complete") {
            message = {
              type: "historical_data_complete",
              data: data,
              timestamp: data.timestamp,
              task_id: data.task_id,
            }
          } else {
            // Generic message handling
            const messageData = data.data || data;
            // Ensure we don't pass empty strings where objects are expected
            const safeData = typeof messageData === 'string' && messageData === ''
              ? {}
              : messageData;

            message = {
              type: data.type || "message",
              data: safeData,
              timestamp: data.timestamp || new Date().toISOString(),
              task_id: data.task_id,
              step_id: data.step_id,
            }
          }

          // Preserve the canonical task-control envelope even when a message
          // type normalizes its payload into ``data`` above.
          message.run_id = data.run_id
          message.state_version = data.state_version
          message.control_state = data.control_state
          message.status = data.status
          message.task = data.task

          setLastMessage(message)
          onMessage?.(message)
        } catch (error) {
          console.error("Error parsing WebSocket message", error)
        }
      }

    } catch (error) {
      console.error('Failed to create WebSocket connection', error)
      const connectionError = error instanceof Error ? error : new Error('Failed to create WebSocket connection')
      setConnectionError(connectionError)
      onError?.(connectionError)
    }
  }, [url, taskId, token, authToken, onConnect, onDisconnect, onError, rejectPendingDeliveries])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    if (socketRef.current) {
      socketRef.current.close()
      socketRef.current = null
    }
    rejectPendingDeliveries(new Error('Disconnected before the message was accepted.'))
    setIsConnected(false)
    isConnectingRef.current = false
  }, [rejectPendingDeliveries])

  // Update taskId ref when taskId changes
  useEffect(() => {
    // If taskId changes, clear any previous connection errors to allow fresh connection attempt
    if (taskId !== taskIdRef.current) {
      setConnectionError(null)
    }

    // If taskId changes and we are connected, disconnect to ensure we connect to the new task
    // logic: if we have a new taskId (different from ref) and we are currently connected
    if (taskId && taskId !== taskIdRef.current && isConnected) {
      disconnect()
    }

    taskIdRef.current = taskId
  }, [taskId, isConnected, disconnect])

  const sendMessage = useCallback((message: Record<string, unknown>) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(message))
    }
  }, [])

  const sendChatMessage = useCallback(async (
    message: string,
    files?: File[],
    force: boolean = false,
    requestedClientMessageId?: string,
  ): Promise<MessageDeliveryAck> => {
    const timestamp = Date.now()
    const currentTaskId = taskIdRef.current
    const socket = socketRef.current
    if (!currentTaskId || socket?.readyState !== WebSocket.OPEN) {
      throw new Error('Message not sent: the task connection is not ready.')
    }

    const clientMessageId = requestedClientMessageId || generateClientMessageId()
    const duplicateMessage = recentMessages.find(
      msg => (
        msg.taskId === currentTaskId
        && msg.message === message
        && msg.clientMessageId !== clientMessageId
        && (timestamp - msg.timestamp) < MESSAGE_DUPLICATE_THRESHOLD
      )
    )
    const duplicateIsPending = duplicateMessage
      ? pendingDeliveriesRef.current.has(duplicateMessage.clientMessageId)
      : false
    if (!force && duplicateIsPending) {
      throw new Error('Duplicate message ignored while the previous send is pending.')
    }

    const messageData: Record<string, unknown> = {
      type: 'chat',
      message,
      task_id: currentTaskId,
      client_message_id: clientMessageId,
    }

    if (files && files.length > 0) {
      type FileWithUploadId = File & { file_id?: string }
      const filesWithUploadIds = files as FileWithUploadId[]
      const filesToUpload = filesWithUploadIds.filter(file => !file.file_id)
      const preUploadedFiles = filesWithUploadIds
        .filter((file): file is FileWithUploadId & { file_id: string } => Boolean(file.file_id))
        .map(file => ({
          file_id: file.file_id,
          name: file.name,
          size: file.size,
          type: file.type || '',
        }))
      let uploadedFiles: Array<{ file_id: string; name?: string; size?: number; type?: string }> = []

      if (filesToUpload.length > 0 && uploadFiles) {
        uploadedFiles = await uploadFiles(filesToUpload, {
          taskId: currentTaskId,
          taskType: 'task',
        })
      } else if (filesToUpload.length > 0) {
        const formData = new FormData()
        filesToUpload.forEach(file => formData.append('files', file))
        formData.append('task_type', 'task')
        formData.append('task_id', currentTaskId.toString())
        const response = await apiRequest(`${getUploadApiUrl()}/api/files/upload`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${tokenRef.current || localStorage.getItem('token') || ''}`,
          },
          body: formData,
        })
        const parsed = await parseApiResponse(response)
        if (!response.ok || !isJsonRecord(parsed.data)) {
          throw new Error(getUploadErrorMessage(response, parsed, {
            generic: 'Upload failed',
            ...UPLOAD_ERROR_MESSAGES,
          }))
        }
        const data = parsed.data
        uploadedFiles = data.success && Array.isArray(data.files)
          ? data.files
            .filter((file): file is { file_id: string; filename?: string; file_size?: number; mime_type?: string } => (
              isJsonRecord(file) && typeof file.file_id === 'string'
            ))
            .map(file => ({
              file_id: file.file_id,
              name: typeof file.filename === 'string' ? file.filename : '',
              size: typeof file.file_size === 'number' ? file.file_size : 0,
              type: typeof file.mime_type === 'string' ? file.mime_type : '',
            }))
          : []
      }
      messageData.files = [...preUploadedFiles, ...uploadedFiles]
    }

    const delivery = new Promise<MessageDeliveryAck>((resolve, reject) => {
      const timeout = setTimeout(() => {
        pendingDeliveriesRef.current.delete(clientMessageId)
        reject(new Error('Message delivery was not acknowledged. Your draft was kept.'))
      }, 30000)
      pendingDeliveriesRef.current.set(clientMessageId, { resolve, reject, timeout })
    })

    try {
      socket.send(JSON.stringify(messageData))
    } catch (error) {
      const pending = pendingDeliveriesRef.current.get(clientMessageId)
      if (pending) {
        clearTimeout(pending.timeout)
        pendingDeliveriesRef.current.delete(clientMessageId)
        pending.reject(error instanceof Error ? error : new Error(String(error)))
      }
      return delivery
    }

    recentMessages.push({
      message,
      timestamp,
      taskId: currentTaskId,
      clientMessageId,
    })
    const cutoffTime = timestamp - 5000
    const firstKeepIndex = recentMessages.findIndex(item => item.timestamp >= cutoffTime)
    if (firstKeepIndex === -1) {
      recentMessages.splice(0, recentMessages.length)
    } else if (firstKeepIndex > 0) {
      recentMessages.splice(0, firstKeepIndex)
    }

    return delivery
  }, [uploadFiles])

  const executeTask = useCallback((taskDescription: string, files?: Array<{ name: string; type: string; size: number; content?: string }>) => {
    if (socketRef.current?.readyState === WebSocket.OPEN && taskIdRef.current) {
      const message = JSON.stringify({
        type: "execute_task",
        task_id: taskIdRef.current,
        description: taskDescription,
        ...(files && files.length > 0 && { files })
      })
      socketRef.current.send(message)
    }
  }, [taskId])

  const pauseTask = useCallback(() => {
    if (socketRef.current?.readyState === WebSocket.OPEN && taskIdRef.current) {
      const message = {
        type: "pause_task",
        task_id: taskIdRef.current,
        command_id: generateClientMessageId(),
      }
      socketRef.current.send(JSON.stringify(message))
    }
  }, [taskId])

  const resumeTask = useCallback(() => {
    if (socketRef.current?.readyState === WebSocket.OPEN && taskIdRef.current) {
      socketRef.current.send(JSON.stringify({
        type: "resume_task",
        task_id: taskIdRef.current,
        command_id: generateClientMessageId(),
      }))
    }
  }, [taskId])

  const requestStatus = useCallback(() => {
    if (socketRef.current?.readyState === WebSocket.OPEN && taskIdRef.current) {
      socketRef.current.send(JSON.stringify({
        type: "status_request",
        task_id: taskIdRef.current,
      }))
    }
  }, [taskId])


  useEffect(() => {
    // Only attempt to connect when taskId changes and autoConnect is enabled
    // We also check connectionError to avoid infinite loops, but we need to react when it's cleared
    // Note: We don't check !isConnected here because:
    // 1. connect() has its own guard checks
    // 2. When switching tasks, isConnected might still be true from the previous task in this render cycle,
    //    preventing the new connection if we check it here.
    if (autoConnect && taskId && !connectionError && !isConnectingRef.current) {
      connect()
    }

    return () => {
      // Clean up on unmount or when dependencies change
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
        reconnectTimeoutRef.current = null
      }
      // Close WebSocket connection to prevent port closed errors
      if (socketRef.current) {
        socketRef.current.close(1000, 'Component unmounting')
        socketRef.current = null
      }
      setIsConnected(false)
      isConnectingRef.current = false
    }
  }, [url, taskId, token, authToken, autoConnect, connectionError]) // Added connectionError to dependencies

  // Separate effect to handle connection state changes
  useEffect(() => {
    if (isConnected) {
      reconnectAttemptsRef.current = 0 // Reset attempts on successful connection
    }
  }, [isConnected])

  return {
    isConnected,
    lastMessage,
    connectionError,
    connect,
    disconnect,
    sendMessage,
    sendChatMessage,
    executeTask,
    pauseTask,
    resumeTask,
    requestStatus,
  }
}

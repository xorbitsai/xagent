"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import { useAuth } from "@/contexts/auth-context"
import { getWsUrl } from "@/lib/utils"

// Duplicate message detection: record recently sent messages
const recentMessages: Array<{ message: string; timestamp: number; taskId: number }> = []
const MESSAGE_DUPLICATE_THRESHOLD = 2000 // Same message within 2 seconds is considered duplicate

interface WebSocketMessage {
  type: string
  data: unknown
  timestamp: string
  task_id?: number
  step_id?: string
  event_id?: string
  event_type?: string
}

interface UseWebSocketOptions {
  url?: string
  taskId?: number
  token?: string
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
  const maxReconnectAttempts = 3

  // Update taskId ref when taskId changes
  useEffect(() => {
    console.log('🔄 taskId useEffect called:', {
      currentTaskId: taskId,
      currentRefValue: taskIdRef.current,
      taskIdType: typeof taskId
    })
    taskIdRef.current = taskId
    console.log('🔄 taskIdRef updated:', taskId)
  }, [taskId])

  // Update token ref when token changes
  useEffect(() => {
    console.log('🔄 token useEffect called:', {
      currentToken: token,
      currentRefValue: tokenRef.current,
      tokenType: typeof token
    })
    tokenRef.current = token || authToken
    console.log('🔄 tokenRef updated:', tokenRef.current)
  }, [token, authToken])

  // Update token ref when auth token changes (for refresh token support)
  useEffect(() => {
    if (!token && authToken) {
      console.log('🔄 Auth token changed, updating WebSocket token:', authToken)
      tokenRef.current = authToken

      // If WebSocket is connected and we got a new token, reconnect with new token
      if (socketRef.current?.readyState === WebSocket.OPEN && taskId) {
        console.log('🔄 Reconnecting WebSocket with new auth token')
        disconnect()
        setTimeout(() => {
          connect()
        }, 1000)
      }
    }
  }, [authToken, token, taskId])

  const connect = useCallback(() => {
    console.log('🔧 Connect called:', {
      currentSocket: socketRef.current,
      readyState: socketRef.current?.readyState,
      isConnecting: isConnectingRef.current,
      taskId
    })

    if (socketRef.current?.readyState === WebSocket.OPEN || isConnectingRef.current) return
    isConnectingRef.current = true

    try {
      // Don't try to connect if there's no task ID
      if (!taskId) {
        isConnectingRef.current = false
        return
      }

      const wsUrl = `${url}/ws/chat/${taskId}${tokenRef.current ? `?token=${tokenRef.current}` : ''}`
      console.log('🚀 Attempting to connect to WebSocket:', wsUrl)

      // Test if the URL is valid before creating WebSocket
      if (!wsUrl.startsWith('ws://') && !wsUrl.startsWith('wss://')) {
        throw new Error(`Invalid WebSocket URL: ${wsUrl}`)
      }

      const socket = new WebSocket(wsUrl)
      socketRef.current = socket

      socket.onopen = () => {
        console.log('✅ WebSocket connection established successfully')
        // Store socket reference for debugging
        ;(window as any).debugWebSocket = socket
        console.log('📊 WebSocket state:', {
          readyState: socket.readyState,
          url: socket.url,
          protocol: socket.protocol,
          extensions: socket.extensions,
          taskId: taskId
        })
        console.log('🎯 About to set isConnected to true')
        setIsConnected(true)
        console.log('✅ isConnected set to true')
        setConnectionError(null)
        reconnectAttemptsRef.current = 0
        isConnectingRef.current = false
        onConnect?.()
      }

      socket.onclose = (event) => {
        console.log('❌ WebSocket connection closed:', event.code, event.reason)
        console.log('📊 Close event details:', {
          code: event.code,
          reason: event.reason,
          wasClean: event.wasClean,
          target: event.target
        })
        setIsConnected(false)
        isConnectingRef.current = false
        onDisconnect?.()

        // Handle authentication errors (4001 = Authentication required)
        if (event.code === 4001) {
          console.log('🔐 WebSocket authentication failed, trying token refresh')
          if (authRefreshToken && typeof authRefreshToken === 'function') {
            try {
              console.log('🔄 Attempting to refresh auth token for WebSocket')
              const refreshTokenFunc = authRefreshToken as () => Promise<boolean>
              refreshTokenFunc().then(refreshSuccess => {
                if (refreshSuccess) {
                  console.log('✅ Auth token refreshed successfully, will reconnect WebSocket')
                  setTimeout(() => {
                    if (taskIdRef.current) {
                      connect()
                    }
                  }, 1000)
                } else {
                  console.log('❌ Auth token refresh failed, WebSocket connection will not be restored')
                  onError?.(new Error('Authentication failed and token refresh failed'))
                }
              }).catch(error => {
                console.error('❌ Error during auth token refresh for WebSocket:', error)
                onError?.(new Error('Authentication failed and token refresh error'))
              })
            } catch (error) {
              console.error('❌ Error during auth token refresh for WebSocket:', error)
              onError?.(new Error('Authentication failed and token refresh error'))
            }
          } else {
            console.log('❌ No refresh token available, cannot recover WebSocket connection')
            onError?.(new Error('Authentication failed and no refresh token available'))
          }
          return
        }

        // Don't reconnect if it's a 404 error or abnormal closure (1006)
        if (event.code === 1006) {
          console.log('🚫 WebSocket endpoint not available, stopping reconnection attempts')
          return
        }

        // Don't reconnect if it's a clean close (might be intentional)
        if (event.code === 1000) {
          console.log('✅ WebSocket connection closed normally, not reconnecting')
          return
        }

        // Don't reconnect if the reason is component unmounting
        if (event.reason === 'Component unmounting') {
          console.log('🛑 Component unmounting, not reconnecting')
          return
        }

        // Only attempt to reconnect if under max attempts and taskId exists
        if (reconnectAttemptsRef.current < maxReconnectAttempts && taskId) {
          reconnectAttemptsRef.current++
          const delay = Math.min(1000 * reconnectAttemptsRef.current, 5000)
          console.log(`🔄 Attempting to reconnect in ${delay}ms... (attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts})`)
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, delay)
        } else {
          console.log('🛑 Max reconnection attempts reached or no taskId, stopping reconnection')
        }
      }

      socket.onerror = (error) => {
        console.log('❌ WebSocket error:', error)
        console.log('❌ WebSocket error details:', {
          type: error.type,
          target: error.target,
          currentTarget: error.currentTarget,
          isTrusted: error.isTrusted,
          readyState: socket.readyState,
          url: wsUrl
        })

        const connectionError = new Error(`WebSocket connection failed to ${wsUrl}. The backend WebSocket endpoint may not be available.`)
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
        console.log('🔥🔥🔥 Raw WebSocket Data 🔥🔥🔥', event.data)
        try {
          const data = JSON.parse(event.data)

          // Handle different message types from the backend
          let message: WebSocketMessage

          if (data.type === "trace_event") {
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

          setLastMessage(message)
          onMessage?.(message)
        } catch (error) {
          console.log("Error parsing WebSocket message:", error)
        }
      }

    } catch (error) {
      console.log('Failed to create WebSocket connection:', error)
      const connectionError = error instanceof Error ? error : new Error('Failed to create WebSocket connection')
      setConnectionError(connectionError)
      onError?.(connectionError)
    }
  }, [url, taskId, token, authToken, onConnect, onDisconnect, onError])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    if (socketRef.current) {
      socketRef.current.close()
      socketRef.current = null
    }
    setIsConnected(false)
    isConnectingRef.current = false
  }, [])

  const sendMessage = useCallback((message: Record<string, unknown>) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(message))
    }
  }, [])

  const sendChatMessage = useCallback((message: string, files?: File[], force: boolean = false) => {
    const timestamp = Date.now()
    console.log(`🚀 sendChatMessage called [${timestamp}]:`, { message, files: files?.map(f => f.name) })

    // Brute force duplicate message detection
    const currentTaskId = taskIdRef.current
    const duplicateMessage = recentMessages.find(
      msg => msg.taskId === currentTaskId && msg.message === message && (timestamp - msg.timestamp) < MESSAGE_DUPLICATE_THRESHOLD
    )

    if (!force && duplicateMessage) {
      console.warn(
        `🚨 DUPLICATE MESSAGE DETECTED (Silently ignored)!\n` +
        `Message: "${message}"\n` +
        `Previous send time: ${duplicateMessage.timestamp} (${timestamp - duplicateMessage.timestamp}ms ago)\n` +
        `Current send time: ${timestamp}\n` +
        `Task ID: ${currentTaskId}`
      )
      return
    }

    if (socketRef.current?.readyState === WebSocket.OPEN && taskIdRef.current) {
      const messageData: any = {
        type: "chat",
        message,
        task_id: taskIdRef.current,
      }

      // If there are files, add file info
      if (files && files.length > 0) {
        console.log(`📁 Processing files [${timestamp}]:`, files.length)

        const filePromises = files.map(file => {
          return new Promise((resolve) => {
            const reader = new FileReader()
            reader.onload = (e) => {
              const fileData = {
                name: file.name,
                size: file.size,
                type: file.type,
                content: typeof e.target?.result === 'string' ? e.target.result.split(',')[1] : '' // Get base64 content
              }
              console.log(`📄 File processed [${timestamp}]:`, fileData.name, 'size:', fileData.size)
              resolve(fileData)
            }
            reader.readAsDataURL(file)
          })
        })

        // Wait for all files to be read
        Promise.all(filePromises).then(fileDataList => {
          console.log(`✅ All files processed [${timestamp}], sending:`, fileDataList.length)
          messageData.files = fileDataList
          console.log(`📤 Sending message with files [${timestamp}]:`, messageData)
          socketRef.current?.send(JSON.stringify(messageData))
          console.log(`✅ Message sent [${timestamp}]`)

          // Record sent message
          recentMessages.push({ message, timestamp, taskId: currentTaskId! })
          // Clear records older than 5 seconds
          const cutoffTime = timestamp - 5000
          const firstKeepIndex = recentMessages.findIndex(msg => msg.timestamp >= cutoffTime)
          if (firstKeepIndex === -1) {
            recentMessages.splice(0, recentMessages.length)
          } else if (firstKeepIndex > 0) {
            recentMessages.splice(0, firstKeepIndex)
          }
        })
      } else {
        console.log(`📤 Sending message without files [${timestamp}]`)
        socketRef.current?.send(JSON.stringify(messageData))
        console.log(`✅ Message sent [${timestamp}]`)

        // Record sent message
        recentMessages.push({ message, timestamp, taskId: currentTaskId! })
        // Clear records older than 5 seconds
        const cutoffTime = timestamp - 5000
        const firstKeepIndex = recentMessages.findIndex(msg => msg.timestamp >= cutoffTime)
        if (firstKeepIndex === -1) {
          recentMessages.splice(0, recentMessages.length)
        } else if (firstKeepIndex > 0) {
          recentMessages.splice(0, firstKeepIndex)
        }
      }
    } else {
      console.log('❌ Cannot send message - WebSocket not ready or no task ID:', {
        readyState: socketRef.current?.readyState,
        taskId: taskIdRef.current
      })
    }
  }, [taskId])

  const executeTask = useCallback((taskDescription: string, files?: Array<{ name: string; type: string; size: number; content?: string }>) => {
    console.log('🔧 executeTask called:', {
      taskDescription,
      taskId: taskIdRef.current,
      files: files?.length || 0,
      readyState: socketRef.current?.readyState,
      readyStateText: socketRef.current ? ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED'][socketRef.current.readyState] : 'NO_SOCKET',
      isOpen: socketRef.current?.readyState === WebSocket.OPEN
    })

    if (socketRef.current?.readyState === WebSocket.OPEN && taskIdRef.current) {
      const message = JSON.stringify({
        type: "execute_task",
        task_id: taskIdRef.current,
        description: taskDescription,
        ...(files && files.length > 0 && { files })
      })
      console.log('📤 Sending execute_task message:', message)
      socketRef.current.send(message)
    } else {
      console.log('❌ Cannot send execute_task - WebSocket not open or no taskId')
      console.log('🔍 Debug info:', {
        hasSocket: !!socketRef.current,
        readyState: socketRef.current?.readyState,
        taskId: taskIdRef.current
      })
    }
  }, [taskId])

  const pauseTask = useCallback(() => {
    console.log('🔘 pauseTask called:', {
      socketReady: socketRef.current?.readyState === WebSocket.OPEN,
      taskId: taskIdRef.current,
      socketState: socketRef.current?.readyState
    })
    if (socketRef.current?.readyState === WebSocket.OPEN && taskIdRef.current) {
      const message = {
        type: "pause_task",
        task_id: taskIdRef.current,
      }
      console.log('📤 Sending pause_task message:', message)
      socketRef.current.send(JSON.stringify(message))
      console.log('✅ pause_task message sent')
    } else {
      console.warn('⚠️ Cannot send pause_task: socket not ready or no taskId')
    }
  }, [taskId])

  const resumeTask = useCallback(() => {
    if (socketRef.current?.readyState === WebSocket.OPEN && taskIdRef.current) {
      socketRef.current.send(JSON.stringify({
        type: "resume_task",
        task_id: taskIdRef.current,
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
    if (autoConnect && taskId && !isConnected && !connectionError && !isConnectingRef.current) {
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
  }, [url, taskId, token, authToken, autoConnect]) // Only connect when taskId or URL changes

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

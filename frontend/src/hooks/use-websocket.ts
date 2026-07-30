"use client"

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import { useAuth } from "@/contexts/auth-context"
import type { AuthSessionSnapshot } from "@/lib/auth-cache"
import { apiRequest, getUploadErrorMessage, isJsonRecord, parseApiResponse, UPLOAD_ERROR_MESSAGES } from "@/lib/api-wrapper"
import { generateClientMessageId, getWsUrl, getUploadApiUrl } from "@/lib/utils"
import { isFinalAnswerStreamEventType } from "@/lib/streaming-final-answer"

interface RecentMessage {
  message: string
  timestamp: number
  connectionIdentity: string
  descriptorKey: ConnectionDescriptorIdentity
  lifecycleEpoch: number
  attemptEpoch: number
  deliveryGeneration: number
  clientMessageId: string
}

const MESSAGE_DUPLICATE_THRESHOLD = 2000 // Same message within 2 seconds is considered duplicate
const HANDSHAKE_TIMEOUT_MS = 10_000
const MAX_AUTH_REFRESH_RETRIES = 3

// Connection values may carry credentials. Lifecycle fencing keeps the
// normalized connection object opaque; it is never serialized or hashed.
type ConnectionDescriptorIdentity = WebSocketConnection

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

export type MessageDeliveryDisposition = "not_sent" | "rejected" | "outcome_unknown"

export class MessageDeliveryError extends Error {
  readonly disposition: MessageDeliveryDisposition
  readonly retryWithNewId: boolean

  constructor(
    message: string,
    disposition: MessageDeliveryDisposition,
    retryWithNewId = false,
  ) {
    super(message)
    this.name = "MessageDeliveryError"
    this.disposition = disposition
    this.retryWithNewId = retryWithNewId
  }
}

const deliveryError = (
  message: string,
  disposition: MessageDeliveryDisposition,
  retryWithNewId = false,
) => new MessageDeliveryError(message, disposition, retryWithNewId)

export type WebSocketCredentialOwner =
  | {
    kind: "auth-context"
    accessToken: string
    userId: string | null
    session?: AuthSessionSnapshot
  }
  | { kind: "external" }

const getInternalSessionId = (
  connection: WebSocketConnection | null,
): string | null => {
  const credentialOwner = connection?.credentialOwner
  return credentialOwner?.kind === "auth-context"
    ? credentialOwner.session?.sessionId ?? null
    : null
}

export interface WebSocketConnection {
  identity: string
  url: string
  protocols?: string[]
  expectedProtocol?: string
  taskId?: number
  chatTaskIdMode: "required" | "omit"
  credentialOwner: WebSocketCredentialOwner
}

export type WebSocketSendResult = "sent" | "not_sent"

export interface WebSocketConnectionFailure {
  recoverable: boolean
  error: Error
}

interface PendingDelivery {
  resolve: (ack: MessageDeliveryAck) => void
  reject: (error: Error) => void
  timeout: ReturnType<typeof setTimeout>
  connectionIdentity: string
  descriptorKey: ConnectionDescriptorIdentity
  lifecycleEpoch: number
  attemptEpoch: number
  deliveryGeneration: number
  socket: WebSocket
}

interface MessagePreparationClaim {
  cancellation: Promise<never>
  cancel: (error: Error) => void
  cancelled: boolean
  connectionIdentity: string
  descriptorKey: ConnectionDescriptorIdentity
  lifecycleEpoch: number
  attemptEpoch: number
  deliveryGeneration: number
  socket: WebSocket
}

interface WebSocketCallbacks {
  onConnectionClose?: (event: CloseEvent) => "handled" | "default"
  onConnectionFailure?: (failure: WebSocketConnectionFailure) => void
  onSessionConnectionClose?: (
    event: CloseEvent,
    connectionIdentity: string,
  ) => "handled" | "default"
  onSessionConnectionFailure?: (
    failure: WebSocketConnectionFailure,
    connectionIdentity: string,
  ) => void
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Error) => void
  onMessage?: (message: WebSocketMessage) => void
}

const reportConnectionFailure = (
  callbacks: WebSocketCallbacks,
  failure: WebSocketConnectionFailure,
  connectionIdentity?: string,
) => {
  const invokeSafely = <T,>(
    callback: ((value: T) => void) | undefined,
    value: T,
  ) => {
    if (!callback) return
    try {
      callback(value)
    } catch {
      console.error("WebSocket connection failure callback failed")
    }
  }

  if (connectionIdentity && callbacks.onSessionConnectionFailure) {
    invokeSafely(
      (value) => callbacks.onSessionConnectionFailure?.(value, connectionIdentity),
      failure,
    )
  } else {
    invokeSafely(callbacks.onConnectionFailure, failure)
  }
  invokeSafely(callbacks.onError, failure.error)
}

interface SocketOwner {
  socket: WebSocket
  connection: WebSocketConnection
  descriptorKey: ConnectionDescriptorIdentity
  lifecycleEpoch: number
  attemptEpoch: number
  callbacks: WebSocketCallbacks
  refreshAccessToken: (
    expectedSession?: AuthSessionSnapshot,
  ) => Promise<boolean>
  disconnectNotified: boolean
  handshakeTimer: ReturnType<typeof setTimeout> | null
}

interface OwnerRetirementOptions {
  pendingError: Error
  preparationError: Error
  close?: { code?: number; reason?: string }
  notifyDisconnect: boolean
}

interface ScheduledRetry {
  lifecycleEpoch: number
  attemptEpoch: number
}

export interface UseWebSocketOptions {
  url?: string
  taskId?: number
  token?: string
  buildWebSocketUrl?: (params: { baseUrl: string; taskId: number; token?: string }) => string
  uploadFiles?: (files: File[], params: { taskId?: number | null; taskType: string }) => Promise<Array<{ file_id: string; name?: string; size?: number; type?: string }>>
  connection?: WebSocketConnection | null
  deliveryGeneration?: number
  onConnectionClose?: (event: CloseEvent) => "handled" | "default"
  onConnectionFailure?: (failure: WebSocketConnectionFailure) => void
  onSessionConnectionClose?: (
    event: CloseEvent,
    connectionIdentity: string,
  ) => "handled" | "default"
  onSessionConnectionFailure?: (
    failure: WebSocketConnectionFailure,
    connectionIdentity: string,
  ) => void
  autoConnect?: boolean
  onMessage?: (message: WebSocketMessage) => void
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Error) => void
}

const useIsomorphicLayoutEffect = typeof window === "undefined"
  ? useEffect
  : useLayoutEffect

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    url,
    taskId,
    token,
    buildWebSocketUrl,
    uploadFiles,
    connection: connectionOption,
    deliveryGeneration = 0,
    onConnectionClose,
    onConnectionFailure,
    onSessionConnectionClose,
    onSessionConnectionFailure,
    autoConnect = true,
    onMessage,
    onConnect,
    onDisconnect,
    onError,
  } = options

  const {
    token: authToken,
    user: authUser,
    session: authSession,
    refreshAccessToken,
  } = useAuth()

  const normalizedConnection = useMemo<WebSocketConnection | null>(() => {
    if (connectionOption !== undefined) return connectionOption
    if (!taskId) return null

    const baseUrl = url ?? getWsUrl()
    const hasExplicitToken = token !== undefined
    const effectiveToken = hasExplicitToken ? token : authToken || undefined
    return {
      identity: `legacy-task:${taskId}`,
      url: buildWebSocketUrl
        ? buildWebSocketUrl({
          baseUrl,
          taskId,
          token: effectiveToken,
        })
        : `${baseUrl}/ws/chat/${taskId}${effectiveToken ? `?token=${effectiveToken}` : ""}`,
      taskId,
      chatTaskIdMode: "required",
      credentialOwner: !hasExplicitToken && authToken
        ? {
          kind: "auth-context",
          accessToken: authToken,
          userId: authUser?.id ? String(authUser.id) : null,
          session: authSession,
        }
        : { kind: "external" },
    }
  }, [
    authToken,
    authUser?.id,
    authSession,
    buildWebSocketUrl,
    connectionOption,
    taskId,
    token,
    url,
  ])
  const connectionDescriptorIdentity = normalizedConnection

  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const [connectionError, setConnectionError] = useState<Error | null>(null)
  const isConnectingRef = useRef(false)

  const socketRef = useRef<WebSocket | null>(null)
  const socketOwnerRef = useRef<SocketOwner | null>(null)
  const connectionRef = useRef<WebSocketConnection | null>(normalizedConnection)
  const descriptorKeyRef = useRef<ConnectionDescriptorIdentity | null>(connectionDescriptorIdentity)
  const retryTimersRef = useRef(new Map<ReturnType<typeof setTimeout>, ScheduledRetry>())
  const reconnectAttemptsRef = useRef(0)
  const authRefreshRetriesRef = useRef(0)
  const authRefreshRetryBudgetSessionIdRef = useRef(getInternalSessionId(normalizedConnection))
  const deliveryGenerationRef = useRef(deliveryGeneration)
  const deliveryIdentityRef = useRef(normalizedConnection?.identity ?? null)
  const lifecycleEpochRef = useRef(0)
  const attemptEpochRef = useRef(0)
  const mountedRef = useRef(false)
  const tokenRef = useRef(token !== undefined ? token : authToken)
  const pendingDeliveriesRef = useRef(new Map<string, PendingDelivery>())
  const preparationsRef = useRef(new Map<string, MessagePreparationClaim>())
  const recentMessagesRef = useRef<RecentMessage[]>([])
  const callbacksRef = useRef<WebSocketCallbacks>({
    onConnectionClose,
    onConnectionFailure,
    onSessionConnectionClose,
    onSessionConnectionFailure,
    onConnect,
    onDisconnect,
    onError,
    onMessage,
  })
  const refreshAccessTokenRef = useRef(refreshAccessToken)
  const maxReconnectAttempts = 3

  const rejectPendingDeliveries = useCallback((
    error: Error,
    matches: (pending: PendingDelivery) => boolean = () => true,
  ) => {
    for (const [clientMessageId, pending] of pendingDeliveriesRef.current) {
      if (!matches(pending)) continue
      clearTimeout(pending.timeout)
      pending.reject(error)
      pendingDeliveriesRef.current.delete(clientMessageId)
    }
  }, [])

  const rejectPreparations = useCallback((
    error: Error,
    matches: (claim: MessagePreparationClaim) => boolean = () => true,
  ) => {
    for (const [clientMessageId, claim] of preparationsRef.current) {
      if (!matches(claim)) continue
      if (preparationsRef.current.get(clientMessageId) !== claim) continue
      preparationsRef.current.delete(clientMessageId)
      claim.cancel(error)
    }
  }, [])

  const clearRecentMessages = useCallback((
    matches: (recent: RecentMessage) => boolean = () => true,
  ) => {
    recentMessagesRef.current = recentMessagesRef.current.filter(
      recent => !matches(recent),
    )
  }, [])

  const clearRetryTimers = useCallback((
    matches: (retry: ScheduledRetry) => boolean = () => true,
  ) => {
    for (const [timer, retry] of retryTimersRef.current) {
      if (!matches(retry)) continue
      clearTimeout(timer)
      retryTimersRef.current.delete(timer)
    }
  }, [])

  const invalidateLifecycle = useCallback(() => {
    lifecycleEpochRef.current++
    clearRetryTimers()
  }, [clearRetryTimers])

  const isCurrentLifecycle = useCallback((
    lifecycleEpoch: number,
  ) => (
    mountedRef.current
    && lifecycleEpochRef.current === lifecycleEpoch
  ), [])

  // Socket callbacks require this exact owner. Async refresh/retry work may
  // outlive retirement, so it is fenced by the narrower attempt predicate.
  const isCurrentOwner = useCallback((owner: SocketOwner) => (
    isCurrentLifecycle(owner.lifecycleEpoch)
    && socketOwnerRef.current === owner
    && socketRef.current === owner.socket
  ), [isCurrentLifecycle])

  const isCurrentAttempt = useCallback((owner: SocketOwner) => (
    isCurrentLifecycle(owner.lifecycleEpoch)
    && attemptEpochRef.current === owner.attemptEpoch
  ), [isCurrentLifecycle])

  const canApplyRetiredOwnerPolicy = useCallback((owner: SocketOwner) => (
    isCurrentAttempt(owner)
    && socketOwnerRef.current === null
    && socketRef.current === null
  ), [isCurrentAttempt])

  const isCurrentSocket = useCallback((socket: WebSocket, identity: string) => {
    const owner = socketOwnerRef.current
    return Boolean(
      owner
      && owner.socket === socket
      && owner.connection.identity === identity
      && isCurrentOwner(owner),
    )
  }, [isCurrentOwner])

  const notifyDisconnect = useCallback((owner: SocketOwner) => {
    if (owner.disconnectNotified) return
    owner.disconnectNotified = true
    try {
      owner.callbacks.onDisconnect?.()
    } catch {
      console.error("WebSocket disconnect handler failed")
    }
  }, [])

  const reportPermanentFailure = useCallback((
    owner: SocketOwner,
    error: Error,
  ) => {
    if (!canApplyRetiredOwnerPolicy(owner)) return
    setConnectionError(error)
    reportConnectionFailure(owner.callbacks, {
      recoverable: false,
      error,
    }, owner.connection.identity)
  }, [canApplyRetiredOwnerPolicy])

  const retireOwnerCore = useCallback((
    owner: SocketOwner,
    options: OwnerRetirementOptions,
  ) => {
    if (owner.handshakeTimer) {
      clearTimeout(owner.handshakeTimer)
      owner.handshakeTimer = null
    }
    const wasCurrent = (
      socketOwnerRef.current === owner
      && socketRef.current === owner.socket
    )
    if (wasCurrent) {
      clearRetryTimers(retry => (
        retry.lifecycleEpoch === owner.lifecycleEpoch
        && retry.attemptEpoch === owner.attemptEpoch
      ))
      socketRef.current = null
      socketOwnerRef.current = null
      if (mountedRef.current) setIsConnected(false)
      isConnectingRef.current = false
    }
    rejectPendingDeliveries(
      options.pendingError,
      pending => (
        pending.socket === owner.socket
        && pending.descriptorKey === owner.descriptorKey
        && pending.lifecycleEpoch === owner.lifecycleEpoch
        && pending.attemptEpoch === owner.attemptEpoch
      ),
    )
    rejectPreparations(
      options.preparationError,
      claim => (
        claim.socket === owner.socket
        && claim.descriptorKey === owner.descriptorKey
        && claim.lifecycleEpoch === owner.lifecycleEpoch
        && claim.attemptEpoch === owner.attemptEpoch
      ),
    )
    clearRecentMessages(
      recent => (
        recent.descriptorKey === owner.descriptorKey
        && recent.lifecycleEpoch === owner.lifecycleEpoch
        && recent.attemptEpoch === owner.attemptEpoch
      ),
    )
    return wasCurrent
  }, [
    clearRecentMessages,
    clearRetryTimers,
    rejectPendingDeliveries,
    rejectPreparations,
  ])

  const retireOwner = useCallback((
    owner: SocketOwner,
    options: OwnerRetirementOptions,
  ) => {
    const wasCurrent = retireOwnerCore(owner, options)
    if (!wasCurrent) return false

    if (
      options.close
      && owner.socket.readyState !== WebSocket.CLOSED
    ) {
      try {
        owner.socket.close(options.close.code, options.close.reason)
      } catch {
        console.error("WebSocket close failed")
      }
    }
    if (options.notifyDisconnect) notifyDisconnect(owner)
    return true
  }, [
    notifyDisconnect,
    retireOwnerCore,
  ])

  const scheduleRetry = useCallback((
    callback: () => void,
    delay: number,
    lifecycleEpoch: number,
    attemptEpoch: number,
  ) => {
    const timer = setTimeout(() => {
      retryTimersRef.current.delete(timer)
      if (
        !isCurrentLifecycle(lifecycleEpoch)
        || attemptEpochRef.current !== attemptEpoch
      ) return
      callback()
    }, delay)
    retryTimersRef.current.set(timer, { lifecycleEpoch, attemptEpoch })
  }, [isCurrentLifecycle])

  useIsomorphicLayoutEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      invalidateLifecycle()
    }
  }, [invalidateLifecycle])

  useIsomorphicLayoutEffect(() => {
    const previousDescriptorKey = descriptorKeyRef.current
    const previousIdentity = deliveryIdentityRef.current
    const previousGeneration = deliveryGenerationRef.current
    const nextIdentity = normalizedConnection?.identity ?? null
    const nextAuthSessionId = getInternalSessionId(normalizedConnection)

    if (previousDescriptorKey !== connectionDescriptorIdentity) {
      invalidateLifecycle()
      if (previousDescriptorKey !== null) {
        rejectPreparations(
          deliveryError("Message not sent: the connection changed before delivery.", "not_sent"),
          claim => claim.descriptorKey === previousDescriptorKey,
        )
      }
    }

    if (
      nextAuthSessionId !== null
      && nextAuthSessionId !== authRefreshRetryBudgetSessionIdRef.current
    ) {
      authRefreshRetriesRef.current = 0
      authRefreshRetryBudgetSessionIdRef.current = nextAuthSessionId
    }

    if (
      previousIdentity !== null
      && previousIdentity === nextIdentity
      && previousGeneration !== deliveryGeneration
    ) {
      rejectPendingDeliveries(
        deliveryError("Message delivery generation changed before acknowledgement.", "outcome_unknown"),
        pending => (
          pending.connectionIdentity === previousIdentity
          && pending.deliveryGeneration === previousGeneration
        ),
      )
      rejectPreparations(
        deliveryError("Message delivery generation changed before preparation completed.", "not_sent"),
        claim => (
          claim.connectionIdentity === previousIdentity
          && claim.deliveryGeneration === previousGeneration
        ),
      )
      clearRecentMessages(
        recent => (
          recent.connectionIdentity === previousIdentity
          && recent.deliveryGeneration === previousGeneration
        ),
      )
    }

    descriptorKeyRef.current = connectionDescriptorIdentity
    connectionRef.current = normalizedConnection
    deliveryIdentityRef.current = nextIdentity
    deliveryGenerationRef.current = deliveryGeneration
    callbacksRef.current = {
      onConnectionClose,
      onConnectionFailure,
      onSessionConnectionClose,
      onSessionConnectionFailure,
      onConnect,
      onDisconnect,
      onError,
      onMessage,
    }
    refreshAccessTokenRef.current = refreshAccessToken
  })

  // Update token ref when token changes
  useEffect(() => {
    tokenRef.current = token !== undefined ? token : authToken
  }, [token, authToken])

  const connect = useCallback(() => {
    if (!mountedRef.current) return
    if (
      isConnectingRef.current
      || (
        socketRef.current
        && socketRef.current.readyState < WebSocket.CLOSING
      )
    ) return

    const connection = connectionRef.current
    const descriptorKey = descriptorKeyRef.current
    if (!connection) return
    if (!descriptorKey) return
    if (connection.chatTaskIdMode === "required" && !connection.taskId) return

    const previousOwner = socketOwnerRef.current
    if (previousOwner) {
      retireOwner(previousOwner, {
        pendingError: deliveryError("Connection replaced before the message was accepted.", "outcome_unknown"),
        preparationError: deliveryError("Connection replaced before the message was prepared.", "not_sent"),
        close: {
          code: 1000,
          reason: "Connection replaced",
        },
        notifyDisconnect: true,
      })
      if (socketOwnerRef.current || isConnectingRef.current) return
    } else if (socketRef.current) {
      socketRef.current = null
    }

    isConnectingRef.current = true
    try {
      // Test if the URL is valid before creating WebSocket
      if (!connection.url.startsWith('ws://') && !connection.url.startsWith('wss://')) {
        throw new Error("Invalid WebSocket URL configuration")
      }

      const attemptEpoch = ++attemptEpochRef.current
      const socket = connection.protocols
        ? new WebSocket(connection.url, connection.protocols)
        : new WebSocket(connection.url)
      const owner: SocketOwner = {
        callbacks: callbacksRef.current,
        connection,
        descriptorKey,
        disconnectNotified: false,
        handshakeTimer: null,
        lifecycleEpoch: lifecycleEpochRef.current,
        attemptEpoch,
        refreshAccessToken: refreshAccessTokenRef.current,
        socket,
      }
      socketRef.current = socket
      socketOwnerRef.current = owner

      owner.handshakeTimer = setTimeout(() => {
        if (!isCurrentOwner(owner)) return
        const handshakeError = new Error("WebSocket connection handshake timed out.")
        const wasCurrent = retireOwner(owner, {
          pendingError: deliveryError("Connection timed out before the message was accepted.", "outcome_unknown"),
          preparationError: deliveryError("Connection timed out before the message was prepared.", "not_sent"),
          close: { code: 1000, reason: "Handshake timeout" },
          notifyDisconnect: true,
        })
        if (!wasCurrent || socketOwnerRef.current || socketRef.current) return
        setConnectionError(handshakeError)
        reportConnectionFailure(owner.callbacks, {
          recoverable: true,
          error: handshakeError,
        }, owner.connection.identity)
      }, HANDSHAKE_TIMEOUT_MS)

      socket.onopen = () => {
        if (!isCurrentOwner(owner)) return
        if (owner.handshakeTimer) {
          clearTimeout(owner.handshakeTimer)
          owner.handshakeTimer = null
        }
        if (connection.expectedProtocol && socket.protocol !== connection.expectedProtocol) {
          const protocolError = new Error("WebSocket subprotocol negotiation failed.")
          // A protocol mismatch is reported as a terminal connection failure;
          // suppressing disconnect avoids starting a second recovery path.
          const wasCurrent = retireOwner(owner, {
            pendingError: deliveryError("Connection closed before the message was accepted.", "outcome_unknown"),
            preparationError: deliveryError("Connection closed before the message was prepared.", "not_sent"),
            close: { code: 1002, reason: "WebSocket subprotocol mismatch" },
            notifyDisconnect: false,
          })
          if (!wasCurrent) return
          setConnectionError(protocolError)
          reportConnectionFailure(owner.callbacks, {
            recoverable: false,
            error: protocolError,
          }, owner.connection.identity)
          return
        }

        setIsConnected(true)
        setConnectionError(null)
        isConnectingRef.current = false
        owner.callbacks.onConnect?.()
      }

      socket.onclose = (event) => {
        if (!isCurrentOwner(owner)) return

        const wasCurrent = retireOwnerCore(owner, {
          pendingError: deliveryError("Connection closed before the message was accepted.", "outcome_unknown"),
          preparationError: deliveryError("Connection closed before the message was prepared.", "not_sent"),
          notifyDisconnect: false,
        })
        if (!wasCurrent) return

        let closeDisposition: "handled" | "default"
        try {
          closeDisposition = owner.callbacks.onSessionConnectionClose
            ? owner.callbacks.onSessionConnectionClose(event, owner.connection.identity)
            : owner.callbacks.onConnectionClose?.(event) ?? "default"
        } catch {
          if (canApplyRetiredOwnerPolicy(owner)) {
            const handlerError = new Error("WebSocket close handler failed.")
            console.error("WebSocket close handler failed")
            reportPermanentFailure(owner, handlerError)
          }
          return
        }

        if (!canApplyRetiredOwnerPolicy(owner)) return
        if (closeDisposition === "handled") return

        notifyDisconnect(owner)
        if (!canApplyRetiredOwnerPolicy(owner)) return

        // Handle authentication errors (4001 = Authentication required)
        if (event.code === 4001) {
          if (owner.connection.credentialOwner.kind !== "auth-context") {
            reportPermanentFailure(owner, new Error("Authentication failed"))
            return
          }
          if (!owner.connection.credentialOwner.session) {
            reportPermanentFailure(owner, new Error("Authentication lineage is unavailable"))
            return
          }
          if (authRefreshRetriesRef.current >= MAX_AUTH_REFRESH_RETRIES) {
            reportPermanentFailure(
              owner,
              new Error("Authentication failed after token refresh retries"),
            )
            return
          }
          authRefreshRetriesRef.current += 1
          try {
            owner.refreshAccessToken(
              owner.connection.credentialOwner.session,
            )
              .then((refreshSuccess) => {
                if (!canApplyRetiredOwnerPolicy(owner)) return
                if (refreshSuccess) {
                  scheduleRetry(
                    connect,
                    1000,
                    owner.lifecycleEpoch,
                    owner.attemptEpoch,
                  )
                } else {
                  reportPermanentFailure(
                    owner,
                    new Error("Authentication failed and token refresh failed"),
                  )
                }
              })
              .catch(() => {
                if (!canApplyRetiredOwnerPolicy(owner)) return
                console.error("Error refreshing auth token for WebSocket")
                reportPermanentFailure(
                  owner,
                  new Error("Authentication failed and token refresh error"),
                )
              })
          } catch {
            if (!canApplyRetiredOwnerPolicy(owner)) return
            console.error("Error refreshing auth token for WebSocket")
            reportPermanentFailure(
              owner,
              new Error("Authentication failed and token refresh error"),
            )
          }
          return
        }

        if (event.code === 4003) {
          reportPermanentFailure(owner, new Error(event.reason || "Access denied"))
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

        // Only attempt to reconnect if under max attempts and the connection is task-bound
        if (reconnectAttemptsRef.current < maxReconnectAttempts && connection.taskId) {
          reconnectAttemptsRef.current++
          const delay = Math.min(1000 * reconnectAttemptsRef.current, 5000)
          scheduleRetry(
            connect,
            delay,
            owner.lifecycleEpoch,
            owner.attemptEpoch,
          )
        }
      }

      socket.onerror = () => {
        if (!isCurrentOwner(owner)) return
        console.error("WebSocket error")
        const connectionError = new Error("WebSocket connection failed. The backend WebSocket endpoint may not be available.")
        const wasCurrent = retireOwner(owner, {
          pendingError: deliveryError("Connection failed before the message was accepted.", "outcome_unknown"),
          preparationError: deliveryError("Connection failed before the message was prepared.", "not_sent"),
          close: { code: 1000, reason: "WebSocket transport error" },
          notifyDisconnect: true,
        })
        if (!wasCurrent || socketOwnerRef.current || socketRef.current) return
        setConnectionError(connectionError)
        reportConnectionFailure(owner.callbacks, {
          recoverable: true,
          error: connectionError,
        }, owner.connection.identity)

      }

      socket.onmessage = (event) => {
        if (!isCurrentOwner(owner)) return
        try {
          const data = JSON.parse(event.data)

          if (data.type === 'message_accepted' || data.type === 'message_rejected') {
            const clientMessageId = data.client_message_id
            const pending = typeof clientMessageId === 'string'
              ? pendingDeliveriesRef.current.get(clientMessageId)
              : undefined
            if (
              pending
              && pending.socket === socket
              && pending.descriptorKey === owner.descriptorKey
              && pending.lifecycleEpoch === owner.lifecycleEpoch
              && pending.attemptEpoch === owner.attemptEpoch
              && pending.deliveryGeneration === deliveryGenerationRef.current
            ) {
              clearTimeout(pending.timeout)
              pendingDeliveriesRef.current.delete(clientMessageId)
              if (data.type === 'message_accepted') {
                pending.resolve({
                  client_message_id: clientMessageId,
                  turn_id: typeof data.turn_id === 'string' ? data.turn_id : clientMessageId,
                })
              } else {
                pending.reject(deliveryError(
                  data.message || "Message was rejected.",
                  data.rejection_outcome === "not_accepted"
                    ? "rejected"
                    : "outcome_unknown",
                  data.retry_with_new_id === true,
                ))
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
          owner.callbacks.onMessage?.(message)
        } catch (error) {
          console.error("Error parsing WebSocket message", error)
        }
      }

    } catch {
      console.error("Failed to create WebSocket connection")
      const connectionError = new Error("Failed to create WebSocket connection.")
      isConnectingRef.current = false
      setConnectionError(connectionError)
      reportConnectionFailure(callbacksRef.current, {
        recoverable: false,
        error: connectionError,
      }, connectionRef.current?.identity)
    }
  }, [
    canApplyRetiredOwnerPolicy,
    isCurrentOwner,
    notifyDisconnect,
    reportPermanentFailure,
    retireOwner,
    retireOwnerCore,
    scheduleRetry,
  ])

  const disconnect = useCallback(() => {
    const owner = socketOwnerRef.current
    invalidateLifecycle()
    reconnectAttemptsRef.current = 0
    if (owner) {
      retireOwner(owner, {
        pendingError: deliveryError("Disconnected before the message was accepted.", "outcome_unknown"),
        preparationError: deliveryError("Disconnected before the message was prepared.", "not_sent"),
        close: {},
        notifyDisconnect: true,
      })
    } else {
      rejectPendingDeliveries(
        deliveryError("Disconnected before the message was accepted.", "outcome_unknown"),
      )
      rejectPreparations(
        deliveryError("Disconnected before the message was prepared.", "not_sent"),
      )
    }
    setIsConnected(false)
    isConnectingRef.current = false
  }, [
    invalidateLifecycle,
    rejectPendingDeliveries,
    rejectPreparations,
    retireOwner,
  ])

  const sendMessage = useCallback((
    message: Record<string, unknown>,
  ): WebSocketSendResult => {
    const connection = connectionRef.current
    const socket = socketRef.current
    if (
      connection
      && socket?.readyState === WebSocket.OPEN
      && isCurrentSocket(socket, connection.identity)
    ) {
      try {
        socket.send(JSON.stringify(message))
        return "sent"
      } catch {
        return "not_sent"
      }
    }
    return "not_sent"
  }, [isCurrentSocket])

  const sendChatMessage = useCallback(async (
    message: string,
    files?: File[],
    force: boolean = false,
    requestedClientMessageId?: string,
  ): Promise<MessageDeliveryAck> => {
    const timestamp = Date.now()
    const owner = socketOwnerRef.current
    const connection = owner?.connection
    const socket = owner?.socket
    if (
      !connection
      || socket?.readyState !== WebSocket.OPEN
      || !owner
      || !isCurrentOwner(owner)
      || (connection.chatTaskIdMode === "required" && !connection.taskId)
    ) {
      throw deliveryError("Message not sent: the connection is not ready.", "not_sent")
    }
    if (connection.chatTaskIdMode === "omit" && files && files.length > 0) {
      throw deliveryError("File delivery is not supported for this connection.", "not_sent")
    }

    const currentTaskId = connection.taskId
    const currentDeliveryGeneration = deliveryGenerationRef.current
    const currentDescriptorKey = owner.descriptorKey
    const currentLifecycleEpoch = owner.lifecycleEpoch
    const currentAttemptEpoch = owner.attemptEpoch
    const clientMessageId = requestedClientMessageId || generateClientMessageId()
    if (
      pendingDeliveriesRef.current.has(clientMessageId)
      || preparationsRef.current.has(clientMessageId)
    ) {
      throw deliveryError("Message not sent: the client message id is already pending.", "not_sent")
    }
    const duplicateMessage = recentMessagesRef.current.find(
      msg => (
        msg.descriptorKey === currentDescriptorKey
        && msg.lifecycleEpoch === currentLifecycleEpoch
        && msg.attemptEpoch === currentAttemptEpoch
        && msg.deliveryGeneration === currentDeliveryGeneration
        && msg.message === message
        && msg.clientMessageId !== clientMessageId
        && (timestamp - msg.timestamp) < MESSAGE_DUPLICATE_THRESHOLD
      )
    )
    const duplicateIsPending = duplicateMessage
      ? pendingDeliveriesRef.current.has(duplicateMessage.clientMessageId)
      : false
    if (!force && duplicateIsPending) {
      throw deliveryError("Duplicate message ignored while the previous send is pending.", "not_sent")
    }

    let rejectCancellation!: (error: Error) => void
    const cancellation = new Promise<never>((_resolve, reject) => {
      rejectCancellation = reject
    })
    const claim: MessagePreparationClaim = {
      cancellation,
      cancel: (error) => {
        if (claim.cancelled) return
        claim.cancelled = true
        rejectCancellation(error)
      },
      cancelled: false,
      connectionIdentity: connection.identity,
      descriptorKey: currentDescriptorKey,
      lifecycleEpoch: currentLifecycleEpoch,
      attemptEpoch: currentAttemptEpoch,
      deliveryGeneration: currentDeliveryGeneration,
      socket,
    }
    preparationsRef.current.set(clientMessageId, claim)

    try {
      const messageData: Record<string, unknown> = {
        type: 'chat',
        message,
        client_message_id: clientMessageId,
        ...(connection.chatTaskIdMode === "required" ? { task_id: currentTaskId } : {}),
      }

      if (files && files.length > 0) {
        if (!currentTaskId) {
          throw deliveryError("File delivery requires a task-bound connection.", "not_sent")
        }
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
          uploadedFiles = await Promise.race([
            uploadFiles(filesToUpload, {
              taskId: currentTaskId,
              taskType: 'task',
            }),
            claim.cancellation,
          ])
        } else if (filesToUpload.length > 0) {
          const uploadRequest = (async () => {
            const formData = new FormData()
            filesToUpload.forEach(file => formData.append('files', file))
            formData.append('task_type', 'task')
            formData.append('task_id', currentTaskId.toString())
            const response = await apiRequest(`${getUploadApiUrl()}/api/files/upload`, {
              method: 'POST',
              headers: {
                'Authorization': `Bearer ${tokenRef.current ?? localStorage.getItem('token') ?? ''}`,
              },
              body: formData,
            })
            const parsed = await parseApiResponse(response)
            if (!response.ok || !isJsonRecord(parsed.data)) {
              throw deliveryError(getUploadErrorMessage(response, parsed, {
                generic: 'Upload failed',
                ...UPLOAD_ERROR_MESSAGES,
              }), "not_sent")
            }
            const data = parsed.data
            return data.success && Array.isArray(data.files)
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
          })()
          uploadedFiles = await Promise.race([uploadRequest, claim.cancellation])
        }
        messageData.files = [...preUploadedFiles, ...uploadedFiles]
      }

      if (
        preparationsRef.current.get(clientMessageId) !== claim
        || claim.cancelled
        || socket.readyState !== WebSocket.OPEN
        || !isCurrentOwner(owner)
        || deliveryGenerationRef.current !== currentDeliveryGeneration
      ) {
        throw deliveryError("Message not sent: the connection changed before delivery.", "not_sent")
      }
      if (pendingDeliveriesRef.current.has(clientMessageId)) {
        throw deliveryError("Message not sent: the client message id is already pending.", "not_sent")
      }

      const delivery = new Promise<MessageDeliveryAck>((resolve, reject) => {
        const pendingDelivery: PendingDelivery = {
          resolve,
          reject,
          timeout: setTimeout(() => {
            if (pendingDeliveriesRef.current.get(clientMessageId) !== pendingDelivery) return
            pendingDeliveriesRef.current.delete(clientMessageId)
            reject(deliveryError("Message delivery was not acknowledged. Your draft was kept.", "outcome_unknown"))
          }, 30000),
          connectionIdentity: connection.identity,
          descriptorKey: currentDescriptorKey,
          lifecycleEpoch: currentLifecycleEpoch,
          attemptEpoch: currentAttemptEpoch,
          deliveryGeneration: currentDeliveryGeneration,
          socket,
        }
        pendingDeliveriesRef.current.set(clientMessageId, pendingDelivery)
      })
      if (preparationsRef.current.get(clientMessageId) === claim) {
        preparationsRef.current.delete(clientMessageId)
      }

      try {
        socket.send(JSON.stringify(messageData))
      } catch (error) {
        const pending = pendingDeliveriesRef.current.get(clientMessageId)
        if (
          pending?.socket === socket
          && pending.descriptorKey === currentDescriptorKey
          && pending.lifecycleEpoch === currentLifecycleEpoch
          && pending.attemptEpoch === currentAttemptEpoch
          && pending.deliveryGeneration === currentDeliveryGeneration
        ) {
          clearTimeout(pending.timeout)
          pendingDeliveriesRef.current.delete(clientMessageId)
          pending.reject(deliveryError(
            error instanceof Error ? error.message : String(error),
            "not_sent",
          ))
        }
        return delivery
      }

      recentMessagesRef.current.push({
        message,
        timestamp,
        connectionIdentity: connection.identity,
        descriptorKey: currentDescriptorKey,
        lifecycleEpoch: currentLifecycleEpoch,
        attemptEpoch: currentAttemptEpoch,
        deliveryGeneration: currentDeliveryGeneration,
        clientMessageId,
      })
      const cutoffTime = timestamp - 5000
      const firstKeepIndex = recentMessagesRef.current.findIndex(
        item => item.timestamp >= cutoffTime,
      )
      if (firstKeepIndex === -1) {
        recentMessagesRef.current = []
      } else if (firstKeepIndex > 0) {
        recentMessagesRef.current.splice(0, firstKeepIndex)
      }

      return delivery
    } catch (error) {
      if (error instanceof MessageDeliveryError) throw error
      throw deliveryError(
        error instanceof Error ? error.message : String(error),
        "not_sent",
      )
    } finally {
      if (preparationsRef.current.get(clientMessageId) === claim) {
        preparationsRef.current.delete(clientMessageId)
      }
    }
  }, [isCurrentOwner, uploadFiles])

  const getCurrentTaskConnection = useCallback(() => {
    const connection = connectionRef.current
    const socket = socketRef.current
    if (
      !connection?.taskId
      || socket?.readyState !== WebSocket.OPEN
      || !isCurrentSocket(socket, connection.identity)
    ) {
      return null
    }
    return { socket, taskId: connection.taskId }
  }, [isCurrentSocket])

  const executeTask = useCallback((taskDescription: string, files?: Array<{ name: string; type: string; size: number; content?: string }>) => {
    const current = getCurrentTaskConnection()
    if (current) {
      const message = JSON.stringify({
        type: "execute_task",
        task_id: current.taskId,
        description: taskDescription,
        ...(files && files.length > 0 && { files })
      })
      current.socket.send(message)
    }
  }, [getCurrentTaskConnection])

  const pauseTask = useCallback(() => {
    const current = getCurrentTaskConnection()
    if (current) {
      const message = {
        type: "pause_task",
        task_id: current.taskId,
        command_id: generateClientMessageId(),
      }
      current.socket.send(JSON.stringify(message))
    }
  }, [getCurrentTaskConnection])

  const resumeTask = useCallback(() => {
    const current = getCurrentTaskConnection()
    if (current) {
      current.socket.send(JSON.stringify({
        type: "resume_task",
        task_id: current.taskId,
        command_id: generateClientMessageId(),
      }))
    }
  }, [getCurrentTaskConnection])

  const requestStatus = useCallback(() => {
    const current = getCurrentTaskConnection()
    if (current) {
      current.socket.send(JSON.stringify({
        type: "status_request",
        task_id: current.taskId,
      }))
    }
  }, [getCurrentTaskConnection])

  useEffect(() => {
    const ownedDescriptorKey = connectionDescriptorIdentity
    setConnectionError(null)
    reconnectAttemptsRef.current = 0
    if (autoConnect && ownedDescriptorKey !== null && !isConnectingRef.current) {
      connect()
    }

    return () => {
      invalidateLifecycle()
      const owner = socketOwnerRef.current
      const ownsCurrentSocket = Boolean(
        owner
        && owner.descriptorKey === ownedDescriptorKey,
      )
      if (owner && ownsCurrentSocket) {
        retireOwner(owner, {
          pendingError: deliveryError("Connection replaced before the message was accepted.", "outcome_unknown"),
          preparationError: deliveryError("Message not sent: the connection changed before delivery.", "not_sent"),
          close: {
            code: 1000,
            reason: "Component unmounting",
          },
          notifyDisconnect: true,
        })
      }
      if (ownedDescriptorKey !== null) {
        rejectPendingDeliveries(
          deliveryError("Connection replaced before the message was accepted.", "outcome_unknown"),
          pending => pending.descriptorKey === ownedDescriptorKey,
        )
        rejectPreparations(
          deliveryError("Message not sent: the connection changed before delivery.", "not_sent"),
          claim => claim.descriptorKey === ownedDescriptorKey,
        )
        clearRecentMessages(
          recent => recent.descriptorKey === ownedDescriptorKey,
        )
      }
      if (mountedRef.current) setIsConnected(false)
      isConnectingRef.current = false
    }
  }, [
    autoConnect,
    clearRecentMessages,
    connect,
    connectionDescriptorIdentity,
    invalidateLifecycle,
    rejectPendingDeliveries,
    rejectPreparations,
    retireOwner,
  ])

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

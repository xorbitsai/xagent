"use client"

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import { useAuth } from "@/contexts/auth-context"
import type { AuthSessionSnapshot } from "@/lib/auth-cache"
import { apiRequest, classifyUploadError, isJsonRecord, parseApiResponse } from "@/lib/api-wrapper"
import { clientErrorFallback, readClientErrorCode, readServerRejectionCode, type ClientErrorCode } from "@/lib/client-errors"
import { normalizeUploadFileIds } from "@/lib/upload-file-ids"
import { isCanonicalUuid } from "@/lib/widget-session-upload"
import { generateClientMessageId, getWsUrl, getUploadApiUrl } from "@/lib/utils"
import { isFinalAnswerStreamEventType } from "@/lib/streaming-final-answer"
import { websocketAuthProtocols } from "@/lib/websocket-auth"

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

// Dead after this long without any frame, once the server confirmed heartbeats.
// The server's HEARTBEAT_INTERVAL_SECONDS fits into it twice, and widget.js
// SESSION_STABLE_CONNECTION_MS is twice this value.
export const DEAD_AFTER_MS = 30_000

// Reconnects never give up: a release restarts the backend for minutes, and a
// page left open across it must come back without a refresh. The delay grows
// by 1.5x from one second to a five-minute cap, so a page notices the end of
// an outage within about half the outage's length.
export const RECONNECT_BASE_DELAY_MS = 1_000
export const RECONNECT_MAX_DELAY_MS = 300_000
const RECONNECT_BACKOFF_FACTOR = 1.5
const RECONNECT_JITTER = 0.2

// `attempt` is 1-based; `random` is a value in [0, 1). Whole milliseconds.
export const reconnectDelayMs = (attempt: number, random: number): number => {
  const step = Math.max(0, attempt - 1)
  const nominal = Math.min(
    RECONNECT_MAX_DELAY_MS,
    RECONNECT_BASE_DELAY_MS * RECONNECT_BACKOFF_FACTOR ** step,
  )
  return Math.round(nominal * (1 - RECONNECT_JITTER + 2 * RECONNECT_JITTER * random))
}

// A server that is draining closes with 1012 (Service Restart) or with this
// reason, asking the client to come back on another instance.
export const RECONNECT_CLOSE_REASON = "reconnect"

// Closes that a new connection would only repeat: protocol and policy errors.
const PERMANENT_CLOSE_CODES = new Set([1002, 1003, 1007, 1008, 1010, 1015])
// Application codes (4000-4999) are permanent unless listed here. 4008 is the
// public chat's connection rate limit, which the back-off itself respects.
const RETRYABLE_APPLICATION_CLOSE_CODES = new Set([4008])

export type CloseDisposition =
  // The other side meant to close (1000); nothing to recover.
  | "intentional"
  // 4001: refresh the access token, then connect again.
  | "reauthenticate"
  // Access denied, task unavailable, protocol or policy errors.
  | "permanent"
  // Abnormal closure (1006), server restart (1001, 1011-1014) and the rest.
  | "retry"

export const classifyClose = (
  event: Pick<CloseEvent, "code" | "reason">,
): CloseDisposition => {
  if (event.code === 1000) {
    return event.reason === RECONNECT_CLOSE_REASON ? "retry" : "intentional"
  }
  if (event.code === 4001) return "reauthenticate"
  if (PERMANENT_CLOSE_CODES.has(event.code)) return "permanent"
  if (
    event.code >= 4000
    && event.code <= 4999
    && !RETRYABLE_APPLICATION_CLOSE_CODES.has(event.code)
  ) {
    return "permanent"
  }
  return "retry"
}

export type ConnectionPhase = "idle" | "connecting" | "connected" | "reconnecting" | "lost"

export interface ConnectionPhaseFacts {
  hasDescriptor: boolean
  autoConnect: boolean
  stoppedByCaller: boolean
  ownerOpen: boolean
  ownerInFlight: boolean
  retryPending: boolean
  lifecycleEnded: boolean
}

// A pure function of facts the connection manager already owns; first match wins.
export const deriveConnectionPhase = (facts: ConnectionPhaseFacts): ConnectionPhase => {
  const hasConnectionObject = facts.ownerOpen || facts.ownerInFlight
  if (!facts.hasDescriptor || facts.stoppedByCaller || (!facts.autoConnect && !hasConnectionObject)) return "idle"
  if (facts.ownerOpen) return "connected"
  if (facts.retryPending || (facts.ownerInFlight && facts.lifecycleEnded)) return "reconnecting"
  return facts.ownerInFlight || !facts.lifecycleEnded ? "connecting" : "lost"
}

const MESSAGE_DUPLICATE_THRESHOLD = 2000 // Same message within 2 seconds is considered duplicate
const HANDSHAKE_TIMEOUT_MS = 10_000
const MAX_AUTH_REFRESH_RETRIES = 3
const SESSION_BINDING_PROTOCOL = "xagent-session-binding-v1"
const SESSION_TASK_PROTOCOL_PREFIX = "xagent-session-task."

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
  stream_run_id?: string | null
  stream_attempt_id?: string | null
  state_version?: number
  control_state?: "idle" | "running" | "pause_requested" | "paused" | "resume_requested" | "waiting_for_user" | "completed" | "failed"
  status?: unknown
  task?: Record<string, unknown>
}

interface MessageDeliveryAck {
  client_message_id: string
  turn_id: string
}

// Files that were already uploaded elsewhere and are attached to a chat frame
// by id. Only connections that omit the task id (the Session transport) accept
// them; the caller passes the server-advertised per-message limit.
export interface ChatMessageFileIds {
  fileIds?: readonly string[]
  maxFileIds?: number
}

export type MessageDeliveryDisposition = "not_sent" | "rejected" | "outcome_unknown"

interface MessageDeliveryErrorOptions {
  retryWithNewId?: boolean
  userFacing?: boolean
  errorCode?: ClientErrorCode | null
  connectionLost?: boolean
}

export class MessageDeliveryError extends Error {
  readonly disposition: MessageDeliveryDisposition
  readonly retryWithNewId: boolean
  /**
   * Whether `message` explains the failure in terms the sender can act on —
   * the server's rejection text. The remaining messages describe connection
   * plumbing ("the connection changed before delivery") and are diagnostics:
   * callers show their own localized string for those rather than putting
   * internal English in front of a visitor.
   */
  readonly userFacing: boolean
  /**
   * Display key for this failure; callers localize it with
   * clientErrorTranslationKey. It may be a client-derived code, so a non-null
   * value is not necessarily what the server sent, and it does not imply that
   * `message` is safe to show (`userFacing` alone says that). For an
   * `outcome_unknown` error, an omitted errorCode and an explicit `null` are
   * equivalent: both become `message_outcome_unknown`, so a send that may have
   * landed is never worded as a plain failure. Such an error stays code-less
   * only when it is constructed with `userFacing: true`. `not_sent` and
   * `rejected` errors get no default code.
   */
  readonly errorCode: ClientErrorCode | null
  /**
   * The connection carrying the message ended (closed, errored, declared dead
   * or replaced by a descriptor change) while it awaited acknowledgement, so
   * the same id may be sent once more on the next connection. False for a
   * caller disconnect, an unmount, a delivery-generation change and an
   * acknowledgement timeout on a connection that is still alive.
   */
  readonly connectionLost: boolean

  constructor(
    message: string,
    disposition: MessageDeliveryDisposition,
    options: MessageDeliveryErrorOptions = {},
  ) {
    super(message)
    this.name = "MessageDeliveryError"
    this.disposition = disposition
    this.retryWithNewId = options.retryWithNewId ?? false
    this.userFacing = options.userFacing ?? false
    this.connectionLost = options.connectionLost ?? false
    this.errorCode = options.errorCode
      ?? (disposition === "outcome_unknown" && !this.userFacing
        ? "message_outcome_unknown"
        : null)
  }
}

export const resolveReportedTimezone = (): string | undefined => {
  if (typeof window !== "undefined") {
    // widget.js copies data-timezone onto the iframe URL; an embedder that knows
    // its user's business zone outranks the machine zone.
    const declared = new URLSearchParams(window.location.search).get("timezone")?.trim()
    if (declared) return declared
  }
  try {
    // Defensive: neither a throwing Intl nor an empty resolved zone may block
    // the send. The caller's truthiness check drops "".
    return Intl.DateTimeFormat().resolvedOptions().timeZone
  } catch {
    return undefined
  }
}

const deliveryError = (
  message: string,
  disposition: MessageDeliveryDisposition,
  options: MessageDeliveryErrorOptions = {},
) => new MessageDeliveryError(message, disposition, options)

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

type LoginKind = "auth" | "external" | "widget-session"

// Only the Session window omits the task id; the embedding page owns its reconnects.
const isSessionConnection = (connection: WebSocketConnection) => connection.chatTaskIdMode === "omit"

const loginKindOf = (connection: WebSocketConnection): LoginKind => {
  if (isSessionConnection(connection)) return "widget-session"
  return connection.credentialOwner.kind === "auth-context" ? "auth" : "external"
}

// Compared in memory only: connection values may carry credentials.
const isSameLoginPrincipal = (previous: WebSocketConnection, next: WebSocketConnection) => {
  const kind = loginKindOf(previous)
  if (kind !== loginKindOf(next)) return false
  if (kind === "external") {
    const before = previous.protocols ?? []
    const after = next.protocols ?? []
    return previous.url === next.url
      && previous.loginToken === next.loginToken
      && before.length === after.length
      && before.every((protocol, index) => protocol === after[index])
  }
  const [a, b] = [previous.credentialOwner, next.credentialOwner]
  return kind === "widget-session" || (a.kind === "auth-context" && b.kind === "auth-context"
    && a.userId === b.userId && a.session?.sessionId === b.session?.sessionId)
}

export interface WebSocketConnection {
  identity: string
  url: string
  protocols?: string[]
  /** Validated only when connecting, so a malformed cache cannot crash render. */
  loginToken?: string
  expectedProtocol?: string
  taskId?: number
  chatTaskIdMode: "required" | "omit"
  taskBindingMode?: "session-subprotocol"
  credentialOwner: WebSocketCredentialOwner
}

const getConnectionProtocols = (
  connection: WebSocketConnection,
  currentTaskId: number | undefined,
): string[] | undefined => {
  if (connection.taskBindingMode !== "session-subprotocol") {
    return connection.loginToken ? websocketAuthProtocols(connection.loginToken) : connection.protocols
  }

  const baseProtocols = (connection.protocols ?? []).filter(protocol => (
    protocol !== SESSION_BINDING_PROTOCOL
    && !protocol.startsWith(SESSION_TASK_PROTOCOL_PREFIX)
  ))
  const taskBinding = Number.isInteger(currentTaskId) && Number(currentTaskId) > 0
    ? String(currentTaskId)
    : "unbound"
  return [
    ...baseProtocols,
    SESSION_BINDING_PROTOCOL,
    `${SESSION_TASK_PROTOCOL_PREFIX}${taskBinding}`,
  ]
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
  framesReceivedAtSend: number
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
  onConnect?: (openedTaskId: number | undefined) => void
  onConnectionCapabilities?: (capabilities: readonly string[]) => void
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
  // `{identity}|{login kind}:{login generation}`; carries no credential.
  connectionKey: string
  openedAt: number | null
  firstFrameAt: number | null
  lastFrameAt: number | null
  framesReceived: number
  heartbeatEnabled: boolean
  deadTimer: ReturnType<typeof setTimeout> | null
  sharedTaskId: number | undefined
}

interface OwnerRetirementOptions {
  pendingError: Error
  preparationError: Error
  close?: { code?: number; reason?: string }
  notifyDisconnect: boolean
  // Descriptor change, unmount and caller disconnect are not connection failures.
  keepsLifecycle?: true
}

interface ScheduledRetry {
  lifecycleEpoch: number
  attemptEpoch: number
  // Runs the retry now; the caller has already cleared the timer.
  fire: () => void
}

export interface UseWebSocketOptions {
  url?: string
  taskId?: number
  token?: string
  /** Builds the socket URL only. Credentials ride the handshake headers
   * (see websocketAuthProtocols), never the URL, so no token is passed in. */
  buildWebSocketUrl?: (params: { baseUrl: string; taskId: number }) => string
  uploadFiles?: (files: File[], params: { taskId?: number | null; taskType: string }) => Promise<Array<{ file_id: string; name?: string; size?: number; type?: string }>>
  connection?: WebSocketConnection | null
  deliveryGeneration?: number
  legacyErrorProse?: "trusted" | "untrusted"
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
  // Receives the task of the socket that opened, which can differ from the
  // caller's current task while a task switch is still being rendered.
  onConnect?: (openedTaskId: number | undefined) => void
  // Receives the capability list of the socket that sent it; the frame itself
  // is never passed to onMessage.
  onConnectionCapabilities?: (capabilities: readonly string[]) => void
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
    legacyErrorProse = "untrusted",
    onConnectionClose,
    onConnectionFailure,
    onSessionConnectionClose,
    onSessionConnectionFailure,
    autoConnect = true,
    onMessage,
    onConnect,
    onConnectionCapabilities,
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
        })
        : `${baseUrl}/ws/chat/${taskId}?liveness=heartbeat`,
      loginToken: effectiveToken || undefined,
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
  const taskIdRef = useRef(taskId)
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
  const legacyErrorProseRef = useRef(legacyErrorProse)
  const preparationsRef = useRef(new Map<string, MessagePreparationClaim>())
  // Keyed by the logical client_message_id, not by physical send. The durable
  // transport compares the whole payload, so a same-id retry that re-resolved
  // the zone would be rejected as different content while the first command may
  // still be executing. Cleared on either terminal ack below.
  const attemptTimezonesRef = useRef(new Map<string, string | undefined>())
  // Same key and lifetime: uploaded files of a same-id retry on the same task
  // and login, so the retry carries the file ids the first attempt already sent.
  const attemptUploadsRef = useRef(new Map<string, {
    taskId: number
    loginIdentity: string
    files: Array<{ file_id: string; name?: string; size?: number; type?: string }>
  }>())
  const recentMessagesRef = useRef<RecentMessage[]>([])
  const callbacksRef = useRef<WebSocketCallbacks>({
    onConnectionClose,
    onConnectionFailure,
    onSessionConnectionClose,
    onSessionConnectionFailure,
    onConnect,
    onConnectionCapabilities,
    onDisconnect,
    onError,
    onMessage,
  })
  const refreshAccessTokenRef = useRef(refreshAccessToken)
  const openedKeysRef = useRef(new Set<string>())
  const lastConnectionKeyRef = useRef<string | null>(null)
  const loginGenerationRef = useRef(0)
  const loginPrincipalRef = useRef<WebSocketConnection | null>(normalizedConnection)
  const lifecycleEndedRef = useRef(false)
  const stoppedByCallerRef = useRef(false)
  const autoConnectRef = useRef(autoConnect)
  const connectRef = useRef<() => void>(() => undefined)
  const [connectionPhase, setConnectionPhase] = useState<ConnectionPhase>(
    normalizedConnection !== null && autoConnect ? "connecting" : "idle",
  )

  // Called at the end of every entry point (the retry timer callback also calls
  // it before its stale-attempt early return), never from internal helpers or
  // layout effects, so a half-applied state is not rendered.
  const syncPhase = useCallback(() => {
    if (!mountedRef.current) return
    const owner = socketOwnerRef.current
    setConnectionPhase(deriveConnectionPhase({
      hasDescriptor: connectionRef.current !== null,
      autoConnect: autoConnectRef.current,
      stoppedByCaller: stoppedByCallerRef.current,
      ownerOpen: owner !== null && owner.openedAt !== null,
      ownerInFlight: owner !== null && owner.openedAt === null,
      retryPending: retryTimersRef.current.size > 0,
      lifecycleEnded: lifecycleEndedRef.current,
    }))
  }, [])

  const withPhaseSync = useCallback(<Args extends unknown[]>(
    entry: (...args: Args) => void,
  ) => (...args: Args) => {
    try {
      entry(...args)
    } finally {
      syncPhase()
    }
  }, [syncPhase])

  const loginIdentityFor = useCallback((connection: WebSocketConnection) => (
    `${loginKindOf(connection)}:${loginGenerationRef.current}`
  ), [])

  // Which login the current connection belongs to; carries no credential.
  const getLoginIdentity = useCallback(() => (
    connectionRef.current ? loginIdentityFor(connectionRef.current) : null
  ), [loginIdentityFor])

  const connectionKeyFor = useCallback((connection: WebSocketConnection) => (
    `${connection.identity}|${loginIdentityFor(connection)}`
  ), [loginIdentityFor])

  const releaseAttemptBindings = useCallback((clientMessageId: string) => {
    attemptTimezonesRef.current.delete(clientMessageId)
    attemptUploadsRef.current.delete(clientMessageId)
  }, [])

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
    lifecycleEndedRef.current = false
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
    if (owner.deadTimer) {
      clearTimeout(owner.deadTimer)
      owner.deadTimer = null
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
      // Without heartbeats, staying open for a full dead window proves health.
      if (
        !owner.heartbeatEnabled
        && owner.openedAt !== null
        && Date.now() - owner.openedAt >= DEAD_AFTER_MS
      ) {
        reconnectAttemptsRef.current = 0
      }
      if (!options.keepsLifecycle) lifecycleEndedRef.current = true
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
    const fire = () => {
      retryTimersRef.current.delete(timer)
      syncPhase() // even a stale timer changed the facts by removing itself
      if (
        !isCurrentLifecycle(lifecycleEpoch)
        || attemptEpochRef.current !== attemptEpoch
      ) return
      callback()
      syncPhase()
    }
    const timer = setTimeout(fire, delay)
    retryTimersRef.current.set(timer, { lifecycleEpoch, attemptEpoch, fire })
  }, [isCurrentLifecycle, syncPhase])

  // Every key is retried on a timer, including a first connection that failed
  // in transit, so a chat page loaded while the backend is down connects once
  // it is back. Before the socket opens, a 1006 can also be a rejected
  // handshake, but /ws/chat accepts before denying task access (4003,
  // permanent). What it still rejects before accepting is a missing or
  // rejected credential and an authentication outage (503); retrying the
  // former costs one handshake per back-off step until the page signs in again.
  const scheduleReconnect = useCallback((owner: SocketOwner) => {
    if (isSessionConnection(owner.connection) || !owner.connection.taskId) return
    if (!canApplyRetiredOwnerPolicy(owner)) return
    const attempt = ++reconnectAttemptsRef.current
    scheduleRetry(
      () => connectRef.current(),
      reconnectDelayMs(attempt, Math.random()),
      owner.lifecycleEpoch,
      owner.attemptEpoch,
    )
  }, [canApplyRetiredOwnerPolicy, scheduleRetry])

  const retireDeadOwner = useCallback((owner: SocketOwner, reason: string) => {
    if (!isCurrentOwner(owner)) return
    if (owner.sharedTaskId !== undefined) {
      owner.callbacks.onMessage?.({
        type: "stream_unavailable", task_id: owner.sharedTaskId,
        data: {}, timestamp: new Date().toISOString(),
      })
    }
    const wasCurrent = retireOwner(owner, {
      pendingError: deliveryError("Connection lost before the message was accepted.", "outcome_unknown", { connectionLost: true }),
      preparationError: deliveryError("Connection lost before the message was prepared.", "not_sent"),
      close: { code: 1000, reason: "Connection liveness timeout" },
      notifyDisconnect: true,
    })
    if (!wasCurrent || socketOwnerRef.current || socketRef.current) return
    const livenessError = new Error(reason)
    setConnectionError(livenessError)
    reportConnectionFailure(owner.callbacks, {
      recoverable: true,
      error: livenessError,
    }, owner.connection.identity)
    scheduleReconnect(owner)
  }, [isCurrentOwner, retireOwner, scheduleReconnect])

  // Not re-armed per frame: on firing it sleeps for the rest of the dead window.
  const watchLiveness = useCallback((owner: SocketOwner) => {
    const arm = (delay: number) => {
      owner.deadTimer = setTimeout(withPhaseSync(() => {
        owner.deadTimer = null
        if (!isCurrentOwner(owner)) return
        const remaining = DEAD_AFTER_MS - (Date.now() - (owner.lastFrameAt ?? 0))
        if (remaining > 0) {
          arm(remaining)
          return
        }
        retireDeadOwner(owner, "WebSocket connection stopped delivering frames.")
      }), delay)
    }
    arm(DEAD_AFTER_MS)
  }, [isCurrentOwner, retireDeadOwner, withPhaseSync])

  // A page that becomes visible again or a network that comes back online
  // retries once now instead of waiting out a back-off of up to five minutes.
  // A failure continues the back-off where it was.
  const retryNow = useCallback(() => {
    if (!mountedRef.current) return
    const owner = socketOwnerRef.current
    if (owner) {
      // Background tabs throttle timers, so the dead timer may not have fired.
      const lastFrameAt = owner.lastFrameAt ?? owner.openedAt
      if (
        !owner.heartbeatEnabled
        || lastFrameAt === null
        || Date.now() - lastFrameAt < DEAD_AFTER_MS
      ) return
      retireDeadOwner(owner, "WebSocket connection stopped delivering frames.")
      if (socketOwnerRef.current) return
    }
    const lifecycleEpoch = lifecycleEpochRef.current
    const attemptEpoch = attemptEpochRef.current
    for (const [timer, retry] of retryTimersRef.current) {
      if (
        retry.lifecycleEpoch !== lifecycleEpoch
        || retry.attemptEpoch !== attemptEpoch
      ) continue
      clearTimeout(timer)
      retry.fire()
      return
    }
  }, [retireDeadOwner])

  useEffect(() => {
    if (typeof window === "undefined") return
    const onVisibilityChange = withPhaseSync(() => {
      if (document.visibilityState === "visible") retryNow()
    })
    const onOnline = withPhaseSync(retryNow)
    document.addEventListener("visibilitychange", onVisibilityChange)
    window.addEventListener("online", onOnline)
    return () => {
      document.removeEventListener("visibilitychange", onVisibilityChange)
      window.removeEventListener("online", onOnline)
    }
  }, [retryNow, withPhaseSync])

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
      if (normalizedConnection !== null) {
        // The generation moves only here, never during render.
        const previousPrincipal = loginPrincipalRef.current
        if (
          previousPrincipal !== null
          && !isSameLoginPrincipal(previousPrincipal, normalizedConnection)
        ) {
          loginGenerationRef.current++
        }
        loginPrincipalRef.current = normalizedConnection
      }
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
    autoConnectRef.current = autoConnect
    taskIdRef.current = taskId
    deliveryIdentityRef.current = nextIdentity
    deliveryGenerationRef.current = deliveryGeneration
    legacyErrorProseRef.current = legacyErrorProse
    callbacksRef.current = {
      onConnectionClose,
      onConnectionFailure,
      onSessionConnectionClose,
      onSessionConnectionFailure,
      onConnect,
      onConnectionCapabilities,
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
    stoppedByCallerRef.current = false
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
        pendingError: deliveryError("Connection replaced before the message was accepted.", "outcome_unknown", { connectionLost: true }),
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
      const protocols = getConnectionProtocols(connection, taskIdRef.current)
      const socket = protocols
        ? new WebSocket(connection.url, protocols)
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
        connectionKey: connectionKeyFor(connection),
        openedAt: null,
        firstFrameAt: null,
        lastFrameAt: null,
        framesReceived: 0,
        heartbeatEnabled: false,
        deadTimer: null,
        sharedTaskId: undefined,
      }
      socketRef.current = socket
      socketOwnerRef.current = owner

      owner.handshakeTimer = setTimeout(withPhaseSync(() => {
        if (!isCurrentOwner(owner)) return
        const handshakeError = new Error("WebSocket connection handshake timed out.")
        const wasCurrent = retireOwner(owner, {
          pendingError: deliveryError("Connection timed out before the message was accepted.", "outcome_unknown", { connectionLost: true }),
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
        scheduleReconnect(owner)
      }), HANDSHAKE_TIMEOUT_MS)

      socket.onopen = withPhaseSync(() => {
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
            pendingError: deliveryError("Connection closed before the message was accepted.", "outcome_unknown", { connectionLost: true }),
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

        owner.openedAt = Date.now()
        openedKeysRef.current.add(owner.connectionKey)
        setIsConnected(true)
        setConnectionError(null)
        isConnectingRef.current = false
        owner.callbacks.onConnect?.(owner.connection.taskId)
      })

      socket.onclose = withPhaseSync((event: CloseEvent) => {
        if (!isCurrentOwner(owner)) return
        if (owner.sharedTaskId !== undefined) {
          owner.callbacks.onMessage?.({
            type: "stream_unavailable", task_id: owner.sharedTaskId,
            data: {}, timestamp: new Date().toISOString(),
          })
        }

        const wasCurrent = retireOwnerCore(owner, {
          pendingError: deliveryError("Connection closed before the message was accepted.", "outcome_unknown", { connectionLost: true }),
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

        const disposition = classifyClose(event)
        if (disposition === "reauthenticate") {
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
                syncPhase()
              })
              .catch(() => {
                if (!canApplyRetiredOwnerPolicy(owner)) return
                console.error("Error refreshing auth token for WebSocket")
                reportPermanentFailure(
                  owner,
                  new Error("Authentication failed and token refresh error"),
                )
                syncPhase()
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

        if (disposition === "permanent") {
          reportPermanentFailure(
            owner,
            new Error(
              event.reason
              || (event.code === 4003
                ? "Access denied"
                : `WebSocket closed with code ${event.code}`),
            ),
          )
          return
        }

        if (disposition === "intentional") return

        scheduleReconnect(owner)
      })

      socket.onerror = withPhaseSync(() => {
        if (!isCurrentOwner(owner)) return
        console.error("WebSocket error")
        const connectionError = new Error("WebSocket connection failed. The backend WebSocket endpoint may not be available.")
        const wasCurrent = retireOwner(owner, {
          pendingError: deliveryError("Connection failed before the message was accepted.", "outcome_unknown", { connectionLost: true }),
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
        scheduleReconnect(owner)
      })

      socket.onmessage = (event) => {
        if (!isCurrentOwner(owner)) return
        // Every frame is proof of life, even unparsable and liveness frames.
        const receivedAt = Date.now()
        owner.lastFrameAt = receivedAt
        owner.framesReceived += 1
        owner.firstFrameAt ??= receivedAt
        // Traffic spanning a full dead window restarts the back-off.
        if (owner.heartbeatEnabled && receivedAt - owner.firstFrameAt >= DEAD_AFTER_MS) {
          reconnectAttemptsRef.current = 0
        }
        try {
          const data = JSON.parse(event.data)
          if (data.type === "connection_capabilities") {
            if (
              Array.isArray(data.capabilities)
              && data.capabilities.includes("heartbeat")
              && !owner.heartbeatEnabled
            ) {
              owner.heartbeatEnabled = true
              watchLiveness(owner)
            }
            owner.callbacks.onConnectionCapabilities?.(
              Array.isArray(data.capabilities)
                ? data.capabilities.filter((entry: unknown): entry is string => typeof entry === "string")
                : [],
            )
            return
          }
          if (data.type === "heartbeat") return
          if (data.type === "task_stream_snapshot" || data.stream_run_id !== undefined) {
            owner.sharedTaskId = data.task_id
          }

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
              if (
                data.type === 'message_accepted'
                || data.rejection_outcome === "not_accepted"
              ) {
                // Terminal for this id: accepted, or rejected without a
                // same-id retry. An unknown outcome keeps the binding.
                releaseAttemptBindings(clientMessageId)
              }
              if (data.type === 'message_accepted') {
                pending.resolve({
                  client_message_id: clientMessageId,
                  turn_id: typeof data.turn_id === 'string' ? data.turn_id : clientMessageId,
                })
              } else {
                const hasErrorCode = Object.prototype.hasOwnProperty.call(data, "error_code")
                const errorCode = readServerRejectionCode(data.error_code)
                const hasLegacyRejectionMessage = (
                  typeof data.message === "string"
                  && data.message.trim() !== ""
                )
                const trustLegacyRejectionMessage = (
                  !hasErrorCode
                  && legacyErrorProseRef.current === "trusted"
                  && hasLegacyRejectionMessage
                )
                const rejectionMessage = errorCode
                  ? clientErrorFallback(errorCode)
                  : hasErrorCode
                    ? "Message was rejected."
                    : trustLegacyRejectionMessage
                      ? data.message
                      : "Message was rejected."
                const disposition: MessageDeliveryDisposition = (
                  data.rejection_outcome === "not_accepted"
                    ? "rejected"
                    : "outcome_unknown"
                )
                // A definitive rejection with no usable server code still gets
                // wording of its own; an unknown outcome takes the default
                // code from the error constructor.
                const displayCode = errorCode ?? (
                  !trustLegacyRejectionMessage && disposition === "rejected"
                    ? "message_not_accepted"
                    : null
                )
                pending.reject(deliveryError(
                  rejectionMessage,
                  disposition,
                  {
                    retryWithNewId: data.retry_with_new_id === true,
                    userFacing: errorCode !== null || trustLegacyRejectionMessage,
                    errorCode: displayCode,
                  },
                ))
              }
            } else if (
              typeof clientMessageId === 'string'
              && !pending
              && !preparationsRef.current.has(clientMessageId)
              && (
                data.type === 'message_accepted'
                || data.rejection_outcome === "not_accepted"
              )
            ) {
              // Terminal ack that arrived after the 30s timeout already dropped
              // the pending entry. isCurrentOwner (top of onmessage) fences stale
              // sockets, and no same-id retry is in flight, so the binding can
              // never be reused; release it. outcome_unknown is excluded so a
              // still-allowed same-id retry keeps its first attempt's zone.
              releaseAttemptBindings(clientMessageId)
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
          message.stream_run_id = data.stream_run_id
          message.stream_attempt_id = data.stream_attempt_id
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
      lifecycleEndedRef.current = true
      setConnectionError(connectionError)
      reportConnectionFailure(callbacksRef.current, {
        recoverable: false,
        error: connectionError,
      }, connectionRef.current?.identity)
    }
  }, [
    canApplyRetiredOwnerPolicy,
    connectionKeyFor,
    isCurrentOwner,
    notifyDisconnect,
    releaseAttemptBindings,
    reportPermanentFailure,
    retireOwner,
    retireOwnerCore,
    scheduleReconnect,
    scheduleRetry,
    syncPhase,
    watchLiveness,
    withPhaseSync,
  ])

  useIsomorphicLayoutEffect(() => {
    connectRef.current = connect
  }, [connect])

  // Timers and effects run `connect` and recompute the phase themselves. A
  // caller's connect starts a new cycle, so an earlier failure, or a retry it
  // left waiting, no longer makes the new attempt read as reconnecting.
  const connectFromCaller = useMemo(() => withPhaseSync(() => {
    lifecycleEndedRef.current = false
    clearRetryTimers()
    connect()
  }), [clearRetryTimers, connect, withPhaseSync])

  const disconnect = useCallback(() => {
    const owner = socketOwnerRef.current
    invalidateLifecycle()
    stoppedByCallerRef.current = true
    reconnectAttemptsRef.current = 0
    if (owner) {
      retireOwner(owner, {
        pendingError: deliveryError("Disconnected before the message was accepted.", "outcome_unknown"),
        preparationError: deliveryError("Disconnected before the message was prepared.", "not_sent"),
        close: {},
        notifyDisconnect: true,
        keepsLifecycle: true,
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
    syncPhase()
  }, [
    invalidateLifecycle,
    rejectPendingDeliveries,
    rejectPreparations,
    retireOwner,
    syncPhase,
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
    requestId?: string,
    attachments?: ChatMessageFileIds,
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
      throw deliveryError(
        "Message not sent: the connection is not ready.",
        "not_sent",
        { errorCode: "connection_not_ready" },
      )
    }
    // Raw File objects cannot be delivered on this kind of connection; files
    // uploaded beforehand travel as ids through `attachments`.
    if (connection.chatTaskIdMode === "omit" && files && files.length > 0) {
      throw deliveryError("File delivery is not supported for this connection.", "not_sent")
    }
    // The server closes the connection on a frame that carries file ids it
    // cannot accept, so every shape it refuses is refused here first.
    const fileIds = attachments?.fileIds && attachments.fileIds.length > 0
      ? attachments.fileIds
      : null
    if (fileIds) {
      const maxFileIds = attachments?.maxFileIds
      if (connection.chatTaskIdMode !== "omit") {
        throw deliveryError("File ids are not supported for this connection.", "not_sent")
      }
      if (requestId) {
        throw deliveryError("Files cannot accompany an answer to a request.", "not_sent")
      }
      if (maxFileIds === undefined || fileIds.length > maxFileIds) {
        throw deliveryError("Message not sent: too many files attached.", "not_sent")
      }
      if (!fileIds.every(isCanonicalUuid)) {
        throw deliveryError("Message not sent: an attached file id is not valid.", "not_sent")
      }
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

    // A binding made before this attempt belongs to an earlier attempt under
    // this id whose outcome is unknown; a later same-id retry still needs it, so
    // a failure of this attempt keeps it.
    const bindingsPredateAttempt = attemptTimezonesRef.current.has(clientMessageId)
    try {
      if (!bindingsPredateAttempt) {
        attemptTimezonesRef.current.set(clientMessageId, resolveReportedTimezone())
      }
      const reportedTimezone = attemptTimezonesRef.current.get(clientMessageId)
      const messageData: Record<string, unknown> = {
        type: 'chat',
        message,
        client_message_id: clientMessageId,
        ...(requestId ? { request_id: requestId } : {}),
        ...(fileIds ? { file_ids: [...fileIds] } : {}),
        ...(reportedTimezone ? { context: { timezone: reportedTimezone } } : {}),
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

        const cachedUpload = attemptUploadsRef.current.get(clientMessageId)
        const loginIdentity = loginIdentityFor(connection)
        if (
          cachedUpload?.taskId === currentTaskId
          && cachedUpload.loginIdentity === loginIdentity
          && filesToUpload.length > 0
        ) {
          uploadedFiles = cachedUpload.files
        } else if (filesToUpload.length > 0 && uploadFiles) {
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
            const data = isJsonRecord(parsed.data) ? parsed.data : null
            if (
              !response.ok
              || data?.success !== true
              || !Array.isArray(data.files)
            ) {
              const uploadError = classifyUploadError(response, parsed)
              throw deliveryError(
                uploadError.message,
                "not_sent",
                {
                  userFacing: true,
                  errorCode: uploadError.errorCode,
                },
              )
            }
            const normalizedFileIds = normalizeUploadFileIds(
              data.files.map(file => isJsonRecord(file) ? file.file_id : undefined),
              filesToUpload.length,
            )
            if (!normalizedFileIds) {
              const uploadError = classifyUploadError(response, parsed)
              throw deliveryError(
                uploadError.message,
                "not_sent",
                { userFacing: true, errorCode: uploadError.errorCode },
              )
            }
            return data.files.map((file, index) => {
              const metadata = isJsonRecord(file) ? file : {}
              return {
                file_id: normalizedFileIds[index],
                name: typeof metadata.filename === 'string' ? metadata.filename : '',
                size: typeof metadata.file_size === 'number' ? metadata.file_size : 0,
                type: typeof metadata.mime_type === 'string' ? metadata.mime_type : '',
              }
            })
          })()
          uploadedFiles = await Promise.race([uploadRequest, claim.cancellation])
        }
        const normalizedUploadedFileIds = normalizeUploadFileIds(
          uploadedFiles.map(file => file.file_id),
          filesToUpload.length,
        )
        if (!normalizedUploadedFileIds) {
          throw deliveryError(
            clientErrorFallback("upload_failed"),
            "not_sent",
            { userFacing: true, errorCode: "upload_failed" },
          )
        }
        const normalizedUploadedFiles = uploadedFiles.map((file, index) => ({
          ...file,
          file_id: normalizedUploadedFileIds[index],
        }))
        if (filesToUpload.length > 0) {
          attemptUploadsRef.current.set(clientMessageId, { taskId: currentTaskId, loginIdentity, files: normalizedUploadedFiles })
        }
        const allFiles = [...preUploadedFiles, ...normalizedUploadedFiles]
        const normalizedFileIds = normalizeUploadFileIds(
          allFiles.map(file => file.file_id),
          filesWithUploadIds.length,
        )
        if (!normalizedFileIds) {
          throw deliveryError(
            clientErrorFallback("upload_failed"),
            "not_sent",
            { userFacing: true, errorCode: "upload_failed" },
          )
        }
        messageData.files = allFiles.map((file, index) => ({
          ...file,
          file_id: normalizedFileIds[index],
        }))
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
            if (
              isCurrentOwner(owner)
              && owner.framesReceived === pendingDelivery.framesReceivedAtSend
            ) {
              // No frame at all since the send: the link is dead, not slow.
              retireDeadOwner(owner, "WebSocket connection did not answer a sent message.")
            } else {
              pendingDeliveriesRef.current.delete(clientMessageId)
              reject(deliveryError("Message delivery was not acknowledged. Your draft was kept.", "outcome_unknown"))
            }
            syncPhase()
          }, 30000),
          connectionIdentity: connection.identity,
          descriptorKey: currentDescriptorKey,
          lifecycleEpoch: currentLifecycleEpoch,
          attemptEpoch: currentAttemptEpoch,
          deliveryGeneration: currentDeliveryGeneration,
          socket,
          framesReceivedAtSend: owner.framesReceived,
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
          // Nothing of this attempt reached the server, so unless an earlier
          // attempt owns the binding, a same-id retry may re-resolve the zone
          // and upload again; drop it rather than leak it.
          if (!bindingsPredateAttempt) releaseAttemptBindings(clientMessageId)
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
      // Pre-send failures never reached the server; free the attempt binding so
      // abandoned drafts do not accumulate. An outcome-unknown send of this
      // attempt cannot land here (it settles through the delivery promise), and
      // a binding an earlier outcome-unknown attempt made is kept for its retry.
      const disposition =
        error instanceof MessageDeliveryError ? error.disposition : "not_sent"
      if (disposition !== "outcome_unknown" && !bindingsPredateAttempt) {
        releaseAttemptBindings(clientMessageId)
      }
      if (error instanceof MessageDeliveryError) throw error
      const errorCode = typeof error === "object" && error !== null
        ? readClientErrorCode((error as { errorCode?: unknown }).errorCode)
        : null
      throw deliveryError(
        errorCode
          ? clientErrorFallback(errorCode)
          : error instanceof Error ? error.message : String(error),
        "not_sent",
        {
          userFacing: errorCode !== null,
          errorCode,
        },
      )
    } finally {
      if (preparationsRef.current.get(clientMessageId) === claim) {
        preparationsRef.current.delete(clientMessageId)
      }
    }
  }, [isCurrentOwner, loginIdentityFor, releaseAttemptBindings, retireDeadOwner, syncPhase, uploadFiles])

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
    const connectionKey = connectionRef.current
      ? connectionKeyFor(connectionRef.current)
      : null
    // The back-off belongs to the key: a same-key credential refresh keeps it.
    if (connectionKey !== lastConnectionKeyRef.current) {
      reconnectAttemptsRef.current = 0
      lastConnectionKeyRef.current = connectionKey
    }
    if (autoConnect && ownedDescriptorKey !== null && !isConnectingRef.current) {
      if (
        connectionKey !== null
        && openedKeysRef.current.has(connectionKey)
      ) {
        // Replacing a connection on an opened key counts as one attempt.
        reconnectAttemptsRef.current++
      }
      connect()
    }
    syncPhase()

    return () => {
      invalidateLifecycle()
      const owner = socketOwnerRef.current
      const ownsCurrentSocket = Boolean(
        owner
        && owner.descriptorKey === ownedDescriptorKey,
      )
      if (owner && ownsCurrentSocket) {
        retireOwner(owner, {
          pendingError: deliveryError("Connection replaced before the message was accepted.", "outcome_unknown", { connectionLost: mountedRef.current }),
          preparationError: deliveryError("Message not sent: the connection changed before delivery.", "not_sent"),
          close: {
            code: 1000,
            reason: "Component unmounting",
          },
          notifyDisconnect: true,
          keepsLifecycle: true,
        })
      }
      if (ownedDescriptorKey !== null) {
        rejectPendingDeliveries(
          deliveryError("Connection replaced before the message was accepted.", "outcome_unknown", { connectionLost: mountedRef.current }),
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
      syncPhase()
    }
  }, [
    autoConnect,
    clearRecentMessages,
    connect,
    connectionDescriptorIdentity,
    connectionKeyFor,
    invalidateLifecycle,
    rejectPendingDeliveries,
    rejectPreparations,
    retireOwner,
    syncPhase,
  ])

  return {
    isConnected,
    connectionPhase,
    getLoginIdentity,
    lastMessage,
    connectionError,
    connect: connectFromCaller,
    disconnect,
    sendMessage,
    sendChatMessage,
    executeTask,
    pauseTask,
    resumeTask,
    requestStatus,
  }
}

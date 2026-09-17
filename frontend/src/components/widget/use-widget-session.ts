"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import type { WebSocketConnectionFailure } from "@/hooks/use-websocket"

const TOKEN_REFRESH_THRESHOLD_MS = 60_000
const EXPIRY_WARNING_LEAD_MS = 10 * 60_000
// Keep this predicate aligned with the two retry signals emitted by
// frontend/public/widget.js; every other parent code fails closed.
const isParentRecoverableCode = (code: string): boolean =>
  code === "network_unavailable" || code === "rate_limited"

export type WidgetSessionStatus = "waiting" | "active" | "refreshing" | "degraded" | "terminal"
export type WidgetSessionReconnectReason = "ws_closed" | "token_expired"

export interface WidgetSessionAgent {
  id: number
  name: string
  description?: string
  logoUrl?: string
  suggestedPrompts: string[]
}

export interface WidgetSession {
  token: string
  tokenExpiresAt: string
  absoluteExpiresAt: string
  agent: WidgetSessionAgent
  generation: number
}

interface WidgetSessionBridgeState {
  status: WidgetSessionStatus
  session: WidgetSession | null
  agent: WidgetSessionAgent | null
  terminalCode: string | null
  isAbsoluteExpiryWarningVisible: boolean
}

interface ProtocolMessage extends Record<string, unknown> {
  xagent: true
  v: 1
  type: "session_update" | "session_degraded" | "session_terminal"
}

const initialState: WidgetSessionBridgeState = {
  status: "waiting",
  session: null,
  agent: null,
  terminalCode: null,
  isAbsoluteExpiryWarningVisible: false,
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const parseDate = (value: unknown): number | null => {
  if (typeof value !== "string" || !value) return null
  const timestamp = Date.parse(value)
  return Number.isFinite(timestamp) ? timestamp : null
}

const isNonBlankString = (value: unknown): value is string =>
  typeof value === "string" && value.trim().length > 0

const parseAgent = (value: unknown): WidgetSessionAgent | null => {
  if (!isRecord(value)) return null
  if (!Number.isInteger(value.id) || (value.id as number) <= 0) return null
  if (typeof value.name !== "string" || !value.name.trim()) return null
  if (value.description !== undefined && value.description !== null && typeof value.description !== "string") return null
  if (value.logo_url !== undefined && value.logo_url !== null && typeof value.logo_url !== "string") return null
  if (!Array.isArray(value.suggested_prompts) || !value.suggested_prompts.every((item) => typeof item === "string")) {
    return null
  }

  return {
    id: value.id as number,
    name: value.name,
    ...(typeof value.description === "string" ? { description: value.description } : {}),
    ...(typeof value.logo_url === "string" ? { logoUrl: value.logo_url } : {}),
    suggestedPrompts: value.suggested_prompts as string[],
  }
}

const isProtocolMessage = (value: unknown): value is ProtocolMessage => {
  if (!isRecord(value)) return false
  return value.xagent === true
    && value.v === 1
    && (
      value.type === "session_update"
      || value.type === "session_degraded"
      || value.type === "session_terminal"
    )
}

export function buildWidgetSessionWebSocketUrl(origin: string): string {
  const url = new URL(origin)
  if (url.protocol === "https:") {
    url.protocol = "wss:"
  } else if (url.protocol === "http:") {
    url.protocol = "ws:"
  } else {
    throw new Error("Widget Session requires an HTTP(S) origin")
  }
  url.pathname = "/v1/external/chat/sessions/ws"
  url.search = ""
  url.hash = ""
  return url.toString()
}

export function useWidgetSession() {
  const [state, setState] = useState<WidgetSessionBridgeState>(initialState)
  const mountedRef = useRef(false)
  const targetOriginRef = useRef<string | null>(null)
  const recoveryInFlightRef = useRef(false)
  const terminalRef = useRef(false)
  const generationRef = useRef(0)
  const activeSessionGenerationRef = useRef<number | null>(null)
  const activeDeliveryIdRef = useRef<string | null>(null)
  const warningTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const clearWarningTimer = useCallback(() => {
    if (warningTimerRef.current) {
      clearTimeout(warningTimerRef.current)
      warningTimerRef.current = null
    }
  }, [])

  const transitionTerminal = useCallback((code: string) => {
    if (terminalRef.current) return
    terminalRef.current = true
    activeSessionGenerationRef.current = null
    activeDeliveryIdRef.current = null
    recoveryInFlightRef.current = false
    clearWarningTimer()
    setState({
      status: "terminal",
      session: null,
      agent: null,
      terminalCode: code,
      isAbsoluteExpiryWarningVisible: false,
    })
  }, [clearWarningTimer])

  const transitionDegraded = useCallback(() => {
    if (terminalRef.current) return
    activeSessionGenerationRef.current = null
    activeDeliveryIdRef.current = null
    clearWarningTimer()
    setState((current) => ({
      status: "degraded",
      session: null,
      agent: current.agent,
      terminalCode: null,
      isAbsoluteExpiryWarningVisible: false,
    }))
  }, [clearWarningTimer])

  const issueReconnectRequest = useCallback((reason: WidgetSessionReconnectReason) => {
    const targetOrigin = targetOriginRef.current
    if (!mountedRef.current || !targetOrigin || terminalRef.current) return

    if (recoveryInFlightRef.current) return
    recoveryInFlightRef.current = true
    clearWarningTimer()
    setState((current) => ({
      status: "refreshing",
      session: null,
      agent: current.agent,
      terminalCode: null,
      isAbsoluteExpiryWarningVisible: false,
    }))
    window.parent.postMessage({ xagent: true, v: 1, type: "reconnect_request", reason }, targetOrigin)
  }, [clearWarningTimer])

  const requestReconnect = useCallback((reason: WidgetSessionReconnectReason) => {
    if (!mountedRef.current || terminalRef.current || recoveryInFlightRef.current) return
    activeSessionGenerationRef.current = null
    issueReconnectRequest(reason)
  }, [issueReconnectRequest])

  const isActiveConnection = useCallback((connectionIdentity?: string) => {
    const activeGeneration = activeSessionGenerationRef.current
    return connectionIdentity === undefined || (
      activeGeneration !== null
      && connectionIdentity === `widget-session:${activeGeneration}`
    )
  }, [])

  const handleConnectionOpen = useCallback((connectionIdentity: string) => {
    const activeGeneration = activeSessionGenerationRef.current
    const deliveryId = activeDeliveryIdRef.current
    const targetOrigin = targetOriginRef.current
    if (
      !mountedRef.current
      || !targetOrigin
      || terminalRef.current
      || activeGeneration === null
      || !deliveryId
      || connectionIdentity !== `widget-session:${activeGeneration}`
    ) return

    window.parent.postMessage({
      xagent: true,
      v: 1,
      type: "session_connection_open",
      session_delivery_id: deliveryId,
    }, targetOrigin)
  }, [])

  const handleConnectionClose = useCallback((
    event: CloseEvent,
    connectionIdentity?: string,
  ): "handled" => {
    if (!isActiveConnection(connectionIdentity)) return "handled"
    if (event.code === 1000 || event.code === 1001) {
      transitionTerminal("unexpected_error")
      return "handled"
    }

    if (event.code === 4403) {
      transitionTerminal("ws_4403")
      return "handled"
    }

    if (event.code === 4408) {
      transitionTerminal("ws_4408")
      return "handled"
    }

    requestReconnect("ws_closed")
    return "handled"
  }, [isActiveConnection, requestReconnect, transitionTerminal])

  const handleConnectionFailure = useCallback((
    failure: WebSocketConnectionFailure,
    connectionIdentity?: string,
  ) => {
    if (!isActiveConnection(connectionIdentity)) return
    if (failure.recoverable) {
      requestReconnect("ws_closed")
      return
    }
    transitionTerminal("unexpected_error")
  }, [isActiveConnection, requestReconnect, transitionTerminal])

  const scheduleExpiryWarning = useCallback((absoluteExpiresAt: number) => {
    clearWarningTimer()
    const delay = Math.max(0, absoluteExpiresAt - Date.now() - EXPIRY_WARNING_LEAD_MS)
    warningTimerRef.current = setTimeout(() => {
      warningTimerRef.current = null
      setState((current) => current.status === "active"
        ? { ...current, isAbsoluteExpiryWarningVisible: true }
        : current)
    }, delay)
  }, [clearWarningTimer])

  useEffect(() => {
    if (window.parent === window) {
      transitionTerminal("unexpected_error")
      return
    }
    mountedRef.current = true

    const onMessage = (event: MessageEvent<unknown>) => {
      if (event.source !== window.parent || !isProtocolMessage(event.data)) return
      if (targetOriginRef.current && event.origin !== targetOriginRef.current) return

      if (!targetOriginRef.current) {
        targetOriginRef.current = event.origin
      }

      if (terminalRef.current) return

      if (event.data.type === "session_terminal") {
        transitionTerminal(typeof event.data.code === "string" && event.data.code
          ? event.data.code
          : "unexpected_error")
        return
      }

      if (event.data.type === "session_degraded") {
        if (
          typeof event.data.code !== "string"
          || !isParentRecoverableCode(event.data.code)
        ) {
          transitionTerminal("unexpected_error")
          return
        }
        transitionDegraded()
        return
      }

      const token = event.data.session_token
      const hasDeliveryId = Object.prototype.hasOwnProperty.call(
        event.data,
        "session_delivery_id",
      )
      const deliveryId = event.data.session_delivery_id
      const tokenExpiresAt = parseDate(event.data.session_token_expires_at)
      const absoluteExpiresAt = parseDate(event.data.absolute_expires_at)
      const agent = parseAgent(event.data.agent)
      if (
        (hasDeliveryId && !isNonBlankString(deliveryId))
        || typeof token !== "string"
        || !token.trim()
        || tokenExpiresAt === null
        || absoluteExpiresAt === null
        || !agent
      ) {
        transitionTerminal("unexpected_error")
        return
      }

      const now = Date.now()
      if (absoluteExpiresAt <= now || absoluteExpiresAt < tokenExpiresAt) {
        transitionTerminal("session_expired")
        return
      }

      if (tokenExpiresAt - now < TOKEN_REFRESH_THRESHOLD_MS) {
        requestReconnect("token_expired")
        return
      }

      recoveryInFlightRef.current = false
      generationRef.current += 1
      const session: WidgetSession = {
        token,
        tokenExpiresAt: event.data.session_token_expires_at as string,
        absoluteExpiresAt: event.data.absolute_expires_at as string,
        agent,
        generation: generationRef.current,
      }
      activeSessionGenerationRef.current = session.generation
      activeDeliveryIdRef.current = hasDeliveryId ? deliveryId as string : null
      setState({
        status: "active",
        session,
        agent,
        terminalCode: null,
        isAbsoluteExpiryWarningVisible: false,
      })
      scheduleExpiryWarning(absoluteExpiresAt)
    }

    window.addEventListener("message", onMessage)
    window.parent.postMessage({ xagent: true, v: 1, type: "ready" }, "*")

    return () => {
      mountedRef.current = false
      activeSessionGenerationRef.current = null
      activeDeliveryIdRef.current = null
      window.removeEventListener("message", onMessage)
      recoveryInFlightRef.current = false
      clearWarningTimer()
    }
    // The bridge owns its parent message subscription for the mounted lifetime.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return {
    ...state,
    requestReconnect,
    handleConnectionOpen,
    handleConnectionClose,
    handleConnectionFailure,
  }
}

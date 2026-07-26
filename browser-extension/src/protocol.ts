export const PROTOCOL_VERSION = 1
export const MAX_RECONNECT_DELAY_MS = 60_000

export type RelayConnectionState =
  | "unpaired"
  | "connecting"
  | "connected"
  | "reconnecting"
  | "offline"

export interface RelayCommand {
  type: "command"
  protocol_version: number
  request_id: string
  command: "observe" | "act" | "capture_media"
  payload: Record<string, unknown>
}

export interface RelayReady {
  type: "ready"
  protocol_version: number
  paired: boolean
  session_token?: string
}

export interface RelayError {
  type: "error"
  protocol_version: number
  error: string
}

export type ServerMessage =
  | RelayCommand
  | RelayReady
  | RelayError
  | { type: "pong"; protocol_version: number }

export interface RelayPublicStatus {
  connected: boolean
  connecting: boolean
  connectionState: RelayConnectionState
  hasSession: boolean
  reconnectAttempt: number
  nextRetryAt: number | null
  attached: boolean
  tabId: number | null
  title: string | null
  url: string | null
  error: string | null
}

export interface RelayPairingSetup {
  relayUrl: string
  pairingToken: string
}

export function parseServerMessage(raw: string): ServerMessage {
  const value: unknown = JSON.parse(raw)
  if (!isRecord(value) || typeof value.type !== "string") {
    throw new Error("Relay message must be an object with a type.")
  }
  if (value.protocol_version !== PROTOCOL_VERSION) {
    throw new Error("Relay protocol version mismatch.")
  }
  return value as unknown as ServerMessage
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

export function normalizeRelayUrl(raw: string): string {
  const url = new URL(raw.trim())
  if (url.protocol !== "ws:" && url.protocol !== "wss:") {
    throw new Error("Relay URL must use ws:// or wss://.")
  }
  if (url.username || url.password || url.search || url.hash) {
    throw new Error("Relay URL must not include credentials, query, or fragment.")
  }
  return url.toString()
}

export function parsePairingSetup(raw: string): RelayPairingSetup {
  let value: unknown
  try {
    value = JSON.parse(raw.trim())
  } catch {
    throw new Error("Pairing setup must be the JSON copied from Xagent Settings.")
  }
  if (!isRecord(value)) {
    throw new Error("Pairing setup must be a JSON object.")
  }
  const relayUrl = normalizeRelayUrl(
    requiredSetupString(value.relayUrl ?? value.websocket_url, "relay URL"),
  )
  const pairingToken = requiredSetupString(
    value.pairingToken ?? value.pairing_token,
    "pairing token",
  )
  return { relayUrl, pairingToken }
}

export function reconnectDelayMs(
  attempt: number,
  random: number = Math.random(),
): number {
  const normalizedAttempt = Math.max(1, Math.floor(attempt))
  const exponential = Math.min(
    MAX_RECONNECT_DELAY_MS,
    1_000 * 2 ** Math.min(normalizedAttempt - 1, 6),
  )
  const boundedRandom = Math.min(1, Math.max(0, random))
  return Math.min(
    MAX_RECONNECT_DELAY_MS,
    Math.round(exponential * (0.8 + boundedRandom * 0.4)),
  )
}

function requiredSetupString(value: unknown, field: string): string {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`Pairing setup is missing the ${field}.`)
  }
  return value.trim()
}

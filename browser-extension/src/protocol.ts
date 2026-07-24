export const PROTOCOL_VERSION = 1

export interface RelayCommand {
  type: "command"
  protocol_version: number
  request_id: string
  command: "observe" | "act"
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
  attached: boolean
  tabId: number | null
  title: string | null
  url: string | null
  error: string | null
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

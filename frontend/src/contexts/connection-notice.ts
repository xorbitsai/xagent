import type { ConnectionPhase } from "@/hooks/use-websocket"

export type ConnectionNotice = "reconnecting" | "lost" | null

interface ConnectionNoticeFacts {
  isSessionTransport: boolean
  // The socket opened at least once for the current task (or, for the Session
  // transport, at all) since this provider mounted.
  hadOpen: boolean
  phase: ConnectionPhase
  isConnected: boolean
}

// A first connect, a first-ever failure, and a routine socket swap (phase
// "connecting") show nothing. The Session transport is reconnected by the
// host page, so it never reports "lost".
export function deriveConnectionNotice({
  isSessionTransport,
  hadOpen,
  phase,
  isConnected,
}: ConnectionNoticeFacts): ConnectionNotice {
  if (!hadOpen) return null
  if (isSessionTransport) return isConnected ? null : "reconnecting"
  return phase === "reconnecting" || phase === "lost" ? phase : null
}

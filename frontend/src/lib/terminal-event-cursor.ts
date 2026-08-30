type TerminalEventCursorConnection = {
  taskId?: number
  credentialOwner:
    | {
      kind: "auth-context"
      userId: string | null
      session?: { sessionId: string | null }
    }
    | { kind: "external" }
}

export const terminalEventCursorKey = (
  connection: TerminalEventCursorConnection,
): string => {
  const owner = connection.credentialOwner
  const ownerScope = owner.kind === "auth-context"
    ? `auth:${owner.session?.sessionId ?? owner.userId ?? "unknown"}`
    : "external"
  return `xagent:terminal-task-event-cursor:${ownerScope}:${connection.taskId ?? ""}`
}

export const readTerminalEventCursor = (key: string): number => {
  if (typeof window === "undefined") return 0
  try {
    const parsed = Number(window.sessionStorage.getItem(key))
    return Number.isSafeInteger(parsed) && parsed >= 0 ? parsed : 0
  } catch {
    return 0
  }
}

export const writeTerminalEventCursor = (key: string, cursor: number): void => {
  if (typeof window === "undefined") return
  try {
    window.sessionStorage.setItem(key, String(cursor))
  } catch {
    // In-memory deduplication remains authoritative when storage is unavailable.
  }
}

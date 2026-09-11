/**
 * Stable identity for assistant messages, mirroring what #2254 gave user turns.
 *
 * A question reaches the client twice across a reconnect: live as its own
 * trace event, then replayed from the transcript. The two carry different
 * text -- the replayed copy has the rendered interaction list appended -- so
 * only identity can pair them. The server sends both under the originating
 * trace event's id (#2292), which is what this keys on.
 */

const ASSISTANT_EVENT_MESSAGE_PREFIX = "msg-agent-event"

/** `null` for identity-less legacy events, which keep a minted id. */
export const stableAssistantMessageId = (eventId: unknown): string | null => {
  if (typeof eventId === "string" && eventId.trim()) {
    return `${ASSISTANT_EVENT_MESSAGE_PREFIX}-${eventId.trim()}`
  }
  return null
}

export const isStableAssistantMessageId = (id: string): boolean =>
  id.startsWith(`${ASSISTANT_EVENT_MESSAGE_PREFIX}-`)

// The host page's widget.js owns panel visibility, sizing, and the auto-open
// decision; it has no direct handle into this iframe's React tree, so intent
// is signalled back over postMessage instead. The host is an arbitrary
// third-party origin from in here, so targetOrigin can't be pinned tighter
// than "*" -- every message sent this way carries no sensitive payload,
// unlike the parent -> iframe session protocol.
// widget_chrome_ready/widget_chrome_not_ready: sent by WidgetChromeControls
// on mount/unmount so widget.js knows whether the iframe currently has its
// own close control -- see the mobile FAB-hiding guard in widget.js.
export type WidgetParentMessageType =
  | "widget_close"
  | "widget_chrome_ready"
  | "widget_chrome_not_ready"
  | "widget_expand"
  | "widget_collapse"
  | "prompt_context_request"
  | "initial_message_request"
  | "initial_message_ack"

// Must stay in step with MAX_INITIAL_MESSAGE_LENGTH in public/widget.js.
export const MAX_INITIAL_MESSAGE_LENGTH = 2000

// The host's reply to this iframe, or null for anything else: another source,
// a malformed envelope, or a different type.
function hostMessageOfType(event: MessageEvent, type: string): Record<string, unknown> | null {
  if (event.source !== window.parent) return null
  const data = event.data as Record<string, unknown> | null
  return data && typeof data === "object"
    && data.xagent === true && data.v === 1 && data.type === type
    ? data
    : null
}

export function postToParentWidget(type: WidgetParentMessageType): void {
  // A direct (non-embedded) visit to this page has window.parent === window;
  // posting there would just send the widget its own message right back.
  if (window.parent !== window) {
    window.parent.postMessage({ xagent: true, v: 1, type }, "*")
  }
}

export function requestPromptContext(
  timeoutMs = 500,
): Promise<Record<string, unknown> | null> {
  if (window.parent === window) return Promise.resolve(null)
  return new Promise((resolve) => {
    const finish = (values: Record<string, unknown> | null) => {
      clearTimeout(timer)
      window.removeEventListener("message", handleMessage)
      resolve(values)
    }
    const handleMessage = (event: MessageEvent) => {
      const data = hostMessageOfType(event, "prompt_context")
      if (!data) return
      const values = data.values
      const proto = values && typeof values === "object" ? Object.getPrototypeOf(values) : undefined
      if (proto !== Object.prototype && proto !== null) {
        finish(null)
        return
      }
      finish(values as Record<string, unknown>)
    }
    window.addEventListener("message", handleMessage)
    const timer = setTimeout(() => finish(null), timeoutMs)
    postToParentWidget("prompt_context_request")
  })
}

// The other direction: widget.js can reject a widget_expand (its own mobile
// guard rejects it) and needs to correct the iframe's optimistic guess about
// its own state; prompt_context replies go through requestPromptContext instead.
// event.source, not origin, is the check here:
// the embedding page is an arbitrary third-party origin the iframe can't
// allowlist in advance (unlike the host, which set iframe.src itself and so
// already knows this iframe's real origin for its own sends).
export type WidgetHostMessageType = "widget_expand_rejected"

export function listenForWidgetHostMessage(
  type: WidgetHostMessageType,
  onMessage: () => void,
): () => void {
  const handleMessage = (event: MessageEvent) => {
    // On a direct (non-embedded) visit window.parent === window, so without
    // this a same-document script's own window.postMessage(..., "*") would
    // satisfy `event.source === window.parent` trivially -- mirrors the
    // same direct-visit guard postToParentWidget already applies above.
    if (window.parent === window || !hostMessageOfType(event, type)) return
    onMessage()
  }
  window.addEventListener("message", handleMessage)
  return () => window.removeEventListener("message", handleMessage)
}

// Posts initial_message_request, then listens. Unlike prompt_context the parent
// may answer long after the request: it holds the message until the visitor
// first opens the panel. No timeout, so the caller owns the lifetime through
// the returned cleanup. A valid message is acknowledged, and that ack is what
// stops the parent from answering again.
export function requestInitialMessage(
  onMessage: (message: string) => void,
): () => void {
  if (window.parent === window) return () => undefined
  const handleMessage = (event: MessageEvent) => {
    const data = hostMessageOfType(event, "initial_message")
    if (!data || typeof data.value !== "string") return
    const message = data.value.trim()
    if (!message || message.length > MAX_INITIAL_MESSAGE_LENGTH) return
    postToParentWidget("initial_message_ack")
    onMessage(message)
  }
  window.addEventListener("message", handleMessage)
  postToParentWidget("initial_message_request")
  return () => window.removeEventListener("message", handleMessage)
}

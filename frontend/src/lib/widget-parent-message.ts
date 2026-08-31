// The host page's widget.js owns panel visibility, sizing, and the auto-open
// decision; it has no direct handle into this iframe's React tree, so intent
// is signalled back over postMessage instead. The host is an arbitrary
// third-party origin from in here, so targetOrigin can't be pinned tighter
// than "*" -- every message sent this way carries no sensitive payload,
// unlike the parent -> iframe session protocol.
export type WidgetParentMessageType = "widget_close" | "widget_expand" | "widget_collapse"

export function postToParentWidget(type: WidgetParentMessageType): void {
  // A direct (non-embedded) visit to this page has window.parent === window;
  // posting there would just send the widget its own message right back.
  if (window.parent !== window) {
    window.parent.postMessage({ xagent: true, v: 1, type }, "*")
  }
}

// The other direction: widget.js can reject a widget_expand (its own mobile
// guard rejects it) and needs to correct the iframe's optimistic guess about
// its own state -- this is the only message the host currently ever
// initiates on this channel. event.source, not origin, is the check here:
// the embedding page is an arbitrary third-party origin the iframe can't
// allowlist in advance (unlike the host, which set iframe.src itself and so
// already knows this iframe's real origin for its own sends).
export type WidgetHostMessageType = "widget_expand_rejected"

export function listenForWidgetHostMessage(
  type: WidgetHostMessageType,
  onMessage: () => void,
): () => void {
  const handleMessage = (event: MessageEvent) => {
    if (event.source !== window.parent) return
    const data: unknown = event.data
    if (
      !data || typeof data !== "object"
      || (data as Record<string, unknown>).xagent !== true
      || (data as Record<string, unknown>).v !== 1
      || (data as Record<string, unknown>).type !== type
    ) return
    onMessage()
  }
  window.addEventListener("message", handleMessage)
  return () => window.removeEventListener("message", handleMessage)
}

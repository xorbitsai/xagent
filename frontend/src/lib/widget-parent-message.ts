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

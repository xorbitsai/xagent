import { useEffect, useRef } from "react"
import { requestInitialMessage } from "@/lib/widget-parent-message"

interface UseWidgetInitialMessageOptions {
  // True once the page knows whether a conversation already exists.
  ready: boolean
  hasConversation: boolean
  onMessage: (message: string) => void | Promise<void>
}

// One shot per mount (widget.js stops answering after the first ack): any
// conversation seen here, restored or started by the visitor, spends it for
// good, so "start new conversation" never replays the host's message.
export function useWidgetInitialMessage({
  ready,
  hasConversation,
  onMessage,
}: UseWidgetInitialMessageOptions): void {
  const consumedRef = useRef(false)
  const onMessageRef = useRef(onMessage)
  onMessageRef.current = onMessage

  useEffect(() => {
    if (ready && hasConversation) consumedRef.current = true
  }, [ready, hasConversation])

  useEffect(() => {
    if (!ready || hasConversation || consumedRef.current) return
    const stop = requestInitialMessage((message) => {
      consumedRef.current = true
      stop()
      void Promise.resolve(onMessageRef.current(message)).catch(() => undefined)
    })
    return stop
  }, [ready, hasConversation])
}

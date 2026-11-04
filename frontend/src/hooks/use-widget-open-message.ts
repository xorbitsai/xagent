import { useCallback, useEffect, useRef, useState } from "react"
import { listenForOpenMessages } from "@/lib/widget-parent-message"

export type WidgetOpenRequest = { seq: number; message: string }

// The console warning every open-message driver gives when its send fails.
export const W_SEND_FAILED = "Xagent Widget: the message from XagentWidget.open() could not be sent."

interface UseWidgetOpenMessageOptions {
  enabled: boolean
}

// Holds the host page's latest XagentWidget.open(message) for this mount: one
// slot, and a newer message replaces one not yet taken. The caller drives the
// send and claims a request with take(seq); a request that arrived after the
// caller read `request` keeps its own seq, so it is never cleared by mistake.
// widget.js re-sends a message until its ack arrives, so the last accepted id
// outlives re-subscription and a repeat of it is dropped; distinct calls always
// carry distinct ids.
export function useWidgetOpenMessage({ enabled }: UseWidgetOpenMessageOptions): {
  request: WidgetOpenRequest | null
  take: (seq: number) => void
} {
  const [request, setRequest] = useState<WidgetOpenRequest | null>(null)
  const lastIdRef = useRef<string | null>(null)
  const seqRef = useRef(0)

  useEffect(() => {
    if (!enabled) return
    return listenForOpenMessages((message, id) => {
      if (id === lastIdRef.current) return
      lastIdRef.current = id
      seqRef.current += 1
      setRequest({ seq: seqRef.current, message })
    })
  }, [enabled])

  const take = useCallback((seq: number) => {
    setRequest((current) => (current?.seq === seq ? null : current))
  }, [])

  return { request, take }
}

import { useEffect, useRef, useState } from "react"
import type { SessionConversationAttachment, useApp } from "@/contexts/app-context-chat"
import { W_SEND_FAILED, type WidgetOpenRequest } from "@/hooks/use-widget-open-message"
import type { WidgetSessionStatus } from "./use-widget-session"

type SessionConversationPhase = ReturnType<typeof useApp>["sessionConversationState"]

export const W_UNSUPPORTED = "Xagent Widget: this chat server cannot start a new conversation for XagentWidget.open(); the message was not sent."
export const W_RELOAD_REQUIRED = "Xagent Widget: the chat must be reloaded before XagentWidget.open() can send a message; the message was not sent."
export const W_RESET_FAILED = "Xagent Widget: a new conversation could not be started; the message from XagentWidget.open() was not sent."
export const W_RESET_SUPERSEDED = "Xagent Widget: a new conversation was started, but the chat moved to another conversation before XagentWidget.open() could send its message; the message was not sent."
export const W_SESSION_ENDED = "Xagent Widget: the chat session has ended; the message from XagentWidget.open() was not sent."

// What a ready driver does with an open message in each phase: send it as the
// first message of a conversation, start a new conversation first, wait for
// the phase to move on, or drop it.
const PHASE_ROUTE: Readonly<Record<SessionConversationPhase, "send" | "reset" | "wait" | "discard">> = {
  unbound: "send",
  bound: "reset",
  reset_requested: "wait",
  replacement_ready: "send",
  replacement_sending: "wait",
  replacement_awaiting_task: "wait",
  reload_required: "discard",
}

interface ActiveOpenMessage {
  message: string
  stage: "waiting" | "resetting" | "sending"
  // Set once a reset for this message has succeeded.
  reset: boolean
}

export interface UseSessionOpenMessageInput {
  request: WidgetOpenRequest | null
  take: (seq: number) => void
  status: WidgetSessionStatus
  hasAgent: boolean
  sessionConversationAttachment: SessionConversationAttachment
  sessionConversationState: SessionConversationPhase
  isMessageDeliveryPending: boolean
  isConversationResetPending: boolean
  isAttachmentBusy: boolean
  // Rejects when the conversation was not reset; the page shows its own
  // failure line before rejecting.
  startNewConversation: () => Promise<void>
  sendStartMessage: (message: string) => Promise<void>
}

// Sends the host page's XagentWidget.open(message) as the first message of a
// new conversation. One message is handled at a time: while it is in hand
// (`active`), a newer message waits in the slot and replaces any older one
// still there. Nothing is sent until the current socket has attached, so a
// conversation the server is still replaying is reset first instead of
// receiving the message. Attaching does not prove there is no previous
// conversation: if the server's replay finds nothing to send (its task was
// deleted or changed owner after the attach check), the page stays unbound and
// the message is sent as a first message, exactly as one typed on the start
// screen would be. A failure is never retried, and a message is reset at most
// once.
export function useSessionOpenMessage({
  request,
  take,
  status,
  hasAgent,
  sessionConversationAttachment,
  sessionConversationState,
  isMessageDeliveryPending,
  isConversationResetPending,
  isAttachmentBusy,
  startNewConversation,
  sendStartMessage,
}: UseSessionOpenMessageInput): void {
  const activeRef = useRef<ActiveOpenMessage | null>(null)
  const mountedRef = useRef(false)
  // Bumped when a reset or send settles, so the next step runs on a render
  // that sees the phase it left behind.
  const [settled, setSettled] = useState(0)

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  useEffect(() => {
    const active = activeRef.current
    // The message to handle next: the one in hand, otherwise the slot's.
    const next = active ?? request
    if (!next) return

    const route = PHASE_ROUTE[sessionConversationState]
    const discardWarning = status === "terminal"
      ? W_SESSION_ENDED
      : route === "discard"
        ? W_RELOAD_REQUIRED
        : sessionConversationAttachment === "unsupported"
          ? W_UNSUPPORTED
          : null
    if (discardWarning) {
      // A reset or send already under way settles first; its message is
      // dropped here on the render after, if it is still waiting.
      if (active?.stage === "waiting") {
        activeRef.current = null
        console.warn(discardWarning)
      }
      if (request) {
        take(request.seq)
        console.warn(discardWarning)
      }
      return
    }

    if (active && active.stage !== "waiting") return

    if (
      status !== "active"
      || !hasAgent
      || sessionConversationAttachment !== "attached"
      || isMessageDeliveryPending
      || isConversationResetPending
      || isAttachmentBusy
      || route === "wait"
    ) {
      return
    }

    let current: ActiveOpenMessage
    if ("seq" in next) {
      take(next.seq)
      current = { message: next.message, stage: "waiting", reset: false }
    } else {
      current = next
    }

    const settle = (next: ActiveOpenMessage | null, warning?: string) => {
      if (!mountedRef.current) return
      activeRef.current = next
      if (warning) console.warn(warning)
      setSettled((count) => count + 1)
    }

    if (route === "reset") {
      // Its reset succeeded, yet the conversation is bound again: drop it
      // rather than start a second new conversation for the same message.
      if (current.reset) {
        settle(null, W_RESET_SUPERSEDED)
        return
      }
      const resetting: ActiveOpenMessage = { ...current, stage: "resetting" }
      activeRef.current = resetting
      // An acknowledged reset moves the phase on before this promise
      // resolves, so the render it triggers sends the message into the new
      // conversation; a phase still or again at bound drops it (above).
      startNewConversation().then(
        () => settle({ message: resetting.message, stage: "waiting", reset: true }),
        () => settle(null, W_RESET_FAILED),
      )
      return
    }

    activeRef.current = { ...current, stage: "sending" }
    sendStartMessage(current.message).then(
      () => settle(null),
      () => settle(null, W_SEND_FAILED),
    )
  }, [
    request,
    take,
    status,
    hasAgent,
    sessionConversationAttachment,
    sessionConversationState,
    isMessageDeliveryPending,
    isConversationResetPending,
    isAttachmentBusy,
    startNewConversation,
    sendStartMessage,
    settled,
  ])
}

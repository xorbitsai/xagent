import { useEffect, useRef, useState } from "react"
import type { SessionConversationAttachment, useApp } from "@/contexts/app-context-chat"
import { PHASE_ROUTE } from "./use-session-open-message"
import type { WidgetSessionStatus } from "./use-widget-session"

type SessionConversationPhase = ReturnType<typeof useApp>["sessionConversationState"]

// A message the visitor wrote to go on from an earlier conversation.
export interface HistoryContinueRequest {
  seq: number
  message: string
  sourceTaskId: number
}

export type HistoryContinueOutcome =
  | "sent"            // the send was accepted
  | "send_failed"     // sendContinueMessage rejected; the page already shows its line
  | "reset_failed"    // startNewConversation rejected; the page already shows its line
  | "superseded"      // the conversation is bound again after this message's reset
  | "unsupported"     // the server does not announce attach completion
  | "reload_required" // the conversation must be reloaded
  | "ended"           // the session ended

interface ActiveContinueMessage {
  message: string
  sourceTaskId: number
  stage: "waiting" | "resetting" | "sending"
  // Set once a reset for this message has succeeded.
  reset: boolean
}

export interface UseHistoryContinueInput {
  request: HistoryContinueRequest | null
  take: (seq: number) => void
  status: WidgetSessionStatus
  hasAgent: boolean
  sessionConversationAttachment: SessionConversationAttachment
  sessionConversationState: SessionConversationPhase
  // True while the provider is delivering any message, including a host page
  // open message already being sent; this driver waits for it. An open
  // message still in its own slot does not count: this driver never waits
  // for one. The order runs the other way only through the page: a page that
  // also runs the open message driver passes this hook's `busy` into that
  // driver's `isMessageDeliveryPending`, so an open message waits until the
  // continuation has ended.
  isMessageDeliveryPending: boolean
  isConversationResetPending: boolean
  isAttachmentBusy: boolean
  // Rejects when the conversation was not reset; the page shows its own
  // failure line before rejecting.
  startNewConversation: () => Promise<void>
  sendContinueMessage: (message: string, sourceTaskId: number) => Promise<void>
  onOutcome: (outcome: HistoryContinueOutcome) => void
}

// Sends a message written over an earlier conversation as the first message
// of a new conversation that continues from it. Works like the host page's
// open message driver: one message is handled at a time, a newer one waits in
// the slot and replaces any older one still there, nothing is sent until the
// current socket has attached, a conversation that is still bound is reset
// first, a message is reset at most once, and a failure is never retried.
// Every way a message ends is reported through onOutcome, except a reset or
// send that settles after unmount, which reports nothing.
//
// `busy` is true from the render a request reaches the slot until its
// message has ended, whether it was sent, failed or dropped.
export function useHistoryContinue({
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
  sendContinueMessage,
  onOutcome,
}: UseHistoryContinueInput): { busy: boolean } {
  const activeRef = useRef<ActiveContinueMessage | null>(null)
  const mountedRef = useRef(false)
  // Bumped when a reset or send settles, so the next step runs on a render
  // that sees the phase it left behind.
  const [settled, setSettled] = useState(0)
  // Whether a message is in hand. Set when one is taken from the slot and
  // cleared whenever it ends, so `busy` turns false on the render after.
  const [held, setHeld] = useState(false)
  // The latest render's onOutcome, so a reset or send that settles later
  // reports to the page's current callback rather than the one it began with.
  const onOutcomeRef = useRef(onOutcome)

  useEffect(() => {
    onOutcomeRef.current = onOutcome
  }, [onOutcome])

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
    const discarded: HistoryContinueOutcome | null = status === "terminal"
      ? "ended"
      : route === "discard"
        ? "reload_required"
        : sessionConversationAttachment === "unsupported"
          ? "unsupported"
          : null
    if (discarded) {
      // A reset or send already under way settles first; its message is
      // dropped here on the render after, if it is still waiting.
      if (active?.stage === "waiting") {
        activeRef.current = null
        setHeld(false)
        onOutcomeRef.current(discarded)
      }
      if (request) {
        take(request.seq)
        onOutcomeRef.current(discarded)
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

    let current: ActiveContinueMessage
    if ("seq" in next) {
      take(next.seq)
      setHeld(true)
      current = { message: next.message, sourceTaskId: next.sourceTaskId, stage: "waiting", reset: false }
    } else {
      current = next
    }

    const settle = (next: ActiveContinueMessage | null, outcome?: HistoryContinueOutcome) => {
      if (!mountedRef.current) return
      activeRef.current = next
      if (outcome) onOutcomeRef.current(outcome)
      setSettled((count) => count + 1)
      setHeld(next !== null)
    }

    if (route === "reset") {
      // Its reset succeeded, yet the conversation is bound again: drop it
      // rather than start a second new conversation for the same message.
      if (current.reset) {
        settle(null, "superseded")
        return
      }
      const resetting: ActiveContinueMessage = { ...current, stage: "resetting" }
      activeRef.current = resetting
      // An acknowledged reset moves the phase on before this promise
      // resolves, so the render it triggers sends the message into the new
      // conversation; a phase still or again at bound drops it (above).
      startNewConversation().then(
        () => settle({ ...resetting, stage: "waiting", reset: true }),
        () => settle(null, "reset_failed"),
      )
      return
    }

    activeRef.current = { ...current, stage: "sending" }
    sendContinueMessage(current.message, current.sourceTaskId).then(
      () => settle(null, "sent"),
      () => settle(null, "send_failed"),
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
    sendContinueMessage,
    // onOutcome is read through onOutcomeRef, so a new callback alone does not
    // run this effect again.
    settled,
  ])

  return { busy: request !== null || held }
}

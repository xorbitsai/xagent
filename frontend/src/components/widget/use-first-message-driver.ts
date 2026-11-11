import { useEffect, useRef, useState } from "react"
import type { SessionConversationAttachment, useApp } from "@/contexts/app-context-chat"
import type { WidgetSessionStatus } from "./use-widget-session"

type SessionConversationPhase = ReturnType<typeof useApp>["sessionConversationState"]

// What a ready driver does with a message in each phase: send it as the first
// message of a conversation, start a new conversation first, wait for the
// phase to move on, or drop it.
export const PHASE_ROUTE: Readonly<Record<SessionConversationPhase, "send" | "reset" | "wait" | "discard">> = {
  unbound: "send",
  bound: "reset",
  reset_requested: "wait",
  replacement_ready: "send",
  replacement_sending: "wait",
  replacement_awaiting_task: "wait",
  reload_required: "discard",
}

export type FirstMessageOutcome =
  | "sent"            // the send was accepted
  | "send_failed"     // send rejected; the page already shows its line
  | "reset_failed"    // startNewConversation rejected; the page already shows its line
  | "superseded"      // the conversation is bound again after this message's reset
  | "unsupported"     // the server does not announce attach completion
  | "reload_required" // the conversation must be reloaded
  | "ended"           // the session ended

interface ActiveFirstMessage<R> {
  request: R
  stage: "waiting" | "resetting" | "sending"
  // Set once a reset for this message has succeeded.
  reset: boolean
}

export interface UseFirstMessageDriverInput<R extends { seq: number }> {
  request: R | null
  take: (seq: number) => void
  status: WidgetSessionStatus
  hasAgent: boolean
  sessionConversationAttachment: SessionConversationAttachment
  sessionConversationState: SessionConversationPhase
  // While true the driver takes, resets and sends nothing; it never drops a
  // message for it.
  isMessageDeliveryPending: boolean
  isConversationResetPending: boolean
  isAttachmentBusy: boolean
  // Rejects when the conversation was not reset; the page shows its own
  // failure line before rejecting.
  startNewConversation: () => Promise<void>
  send: (request: R) => Promise<void>
  onOutcome: (outcome: FirstMessageOutcome) => void
}

// Sends a message from a slot as the first message of a new conversation. One
// message is handled at a time: while it is in hand (`active`), a newer one
// waits in the slot and replaces any older one still there. Nothing is sent
// until the current socket has attached, a conversation that is still bound is
// reset first, a message is reset at most once, and a failure is never
// retried. Every way a message ends is reported through onOutcome, except a
// reset or send that settles after unmount, which reports nothing.
//
// `busy` is true from the render a request reaches the slot until its
// message has ended, whether it was sent, failed or dropped.
export function useFirstMessageDriver<R extends { seq: number }>({
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
  send,
  onOutcome,
}: UseFirstMessageDriverInput<R>): { busy: boolean } {
  const activeRef = useRef<ActiveFirstMessage<R> | null>(null)
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
    if (!active && !request) return

    const route = PHASE_ROUTE[sessionConversationState]
    const discarded: FirstMessageOutcome | null = status === "terminal"
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

    // The message to handle next: the one in hand, otherwise the slot's, which
    // is there because the effect returned at its start when neither was.
    let current: ActiveFirstMessage<R>
    if (active) {
      current = active
    } else {
      const taken = request as R
      take(taken.seq)
      setHeld(true)
      current = { request: taken, stage: "waiting", reset: false }
    }

    const settle = (next: ActiveFirstMessage<R> | null, outcome?: FirstMessageOutcome) => {
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
      const resetting: ActiveFirstMessage<R> = { ...current, stage: "resetting" }
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
    send(current.request).then(
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
    send,
    // onOutcome is read through onOutcomeRef, so a new callback alone does not
    // run this effect again.
    settled,
  ])

  return { busy: request !== null || held }
}

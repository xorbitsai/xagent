import type { MessageDeliveryDisposition } from "@/hooks/use-websocket"
import type { TaskStatus } from "@/lib/task-status"

// What the Session window does with a message sent while an earlier turn has
// not ended. Everything here is pure: the provider feeds it events and applies
// the state it returns.

export const MESSAGE_QUEUE_LIMIT = 5

export interface QueuedMessage {
  clientMessageId: string
  text: string
  // Submission order, assigned when the visitor pressed send.
  seq: number
  status: "waiting" | "sending" | "failed"
}

export interface ObservedTaskState {
  version: number | undefined
  status: TaskStatus | undefined
}

export type MessageQueuePhase =
  | { kind: "idle" }
  // The turn in progress has not been seen to end. `since` is the observed
  // state version read just before the send that started the wait (undefined
  // when that send created the conversation's task). `answerRequired`: the
  // server refused the head because the task waits on a question the version
  // comparison cannot show, since that question was observed before `since`.
  | { kind: "waiting"; since: number | undefined; answerRequired: boolean }
  // The head goes out once the send checks pass, and the server decides.
  | { kind: "ready" }
  | { kind: "sending"; clientMessageId: string; since: number | undefined }
  // Nothing goes out until the visitor presses continue.
  | { kind: "held"; reason: "turn_failed" | "turn_paused" | "send_failed" }

export interface MessageQueueState {
  items: QueuedMessage[]
  phase: MessageQueuePhase
  observed: ObservedTaskState
}

export const initialMessageQueueState: MessageQueueState = {
  items: [],
  phase: { kind: "idle" },
  observed: { version: undefined, status: undefined },
}

// "unseen": nothing newer than the wait's start. "question": the turn ended on
// a question for the visitor. "running": running, pending, or a status this
// client does not know.
export type TurnReading = "unseen" | "completed" | "failed" | "paused" | "question" | "running"

export function readTurn(observed: ObservedTaskState, since: number | undefined): TurnReading {
  if (observed.version === undefined) return "unseen"
  if (since !== undefined && observed.version <= since) return "unseen"
  switch (observed.status) {
    case "completed": return "completed"
    case "failed": return "failed"
    case "paused": return "paused"
    case "waiting_for_user": return "question"
    default: return "running"
  }
}

// Statuses in which the task can take a new turn.
const APPENDABLE: ReadonlySet<TaskStatus> = new Set(["completed", "failed", "paused"])

export type QueuedSendVerdict = "wait" | "answer_first" | "hold"

export type MessageQueueEvent =
  // `snapshot`: the frame is a periodic reconciliation snapshot.
  | { type: "OBSERVE"; observed: ObservedTaskState; snapshot: boolean }
  | { type: "DELIVERY_ACCEPTED"; clientMessageId: string; sinceVersion: number | undefined }
  | { type: "ENQUEUE"; item: QueuedMessage }
  | { type: "REQUEUE_REFUSED"; item: QueuedMessage; sinceVersion: number | undefined }
  | { type: "SEND_STARTED"; clientMessageId: string; sinceVersion: number | undefined }
  | { type: "SEND_FAILED"; clientMessageId: string; verdict: QueuedSendVerdict; newClientMessageId?: string }
  | { type: "CONTINUE" }
  | { type: "REMOVE"; clientMessageId: string }
  | { type: "CLEAR" }
  | { type: "SUPPORT_LOST" }

// What the caller tells the visitor, once, after applying a transition.
export type MessageQueueEffect = null | "send_failed" | "support_lost"

const IDLE: MessageQueuePhase = { kind: "idle" }
const READY: MessageQueuePhase = { kind: "ready" }
const held = (reason: "turn_failed" | "turn_paused" | "send_failed"): MessageQueuePhase => ({ kind: "held", reason })

// Moves a waiting queue on once the awaited turn has been seen to end. A
// snapshot that finds the task able to take a new turn while nothing newer than
// the wait's start has been seen sends the head once and lets the server decide.
// That is how a start the task process never took up, a turn that ended without
// a frame, and a refusal during a turn's wind-down are found. It needs a
// snapshot, a known start, an unchanged version, an appendable status and a
// waiting head.
function settle(state: MessageQueueState, snapshot: boolean): MessageQueueState {
  const { phase, items, observed } = state
  if (phase.kind !== "waiting") return state
  const reading = readTurn(observed, phase.since)
  if (reading === "completed") return { ...state, phase: items.length > 0 ? READY : IDLE }
  if (reading === "failed" || reading === "paused") {
    return { ...state, phase: items.length > 0 ? held(reading === "failed" ? "turn_failed" : "turn_paused") : IDLE }
  }
  const retry = reading === "unseen"
    && snapshot
    // Defensive: with no known start any versioned observation already reads
    // as the end of this turn, so an unseen reading never has an unknown start.
    && phase.since !== undefined
    && observed.status !== undefined
    && APPENDABLE.has(observed.status)
    && items[0]?.status === "waiting"
  return retry ? { ...state, phase: READY } : state
}

// Every move into waiting goes through here. Its re-check is never a snapshot's:
// a refused head waits for the next snapshot instead of going out again at once.
const enterWaiting = (
  state: MessageQueueState,
  since: number | undefined,
  answerRequired = false,
): MessageQueueState => settle({ ...state, phase: { kind: "waiting", since, answerRequired } }, false)

// Back to idle when nothing is left and no turn is awaited.
const settleEmpty = (state: MessageQueueState): MessageQueueState =>
  state.items.length === 0 && state.phase.kind !== "waiting" && state.phase.kind !== "sending"
    ? { ...state, phase: IDLE }
    : state

const mapItem = (
  items: QueuedMessage[],
  clientMessageId: string,
  change: Partial<QueuedMessage>,
) => items.map(item => (item.clientMessageId === clientMessageId ? { ...item, ...change } : item))

const without = (items: QueuedMessage[], clientMessageId: string) =>
  items.filter(item => item.clientMessageId !== clientMessageId)

export function transitionMessageQueue(
  state: MessageQueueState,
  event: MessageQueueEvent,
): { next: MessageQueueState; effect: MessageQueueEffect } {
  const { phase, items } = state
  const to = (next: MessageQueueState, effect: MessageQueueEffect = null) => ({ next, effect })
  switch (event.type) {
    case "OBSERVE": {
      const { version, status } = state.observed
      const unchanged = version === event.observed.version && status === event.observed.status
      return to(settle(unchanged ? state : { ...state, observed: event.observed }, event.snapshot))
    }
    case "DELIVERY_ACCEPTED": {
      const removed = { ...state, items: without(items, event.clientMessageId) }
      if (phase.kind === "held") {
        // Taking the last queued message out of a stopped queue leaves nothing
        // held: the accepted message is the turn now awaited.
        const emptied = items.length > 0 && removed.items.length === 0
        return to(emptied ? enterWaiting(removed, event.sinceVersion) : removed)
      }
      if (phase.kind === "sending" && phase.clientMessageId !== event.clientMessageId) return to(state)
      return to(enterWaiting(removed, event.sinceVersion))
    }
    case "ENQUEUE":
      return to({ ...state, items: [...items, event.item], phase: phase.kind === "idle" ? READY : phase })
    case "REQUEUE_REFUSED": {
      const at = items.findIndex(other => other.seq > event.item.seq)
      const inserted = { ...state, items: at < 0 ? [...items, event.item] : [...items.slice(0, at), event.item, ...items.slice(at)] }
      return to(phase.kind === "idle" ? enterWaiting(inserted, event.sinceVersion) : inserted)
    }
    case "SEND_STARTED":
      if (phase.kind !== "ready" || items[0]?.clientMessageId !== event.clientMessageId) return to(state)
      return to({
        ...state,
        items: mapItem(items, event.clientMessageId, { status: "sending" }),
        phase: { kind: "sending", clientMessageId: event.clientMessageId, since: event.sinceVersion },
      })
    case "SEND_FAILED": {
      if (phase.kind !== "sending" || phase.clientMessageId !== event.clientMessageId) return to(state)
      if (event.verdict === "hold") {
        const change = { status: "failed" as const, ...(event.newClientMessageId ? { clientMessageId: event.newClientMessageId } : {}) }
        return to({ ...state, items: mapItem(items, event.clientMessageId, change), phase: held("send_failed") }, "send_failed")
      }
      const waiting = { ...state, items: mapItem(items, event.clientMessageId, { status: "waiting" }) }
      return to(enterWaiting(waiting, phase.since, event.verdict === "answer_first"))
    }
    case "CONTINUE":
      if (phase.kind !== "held") return to(state)
      return to(settleEmpty({ ...state, phase: READY, items: items.map(item => ({ ...item, status: "waiting" })) }))
    case "REMOVE":
      if (phase.kind === "sending" && phase.clientMessageId === event.clientMessageId) return to(state)
      return to(settleEmpty({ ...state, items: without(items, event.clientMessageId) }))
    case "CLEAR":
      return to(settleEmpty({ ...state, items: phase.kind === "sending" ? items.filter(item => item.status === "sending") : [] }))
    case "SUPPORT_LOST": {
      if (items.length === 0) return to(phase.kind === "idle" ? state : { ...state, phase: IDLE })
      const failed = { ...state, items: items.map(item => ({ ...item, status: "failed" as const })) }
      return phase.kind === "held" ? to(failed) : to({ ...failed, phase: held("send_failed") }, "support_lost")
    }
  }
}

// Whether the line "answer the question above and the queued messages will
// follow" applies: the task waits on a question the visitor has not answered.
export function isAwaitingAnswer(state: MessageQueueState): boolean {
  return state.items.length > 0
    && state.phase.kind === "waiting"
    && state.observed.status === "waiting_for_user"
    && (state.phase.answerRequired || readTurn(state.observed, state.phase.since) === "question")
}

export type ComposerSubmit = "send" | "queue" | "refuse_files" | "refuse_full"

// `active`: the queue is on for this connection and the conversation is bound.
export function decideComposerSubmit(
  state: MessageQueueState,
  { active, deliveryInFlight, hasFiles }: { active: boolean; deliveryInFlight: boolean; hasFiles: boolean },
): ComposerSubmit {
  if (!active) return "send"
  // A task waiting on a question takes the visitor's message as the answer,
  // whatever the queue holds.
  if (state.observed.status === "waiting_for_user") return "send"
  if (state.phase.kind === "idle" && state.items.length === 0 && !deliveryInFlight) return "send"
  if (hasFiles) return "refuse_files"
  return state.items.length >= MESSAGE_QUEUE_LIMIT ? "refuse_full" : "queue"
}

interface DeliveryFailure {
  disposition: MessageDeliveryDisposition | null
  errorCode: unknown
  retryWithNewId: boolean
}

const readDeliveryFailure = (error: unknown): DeliveryFailure => {
  const fields = (typeof error === "object" && error !== null ? error : {}) as Record<string, unknown>
  const { disposition } = fields
  return {
    disposition: disposition === "not_sent" || disposition === "rejected" || disposition === "outcome_unknown"
      ? disposition
      : null,
    errorCode: fields.errorCode,
    retryWithNewId: fields.retryWithNewId === true,
  }
}

// A direct send the server refused only because a turn is still running.
export const isBusyRefusal = (error: unknown): boolean => {
  const { disposition, errorCode } = readDeliveryFailure(error)
  return disposition === "rejected" && errorCode === "task_busy"
}

// Where a failed send of the queue's head goes. `replaceId`: the server will
// not take that id again.
export function classifyQueuedSendFailure(error: unknown): { verdict: QueuedSendVerdict; replaceId: boolean } {
  const { disposition, errorCode, retryWithNewId } = readDeliveryFailure(error)
  const outcome = (verdict: QueuedSendVerdict, replaceId = false) => ({ verdict, replaceId })
  switch (disposition) {
    case "not_sent":
      return outcome("wait")
    case "outcome_unknown":
      return outcome(errorCode === "message_acceptance_pending" ? "wait" : "hold")
    case "rejected":
      if (retryWithNewId) return outcome("hold", true)
      if (errorCode === "task_busy") return outcome("wait")
      return outcome(errorCode === "interaction_response_required" ? "answer_first" : "hold")
    case null:
      return outcome("hold")
    default: {
      const unhandled: never = disposition
      return unhandled
    }
  }
}

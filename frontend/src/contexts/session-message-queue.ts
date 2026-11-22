import type { MessageDeliveryDisposition } from "@/hooks/use-websocket"
import type { TaskStatus } from "@/lib/task-status"

// What the Session window does with a message sent while an earlier turn has
// not ended. Everything here is pure: the provider feeds it events and applies
// the state it returns.

export const MESSAGE_QUEUE_LIMIT = 5

// How long a "not started yet" refusal waits before the same message is sent
// into the running turn again, and how many such retries one message gets.
export const SEND_NOW_RETRY_DELAY_MS = 1000
export const SEND_NOW_RETRY_LIMIT = 5

// The bookkeeping of a message being sent into the turn in progress.
export interface InsertState {
  // The server accepted the request: it holds the message until the turn takes
  // it in or hands it back.
  confirmed: boolean
  // One request is waiting for its acknowledgement. A message has at most one.
  inFlight: boolean
  // The observed state version when the request went out.
  since: number | undefined
  // The connection identity the request went out on.
  connection: string | null
  // The observed version at which the request was last sent again because the
  // turn ended without the message being seen.
  resentAt: number | undefined
  // How many "not started yet" refusals have been retried.
  retries: number
  // When the next retry is due (milliseconds); undefined when none is waiting.
  retryAt: number | undefined
}

export interface QueuedMessage {
  clientMessageId: string
  text: string
  // Submission order, assigned when the visitor pressed send.
  seq: number
  // "inserting": being sent into the turn in progress.
  status: "waiting" | "sending" | "failed" | "inserting"
  // The message could not join this turn and goes out first on the next one.
  next?: boolean
  insert?: InsertState
}

export interface ObservedTaskState {
  version: number | undefined
  status: TaskStatus | undefined
}

// "earlier_send_failed": a message sent past the queue failed after a message
// queued behind it was submitted, so that message would have gone out first.
// "send_now_unknown": a message sent into the running turn may or may not have
// arrived.
export type HeldReason =
  | "turn_failed"
  | "turn_paused"
  | "send_failed"
  | "earlier_send_failed"
  | "send_now_unknown"

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
  | { kind: "held"; reason: HeldReason }

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

// Where a failed send into the running turn goes: "next" goes out first on the
// next turn, "retry" was never sent, "later" is tried again shortly, "hold"
// stops the queue, "unknown" stops it because the message may have arrived.
export type SendNowVerdict = "next" | "hold" | "unknown" | "retry" | "later"

export type LiveInputReturnReason =
  | "run_ended"
  | "not_started"
  | "not_applied"
  | "unavailable"
  | "taken"
  | "refused"
  | "conflict"
  | "outcome_unknown"

export type MessageQueueEvent =
  // `snapshot`: the frame is a periodic reconciliation snapshot.
  | { type: "OBSERVE"; observed: ObservedTaskState; snapshot: boolean }
  | { type: "DELIVERY_ACCEPTED"; clientMessageId: string; sinceVersion: number | undefined }
  // A send failed for a reason other than a busy turn. `queuedFrom`: the
  // submission order the next queued message would get when that send started.
  | { type: "DELIVERY_FAILED"; queuedFrom: number }
  | { type: "ENQUEUE"; item: QueuedMessage }
  | { type: "REQUEUE_REFUSED"; item: QueuedMessage; sinceVersion: number | undefined }
  | { type: "SEND_STARTED"; clientMessageId: string; sinceVersion: number | undefined }
  | { type: "SEND_FAILED"; clientMessageId: string; verdict: QueuedSendVerdict; newClientMessageId?: string }
  | { type: "CONTINUE" }
  | { type: "REMOVE"; clientMessageId: string }
  | { type: "CLEAR" }
  | { type: "SUPPORT_LOST" }
  | {
    type: "SEND_NOW_REQUESTED"
    clientMessageId: string
    sinceVersion: number | undefined
    connection: string | null
    // The turn can take the message: nothing is waiting on an answer.
    canInsert: boolean
  }
  | { type: "SEND_NOW_ACCEPTED"; clientMessageId: string }
  | {
    type: "SEND_NOW_FAILED"
    clientMessageId: string
    verdict: SendNowVerdict
    newClientMessageId?: string
    now: number
  }
  | { type: "SEND_NOW_RESENT"; clientMessageId: string; atVersion: number | undefined; connection: string | null }
  | { type: "LIVE_INPUT_APPLIED"; clientMessageId: string }
  | {
    type: "LIVE_INPUT_RETURNED"
    clientMessageId: string
    reason: LiveInputReturnReason
    newClientMessageId?: string
    now: number
  }
  | { type: "SEND_NOW_SUPPORT_LOST" }

// What the caller tells the visitor, once, after applying a transition.
export type MessageQueueEffect = null | "send_failed" | "support_lost" | "send_now_unknown"

const IDLE: MessageQueuePhase = { kind: "idle" }
const READY: MessageQueuePhase = { kind: "ready" }
const held = (reason: HeldReason): MessageQueuePhase => ({ kind: "held", reason })

// Moves a waiting queue on once the awaited turn has been seen to end. A
// snapshot that finds the task able to take a new turn while nothing newer than
// the wait's start has been seen sends the head once and lets the server decide.
// That is how a start the task process never took up, a turn that ended without
// a frame, and a refusal during a turn's wind-down are found. It needs a
// snapshot, a known start, an unchanged version, an appendable status and a
// waiting head, and it is skipped while the head was refused until a question
// is answered: that wait ends with the turn the answer starts.
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
    && !phase.answerRequired
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

const isPlaced = (item: QueuedMessage) =>
  item.status === "sending" || item.status === "inserting" || item.next === true

// A message sent into the running turn whose request is on its way, or that the
// server holds, cannot be taken back: neither removing it nor clearing the
// queue drops it.
export const isPinned = (item: QueuedMessage): boolean =>
  item.status === "inserting" && (item.insert?.inFlight === true || item.insert?.confirmed === true)

const dropInsert = (item: QueuedMessage): QueuedMessage => {
  const copy = { ...item }
  delete copy.insert
  return copy
}

const dropNext = (item: QueuedMessage): QueuedMessage => {
  const copy = { ...item }
  delete copy.next
  return copy
}

// Moves a message behind the one being sent and every message already going
// into this turn or first on the next one, ahead of the rest.
function placeFirst(items: QueuedMessage[], clientMessageId: string): QueuedMessage[] {
  const target = items.find(item => item.clientMessageId === clientMessageId)
  if (!target) return items
  const rest = items.filter(item => item !== target)
  const at = rest.reduce((last, item, index) => (isPlaced(item) ? index + 1 : last), 0)
  return [...rest.slice(0, at), target, ...rest.slice(at)]
}

// Puts a message in submission order behind every message going into this turn
// or first on the next one.
function insertBySeq(items: QueuedMessage[], item: QueuedMessage): QueuedMessage[] {
  const lower = items.reduce(
    (last, other, index) => (other.status === "inserting" || other.next === true ? index + 1 : last),
    0,
  )
  const offset = items.slice(lower).findIndex(other => other.seq > item.seq)
  const at = offset < 0 ? items.length : lower + offset
  return [...items.slice(0, at), item, ...items.slice(at)]
}

function placeBySeq(items: QueuedMessage[], clientMessageId: string): QueuedMessage[] {
  const target = items.find(item => item.clientMessageId === clientMessageId)
  if (!target) return items
  return insertBySeq(items.filter(item => item !== target), dropNext(target))
}

// A message that could not join this turn: it waits at the front for the next.
const toNext = (item: QueuedMessage, newClientMessageId?: string): QueuedMessage => ({
  ...dropInsert(item),
  status: "waiting",
  next: true,
  ...(newClientMessageId ? { clientMessageId: newClientMessageId } : {}),
})

const toFailed = (item: QueuedMessage, newClientMessageId?: string): QueuedMessage => ({
  ...dropInsert(item),
  status: "failed",
  ...(newClientMessageId ? { clientMessageId: newClientMessageId } : {}),
})

// A message the turn has not started yet is tried again after a pause, up to
// the retry limit, and then goes out first on the next turn. `clearInFlight`:
// the pause follows the result of the request that was waiting.
function toRetryLater(item: QueuedMessage, now: number, clearInFlight: boolean): QueuedMessage {
  const { insert } = item
  if (!insert) return item
  if (insert.retries >= SEND_NOW_RETRY_LIMIT) return toNext(item)
  return {
    ...item,
    insert: {
      ...insert,
      retries: insert.retries + 1,
      retryAt: now + SEND_NOW_RETRY_DELAY_MS,
      confirmed: false,
      inFlight: clearInFlight ? false : insert.inFlight,
    },
  }
}

// Leaving the sending phase with a failed message stops the queue: a message
// sent into the running turn failed while another message was being sent.
function holdIfFailedItems(result: { next: MessageQueueState; effect: MessageQueueEffect }) {
  const { next } = result
  if (next.phase.kind === "sending" || !next.items.some(item => item.status === "failed")) return result
  return { next: { ...next, phase: held("send_now_unknown") }, effect: "send_now_unknown" as MessageQueueEffect }
}

type InsertOutcome =
  | { kind: "next"; newClientMessageId?: string }
  | { kind: "retry" }
  | { kind: "later"; now: number; clearInFlight: boolean }
  | { kind: "hold"; newClientMessageId?: string }
  | { kind: "unknown" }

// Applies how a message sent into the running turn ended. Only a message that
// is still being inserted can end.
function endInsert(
  state: MessageQueueState,
  clientMessageId: string,
  outcome: InsertOutcome,
): { next: MessageQueueState; effect: MessageQueueEffect } {
  const { phase, items } = state
  const target = items.find(item => item.clientMessageId === clientMessageId)
  if (phase.kind === "idle" || !target || target.status !== "inserting") return { next: state, effect: null }
  const replace = (replacement: QueuedMessage) =>
    items.map(item => (item === target ? replacement : item))
  switch (outcome.kind) {
    case "next":
      return { next: { ...state, items: replace(toNext(target, outcome.newClientMessageId)) }, effect: null }
    case "retry":
      return {
        next: { ...state, items: placeBySeq(replace(dropInsert({ ...target, status: "waiting" })), target.clientMessageId) },
        effect: null,
      }
    case "later":
      return { next: { ...state, items: replace(toRetryLater(target, outcome.now, outcome.clearInFlight)) }, effect: null }
    case "hold":
    case "unknown": {
      const unknown = outcome.kind === "unknown"
      const failed = {
        ...state,
        items: replace(toFailed(target, outcome.kind === "hold" ? outcome.newClientMessageId : undefined)),
      }
      // While a message is being sent, or the queue is already stopped, the
      // phase stays and the end of the sending phase stops the queue.
      if (phase.kind === "sending" || phase.kind === "held") return { next: failed, effect: null }
      return {
        next: { ...failed, phase: held(unknown ? "send_now_unknown" : "send_failed") },
        effect: unknown ? "send_now_unknown" : "send_failed",
      }
    }
  }
}

export function transitionMessageQueue(
  state: MessageQueueState,
  event: MessageQueueEvent,
): { next: MessageQueueState; effect: MessageQueueEffect } {
  const { phase, items } = state
  const to = (next: MessageQueueState, effect: MessageQueueEffect = null) => ({ next, effect })
  const inserting = (clientMessageId: string) =>
    items.find(item => item.clientMessageId === clientMessageId && item.status === "inserting")
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
      return holdIfFailedItems(to(enterWaiting(removed, event.sinceVersion)))
    }
    case "DELIVERY_FAILED": {
      // The queue's own head reports through SEND_FAILED, and a held queue
      // already waits for the visitor.
      if (phase.kind !== "ready" && phase.kind !== "waiting") return to(state)
      const overtakes = items.some(item => item.seq >= event.queuedFrom)
      return to(overtakes ? { ...state, phase: held("earlier_send_failed") } : state)
    }
    case "ENQUEUE":
      return to({ ...state, items: [...items, event.item], phase: phase.kind === "idle" ? READY : phase })
    case "REQUEUE_REFUSED": {
      const inserted = { ...state, items: insertBySeq(items, event.item) }
      return to(phase.kind === "idle" || phase.kind === "ready" ? enterWaiting(inserted, event.sinceVersion) : inserted)
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
      return holdIfFailedItems(to(enterWaiting(waiting, phase.since, event.verdict === "answer_first")))
    }
    case "CONTINUE":
      if (phase.kind !== "held") return to(state)
      return to(settleEmpty({
        ...state,
        phase: READY,
        items: items.map(item => (item.status === "inserting" ? item : { ...item, status: "waiting" as const })),
      }))
    case "REMOVE": {
      if (phase.kind === "sending" && phase.clientMessageId === event.clientMessageId) return to(state)
      // A message the server holds, or whose request is on its way, cannot be
      // taken back; one that only waits to be tried again can.
      const target = items.find(item => item.clientMessageId === event.clientMessageId)
      if (target !== undefined && isPinned(target)) return to(state)
      return to(settleEmpty({ ...state, items: without(items, event.clientMessageId) }))
    }
    case "CLEAR": {
      // Messages the server holds or a request is on its way for stay.
      const keeps = (item: QueuedMessage) => isPinned(item) || (phase.kind === "sending" && item.status === "sending")
      const kept = { ...state, items: items.filter(keeps) }
      if (phase.kind === "held" && kept.items.length > 0) return to(enterWaiting(kept, state.observed.version))
      return to(settleEmpty(kept))
    }
    case "SUPPORT_LOST": {
      if (items.length === 0) return to(phase.kind === "idle" ? state : { ...state, phase: IDLE })
      const failed = { ...state, items: items.map(item => ({ ...item, status: "failed" as const })) }
      return phase.kind === "held" ? to(failed) : to({ ...failed, phase: held("send_failed") }, "support_lost")
    }
    case "SEND_NOW_REQUESTED": {
      const target = items.find(item => item.clientMessageId === event.clientMessageId)
      if (phase.kind === "idle" || !target) return to(state)
      if (phase.kind === "held") {
        if (target.status !== "waiting" && target.status !== "failed") return to(state)
        const marked = items.map(item => (item === target ? { ...item, next: true } : item))
        return transitionMessageQueue(
          { ...state, items: placeFirst(marked, event.clientMessageId) },
          { type: "CONTINUE" },
        )
      }
      if (target.status !== "waiting") return to(state)
      if (phase.kind === "ready") {
        const marked = items.map(item => (item === target ? { ...item, next: true } : item))
        return to({ ...state, items: placeFirst(marked, event.clientMessageId) })
      }
      if (!event.canInsert) return to(state)
      const insert: InsertState = {
        confirmed: false,
        inFlight: true,
        since: event.sinceVersion,
        connection: event.connection,
        resentAt: undefined,
        retries: 0,
        retryAt: undefined,
      }
      const marked = items.map(item => (item === target ? { ...item, status: "inserting" as const, insert } : item))
      return to({ ...state, items: placeFirst(marked, event.clientMessageId) })
    }
    case "SEND_NOW_ACCEPTED": {
      const target = inserting(event.clientMessageId)
      if (phase.kind === "idle" || !target?.insert) return to(state)
      // An acknowledgement that arrives while the message waits to be tried
      // again belongs to the attempt the "not started yet" answer ended.
      const insert = { ...target.insert, inFlight: false, confirmed: target.insert.retryAt === undefined }
      return to({ ...state, items: mapItem(items, event.clientMessageId, { insert }) })
    }
    case "SEND_NOW_FAILED": {
      const { verdict } = event
      switch (verdict) {
        case "next": return endInsert(state, event.clientMessageId, { kind: "next" })
        case "retry": return endInsert(state, event.clientMessageId, { kind: "retry" })
        case "later": return endInsert(state, event.clientMessageId, { kind: "later", now: event.now, clearInFlight: true })
        case "hold": return endInsert(state, event.clientMessageId, { kind: "hold", newClientMessageId: event.newClientMessageId })
        case "unknown": return endInsert(state, event.clientMessageId, { kind: "unknown" })
        default: {
          const unhandled: never = verdict
          return unhandled
        }
      }
    }
    case "SEND_NOW_RESENT": {
      const target = inserting(event.clientMessageId)
      if (phase.kind === "idle" || !target?.insert) return to(state)
      const insert = {
        ...target.insert,
        inFlight: true,
        confirmed: false,
        connection: event.connection,
        resentAt: event.atVersion,
        retryAt: undefined,
      }
      return to({ ...state, items: mapItem(items, event.clientMessageId, { insert }) })
    }
    case "LIVE_INPUT_APPLIED":
      if (phase.kind === "idle" || !inserting(event.clientMessageId)) return to(state)
      return to(settleEmpty({ ...state, items: without(items, event.clientMessageId) }))
    case "LIVE_INPUT_RETURNED": {
      const { clientMessageId, reason } = event
      switch (reason) {
        case "run_ended": return endInsert(state, clientMessageId, { kind: "next" })
        case "not_applied":
          return endInsert(state, clientMessageId, { kind: "next", newClientMessageId: event.newClientMessageId })
        case "not_started":
          return endInsert(state, clientMessageId, { kind: "later", now: event.now, clearInFlight: false })
        case "taken":
          if (phase.kind === "idle" || !inserting(clientMessageId)) return to(state)
          return to(settleEmpty({ ...state, items: without(items, clientMessageId) }))
        case "unavailable":
        case "refused":
        case "conflict":
          return endInsert(state, clientMessageId, { kind: "hold" })
        case "outcome_unknown": return endInsert(state, clientMessageId, { kind: "unknown" })
        default: {
          const unhandled: never = reason
          return unhandled
        }
      }
    }
    case "SEND_NOW_SUPPORT_LOST": {
      if (phase.kind === "idle" || !items.some(item => item.status === "inserting")) return to(state)
      const failed = { ...state, items: items.map(item => (item.status === "inserting" ? toFailed(item) : item)) }
      if (phase.kind === "sending" || phase.kind === "held") return to(failed)
      return to({ ...failed, phase: held("send_now_unknown") }, "send_now_unknown")
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

// Where a failed send into the running turn goes. `replaceId`: the server will
// not take that id again. Null: the connection was lost while the request
// waited, so the same id is sent again on the next connection instead.
export function classifySendNowFailure(
  error: unknown,
): { verdict: SendNowVerdict; replaceId: boolean } | null {
  const { disposition, errorCode, retryWithNewId } = readDeliveryFailure(error)
  const fields = (typeof error === "object" && error !== null ? error : {}) as Record<string, unknown>
  const outcome = (verdict: SendNowVerdict, replaceId = false) => ({ verdict, replaceId })
  switch (disposition) {
    case "not_sent":
      return outcome("retry")
    case "outcome_unknown":
      if (errorCode === "live_input_outcome_unknown") return outcome("unknown")
      return fields.connectionLost === true ? null : outcome("unknown")
    case "rejected":
      if (retryWithNewId) return outcome("hold", true)
      switch (errorCode) {
        case "live_input_not_running":
        case "interaction_response_required":
          return outcome("next")
        case "live_input_not_started":
          return outcome("later")
        case "live_input_outcome_unknown":
          return outcome("unknown")
        default:
          return outcome("hold")
      }
    case null:
      return outcome("hold")
    default: {
      const unhandled: never = disposition
      return unhandled
    }
  }
}

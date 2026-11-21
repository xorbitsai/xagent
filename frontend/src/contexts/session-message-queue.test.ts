import { describe, expect, it } from "vitest"

import { MessageDeliveryError } from "@/hooks/use-websocket"
import type { ClientErrorCode } from "@/lib/client-errors"
import type { TaskStatus } from "@/lib/task-status"
import {
  MESSAGE_QUEUE_LIMIT,
  classifyQueuedSendFailure,
  classifySendNowFailure,
  decideComposerSubmit,
  isAwaitingAnswer,
  isBusyRefusal,
  transitionMessageQueue,
  type InsertState,
  type LiveInputReturnReason,
  type MessageQueueEffect,
  type MessageQueueEvent,
  type MessageQueuePhase,
  type MessageQueueState,
  type QueuedMessage,
} from "./session-message-queue"

const item = (id: string, seq: number, status: QueuedMessage["status"] = "waiting"): QueuedMessage =>
  ({ clientMessageId: id, text: id, seq, status })
const insertOf = (over: Partial<InsertState> = {}): InsertState => ({
  confirmed: false, inFlight: true, since: 1, connection: "c1", resentAt: undefined, retries: 0, retryAt: undefined, ...over,
})
const ins = (id: string, seq: number, over: Partial<InsertState> = {}): QueuedMessage =>
  ({ clientMessageId: id, text: id, seq, status: "inserting", insert: insertOf(over) })
const nxt = (id: string, seq: number, status: QueuedMessage["status"] = "waiting"): QueuedMessage =>
  ({ ...item(id, seq, status), next: true })
const state = (
  phase: MessageQueuePhase,
  items: QueuedMessage[] = [],
  observed: { version?: number; status?: TaskStatus } = {},
): MessageQueueState => ({ items, phase, observed: { version: observed.version, status: observed.status } })
const waiting = (since: number | undefined, answerRequired = false): MessageQueuePhase =>
  ({ kind: "waiting", since, answerRequired })
const sending = (id: string, since?: number): MessageQueuePhase => ({ kind: "sending", clientMessageId: id, since })
const HELD: MessageQueuePhase = { kind: "held", reason: "turn_failed" }
const READY: MessageQueuePhase = { kind: "ready" }
const IDLE: MessageQueuePhase = { kind: "idle" }
const apply = (from: MessageQueueState, event: MessageQueueEvent) => transitionMessageQueue(from, event)
const observe = (version: number | undefined, status: TaskStatus | undefined, snapshot = false): MessageQueueEvent =>
  ({ type: "OBSERVE", observed: { version, status }, snapshot })
const ids = (from: MessageQueueState) => from.items.map(entry => `${entry.clientMessageId}:${entry.status}`)
const firstInsert = (from: MessageQueueState, id: string) =>
  from.items.find(entry => entry.clientMessageId === id)?.insert

describe("decideComposerSubmit", () => {
  const A = { active: true, deliveryInFlight: false, hasFiles: false }
  const full = Array.from({ length: MESSAGE_QUEUE_LIMIT }, (_, index) => item(`m${index}`, index))
  const phases: Array<[string, MessageQueuePhase]> = [
    ["idle", IDLE], ["waiting", waiting(1)], ["ready", READY], ["sending", sending("m0")], ["held", HELD],
  ]

  it("returns the listed outcome for every submit case", () => {
    const cases: Array<[string, MessageQueueState, Parameters<typeof decideComposerSubmit>[1], string]> = [
      ["queue off", state(waiting(1), [item("a", 1)]), { ...A, active: false }, "send"],
      ["idle and nothing in flight", state(IDLE), A, "send"],
      ["idle, files are fine", state(IDLE), { ...A, hasFiles: true }, "send"],
      ["idle, another delivery in flight", state(IDLE), { ...A, deliveryInFlight: true }, "queue"],
      ["idle, another delivery in flight, files", state(IDLE), { ...A, deliveryInFlight: true, hasFiles: true }, "refuse_files"],
      ["waiting", state(waiting(1), [item("a", 1)]), A, "queue"],
      ["waiting, files", state(waiting(1)), { ...A, hasFiles: true }, "refuse_files"],
      ["ready", state(READY, [item("a", 1)]), A, "queue"],
      ["sending", state(sending("a"), [item("a", 1, "sending")]), A, "queue"],
      ["held", state(HELD, [item("a", 1)]), A, "queue"],
      ["one below the limit", state(waiting(1), full.slice(1)), A, "queue"],
      ["at the limit", state(waiting(1), full), A, "refuse_full"],
      ["at the limit, files", state(waiting(1), full), { ...A, hasFiles: true }, "refuse_files"],
    ]
    for (const [name, from, input, expected] of cases) {
      expect(decideComposerSubmit(from, input), name).toBe(expected)
    }
  })

  it.each(phases)("sends an answer to a pending question straight away while %s", (_name, phase) => {
    const from = state(phase, full, { version: 3, status: "waiting_for_user" })
    expect(decideComposerSubmit(from, { ...A, deliveryInFlight: true, hasFiles: true })).toBe("send")
  })
})

describe("classifyQueuedSendFailure", () => {
  const delivery = (
    disposition: "not_sent" | "rejected" | "outcome_unknown",
    errorCode: ClientErrorCode | null,
    retryWithNewId = false,
  ) => new MessageDeliveryError("failed", disposition, { errorCode, retryWithNewId })

  it.each([
    ["not sent", delivery("not_sent", null), "wait", false],
    ["busy", delivery("rejected", "task_busy"), "wait", false],
    ["answer first", delivery("rejected", "interaction_response_required"), "answer_first", false],
    ["still processing elsewhere", delivery("outcome_unknown", "message_acceptance_pending"), "wait", false],
    ["id unusable", delivery("rejected", "message_not_accepted", true), "hold", true],
    ["planned restart", delivery("rejected", "execution_queue_full"), "hold", false],
    ["other refusal", delivery("rejected", "task_unavailable"), "hold", false],
    ["pointer contention", delivery("rejected", "message_acceptance_pending"), "hold", false],
    ["outcome unknown", delivery("outcome_unknown", "message_outcome_unknown"), "hold", false],
    ["plain error", new Error("boom"), "hold", false],
    ["not an error", undefined, "hold", false],
  ])("%s", (_name, error, verdict, replaceId) => {
    expect(classifyQueuedSendFailure(error)).toEqual({ verdict, replaceId })
  })

  it("reads only a rejected busy refusal as a busy direct send", () => {
    expect(isBusyRefusal(delivery("rejected", "task_busy"))).toBe(true)
    expect(isBusyRefusal(delivery("outcome_unknown", "task_busy"))).toBe(false)
    expect(isBusyRefusal(delivery("rejected", "interaction_response_required"))).toBe(false)
    expect(isBusyRefusal(new Error("busy"))).toBe(false)
  })
})

describe("classifySendNowFailure", () => {
  const delivery = (
    disposition: "not_sent" | "rejected" | "outcome_unknown",
    errorCode: ClientErrorCode | null,
    options: { retryWithNewId?: boolean; connectionLost?: boolean } = {},
  ) => new MessageDeliveryError("failed", disposition, { errorCode, ...options })

  it.each([
    ["not sent", delivery("not_sent", null), { verdict: "retry", replaceId: false }],
    ["no run in progress", delivery("rejected", "live_input_not_running"), { verdict: "next", replaceId: false }],
    ["a question is pending", delivery("rejected", "interaction_response_required"), { verdict: "next", replaceId: false }],
    ["run not started yet", delivery("rejected", "live_input_not_started"), { verdict: "later", replaceId: false }],
    ["unavailable", delivery("rejected", "live_input_unavailable"), { verdict: "hold", replaceId: false }],
    ["id conflict", delivery("rejected", "message_id_conflict", { retryWithNewId: true }), { verdict: "hold", replaceId: true }],
    ["quota", delivery("rejected", "message_not_accepted"), { verdict: "hold", replaceId: false }],
    ["server outcome unknown, rejected", delivery("rejected", "live_input_outcome_unknown"), { verdict: "unknown", replaceId: false }],
    ["server outcome unknown", delivery("outcome_unknown", "live_input_outcome_unknown"), { verdict: "unknown", replaceId: false }],
    ["timed out", delivery("outcome_unknown", "message_outcome_unknown"), { verdict: "unknown", replaceId: false }],
    ["connection lost while waiting", delivery("outcome_unknown", "message_outcome_unknown", { connectionLost: true }), null],
    ["plain error", new Error("boom"), { verdict: "hold", replaceId: false }],
  ])("%s", (_name, error, expected) => {
    expect(classifySendNowFailure(error)).toEqual(expected)
  })
})

describe("transitionMessageQueue", () => {
  const A = item("a", 1)
  const B = item("b", 2)

  it("follows the transition table", () => {
    type Row = [string, MessageQueueState, MessageQueueEvent, MessageQueuePhase, string[]?]
    const rows: Row[] = [
      ["observe writes the value and keeps an idle phase", state(IDLE), observe(2, "running"), IDLE],
      ["observe: a newer completed turn readies the head", state(waiting(1), [A]), observe(2, "completed"), READY],
      ["observe: a newer completed turn with nothing queued is idle", state(waiting(1)), observe(2, "completed"), IDLE],
      ["observe: a newer failed turn holds", state(waiting(1), [A]), observe(2, "failed"), { kind: "held", reason: "turn_failed" }],
      ["observe: a newer paused turn holds", state(waiting(1), [A]), observe(2, "paused"), { kind: "held", reason: "turn_paused" }],
      ["observe: a failed turn with nothing queued is idle", state(waiting(1)), observe(2, "failed"), IDLE],
      ["observe: a question does not move on", state(waiting(1), [A]), observe(2, "waiting_for_user", true), waiting(1)],
      ["observe: a running turn does not move on", state(waiting(1), [A]), observe(2, "running", true), waiting(1)],
      ["observe: a snapshot at the start version readies the head", state(waiting(1), [A]), observe(1, "completed", true), READY],
      ["observe: a snapshot of a failed task at the start readies the head", state(waiting(1), [A]), observe(1, "failed", true), READY],
      ["observe: a snapshot does not resend a head refused until a question is answered", state(waiting(1, true), [A]), observe(1, "completed", true), waiting(1, true)],
      ["observe: not a snapshot", state(waiting(1), [A]), observe(1, "completed", false), waiting(1)],
      ["observe: a created task's first completed observation ends the wait", state(waiting(undefined), [A]), observe(1, "completed", false), READY],
      ["observe: status not appendable", state(waiting(1), [A]), observe(1, "running", true), waiting(1)],
      ["observe: status waiting for an answer", state(waiting(1), [A]), observe(1, "waiting_for_user", true), waiting(1)],
      ["observe: head not waiting", state(waiting(1), [item("a", 1, "failed")]), observe(1, "completed", true), waiting(1)],
      ["observe: a held queue stays held", state(HELD, [A]), observe(5, "completed", true), HELD],
      ["observe: a sending queue stays sending", state(sending("a", 1), [item("a", 1, "sending")]), observe(5, "completed", true), sending("a", 1)],
      ["accepted: idle waits from the version it read", state(IDLE), { type: "DELIVERY_ACCEPTED", clientMessageId: "x", sinceVersion: 3 }, waiting(3)],
      ["accepted: waiting restarts from the new version", state(waiting(1), [A]), { type: "DELIVERY_ACCEPTED", clientMessageId: "x", sinceVersion: 4 }, waiting(4)],
      ["accepted: ready waits", state(READY, [A]), { type: "DELIVERY_ACCEPTED", clientMessageId: "x", sinceVersion: 4 }, waiting(4)],
      ["accepted: the end frame came first", state(IDLE, [], { version: 4, status: "completed" }), { type: "DELIVERY_ACCEPTED", clientMessageId: "x", sinceVersion: 3 }, IDLE],
      ["accepted: the head being sent leaves the list", state(sending("a", 2), [item("a", 1, "sending"), B]), { type: "DELIVERY_ACCEPTED", clientMessageId: "a", sinceVersion: 2 }, waiting(2), ["b:waiting"]],
      ["accepted: another id while sending changes nothing", state(sending("a", 2), [item("a", 1, "sending")]), { type: "DELIVERY_ACCEPTED", clientMessageId: "x", sinceVersion: 9 }, sending("a", 2), ["a:sending"]],
      ["accepted: a held queue stays held", state(HELD, [A]), { type: "DELIVERY_ACCEPTED", clientMessageId: "x", sinceVersion: 9 }, HELD],
      ["accepted: the last queued message accepted from a held queue waits", state(HELD, [A]), { type: "DELIVERY_ACCEPTED", clientMessageId: "a", sinceVersion: 9 }, waiting(9), []],
      ["accepted: one of several queued messages leaves a held queue held", state(HELD, [A, B]), { type: "DELIVERY_ACCEPTED", clientMessageId: "a", sinceVersion: 9 }, HELD, ["b:waiting"]],
      ["delivery failed: ready holds when a message was queued after the failed send started", state(READY, [A, B]), { type: "DELIVERY_FAILED", queuedFrom: 2 }, { kind: "held", reason: "earlier_send_failed" }, ["a:waiting", "b:waiting"]],
      ["delivery failed: ready goes on when every queued message came before it", state(READY, [A]), { type: "DELIVERY_FAILED", queuedFrom: 2 }, READY, ["a:waiting"]],
      ["delivery failed: waiting holds when a message was queued after the failed send started", state(waiting(1), [A, B]), { type: "DELIVERY_FAILED", queuedFrom: 2 }, { kind: "held", reason: "earlier_send_failed" }, ["a:waiting", "b:waiting"]],
      ["delivery failed: waiting on a question keeps waiting when every queued message came before it", state(waiting(1, true), [A]), { type: "DELIVERY_FAILED", queuedFrom: 2 }, waiting(1, true), ["a:waiting"]],
      ["delivery failed: waiting with nothing queued keeps waiting", state(waiting(1)), { type: "DELIVERY_FAILED", queuedFrom: 0 }, waiting(1), []],
      ["delivery failed: idle stays idle", state(IDLE), { type: "DELIVERY_FAILED", queuedFrom: 0 }, IDLE, []],
      ["delivery failed: the head being sent decides for itself", state(sending("a", 1), [item("a", 1, "sending"), B]), { type: "DELIVERY_FAILED", queuedFrom: 0 }, sending("a", 1), ["a:sending", "b:waiting"]],
      ["delivery failed: a held queue keeps its reason", state(HELD, [A, B]), { type: "DELIVERY_FAILED", queuedFrom: 0 }, HELD, ["a:waiting", "b:waiting"]],
      ["enqueue: idle becomes ready", state(IDLE), { type: "ENQUEUE", item: A }, READY, ["a:waiting"]],
      ["enqueue: held stays held", state(HELD, [A]), { type: "ENQUEUE", item: B }, HELD, ["a:waiting", "b:waiting"]],
      ["requeue: idle waits and orders by submission", state(IDLE, [B]), { type: "REQUEUE_REFUSED", item: A, sinceVersion: 3 }, waiting(3), ["a:waiting", "b:waiting"]],
      ["requeue: ready waits from the refused send's start", state(READY, [B]), { type: "REQUEUE_REFUSED", item: A, sinceVersion: 3 }, waiting(3), ["a:waiting", "b:waiting"]],
      ["requeue: waiting keeps its start", state(waiting(1), [A]), { type: "REQUEUE_REFUSED", item: B, sinceVersion: 7 }, waiting(1), ["a:waiting", "b:waiting"]],
      ["started: the head is sent from ready", state(READY, [A, B]), { type: "SEND_STARTED", clientMessageId: "a", sinceVersion: 2 }, sending("a", 2), ["a:sending", "b:waiting"]],
      ["started: not the head", state(READY, [A, B]), { type: "SEND_STARTED", clientMessageId: "b", sinceVersion: 2 }, READY, ["a:waiting", "b:waiting"]],
      ["started: not ready", state(waiting(1), [A]), { type: "SEND_STARTED", clientMessageId: "a", sinceVersion: 2 }, waiting(1)],
      ["failed: wait returns to the start of the refused send", state(sending("a", 2), [item("a", 1, "sending")]), { type: "SEND_FAILED", clientMessageId: "a", verdict: "wait" }, waiting(2), ["a:waiting"]],
      ["failed: answer first sets the flag", state(sending("a", 2), [item("a", 1, "sending")]), { type: "SEND_FAILED", clientMessageId: "a", verdict: "answer_first" }, waiting(2, true), ["a:waiting"]],
      ["failed: hold with a new id", state(sending("a", 2), [item("a", 1, "sending"), B]), { type: "SEND_FAILED", clientMessageId: "a", verdict: "hold", newClientMessageId: "a2" }, { kind: "held", reason: "send_failed" }, ["a2:failed", "b:waiting"]],
      ["failed: another id is ignored", state(sending("a", 2), [item("a", 1, "sending")]), { type: "SEND_FAILED", clientMessageId: "x", verdict: "hold" }, sending("a", 2), ["a:sending"]],
      ["continue: held readies and revives failed items", state(HELD, [item("a", 1, "failed"), B]), { type: "CONTINUE" }, READY, ["a:waiting", "b:waiting"]],
      ["continue: only from held", state(waiting(1), [A]), { type: "CONTINUE" }, waiting(1)],
      ["remove: waiting stays waiting when emptied", state(waiting(1), [A]), { type: "REMOVE", clientMessageId: "a" }, waiting(1), []],
      ["remove: ready is idle when emptied", state(READY, [A]), { type: "REMOVE", clientMessageId: "a" }, IDLE, []],
      ["remove: held is idle when emptied", state(HELD, [A]), { type: "REMOVE", clientMessageId: "a" }, IDLE, []],
      ["remove: the head being sent stays", state(sending("a", 1), [item("a", 1, "sending"), B]), { type: "REMOVE", clientMessageId: "a" }, sending("a", 1), ["a:sending", "b:waiting"]],
      ["remove: a later item goes while sending", state(sending("a", 1), [item("a", 1, "sending"), B]), { type: "REMOVE", clientMessageId: "b" }, sending("a", 1), ["a:sending"]],
      ["clear: waiting stays waiting", state(waiting(1), [A, B]), { type: "CLEAR" }, waiting(1), []],
      ["clear: held is idle", state(HELD, [A, B]), { type: "CLEAR" }, IDLE, []],
      ["clear: the head being sent stays", state(sending("a", 1), [item("a", 1, "sending"), B]), { type: "CLEAR" }, sending("a", 1), ["a:sending"]],
      ["support lost: a waiting queue fails", state(waiting(1), [A, B]), { type: "SUPPORT_LOST" }, { kind: "held", reason: "send_failed" }, ["a:failed", "b:failed"]],
      ["support lost: the head being sent fails too", state(sending("a", 1), [item("a", 1, "sending"), B]), { type: "SUPPORT_LOST" }, { kind: "held", reason: "send_failed" }, ["a:failed", "b:failed"]],
      ["support lost: nothing queued while waiting", state(waiting(1)), { type: "SUPPORT_LOST" }, IDLE, []],
      ["support lost: nothing happening", state(IDLE), { type: "SUPPORT_LOST" }, IDLE, []],
    ]
    for (const [name, from, event, phase, expectedIds] of rows) {
      const { next } = apply(from, event)
      expect(next.phase, name).toEqual(phase)
      if (expectedIds) expect(ids(next), name).toEqual(expectedIds)
    }
  })

  it("re-checks a refused head only against the next snapshot", () => {
    // The task already reads completed at the start version, so a re-check that
    // counted as a snapshot would send the refused head again at once.
    const started = apply(state(READY, [A], { version: 2, status: "completed" }), { type: "SEND_STARTED", clientMessageId: "a", sinceVersion: 2 }).next
    const refused = apply(started, { type: "SEND_FAILED", clientMessageId: "a", verdict: "wait" }).next
    expect(refused.phase).toEqual(waiting(2))
    expect(apply(refused, observe(2, "completed", false)).next.phase).toEqual(waiting(2))
    expect(apply(refused, observe(2, "completed", true)).next.phase).toEqual(READY)
  })

  it("returns the same state for an observation that changes nothing", () => {
    const unchanged = state(waiting(1), [A], { version: 1, status: "running" })
    expect(apply(unchanged, observe(1, "running", true)).next).toBe(unchanged)
  })

  it("clears the answer flag with the next wait", () => {
    const flagged = state(waiting(2, true), [A], { version: 2, status: "waiting_for_user" })
    expect(isAwaitingAnswer(flagged)).toBe(true)
    const answered = apply(flagged, { type: "DELIVERY_ACCEPTED", clientMessageId: "x", sinceVersion: 2 }).next
    expect(answered.phase).toEqual(waiting(2))
    expect(isAwaitingAnswer(answered)).toBe(false)
  })

  it("tells the visitor once per transition that needs it", () => {
    const failing = state(sending("a", 1), [item("a", 1, "sending")])
    expect(apply(failing, { type: "SEND_FAILED", clientMessageId: "a", verdict: "hold" }).effect).toBe("send_failed")
    expect(apply(failing, { type: "SEND_FAILED", clientMessageId: "a", verdict: "wait" }).effect).toBeNull()
    expect(apply(state(waiting(1), [A]), { type: "SUPPORT_LOST" }).effect).toBe("support_lost")
    expect(apply(state(HELD, [A]), { type: "SUPPORT_LOST" }).effect).toBeNull()
    expect(apply(state(READY, [A]), { type: "DELIVERY_FAILED", queuedFrom: 0 }).effect).toBeNull()
    const idle = state(IDLE)
    expect(apply(idle, { type: "SUPPORT_LOST" })).toEqual({ next: idle, effect: null })
    expect(apply(idle, { type: "SUPPORT_LOST" }).next).toBe(idle)
  })

  it.each([
    ["no items", state(waiting(1), [], { version: 2, status: "waiting_for_user" }), false],
    ["not waiting", state(READY, [A], { version: 2, status: "waiting_for_user" }), false],
    ["status is not a question", state(waiting(1), [A], { version: 2, status: "running" }), false],
    ["a question newer than the start", state(waiting(1), [A], { version: 2, status: "waiting_for_user" }), true],
    ["a question seen before the start", state(waiting(2), [A], { version: 2, status: "waiting_for_user" }), false],
    ["a question the server said must be answered", state(waiting(2, true), [A], { version: 2, status: "waiting_for_user" }), true],
  ])("isAwaitingAnswer: %s", (_name, from, expected) => {
    expect(isAwaitingAnswer(from)).toBe(expected)
  })
})

describe("sending a queued message into the running turn", () => {
  const A = item("a", 1)
  const B = item("b", 2)
  const H = item("h", 0, "sending")
  const NOW = 10_000
  const requested = (id: string, canInsert = true): MessageQueueEvent =>
    ({ type: "SEND_NOW_REQUESTED", clientMessageId: id, sinceVersion: 1, connection: "c1", canInsert })
  const failedWith = (id: string, verdict: "next" | "hold" | "unknown" | "retry" | "later", extra: { newClientMessageId?: string } = {}): MessageQueueEvent =>
    ({ type: "SEND_NOW_FAILED", clientMessageId: id, verdict, now: NOW, ...extra })
  const returned = (id: string, reason: LiveInputReturnReason, extra: { newClientMessageId?: string } = {}): MessageQueueEvent =>
    ({ type: "LIVE_INPUT_RETURNED", clientMessageId: id, reason, now: NOW, ...extra })
  const SENDING = sending("h", 1)
  const HELD_SENT: MessageQueuePhase = { kind: "held", reason: "send_failed" }
  const HELD_UNKNOWN: MessageQueuePhase = { kind: "held", reason: "send_now_unknown" }

  it("follows every new cell of the transition table", () => {
    const rows: Array<[string, MessageQueueState, MessageQueueEvent, MessageQueuePhase, string[], MessageQueueEffect?]> = [
      // REQUEUE_REFUSED goes behind every message being inserted or going out next.
      ["requeue: waiting", state(waiting(1), [ins("x", 3), B]), { type: "REQUEUE_REFUSED", item: A, sinceVersion: 9 }, waiting(1), ["x:inserting", "a:waiting", "b:waiting"]],
      ["requeue: ready", state(READY, [nxt("x", 3), B]), { type: "REQUEUE_REFUSED", item: A, sinceVersion: 9 }, waiting(9), ["x:waiting", "a:waiting", "b:waiting"]],
      ["requeue: sending", state(SENDING, [H, ins("x", 3), B]), { type: "REQUEUE_REFUSED", item: A, sinceVersion: 9 }, SENDING, ["h:sending", "x:inserting", "a:waiting", "b:waiting"]],
      ["requeue: held", state(HELD, [ins("x", 3), B]), { type: "REQUEUE_REFUSED", item: A, sinceVersion: 9 }, HELD, ["x:inserting", "a:waiting", "b:waiting"]],
      // CONTINUE leaves a message being inserted alone.
      ["continue: held", state(HELD, [ins("x", 3), item("a", 1, "failed"), B]), { type: "CONTINUE" }, READY, ["x:inserting", "a:waiting", "b:waiting"]],
      // REMOVE: refused while a request is on its way or the server holds the message.
      ["remove: waiting, request in flight", state(waiting(1), [ins("x", 1), B]), { type: "REMOVE", clientMessageId: "x" }, waiting(1), ["x:inserting", "b:waiting"]],
      ["remove: ready, retry waiting", state(READY, [ins("x", 1, { inFlight: false, retryAt: NOW }), B]), { type: "REMOVE", clientMessageId: "x" }, READY, ["b:waiting"]],
      ["remove: sending, confirmed", state(SENDING, [H, ins("x", 1, { inFlight: false, confirmed: true })]), { type: "REMOVE", clientMessageId: "x" }, SENDING, ["h:sending", "x:inserting"]],
      ["remove: held, retry waiting", state(HELD, [ins("x", 1, { inFlight: false, retryAt: NOW })]), { type: "REMOVE", clientMessageId: "x" }, IDLE, []],
      // CLEAR keeps what the server holds or a request is on its way for.
      ["clear: waiting", state(waiting(1), [ins("x", 1, { inFlight: false, confirmed: true }), ins("y", 2, { inFlight: false, retryAt: NOW }), item("c", 3)]), { type: "CLEAR" }, waiting(1), ["x:inserting"]],
      ["clear: ready, emptied", state(READY, [ins("y", 2, { inFlight: false, retryAt: NOW }), item("c", 3)]), { type: "CLEAR" }, IDLE, []],
      ["clear: sending", state(SENDING, [H, ins("x", 1), item("c", 3)]), { type: "CLEAR" }, SENDING, ["h:sending", "x:inserting"]],
      ["clear: held keeps a message being inserted and waits", state(HELD, [ins("x", 1, { confirmed: true }), item("c", 3)], { version: 4, status: "failed" }), { type: "CLEAR" }, waiting(4), ["x:inserting"]],
      // Leaving the sending phase with a failed message stops the queue.
      ["accepted: leaving sending with a failed message holds", state(SENDING, [H, item("f", 1, "failed")]), { type: "DELIVERY_ACCEPTED", clientMessageId: "h", sinceVersion: 2 }, HELD_UNKNOWN, ["f:failed"], "send_now_unknown"],
      ["send failed: leaving sending with a failed message holds", state(SENDING, [H, item("f", 1, "failed")]), { type: "SEND_FAILED", clientMessageId: "h", verdict: "wait" }, HELD_UNKNOWN, ["h:waiting", "f:failed"], "send_now_unknown"],
      // SEND_NOW_REQUESTED
      ["requested: waiting inserts at the front", state(waiting(1), [A, B]), requested("b"), waiting(1), ["b:inserting", "a:waiting"], null],
      ["requested: ready marks it next, no frame", state(READY, [A, B]), requested("b"), READY, ["b:waiting", "a:waiting"], null],
      ["requested: sending goes behind the head being sent", state(SENDING, [H, A, B]), requested("b"), SENDING, ["h:sending", "b:inserting", "a:waiting"], null],
      ["requested: held marks it next and continues", state(HELD, [item("a", 1, "failed"), B]), requested("b"), READY, ["b:waiting", "a:waiting"], null],
      // SEND_NOW_ACCEPTED
      ["accepted now: waiting", state(waiting(1), [ins("x", 1)]), { type: "SEND_NOW_ACCEPTED", clientMessageId: "x" }, waiting(1), ["x:inserting"]],
      ["accepted now: ready", state(READY, [ins("x", 1)]), { type: "SEND_NOW_ACCEPTED", clientMessageId: "x" }, READY, ["x:inserting"]],
      ["accepted now: sending", state(SENDING, [H, ins("x", 1)]), { type: "SEND_NOW_ACCEPTED", clientMessageId: "x" }, SENDING, ["h:sending", "x:inserting"]],
      ["accepted now: held", state(HELD, [ins("x", 1)]), { type: "SEND_NOW_ACCEPTED", clientMessageId: "x" }, HELD, ["x:inserting"]],
      // SEND_NOW_FAILED
      ["failed now: waiting, hold stops the queue", state(waiting(1), [ins("x", 1)]), failedWith("x", "hold"), HELD_SENT, ["x:failed"], "send_failed"],
      ["failed now: ready, unknown stops the queue", state(READY, [ins("x", 1)]), failedWith("x", "unknown"), HELD_UNKNOWN, ["x:failed"], "send_now_unknown"],
      ["failed now: sending, hold only marks it", state(SENDING, [H, ins("x", 1)]), failedWith("x", "hold"), SENDING, ["h:sending", "x:failed"]],
      ["failed now: held, unknown only marks it", state(HELD, [ins("x", 1)]), failedWith("x", "unknown"), HELD, ["x:failed"]],
      // SEND_NOW_RESENT
      ["resent: waiting", state(waiting(1), [ins("x", 1, { inFlight: false, confirmed: true })]), { type: "SEND_NOW_RESENT", clientMessageId: "x", atVersion: 4, connection: "c2" }, waiting(1), ["x:inserting"]],
      ["resent: ready", state(READY, [ins("x", 1, { inFlight: false, confirmed: true })]), { type: "SEND_NOW_RESENT", clientMessageId: "x", atVersion: 4, connection: "c2" }, READY, ["x:inserting"]],
      ["resent: sending", state(SENDING, [H, ins("x", 1, { inFlight: false, retryAt: NOW })]), { type: "SEND_NOW_RESENT", clientMessageId: "x", atVersion: undefined, connection: "c2" }, SENDING, ["h:sending", "x:inserting"]],
      ["resent: held", state(HELD, [ins("x", 1, { inFlight: false, confirmed: true })]), { type: "SEND_NOW_RESENT", clientMessageId: "x", atVersion: 4, connection: "c2" }, HELD, ["x:inserting"]],
      // LIVE_INPUT_APPLIED
      ["applied: waiting keeps waiting when emptied", state(waiting(1), [ins("x", 1, { confirmed: true })]), { type: "LIVE_INPUT_APPLIED", clientMessageId: "x" }, waiting(1), []],
      ["applied: ready is idle when emptied", state(READY, [ins("x", 1, { confirmed: true })]), { type: "LIVE_INPUT_APPLIED", clientMessageId: "x" }, IDLE, []],
      ["applied: sending", state(SENDING, [H, ins("x", 1, { confirmed: true })]), { type: "LIVE_INPUT_APPLIED", clientMessageId: "x" }, SENDING, ["h:sending"]],
      ["applied: held is idle when emptied", state(HELD, [ins("x", 1, { confirmed: true })]), { type: "LIVE_INPUT_APPLIED", clientMessageId: "x" }, IDLE, []],
      // LIVE_INPUT_RETURNED
      ["returned: waiting, run ended", state(waiting(1), [ins("x", 1), B]), returned("x", "run_ended"), waiting(1), ["x:waiting", "b:waiting"]],
      ["returned: ready, taken", state(READY, [ins("x", 1), B]), returned("x", "taken"), READY, ["b:waiting"]],
      ["returned: sending, refused only marks it", state(SENDING, [H, ins("x", 1)]), returned("x", "refused"), SENDING, ["h:sending", "x:failed"]],
      ["returned: held, outcome unknown only marks it", state(HELD, [ins("x", 1)]), returned("x", "outcome_unknown"), HELD, ["x:failed"]],
      // SEND_NOW_SUPPORT_LOST
      ["support lost: waiting stops the queue", state(waiting(1), [ins("x", 1), B]), { type: "SEND_NOW_SUPPORT_LOST" }, HELD_UNKNOWN, ["x:failed", "b:waiting"], "send_now_unknown"],
      ["support lost: ready stops the queue", state(READY, [ins("x", 1), B]), { type: "SEND_NOW_SUPPORT_LOST" }, HELD_UNKNOWN, ["x:failed", "b:waiting"], "send_now_unknown"],
      ["support lost: sending only marks it", state(SENDING, [H, ins("x", 1)]), { type: "SEND_NOW_SUPPORT_LOST" }, SENDING, ["h:sending", "x:failed"]],
      ["support lost: held only marks it", state(HELD, [ins("x", 1)]), { type: "SEND_NOW_SUPPORT_LOST" }, HELD, ["x:failed"]],
    ]
    for (const [name, from, event, phase, expectedIds, effect] of rows) {
      const { next, effect: actual } = apply(from, event)
      expect(next.phase, name).toEqual(phase)
      expect(ids(next), name).toEqual(expectedIds)
      if (effect !== undefined) expect(actual, name).toBe(effect)
    }
    expect(rows).toHaveLength(43)
  })

  it("ignores every new event while idle", () => {
    const idle = state(IDLE)
    const events: MessageQueueEvent[] = [
      requested("x"), { type: "SEND_NOW_ACCEPTED", clientMessageId: "x" }, failedWith("x", "next"),
      { type: "SEND_NOW_RESENT", clientMessageId: "x", atVersion: 1, connection: "c1" },
      { type: "LIVE_INPUT_APPLIED", clientMessageId: "x" }, returned("x", "run_ended"), { type: "SEND_NOW_SUPPORT_LOST" },
    ]
    for (const event of events) expect(apply(idle, event).next).toBe(idle)
  })

  it.each([
    ["failed", { version: 4, status: "failed" as const }],
    ["paused", { version: 4, status: "paused" as const }],
  ])("clearing a held queue that keeps a message being inserted waits from the observed version (%s)", (_name, observed) => {
    const from = state(HELD, [ins("x", 1, { confirmed: true }), A], observed)
    const { next } = apply(from, { type: "CLEAR" })
    expect(next.phase).toEqual(waiting(observed.version))
    expect(ids(next)).toEqual(["x:inserting"])
  })

  describe("every phase column of the send-now cells", () => {
    const COLUMNS = ["waiting", "ready", "sending", "held"] as const
    type Column = (typeof COLUMNS)[number]
    const phaseOf = (column: Column): MessageQueuePhase =>
      ({ waiting: waiting(1), ready: READY, sending: SENDING, held: HELD })[column]
    // The message being inserted at the front, with A behind it; a column that
    // sends has its head in front of both.
    const fromColumn = (column: Column, over: Partial<InsertState> = {}) =>
      state(phaseOf(column), [...(column === "sending" ? [H] : []), ins("x", 3, over), A])
    const lead = (column: Column) => (column === "sending" ? ["h:sending"] : [])
    // Whether a failed insert stops the queue itself in this column.
    const stopsAlone = (column: Column) => column === "waiting" || column === "ready"

    it.each(COLUMNS)("a confirmation in the %s column clears the request and confirms unless a retry waits", (column) => {
      const confirmed = apply(fromColumn(column), { type: "SEND_NOW_ACCEPTED", clientMessageId: "x" }).next
      expect(firstInsert(confirmed, "x")).toMatchObject({ inFlight: false, confirmed: true })
      const waitingRetry = fromColumn(column, { inFlight: true, retryAt: NOW })
      const late = apply(waitingRetry, { type: "SEND_NOW_ACCEPTED", clientMessageId: "x" }).next
      expect(firstInsert(late, "x")).toMatchObject({ inFlight: false, confirmed: false, retryAt: NOW })
    })

    it.each(COLUMNS)("sending again in the %s column starts a request and ends any wait", (column) => {
      const from = fromColumn(column, { inFlight: false, confirmed: true, retryAt: NOW, connection: "c1", resentAt: undefined })
      const { next } = apply(from, { type: "SEND_NOW_RESENT", clientMessageId: "x", atVersion: 4, connection: "c2" })
      expect(firstInsert(next, "x")).toMatchObject({
        inFlight: true, confirmed: false, retryAt: undefined, connection: "c2", resentAt: 4, retries: 0, since: 1,
      })
      expect(next.phase).toEqual(from.phase)
    })

    it.each(["sending", "waiting"] as const)("a request that cannot be taken in is ignored in the %s column", (column) => {
      const from = state(phaseOf(column), [...(column === "sending" ? [H] : []), A, B])
      expect(apply(from, { ...requested("b"), canInsert: false } as MessageQueueEvent).next).toBe(from)
    })

    describe.each(COLUMNS)("the %s column", (column) => {
      it.each([
        ["next", ["x:waiting:next", "a:waiting"]],
        ["retry", ["a:waiting", "x:waiting"]],
        ["later", ["x:inserting", "a:waiting"]],
        ["hold", ["x2:failed", "a:waiting"]],
        ["unknown", ["x:failed", "a:waiting"]],
      ] as const)("a failed send ending as %s", (verdict, expected) => {
        const from = fromColumn(column)
        const result = apply(from, failedWith("x", verdict, verdict === "hold" ? { newClientMessageId: "x2" } : {}))
        expect([...lead(column), ...expected]).toEqual(
          result.next.items.map(entry => `${entry.clientMessageId}:${entry.status}${entry.next ? ":next" : ""}`),
        )
        const stops = verdict === "hold" || verdict === "unknown"
        if (stops && stopsAlone(column)) {
          expect(result.next.phase).toEqual(verdict === "hold" ? HELD_SENT : HELD_UNKNOWN)
          expect(result.effect).toBe(verdict === "hold" ? "send_failed" : "send_now_unknown")
        } else {
          expect(result.next.phase).toEqual(from.phase)
          expect(result.effect).toBeNull()
        }
        if (verdict === "later") expect(firstInsert(result.next, "x")).toMatchObject({ retries: 1, retryAt: NOW + 1000, inFlight: false })
      })

      it.each([
        ["run_ended", ["x:waiting:next", "a:waiting"], null],
        ["not_applied", ["x2:waiting:next", "a:waiting"], null],
        ["not_started", ["x:inserting", "a:waiting"], null],
        ["taken", ["a:waiting"], null],
        ["unavailable", ["x:failed", "a:waiting"], "send_failed"],
        ["refused", ["x:failed", "a:waiting"], "send_failed"],
        ["conflict", ["x:failed", "a:waiting"], "send_failed"],
        ["outcome_unknown", ["x:failed", "a:waiting"], "send_now_unknown"],
      ] as const)("a returned message with reason %s", (reason, expected, effect) => {
        const from = fromColumn(column)
        const result = apply(from, returned("x", reason, reason === "not_applied" ? { newClientMessageId: "x2" } : {}))
        const shown = result.next.items.map(entry => `${entry.clientMessageId}:${entry.status}${entry.next ? ":next" : ""}`)
        expect(shown).toEqual([...lead(column), ...expected])
        expect(result.effect).toBe(stopsAlone(column) ? effect : null)
        if (reason === "not_started") expect(firstInsert(result.next, "x")).toMatchObject({ inFlight: true, retries: 1, retryAt: NOW + 1000 })
      })
    })
  })

  it("ignores an applied echo for a message that is not being inserted", () => {
    const from = state(waiting(1), [A])
    expect(apply(from, { type: "LIVE_INPUT_APPLIED", clientMessageId: "a" }).next).toBe(from)
  })

  it("confirms a request and clears the request in flight", () => {
    const from = state(waiting(1), [ins("x", 1)])
    const accepted = apply(from, { type: "SEND_NOW_ACCEPTED", clientMessageId: "x" }).next
    expect(firstInsert(accepted, "x")).toMatchObject({ inFlight: false, confirmed: true })
  })

  it.each([
    ["next", "waiting", true],
    ["retry", "waiting", false],
    ["later", "inserting", false],
    ["hold", "failed", false],
    ["unknown", "failed", false],
  ] as const)("a failed send into the turn ends as %s", (verdict, status, next) => {
    const from = state(waiting(1), [ins("x", 2), A])
    const result = apply(from, failedWith("x", verdict, verdict === "hold" ? { newClientMessageId: "x2" } : {})).next
    const moved = result.items.find(entry => entry.text === "x")
    expect(moved?.status).toBe(status)
    expect(moved?.next === true).toBe(next)
    expect(moved?.clientMessageId).toBe(verdict === "hold" ? "x2" : "x")
    if (verdict === "retry") expect(ids(result)).toEqual(["a:waiting", "x:waiting"])
    if (verdict === "later") expect(moved?.insert).toMatchObject({ retries: 1, retryAt: NOW + 1000, confirmed: false, inFlight: false })
    else expect(moved?.insert).toBeUndefined()
  })

  it.each([
    ["run_ended", "waiting", true, undefined],
    ["not_applied", "waiting", true, "fresh"],
    ["not_started", "inserting", false, undefined],
    ["taken", undefined, false, undefined],
    ["unavailable", "failed", false, undefined],
    ["refused", "failed", false, undefined],
    ["conflict", "failed", false, undefined],
    ["outcome_unknown", "failed", false, undefined],
  ] as const)("a returned message with reason %s", (reason, status, next, newId) => {
    const from = state(waiting(1), [ins("x", 2), A])
    const result = apply(from, returned("x", reason, newId ? { newClientMessageId: newId } : {})).next
    const moved = result.items.find(entry => entry.text === "x")
    expect(moved?.status).toBe(status)
    expect(moved?.next === true).toBe(next)
    if (moved) {
      expect(moved.clientMessageId).toBe(newId ?? "x")
      expect(moved.seq).toBe(2)
    }
    if (reason === "not_started") expect(moved?.insert).toMatchObject({ retries: 1, retryAt: NOW + 1000, inFlight: true })
  })

  it("keeps a request in flight when the turn reports it has not started", () => {
    const next = apply(state(waiting(1), [ins("x", 1)]), returned("x", "not_started")).next
    expect(firstInsert(next, "x")).toMatchObject({ inFlight: true, confirmed: false, retries: 1, retryAt: NOW + 1000 })
  })

  it("gives up retrying after the sixth refusal and sends the message first on the next turn", () => {
    let current = state(waiting(1), [ins("x", 1), A])
    for (let attempt = 1; attempt <= 5; attempt += 1) {
      current = apply(current, failedWith("x", "later")).next
      expect(firstInsert(current, "x")).toMatchObject({ retries: attempt, inFlight: false })
      current = apply(current, { type: "SEND_NOW_RESENT", clientMessageId: "x", atVersion: undefined, connection: "c1" }).next
      expect(firstInsert(current, "x")).toMatchObject({ inFlight: true, retryAt: undefined })
    }
    current = apply(current, failedWith("x", "later")).next
    expect(ids(current)).toEqual(["x:waiting", "a:waiting"])
    expect(current.items[0].next).toBe(true)
    expect(current.items[0].insert).toBeUndefined()
  })

  it("late ack after not started keeps the retry wait", () => {
    let current = state(waiting(1), [ins("x", 1)])
    current = apply(current, returned("x", "not_started")).next
    current = apply(current, { type: "SEND_NOW_ACCEPTED", clientMessageId: "x" }).next
    expect(firstInsert(current, "x")).toMatchObject({ confirmed: false, inFlight: false, retryAt: NOW + 1000 })
    // The visitor can still take the message back.
    expect(ids(apply(current, { type: "REMOVE", clientMessageId: "x" }).next)).toEqual([])
  })

  it("lets a message be removed only while no request is on its way and nothing is confirmed", () => {
    const base = (over: Partial<InsertState>) => state(waiting(1), [ins("x", 1, over)])
    const remove = (from: MessageQueueState) => ids(apply(from, { type: "REMOVE", clientMessageId: "x" }).next)
    expect(remove(base({}))).toEqual(["x:inserting"])
    expect(remove(base({ inFlight: false, confirmed: true }))).toEqual(["x:inserting"])
    expect(remove(base({ inFlight: false, retryAt: NOW }))).toEqual([])
  })

  it.each([
    ["run_ended", "waiting"], ["run_ended", "ready"], ["run_ended", "held"],
    ["not_applied", "waiting"], ["not_applied", "ready"], ["not_applied", "held"],
  ] as const)("a returned message goes first [%s %s]", (reason, phase) => {
    const phases = { waiting: waiting(1), ready: READY, held: HELD } as const
    const from = state(phases[phase], [ins("x", 3), item("a", 1), item("b", 2)])
    const { next } = apply(from, returned("x", reason, reason === "not_applied" ? { newClientMessageId: "x2" } : {}))
    const head = next.items[0]
    expect(head).toMatchObject({ status: "waiting", next: true, seq: 3 })
    expect(head.clientMessageId).toBe(reason === "not_applied" ? "x2" : "x")
    expect(next.items.slice(1).map(entry => entry.clientMessageId)).toEqual(["a", "b"])
  })
})

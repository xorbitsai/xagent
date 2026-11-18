import { describe, expect, it } from "vitest"

import { MessageDeliveryError } from "@/hooks/use-websocket"
import type { ClientErrorCode } from "@/lib/client-errors"
import type { TaskStatus } from "@/lib/task-status"
import {
  MESSAGE_QUEUE_LIMIT,
  classifyQueuedSendFailure,
  decideComposerSubmit,
  isAwaitingAnswer,
  isBusyRefusal,
  transitionMessageQueue,
  type MessageQueueEvent,
  type MessageQueuePhase,
  type MessageQueueState,
  type QueuedMessage,
} from "./session-message-queue"

const item = (id: string, seq: number, status: QueuedMessage["status"] = "waiting"): QueuedMessage =>
  ({ clientMessageId: id, text: id, seq, status })
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

import React from "react"
import { readFileSync } from "node:fs"
import path from "node:path"
import { act, cleanup, render } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

const authUserRef = { current: { id: "u1" } as { id: string } | null }
vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ user: authUserRef.current }),
}))

import {
  ConnectorRuntimeDialogProvider,
  useConnectorRuntimeDialog,
  useConnectorRuntimeDialogActions,
  useConnectorRuntimeDialogActionsIfMounted,
  transitionRequest,
  gatesToSettle,
  type ConnectorRuntimeDialogActions,
  type ConnectorRuntimeDialogCloseOutcome,
  type ConnectorRuntimeDialogRequest,
  type ConnectorRuntimeDialogState,
  type ConnectorRuntimeDialogValue,
  type ConnectorRuntimeResendPayload,
  type FirstGateDecision,
  type HeldGate,
  type RequestInput,
  type SessionCheckCause,
} from "./connector-runtime-dialog-context"

afterEach(() => {
  cleanup()
  authUserRef.current = { id: "u1" }
})

// Lets committed effects and the promise callbacks they resolve run.
const flush = () => act(async () => {})

describe("useConnectorRuntimeDialogActionsIfMounted", () => {
  it("does not warn when called with no provider above it", () => {
    // A fresh module instance for this test file (vitest isolates modules
    // per file by default): warnCalledOutsideProvider's own "already warned
    // about this action" set starts empty here, so this is a real, not
    // vacuous, check that no warning fires -- unlike calling the same
    // action name from within the large app-context-chat.test.tsx suite,
    // where an unrelated earlier test can already have exhausted that
    // action name's one-time warning.
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {})
    let actions: ConnectorRuntimeDialogActions | undefined
    function Probe() {
      actions = useConnectorRuntimeDialogActionsIfMounted()
      return null
    }
    render(<Probe />)
    act(() => {
      actions?.openForTask(1)
      actions?.close("dismissed", 1)
      actions?.recordDelivery({ taskId: 1, clientMessageId: "x", text: "hi" })
      actions?.retainOnlyTask(1)
      actions?.forgetDelivery(1)
    })
    expect(warnSpy).not.toHaveBeenCalled()
    warnSpy.mockRestore()
  })

  it("still warns through the plain hook with no provider above it (the wiring-mistake case this dev warning exists for)", () => {
    // Same fresh-module guarantee as above, in the other direction: proves
    // the silent behavior above is specific to the "if mounted" entry point
    // and not a change to warnCalledOutsideProvider itself, which must keep
    // catching an actual wiring mistake (a consumer mounted as a sibling of
    // the provider instead of inside it).
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {})
    let actions: ConnectorRuntimeDialogActions | undefined
    function Probe() {
      actions = useConnectorRuntimeDialogActions()
      return null
    }
    render(<Probe />)
    act(() => { actions?.openForTask(1) })
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining("called outside ConnectorRuntimeDialogProvider"),
    )
    warnSpy.mockRestore()
  })

  it("returns the same actions a real provider hands to the plain hook", () => {
    // Inside a real provider there is only one actions object (the
    // provider's own useMemo), so this must read as that same object, not a
    // second one -- otherwise a consumer holding onto this hook's result in
    // a ref (as app-context-chat.tsx does) would diverge from one reading
    // useConnectorRuntimeDialogActions() directly.
    let ifMounted: ConnectorRuntimeDialogActions | undefined
    let plain: ConnectorRuntimeDialogActions | undefined
    function Probe() {
      ifMounted = useConnectorRuntimeDialogActionsIfMounted()
      plain = useConnectorRuntimeDialogActions()
      return null
    }
    render(
      <ConnectorRuntimeDialogProvider>
        <Probe />
      </ConnectorRuntimeDialogProvider>,
    )
    expect(ifMounted).toBe(plain)
  })
})

describe("pending candidate state machine", () => {
  let latestActions: ConnectorRuntimeDialogActions
  let latestState: ConnectorRuntimeDialogValue

  function Probe() {
    latestActions = useConnectorRuntimeDialog()
    latestState = useConnectorRuntimeDialog()
    return null
  }

  function renderProbe() {
    return render(
      <ConnectorRuntimeDialogProvider>
        <Probe />
      </ConnectorRuntimeDialogProvider>,
    )
  }

  it("promotes a staged candidate to the stash once its delivery is recorded (transition)", () => {
    renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-1", text: "hi" })
    })
    expect(latestState.payload).toBeNull()
    act(() => {
      latestActions.recordDelivery({ taskId: 1, clientMessageId: "cm-1", text: "hi" })
    })
    expect(latestState.payload).toEqual({ taskId: 1, clientMessageId: "cm-1", text: "hi", files: [] })
  })

  it("withdraws a staged candidate on discard, so its late delivery is not recorded (cancellation)", () => {
    renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-1", text: "hi" })
    })
    act(() => {
      latestActions.discardPendingDelivery("cm-1")
    })
    // The send that staged this candidate threw or returned early; its
    // delivery acknowledgement arriving late must not resurrect the stash --
    // recordDelivery only redeems a ticket that is still outstanding.
    act(() => {
      latestActions.recordDelivery({ taskId: 1, clientMessageId: "cm-1", text: "hi" })
    })
    expect(latestState.payload).toBeNull()
  })

  it("drops a staged candidate when its task settles, so a late delivery is not recorded (settlement discard)", () => {
    renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-1", text: "hi" })
    })
    // A non-triggering settlement frame (task_completed, or a task_error not
    // in the trigger set) reaches forgetDelivery, not openForTask.
    act(() => {
      latestActions.forgetDelivery(1)
    })
    act(() => {
      latestActions.recordDelivery({ taskId: 1, clientMessageId: "cm-1", text: "hi" })
    })
    expect(latestState.payload).toBeNull()
  })

  it("does not claim either of two in-flight candidates for the same task (T5)", () => {
    renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-1", text: "first" })
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-2", text: "second" })
    })
    act(() => {
      latestActions.openForTask(1)
    })
    expect(latestState.request?.resendPayload).toBeNull()
  })

  it("replaces a repeat stage under the same clientMessageId in place", () => {
    // A retry that reuses the caller-supplied id stages a second time under
    // that same id. Appending instead of replacing would leave two entries
    // for one turn, which openForTask reads as two in-flight turns and
    // withholds the whole resend chain for -- so the retry-id reuse this
    // dialog depends on would cost the user the resend button.
    renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-1", text: "first" })
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-1", text: "second" })
    })
    act(() => {
      latestActions.openForTask(1)
    })
    expect(latestState.request?.resendPayload).toEqual({
      taskId: 1, clientMessageId: "cm-1", text: "second", files: [],
    })
  })

  it("does not fall through to the confirmed stash on ambiguity (Major A)", () => {
    renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-a", text: "AAA" })
    })
    act(() => {
      latestActions.recordDelivery({ taskId: 1, clientMessageId: "cm-a", text: "AAA" })
    })
    // A second and third turn are staged after "AAA" was already delivered
    // and confirmed -- the ambiguous pair must not make openForTask fall
    // through the rest of the chain and hand out that older, already-sent
    // turn instead of degrading to save-only, same as T5 above (which has
    // no confirmed stash to fall through to in the first place).
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-b", text: "BBB" })
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-c", text: "CCC" })
    })
    act(() => {
      latestActions.openForTask(1)
    })
    expect(latestState.request?.resendPayload).toBeNull()
  })

  it("keeps an already-open dialog's own snapshot through a later ambiguous frame", () => {
    renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-a", text: "AAA" })
    })
    act(() => {
      latestActions.recordDelivery({ taskId: 1, clientMessageId: "cm-a", text: "AAA" })
    })
    act(() => {
      latestActions.openForTask(1)
    })
    expect(latestState.request?.resendPayload).toEqual({
      taskId: 1, clientMessageId: "cm-a", text: "AAA", files: [],
    })
    // A newer turn is sent and confirmed while this dialog is still open...
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-x", text: "XXX" })
    })
    act(() => {
      latestActions.recordDelivery({ taskId: 1, clientMessageId: "cm-x", text: "XXX" })
    })
    // ...and then two more turns are staged before a second terminal frame
    // retargets this same open dialog.
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-b", text: "BBB" })
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-c", text: "CCC" })
    })
    act(() => {
      latestActions.openForTask(1)
    })
    // Ambiguity discards the confirmed stash ("XXX") same as always, but
    // must not make the button the user is already looking at disappear or
    // switch to a different turn: it keeps carrying "AAA".
    expect(latestState.request?.resendPayload).toEqual({
      taskId: 1, clientMessageId: "cm-a", text: "AAA", files: [],
    })
  })

  it("hands the confirmed stash to a plain single-turn resend with nothing staged", () => {
    renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-a", text: "AAA" })
    })
    act(() => {
      latestActions.recordDelivery({ taskId: 1, clientMessageId: "cm-a", text: "AAA" })
    })
    act(() => {
      latestActions.openForTask(1)
    })
    expect(latestState.request?.resendPayload).toEqual({
      taskId: 1, clientMessageId: "cm-a", text: "AAA", files: [],
    })
  })

  it("prefers a staged candidate over an already-confirmed stash (T6)", () => {
    renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-old", text: "old" })
    })
    act(() => {
      latestActions.recordDelivery({ taskId: 1, clientMessageId: "cm-old", text: "old" })
    })
    // A second turn sent afterward is staged but not yet acknowledged when
    // the triggering terminal frame arrives.
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-new", text: "new" })
    })
    act(() => {
      latestActions.openForTask(1)
    })
    expect(latestState.request?.resendPayload).toEqual({
      taskId: 1, clientMessageId: "cm-new", text: "new", files: [],
    })
  })

  it("never claims a candidate staged for a different task (T7)", () => {
    renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-1", text: "hi" })
    })
    act(() => {
      latestActions.openForTask(2)
    })
    expect(latestState.request?.resendPayload).toBeNull()
    // Task 1's own candidate is untouched by task 2's settlement.
    act(() => {
      latestActions.openForTask(1)
    })
    expect(latestState.request?.resendPayload).toEqual({
      taskId: 1, clientMessageId: "cm-1", text: "hi", files: [],
    })
  })

  it("clears a pending candidate when the task it belongs to is switched away from (T8)", () => {
    renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-1", text: "hi" })
    })
    act(() => {
      latestActions.retainOnlyTask(2)
    })
    act(() => {
      latestActions.openForTask(1)
    })
    expect(latestState.request?.resendPayload).toBeNull()
  })

  it("clears a pending candidate when the signed-in identity changes (T8)", () => {
    const { rerender } = renderProbe()
    act(() => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "cm-1", text: "hi" })
    })
    authUserRef.current = { id: "u2" }
    rerender(
      <ConnectorRuntimeDialogProvider>
        <Probe />
      </ConnectorRuntimeDialogProvider>,
    )
    act(() => {
      latestActions.openForTask(1)
    })
    expect(latestState.request?.resendPayload).toBeNull()
  })
})

// Shared by the two transitionRequest tables below.
const delivery = (taskId: number, clientMessageId: string) => ({
  taskId, clientMessageId, text: clientMessageId, files: [] as File[],
})
const stash1 = delivery(1, "stash-1")
const stash2 = delivery(2, "stash-2")
const staged1 = delivery(1, "staged-1")
const staged1b = delivery(1, "staged-1b")
const staged2 = delivery(2, "staged-2")
const keptSnap = delivery(1, "kept-1")
const request = (
  taskId: number,
  seq: number,
  resendPayload: ConnectorRuntimeResendPayload | null = null,
  trigger: ConnectorRuntimeDialogRequest["trigger"] = "turn_failure",
): ConnectorRuntimeDialogRequest => ({ taskId, seq, resendPayload, trigger, gateId: null })
const state = (overrides: Partial<ConnectorRuntimeDialogState> = {}): ConnectorRuntimeDialogState => (
  { seq: 4, request: null, payload: null, pending: [], dismissedCheck: null, gates: [], ...overrides }
)
const open = (taskId: number): RequestInput => ({ type: "open", taskId, trigger: "turn_failure" })
const closeAs = (outcome: ConnectorRuntimeDialogCloseOutcome, taskId = 1): RequestInput => (
  { type: "close", taskId, outcome }
)
const check = (taskId: number, cause: SessionCheckCause): RequestInput => (
  { type: "open", taskId, trigger: "session_open", cause }
)
const sessionCheck = (taskId: number, seq = 4) => request(taskId, seq, null, "session_open")


// Every request transition the provider's actions made before they were
// moved into transitionRequest, pinned as data: each row is one previous
// state, one input, and the state that must come out. A row whose expected
// state is `"same"` must return the very object it was given -- the
// provider relies on that to skip a re-render when nothing changed.
describe("transitionRequest keeps today's transitions", () => {
  const rows: Array<[string, ConnectorRuntimeDialogState, RequestInput, ConnectorRuntimeDialogState | "same"]> = [
    ["open with nothing to claim", state(), open(1),
      state({ seq: 5, request: request(1, 5) })],
    ["open for a task whose request already carries a snapshot keeps it", state({ request: request(1, 4, keptSnap) }), open(1),
      state({ seq: 5, request: request(1, 5, keptSnap) })],
    ["open claims exactly one staged turn over the stash, and clears both", state({ payload: stash1, pending: [staged1, staged2] }), open(1),
      state({ seq: 5, request: request(1, 5, staged1), payload: null, pending: [staged2] })],
    ["open claims the stash when nothing is staged", state({ payload: stash1 }), open(1),
      state({ seq: 5, request: request(1, 5, stash1), payload: null })],
    ["open leaves another task's stash alone", state({ payload: stash2 }), open(1),
      state({ seq: 5, request: request(1, 5), payload: stash2 })],
    ["open claims nothing when two turns are staged", state({ payload: stash1, pending: [staged1, staged1b] }), open(1),
      state({ seq: 5, request: request(1, 5), payload: null, pending: [] })],
    ["open keeps an open request's snapshot when two turns are staged", state({ request: request(1, 4, keptSnap), pending: [staged1, staged1b] }), open(1),
      state({ seq: 5, request: request(1, 5, keptSnap), pending: [] })],
    ["open for another task replaces the request and drops its snapshot", state({ request: request(2, 4, stash2) }), open(1),
      state({ seq: 5, request: request(1, 5) })],
    ["close with no request", state({ payload: stash1 }), closeAs("dismissed"), "same"],
    ["close dismissed drops the stash (and sets the dismissed-check mark, see below)", state({ request: request(1, 4), payload: stash1 }), closeAs("dismissed"),
      state({ payload: null, dismissedCheck: 1 })],
    ["close resent keeps the stash", state({ request: request(1, 4), payload: stash1 }), closeAs("resent"),
      state({ payload: stash1 })],
    ["close not-shown keeps the stash", state({ request: request(1, 4), payload: stash1 }), closeAs("not-shown"),
      state({ payload: stash1 })],
    ["close left-host keeps the stash", state({ request: request(1, 4), payload: stash1 }), closeAs("left-host"),
      state({ payload: stash1 })],
    ["retain the request's own task", state({ request: request(1, 4), payload: stash2, pending: [staged1, staged2] }), { type: "retain", taskId: 1 },
      state({ request: request(1, 4), payload: null, pending: [staged1] })],
    ["retain another task", state({ request: request(1, 4), payload: stash1, pending: [staged1] }), { type: "retain", taskId: 2 },
      state()],
    ["retain no task", state({ request: request(1, 4), payload: stash1, pending: [staged2] }), { type: "retain", taskId: null },
      state()],
    ["retain with nothing to drop", state({ request: request(1, 4), payload: stash1, pending: [staged1] }), { type: "retain", taskId: 1 }, "same"],
    ["forget takes the snapshot off an open request, seq unchanged", state({ request: request(1, 4, keptSnap), payload: stash1, pending: [staged1, staged2] }), { type: "forget", taskId: 1 },
      state({ request: request(1, 4), payload: null, pending: [staged2] })],
    ["forget leaves a request with no snapshot as is", state({ request: request(1, 4), payload: stash1 }), { type: "forget", taskId: 1 },
      state({ request: request(1, 4), payload: null })],
    ["forget with nothing for that task", state({ request: request(2, 4, stash2), payload: stash2, pending: [staged2] }), { type: "forget", taskId: 1 }, "same"],
    ["identity change with nothing held", state(), { type: "identity-changed" }, "same"],
    ["identity change drops the request, the stash and every staged turn", state({ request: request(1, 4, keptSnap), payload: stash2, pending: [staged1] }), { type: "identity-changed" },
      state()],
  ]

  it.each(rows)("%s", (_name, prev, input, expected) => {
    const next = transitionRequest(prev, input)
    if (expected === "same") expect(next).toBe(prev)
    else expect(next).toEqual(expected)
  })

  it("keeps an unchanged request object when retaining its own task", () => {
    const prev = state({ request: request(1, 4, keptSnap), payload: stash2 })
    expect(transitionRequest(prev, { type: "retain", taskId: 1 }).request).toBe(prev.request)
  })
})

// Session checks, the per-task close and the dismissed-check mark, as data.
// Columns follow the current request: none, a turn failure (TF) or a
// session check (SO); the first-gate column does not exist yet.
describe("transitionRequest opens and closes requests by trigger", () => {
  const tf = (taskId: number, snap: ConnectorRuntimeResendPayload | null = null) => request(taskId, 4, snap)
  const rows: Array<[string, ConnectorRuntimeDialogState, RequestInput, ConnectorRuntimeDialogState | "same"]> = [
    // B0: a close from another task's dialog leaves every request alone.
    ["B0 none: close from task 2", state({ payload: stash1 }), closeAs("dismissed", 2), "same"],
    ["B0 TF: close from task 2 keeps task 1's turn failure", state({ request: tf(1, keptSnap), payload: stash1 }), closeAs("dismissed", 2), "same"],
    ["B0 SO: close from task 2 keeps task 1's session check", state({ request: sessionCheck(1) }), closeAs("not-shown", 2), "same"],
    // B1: a turn failure for the same task upgrades a session check.
    ["B1a SO: upgrade claims the stash", state({ request: sessionCheck(1), payload: stash1 }), open(1),
      state({ seq: 5, request: request(1, 5, stash1), payload: null })],
    ["B1a SO: upgrade claims the one staged turn", state({ request: sessionCheck(1), payload: stash1, pending: [staged1] }), open(1),
      state({ seq: 5, request: request(1, 5, staged1), payload: null, pending: [] })],
    ["B1b SO: another task's turn failure replaces it", state({ request: sessionCheck(2) }), open(1),
      state({ seq: 5, request: request(1, 5) })],
    // B2a: nothing new for a task that already has a request.
    ["B2a TF: opened", state({ request: tf(1, keptSnap) }), check(1, "opened"), "same"],
    ["B2a TF: reconnected", state({ request: tf(1, keptSnap) }), check(1, "reconnected"), "same"],
    ["B2a SO: opened", state({ request: sessionCheck(1) }), check(1, "opened"), "same"],
    // B2b: a session check touches neither the stash nor the tickets.
    ["B2b none: opened", state({ payload: stash2, pending: [staged2] }), check(1, "opened"),
      state({ seq: 5, request: request(1, 5, null, "session_open"), payload: stash2, pending: [staged2] })],
    ["B2b TF: replaces another task's turn failure", state({ request: tf(2, stash2), payload: stash2 }), check(1, "opened"),
      state({ seq: 5, request: request(1, 5, null, "session_open"), payload: stash2 })],
    ["B2b SO: replaces another task's session check", state({ request: sessionCheck(2) }), check(1, "reconnected"),
      state({ seq: 5, request: request(1, 5, null, "session_open") })],
    ["B2b none: a reconnect does not yield to this task's stash", state({ payload: stash1 }), check(1, "reconnected"),
      state({ seq: 5, request: request(1, 5, null, "session_open"), payload: stash1 })],
    // B2c: a ticket for this task, for either cause.
    ["B2c none: opened yields to a ticket", state({ pending: [staged1] }), check(1, "opened"), "same"],
    ["B2c none: reconnected yields to a ticket", state({ pending: [staged1] }), check(1, "reconnected"), "same"],
    ["B2c TF: yields without replacing another task's request", state({ request: tf(2), pending: [staged1] }), check(1, "opened"), "same"],
    ["B2c SO: yields without replacing another task's request", state({ request: sessionCheck(2), pending: [staged1] }), check(1, "reconnected"), "same"],
    // B2d: a fresh view yields to this task's stash.
    ["B2d none: opened yields to the stash", state({ payload: stash1 }), check(1, "opened"), "same"],
    ["B2d SO: opened yields to the stash", state({ request: sessionCheck(2), payload: stash1 }), check(1, "opened"), "same"],
    // B2e: a reconnect stays quiet for a check the user closed this view.
    ["B2e none: reconnected after the user closed it", state({ dismissedCheck: 1 }), check(1, "reconnected"), "same"],
    ["B2e TF: reconnected after the user closed it", state({ request: tf(2), dismissedCheck: 1 }), check(1, "reconnected"), "same"],
    ["B2e none: another task's mark does not keep a reconnect quiet", state({ dismissedCheck: 2 }), check(1, "reconnected"),
      state({ seq: 5, request: request(1, 5, null, "session_open"), dismissedCheck: 2 })],
    // B4-B7: closing a session check never drops the stash.
    ["B4 SO: dismissed keeps the stash and marks the task", state({ request: sessionCheck(1), payload: stash1 }), closeAs("dismissed"),
      state({ payload: stash1, dismissedCheck: 1 })],
    ["B5 SO: resent keeps the stash", state({ request: sessionCheck(1), payload: stash1 }), closeAs("resent"),
      state({ payload: stash1 })],
    ["B6 SO: not-shown keeps the stash, no mark", state({ request: sessionCheck(1), payload: stash1 }), closeAs("not-shown"),
      state({ payload: stash1 })],
    ["B7 SO: left-host keeps the stash, no mark", state({ request: sessionCheck(1), payload: stash1 }), closeAs("left-host"),
      state({ payload: stash1 })],
    ["B4 TF: dismissed drops the stash and marks the task", state({ request: tf(1), payload: stash1 }), closeAs("dismissed"),
      state({ dismissedCheck: 1 })],
    // B8-B10 for a session check.
    ["B8a SO: retain its own task", state({ request: sessionCheck(1) }), { type: "retain", taskId: 1 }, "same"],
    ["B8b SO: retain another task", state({ request: sessionCheck(1) }), { type: "retain", taskId: 2 }, state()],
    ["B9a SO: forget keeps the request object", state({ request: sessionCheck(1), payload: stash1 }), { type: "forget", taskId: 1 },
      state({ request: sessionCheck(1) })],
    ["B9b SO: forget another task", state({ request: sessionCheck(1) }), { type: "forget", taskId: 2 }, "same"],
    ["B10 SO: identity change", state({ request: sessionCheck(1) }), { type: "identity-changed" }, state()],
    // The dismissed-check mark: what clears it.
    ["mark: a fresh view clears it and checks again", state({ dismissedCheck: 1 }), check(1, "opened"),
      state({ seq: 5, request: request(1, 5, null, "session_open") })],
    ["mark: a fresh view of another task clears it", state({ dismissedCheck: 1, pending: [staged2] }), check(2, "opened"),
      state({ pending: [staged2] })],
    ["mark: a turn failure for that task clears it", state({ dismissedCheck: 1 }), open(1),
      state({ seq: 5, request: request(1, 5) })],
    ["mark: a turn failure for another task keeps it", state({ dismissedCheck: 1 }), open(2),
      state({ seq: 5, request: request(2, 5), dismissedCheck: 1 })],
    ["mark: switching away clears it", state({ dismissedCheck: 1 }), { type: "retain", taskId: 2 }, state()],
    ["mark: staying on that task keeps it", state({ dismissedCheck: 1 }), { type: "retain", taskId: 1 }, "same"],
    ["mark: an identity change clears it even with nothing else held", state({ dismissedCheck: 1 }), { type: "identity-changed" }, state()],
  ]

  it.each(rows)("%s", (_name, prev, input, expected) => {
    const next = transitionRequest(prev, input)
    if (expected === "same") expect(next).toBe(prev)
    else expect(next).toEqual(expected)
  })

  it("stays quiet on a reconnect only until the task is viewed afresh", () => {
    const steps: RequestInput[] = [check(1, "opened"), closeAs("dismissed"), check(1, "reconnected")]
    const afterReconnect = steps.reduce(transitionRequest, state())
    expect(afterReconnect.request).toBeNull()
    const reopened = transitionRequest(afterReconnect, check(1, "opened"))
    expect(reopened.request).toEqual(request(1, 6, null, "session_open"))
    const closedUnseen = transitionRequest(reopened, closeAs("not-shown"))
    expect(transitionRequest(closedUnseen, check(1, "reconnected")).request).toEqual(request(1, 7, null, "session_open"))
  })

  it("stays quiet on a reconnect after the user closes a turn-failure dialog too", () => {
    const closed = [open(1), closeAs("dismissed")].reduce(transitionRequest, state())
    expect(transitionRequest(closed, check(1, "reconnected"))).toBe(closed)
    const upgraded = [check(1, "opened"), open(1), closeAs("dismissed")].reduce(transitionRequest, state())
    expect(transitionRequest(upgraded, check(1, "reconnected"))).toBe(upgraded)
    const resent = [open(1), closeAs("resent")].reduce(transitionRequest, state())
    expect(transitionRequest(resent, check(1, "reconnected")).request).toEqual(request(1, 6, null, "session_open"))
  })

  it("keeps a session check when another task's dialog closes in the same batch", () => {
    let actions: ConnectorRuntimeDialogActions | undefined
    let value: ConnectorRuntimeDialogValue | undefined
    function Probe() {
      actions = useConnectorRuntimeDialogActions()
      value = useConnectorRuntimeDialog()
      return null
    }
    render(<ConnectorRuntimeDialogProvider><Probe /></ConnectorRuntimeDialogProvider>)
    act(() => { actions?.openSessionCheck(1, "opened") })
    act(() => {
      actions?.openSessionCheck(2, "opened")
      actions?.close("not-shown", 1)
    })
    expect(value?.request).toMatchObject({ taskId: 2, trigger: "session_open" })
    act(() => {
      actions?.openSessionCheck(0, "opened")
      actions?.openSessionCheck(-3, "reconnected")
      actions?.openSessionCheck(1.5, "opened")
    })
    expect(value?.request).toMatchObject({ taskId: 2 })
  })

  it("gives openSessionCheck the same two no-provider defaults as every other action", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {})
    let silent: ConnectorRuntimeDialogActions | undefined
    let plain: ConnectorRuntimeDialogActions | undefined
    function Probe() {
      silent = useConnectorRuntimeDialogActionsIfMounted()
      plain = useConnectorRuntimeDialogActions()
      return null
    }
    render(<Probe />)
    silent?.openSessionCheck(1, "opened")
    expect(warnSpy).not.toHaveBeenCalled()
    plain?.openSessionCheck(1, "opened")
    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("openSessionCheck() was called outside"))
    warnSpy.mockRestore()
  })
})

// First gates, as data: the first-gate (FG) column of every transition, and
// opening one with no request. A held first message's account must move in
// the very step that moves its request, whichever input takes it away.
describe("transitionRequest settles a first gate exactly once, whoever ends it", () => {
  const fg = (taskId: number, gateId: number, seq = 4): ConnectorRuntimeDialogRequest => (
    { taskId, seq, resendPayload: null, trigger: "first_gate", gateId }
  )
  const held = (id: number): HeldGate => ({ id, decision: null })
  const ended = (id: number, decision: FirstGateDecision): HeldGate => ({ id, decision })
  const gate = (taskId: number, gateId: number): RequestInput => ({ type: "open", taskId, trigger: "first_gate", gateId })
  const holding = (overrides: Partial<ConnectorRuntimeDialogState> = {}) => (
    state({ request: fg(1, 7), gates: [held(7)], ...overrides })
  )
  const rows: Array<[string, ConnectorRuntimeDialogState, RequestInput, ConnectorRuntimeDialogState | "same"]> = [
    ["B3 none: opens and holds, touching neither the stash nor the tickets", state({ payload: stash1, pending: [staged1] }), gate(1, 7),
      state({ seq: 5, request: fg(1, 7, 5), payload: stash1, pending: [staged1], gates: [held(7)] })],
    ["B0 FG: a close from task 2 leaves it held", holding(), closeAs("resent", 2), "same"],
    ["B1a FG: a failure frame for its task claims nothing and keeps it held", holding({ payload: stash1, pending: [staged1, staged2] }), open(1),
      holding({ payload: null, pending: [staged2] })],
    ["B1b FG: another task's failure frame clears it", holding(), open(2),
      state({ seq: 5, request: request(2, 5), gates: [ended(7, "cleared")] })],
    ["B2a FG: a session check for its task does nothing", holding(), check(1, "opened"), "same"],
    ["B2b FG: another task's session check clears it", holding(), check(2, "opened"),
      state({ seq: 5, request: sessionCheck(2, 5), gates: [ended(7, "cleared")] })],
    ["B2c FG: a check yielding to its own task's ticket leaves it held", holding({ pending: [staged2] }), check(2, "reconnected"), "same"],
    ["B2d FG: a check yielding to its own task's stash leaves it held", holding({ payload: stash2 }), check(2, "opened"), "same"],
    ["B2e FG: a reconnect check kept quiet by the mark leaves it held", holding({ dismissedCheck: 2 }), check(2, "reconnected"), "same"],
    ["B3a FG: a second gate for its task replaces it and clears the first", holding(), gate(1, 8),
      state({ seq: 5, request: fg(1, 8, 5), gates: [ended(7, "cleared"), held(8)] })],
    ["B3b FG: a gate for another task replaces it and clears the first", holding(), gate(2, 8),
      state({ seq: 5, request: fg(2, 8, 5), gates: [ended(7, "cleared"), held(8)] })],
    ["B4 FG: dismissed discards it, keeps the stash, sets the mark", holding({ payload: stash1 }), closeAs("dismissed"),
      state({ payload: stash1, dismissedCheck: 1, gates: [ended(7, "discarded")] })],
    ["B5 FG: resent releases it", holding(), closeAs("resent"), state({ gates: [ended(7, "released")] })],
    ["B6 FG: not-shown releases it", holding(), closeAs("not-shown"), state({ gates: [ended(7, "released")] })],
    ["B7 FG: left-host clears it", holding(), closeAs("left-host"), state({ gates: [ended(7, "cleared")] })],
    ["B8a FG: retaining its task keeps it held", holding(), { type: "retain", taskId: 1 }, "same"],
    ["B8b FG: retaining another task clears it", holding(), { type: "retain", taskId: 2 }, state({ gates: [ended(7, "cleared")] })],
    ["B8c FG: retaining no task clears it", holding(), { type: "retain", taskId: null }, state({ gates: [ended(7, "cleared")] })],
    ["B9a FG: a settlement for its task drops tickets and stash, keeps it held", holding({ payload: stash1, pending: [staged1] }), { type: "forget", taskId: 1 },
      holding()],
    ["B9b FG: a settlement for another task", holding(), { type: "forget", taskId: 2 }, "same"],
    ["B10 FG: an identity change clears it", holding({ payload: stash1 }), { type: "identity-changed" }, state({ gates: [ended(7, "cleared")] })],
    ["B12: handed-over entries leave the account, the held one stays", state({ request: fg(1, 8), gates: [ended(7, "released"), held(8)] }),
      { type: "gates-drained", ids: [7] }, state({ request: fg(1, 8), gates: [held(8)] })],
    ["B12: nothing to drop", holding(), { type: "gates-drained", ids: [9] }, "same"],
    ["an ended entry outlives a branch that builds its state from scratch", state({ gates: [ended(7, "released")] }), open(2),
      state({ seq: 5, request: request(2, 5), gates: [ended(7, "released")] })],
  ]

  it.each(rows)("%s", (_name, prev, input, expected) => {
    const next = transitionRequest(prev, input)
    if (expected === "same") expect(next).toBe(prev)
    else expect(next).toEqual(expected)
  })

  it.each([
    ["a failure frame for its task", open(1)],
    ["a settlement for its task", { type: "forget", taskId: 1 } as RequestInput],
  ])("keeps the very request object, seq and all, through %s", (_name, input) => {
    const prev = holding({ payload: stash1, pending: [staged1] })
    expect(transitionRequest(prev, input).request).toBe(prev.request)
  })

  it("keeps the request and payload objects when the account is drained", () => {
    const prev = state({ request: request(2, 4, stash2), payload: stash2, gates: [ended(7, "cleared")] })
    const next = transitionRequest(prev, { type: "gates-drained", ids: [7] })
    expect(next.request).toBe(prev.request)
    expect(next.payload).toBe(prev.payload)
  })

  it.each([
    ["an ended entry", state({ gates: [ended(7, "discarded")] }), [{ id: 7, decision: "discarded" }]],
    ["an entry its committed request still holds", holding(), []],
    ["an entry held by a request that is not the committed one", state({ request: fg(1, 8), gates: [held(7), held(8)] }),
      [{ id: 7, decision: "cleared" }]],
    ["an entry held with no committed request", state({ gates: [held(7)] }), [{ id: 7, decision: "cleared" }]],
    ["a state with no account", state(), []],
  ] as const)("hands over %s", (_name, committed, due) => {
    expect(gatesToSettle(committed)).toEqual(due)
  })
})

describe("the provider hands a held first message its ending after commit", () => {
  let actions: ConnectorRuntimeDialogActions | undefined
  let value: ConnectorRuntimeDialogValue | undefined
  function Probe() {
    actions = useConnectorRuntimeDialogActions()
    value = useConnectorRuntimeDialog()
    return null
  }
  const tree = () => <ConnectorRuntimeDialogProvider><Probe /></ConnectorRuntimeDialogProvider>
  function hold(taskId: number): FirstGateDecision[] {
    const seen: FirstGateDecision[] = []
    act(() => { void actions?.openFirstGate(taskId)?.then(d => { seen.push(d) }) })
    return seen
  }

  it("opens nothing for an invalid task id", () => {
    render(tree())
    for (const taskId of [0, -3, 1.5]) expect(actions?.openFirstGate(taskId)).toBeNull()
    expect(value?.request).toBeNull()
  })

  it("gives openFirstGate the same two no-provider defaults as every other action", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {})
    let silent: ConnectorRuntimeDialogActions | undefined
    let plain: ConnectorRuntimeDialogActions | undefined
    function Outside() {
      silent = useConnectorRuntimeDialogActionsIfMounted()
      plain = useConnectorRuntimeDialogActions()
      return null
    }
    render(<Outside />)
    expect(silent?.openFirstGate(1)).toBeNull()
    expect(warnSpy).not.toHaveBeenCalled()
    expect(plain?.openFirstGate(1)).toBeNull()
    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("openFirstGate() was called outside"))
    warnSpy.mockRestore()
  })

  it.each([
    ["resent", "released"],
    ["not-shown", "released"],
    ["dismissed", "discarded"],
    ["left-host", "cleared"],
  ] as const)("answers a %s close with %s, once", async (outcome, decision) => {
    const view = render(tree())
    const seen = hold(1)
    expect(value?.request).toMatchObject({ taskId: 1, trigger: "first_gate", resendPayload: null })
    act(() => { actions?.close(outcome, 1) })
    await flush()
    expect(seen).toEqual([decision])
    view.unmount()
    await Promise.resolve()
    expect(seen).toEqual([decision])
  })

  it.each([
    ["a task switch", () => actions?.retainOnlyTask(null)],
    ["another task's failure frame", () => actions?.openForTask(2)],
    ["another task's session check", () => actions?.openSessionCheck(2, "opened")],
  ])("answers cleared when %s takes the request away", async (_name, takeAway) => {
    render(tree())
    const seen = hold(1)
    act(() => { takeAway() })
    await flush()
    expect(seen).toEqual(["cleared"])
  })

  it("answers cleared when the signed-in identity changes", async () => {
    const view = render(tree())
    const seen = hold(1)
    authUserRef.current = { id: "u2" }
    view.rerender(tree())
    await flush()
    expect(seen).toEqual(["cleared"])
  })

  it("answers cleared when the provider unmounts with the message still held", async () => {
    const view = render(tree())
    const seen = hold(1)
    await Promise.resolve()
    expect(seen).toEqual([])
    view.unmount()
    await flush()
    expect(seen).toEqual(["cleared"])
  })

  it("keeps a gate held when another task's dialog closes in the same batch", async () => {
    render(tree())
    act(() => { actions?.openSessionCheck(1, "opened") })
    const seen: FirstGateDecision[] = []
    act(() => {
      void actions?.openFirstGate(2)?.then(d => { seen.push(d) })
      actions?.close("not-shown", 1)
    })
    await Promise.resolve()
    expect(value?.request).toMatchObject({ taskId: 2, trigger: "first_gate" })
    expect(seen).toEqual([])
  })

  it("does not re-render a subscriber when it drains the account", () => {
    let renders = 0
    const requests: unknown[] = []
    const payloads: unknown[] = []
    function Subscriber() {
      const { request, payload } = useConnectorRuntimeDialog()
      renders += 1
      requests.push(request)
      payloads.push(payload)
      return null
    }
    render(<ConnectorRuntimeDialogProvider><Probe /><Subscriber /></ConnectorRuntimeDialogProvider>)
    hold(1)
    const before = renders
    // Replacing the gate's request records "cleared"; draining it afterwards
    // must not hand a subscriber a new request or payload object.
    act(() => { actions?.openForTask(2) })
    expect(renders).toBe(before + 1)
    expect(new Set(requests.slice(before)).size).toBe(1)
    expect(new Set(payloads.slice(before)).size).toBe(1)
    // Control: a change to the state half is seen by this subscriber.
    act(() => {
      actions?.stagePendingDelivery({ taskId: 3, clientMessageId: "c3", text: "x" })
      actions?.recordDelivery({ taskId: 3, clientMessageId: "c3", text: "x" })
    })
    expect(renders).toBe(before + 2)
  })
})

describe("a first gate under StrictMode", () => {
  // StrictMode re-runs the provider's effects only on mount, and a gate
  // opened in that same commit (a child's mount effect) ends cleared by the
  // identity effect's own mount run whatever the re-run does, so this cannot
  // show an effect teardown the provider survives; the unmount fallback's
  // check for that case is pinned by the source scan below. What this pins:
  // every gate opened after mount is answered exactly once, and the real
  // unmount answers the one still held.
  it("answers a gate opened after mount once, and the real unmount answers cleared", async () => {
    let actions: ConnectorRuntimeDialogActions | undefined
    function Probe() {
      actions = useConnectorRuntimeDialogActions()
      return null
    }
    const tree = () => (
      <React.StrictMode>
        <ConnectorRuntimeDialogProvider><Probe /></ConnectorRuntimeDialogProvider>
      </React.StrictMode>
    )
    const view = render(tree())
    const first: FirstGateDecision[] = []
    act(() => { void actions?.openFirstGate(1)?.then(d => { first.push(d) }) })
    act(() => { actions?.close("resent", 1) })
    await flush()
    expect(first).toEqual(["released"])
    const second: FirstGateDecision[] = []
    act(() => { void actions?.openFirstGate(2)?.then(d => { second.push(d) }) })
    await Promise.resolve()
    expect(second).toEqual([])
    view.unmount()
    await flush()
    expect(second).toEqual(["cleared"])
    expect(first).toEqual(["released"])
  })
})

// Where the provider may write its state and hand a held message its ending,
// read off the source: every request and account write goes through apply
// (transitionRequest), and an ending is handed over only after commit. Each
// rule names the only regions a call may stand in, cut out by bracket
// balance rather than by line, so reformatting does not matter and a call
// moved to another place fails even when the count stays the same.
describe("keeps the held-message account behind one writer and one exit", () => {
  const source = readFileSync(path.resolve(__dirname, "./connector-runtime-dialog-context.tsx"), "utf8")
  // Comments go first: prose may name these functions and hold brackets.
  const code = source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "")
  const count = (text: string, pattern: RegExp) => Array.from(text.matchAll(pattern)).length

  // The bracketed group that opens at the first `open` at or after `from`.
  function group(from: number, open: "(" | "{"): string {
    const close = open === "(" ? ")" : "}"
    const start = code.indexOf(open, from)
    expect(start).toBeGreaterThanOrEqual(0)
    let depth = 0
    for (let i = start; i < code.length; i++) {
      if (code[i] === open) depth += 1
      else if (code[i] === close && --depth === 0) return code.slice(start, i + 1)
    }
    throw new Error(`unbalanced ${open} at ${start}`)
  }
  // The group right after the one place `pattern` matches.
  function after(pattern: RegExp, open: "(" | "{"): string {
    const hits = Array.from(code.matchAll(new RegExp(pattern.source, "g")))
    expect(hits).toHaveLength(1)
    const hit = hits[0]
    return group((hit?.index ?? 0) + (hit?.[0].length ?? 0), open)
  }
  // The argument list of every useEffect call, in source order.
  const effects = () => Array.from(code.matchAll(/\buseEffect\s*(?=\()/g), m => group(m.index ?? 0, "("))

  it("calls setState only in apply and the three ticket actions", () => {
    const apply = after(/\bconst\s+apply\s*=\s*useCallback\s*/, "(")
    expect(apply).toMatch(/^\(\s*\(\s*input\s*:\s*RequestInput\s*\)\s*=>\s*setState\s*\(\s*prev\s*=>\s*transitionRequest\s*\(\s*prev\s*,\s*input\s*\)\s*\)/)
    // The three ticket actions, as the provider's action object defines them
    // (the interface and the no-provider defaults name them too).
    const actionsObject = after(/\bconst\s+actions\s*=\s*useMemo\s*<\s*ConnectorRuntimeDialogActions\s*>\s*/, "(")
    const tickets = ["recordDelivery", "stagePendingDelivery", "discardPendingDelivery"].map(name => {
      const at = actionsObject.search(new RegExp(`\\b${name}\\s*:\\s*\\([^)]*\\)\\s*=>\\s*\\{`))
      expect(at).toBeGreaterThanOrEqual(0)
      return group(code.indexOf(actionsObject) + at, "{")
    })
    expect([apply, ...tickets].map(text => count(text, /\bsetState\s*\(/g))).toEqual([1, 1, 1, 1])
    expect(count(code, /\bsetState\s*\(/g)).toBe(4)
  })

  it("settles a gate only from the exit effect and the unmount fallback", () => {
    const settles = /\bsettleGate\s*\(/g
    // The declaration, plus one call in each of the two effects below.
    expect(count(code, settles)).toBe(3)
    expect(count(after(/\bfunction\s+settleGate\s*/, "("), settles)).toBe(0)
    const exit = effects().filter(effect => /\bgatesToSettle\s*\(\s*state\s*\)/.test(effect))
    const fallback = effects().filter(effect => /\bqueueMicrotask\s*\(/.test(effect))
    expect([exit.length, fallback.length]).toEqual([1, 1])
    expect([count(exit[0] ?? "", settles), count(fallback[0] ?? "", settles)]).toEqual([1, 1])
    expect(fallback[0]).toMatch(/settleGate\s*\(\s*resolvers\s*,\s*id\s*,\s*"cleared"\s*\)/)
    // Answers only a real unmount: the teardown marks the provider gone,
    // and the deferred check gives up if a re-setup marked it back.
    expect(fallback[0]).toMatch(/^\(\s*\(\s*\)\s*=>\s*\{\s*mountedRef\.current\s*=\s*true\b/)
    expect(fallback[0]).toMatch(/return\s*\(\s*\)\s*=>\s*\{\s*mountedRef\.current\s*=\s*false\s*queueMicrotask\s*\(\s*\(\s*\)\s*=>\s*\{\s*if\s*\(\s*mountedRef\.current\s*\)\s*return\b/)
    expect(count(code, /\.get\s*\(/g)).toBe(1)
    expect(after(/\bfunction\s+settleGate\s*\([^)]*\)\s*:\s*void\s*/, "{")).toMatch(/\bresolvers\.get\s*\(\s*id\s*\)/)
  })
})

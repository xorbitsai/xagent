import React from "react"
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
  type ConnectorRuntimeDialogActions,
  type ConnectorRuntimeDialogValue,
} from "./connector-runtime-dialog-context"

afterEach(() => {
  cleanup()
  authUserRef.current = { id: "u1" }
})

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
      actions?.close("dismissed")
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

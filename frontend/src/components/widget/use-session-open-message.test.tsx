import { useCallback, useRef, useState } from "react"
import { act, cleanup, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest"
import type { useApp } from "@/contexts/app-context-chat"
import { W_SEND_FAILED, type WidgetOpenRequest } from "@/hooks/use-widget-open-message"
import {
  W_RELOAD_REQUIRED,
  W_RESET_FAILED,
  W_RESET_SUPERSEDED,
  W_SESSION_ENDED,
  W_UNSUPPORTED,
  useSessionOpenMessage,
  type UseSessionOpenMessageInput,
} from "./use-session-open-message"

type AppControls = ReturnType<typeof useApp>
type Phase = AppControls["sessionConversationState"]
type Controls = Omit<UseSessionOpenMessageInput, "request" | "take">

const deferred = () => {
  let resolve!: () => void
  let reject!: (error: Error) => void
  const promise = new Promise<void>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
}

// Stands in for useWidgetOpenMessage: one slot, a newer message replaces one
// not yet taken, and take(seq) clears only the request it names.
let deliver: (message: string) => void = () => undefined
const take = vi.fn()
function useDriver(controls: Controls) {
  const [request, setRequest] = useState<WidgetOpenRequest | null>(null)
  const seqRef = useRef(0)
  deliver = (message) => {
    seqRef.current += 1
    setRequest({ seq: seqRef.current, message })
  }
  const takeRequest = useCallback((seq: number) => {
    take(seq)
    setRequest((current) => (current?.seq === seq ? null : current))
  }, [])
  useSessionOpenMessage({ ...controls, request, take: takeRequest })
}

let startNewConversation = vi.fn<() => Promise<void>>()
let sendStartMessage = vi.fn<(message: string) => Promise<void>>()
let warn: MockInstance<typeof console.warn>

const controls = (overrides: Partial<Controls> = {}): Controls => ({
  status: "active",
  hasAgent: true,
  sessionConversationAttachment: "attached",
  sessionConversationState: "unbound",
  isMessageDeliveryPending: false,
  isConversationResetPending: false,
  isAttachmentBusy: false,
  startNewConversation,
  sendStartMessage,
  ...overrides,
})

const mount = (overrides: Partial<Controls> = {}) => {
  let current = controls(overrides)
  const hook = renderHook((props: Controls) => useDriver(props), { initialProps: current })
  const update = (next: Partial<Controls>) => {
    current = { ...current, ...next }
    hook.rerender(current)
  }
  return { ...hook, update }
}

const open = (message: string) => act(() => deliver(message))
const settle = (step: () => void) => act(async () => { step() })
const sent = () => sendStartMessage.mock.calls

beforeEach(() => {
  take.mockReset()
  startNewConversation = vi.fn<() => Promise<void>>(() => new Promise(() => {}))
  sendStartMessage = vi.fn<(message: string) => Promise<void>>(() => new Promise(() => {}))
  warn = vi.spyOn(console, "warn").mockImplementation(() => {})
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe("useSessionOpenMessage", () => {
  // Bound with a reset under way that the test settles itself; the reset is
  // the first startNewConversation call.
  const startResetFromBound = (overrides: Partial<Controls> = {}) => {
    const reset = deferred()
    startNewConversation.mockReturnValueOnce(reset.promise)
    const driver = mount({ sessionConversationState: "bound", ...overrides })
    open("A")
    return { ...driver, reset }
  }

  describe("a message in the slot, by phase", () => {
    const SLOT_CASES: Record<Phase, () => Promise<void>> = {
      unbound: async () => {
        mount({ sessionConversationState: "unbound" })
        open("A")
        expect(sent()).toEqual([["A"]])
        expect(startNewConversation).not.toHaveBeenCalled()
      },
      bound: async () => {
        const { update, reset } = startResetFromBound()
        expect(startNewConversation).toHaveBeenCalledTimes(1)
        expect(sent()).toEqual([])

        // The acknowledged reset moves the phase before its promise settles.
        update({ sessionConversationState: "replacement_ready" })
        await settle(() => reset.resolve())

        expect(sent()).toEqual([["A"]])
        expect(startNewConversation).toHaveBeenCalledTimes(1)
      },
      replacement_ready: async () => {
        mount({ sessionConversationState: "replacement_ready" })
        open("A")
        expect(sent()).toEqual([["A"]])
        expect(startNewConversation).not.toHaveBeenCalled()
      },
      reset_requested: async () => {
        const { update } = mount({ sessionConversationState: "reset_requested" })
        open("A")
        expect(startNewConversation).not.toHaveBeenCalled()
        expect(sent()).toEqual([])
        update({ sessionConversationState: "bound" })
        expect(startNewConversation).toHaveBeenCalledTimes(1)
      },
      replacement_sending: async () => {
        const { update } = mount({ sessionConversationState: "replacement_sending" })
        open("A")
        expect(startNewConversation).not.toHaveBeenCalled()
        expect(sent()).toEqual([])
        update({ sessionConversationState: "bound" })
        expect(startNewConversation).toHaveBeenCalledTimes(1)
      },
      replacement_awaiting_task: async () => {
        const { update } = mount({ sessionConversationState: "replacement_awaiting_task" })
        open("A")
        expect(startNewConversation).not.toHaveBeenCalled()
        expect(sent()).toEqual([])
        update({ sessionConversationState: "bound" })
        expect(startNewConversation).toHaveBeenCalledTimes(1)
      },
      reload_required: async () => {
        mount({ sessionConversationState: "reload_required" })
        open("A")
        expect(startNewConversation).not.toHaveBeenCalled()
        expect(sent()).toEqual([])
        expect(take.mock.calls).toEqual([[1]])
        expect(warn.mock.calls).toEqual([[W_RELOAD_REQUIRED]])
      },
    }

    it.each(Object.keys(SLOT_CASES) as Phase[])("handles a slot message in the %s phase", async (phase) => {
      await SLOT_CASES[phase]()
      if (phase !== "reload_required") expect(warn).not.toHaveBeenCalled()
    })
  })

  describe("a message waiting after its reset, by phase", () => {
    // Counts after the setup: startNewConversation is cumulative (the setup
    // made the first call); take counts only calls after the setup's take(1).
    interface WaitingCase {
      starts: number
      sends: string[][]
      warns: string[][]
    }
    const WAITING_CASES: Record<Phase, WaitingCase> = {
      // Its reset already succeeded once, so it is dropped, not reset again.
      bound: { starts: 1, sends: [], warns: [[W_RESET_SUPERSEDED]] },
      replacement_ready: { starts: 1, sends: [["A"]], warns: [] },
      unbound: { starts: 1, sends: [["A"]], warns: [] },
      reset_requested: { starts: 1, sends: [], warns: [] },
      replacement_sending: { starts: 1, sends: [], warns: [] },
      replacement_awaiting_task: { starts: 1, sends: [], warns: [] },
      reload_required: { starts: 1, sends: [], warns: [[W_RELOAD_REQUIRED]] },
    }

    it.each(Object.keys(WAITING_CASES) as Phase[])("handles the waiting message in the %s phase", async (phase) => {
      const { update, reset } = startResetFromBound()
      expect(take.mock.calls).toEqual([[1]])
      take.mockClear()

      // The target phase arrives first, while the reset is still in hand.
      update({ sessionConversationState: phase })
      await settle(() => reset.resolve())

      const expected = WAITING_CASES[phase]
      expect(startNewConversation).toHaveBeenCalledTimes(expected.starts)
      expect(sent()).toEqual(expected.sends)
      expect(take).not.toHaveBeenCalled()
      expect(warn.mock.calls).toEqual(expected.warns)
      if (phase === "reload_required") {
        update({ sessionConversationState: "replacement_ready" })
        expect(sent()).toEqual([])
      }
    })

    it.each([
      ["the server does not support it", { sessionConversationAttachment: "unsupported" }, W_UNSUPPORTED],
      ["the session has ended", { status: "terminal" }, W_SESSION_ENDED],
    ] as const)("drops the waiting message when %s", async (_name, change, warning) => {
      const { update, reset } = startResetFromBound()
      take.mockClear()

      update({ sessionConversationState: "replacement_ready", ...change })
      await settle(() => reset.resolve())

      expect(startNewConversation).toHaveBeenCalledTimes(1)
      expect(sent()).toEqual([])
      expect(take).not.toHaveBeenCalled()
      expect(warn.mock.calls).toEqual([[warning]])
    })
  })

  describe("the current socket's attachment", () => {
    it("waits while pending and sends once attached", () => {
      const { update } = mount({ sessionConversationAttachment: "pending" })
      open("A")
      expect(sent()).toEqual([])
      expect(take).not.toHaveBeenCalled()

      update({ sessionConversationAttachment: "attached" })
      expect(sent()).toEqual([["A"]])
    })

    it("sends at once when attached", () => {
      mount({ sessionConversationAttachment: "attached" })
      open("A")
      expect(sent()).toEqual([["A"]])
    })

    it("drops the message when the server does not support it", () => {
      mount({ sessionConversationAttachment: "unsupported" })
      open("A")
      expect(sent()).toEqual([])
      expect(take.mock.calls).toEqual([[1]])
      expect(warn.mock.calls).toEqual([[W_UNSUPPORTED]])
    })

    it("does not reset while a reconnect has the attachment pending", () => {
      const { update } = mount({ sessionConversationState: "bound" })
      update({ sessionConversationAttachment: "pending" })
      open("A")
      expect(startNewConversation).not.toHaveBeenCalled()

      update({ sessionConversationAttachment: "attached" })
      expect(startNewConversation).toHaveBeenCalledTimes(1)
    })
  })

  describe("work already in flight", () => {
    it.each([
      ["a message delivery", "isMessageDeliveryPending"],
      ["a conversation reset", "isConversationResetPending"],
      ["an attachment upload", "isAttachmentBusy"],
    ] as const)("waits for %s to finish", (_name, flag) => {
      const { update } = mount({ sessionConversationState: "bound", [flag]: true } as Partial<Controls>)
      open("A")
      expect(startNewConversation).not.toHaveBeenCalled()

      update({ [flag]: false } as Partial<Controls>)
      expect(startNewConversation).toHaveBeenCalledTimes(1)
    })
  })

  describe("the page status", () => {
    it.each(["waiting", "refreshing", "degraded"] as const)("waits while the session is %s", (status) => {
      const { update } = mount({ status })
      open("A")
      expect(sent()).toEqual([])
      expect(take).not.toHaveBeenCalled()
      expect(warn).not.toHaveBeenCalled()

      update({ status: "active" })
      expect(sent()).toEqual([["A"]])
    })

    it("drops the message when the session has ended", () => {
      mount({ status: "terminal" })
      open("A")
      expect(sent()).toEqual([])
      expect(take.mock.calls).toEqual([[1]])
      expect(warn.mock.calls).toEqual([[W_SESSION_ENDED]])
    })

    it("handles the message by phase while the session is active", () => {
      mount({ status: "active" })
      open("A")
      expect(sent()).toEqual([["A"]])
    })

    it("waits while the agent is unknown", () => {
      const { update } = mount({ hasAgent: false })
      open("A")
      expect(sent()).toEqual([])

      update({ hasAgent: true })
      expect(sent()).toEqual([["A"]])
    })
  })

  describe("failures", () => {
    it("resets a message at most once when the conversation stays bound", async () => {
      // Throws from the second call on, so a missing guard fails here at once
      // instead of resetting without end.
      startNewConversation.mockImplementation(() => {
        if (startNewConversation.mock.calls.length > 1) {
          throw new Error("reset a second time")
        }
        return Promise.resolve()
      })
      mount({ sessionConversationState: "bound" })
      await act(async () => { deliver("A") })

      expect(startNewConversation).toHaveBeenCalledTimes(1)
      expect(sendStartMessage).not.toHaveBeenCalled()
      expect(warn.mock.calls).toEqual([[W_RESET_SUPERSEDED]])
    }, 2_000)

    it("drops the message when the new conversation cannot be started", async () => {
      const { update, reset } = startResetFromBound()
      await settle(() => reset.reject(new Error("reset refused")))

      expect(startNewConversation).toHaveBeenCalledTimes(1)
      expect(sent()).toEqual([])
      expect(warn.mock.calls).toEqual([[W_RESET_FAILED]])

      // The next message is handled as usual.
      update({ sessionConversationState: "bound" })
      open("B")
      expect(startNewConversation).toHaveBeenCalledTimes(2)
    })

    it("sends once and does not retry a failed send", async () => {
      const send = deferred()
      sendStartMessage.mockReturnValueOnce(send.promise)
      mount({ sessionConversationState: "unbound" })
      open("A")
      await settle(() => send.reject(new Error("not delivered")))

      expect(sent()).toEqual([["A"]])
      expect(warn.mock.calls).toEqual([[W_SEND_FAILED]])
    })
  })

  describe("host page calls while the chat is open", () => {
    // The driver takes no isProcessing input: a conversation whose task is
    // still running or waiting for the visitor is handled like any bound one.
    it("starts a new conversation over an established one, answered or not", async () => {
      const { update, reset } = startResetFromBound()
      update({ sessionConversationState: "replacement_ready" })
      await settle(() => reset.resolve())

      expect(startNewConversation).toHaveBeenCalledTimes(1)
      expect(sent()).toEqual([["A"]])
    })

    it("holds a second message until the first one's conversation is adopted", async () => {
      const sendA = deferred()
      sendStartMessage.mockReturnValueOnce(sendA.promise)
      const { update } = mount({ sessionConversationState: "unbound" })
      open("A")
      open("B")
      update({ sessionConversationState: "replacement_awaiting_task" })
      expect(take.mock.calls).toEqual([[1]])
      expect(startNewConversation).not.toHaveBeenCalled()

      // The new task's task_info is adopted before the send's acceptance.
      update({ sessionConversationState: "bound" })
      expect(startNewConversation).not.toHaveBeenCalled()
      const resetB = deferred()
      startNewConversation.mockReturnValueOnce(resetB.promise)
      await settle(() => sendA.resolve())

      expect(take.mock.calls).toEqual([[1], [2]])
      expect(startNewConversation).toHaveBeenCalledTimes(1)
      update({ sessionConversationState: "replacement_ready" })
      await settle(() => resetB.resolve())
      expect(sent()).toEqual([["A"], ["B"]])
    })

    it("sends the first and the last of three quick clicks made once ready", async () => {
      const sendA = deferred()
      sendStartMessage.mockReturnValueOnce(sendA.promise)
      const { update, reset } = startResetFromBound()
      open("B")
      open("C")

      update({ sessionConversationState: "replacement_ready" })
      await settle(() => reset.resolve())
      expect(sent()).toEqual([["A"]])

      update({ sessionConversationState: "bound" })
      const resetC = deferred()
      startNewConversation.mockReturnValueOnce(resetC.promise)
      await settle(() => sendA.resolve())
      update({ sessionConversationState: "replacement_ready" })
      await settle(() => resetC.resolve())

      expect(sent()).toEqual([["A"], ["C"]])
      expect(startNewConversation).toHaveBeenCalledTimes(2)
      expect(take.mock.calls).toEqual([[1], [3]])
    })

    it("sends only the last of three clicks made before the socket attached", async () => {
      const { update } = mount({
        sessionConversationState: "bound",
        sessionConversationAttachment: "pending",
      })
      open("A")
      open("B")
      open("C")
      const reset = deferred()
      startNewConversation.mockReturnValueOnce(reset.promise)

      update({ sessionConversationAttachment: "attached" })
      update({ sessionConversationState: "replacement_ready" })
      await settle(() => reset.resolve())

      expect(startNewConversation).toHaveBeenCalledTimes(1)
      expect(sent()).toEqual([["C"]])
    })
  })

  describe("unmount", () => {
    it("does not send after a reset that settles once unmounted", async () => {
      const { unmount, reset } = startResetFromBound()
      take.mockClear()
      unmount()
      await settle(() => reset.resolve())

      expect(sent()).toEqual([])
      expect(take).not.toHaveBeenCalled()
      expect(warn).not.toHaveBeenCalled()
    })

    it("does not warn for a reset that fails once unmounted", async () => {
      const { unmount, reset } = startResetFromBound()
      unmount()
      await settle(() => reset.reject(new Error("reset refused")))

      expect(warn).not.toHaveBeenCalled()
    })

    it("does not warn for a send that fails once unmounted", async () => {
      const send = deferred()
      sendStartMessage.mockReturnValueOnce(send.promise)
      const { unmount } = mount({ sessionConversationState: "unbound" })
      open("A")
      unmount()
      await settle(() => send.reject(new Error("not delivered")))

      expect(warn).not.toHaveBeenCalled()
    })
  })
})

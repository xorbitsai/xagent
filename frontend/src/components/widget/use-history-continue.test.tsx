import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { useCallback, useRef, useState } from "react"
import { act, cleanup, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { useApp } from "@/contexts/app-context-chat"
import type { WidgetOpenRequest } from "@/hooks/use-widget-open-message"
import {
  useHistoryContinue,
  type HistoryContinueOutcome,
  type HistoryContinueRequest,
  type UseHistoryContinueInput,
} from "./use-history-continue"
import { PHASE_ROUTE, useSessionOpenMessage } from "./use-session-open-message"

type AppControls = ReturnType<typeof useApp>
type Phase = AppControls["sessionConversationState"]
type Controls = Omit<UseHistoryContinueInput, "request" | "take">

const deferred = () => {
  let resolve!: () => void
  let reject!: (error: Error) => void
  const promise = new Promise<void>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
}

// Each test message continues from its own earlier conversation.
const SOURCES: Record<string, number> = { A: 101, B: 102, C: 103 }

// Stands in for the page's slot: one request, a newer one replaces one not
// yet taken, and take(seq) clears only the request it names.
let deliver: (message: string) => void = () => undefined
const take = vi.fn()
function useDriver(controls: Controls) {
  const [request, setRequest] = useState<HistoryContinueRequest | null>(null)
  const seqRef = useRef(0)
  deliver = (message) => {
    seqRef.current += 1
    setRequest({ seq: seqRef.current, message, sourceTaskId: SOURCES[message] })
  }
  const takeRequest = useCallback((seq: number) => {
    take(seq)
    setRequest((current) => (current?.seq === seq ? null : current))
  }, [])
  const { busy } = useHistoryContinue({ ...controls, request, take: takeRequest })
  return { busy, request }
}

let startNewConversation = vi.fn<() => Promise<void>>()
let sendContinueMessage = vi.fn<(message: string, sourceTaskId: number) => Promise<void>>()
let onOutcome = vi.fn<(outcome: HistoryContinueOutcome) => void>()

const controls = (overrides: Partial<Controls> = {}): Controls => ({
  status: "active",
  hasAgent: true,
  sessionConversationAttachment: "attached",
  sessionConversationState: "unbound",
  isMessageDeliveryPending: false,
  isConversationResetPending: false,
  isAttachmentBusy: false,
  startNewConversation,
  sendContinueMessage,
  onOutcome,
  ...overrides,
})

const mount = (overrides: Partial<Controls> = {}) => {
  let current = controls(overrides)
  const hook = renderHook((props: Controls) => useDriver(props), { initialProps: current })
  const update = (next: Partial<Controls>) => {
    current = { ...current, ...next }
    hook.rerender(current)
  }
  const busy = () => hook.result.current.busy
  return { ...hook, update, busy }
}

const open = (message: string) => act(() => deliver(message))
const settle = (step: () => void) => act(async () => { step() })
const sent = () => sendContinueMessage.mock.calls
const outcomes = () => onOutcome.mock.calls

beforeEach(() => {
  take.mockReset()
  startNewConversation = vi.fn<() => Promise<void>>(() => new Promise(() => {}))
  sendContinueMessage = vi.fn<(message: string, sourceTaskId: number) => Promise<void>>(() => new Promise(() => {}))
  onOutcome = vi.fn<(outcome: HistoryContinueOutcome) => void>()
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe("useHistoryContinue", () => {
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
        expect(sent()).toEqual([["A", 101]])
        expect(startNewConversation).not.toHaveBeenCalled()
      },
      bound: async () => {
        const { update, reset } = startResetFromBound()
        expect(startNewConversation).toHaveBeenCalledTimes(1)
        expect(sent()).toEqual([])

        // The acknowledged reset moves the phase before its promise settles.
        update({ sessionConversationState: "replacement_ready" })
        await settle(() => reset.resolve())

        expect(sent()).toEqual([["A", 101]])
        expect(startNewConversation).toHaveBeenCalledTimes(1)
      },
      replacement_ready: async () => {
        mount({ sessionConversationState: "replacement_ready" })
        open("A")
        expect(sent()).toEqual([["A", 101]])
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
        expect(outcomes()).toEqual([["reload_required"]])
      },
    }

    it.each(Object.keys(SLOT_CASES) as Phase[])("handles a slot message in the %s phase", async (phase) => {
      await SLOT_CASES[phase]()
      if (phase !== "reload_required") expect(onOutcome).not.toHaveBeenCalled()
    })
  })

  describe("a message waiting after its reset, by phase", () => {
    // Counts after the setup: startNewConversation is cumulative (the setup
    // made the first call); take counts only calls after the setup's take(1).
    interface WaitingCase {
      starts: number
      sends: Array<[string, number]>
      outcomes: HistoryContinueOutcome[][]
    }
    const WAITING_CASES: Record<Phase, WaitingCase> = {
      // Its reset already succeeded once, so it is dropped, not reset again.
      bound: { starts: 1, sends: [], outcomes: [["superseded"]] },
      replacement_ready: { starts: 1, sends: [["A", 101]], outcomes: [] },
      unbound: { starts: 1, sends: [["A", 101]], outcomes: [] },
      reset_requested: { starts: 1, sends: [], outcomes: [] },
      replacement_sending: { starts: 1, sends: [], outcomes: [] },
      replacement_awaiting_task: { starts: 1, sends: [], outcomes: [] },
      reload_required: { starts: 1, sends: [], outcomes: [["reload_required"]] },
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
      expect(outcomes()).toEqual(expected.outcomes)
      if (phase === "reload_required") {
        update({ sessionConversationState: "replacement_ready" })
        expect(sent()).toEqual([])
      }
    })

    it.each([
      ["the server does not support it", { sessionConversationAttachment: "unsupported" }, "unsupported"],
      ["the session has ended", { status: "terminal" }, "ended"],
    ] as const)("drops the waiting message when %s", async (_name, change, outcome) => {
      const { update, reset, busy } = startResetFromBound()
      take.mockClear()

      update({ sessionConversationState: "replacement_ready", ...change })
      await settle(() => reset.resolve())

      expect(startNewConversation).toHaveBeenCalledTimes(1)
      expect(sent()).toEqual([])
      expect(take).not.toHaveBeenCalled()
      expect(outcomes()).toEqual([[outcome]])
      expect(busy()).toBe(false)
    })
  })

  describe("the current socket's attachment", () => {
    it("waits while pending and sends once attached", () => {
      const { update } = mount({ sessionConversationAttachment: "pending" })
      open("A")
      expect(sent()).toEqual([])
      expect(take).not.toHaveBeenCalled()

      update({ sessionConversationAttachment: "attached" })
      expect(sent()).toEqual([["A", 101]])
    })

    it("sends at once when attached", () => {
      mount({ sessionConversationAttachment: "attached" })
      open("A")
      expect(sent()).toEqual([["A", 101]])
    })

    it("drops the message when the server does not support it", () => {
      const { busy } = mount({ sessionConversationAttachment: "unsupported" })
      open("A")
      expect(sent()).toEqual([])
      expect(take.mock.calls).toEqual([[1]])
      expect(outcomes()).toEqual([["unsupported"]])
      expect(busy()).toBe(false)
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
      expect(onOutcome).not.toHaveBeenCalled()

      update({ status: "active" })
      expect(sent()).toEqual([["A", 101]])
    })

    it("drops the message when the session has ended", () => {
      const { busy } = mount({ status: "terminal" })
      open("A")
      expect(sent()).toEqual([])
      expect(take.mock.calls).toEqual([[1]])
      expect(outcomes()).toEqual([["ended"]])
      expect(busy()).toBe(false)
    })

    it("handles the message by phase while the session is active", () => {
      mount({ status: "active" })
      open("A")
      expect(sent()).toEqual([["A", 101]])
    })

    it("waits while the agent is unknown", () => {
      const { update } = mount({ hasAgent: false })
      open("A")
      expect(sent()).toEqual([])

      update({ hasAgent: true })
      expect(sent()).toEqual([["A", 101]])
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
      const { busy } = mount({ sessionConversationState: "bound" })
      await act(async () => { deliver("A") })

      expect(startNewConversation).toHaveBeenCalledTimes(1)
      expect(sendContinueMessage).not.toHaveBeenCalled()
      expect(outcomes()).toEqual([["superseded"]])
      expect(busy()).toBe(false)
    }, 2_000)

    it("drops the message when the new conversation cannot be started", async () => {
      const { update, reset, busy } = startResetFromBound()
      await settle(() => reset.reject(new Error("reset refused")))

      expect(startNewConversation).toHaveBeenCalledTimes(1)
      expect(sent()).toEqual([])
      expect(outcomes()).toEqual([["reset_failed"]])
      expect(busy()).toBe(false)

      // The next message is handled as usual.
      update({ sessionConversationState: "bound" })
      open("B")
      expect(startNewConversation).toHaveBeenCalledTimes(2)
    })

    it("sends once and does not retry a failed send", async () => {
      const send = deferred()
      sendContinueMessage.mockReturnValueOnce(send.promise)
      const { busy } = mount({ sessionConversationState: "unbound" })
      open("A")
      await settle(() => send.reject(new Error("not delivered")))

      expect(sent()).toEqual([["A", 101]])
      expect(outcomes()).toEqual([["send_failed"]])
      expect(busy()).toBe(false)
    })
  })

  describe("messages arriving while one is in hand", () => {
    // The driver takes no isProcessing input: a conversation whose task is
    // still running or waiting for the visitor is handled like any bound one.
    it("starts a new conversation over an established one, answered or not", async () => {
      const { update, reset } = startResetFromBound()
      update({ sessionConversationState: "replacement_ready" })
      await settle(() => reset.resolve())

      expect(startNewConversation).toHaveBeenCalledTimes(1)
      expect(sent()).toEqual([["A", 101]])
    })

    it("holds a second message until the first one's conversation is adopted", async () => {
      const sendA = deferred()
      sendContinueMessage.mockReturnValueOnce(sendA.promise)
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

      expect(outcomes()).toEqual([["sent"]])
      expect(take.mock.calls).toEqual([[1], [2]])
      expect(startNewConversation).toHaveBeenCalledTimes(1)
      update({ sessionConversationState: "replacement_ready" })
      await settle(() => resetB.resolve())
      expect(sent()).toEqual([["A", 101], ["B", 102]])
    })

    it("sends the first and the last of three quick clicks made once ready", async () => {
      const sendA = deferred()
      sendContinueMessage.mockReturnValueOnce(sendA.promise)
      const { update, reset } = startResetFromBound()
      open("B")
      open("C")

      update({ sessionConversationState: "replacement_ready" })
      await settle(() => reset.resolve())
      expect(sent()).toEqual([["A", 101]])

      update({ sessionConversationState: "bound" })
      const resetC = deferred()
      startNewConversation.mockReturnValueOnce(resetC.promise)
      await settle(() => sendA.resolve())
      update({ sessionConversationState: "replacement_ready" })
      await settle(() => resetC.resolve())

      expect(sent()).toEqual([["A", 101], ["C", 103]])
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
      expect(sent()).toEqual([["C", 103]])
    })
  })

  describe("the outcome callback", () => {
    it.each([
      ["a send", "sent"],
      ["a reset", "reset_failed"],
    ] as const)("reports %s that settles later to the latest render's callback", async (_name, outcome) => {
      const step = deferred()
      if (outcome === "sent") sendContinueMessage.mockReturnValueOnce(step.promise)
      else startNewConversation.mockReturnValueOnce(step.promise)
      const earlier = onOutcome
      const { update } = mount({ sessionConversationState: outcome === "sent" ? "unbound" : "bound" })
      open("A")

      const latest = vi.fn<(outcome: HistoryContinueOutcome) => void>()
      update({ onOutcome: latest })
      await settle(() => (outcome === "sent" ? step.resolve() : step.reject(new Error("refused"))))

      expect(latest.mock.calls).toEqual([[outcome]])
      expect(earlier).not.toHaveBeenCalled()
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
      expect(onOutcome).not.toHaveBeenCalled()
    })

    it("does not report a reset that fails once unmounted", async () => {
      const { unmount, reset } = startResetFromBound()
      unmount()
      await settle(() => reset.reject(new Error("reset refused")))

      expect(onOutcome).not.toHaveBeenCalled()
    })

    it("does not report a send that fails once unmounted", async () => {
      const send = deferred()
      sendContinueMessage.mockReturnValueOnce(send.promise)
      const { unmount } = mount({ sessionConversationState: "unbound" })
      open("A")
      unmount()
      await settle(() => send.reject(new Error("not delivered")))

      expect(onOutcome).not.toHaveBeenCalled()
    })
  })

  describe("the phase routes", () => {
    it("reads the open message driver's phase table instead of keeping its own", () => {
      const source = readFileSync(resolve(process.cwd(), "src/components/widget/use-history-continue.ts"), "utf8")
      expect(source).not.toMatch(/["'`]replacement_awaiting_task["'`]/)
      expect(source).toMatch(/import \{ PHASE_ROUTE \} from "\.\/use-session-open-message"/)
    })

    it.each(Object.entries(PHASE_ROUTE) as Array<[Phase, (typeof PHASE_ROUTE)[Phase]]>)(
      "routes a message in the %s phase as %s",
      (phase, route) => {
        mount({ sessionConversationState: phase })
        open("A")

        expect(startNewConversation).toHaveBeenCalledTimes(route === "reset" ? 1 : 0)
        expect(sent()).toEqual(route === "send" ? [["A", 101]] : [])
        expect(outcomes()).toEqual(route === "discard" ? [["reload_required"]] : [])
        expect(take).toHaveBeenCalledTimes(route === "wait" ? 0 : 1)
      },
    )
  })

  describe("busy", () => {
    it("is true from the render the request arrives in, before it is taken", () => {
      const { result, update } = mount({ sessionConversationAttachment: "pending" })
      expect(result.current.busy).toBe(false)

      open("A")
      expect(result.current.request).not.toBeNull()
      expect(result.current.busy).toBe(true)
      expect(take).not.toHaveBeenCalled()

      update({ sessionConversationAttachment: "attached" })
      expect(take.mock.calls).toEqual([[1]])
      expect(result.current.request).toBeNull()
      expect(result.current.busy).toBe(true)
    })

    it("stays true while the message is in hand and turns false once it settles", async () => {
      const send = deferred()
      sendContinueMessage.mockReturnValueOnce(send.promise)
      const { update, reset, busy } = startResetFromBound()
      expect(busy()).toBe(true)

      update({ sessionConversationState: "replacement_ready" })
      await settle(() => reset.resolve())
      expect(sent()).toEqual([["A", 101]])
      expect(busy()).toBe(true)

      await settle(() => send.resolve())
      expect(outcomes()).toEqual([["sent"]])
      expect(busy()).toBe(false)
    })

    it("stays true while a newer request waits behind the one in hand", async () => {
      const send = deferred()
      sendContinueMessage.mockReturnValueOnce(send.promise)
      const { update, busy } = mount({ sessionConversationState: "unbound" })
      open("A")
      open("B")
      update({ sessionConversationState: "replacement_awaiting_task" })

      await settle(() => send.resolve())
      expect(busy()).toBe(true)
    })

    // Every way a message can end leaves the driver idle on the next render.
    const ENDINGS: Record<string, (driver: ReturnType<typeof mount>) => Promise<void>> = {
      "an accepted send": async () => {
        const send = deferred()
        sendContinueMessage.mockReturnValueOnce(send.promise)
        open("A")
        await settle(() => send.resolve())
      },
      "a refused send": async () => {
        const send = deferred()
        sendContinueMessage.mockReturnValueOnce(send.promise)
        open("A")
        await settle(() => send.reject(new Error("refused")))
      },
      "a failed reset": async ({ update }) => {
        const reset = deferred()
        startNewConversation.mockReturnValueOnce(reset.promise)
        update({ sessionConversationState: "bound" })
        open("A")
        await settle(() => reset.reject(new Error("refused")))
      },
      "a superseded reset": async ({ update }) => {
        const reset = deferred()
        startNewConversation.mockReturnValueOnce(reset.promise)
        update({ sessionConversationState: "bound" })
        open("A")
        await settle(() => reset.resolve())
      },
      "a reconnect without attach support after the reset": async ({ update }) => {
        const reset = deferred()
        startNewConversation.mockReturnValueOnce(reset.promise)
        update({ sessionConversationState: "bound" })
        open("A")
        update({ sessionConversationState: "replacement_ready", sessionConversationAttachment: "unsupported" })
        await settle(() => reset.resolve())
      },
      "a reload required after the reset": async ({ update }) => {
        const reset = deferred()
        startNewConversation.mockReturnValueOnce(reset.promise)
        update({ sessionConversationState: "bound" })
        open("A")
        update({ sessionConversationState: "reload_required" })
        await settle(() => reset.resolve())
      },
      "a session that ended after the reset": async ({ update }) => {
        const reset = deferred()
        startNewConversation.mockReturnValueOnce(reset.promise)
        update({ sessionConversationState: "bound" })
        open("A")
        update({ sessionConversationState: "replacement_ready", status: "terminal" })
        await settle(() => reset.resolve())
      },
      "a slot message dropped without attach support": async ({ update }) => {
        update({ sessionConversationAttachment: "unsupported" })
        open("A")
      },
      "a slot message dropped for a reload": async ({ update }) => {
        update({ sessionConversationState: "reload_required" })
        open("A")
      },
      "a slot message dropped once the session ended": async ({ update }) => {
        update({ status: "terminal" })
        open("A")
      },
    }

    it.each(Object.keys(ENDINGS))("is false on the render after %s", async (ending) => {
      const driver = mount()
      await ENDINGS[ending](driver)

      expect(onOutcome).toHaveBeenCalledTimes(1)
      expect(driver.result.current.request).toBeNull()
      expect(driver.busy()).toBe(false)
    })
  })

  describe("next to the open message driver", () => {
    // Wired the way a page that uses both drivers would wire them: this
    // driver first, and its busy fed into the open driver's
    // isMessageDeliveryPending input.
    let continueWith: (message: string) => void = () => undefined
    let openWith: (message: string) => void = () => undefined
    const log: string[] = []
    const continueReset = vi.fn<() => Promise<void>>()
    const openReset = vi.fn<() => Promise<void>>()
    const openSend = vi.fn<(message: string) => Promise<void>>()

    type PairControls = Omit<Controls, "startNewConversation" | "sendContinueMessage" | "onOutcome">

    function usePair(pair: PairControls) {
      const [continueRequest, setContinueRequest] = useState<HistoryContinueRequest | null>(null)
      const [openRequest, setOpenRequest] = useState<WidgetOpenRequest | null>(null)
      const seqRef = useRef(0)
      continueWith = (message) => {
        seqRef.current += 1
        setContinueRequest({ seq: seqRef.current, message, sourceTaskId: SOURCES[message] })
      }
      openWith = (message) => {
        seqRef.current += 1
        setOpenRequest({ seq: seqRef.current, message })
      }
      const takeContinue = useCallback((seq: number) => {
        setContinueRequest((current) => (current?.seq === seq ? null : current))
      }, [])
      const takeOpen = useCallback((seq: number) => {
        setOpenRequest((current) => (current?.seq === seq ? null : current))
      }, [])
      const { busy } = useHistoryContinue({
        ...pair,
        request: continueRequest,
        take: takeContinue,
        startNewConversation: continueReset,
        sendContinueMessage,
        onOutcome,
      })
      useSessionOpenMessage({
        ...pair,
        isMessageDeliveryPending: pair.isMessageDeliveryPending || busy,
        request: openRequest,
        take: takeOpen,
        startNewConversation: openReset,
        sendStartMessage: openSend,
      })
      return { busy, continueRequest, openRequest }
    }

    const pairControls = (overrides: Partial<PairControls> = {}): PairControls => ({
      status: "active",
      hasAgent: true,
      sessionConversationAttachment: "attached",
      sessionConversationState: "bound",
      isMessageDeliveryPending: false,
      isConversationResetPending: false,
      isAttachmentBusy: false,
      ...overrides,
    })

    const mountPair = (overrides: Partial<PairControls> = {}) => {
      let current = pairControls(overrides)
      const hook = renderHook((props: PairControls) => usePair(props), { initialProps: current })
      const update = (next: Partial<PairControls>) => {
        current = { ...current, ...next }
        hook.rerender(current)
      }
      return { ...hook, update }
    }

    // Each step settles under the test's control; the log records the order in
    // which the two drivers acted.
    let pending: Record<string, ReturnType<typeof deferred>> = {}
    beforeEach(() => {
      log.length = 0
      pending = {}
      vi.spyOn(console, "warn").mockImplementation(() => {})
      const step = (name: string) => () => {
        log.push(name)
        pending[name] = deferred()
        return pending[name].promise
      }
      continueReset.mockReset().mockImplementation(step("continue reset"))
      openReset.mockReset().mockImplementation(step("open reset"))
      sendContinueMessage.mockImplementation((message, sourceTaskId) => step(`continue ${message} from ${sourceTaskId}`)())
      openSend.mockReset().mockImplementation((message) => step(`open ${message}`)())
    })

    // Plays the provider: a reset acknowledgement reaches replacement_ready,
    // and an accepted first message binds the new conversation.
    const acknowledgeReset = async (update: (next: Partial<PairControls>) => void, name: string) => {
      update({ sessionConversationState: "replacement_ready" })
      await settle(() => pending[name].resolve())
    }
    const acceptSend = async (update: (next: Partial<PairControls>) => void, name: string) => {
      update({ sessionConversationState: "bound" })
      await settle(() => pending[name].resolve())
    }

    it("keeps the open message waiting when both arrive in the same render", async () => {
      const { result } = mountPair()

      act(() => {
        continueWith("A")
        openWith("hi")
      })

      expect(result.current.busy).toBe(true)
      expect(log).toEqual(["continue reset"])
      expect(openReset).not.toHaveBeenCalled()
      expect(openSend).not.toHaveBeenCalled()
      expect(result.current.openRequest).not.toBeNull()
    })

    it.each([
      ["its reset is in flight", "in flight"],
      ["its reset is acknowledged but not settled", "acknowledged"],
      ["it is still in the slot", "slot"],
    ] as const)("finishes the continuation first when the host page opens while %s", async (_name, stage) => {
      const { result, update } = mountPair(stage === "slot" ? { sessionConversationAttachment: "pending" } : {})
      act(() => continueWith("A"))
      if (stage === "acknowledged") update({ sessionConversationState: "replacement_ready" })

      act(() => openWith("hi"))
      expect(openReset).not.toHaveBeenCalled()
      expect(openSend).not.toHaveBeenCalled()

      if (stage === "slot") update({ sessionConversationAttachment: "attached" })
      if (stage === "acknowledged") {
        await settle(() => pending["continue reset"].resolve())
      } else {
        await acknowledgeReset(update, "continue reset")
      }
      expect(openReset).not.toHaveBeenCalled()
      await acceptSend(update, "continue A from 101")
      await acknowledgeReset(update, "open reset")
      await acceptSend(update, "open hi")

      expect(log).toEqual(["continue reset", "continue A from 101", "open reset", "open hi"])
      expect(outcomes()).toEqual([["sent"]])
      expect(result.current.busy).toBe(false)
      expect(result.current.continueRequest).toBeNull()
      expect(result.current.openRequest).toBeNull()
      expect(console.warn).not.toHaveBeenCalled()
    })
  })
})

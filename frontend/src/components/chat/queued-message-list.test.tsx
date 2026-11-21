import React from "react"
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { SessionMessageQueueControls } from "@/contexts/app-context-chat"
import {
  initialMessageQueueState,
  type MessageQueuePhase,
  type MessageQueueState,
  type QueuedMessage,
} from "@/contexts/session-message-queue"

vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t: (key: string) => key }) }))

import { QueuedMessageList } from "./queued-message-list"

afterEach(cleanup)

const item = (id: string, status: QueuedMessage["status"], text = `text ${id}`): QueuedMessage => ({
  clientMessageId: id, text, seq: id.charCodeAt(0), status,
})

const controls = (
  items: QueuedMessage[],
  phase: MessageQueuePhase,
  observed: MessageQueueState["observed"] = initialMessageQueueState.observed,
): SessionMessageQueueControls => ({
  state: { items, phase, observed },
  submit: vi.fn(),
  continueSending: vi.fn(),
  sendNow: vi.fn(),
  remove: vi.fn(),
  clear: vi.fn(),
})

const WAITING: MessageQueuePhase = { kind: "waiting", since: 1, answerRequired: false }

describe("QueuedMessageList", () => {
  it("renders nothing without queued messages", () => {
    const { container } = render(<QueuedMessageList queue={controls([], { kind: "idle" })} canContinue />)
    expect(container).toBeEmptyDOMElement()
  })

  it.each([
    ["waiting", "messageQueue.status.waiting", false],
    ["sending", "messageQueue.status.sending", true],
    ["failed", "messageQueue.status.failed", false],
  ] as const)("shows a %s row with its status and only the allowed removal", (status, label, removeDisabled) => {
    const queue = controls([item("a", status)], WAITING)
    render(<QueuedMessageList queue={queue} canContinue />)
    const row = within(screen.getByRole("list", { name: "messageQueue.listLabel" })).getByRole("listitem")
    expect(within(row).getByText(label)).toHaveAttribute("aria-live", "polite")
    const remove = within(row).getByRole("button", { name: "messageQueue.remove" })
    expect(remove).toHaveProperty("disabled", removeDisabled)
    fireEvent.click(remove)
    expect(queue.remove).toHaveBeenCalledTimes(removeDisabled ? 0 : 1)
    if (!removeDisabled) expect(queue.remove).toHaveBeenCalledWith("a")
    expect(screen.queryByText("messageQueue.continue")).toBeNull()
  })

  it("shows the whole text only when expanded and never parses it", () => {
    const text = "<b>bold</b> **x**\nsecond line"
    render(<QueuedMessageList queue={controls([item("a", "waiting", text)], WAITING)} canContinue />)
    const body = screen.getByText((_, node) => node?.tagName === "SPAN" && node.textContent === text)
    expect(body.className).toContain("truncate")
    expect(body.querySelector("b")).toBeNull()
    fireEvent.click(screen.getByRole("button", { name: "messageQueue.expand" }))
    expect(body.className).toContain("break-words")
    expect(screen.getByRole("button", { name: "messageQueue.collapse" })).toHaveAttribute("aria-expanded", "true")
  })

  it("keeps the other rows' expansion when a row is removed, and ties each button to its row's text", () => {
    const items = [item("a", "waiting", "first"), item("b", "waiting", "second")]
    const view = render(<QueuedMessageList queue={controls(items, WAITING)} canContinue />)
    const [, expandSecond] = screen.getAllByRole("button", { name: "messageQueue.expand" })
    fireEvent.click(expandSecond)
    const second = screen.getByText("second")
    expect(expandSecond).toHaveAttribute("aria-describedby", second.id)
    expect(screen.getAllByRole("button", { name: "messageQueue.remove" })[1]).toHaveAttribute("aria-describedby", second.id)

    view.rerender(<QueuedMessageList queue={controls(items.slice(1), WAITING)} canContinue />)
    expect(screen.getByRole("button", { name: "messageQueue.collapse" })).toHaveAttribute("aria-expanded", "true")
    expect(screen.getByText("second").className).toContain("break-words")
  })

  it("keeps a row expanded when its message gets a new id", () => {
    const view = render(<QueuedMessageList queue={controls([item("a", "failed")], WAITING)} canContinue />)
    fireEvent.click(screen.getByRole("button", { name: "messageQueue.expand" }))
    view.rerender(<QueuedMessageList queue={controls([{ ...item("a2", "failed"), seq: item("a", "failed").seq }], WAITING)} canContinue />)
    expect(screen.getByRole("button", { name: "messageQueue.collapse" })).toHaveAttribute("aria-expanded", "true")
  })

  it("scrolls inside a bounded list", () => {
    render(<QueuedMessageList queue={controls([item("a", "waiting")], WAITING)} canContinue />)
    expect(screen.getByRole("list").className).toContain("overflow-y-auto")
  })

  it.each([
    ["turn_failed", "messageQueue.held.turnFailed"],
    ["turn_paused", "messageQueue.held.turnInterrupted"],
    ["send_failed", "messageQueue.held.sendFailed"],
    ["earlier_send_failed", "messageQueue.held.earlierSendFailed"],
  ] as const)("explains a queue held for %s and offers continue and clear", (reason, text) => {
    const queue = controls([item("a", "waiting"), item("b", "waiting")], { kind: "held", reason })
    render(<QueuedMessageList queue={queue} canContinue />)
    const status = screen.getByRole("status")
    expect(status.tagName).toBe("P")
    expect(status.textContent).toBe(text)
    const continueButton = screen.getByRole("button", { name: "messageQueue.continue" })
    expect(continueButton).not.toHaveAttribute("aria-describedby")
    expect(screen.queryByText("messageQueue.continueUnavailable")).toBeNull()
    fireEvent.click(continueButton)
    fireEvent.click(screen.getByRole("button", { name: "messageQueue.clear" }))
    expect(queue.continueSending).toHaveBeenCalledTimes(1)
    expect(queue.clear).toHaveBeenCalledTimes(1)
  })

  it("disables continue, but not clear, when the queue cannot send", () => {
    const queue = controls([item("a", "failed")], { kind: "held", reason: "send_failed" })
    render(<QueuedMessageList queue={queue} canContinue={false} />)
    expect(screen.getByRole("button", { name: "messageQueue.continue" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "messageQueue.clear" })).toBeEnabled()
    const reason = screen.getByText("messageQueue.continueUnavailable")
    expect(reason.id).not.toBe("")
    expect(screen.getByRole("button", { name: "messageQueue.continue" })).toHaveAttribute("aria-describedby", reason.id)
    expect(screen.getByRole("status")).not.toContainElement(reason)
    expect(screen.getByRole("status").textContent).toBe("messageQueue.held.sendFailed")
  })

  it.each([
    ["the task waits on a question", { version: 2, status: "waiting_for_user" as const }, true],
    ["the task is running", { version: 2, status: "running" as const }, false],
  ])("shows the answer-first line only when %s", (_name, observed, shown) => {
    render(<QueuedMessageList queue={controls([item("a", "waiting")], { kind: "waiting", since: 1, answerRequired: false }, observed)} canContinue />)
    expect(screen.queryByText("messageQueue.waitingAnswer") !== null).toBe(shown)
  })

  it("has no send-now button when the connection does not announce it", () => {
    render(<QueuedMessageList queue={controls([item("a", "waiting")], { kind: "held", reason: "turn_failed" })} canContinue />)
    expect(screen.getAllByRole("button").map(button => button.getAttribute("aria-label") ?? button.textContent)).toEqual([
      "messageQueue.continue", "messageQueue.clear", "messageQueue.expand", "messageQueue.remove",
    ])
  })
  describe("send-now button", () => {
    const inserting = (id: string, insert: Partial<NonNullable<QueuedMessage["insert"]>> = {}): QueuedMessage => ({
      ...item(id, "inserting"),
      insert: { confirmed: false, inFlight: true, since: 1, connection: "c", resentAt: undefined, retries: 0, retryAt: undefined, ...insert },
    })
    const HELD: MessageQueuePhase = { kind: "held", reason: "turn_failed" }
    const sendNowButton = () => screen.queryByRole("button", { name: "messageQueue.sendNow" })

    it.each([
      ["the connection does not announce it", false, WAITING, item("a", "waiting"), undefined, "hidden"],
      ["the row is being sent", true, WAITING, item("a", "sending"), undefined, "disabled"],
      ["the row is being sent into the turn", true, WAITING, inserting("a"), undefined, "disabled"],
      ["the task waits on an answer", true, WAITING, item("a", "waiting"), { version: 2, status: "waiting_for_user" as const }, "disabled"],
      ["a stopped queue holds a waiting row", true, HELD, item("a", "waiting"), undefined, "enabled"],
      ["a stopped queue holds a failed row", true, HELD, item("a", "failed"), undefined, "enabled"],
      ["a waiting row in a waiting queue", true, WAITING, item("a", "waiting"), undefined, "enabled"],
      ["a failed row outside a stopped queue", true, WAITING, item("a", "failed"), undefined, "disabled"],
      ["a row that already goes first on the next turn", true, WAITING, { ...item("a", "waiting"), next: true }, undefined, "disabled"],
    ] as const)("send-now button state [%s]", (_name, available, phase, row, observed, expected) => {
      const queue = controls([row], phase, observed)
      render(<QueuedMessageList queue={queue} canContinue canSendNow={available} />)
      const button = sendNowButton()
      if (expected === "hidden") {
        expect(button).toBeNull()
        return
      }
      expect(button).toHaveProperty("disabled", expected === "disabled")
      fireEvent.click(button!)
      expect(queue.sendNow).toHaveBeenCalledTimes(expected === "enabled" ? 1 : 0)
      if (expected === "enabled") expect(queue.sendNow).toHaveBeenCalledWith("a")
    })

    it.each([
      ["a request is on its way", { inFlight: true }, true],
      ["the server holds it", { inFlight: false, confirmed: true }, true],
      ["only a retry waits", { inFlight: false, retryAt: 5000 }, false],
    ])("remove enabled only while a send-now retry is waiting [%s]", (_name, insert, disabled) => {
      render(<QueuedMessageList queue={controls([inserting("a", insert)], WAITING)} canContinue canSendNow />)
      expect(screen.getByRole("button", { name: "messageQueue.remove" })).toHaveProperty("disabled", disabled)
    })

    it("status shows Sends next", () => {
      render(<QueuedMessageList queue={controls([{ ...item("a", "waiting"), next: true }, inserting("b"), item("c", "waiting")], WAITING)} canContinue canSendNow />)
      expect(screen.getByText("messageQueue.status.next")).toBeInTheDocument()
      expect(screen.getByText("messageQueue.status.inserting")).toBeInTheDocument()
      expect(screen.getByText("messageQueue.status.waiting")).toBeInTheDocument()
    })

    it("send-now button names its row for screen readers", () => {
      render(<QueuedMessageList queue={controls([item("a", "waiting", "first"), item("b", "waiting", "second")], WAITING)} canContinue canSendNow />)
      const [, second] = screen.getAllByRole("button", { name: "messageQueue.sendNow" })
      expect(second).toHaveAttribute("aria-describedby", screen.getByText("second").id)
    })
  })
})

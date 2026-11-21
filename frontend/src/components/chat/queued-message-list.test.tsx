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

  it("has no send-now button", () => {
    render(<QueuedMessageList queue={controls([item("a", "waiting")], { kind: "held", reason: "turn_failed" })} canContinue />)
    expect(screen.getAllByRole("button").map(button => button.getAttribute("aria-label") ?? button.textContent)).toEqual([
      "messageQueue.continue", "messageQueue.clear", "messageQueue.expand", "messageQueue.remove",
    ])
  })
})

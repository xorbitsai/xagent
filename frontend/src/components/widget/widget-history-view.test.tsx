import React from "react"
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const history = vi.hoisted(() => ({ read: vi.fn() }))
type ComposerProps = {
  onSend: (message: string, config?: Record<string, unknown>) => void | Promise<void>
  isLoading?: boolean
  [key: string]: unknown
}
type MessageProps = {
  role: string
  content: React.ReactNode
  rawContent?: string
  timestamp?: string
  showProcessView?: boolean
}
const surfaces = vi.hoisted(() => ({
  input: null as null | ComposerProps,
  messages: [] as MessageProps[],
}))

vi.mock("@/lib/widget-session-history", async () => {
  const actual = await vi.importActual<typeof import("@/lib/widget-session-history")>(
    "@/lib/widget-session-history",
  )
  return { ...actual, readWidgetConversation: history.read }
})

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

vi.mock("@/components/chat/ChatInput", () => ({
  ChatInput: (props: ComposerProps) => {
    surfaces.input = props
    return (
      <button
        type="button"
        data-testid="composer"
        data-loading={String(Boolean(props.isLoading))}
        onClick={() => { void Promise.resolve(props.onSend("a question", { mode: "x" })).catch(() => undefined) }}
      />
    )
  },
}))

vi.mock("@/components/chat/ChatMessage", () => ({
  ChatMessage: (props: MessageProps) => {
    surfaces.messages.push(props)
    return (
      <div data-testid={`message-${props.role}`} data-process={String(props.showProcessView)}>
        {props.content}
      </div>
    )
  },
}))

import { PHASE_ROUTE } from "./use-session-open-message"
import { WidgetSessionHistoryError } from "@/lib/widget-session-history"
import {
  HistoryPreflightError,
  WidgetHistoryView,
  type WidgetHistoryViewProps,
} from "./widget-history-view"

const ORIGIN = "https://chat.example.test"
const AUTH = { getToken: vi.fn(async () => "st_token") }

const message = (
  id: number,
  role: "user" | "assistant",
  content: string,
  attachments: Array<{ filename: string; fileSize: number | null; mimeType: string | null }> = [],
) => ({ id, role, content, createdAt: "2026-01-02T03:04:05Z", attachments })

const messagePage = (messages = [message(1, "user", "Where is my order?")], nextCursor: string | null = null) => ({
  messages,
  nextCursor,
})

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
}

const defaults = (): WidgetHistoryViewProps => ({
  origin: ORIGIN,
  getAuth: () => AUTH,
  unauthorizedNotice: "tryLater",
  taskId: 101,
  title: "Order question",
  sessionPhase: "unbound",
  sessionAttachment: "attached",
  onBack: vi.fn(),
  onContinue: vi.fn(),
})

const renderView = (overrides: Partial<WidgetHistoryViewProps> = {}) => {
  const props = { ...defaults(), ...overrides }
  return { props, ...render(<WidgetHistoryView {...props} />) }
}

const send = async () => {
  await act(async () => { fireEvent.click(screen.getByTestId("composer")) })
}

describe("WidgetHistoryView", () => {
  beforeEach(() => {
    history.read.mockReset()
    surfaces.input = null
    surfaces.messages = []
  })
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  describe("reading the earlier conversation", () => {
    it("shows the title and the messages oldest first, without the process view", async () => {
      history.read.mockResolvedValue(messagePage([
        message(1, "user", "Where is my order?"),
        message(2, "assistant", "It ships today."),
      ]))
      renderView()

      await screen.findByText("Where is my order?")
      expect(screen.getByText("Order question")).toBeInTheDocument()
      expect(screen.getAllByTestId(/^message-/).map((node) => node.textContent)).toEqual([
        "Where is my order?",
        "It ships today.",
      ])
      expect(screen.getAllByTestId(/^message-/).every((node) => node.dataset.process === "false")).toBe(true)
      expect(surfaces.messages.map((props) => [props.role, props.rawContent, props.timestamp])).toEqual([
        ["user", undefined, "2026-01-02T03:04:05Z"],
        ["assistant", "It ships today.", "2026-01-02T03:04:05Z"],
      ])
    })

    it("reads the first page of the chosen conversation with the current auth", async () => {
      history.read.mockResolvedValue(messagePage())
      renderView({ taskId: 77 })

      await screen.findByText("Where is my order?")
      const [origin, auth, taskId, pageArg, signal] = history.read.mock.calls[0]
      expect([origin, auth, taskId, pageArg]).toEqual([ORIGIN, AUTH, 77, { cursor: null }])
      expect(signal).toBeInstanceOf(AbortSignal)
    })

    it("shows an attached file by name only, without an id or a way to open it", async () => {
      history.read.mockResolvedValue(messagePage([
        message(1, "user", "See attached", [
          { filename: "receipt.pdf", fileSize: 11, mimeType: "application/pdf" },
          { filename: "note.txt", fileSize: null, mimeType: null },
        ]),
      ]))
      renderView()

      await screen.findByText(/See attached/)
      const user = screen.getByTestId("message-user")
      expect(within(user).getByText("receipt.pdf")).toBeInTheDocument()
      expect(within(user).getByText("note.txt")).toBeInTheDocument()
      expect(user.querySelector("button, a")).toBeNull()
      expect(user.innerHTML).not.toContain("file_id")
    })

    it("moves focus to the back button when it opens", async () => {
      history.read.mockResolvedValue(messagePage())
      renderView()

      expect(screen.getByRole("button", { name: "widgetSession.history.back" })).toHaveFocus()
      await screen.findByText("Where is my order?")
    })

    it("goes back", async () => {
      history.read.mockResolvedValue(messagePage())
      const { props } = renderView()

      await screen.findByText("Where is my order?")
      fireEvent.click(screen.getByRole("button", { name: "widgetSession.history.back" }))

      expect(props.onBack).toHaveBeenCalledTimes(1)
    })

    it("shows a loading state until the first page arrives", async () => {
      const first = deferred<ReturnType<typeof messagePage>>()
      history.read.mockReturnValue(first.promise)
      renderView()

      expect(screen.getByRole("status")).toHaveTextContent("common.loading")
      await act(async () => first.resolve(messagePage()))
      expect(screen.queryByRole("status", { name: "" })).not.toBeInTheDocument()
    })

    it("says a conversation without messages is empty and still lets the visitor continue it", async () => {
      history.read.mockResolvedValue(messagePage([]))
      renderView()

      expect(await screen.findByText("widgetSession.history.viewEmpty")).toBeInTheDocument()
      expect(screen.getByTestId("composer")).toBeInTheDocument()
    })

    it("shows that the conversation can no longer be viewed and takes the composer away", async () => {
      history.read.mockRejectedValue(new WidgetSessionHistoryError("not_found"))
      renderView()

      expect(await screen.findByRole("alert")).toHaveTextContent("widgetSession.history.notFound")
      expect(screen.queryByTestId("composer")).not.toBeInTheDocument()
      expect(screen.getByRole("button", { name: "widgetSession.history.back" })).toBeInTheDocument()
    })

    it("shows that history is turned off and takes the composer away", async () => {
      history.read.mockRejectedValue(new WidgetSessionHistoryError("disabled"))
      renderView()

      expect(await screen.findByRole("alert")).toHaveTextContent("widgetSession.history.disabled")
      expect(screen.queryByTestId("composer")).not.toBeInTheDocument()
    })

    it("asks for a reload and offers no retry when a rejected token cannot be refreshed", async () => {
      history.read.mockRejectedValueOnce(new WidgetSessionHistoryError("unauthorized"))
      renderView({ unauthorizedNotice: "refreshToLoad" })

      expect(await screen.findByRole("alert")).toHaveTextContent("widgetSession.history.refreshToLoad")
      expect(screen.queryByRole("button", { name: "widgetSession.history.retry" })).not.toBeInTheDocument()
      // A check before a hand-over would be rejected the same way.
      expect(screen.getByTestId("composer").dataset.loading).toBe("true")
    })

    it("asks for a reload and locks the composer when a later page meets a rejected token", async () => {
      history.read
        .mockResolvedValueOnce(messagePage([message(1, "user", "Where is my order?")], "c2"))
        .mockRejectedValueOnce(new WidgetSessionHistoryError("unauthorized"))
      renderView({ unauthorizedNotice: "refreshToLoad" })
      await screen.findByText("Where is my order?")

      fireEvent.click(screen.getByRole("button", { name: "widgetSession.history.loadMore" }))

      expect(await screen.findByRole("alert")).toHaveTextContent("widgetSession.history.refreshToLoad")
      expect(screen.queryByRole("button", { name: "widgetSession.history.retry" })).not.toBeInTheDocument()
      expect(screen.getByTestId("composer").dataset.loading).toBe("true")
    })

    it("shows the try-later notice for a rejected token and retries", async () => {
      history.read
        .mockRejectedValueOnce(new WidgetSessionHistoryError("unauthorized"))
        .mockResolvedValueOnce(messagePage())
      renderView({ unauthorizedNotice: "tryLater" })

      expect(await screen.findByRole("alert")).toHaveTextContent("widgetSession.history.tryLater")
      expect(screen.getByTestId("composer").dataset.loading).toBe("false")
      fireEvent.click(screen.getByRole("button", { name: "widgetSession.history.retry" }))

      expect(await screen.findByText("Where is my order?")).toBeInTheDocument()
    })

    it("shows a failure line with a retry for any other failure and keeps the composer", async () => {
      history.read
        .mockRejectedValueOnce(new Error("boom"))
        .mockResolvedValueOnce(messagePage())
      renderView()

      expect(await screen.findByRole("alert")).toHaveTextContent("widgetSession.history.loadFailed")
      expect(screen.getByTestId("composer")).toBeInTheDocument()
      fireEvent.click(screen.getByRole("button", { name: "widgetSession.history.retry" }))

      await screen.findByText("Where is my order?")
      expect(history.read.mock.calls[1][3]).toEqual({ cursor: null })
      expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    })
  })

  describe("paging", () => {
    it("appends the next page after the cursor and drops the button on the last page", async () => {
      history.read
        .mockResolvedValueOnce(messagePage([message(1, "user", "first")], "cursor-2"))
        .mockResolvedValueOnce(messagePage([message(2, "assistant", "second")]))
      renderView()

      await screen.findByText("first")
      fireEvent.click(screen.getByRole("button", { name: "widgetSession.history.loadMore" }))

      await screen.findByText("second")
      expect(history.read.mock.calls[1][3]).toEqual({ cursor: "cursor-2" })
      expect(screen.getAllByTestId(/^message-/).map((node) => node.textContent)).toEqual(["first", "second"])
      expect(screen.queryByRole("button", { name: "widgetSession.history.loadMore" })).not.toBeInTheDocument()
    })

    it("keeps what is loaded when the next page fails, and retries it", async () => {
      history.read
        .mockResolvedValueOnce(messagePage([message(1, "user", "first")], "cursor-2"))
        .mockRejectedValueOnce(new WidgetSessionHistoryError("load_failed"))
        .mockResolvedValueOnce(messagePage([message(2, "assistant", "second")]))
      renderView()

      await screen.findByText("first")
      fireEvent.click(screen.getByRole("button", { name: "widgetSession.history.loadMore" }))

      expect(await screen.findByRole("alert")).toHaveTextContent("widgetSession.history.loadFailed")
      expect(screen.getByText("first")).toBeInTheDocument()
      expect(screen.getByTestId("composer")).toBeInTheDocument()
      fireEvent.click(screen.getByRole("button", { name: "widgetSession.history.retry" }))

      await screen.findByText("second")
      expect(history.read.mock.calls[2][3]).toEqual({ cursor: "cursor-2" })
    })

    it("disables the button while the next page loads", async () => {
      const next = deferred<ReturnType<typeof messagePage>>()
      history.read
        .mockResolvedValueOnce(messagePage([message(1, "user", "first")], "cursor-2"))
        .mockReturnValueOnce(next.promise)
      renderView()

      await screen.findByText("first")
      const button = screen.getByRole("button", { name: "widgetSession.history.loadMore" })
      fireEvent.click(button)
      expect(button).toBeDisabled()
      await act(async () => next.resolve(messagePage([message(2, "assistant", "second")])))
    })
  })

  describe("a request cancelled by a second mount", () => {
    it("does not report the cancelled request as a failure", async () => {
      const second = deferred<ReturnType<typeof messagePage>>()
      history.read
        .mockImplementationOnce(
          (_o: string, _a: unknown, _t: number, _p: unknown, signal: AbortSignal) =>
            new Promise((_resolve, reject) => {
              signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))
            }),
        )
        .mockReturnValueOnce(second.promise)
      render(
        <React.StrictMode>
          <WidgetHistoryView {...defaults()} />
        </React.StrictMode>,
      )

      await act(async () => undefined)
      expect(history.read).toHaveBeenCalledTimes(2)
      expect(screen.queryByText("widgetSession.history.loadFailed")).not.toBeInTheDocument()
      await act(async () => second.resolve(messagePage()))
      await screen.findByText("Where is my order?")
      expect(screen.queryByText("widgetSession.history.loadFailed")).not.toBeInTheDocument()
    })
  })

  describe("a response that arrives after the view closed", () => {
    it.each([
      ["a page", (first: ReturnType<typeof deferred<ReturnType<typeof messagePage>>>) => first.resolve(messagePage())],
      ["a failure", (first: ReturnType<typeof deferred<ReturnType<typeof messagePage>>>) =>
        first.reject(new WidgetSessionHistoryError("load_failed"))],
    ])("is dropped quietly when it is %s", async (_label, settle) => {
      const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined)
      const first = deferred<ReturnType<typeof messagePage>>()
      history.read.mockReturnValue(first.promise)
      const { unmount } = renderView()

      const signal = history.read.mock.calls[0][4] as AbortSignal
      unmount()
      expect(signal.aborted).toBe(true)
      await act(async () => settle(first))
      await act(async () => undefined)

      expect(consoleError).not.toHaveBeenCalled()
    })
  })

  describe("whether the visitor can send from here", () => {
    const phases = Object.keys(PHASE_ROUTE) as Array<keyof typeof PHASE_ROUTE>
    const attachments = ["pending", "attached", "unsupported"] as const
    const cases = phases.flatMap((phase) => attachments.map((attachment) => [phase, attachment] as const))

    it.each(cases)("phase %s, replay %s", async (phase, attachment) => {
      history.read.mockResolvedValue(messagePage())
      renderView({ sessionPhase: phase, sessionAttachment: attachment })
      await screen.findByText("Where is my order?")

      const sendable = PHASE_ROUTE[phase] === "send" || PHASE_ROUTE[phase] === "reset"
      const expectedHint = attachment === "unsupported"
        ? "widgetSession.history.sendUnavailable"
        : attachment === "pending" || !sendable
          ? "widgetSession.history.waitForCurrent"
          : null

      expect(screen.getByTestId("composer").dataset.loading).toBe(String(expectedHint !== null))
      if (expectedHint) {
        expect(screen.getByRole("status", { name: "" })).toHaveTextContent(expectedHint)
      } else {
        expect(screen.queryByText(/sendUnavailable|waitForCurrent/)).not.toBeInTheDocument()
      }
    })

    it("configures the composer as a text-only box", async () => {
      history.read.mockResolvedValue(messagePage())
      renderView()
      await screen.findByText("Where is my order?")

      expect(surfaces.input).toMatchObject({
        filesDisabled: true,
        hideFileUpload: true,
        hideConfig: true,
        readOnlyConfig: true,
        compact: true,
        voiceInputEnabled: false,
      })
    })
  })

  describe("sending", () => {
    it.each(["unbound", "bound", "replacement_ready"] as const)(
      "checks the source with a one-message read in phase %s, then hands the message over",
      async (sessionPhase) => {
        history.read.mockResolvedValue(messagePage())
        const { props } = renderView({ sessionPhase })
        await screen.findByText("Where is my order?")
        history.read.mockClear()
        history.read.mockResolvedValue(messagePage([]))

        await send()

        expect(history.read).toHaveBeenCalledTimes(1)
        const [origin, auth, taskId, pageArg] = history.read.mock.calls[0]
        expect([origin, auth, taskId, pageArg]).toEqual([ORIGIN, AUTH, 101, { limit: 1 }])
        expect(props.onContinue).toHaveBeenCalledTimes(1)
        expect(props.onContinue).toHaveBeenCalledWith("a question", 101)
      },
    )

    it("hands the message over after a source that has no messages", async () => {
      history.read.mockResolvedValueOnce(messagePage([])).mockResolvedValueOnce(messagePage([]))
      const { props } = renderView()
      await screen.findByText("widgetSession.history.viewEmpty")

      await send()

      expect(props.onContinue).toHaveBeenCalledWith("a question", 101)
    })

    it("asks for the auth again when it sends", async () => {
      history.read.mockResolvedValue(messagePage())
      const refreshed = { getToken: vi.fn(async () => "later") }
      const getAuth = vi.fn().mockReturnValueOnce(AUTH).mockReturnValue(refreshed)
      renderView({ getAuth })
      await screen.findByText("Where is my order?")

      await send()

      expect(history.read.mock.calls[1][1]).toBe(refreshed)
    })

    it("keeps the composer locked while the source is being checked, and waits for a refreshed token", async () => {
      history.read.mockResolvedValueOnce(messagePage())
      const check = deferred<ReturnType<typeof messagePage>>()
      history.read.mockReturnValueOnce(check.promise)
      const { props } = renderView()
      await screen.findByText("Where is my order?")

      await send()
      expect(screen.getByTestId("composer").dataset.loading).toBe("true")
      expect(props.onContinue).not.toHaveBeenCalled()
      await act(async () => check.resolve(messagePage([])))

      expect(props.onContinue).toHaveBeenCalledTimes(1)
    })

    it.each([
      ["a conversation that is gone", new WidgetSessionHistoryError("not_found"),
        "widgetSession.history.notFound", false],
      ["history turned off", new WidgetSessionHistoryError("disabled"),
        "widgetSession.history.disabled", false],
      ["a token that may not be refreshed", new WidgetSessionHistoryError("unauthorized"),
        "widgetSession.history.tryLater", true],
      ["a failed read", new WidgetSessionHistoryError("load_failed"),
        "widgetSession.startMessageFailed", true],
      ["an error of another kind", new Error("boom"),
        "widgetSession.startMessageFailed", true],
    ])("stays here and says so for %s", async (_label, failure, text, keepsComposer) => {
      history.read.mockResolvedValueOnce(messagePage())
      const { props } = renderView()
      await screen.findByText("Where is my order?")
      history.read.mockRejectedValueOnce(failure)
      let rejection: unknown
      await act(async () => {
        try {
          await surfaces.input?.onSend("a question", {})
        } catch (error) {
          rejection = error
        }
      })

      expect(rejection).toBeInstanceOf(HistoryPreflightError)
      expect(rejection).toMatchObject({ notificationHandled: true })
      expect(props.onContinue).not.toHaveBeenCalled()
      expect(screen.getByRole("alert")).toHaveTextContent(text)
      expect(Boolean(screen.queryByTestId("composer"))).toBe(keepsComposer)
    })

    it("tells a rejected token in the replacement phase to reload the chat", async () => {
      history.read.mockResolvedValueOnce(messagePage())
      renderView({ sessionPhase: "replacement_ready", unauthorizedNotice: "refreshToLoad" })
      await screen.findByText("Where is my order?")
      history.read.mockRejectedValueOnce(new WidgetSessionHistoryError("unauthorized"))

      await send()

      expect(screen.getByRole("alert")).toHaveTextContent("widgetSession.history.refreshToLoad")
      // A reload is the only way past it, so the composer stays locked and
      // keeps the visitor's text.
      expect(screen.getByTestId("composer").dataset.loading).toBe("true")
    })

    it("clears the failure line and sends on the next try", async () => {
      history.read.mockResolvedValueOnce(messagePage())
      const { props } = renderView()
      await screen.findByText("Where is my order?")
      history.read.mockRejectedValueOnce(new WidgetSessionHistoryError("load_failed"))
      await send()
      expect(screen.getByRole("alert")).toBeInTheDocument()
      history.read.mockResolvedValueOnce(messagePage([]))

      await send()

      expect(props.onContinue).toHaveBeenCalledTimes(1)
      expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    })

    it("drops the result of a check that finishes after the view closed", async () => {
      const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined)
      history.read.mockResolvedValueOnce(messagePage())
      const check = deferred<ReturnType<typeof messagePage>>()
      history.read.mockReturnValueOnce(check.promise)
      const { props, unmount } = renderView()
      await screen.findByText("Where is my order?")

      await send()
      unmount()
      await act(async () => check.resolve(messagePage([])))

      expect(props.onContinue).not.toHaveBeenCalled()
      expect(consoleError).not.toHaveBeenCalled()
    })

    it("cancels the check when the view closes", async () => {
      history.read.mockResolvedValueOnce(messagePage())
      const check = deferred<ReturnType<typeof messagePage>>()
      history.read.mockReturnValueOnce(check.promise)
      const { unmount } = renderView()
      await screen.findByText("Where is my order?")

      await send()
      const signal = history.read.mock.calls[1][4] as AbortSignal
      expect(signal.aborted).toBe(false)
      unmount()

      expect(signal.aborted).toBe(true)
    })

    it("drops a failed check that finishes after the view closed", async () => {
      const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined)
      history.read.mockResolvedValueOnce(messagePage())
      const check = deferred<ReturnType<typeof messagePage>>()
      history.read.mockReturnValueOnce(check.promise)
      const { props, unmount } = renderView()
      await screen.findByText("Where is my order?")

      await send()
      unmount()
      await act(async () => check.reject(new WidgetSessionHistoryError("not_found")))

      expect(props.onContinue).not.toHaveBeenCalled()
      expect(consoleError).not.toHaveBeenCalled()
    })

    it("lets the visitor go back while the source is being checked", async () => {
      history.read.mockResolvedValueOnce(messagePage())
      const check = deferred<ReturnType<typeof messagePage>>()
      history.read.mockReturnValueOnce(check.promise)
      const { props } = renderView()
      await screen.findByText("Where is my order?")

      await send()
      fireEvent.click(screen.getByRole("button", { name: "widgetSession.history.back" }))

      expect(props.onBack).toHaveBeenCalledTimes(1)
      expect(props.onContinue).not.toHaveBeenCalled()
    })
  })
})

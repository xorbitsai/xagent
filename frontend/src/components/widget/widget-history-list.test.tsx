import React from "react"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const history = vi.hoisted(() => ({ list: vi.fn() }))

vi.mock("@/lib/widget-session-history", async () => {
  const actual = await vi.importActual<typeof import("@/lib/widget-session-history")>(
    "@/lib/widget-session-history",
  )
  return { ...actual, listWidgetConversations: history.list }
})

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

import { WidgetSessionHistoryError } from "@/lib/widget-session-history"
import { WidgetHistoryList, type WidgetHistoryListProps } from "./widget-history-list"

const ORIGIN = "https://chat.example.test"
const AUTH = { getToken: vi.fn(async () => "st_token") }

const page = (ids: number[], nextCursor: string | null = null) => ({
  conversations: ids.map((taskId) => ({
    taskId,
    title: `Conversation ${taskId}`,
    createdAt: "2026-01-02T03:04:05Z",
  })),
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

const handlers = () => ({
  onBack: vi.fn(),
  onOpenCurrent: vi.fn(),
  onOpen: vi.fn(),
})

const renderList = (overrides: Partial<WidgetHistoryListProps> = {}) => {
  const props = {
    origin: ORIGIN,
    getAuth: () => AUTH,
    unauthorizedNotice: "tryLater" as const,
    currentTaskId: null,
    ...handlers(),
    ...overrides,
  }
  return { props, ...render(<WidgetHistoryList {...props} />) }
}

describe("WidgetHistoryList", () => {
  beforeEach(() => {
    history.list.mockReset()
  })
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it("lists the earlier conversations in the order the server returned them", async () => {
    history.list.mockResolvedValue(page([30, 20, 10]))
    renderList()

    await screen.findByText("Conversation 30")
    const titles = screen.getAllByRole("listitem").map(
      (item) => item.querySelector("span")?.textContent,
    )
    expect(titles).toEqual(["Conversation 30", "Conversation 20", "Conversation 10"])
    expect(screen.getByText("widgetSession.history.listTitle")).toBeInTheDocument()
  })

  it("reads the first page with the current auth, the origin and no cursor", async () => {
    history.list.mockResolvedValue(page([1]))
    const getAuth = vi.fn(() => AUTH)
    renderList({ getAuth })

    await screen.findByText("Conversation 1")
    expect(history.list).toHaveBeenCalledTimes(1)
    const [origin, auth, pageArg, signal] = history.list.mock.calls[0]
    expect(origin).toBe(ORIGIN)
    expect(auth).toBe(AUTH)
    expect(pageArg).toEqual({ cursor: null })
    expect(signal).toBeInstanceOf(AbortSignal)
  })

  it("moves focus to the back button when it opens", async () => {
    history.list.mockResolvedValue(page([1]))
    renderList()

    expect(screen.getByRole("button", { name: "widgetSession.history.back" })).toHaveFocus()
    await screen.findByText("Conversation 1")
  })

  it("shows a loading state until the first page arrives", async () => {
    const first = deferred<ReturnType<typeof page>>()
    history.list.mockReturnValue(first.promise)
    renderList()

    expect(screen.getByRole("status")).toBeInTheDocument()
    await act(async () => first.resolve(page([1])))
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
  })

  it("marks the current conversation and returns to the chat without reading anything", async () => {
    history.list.mockResolvedValue(page([30, 20]))
    const { props } = renderList({ currentTaskId: 20 })

    await screen.findByText("Conversation 30")
    expect(screen.getAllByText("widgetSession.history.currentConversation")).toHaveLength(1)
    fireEvent.click(screen.getByRole("button", { name: /Conversation 20/ }))

    expect(props.onOpenCurrent).toHaveBeenCalledTimes(1)
    expect(props.onOpen).not.toHaveBeenCalled()
    expect(history.list).toHaveBeenCalledTimes(1)
  })

  it("opens another conversation with its id and title", async () => {
    history.list.mockResolvedValue(page([30, 20]))
    const { props } = renderList({ currentTaskId: 20 })

    await screen.findByText("Conversation 30")
    fireEvent.click(screen.getByRole("button", { name: /Conversation 30/ }))

    expect(props.onOpen).toHaveBeenCalledWith(30, "Conversation 30")
    expect(props.onOpenCurrent).not.toHaveBeenCalled()
  })

  it("goes back", async () => {
    history.list.mockResolvedValue(page([1]))
    const { props } = renderList()

    await screen.findByText("Conversation 1")
    fireEvent.click(screen.getByRole("button", { name: "widgetSession.history.back" }))

    expect(props.onBack).toHaveBeenCalledTimes(1)
  })

  it("says so when there are no earlier conversations", async () => {
    history.list.mockResolvedValue(page([]))
    renderList()

    expect(await screen.findByText("widgetSession.history.listEmpty")).toBeInTheDocument()
    expect(screen.queryByText("widgetSession.history.loadMore")).not.toBeInTheDocument()
  })

  it("formats the creation time of each conversation", async () => {
    history.list.mockResolvedValue({
      conversations: [{ taskId: 5, title: "Dated", createdAt: "2026-01-02T03:04:05Z" }],
      nextCursor: null,
    })
    renderList()

    const row = await screen.findByRole("button", { name: /Dated/ })
    const expected = new Date("2026-01-02T03:04:05Z").toLocaleString([], {
      month: "numeric",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    })
    expect(row).toHaveTextContent(expected)
  })

  describe("when the first page cannot be read", () => {
    it("shows a failure line and retries from the first page", async () => {
      history.list
        .mockRejectedValueOnce(new WidgetSessionHistoryError("load_failed"))
        .mockResolvedValueOnce(page([7]))
      renderList()

      const alert = await screen.findByRole("alert")
      expect(alert).toHaveTextContent("widgetSession.history.loadFailed")
      fireEvent.click(screen.getByRole("button", { name: "widgetSession.history.retry" }))

      expect(await screen.findByText("Conversation 7")).toBeInTheDocument()
      expect(screen.queryByRole("alert")).not.toBeInTheDocument()
      expect(history.list.mock.calls[1][2]).toEqual({ cursor: null })
    })

    it.each([
      ["a missing route", "not_found"],
      ["a generic failure", "load_failed"],
    ] as const)("treats %s as a load failure", async (_label, kind) => {
      history.list.mockRejectedValue(new WidgetSessionHistoryError(kind))
      renderList()

      expect(await screen.findByRole("alert")).toHaveTextContent("widgetSession.history.loadFailed")
    })

    it("says history is turned off when the server refuses", async () => {
      history.list.mockRejectedValue(new WidgetSessionHistoryError("disabled"))
      renderList()

      expect(await screen.findByRole("alert")).toHaveTextContent("widgetSession.history.disabled")
    })

    it("asks for a reload and offers no retry when a rejected token cannot be refreshed", async () => {
      history.list.mockRejectedValue(new WidgetSessionHistoryError("unauthorized"))
      renderList({ unauthorizedNotice: "refreshToLoad" })

      expect(await screen.findByRole("alert")).toHaveTextContent("widgetSession.history.refreshToLoad")
      expect(screen.queryByRole("button", { name: "widgetSession.history.retry" })).not.toBeInTheDocument()
    })

    it("says to try later and still offers a retry for a rejected token", async () => {
      history.list.mockRejectedValue(new WidgetSessionHistoryError("unauthorized"))
      renderList({ unauthorizedNotice: "tryLater" })

      expect(await screen.findByRole("alert")).toHaveTextContent("widgetSession.history.tryLater")
      expect(screen.getByRole("button", { name: "widgetSession.history.retry" })).toBeInTheDocument()
    })

    it("treats an error that is not a history error as a load failure", async () => {
      history.list.mockRejectedValue(new Error("boom"))
      renderList()

      expect(await screen.findByRole("alert")).toHaveTextContent("widgetSession.history.loadFailed")
    })
  })

  describe("paging", () => {
    it("appends the next page after the cursor and hides the button on the last page", async () => {
      history.list
        .mockResolvedValueOnce(page([30, 20], "cursor-2"))
        .mockResolvedValueOnce(page([10], null))
      renderList()

      await screen.findByText("Conversation 30")
      fireEvent.click(screen.getByRole("button", { name: "widgetSession.history.loadMore" }))

      await screen.findByText("Conversation 10")
      expect(history.list.mock.calls[1][2]).toEqual({ cursor: "cursor-2" })
      expect(screen.getByText("Conversation 30")).toBeInTheDocument()
      expect(screen.queryByRole("button", { name: "widgetSession.history.loadMore" })).not.toBeInTheDocument()
    })

    it("disables the button while the next page is loading", async () => {
      const next = deferred<ReturnType<typeof page>>()
      history.list.mockResolvedValueOnce(page([30], "cursor-2")).mockReturnValueOnce(next.promise)
      renderList()

      await screen.findByText("Conversation 30")
      const button = screen.getByRole("button", { name: "widgetSession.history.loadMore" })
      fireEvent.click(button)
      expect(button).toBeDisabled()
      fireEvent.click(button)
      expect(history.list).toHaveBeenCalledTimes(2)
      await act(async () => next.resolve(page([10])))
    })

    it("keeps what is loaded when the next page fails, and retries it", async () => {
      history.list
        .mockResolvedValueOnce(page([30], "cursor-2"))
        .mockRejectedValueOnce(new WidgetSessionHistoryError("load_failed"))
        .mockResolvedValueOnce(page([10]))
      renderList()

      await screen.findByText("Conversation 30")
      fireEvent.click(screen.getByRole("button", { name: "widgetSession.history.loadMore" }))

      expect(await screen.findByRole("alert")).toHaveTextContent("widgetSession.history.loadFailed")
      expect(screen.getByText("Conversation 30")).toBeInTheDocument()
      fireEvent.click(screen.getByRole("button", { name: "widgetSession.history.retry" }))

      await screen.findByText("Conversation 10")
      expect(history.list.mock.calls[2][2]).toEqual({ cursor: "cursor-2" })
      expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    })

    it("asks for the auth again for every request", async () => {
      history.list.mockResolvedValueOnce(page([30], "cursor-2")).mockResolvedValueOnce(page([10]))
      const first = { getToken: vi.fn(async () => "one") }
      const second = { getToken: vi.fn(async () => "two") }
      const getAuth = vi.fn().mockReturnValueOnce(first).mockReturnValue(second)
      renderList({ getAuth })

      await screen.findByText("Conversation 30")
      fireEvent.click(screen.getByRole("button", { name: "widgetSession.history.loadMore" }))
      await screen.findByText("Conversation 10")

      expect(history.list.mock.calls[0][1]).toBe(first)
      expect(history.list.mock.calls[1][1]).toBe(second)
    })
  })

  describe("responses that arrive out of order", () => {
    it("cancels the earlier request when the visitor retries", async () => {
      const first = deferred<ReturnType<typeof page>>()
      const second = deferred<ReturnType<typeof page>>()
      const third = deferred<ReturnType<typeof page>>()
      history.list
        .mockReturnValueOnce(first.promise)
        .mockReturnValueOnce(second.promise)
        .mockReturnValueOnce(third.promise)
      renderList()

      await act(async () => first.reject(new WidgetSessionHistoryError("load_failed")))
      fireEvent.click(await screen.findByRole("button", { name: "widgetSession.history.retry" }))
      await act(async () => second.reject(new WidgetSessionHistoryError("load_failed")))
      fireEvent.click(await screen.findByRole("button", { name: "widgetSession.history.retry" }))
      await act(async () => third.resolve(page([3])))
      await screen.findByText("Conversation 3")

      expect(history.list).toHaveBeenCalledTimes(3)
      // The request a newer one replaced is cancelled.
      expect((history.list.mock.calls[1][3] as AbortSignal).aborted).toBe(true)
      expect((history.list.mock.calls[2][3] as AbortSignal).aborted).toBe(false)
    })

    it("does not report a cancelled request as a failure when the list mounts twice", async () => {
      const second = deferred<ReturnType<typeof page>>()
      history.list
        .mockImplementationOnce(
          (_origin: string, _auth: unknown, _page: unknown, signal: AbortSignal) =>
            new Promise((_resolve, reject) => {
              signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))
            }),
        )
        .mockReturnValueOnce(second.promise)
      render(
        <React.StrictMode>
          <WidgetHistoryList
            origin={ORIGIN}
            getAuth={() => AUTH}
            unauthorizedNotice="tryLater"
            currentTaskId={null}
            {...handlers()}
          />
        </React.StrictMode>,
      )

      await act(async () => undefined)
      expect(history.list).toHaveBeenCalledTimes(2)
      expect(screen.queryByText("widgetSession.history.loadFailed")).not.toBeInTheDocument()
      await act(async () => second.resolve(page([7])))
      await screen.findByText("Conversation 7")
      expect(screen.queryByText("widgetSession.history.loadFailed")).not.toBeInTheDocument()
    })

    it("neither writes nor complains when a response arrives after it closed", async () => {
      const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined)
      const first = deferred<ReturnType<typeof page>>()
      history.list.mockReturnValue(first.promise)
      const { unmount } = renderList()

      const signal = history.list.mock.calls[0][3] as AbortSignal
      unmount()
      expect(signal.aborted).toBe(true)
      await act(async () => first.resolve(page([1])))
      await act(async () => undefined)

      expect(consoleError).not.toHaveBeenCalled()
    })

    it("neither writes nor complains when a failure arrives after it closed", async () => {
      const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined)
      const first = deferred<ReturnType<typeof page>>()
      history.list.mockReturnValue(first.promise)
      const { unmount } = renderList()

      unmount()
      await act(async () => first.reject(new WidgetSessionHistoryError("load_failed")))
      await act(async () => undefined)

      expect(consoleError).not.toHaveBeenCalled()
    })
  })

  it("shows the retried page once a failed next page is requested again", async () => {
    const retry = deferred<ReturnType<typeof page>>()
    history.list
      .mockResolvedValueOnce(page([30], "cursor-2"))
      .mockRejectedValueOnce(new WidgetSessionHistoryError("load_failed"))
      .mockReturnValueOnce(retry.promise)
    renderList()

    await screen.findByText("Conversation 30")
    fireEvent.click(screen.getByRole("button", { name: "widgetSession.history.loadMore" }))
    fireEvent.click(await screen.findByRole("button", { name: "widgetSession.history.retry" }))
    expect(screen.getByRole("button", { name: "widgetSession.history.loadMore" })).toBeDisabled()
    await act(async () => retry.resolve(page([10])))

    expect(await screen.findByText("Conversation 10")).toBeInTheDocument()
  })
})

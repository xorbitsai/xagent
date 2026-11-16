import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastMock = vi.hoisted(() => ({ error: vi.fn(), success: vi.fn() }))

vi.mock("sonner", () => ({ toast: toastMock }))

// TraceEventRenderer (reused for tool-call display) pulls in next/navigation
// useRouter() and the app-context useApp(); both need providers that aren't
// mounted when rendering in isolation under jsdom, so stub them.
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    openFilePreview: vi.fn(),
    dispatch: vi.fn(),
  }),
}))

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper"
  )
  return {
    ...actual,
    apiRequest: apiRequestMock,
  }
})

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return {
    ...actual,
    getApiUrl: () => "http://api.local",
  }
})

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    locale: "zh",
    setLocale: vi.fn(),
    t: (key: string, vars?: Record<string, string | number>) => {
      // TraceEventRenderer labels the tool button via t(key, { tool }); mirror
      // the interpolation so the tool name is assertable.
      if (vars?.tool) return `${key}:${vars.tool}`
      // The expired-detail and expired-traceEvents notes interpolate a
      // formatted date via t(key, { date }); mirror it so the rendered date
      // string is assertable instead of being swallowed by the key fallback.
      if (vars?.date) return `${key}:${vars.date}`
      // The export limit messages interpolate the limit; mirror it so the
      // number is assertable.
      if (vars?.limit !== undefined) return `${key}:${vars.limit}`
      if (vars?.limitMb !== undefined) return `${key}:${vars.limitMb}`
      const labels: Record<string, string> = {
        "conversationLogs.title": "Conversation Logs",
        "conversationLogs.searchPlaceholder": "Search conversations",
        "conversationLogs.agentFilterAll": "All agents",
        "conversationLogs.readOnly": "Read-only",
        "conversationLogs.empty.title": "No conversation logs yet",
        "conversationLogs.empty.filteredTitle": "No matching logs",
        "conversationLogs.sources.all": "All",
        "conversationLogs.sources.widget": "Widget",
        "conversationLogs.sources.restApi": "REST API",
        "conversationLogs.sources.sharedLink": "Shareable Link",
        "conversationLogs.sources.webhook": "Webhook",
      }
      return labels[key] || key
    },
  }),
}))

import { ConversationLogsPage } from "./conversation-logs"

const NativeRelativeTimeFormat = Intl.RelativeTimeFormat

const listPayload = {
  logs: [
    {
      task_id: 101,
      title: "REST lead intake",
      description: "Lead from REST",
      status: "completed",
      source: "rest_api",
      source_label: "REST API",
      agent_id: 7,
      agent_name: "Sales Agent",
      created_at: "2026-06-29T01:00:00Z",
      updated_at: "2026-06-29T01:01:00Z",
      last_activity_at: "2026-06-29T01:01:00Z",
      total_tokens: 12,
      message_count: 2,
    },
    {
      task_id: 202,
      title: "Webhook CRM event",
      description: "Webhook payload",
      status: "completed",
      source: "webhook",
      source_label: "Webhook",
      agent_id: 7,
      agent_name: "Sales Agent",
      created_at: "2026-06-29T02:00:00Z",
      updated_at: "2026-06-29T02:01:00Z",
      last_activity_at: "2026-06-29T02:01:00Z",
      total_tokens: 20,
      message_count: 2,
    },
  ],
  source_counts: {
    all: 2,
    widget: 0,
    rest_api: 1,
    shared_link: 0,
    webhook: 1,
  },
  agents: [
    {
      agent_id: 7,
      agent_name: "Sales Agent",
      agent_logo_url: null,
    },
  ],
  pagination: {
    page: 1,
    per_page: 20,
    total: 2,
    total_pages: 1,
  },
}

const detailPayload = {
  log: listPayload.logs[0],
  transcript: [
    {
      id: 1,
      role: "user",
      content: "Qualify this lead",
      message_type: "chat",
      created_at: "2026-06-29T01:00:10Z",
    },
    {
      id: 2,
      role: "assistant",
      content: "**Lead** qualified",
      message_type: "chat",
      created_at: "2026-06-29T01:00:20Z",
    },
  ],
  metadata: {
    task: {
      task_id: 101,
      input: "Qualify this lead",
      output: "Lead qualified",
      error_message: null,
      description: "Lead from REST",
      agent_config: {},
    },
    trigger: null,
    public_context: null,
  },
  trace_events: [
    {
      event_id: "step-1",
      event_type: "react_task_start",
      step_id: "step-1",
      timestamp: 1750000000,
      data: { step_name: "Search", description: "Search" },
    },
    {
      event_id: "tool-start",
      event_type: "tool_execution_start",
      step_id: "step-1",
      timestamp: 1750000001,
      data: { tool_name: "web_search", tool_args: { q: "x" } },
    },
    {
      event_id: "tool-end",
      event_type: "tool_execution_end",
      step_id: "step-1",
      timestamp: 1750000002,
      data: { result: { success: true } },
    },
  ],
  read_only: true,
}

const webhookDetailPayload = {
  ...detailPayload,
  log: listPayload.logs[1],
  transcript: [
    {
      id: 3,
      role: "user",
      content: "Handle webhook event",
      message_type: "chat",
      created_at: "2026-06-29T02:00:10Z",
    },
  ],
  metadata: {
    task: {
      task_id: 202,
      input: "Handle webhook event",
      output: "Webhook handled",
      error_message: null,
      description: "Webhook payload",
      agent_config: {},
    },
    trigger: null,
    public_context: null,
  },
}

describe("ConversationLogsPage", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastMock.error.mockReset()
    apiRequestMock.mockImplementation((url: string) => {
      const parsed = new URL(url)
      if (parsed.pathname === "/api/conversation-logs/101") {
        return Promise.resolve(
          new Response(JSON.stringify(detailPayload), { status: 200 })
        )
      }
      if (parsed.pathname === "/api/conversation-logs") {
        return Promise.resolve(
          new Response(JSON.stringify(listPayload), { status: 200 })
        )
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it("loads logs, source tabs, and a read-only selected transcript", async () => {
    render(<ConversationLogsPage />)

    expect(await screen.findByText("Conversation Logs")).toBeInTheDocument()
    expect((await screen.findAllByText("Sales Agent")).length).toBeGreaterThan(0)
    expect(screen.getByText("All 2")).toBeInTheDocument()
    expect(screen.getByText("REST API 1")).toBeInTheDocument()
    expect(screen.getByText("Webhook 1")).toBeInTheDocument()
    expect(await screen.findByText("Qualify this lead")).toBeInTheDocument()
    expect(await screen.findByText("Lead", { selector: "strong" })).toBeInTheDocument()
    expect(screen.getByText("Read-only")).toBeInTheDocument()
    expect(screen.queryByText("Upload")).not.toBeInTheDocument()
    expect(screen.queryByText("Send")).not.toBeInTheDocument()
  })

  it("renders assistant markdown as formatted html", async () => {
    render(<ConversationLogsPage />)
    const strong = await screen.findByText("Lead", { selector: "strong" })
    expect(strong).toBeInTheDocument()
  })

  it("renders tool calls from trace events", async () => {
    render(<ConversationLogsPage />)
    // A completed task's trace is collapsed behind a toggle; expand it first.
    fireEvent.click(await screen.findByRole("button", { name: /showProcess/ }))
    expect(
      await screen.findByRole("button", { name: /Web Search/ })
    ).toBeInTheDocument()
  })

  it("formats relative times with the app locale", async () => {
    const relativeTimeFormatMock = vi.fn(
      function (locale?: Intl.LocalesArgument, options?: Intl.RelativeTimeFormatOptions) {
        return new NativeRelativeTimeFormat(locale, options)
      }
    )
    vi.stubGlobal("Intl", {
      ...Intl,
      RelativeTimeFormat: relativeTimeFormatMock,
    })

    render(<ConversationLogsPage />)

    await screen.findByText("Conversation Logs")
    await waitFor(() => {
      expect(relativeTimeFormatMock).toHaveBeenCalled()
    })
    expect(relativeTimeFormatMock.mock.calls.every(([locale]) => locale === "zh")).toBe(
      true
    )
  })

  it("sends source tab and search state to the list API", async () => {
    render(<ConversationLogsPage />)

    await screen.findAllByText("Sales Agent")
    fireEvent.click(screen.getByText("Webhook 1"))
    fireEvent.change(screen.getByPlaceholderText("Search conversations"), {
      target: { value: "crm" },
    })

    await waitFor(() => {
      const listUrls = apiRequestMock.mock.calls
        .map(([url]) => String(url))
        .filter((url) => new URL(url).pathname === "/api/conversation-logs")
      expect(listUrls.some((url) => url.includes("source=webhook"))).toBe(true)
      expect(listUrls.some((url) => url.includes("search=crm"))).toBe(true)
    })
  })

  it("ignores stale detail responses after a newer log is selected", async () => {
    let resolveFirstDetail: (response: Response) => void = () => {}
    const firstDetailPromise = new Promise<Response>((resolve) => {
      resolveFirstDetail = resolve
    })

    apiRequestMock.mockImplementation((url: string) => {
      const parsed = new URL(url)
      if (parsed.pathname === "/api/conversation-logs/101") {
        return firstDetailPromise
      }
      if (parsed.pathname === "/api/conversation-logs/202") {
        return Promise.resolve(
          new Response(JSON.stringify(webhookDetailPayload), { status: 200 })
        )
      }
      if (parsed.pathname === "/api/conversation-logs") {
        return Promise.resolve(
          new Response(JSON.stringify(listPayload), { status: 200 })
        )
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    render(<ConversationLogsPage />)

    await screen.findAllByText("Sales Agent")
    fireEvent.click(screen.getByText("Webhook"))
    expect(await screen.findByText("Handle webhook event")).toBeInTheDocument()

    await act(async () => {
      resolveFirstDetail(new Response(JSON.stringify(detailPayload), { status: 200 }))
      await Promise.resolve()
    })

    expect(screen.getByText("Handle webhook event")).toBeInTheDocument()
    expect(screen.queryByText("Qualify this lead")).not.toBeInTheDocument()
  })

  function mockDetailResponse(detail: () => Response) {
    apiRequestMock.mockImplementation((url: string) => {
      const parsed = new URL(url)
      if (parsed.pathname === "/api/conversation-logs/101") {
        return Promise.resolve(detail())
      }
      if (parsed.pathname === "/api/conversation-logs") {
        return Promise.resolve(
          new Response(JSON.stringify(listPayload), { status: 200 })
        )
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })
  }

  it("says a log expired when retention removed it after it was listed", async () => {
    const expiredAt = "2026-09-01T12:00:00+00:00"
    mockDetailResponse(
      () =>
        new Response(
          JSON.stringify({
            detail: {
              code: "task_expired",
              message: "Conversation log expired under the retention policy",
              task_id: 101,
              expired_at: expiredAt,
            },
          }),
          { status: 410 }
        )
    )

    render(<ConversationLogsPage />)

    const expectedDate = new Date(expiredAt).toLocaleDateString()
    expect(
      await screen.findByText(`conversationLogs.expired.detail:${expectedDate}`)
    ).toBeInTheDocument()
    expect(screen.queryByText("Failed to load conversation detail")).not.toBeInTheDocument()
  })

  it("clears the expired note once a different log is selected", async () => {
    const expiredAt = "2026-09-01T12:00:00+00:00"
    apiRequestMock.mockImplementation((url: string) => {
      const parsed = new URL(url)
      if (parsed.pathname === "/api/conversation-logs/101") {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              detail: {
                code: "task_expired",
                message: "Conversation log expired under the retention policy",
                task_id: 101,
                expired_at: expiredAt,
              },
            }),
            { status: 410 }
          )
        )
      }
      if (parsed.pathname === "/api/conversation-logs/202") {
        return Promise.resolve(
          new Response(JSON.stringify(webhookDetailPayload), { status: 200 })
        )
      }
      if (parsed.pathname === "/api/conversation-logs") {
        return Promise.resolve(
          new Response(JSON.stringify(listPayload), { status: 200 })
        )
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    render(<ConversationLogsPage />)

    const expectedDate = new Date(expiredAt).toLocaleDateString()
    expect(
      await screen.findByText(`conversationLogs.expired.detail:${expectedDate}`)
    ).toBeInTheDocument()

    // Copies the selection pattern from "ignores stale detail responses
    // after a newer log is selected": pick the other listed log (source
    // "webhook", task 202) and confirm its detail replaces the expired note
    // instead of the note lingering from the previous selection.
    fireEvent.click(screen.getByText("Webhook"))

    expect(await screen.findByText("Handle webhook event")).toBeInTheDocument()
    expect(
      screen.queryByText(`conversationLogs.expired.detail:${expectedDate}`)
    ).not.toBeInTheDocument()
  })

  it("treats a 410 without the task_expired code as an ordinary failure", async () => {
    mockDetailResponse(
      () => new Response(JSON.stringify({ detail: "Gone" }), { status: 410 })
    )

    render(<ConversationLogsPage />)

    expect(await screen.findByText("Failed to load conversation detail")).toBeInTheDocument()
    expect(screen.queryByText("conversationLogs.expired.detail")).not.toBeInTheDocument()
  })

  it("notes when retention removed earlier execution steps", async () => {
    const expiredAt = "2026-09-01T12:00:00+00:00"
    mockDetailResponse(
      () =>
        new Response(
          JSON.stringify({
            ...detailPayload,
            trace_events: [],
            trace_events_expired_at: expiredAt,
          }),
          { status: 200 }
        )
    )

    render(<ConversationLogsPage />)

    const expectedDate = new Date(expiredAt).toLocaleDateString()
    expect(await screen.findByText("Qualify this lead")).toBeInTheDocument()
    expect(
      screen.getByText(`conversationLogs.expired.traceEvents:${expectedDate}`)
    ).toBeInTheDocument()
  })

  it("shows no retention note for a complete trace", async () => {
    render(<ConversationLogsPage />)

    expect(await screen.findByText("Qualify this lead")).toBeInTheDocument()
    expect(screen.queryByText("conversationLogs.expired.traceEvents")).not.toBeInTheDocument()
  })

  describe("time window and export", () => {
    const FROM_LOCAL = "2026-06-29T08:00"
    const TO_LOCAL = "2026-06-30T08:00"

    function listUrls(): URL[] {
      return apiRequestMock.mock.calls
        .map(([url]) => new URL(String(url)))
        .filter((url) => url.pathname === "/api/conversation-logs")
    }

    function exportUrls(): URL[] {
      return apiRequestMock.mock.calls
        .map(([url]) => new URL(String(url)))
        .filter((url) => url.pathname === "/api/conversation-logs/export")
    }

    function mockExport(respond: () => Response) {
      const listImplementation = apiRequestMock.getMockImplementation()
      apiRequestMock.mockImplementation((url: string) => {
        if (new URL(url).pathname === "/api/conversation-logs/export") {
          return Promise.resolve(respond())
        }
        return listImplementation?.(url)
      })
    }

    function setWindow(from: string, to: string) {
      fireEvent.change(screen.getByLabelText("conversationLogs.activeFromLabel"), {
        target: { value: from },
      })
      fireEvent.change(screen.getByLabelText("conversationLogs.activeToLabel"), {
        target: { value: to },
      })
    }

    function exportButton() {
      return screen.getByRole("button", { name: /^conversationLogs\.export(ing)?$/ })
    }

    it("sends the chosen window to the list API as UTC ISO timestamps", async () => {
      render(<ConversationLogsPage />)
      await screen.findAllByText("Sales Agent")

      setWindow(FROM_LOCAL, TO_LOCAL)

      await waitFor(() => {
        const last = listUrls().at(-1)!
        expect(last.searchParams.get("active_from")).toBe(
          new Date(FROM_LOCAL).toISOString()
        )
        expect(last.searchParams.get("active_to")).toBe(new Date(TO_LOCAL).toISOString())
      })
      expect(listUrls()[0].searchParams.has("active_from")).toBe(false)
    })

    it("drops a time parameter once its input is cleared", async () => {
      render(<ConversationLogsPage />)
      await screen.findAllByText("Sales Agent")

      setWindow(FROM_LOCAL, TO_LOCAL)
      await waitFor(() => expect(listUrls().at(-1)!.searchParams.has("active_from")).toBe(true))
      fireEvent.click(
        screen.getByRole("button", {
          name: "conversationLogs.clearTime - conversationLogs.activeFromLabel",
        })
      )

      await waitFor(() => {
        const last = listUrls().at(-1)!
        expect(last.searchParams.has("active_from")).toBe(false)
        expect(last.searchParams.get("active_to")).toBe(new Date(TO_LOCAL).toISOString())
      })
    })

    it("does not request or allow export when the start is not before the end", async () => {
      render(<ConversationLogsPage />)
      await screen.findAllByText("Sales Agent")
      expect(exportButton()).toBeEnabled()
      const requestsBefore = listUrls().length

      setWindow(TO_LOCAL, FROM_LOCAL)

      expect(await screen.findByRole("alert")).toHaveTextContent(
        "conversationLogs.timeRangeInvalid"
      )
      expect(exportButton()).toBeDisabled()
      await act(async () => {
        await Promise.resolve()
      })
      const afterInvalid = listUrls().slice(requestsBefore)
      expect(afterInvalid.some((url) => url.searchParams.has("active_from") &&
        url.searchParams.has("active_to"))).toBe(false)
      // Equal bounds are the same empty window.
      setWindow(FROM_LOCAL, FROM_LOCAL)
      expect(await screen.findByRole("alert")).toBeInTheDocument()
      expect(exportButton()).toBeDisabled()
    })

    it("disables export when the result is empty", async () => {
      apiRequestMock.mockImplementation((url: string) => {
        if (new URL(url).pathname === "/api/conversation-logs") {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                ...listPayload,
                logs: [],
                pagination: { ...listPayload.pagination, total: 0 },
              }),
              { status: 200 }
            )
          )
        }
        throw new Error(`Unhandled apiRequest: ${url}`)
      })
      render(<ConversationLogsPage />)

      await screen.findByText("No conversation logs yet")
      expect(exportButton()).toBeDisabled()
    })

    it("exports with exactly the filters of the current list request", async () => {
      const createObjectURL = vi.fn(() => "blob:test")
      const revokeObjectURL = vi.fn()
      vi.stubGlobal("URL", Object.assign(URL, { createObjectURL, revokeObjectURL }))
      const downloads: string[] = []
      const clickSpy = vi
        .spyOn(HTMLAnchorElement.prototype, "click")
        .mockImplementation(function (this: HTMLAnchorElement) {
          downloads.push(this.download)
        })
      mockExport(() => new Response("a,b\r\n", { status: 200 }))

      render(<ConversationLogsPage />)
      await screen.findAllByText("Sales Agent")
      fireEvent.click(screen.getByText("Webhook 1"))
      fireEvent.change(screen.getByPlaceholderText("Search conversations"), {
        target: { value: "crm" },
      })
      setWindow(FROM_LOCAL, TO_LOCAL)
      await waitFor(() => {
        const last = listUrls().at(-1)!
        expect(last.searchParams.get("source")).toBe("webhook")
        expect(last.searchParams.get("search")).toBe("crm")
        expect(last.searchParams.has("active_to")).toBe(true)
      })
      const currentList = listUrls().at(-1)!

      fireEvent.click(exportButton())

      await waitFor(() => expect(exportUrls()).toHaveLength(1))
      const listFilters = new URLSearchParams(currentList.searchParams)
      listFilters.delete("page")
      listFilters.delete("per_page")
      expect(Array.from(exportUrls()[0].searchParams.entries()).sort()).toEqual(
        Array.from(listFilters.entries()).sort()
      )
      await waitFor(() => expect(downloads).toHaveLength(1))
      expect(downloads[0]).toMatch(/^conversation-logs-\d{8}-\d{6}\.csv$/)
      expect(toastMock.error).not.toHaveBeenCalled()
      clickSpy.mockRestore()
    })

    it.each([
      [
        "too many rows",
        422,
        { detail: { code: "conversation_log_export_too_large", reason: "rows", limit: 10000 } },
        "conversationLogs.exportTooManyRows:10000",
      ],
      [
        "too large a file",
        422,
        {
          detail: {
            code: "conversation_log_export_too_large",
            reason: "bytes",
            limit: 52428800,
          },
        },
        "conversationLogs.exportTooLargeFile:50",
      ],
      [
        "an unknown reason",
        422,
        { detail: { code: "conversation_log_export_too_large", reason: "disk", limit: 5 } },
        "conversationLogs.exportFailed",
      ],
      [
        "a non-numeric limit",
        422,
        { detail: { code: "conversation_log_export_too_large", reason: "rows", limit: "x" } },
        "conversationLogs.exportFailed",
      ],
      [
        "another 422",
        422,
        { detail: [{ msg: "bad", loc: ["query", "active_from"] }] },
        "conversationLogs.exportFailed",
      ],
      ["a server error", 500, { detail: "boom" }, "conversationLogs.exportFailed"],
    ])("reports %s on export", async (_name, status, body, expectedMessage) => {
      mockExport(() => new Response(JSON.stringify(body), { status }))
      render(<ConversationLogsPage />)
      await screen.findAllByText("Sales Agent")

      fireEvent.click(exportButton())

      await waitFor(() => expect(toastMock.error).toHaveBeenCalledWith(expectedMessage))
      expect(toastMock.error).toHaveBeenCalledTimes(1)
      await waitFor(() => expect(exportButton()).toBeEnabled())
    })
  })
})

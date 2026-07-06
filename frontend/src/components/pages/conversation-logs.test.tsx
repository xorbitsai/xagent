import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())

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
      await screen.findByRole("button", { name: /web_search/ })
    ).toBeInTheDocument()
  })

  it("formats relative times with the app locale", async () => {
    const relativeTimeFormatMock = vi.fn(
      (locale?: Intl.LocalesArgument, options?: Intl.RelativeTimeFormatOptions) =>
        new NativeRelativeTimeFormat(locale, options)
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
})

import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { AppProviderTransportConfig } from "@/contexts/app-context-chat"

const app = vi.hoisted(() => ({
  dispatch: vi.fn(),
  sendMessage: vi.fn(),
  setTaskId: vi.fn(),
  state: {
    taskId: null as number | null,
    currentTask: null,
    isHistoryLoading: false,
    isProcessing: false,
  },
  rerender: null as null | React.Dispatch<React.SetStateAction<number>>,
  provider: null as null | {
    token: string
    transport?: AppProviderTransportConfig
  },
}))

const i18n = vi.hoisted(() => ({ t: (key: string) => key }))

vi.mock("@/contexts/app-context-chat", () => ({
  AppProvider: ({
    children,
    token,
    transport,
  }: {
    children: React.ReactNode
    token: string
    transport?: AppProviderTransportConfig
  }) => {
    app.provider = { token, transport }
    return <>{children}</>
  },
  useApp: () => {
    const [, rerender] = React.useState(0)
    app.rerender = rerender
    return {
      state: app.state,
      dispatch: app.dispatch,
      sendMessage: app.sendMessage,
      setTaskId: app.setTaskId,
    }
  },
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => i18n,
}))

vi.mock("@/components/chat/ChatStartScreen", () => ({
  ChatStartScreen: ({ onSend, title }: { onSend: (message: string, files: File[], config?: Record<string, string>) => Promise<void>; title: string }) => (
    <button type="button" onClick={() => onSend("first message", [], { mode: "balanced" })}>
      start:{title}
    </button>
  ),
}))

vi.mock("@/components/task/task-conversation-panel", () => ({
  TaskConversationPanel: () => <div data-testid="conversation-panel" />,
}))

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return {
    ...actual,
    getApiUrl: () => "https://api.example",
  }
})

import { PublicAgentChatPage } from "./public-agent-chat-page"

const successfulAgentAuth = {
  access_token: "public-access-token",
  token_type: "bearer",
  agent_id: 17,
  agent_name: "Support Agent",
  agent_logo: null,
  agent_description: "Answers questions",
  suggested_prompts: [],
  workforce_id: null,
}

const successfulWorkforceAuth = {
  access_token: "public-access-token",
  token_type: "bearer",
  agent_id: null,
  agent_name: "Support Workforce",
  agent_logo: null,
  agent_description: null,
  suggested_prompts: [],
  workforce_id: 8,
}

function jsonResponse(payload: unknown, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function renderWidgetPage(overrides: Partial<React.ComponentProps<typeof PublicAgentChatPage>> = {}) {
  return render(
    <PublicAgentChatPage
      authMode="widget"
      routeToken="default"
      guestId="guest-1"
      searchAgentId={17}
      {...overrides}
    />,
  )
}

async function expectWidgetAuthFailure(detail: string) {
  expect(await screen.findByText(detail)).toBeInTheDocument()
  expect(screen.queryByRole("button", { name: /start:/ })).toBeNull()
  expect(sessionStorage.getItem("xagent_public_access_token")).toBeNull()
  expect(app.setTaskId).not.toHaveBeenCalled()
}

function expectPublicProviderToken() {
  expect(sessionStorage.getItem("xagent_public_access_token")).toBe("public-access-token")
  expect(app.provider).toMatchObject({ token: "public-access-token" })
  expect(app.provider?.transport?.buildWebSocketUrl?.({
    baseUrl: "wss://api.example",
    taskId: 42,
    token: app.provider?.token,
  })).toBe("wss://api.example/api/widget/chat/ws/42?token=public-access-token")
}

function seedStalePublicToken() {
  sessionStorage.setItem("xagent_public_access_token", "stale-public-token")
}

function widgetTaskResponse(taskId: number, status: "pending" | "running") {
  return {
    task_id: taskId,
    title: "first message",
    status,
    created_at: "2026-07-24T00:00:00Z",
    model_id: null,
    small_fast_model_id: null,
    visual_model_id: null,
    compact_model_id: null,
    model_name: null,
    small_fast_model_name: null,
    visual_model_name: null,
    compact_model_name: null,
    execution_mode: null,
    channel_id: null,
    channel_name: "Web Widget",
    agent_id: null,
    agent_name: null,
    agent_logo_url: null,
    run_id: null,
    state_version: 0,
    control_state: "idle",
  }
}

describe("PublicAgentChatPage", () => {
  const fetchMock = vi.fn()

  beforeEach(() => {
    app.dispatch.mockReset()
    app.sendMessage.mockReset()
    app.setTaskId.mockImplementation((taskId: number | null) => {
      app.state = { ...app.state, taskId }
      app.rerender?.((value) => value + 1)
    })
    app.setTaskId.mockClear()
    app.state = {
      taskId: null,
      currentTask: null,
      isHistoryLoading: false,
      isProcessing: false,
    }
    app.rerender = null
    app.provider = null
    sessionStorage.clear()
    fetchMock.mockReset()
    vi.stubGlobal("fetch", fetchMock)
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it("authenticates embedded widgets with the ticket and never sends the widget key", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(successfulAgentAuth))

    renderWidgetPage({ embedTicket: "embed-ticket", widgetKey: "widget-secret" })

    await screen.findByRole("button", { name: "start:Support Agent" })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "https://api.example/api/widget/auth",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          guest_id: "guest-1",
          agent_id: 17,
          embed_ticket: "embed-ticket",
        }),
      },
    )
    expectPublicProviderToken()
  })

  it("authenticates direct widget visits with their widget key", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(successfulAgentAuth))

    renderWidgetPage({ widgetKey: "widget-secret" })

    await screen.findByRole("button", { name: "start:Support Agent" })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "https://api.example/api/widget/auth",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          guest_id: "guest-1",
          agent_id: 17,
          widget_key: "widget-secret",
        }),
      },
    )
    expectPublicProviderToken()
  })

  it("fails closed for an invalid direct widget key", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: "Invalid widget key" }, 403))
    seedStalePublicToken()

    renderWidgetPage({ widgetKey: "wk-not-a-real-key" })

    await expectWidgetAuthFailure("Invalid widget key")
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "https://api.example/api/widget/auth",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          guest_id: "guest-1",
          agent_id: 17,
          widget_key: "wk-not-a-real-key",
        }),
      },
    )
    expect(errorSpy).toHaveBeenCalledWith(expect.objectContaining({ message: "Invalid widget key" }))
  })

  it("fails closed when an embed ticket domain is no longer allowed", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockResolvedValueOnce(jsonResponse({
      detail: "Domain not allowed: embedded.example",
    }, 403))
    seedStalePublicToken()

    renderWidgetPage({ embedTicket: "domain-bound-ticket", widgetKey: "widget-secret" })

    await expectWidgetAuthFailure("Domain not allowed: embedded.example")
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "https://api.example/api/widget/auth",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          guest_id: "guest-1",
          agent_id: 17,
          embed_ticket: "domain-bound-ticket",
        }),
      },
    )
    expect(errorSpy).toHaveBeenCalledWith(expect.objectContaining({
      message: "Domain not allowed: embedded.example",
    }))
  })

  it("fails closed when a widget is disabled after its embed ticket was issued", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockResolvedValueOnce(jsonResponse({
      detail: "Widget is disabled for this agent",
    }, 403))
    seedStalePublicToken()

    renderWidgetPage({
      embedTicket: "ticket-issued-before-disable",
      widgetKey: "widget-secret",
    })

    await expectWidgetAuthFailure("Widget is disabled for this agent")
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "https://api.example/api/widget/auth",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          guest_id: "guest-1",
          agent_id: 17,
          embed_ticket: "ticket-issued-before-disable",
        }),
      },
    )
    expect(errorSpy).toHaveBeenCalledWith(expect.objectContaining({
      message: "Widget is disabled for this agent",
    }))
  })

  it("resumes the saved task instead of showing a new-chat screen", async () => {
    localStorage.setItem("widget_task_17_guest-1", "71")
    fetchMock.mockResolvedValueOnce(jsonResponse(successfulAgentAuth))

    renderWidgetPage()

    expect(await screen.findByTestId("conversation-panel")).toBeInTheDocument()
    expect(app.setTaskId).toHaveBeenCalledWith(71, { navigate: false })
  })

  it("shows the start screen and defers task creation until the first agent message", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(successfulAgentAuth))

    renderWidgetPage()

    expect(await screen.findByRole("button", { name: "start:Support Agent" })).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("creates an agent task and then sends its opening message", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(successfulAgentAuth))
      .mockResolvedValueOnce(jsonResponse(widgetTaskResponse(42, "pending")))

    const { unmount } = renderWidgetPage({ widgetKey: "widget-secret" })

    fireEvent.click(await screen.findByRole("button", { name: "start:Support Agent" }))

    await waitFor(() => {
      expect(app.sendMessage).toHaveBeenCalledWith(
        "first message",
        { mode: "balanced", targetTaskId: 42 },
        [],
      )
    })
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "https://api.example/api/widget/auth",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          guest_id: "guest-1",
          agent_id: 17,
          widget_key: "widget-secret",
        }),
      },
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "https://api.example/api/widget/chat/task/create",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer public-access-token",
        },
        body: JSON.stringify({
          title: "first message",
          description: "first message",
          agent_id: 17,
        }),
      },
    )
    await waitFor(() => {
      expect(localStorage.getItem("widget_task_17_guest-1")).toBe("42")
    })

    unmount()
    app.state = {
      taskId: null,
      currentTask: null,
      isHistoryLoading: false,
      isProcessing: false,
    }
    app.setTaskId.mockClear()
    app.rerender = null
    fetchMock.mockReset()
    fetchMock.mockResolvedValueOnce(jsonResponse(successfulAgentAuth))

    renderWidgetPage({ widgetKey: "widget-secret" })

    expect(await screen.findByTestId("conversation-panel")).toBeInTheDocument()
    expect(app.setTaskId).toHaveBeenCalledWith(42, { navigate: false })
  })

  it("lets workforce task creation start the opening turn without sending it again", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(successfulWorkforceAuth))
      .mockResolvedValueOnce(jsonResponse(widgetTaskResponse(43, "running")))

    renderWidgetPage({ searchAgentId: null, widgetKey: "widget-secret" })

    fireEvent.click(await screen.findByRole("button", { name: "start:Support Workforce" }))

    await waitFor(() => {
      expect(app.setTaskId).toHaveBeenCalledWith(43, { navigate: false })
    })
    expect(app.sendMessage).not.toHaveBeenCalled()
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "https://api.example/api/widget/auth",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          guest_id: "guest-1",
          agent_id: null,
          widget_key: "widget-secret",
        }),
      },
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "https://api.example/api/widget/chat/task/create",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer public-access-token",
        },
        body: JSON.stringify({
          title: "first message",
          description: "first message",
        }),
      },
    )
    await waitFor(() => {
      expect(localStorage.getItem("widget_task_wf8_guest-1")).toBe("43")
    })
  })
})

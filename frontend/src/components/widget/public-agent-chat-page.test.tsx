import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { AppProviderTransportConfig } from "@/contexts/app-context-chat"

const app = vi.hoisted(() => ({
  dispatch: vi.fn(),
  sendMessage: vi.fn(),
  setTaskId: vi.fn(),
  connectionError: null as Error | null,
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
  startScreenProps: null as null | {
    voiceInputEnabled?: boolean
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
      connectionError: app.connectionError,
      voiceInputEnabled:
        app.provider?.transport?.capabilities?.voice !== "disabled",
    }
  },
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => i18n,
}))

vi.mock("@/components/chat/ChatStartScreen", () => ({
  ChatStartScreen: (props: {
    onSend: (message: string, files: File[], config?: Record<string, string>) => Promise<void>
    title: string
    voiceInputEnabled?: boolean
  }) => {
    app.startScreenProps = props
    return (
      <button
        type="button"
        onClick={() => {
          void props.onSend("first message", [], { mode: "balanced" }).catch(() => undefined)
        }}
      >
        start:{props.title}
      </button>
    )
  },
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

function renderSharePage(overrides: Partial<React.ComponentProps<typeof PublicAgentChatPage>> = {}) {
  return render(
    <PublicAgentChatPage
      authMode="share"
      routeToken="share-tok"
      guestId={null}
      searchAgentId={null}
      {...overrides}
    />,
  )
}

const SHARE_AUTH_KEY = "share_auth_share-tok"

// Build a JWT-shaped token whose (unverified) exp claim the client reads as a
// liveness pre-filter. base64url payload, matching isShareTokenExpired's decode.
function makeShareJwt(claims: Record<string, unknown>): string {
  const b64 = (obj: object) =>
    btoa(JSON.stringify(obj)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "")
  return `${b64({ alg: "none" })}.${b64(claims)}.sig`
}

async function expectWidgetAuthFailure(detail: string) {
  expect(await screen.findByText(detail)).toBeInTheDocument()
  expect(screen.queryByRole("button", { name: /start:/ })).toBeNull()
  expect(sessionStorage.getItem("xagent_public_access_token")).toBeNull()
  expect(app.setTaskId).not.toHaveBeenCalled()
}

function expectPublicProviderToken() {
  expect(sessionStorage.getItem("xagent_public_access_token")).toBeNull()
  expect(app.provider).toMatchObject({ token: "public-access-token" })
  expect(app.provider?.transport?.capabilities).toEqual({
    agentCards: "disabled",
    voice: "disabled",
  })
  expect(app.startScreenProps?.voiceInputEnabled).toBe(false)
  expect(app.provider?.transport?.buildWebSocketUrl?.({
    baseUrl: "wss://api.example",
    taskId: 42,
    token: app.provider?.token,
  })).toBe("wss://api.example/api/widget/chat/ws/42?token=public-access-token")
  expect(app.provider?.transport?.fileAccess?.inlinePreviewUrl("public-file")).toBe(
    "https://api.example/api/files/public/preview/public-file?token=public-access-token",
  )
  expect(app.provider?.transport?.fileAccess?.inlineDownloadUrl("public-file")).toBe(
    "https://api.example/api/files/public/download/public-file?token=public-access-token",
  )
  expect(app.provider?.transport?.uploadFiles).toEqual(expect.any(Function))
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
    app.connectionError = null
    app.state = {
      taskId: null,
      currentTask: null,
      isHistoryLoading: false,
      isProcessing: false,
    }
    app.rerender = null
    app.provider = null
    app.startScreenProps = null
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

  it("authenticates a share link and persists the guest token for reuse", async () => {
    localStorage.clear()
    fetchMock.mockResolvedValueOnce(jsonResponse(successfulAgentAuth))

    renderSharePage()

    await screen.findByRole("button", { name: "start:Support Agent" })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "https://api.example/api/share/auth",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ share_token: "share-tok" }),
      },
    )
    // The server-minted guest token is persisted per link so a reload reuses
    // the same guest_id instead of minting a fresh (task-orphaning) session.
    expect(JSON.parse(localStorage.getItem(SHARE_AUTH_KEY) ?? "null")).toMatchObject({
      access_token: "public-access-token",
    })
    expect(app.provider).toMatchObject({ token: "public-access-token" })
    expect(app.provider?.transport?.buildWebSocketUrl?.({
      baseUrl: "wss://api.example",
      taskId: 42,
      token: app.provider?.token,
    })).toBe("wss://api.example/api/share/chat/ws/42?token=public-access-token")
  })

  it("reuses a persisted, unexpired share token without re-authing", async () => {
    localStorage.clear()
    localStorage.setItem(SHARE_AUTH_KEY, JSON.stringify(successfulAgentAuth))

    renderSharePage()

    await screen.findByRole("button", { name: "start:Support Agent" })
    expect(fetchMock).not.toHaveBeenCalled()
    expect(app.provider).toMatchObject({ token: "public-access-token" })
  })

  it("drops an expired persisted share token and re-authenticates", async () => {
    localStorage.clear()
    const expired = makeShareJwt({ exp: Math.floor(Date.now() / 1000) - 60 })
    localStorage.setItem(
      SHARE_AUTH_KEY,
      JSON.stringify({ ...successfulAgentAuth, access_token: expired }),
    )
    fetchMock.mockResolvedValueOnce(jsonResponse(successfulAgentAuth))

    renderSharePage()

    await screen.findByRole("button", { name: "start:Support Agent" })
    // Expired token ignored -> a fresh /api/share/auth is issued.
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "https://api.example/api/share/auth",
      expect.objectContaining({ method: "POST" }),
    )
  })

  it("ignores a corrupt persisted share auth blob and re-authenticates", async () => {
    localStorage.clear()
    // No access_token -> fails the shape guard -> clean re-auth.
    localStorage.setItem(SHARE_AUTH_KEY, JSON.stringify({ agent_id: 17 }))
    fetchMock.mockResolvedValueOnce(jsonResponse(successfulAgentAuth))

    renderSharePage()

    await screen.findByRole("button", { name: "start:Support Agent" })
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("drops the guest token and re-auths when share task creation is rejected", async () => {
    localStorage.clear()
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock
      .mockResolvedValueOnce(jsonResponse(successfulAgentAuth))
      .mockResolvedValueOnce(jsonResponse({ detail: "Share link is unavailable" }, 403))
      .mockResolvedValueOnce(jsonResponse(successfulAgentAuth))

    renderSharePage()

    fireEvent.click(await screen.findByRole("button", { name: "start:Support Agent" }))

    // A 401/403 on task-create invalidates the guest token: it's dropped and a
    // fresh /api/share/auth is forced instead of stranding the visitor.
    await waitFor(() => {
      const authCalls = fetchMock.mock.calls.filter(
        (call) => call[0] === "https://api.example/api/share/auth",
      )
      expect(authCalls.length).toBe(2)
    })
    errorSpy.mockRestore()
  })

  const futureExp = () => Math.floor(Date.now() / 1000) + 3600

  it("resumes a task persisted under the caller's own guest_id", async () => {
    localStorage.clear()
    const token = makeShareJwt({ guest_id: "guest-A", exp: futureExp() })
    localStorage.setItem(
      SHARE_AUTH_KEY,
      JSON.stringify({ ...successfulAgentAuth, access_token: token }),
    )
    // The task-id key is scoped by guest_id (agent 17 / guest-A).
    localStorage.setItem("share_task_share-tok_17_guest-A", "71")

    renderSharePage()

    expect(await screen.findByTestId("conversation-panel")).toBeInTheDocument()
    expect(app.setTaskId).toHaveBeenCalledWith(71, { navigate: false })
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("does not resume a task persisted under a different guest_id", async () => {
    localStorage.clear()
    const token = makeShareJwt({ guest_id: "guest-A", exp: futureExp() })
    localStorage.setItem(
      SHARE_AUTH_KEY,
      JSON.stringify({ ...successfulAgentAuth, access_token: token }),
    )
    // A task minted under guest-B lives under guest-B's key. guest-A must never
    // read it back — otherwise that foreign task would be permanently orphaned.
    localStorage.setItem("share_task_share-tok_17_guest-B", "99")

    renderSharePage()

    expect(await screen.findByRole("button", { name: "start:Support Agent" })).toBeInTheDocument()
    expect(app.setTaskId).toHaveBeenCalledWith(null, { navigate: false })
    expect(app.setTaskId).not.toHaveBeenCalledWith(99, { navigate: false })
  })

  it("clears a stale task when the server denies access for this guest", async () => {
    localStorage.clear()
    const token = makeShareJwt({ guest_id: "guest-A", exp: futureExp() })
    localStorage.setItem(
      SHARE_AUTH_KEY,
      JSON.stringify({ ...successfulAgentAuth, access_token: token }),
    )
    const taskKey = "share_task_share-tok_17_guest-A"
    localStorage.setItem(taskKey, "71")
    // The WS layer reports a per-guest access denial as a post-accept 4003
    // whose reason surfaces here (see use-websocket.ts onclose 4003 handling).
    // The backend reuses its not-found detail for guest mismatches so task ids
    // can't be enumerated (#973).
    app.connectionError = new Error("Task not found or access denied")

    renderSharePage()

    // The recovery effect drops the stale task and its persisted pointer so the
    // visitor lands on the start screen instead of a dead session.
    await waitFor(() => {
      expect(app.setTaskId).toHaveBeenCalledWith(null, { navigate: false })
    })
    expect(localStorage.getItem(taskKey)).toBeNull()
  })

  it("keeps an active task on a transient (non-access-denied) connection error", async () => {
    localStorage.clear()
    const token = makeShareJwt({ guest_id: "guest-A", exp: futureExp() })
    localStorage.setItem(
      SHARE_AUTH_KEY,
      JSON.stringify({ ...successfulAgentAuth, access_token: token }),
    )
    const taskKey = "share_task_share-tok_17_guest-A"
    localStorage.setItem(taskKey, "71")
    // A generic transport drop is NOT in SHARE_ACCESS_DENIED_REASONS — the
    // load-bearing guard must leave the live session untouched (a reconnect
    // resumes it) rather than wiping the task like an access denial would.
    app.connectionError = new Error(
      "WebSocket connection failed. The backend WebSocket endpoint may not be available.",
    )

    renderSharePage()

    // The session resumes and is never torn down by the recovery effect.
    expect(await screen.findByTestId("conversation-panel")).toBeInTheDocument()
    expect(app.setTaskId).toHaveBeenCalledWith(71, { navigate: false })
    expect(app.setTaskId).not.toHaveBeenCalledWith(null, { navigate: false })
    expect(localStorage.getItem(taskKey)).toBe("71")
  })
})

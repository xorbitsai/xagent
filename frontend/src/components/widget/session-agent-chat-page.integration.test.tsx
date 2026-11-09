import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const authState = vi.hoisted(() => ({
  token: null as string | null,
  user: null as { id: string } | null,
  refreshAccessToken: vi.fn(async () => false),
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => authState,
}))

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    refresh: vi.fn(),
  }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

// The page's toast proxy (@/components/ui/sonner) calls this mock's
// toast.error, so a test can assert on what the visitor was told.
const toastError = vi.hoisted(() => vi.fn())

vi.mock("sonner", () => ({
  toast: {
    error: toastError,
    success: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
  },
}))

// The latest props each mocked surface was rendered with, so a test can act as
// the visitor through the callbacks the page hands them. A test that sets
// `real` renders the actual start screen and panel instead, with the real
// composer inside, and acts through the DOM.
const surfaces = vi.hoisted(() => ({
  start: null as null | Record<string, any>,
  panel: null as null | Record<string, any>,
  real: false,
}))

vi.mock("@/components/chat/ChatStartScreen", async () => {
  const actual = await vi.importActual<typeof import("@/components/chat/ChatStartScreen")>(
    "@/components/chat/ChatStartScreen",
  )
  return {
    ChatStartScreen: (props: Record<string, any>) => {
      surfaces.start = props
      if (surfaces.real) return <actual.ChatStartScreen {...(props as any)} />
      return <div data-testid="session-start">{props.title}</div>
    },
  }
})

vi.mock("@/components/task/task-conversation-panel", async () => {
  const actual = await vi.importActual<typeof import("@/components/task/task-conversation-panel")>(
    "@/components/task/task-conversation-panel",
  )
  return {
    TaskConversationPanel: (props: Record<string, any>) => {
      surfaces.panel = props
      if (surfaces.real) return <actual.TaskConversationPanel {...(props as any)} />
      return <div data-testid="session-conversation" />
    },
  }
})

// The history list and view stand in for themselves unless a test sets `real`:
// the stand-ins hand the page the same callbacks the real ones do.
const historySurfaces = vi.hoisted(() => ({ real: false }))

vi.mock("./widget-history-list", async () => {
  const actual = await vi.importActual<typeof import("./widget-history-list")>("./widget-history-list")
  return {
    WidgetHistoryList: (props: React.ComponentProps<typeof actual.WidgetHistoryList>) => {
      if (historySurfaces.real) return <actual.WidgetHistoryList {...props} />
      return (
        <div>
          <button type="button" onClick={() => props.onOpen(41, "Earlier chat")}>pick-earlier</button>
          <button type="button" onClick={props.onBack}>back-to-chat</button>
        </div>
      )
    },
  }
})

vi.mock("./widget-history-view", async () => {
  const actual = await vi.importActual<typeof import("./widget-history-view")>("./widget-history-view")
  return {
    ...actual,
    WidgetHistoryView: (props: React.ComponentProps<typeof actual.WidgetHistoryView>) => {
      if (historySurfaces.real) return <actual.WidgetHistoryView {...props} />
      return (
        <div>
          <button type="button" onClick={() => props.onContinue("Continue this", props.taskId)}>hand-over</button>
          <button type="button" onClick={props.onBack}>back-to-list</button>
        </div>
      )
    },
  }
})

import { SessionAgentChatPage } from "./session-agent-chat-page"
import { W_UNSUPPORTED } from "./use-session-open-message"

const PARENT_ORIGIN = "https://shiftcare.example"
const SESSION_TOKEN = "st_integration_secret"
const EMBEDDED_PARENT = {
  postMessage: (...args: Parameters<Window["postMessage"]>) => window.postMessage(...args),
}
const originalParentDescriptor = Object.getOwnPropertyDescriptor(window, "parent")

class MockWebSocket {
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3
  static instances: MockWebSocket[] = []
  static failConstruction = false
  static constructionCount = 0

  readyState = 0
  protocol = ""
  onopen: (() => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  send = vi.fn()
  close = vi.fn(() => {
    this.readyState = MockWebSocket.CLOSED
  })

  constructor(
    public url: string,
    public protocols?: string | string[],
  ) {
    MockWebSocket.constructionCount += 1
    if (MockWebSocket.failConstruction) {
      throw new Error(`native constructor rejected ${SESSION_TOKEN}`)
    }
    MockWebSocket.instances.push(this)
  }

  open(protocol: string) {
    this.protocol = protocol
    this.readyState = MockWebSocket.OPEN
    this.onopen?.()
  }

  triggerError() {
    this.onerror?.(new Event("error"))
  }

  triggerClose(code = 1006) {
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.(new CloseEvent("close", { code }))
  }
}

const sessionUpdate = (overrides: Record<string, unknown> = {}) => {
  const now = Date.now()
  return {
    xagent: true,
    v: 1,
    type: "session_update",
    session_delivery_id: "delivery-a",
    session_token: SESSION_TOKEN,
    session_token_expires_at: new Date(now + 15 * 60_000).toISOString(),
    absolute_expires_at: new Date(now + 30 * 60_000).toISOString(),
    agent: {
      id: 42,
      name: "Support Agent",
      suggested_prompts: [],
    },
    ...overrides,
  }
}

const dispatchSession = (overrides: Record<string, unknown> = {}) => {
  act(() => {
    window.dispatchEvent(new MessageEvent("message", {
      data: sessionUpdate(overrides),
      origin: PARENT_ORIGIN,
      source: EMBEDDED_PARENT as unknown as MessageEventSource,
    }))
  })
}

const dispatchDegradedSession = () => {
  act(() => {
    window.dispatchEvent(new MessageEvent("message", {
      data: {
        xagent: true,
        v: 1,
        type: "session_degraded",
        code: "network_unavailable",
      },
      origin: PARENT_ORIGIN,
      source: EMBEDDED_PARENT as unknown as MessageEventSource,
    }))
  })
}

describe("SessionAgentChatPage connection failure integration", () => {
  beforeEach(() => {
    MockWebSocket.instances = []
    MockWebSocket.failConstruction = false
    MockWebSocket.constructionCount = 0
    localStorage.clear()
    sessionStorage.clear()
    vi.stubGlobal("WebSocket", MockWebSocket)
    Object.defineProperty(window, "parent", {
      configurable: true,
      value: EMBEDDED_PARENT,
    })
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    if (originalParentDescriptor) Object.defineProperty(window, "parent", originalParentDescriptor)
  })

  it.each([
    ["constructor setup", true],
    ["required protocol negotiation", false],
  ])(
    "moves to a terminal credential-free UI after %s failure",
    async (_label, failConstruction) => {
      MockWebSocket.failConstruction = failConstruction
      const storageSet = vi.spyOn(Storage.prototype, "setItem")
      const parentPostMessage = vi.spyOn(window, "postMessage")
        .mockImplementation(() => {})
      const consoleLog = vi.spyOn(console, "log")
        .mockImplementation(() => {})
      const consoleWarn = vi.spyOn(console, "warn")
        .mockImplementation(() => {})
      const consoleError = vi.spyOn(console, "error")
        .mockImplementation(() => {})

      render(<SessionAgentChatPage />)
      dispatchSession()

      if (!failConstruction) {
        await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
        act(() => MockWebSocket.instances[0].open(""))
      }

      await waitFor(() => {
        expect(screen.getByRole("heading", {
          name: "widgetSession.unavailable.title",
        })).toBeInTheDocument()
      })

      expect(screen.queryByText("Support Agent")).not.toBeInTheDocument()
      expect(document.documentElement.innerHTML).not.toContain(SESSION_TOKEN)
      expect(window.location.href).not.toContain(SESSION_TOKEN)
      expect(storageSet).not.toHaveBeenCalled()
      const serializedLogs = [
        ...consoleLog.mock.calls,
        ...consoleWarn.mock.calls,
        ...consoleError.mock.calls,
      ].flat().map(String).join(" ")
      expect(serializedLogs).not.toContain(SESSION_TOKEN)
      expect(parentPostMessage.mock.calls.filter(
        ([message]) => message?.type === "reconnect_request",
      )).toHaveLength(0)
      expect(MockWebSocket.constructionCount).toBe(1)
    },
  )

  it("requests one parent reconnect for a physical failure and its abnormal close", async () => {
    const parentPostMessage = vi.spyOn(window, "postMessage")
      .mockImplementation(() => {})
    vi.spyOn(console, "error").mockImplementation(() => {})

    render(<SessionAgentChatPage />)
    dispatchSession()
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))

    const socket = MockWebSocket.instances[0]
    act(() => {
      socket.open("xagent-session-v1")
      socket.triggerError()
      socket.triggerClose()
    })

    await waitFor(() => {
      expect(parentPostMessage.mock.calls.filter(
        ([message]) => message?.type === "reconnect_request",
      )).toEqual([[
        {
          xagent: true,
          v: 1,
          type: "reconnect_request",
          reason: "ws_closed",
        },
        PARENT_ORIGIN,
      ]])
    })
    expect(screen.getByRole("heading", { name: "Support Agent" }))
      .toBeInTheDocument()
    expect(screen.getByText("widgetChat.status.connecting"))
      .toBeInTheDocument()
    expect(MockWebSocket.constructionCount).toBe(1)
  })

  it("echoes the exact delivery discriminator after the current socket opens", async () => {
    const parentPostMessage = vi.spyOn(window, "postMessage")
      .mockImplementation(() => {})

    render(<SessionAgentChatPage />)
    dispatchSession()
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    act(() => MockWebSocket.instances[0].open("xagent-session-v1"))

    expect(parentPostMessage).toHaveBeenCalledWith(
      {
        xagent: true,
        v: 1,
        type: "session_connection_open",
        session_delivery_id: "delivery-a",
      },
      PARENT_ORIGIN,
    )
  })

  it("renders an existing Session as unavailable after a degraded parent update", async () => {
    render(<SessionAgentChatPage />)
    dispatchSession()
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))

    const socket = MockWebSocket.instances[0]
    act(() => socket.open("xagent-session-v1"))
    dispatchDegradedSession()

    await waitFor(() => {
      expect(screen.getByRole("heading", {
        name: "widgetSession.unavailable.title",
      })).toBeInTheDocument()
    })
    expect(screen.queryByTestId("session-start")).not.toBeInTheDocument()
    expect(MockWebSocket.constructionCount).toBe(1)
    expect(socket.close).toHaveBeenCalledTimes(1)
    expect(socket.readyState).toBe(MockWebSocket.CLOSED)
  })
})

describe("SessionAgentChatPage file attachments", () => {
  const FILE_A = "0b1e5c9e-3f1a-4a63-9b0e-6c1d2f3a4b5c"
  const FILE_B = "7d2f0c1a-8b3e-4c55-a1f0-2e9d4b6c8a10"
  const capabilities = {
    file_uploads: {
      allowed_extensions: [".pdf", ".png"],
      max_file_bytes: 1000,
      max_files_per_message: 3,
    },
  }
  const fetchMock = vi.fn()

  const created = (fileId: string, name = "a.pdf") =>
    new Response(JSON.stringify({
      file_id: fileId,
      filename: name,
      file_size: 3,
      mime_type: "application/pdf",
      created_at: "2026-01-01T00:00:00Z",
    }), { status: 201, headers: { "Content-Type": "application/json" } })
  const errorResponse = (status: number, code: string, headers: Record<string, string> = {}) =>
    new Response(JSON.stringify({ error: { code, message: "no" } }), {
      status,
      headers: { "Content-Type": "application/json", ...headers },
    })
  const pdf = (name: string) => new File(["abc"], name, { type: "application/pdf" })
  const deferred = <T,>() => {
    let resolve!: (value: T) => void
    const promise = new Promise<T>((resolvePromise) => { resolve = resolvePromise })
    return { promise, resolve }
  }
  // An upload request that never answers and fails only when it is aborted.
  const pendingUntilAborted = (_url: string, init: RequestInit) =>
    new Promise<Response>((_resolve, reject) => {
      init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))
    })
  const tightened = {
    file_uploads: { ...capabilities.file_uploads, max_files_per_message: 1 },
  }
  // [what the replacing session does, its update, the failure message, the
  // start-page files afterwards]. Without a capability the page hides the
  // cards but keeps the files.
  const replacements: Array<[string, Record<string, unknown>, string, (a: File, b: File) => File[]]> = [
    ["lowers the per-message limit", { capabilities: tightened }, "widgetSession.attachments.tooManyFiles", (a, b) => [a, b]],
    ["withdraws the capability", {}, "widgetSession.attachments.notAvailable", () => []],
  ]

  const mountActive = async (overrides: Record<string, unknown> = { capabilities }) => {
    render(<SessionAgentChatPage />)
    dispatchSession(overrides)
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open("xagent-session-v1"))
    return socket
  }
  const receive = (socket: MockWebSocket, frame: Record<string, unknown>) => {
    act(() => socket.onmessage?.({ data: JSON.stringify(frame) } as MessageEvent))
  }
  const accept = (socket: MockWebSocket, clientMessageId: string) =>
    receive(socket, { type: "message_accepted", client_message_id: clientMessageId })
  const bindTask = (socket: MockWebSocket) =>
    receive(socket, {
      type: "trace_event",
      timestamp: "2026-05-27T05:00:00Z",
      data: {
        event_id: "task-info-42",
        event_type: "task_info",
        data: {
          id: 42,
          title: "Task 42",
          status: "running",
          description: "Session task",
          created_at: "2026-05-27T05:00:00Z",
          updated_at: "2026-05-27T05:00:00Z",
        },
      },
    })
  const frames = (socket: MockWebSocket) =>
    socket.send.mock.calls.map((call) => JSON.parse(call[0] as string) as Record<string, any>)
  const chatFrames = (socket: MockWebSocket) => frames(socket).filter((frame) => frame.type === "chat")
  const uploadCalls = () => fetchMock.mock.calls.filter(
    (call) => String(call[0]).endsWith("/v1/external/chat/sessions/files"),
  )
  const statusOf = (file: File) => surfaces.start?.attachmentStatuses?.get(file)
  const settle = (promise: Promise<unknown>) =>
    promise.then(() => null, (error: unknown) => error)
  const reconnectRequests = (spy: { mock: { calls: unknown[][] } }) =>
    spy.mock.calls.filter(([message]) => (message as { type?: string } | null)?.type === "reconnect_request")

  beforeEach(() => {
    MockWebSocket.instances = []
    MockWebSocket.failConstruction = false
    MockWebSocket.constructionCount = 0
    surfaces.start = null
    surfaces.panel = null
    fetchMock.mockReset()
    toastError.mockReset()
    localStorage.clear()
    sessionStorage.clear()
    localStorage.setItem("token", "main-site-token")
    vi.stubGlobal("WebSocket", MockWebSocket)
    vi.stubGlobal("fetch", fetchMock)
    vi.spyOn(console, "log").mockImplementation(() => {})
    Object.defineProperty(window, "parent", { configurable: true, value: EMBEDDED_PARENT })
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    if (originalParentDescriptor) Object.defineProperty(window, "parent", originalParentDescriptor)
  })

  describe("the upload entry follows the advertised capability", () => {
    it("offers uploads on the start page and the conversation panel with the advertised limits", async () => {
      const socket = await mountActive()

      expect(surfaces.start?.attachments).toEqual({
        accept: ".pdf,.png",
        maxFiles: 3,
        maxFileBytes: 1000,
      })
      expect(surfaces.start?.filesDisabled).toBe(true)

      bindTask(socket)
      await waitFor(() => expect(screen.getByTestId("session-conversation")).toBeInTheDocument())
      expect(surfaces.panel?.attachments).toEqual({
        accept: ".pdf,.png",
        maxFiles: 3,
        maxFileBytes: 1000,
      })
    })

    it("offers nothing and sends nothing extra when no capability is advertised", async () => {
      const socket = await mountActive({})

      expect(surfaces.start?.attachments).toBeUndefined()
      expect(surfaces.start?.files).toEqual([])
      let sending!: Promise<unknown>
      act(() => {
        sending = surfaces.start?.onSend("hello", [], { clientMessageId: "cid-plain" })
      })
      await waitFor(() => expect(chatFrames(socket)).toHaveLength(1))
      accept(socket, "cid-plain")
      await act(async () => { await sending })

      expect(chatFrames(socket)[0]).not.toHaveProperty("file_ids")
      expect(uploadCalls()).toHaveLength(0)
    })

    it("treats a malformed capability as none and keeps the session active", async () => {
      await mountActive({
        capabilities: { file_uploads: { allowed_extensions: [], max_file_bytes: 1, max_files_per_message: 1 } },
      })

      expect(surfaces.start?.attachments).toBeUndefined()
      expect(screen.getByTestId("session-start")).toBeInTheDocument()
    })

    it("withdraws the entry when a later update advertises nothing", async () => {
      await mountActive()
      expect(surfaces.start?.attachments).toBeDefined()

      dispatchSession({ session_delivery_id: "delivery-b" })

      await waitFor(() => expect(surfaces.start?.attachments).toBeUndefined())
      expect(screen.getByTestId("session-start")).toBeInTheDocument()
    })

    it("keeps the last advertisement while the token is being refreshed", async () => {
      vi.spyOn(window, "postMessage").mockImplementation(() => {})
      const socket = await mountActive()
      act(() => socket.triggerClose())

      await waitFor(() => expect(screen.getByText("widgetChat.status.connecting")).toBeInTheDocument())
      expect(surfaces.start?.attachments).toBeDefined()
    })
  })

  describe("sending files from the start page", () => {
    it("uploads the files one after another and sends their ids in order", async () => {
      const socket = await mountActive()
      const first = deferred<Response>()
      const second = deferred<Response>()
      fetchMock.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise)
      const [a, b] = [pdf("a.pdf"), pdf("b.pdf")]
      let sending!: Promise<unknown>

      act(() => {
        sending = surfaces.start?.onSend("with files", [a, b], { clientMessageId: "cid-1" })
      })
      await waitFor(() => expect(uploadCalls()).toHaveLength(1))
      expect(statusOf(a)).toBe("uploading")
      expect(statusOf(b)).toBeUndefined()
      expect(chatFrames(socket)).toHaveLength(0)

      await act(async () => { first.resolve(created(FILE_A, "a.pdf")) })
      await waitFor(() => expect(uploadCalls()).toHaveLength(2))
      expect(statusOf(a)).toBe("uploaded")
      expect(statusOf(b)).toBe("uploading")
      expect(chatFrames(socket)).toHaveLength(0)

      await act(async () => { second.resolve(created(FILE_B, "b.pdf")) })
      await waitFor(() => expect(chatFrames(socket)).toHaveLength(1))
      expect(chatFrames(socket)[0]).toMatchObject({
        message: "with files",
        client_message_id: "cid-1",
        file_ids: [FILE_A, FILE_B],
      })
      expect(chatFrames(socket)[0]).not.toHaveProperty("request_id")
      accept(socket, "cid-1")
      await act(async () => { await sending })
      expect(surfaces.start?.files).toEqual([])
    })

    it("uses only the session token and never touches the stored login or file endpoints", async () => {
      const socket = await mountActive()
      fetchMock.mockImplementation(async () => created(FILE_A))
      let sending!: Promise<unknown>

      act(() => {
        sending = surfaces.start?.onSend("hello", [pdf("a.pdf")], { clientMessageId: "cid-sec" })
      })
      await waitFor(() => expect(chatFrames(socket)).toHaveLength(1))
      accept(socket, "cid-sec")
      await act(async () => { await sending })

      for (const [url, init] of fetchMock.mock.calls) {
        expect(String(url)).not.toContain("/api/files/")
        expect(init.headers.Authorization).toBe(`Bearer ${SESSION_TOKEN}`)
        expect(init.credentials).toBe("omit")
      }
      expect(JSON.stringify(fetchMock.mock.calls.map((call) => call[1].headers))).not.toContain("main-site-token")
    })

    it("refreshes a nearly expired token first, uploads with the new one and waits for the new socket", async () => {
      vi.useFakeTimers({ toFake: ["Date"] })
      const parentPostMessage = vi.spyOn(window, "postMessage").mockImplementation(() => {})
      const socket = await mountActive({
        capabilities,
        session_token_expires_at: new Date(Date.now() + 61_000).toISOString(),
      })
      vi.setSystemTime(Date.now() + 31_000)
      fetchMock.mockImplementation(async () => created(FILE_A))
      let sending!: Promise<unknown>

      act(() => {
        sending = surfaces.start?.onSend("hello", [pdf("a.pdf")], { clientMessageId: "cid-fresh" })
      })
      await waitFor(() => expect(reconnectRequests(parentPostMessage)).toHaveLength(1))
      expect(uploadCalls()).toHaveLength(0)
      expect(reconnectRequests(parentPostMessage)[0][0]).toMatchObject({ reason: "token_expired" })

      dispatchSession({
        session_delivery_id: "delivery-b",
        session_token: "st_renewed_token",
        capabilities,
      })
      await waitFor(() => expect(uploadCalls()).toHaveLength(1))
      expect(uploadCalls()[0][1].headers.Authorization).toBe("Bearer st_renewed_token")

      // The file is already stored; the new generation's socket opens late.
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
      const next = MockWebSocket.instances[1]
      await act(async () => { await new Promise((resolve) => setTimeout(resolve, 200)) })
      expect(chatFrames(next)).toHaveLength(0)
      expect(chatFrames(socket)).toHaveLength(0)

      act(() => next.open("xagent-session-v1"))
      await waitFor(() => expect(chatFrames(next)).toHaveLength(1))
      expect(chatFrames(next)[0].file_ids).toEqual([FILE_A])
      accept(next, "cid-fresh")
      await act(async () => { await sending })
      expect(reconnectRequests(parentPostMessage)).toHaveLength(1)
    })

    it("fails a failed batch without sending and uploads only the missing file on the next attempt", async () => {
      const socket = await mountActive()
      fetchMock
        .mockImplementationOnce(async () => created(FILE_A, "a.pdf"))
        .mockImplementationOnce(async () => errorResponse(429, "quota_exceeded"))
      const [a, b] = [pdf("a.pdf"), pdf("b.pdf")]
      let failure: unknown

      await act(async () => {
        failure = await settle(surfaces.start?.onSend("first", [a, b], { clientMessageId: "cid-q" }))
      })

      expect(failure).toMatchObject({ errorCode: "upload_quota_exceeded" })
      expect(chatFrames(socket)).toHaveLength(0)
      expect(statusOf(a)).toBe("uploaded")
      expect(statusOf(b)).toBe("failed")

      fetchMock.mockImplementationOnce(async () => created(FILE_B, "b.pdf"))
      let sending!: Promise<unknown>
      act(() => {
        sending = surfaces.start?.onSend("first, reworded", [a, b], { clientMessageId: "cid-q2" })
      })
      await waitFor(() => expect(chatFrames(socket)).toHaveLength(1))
      expect(uploadCalls()).toHaveLength(3)
      expect(chatFrames(socket)[0].file_ids).toEqual([FILE_A, FILE_B])
      accept(socket, "cid-q2")
      await act(async () => { await sending })
    })

    it("keeps the retry of a failed file disabled while another file uploads, and enables it afterwards", async () => {
      await mountActive()
      const failed = pdf("failed.pdf")
      const other = pdf("other.pdf")
      fetchMock.mockImplementationOnce(async () => errorResponse(500, "server_error"))
      await act(async () => {
        await settle(surfaces.start?.onSend("hello", [failed], { clientMessageId: "cid-busy" }))
      })
      expect(statusOf(failed)).toBe("failed")
      expect(surfaces.start?.attachmentsBusy).toBe(false)

      const reply = deferred<Response>()
      fetchMock.mockImplementationOnce(() => reply.promise)
      let retrying!: Promise<unknown>
      act(() => { retrying = surfaces.start?.onRetryAttachment(other) })
      await waitFor(() => expect(statusOf(other)).toBe("uploading"))

      expect(statusOf(failed)).toBe("failed")
      expect(surfaces.start?.attachmentsBusy).toBe(true)

      reply.resolve(created(FILE_A, "other.pdf"))
      await act(async () => { await retrying })

      expect(statusOf(other)).toBe("uploaded")
      expect(surfaces.start?.attachmentsBusy).toBe(false)
    })

    it("passes the busy state to the conversation panel as well", async () => {
      const socket = await mountActive()
      bindTask(socket)
      await waitFor(() => expect(screen.getByTestId("session-conversation")).toBeInTheDocument())
      expect(surfaces.panel?.attachmentsBusy).toBe(false)

      const reply = deferred<Response>()
      fetchMock.mockImplementationOnce(() => reply.promise)
      let retrying!: Promise<unknown>
      act(() => { retrying = surfaces.panel?.onRetryAttachment(pdf("again.pdf")) })
      await waitFor(() => expect(surfaces.panel?.attachmentsBusy).toBe(true))

      reply.resolve(created(FILE_A, "again.pdf"))
      await act(async () => { await retrying })
      expect(surfaces.panel?.attachmentsBusy).toBe(false)
    })

    it("shows the file as uploading while a rate-limited request waits, then offers a per-file retry", async () => {
      const socket = await mountActive()
      vi.useFakeTimers()
      fetchMock.mockImplementation(async () => errorResponse(429, "rate_limited"))
      const a = pdf("a.pdf")
      const b = pdf("b.pdf")
      let failure: unknown
      let sending!: Promise<unknown>

      act(() => {
        sending = surfaces.start?.onSend("hello", [a, b], { clientMessageId: "cid-rl" })
          .then(() => null, (error: unknown) => { failure = error })
      })
      await act(async () => { await vi.advanceTimersByTimeAsync(3_000) })
      expect(uploadCalls().length).toBeGreaterThanOrEqual(2)
      expect(statusOf(a)).toBe("uploading")
      expect(toastError).not.toHaveBeenCalled()
      expect(failure).toBeUndefined()

      await act(async () => { await vi.advanceTimersByTimeAsync(60_000) })
      await act(async () => { await sending })

      expect(failure).toMatchObject({ errorCode: "upload_rate_limited" })
      expect(uploadCalls()).toHaveLength(4)
      expect(statusOf(a)).toBe("failed")
      expect(statusOf(b)).toBeUndefined()
      expect(chatFrames(socket)).toHaveLength(0)

      fetchMock.mockReset()
      fetchMock.mockImplementation(async () => created(FILE_A))
      await act(async () => { await surfaces.start?.onRetryAttachment(a) })

      expect(uploadCalls()).toHaveLength(1)
      expect(statusOf(a)).toBe("uploaded")
      expect(chatFrames(socket)).toHaveLength(0)
    })

    it("refreshes once after a 401 and retries, but fails after a second 401", async () => {
      const parentPostMessage = vi.spyOn(window, "postMessage").mockImplementation((message) => {
        if ((message as { type?: string }).type === "reconnect_request") {
          queueMicrotask(() => dispatchSession({
            session_delivery_id: `delivery-${Math.random()}`,
            session_token: "st_after_401",
            capabilities,
          }))
        }
      })
      const socket = await mountActive()
      fetchMock
        .mockImplementationOnce(async () => errorResponse(401, "invalid_api_key"))
        .mockImplementationOnce(async () => created(FILE_A))
      let sending!: Promise<unknown>

      act(() => {
        sending = surfaces.start?.onSend("hello", [pdf("a.pdf")], { clientMessageId: "cid-401" })
      })
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
      act(() => MockWebSocket.instances[1].open("xagent-session-v1"))
      await waitFor(() => expect(chatFrames(MockWebSocket.instances[1])).toHaveLength(1))
      expect(uploadCalls().map((call) => call[1].headers.Authorization)).toEqual([
        `Bearer ${SESSION_TOKEN}`,
        "Bearer st_after_401",
      ])
      expect(reconnectRequests(parentPostMessage)).toHaveLength(1)
      accept(MockWebSocket.instances[1], "cid-401")
      await act(async () => { await sending })
      expect(chatFrames(socket)).toHaveLength(0)

      fetchMock.mockReset()
      fetchMock.mockImplementation(async () => errorResponse(401, "invalid_api_key"))
      let failure: unknown
      await act(async () => {
        failure = await settle(surfaces.start?.onSend("again", [pdf("b.pdf")], { clientMessageId: "cid-401b" }))
      })
      expect(failure).toMatchObject({ errorCode: "upload_failed" })
      expect(uploadCalls()).toHaveLength(2)
    })

    it("fails the send when the session issued after a 401 no longer advertises uploads", async () => {
      vi.spyOn(window, "postMessage").mockImplementation((message) => {
        if ((message as { type?: string }).type === "reconnect_request") {
          queueMicrotask(() => dispatchSession({
            session_delivery_id: "delivery-after-401",
            session_token: "st_after_401",
          }))
        }
      })
      const socket = await mountActive()
      fetchMock.mockImplementationOnce(async () => errorResponse(401, "invalid_api_key"))
      const a = pdf("a.pdf")
      let failure: unknown

      await act(async () => {
        failure = await settle(surfaces.start?.onSend("hello", [a], { clientMessageId: "cid-401-cap" }))
      })

      expect(failure).toMatchObject({ message: "widgetSession.attachments.notAvailable", userFacing: true })
      expect(uploadCalls()).toHaveLength(1)
      expect(chatFrames(socket)).toHaveLength(0)
      expect(MockWebSocket.instances.flatMap((instance) => chatFrames(instance))).toHaveLength(0)
    })

    it("locks the start page and new-conversation and marks the conversation panel busy while files upload", async () => {
      const socket = await mountActive()
      const pending = deferred<Response>()
      fetchMock.mockReturnValueOnce(pending.promise)
      let sending!: Promise<unknown>

      act(() => {
        sending = surfaces.start?.onSend("hello", [pdf("a.pdf")], { clientMessageId: "cid-lock" })
          .catch(() => undefined)
      })
      await waitFor(() => expect(uploadCalls()).toHaveLength(1))
      expect(surfaces.start?.isSending).toBe(true)

      bindTask(socket)
      await waitFor(() => expect(screen.getByTestId("session-conversation")).toBeInTheDocument())
      expect(surfaces.panel?.attachmentsBusy).toBe(true)
      fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
      expect(screen.getByRole("menuitem", { name: "widgetSession.startNewConversation" })).toBeDisabled()

      await act(async () => { pending.resolve(created(FILE_A)) })
      await waitFor(() => expect(chatFrames(socket)).toHaveLength(1))
      // The panel stays marked busy until the send has gone out and been answered.
      expect(surfaces.panel?.attachmentsBusy).toBe(true)
      accept(socket, "cid-lock")
      await act(async () => { await sending })
      await waitFor(() => expect(surfaces.panel?.attachmentsBusy).toBe(false))
    })

    it("aborts an in-flight upload when the visitor removes the file", async () => {
      await mountActive()
      fetchMock.mockImplementation((_url: string, init: RequestInit) => new Promise((_resolve, reject) => {
        init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))
      }))
      const a = pdf("a.pdf")
      let failure: unknown
      let sending!: Promise<unknown>
      act(() => {
        surfaces.start?.onFilesChange([a])
      })
      act(() => {
        sending = surfaces.start?.onSend("hello", [a], { clientMessageId: "cid-abort" })
          .then(() => null, (error: unknown) => { failure = error })
      })
      await waitFor(() => expect(uploadCalls()).toHaveLength(1))
      const signal = uploadCalls()[0][1].signal as AbortSignal
      expect(signal.aborted).toBe(false)

      act(() => { surfaces.start?.onFilesChange([]) })
      await act(async () => { await sending })

      expect(signal.aborted).toBe(true)
      expect((failure as Error).name).toBe("AbortError")
      expect(failure).toMatchObject({ notificationHandled: true })
      expect(screen.queryByRole("alert")).toBeNull()
      expect(statusOf(a)).toBeUndefined()
    })

    it("aborts uploads in flight when the session ends and when the page unmounts", async () => {
      const socket = await mountActive()
      fetchMock.mockImplementation((_url: string, init: RequestInit) => new Promise((_resolve, reject) => {
        init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))
      }))
      let sending!: Promise<unknown>
      act(() => {
        sending = surfaces.start?.onSend("hello", [pdf("a.pdf")], { clientMessageId: "cid-end" })
          .catch(() => undefined)
      })
      await waitFor(() => expect(uploadCalls()).toHaveLength(1))
      const signal = uploadCalls()[0][1].signal as AbortSignal

      act(() => {
        window.dispatchEvent(new MessageEvent("message", {
          data: { xagent: true, v: 1, type: "session_terminal", code: "session_expired" },
          origin: PARENT_ORIGIN,
          source: EMBEDDED_PARENT as unknown as MessageEventSource,
        }))
      })
      await act(async () => { await sending })

      expect(signal.aborted).toBe(true)
      expect(socket.close).toHaveBeenCalled()
    })

    it("lets a start-page file be sent again after the session was unavailable and came back", async () => {
      await mountActive()
      fetchMock.mockImplementationOnce(pendingUntilAborted)
      const a = pdf("a.pdf")
      let failure: unknown
      act(() => { surfaces.start?.onFilesChange([a]) })
      act(() => {
        void surfaces.start?.onSend("hello", [a], { clientMessageId: "cid-away" })
          .then(() => null, (error: unknown) => { failure = error })
      })
      await waitFor(() => expect(uploadCalls()).toHaveLength(1))
      expect(statusOf(a)).toBe("uploading")

      dispatchDegradedSession()
      await waitFor(() => expect(failure).toBeDefined())
      dispatchSession({ session_delivery_id: "delivery-back", capabilities })
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
      const next = MockWebSocket.instances[1]
      act(() => next.open("xagent-session-v1"))

      await waitFor(() => expect(surfaces.start?.files).toEqual([a]))
      expect(statusOf(a)).toBeUndefined()
      expect(surfaces.start?.attachmentsBusy).toBe(false)
      fetchMock.mockImplementationOnce(async () => created(FILE_A, "a.pdf"))
      let sending!: Promise<unknown>
      act(() => {
        sending = surfaces.start?.onSend("hello", [a], { clientMessageId: "cid-back" })
      })
      await waitFor(() => expect(chatFrames(next)).toHaveLength(1))
      expect(chatFrames(next)[0].file_ids).toEqual([FILE_A])
      accept(next, "cid-back")
      await act(async () => { await sending })
    })

    it("keeps start-page files hidden while uploads are not advertised through a text-only send", async () => {
      await mountActive()
      const a = pdf("a.pdf")
      act(() => { surfaces.start?.onFilesChange([a]) })
      dispatchSession({ session_delivery_id: "delivery-withdrawn" })
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
      const next = MockWebSocket.instances[1]
      act(() => next.open("xagent-session-v1"))
      expect(surfaces.start?.files).toEqual([])

      let sending!: Promise<unknown>
      act(() => {
        sending = surfaces.start?.onSend("text only", [], { clientMessageId: "cid-hidden" })
      })
      await waitFor(() => expect(chatFrames(next)).toHaveLength(1))
      accept(next, "cid-hidden")
      await act(async () => { await sending })
      dispatchSession({ session_delivery_id: "delivery-back", capabilities })

      expect(chatFrames(next)[0]).not.toHaveProperty("file_ids")
      await waitFor(() => expect(surfaces.start?.files).toEqual([a]))
    })

    it("aborts uploads in flight when the page unmounts", async () => {
      const { unmount } = render(<SessionAgentChatPage />)
      dispatchSession({ capabilities })
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
      act(() => MockWebSocket.instances[0].open("xagent-session-v1"))
      fetchMock.mockImplementation((_url: string, init: RequestInit) => new Promise((_resolve, reject) => {
        init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))
      }))
      let sending!: Promise<unknown>
      act(() => {
        sending = surfaces.start?.onSend("hello", [pdf("a.pdf")], { clientMessageId: "cid-unmount" })
          .catch(() => undefined)
      })
      await waitFor(() => expect(uploadCalls()).toHaveLength(1))
      const signal = uploadCalls()[0][1].signal as AbortSignal

      unmount()
      await sending

      expect(signal.aborted).toBe(true)
    })

    it("keeps the composer locked while the socket opens and drops a file removed in that wait", async () => {
      render(<SessionAgentChatPage />)
      dispatchSession({ capabilities })
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
      const socket = MockWebSocket.instances[0]
      fetchMock
        .mockImplementationOnce(async () => created(FILE_A, "a.pdf"))
        .mockImplementationOnce(async () => created(FILE_B, "b.pdf"))
      const [a, b] = [pdf("a.pdf"), pdf("b.pdf")]
      let failure: unknown
      let sending!: Promise<unknown>

      act(() => {
        surfaces.start?.onFilesChange([a, b])
      })
      act(() => {
        sending = surfaces.start?.onSend("hello", [a, b], { clientMessageId: "cid-wait" })
          .then(() => null, (error: unknown) => { failure = error })
      })
      await waitFor(() => expect(uploadCalls()).toHaveLength(2))
      await waitFor(() => expect(statusOf(b)).toBe("uploaded"))

      // Every upload is done; only the socket is missing.
      expect(surfaces.start?.isSending).toBe(true)
      expect(chatFrames(socket)).toHaveLength(0)

      // Removing a file ends the send at once, without waiting for the socket.
      act(() => { surfaces.start?.onFilesChange([a]) })
      await act(async () => { await sending })

      expect((failure as Error).name).toBe("AbortError")
      expect(failure).toMatchObject({ notificationHandled: true })
      expect(socket.readyState).not.toBe(MockWebSocket.OPEN)
      expect(screen.queryByRole("alert")).toBeNull()
      act(() => socket.open("xagent-session-v1"))
      expect(chatFrames(socket)).toHaveLength(0)
      await waitFor(() => expect(surfaces.start?.isSending).toBe(false))
    })

    it("ends the send at once when an uploaded file is removed, cancelling the upload it waits on", async () => {
      const socket = await mountActive()
      fetchMock
        .mockImplementationOnce(async () => created(FILE_A, "a.pdf"))
        .mockImplementationOnce(pendingUntilAborted)
      const [a, b, c] = [pdf("a.pdf"), pdf("b.pdf"), pdf("c.pdf")]
      let failure: unknown
      act(() => { surfaces.start?.onFilesChange([a, b, c]) })
      act(() => {
        void surfaces.start?.onSend("hello", [a, b, c], { clientMessageId: "cid-rm2" })
          .then(() => null, (error: unknown) => { failure = error })
      })
      await waitFor(() => expect(uploadCalls()).toHaveLength(2))
      const signalB = uploadCalls()[1][1].signal as AbortSignal
      expect(statusOf(b)).toBe("uploading")

      act(() => { surfaces.start?.onFilesChange([b, c]) })
      await waitFor(() => expect(failure).toBeDefined())

      expect(signalB.aborted).toBe(true)
      expect((failure as Error).name).toBe("AbortError")
      expect(failure).toMatchObject({ notificationHandled: true })
      expect(uploadCalls()).toHaveLength(2)
      expect(statusOf(b)).toBeUndefined()
      expect(chatFrames(socket)).toHaveLength(0)
      await waitFor(() => expect(surfaces.start?.isSending).toBe(false))
      expect(surfaces.start?.attachmentsBusy).toBe(false)
      expect(surfaces.start?.files).toEqual([b, c])
    })

    it("does not upload a retried file removed while the retry waits for a token, and the refresh still serves another upload", async () => {
      vi.useFakeTimers({ toFake: ["Date"] })
      const parentPostMessage = vi.spyOn(window, "postMessage").mockImplementation(() => {})
      const socket = await mountActive({
        capabilities,
        session_token_expires_at: new Date(Date.now() + 61_000).toISOString(),
      })
      const [a, b] = [pdf("a.pdf"), pdf("b.pdf")]
      act(() => { surfaces.start?.onFilesChange([a]) })
      fetchMock.mockImplementationOnce(async () => errorResponse(500, "server_error"))
      await act(async () => {
        await settle(surfaces.start?.onSend("first", [a], { clientMessageId: "cid-r0" }))
      })
      expect(statusOf(a)).toBe("failed")
      toastError.mockReset()

      vi.setSystemTime(Date.now() + 31_000)
      act(() => { surfaces.start?.onFilesChange([a, b]) })
      let retrying!: Promise<unknown>
      act(() => { retrying = surfaces.start?.onRetryAttachment(a) })
      await waitFor(() => expect(reconnectRequests(parentPostMessage)).toHaveLength(1))
      expect(statusOf(a)).toBe("uploading")
      let sending!: Promise<unknown>
      act(() => {
        sending = surfaces.start?.onSend("second", [b], { clientMessageId: "cid-r1" })
      })

      act(() => { surfaces.start?.onFilesChange([b]) })
      await act(async () => { await retrying })
      expect(statusOf(a)).toBeUndefined()

      fetchMock.mockImplementation(async () => created(FILE_B, "b.pdf"))
      dispatchSession({
        session_delivery_id: "delivery-b",
        session_token: "st_renewed_token",
        capabilities,
      })
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
      const next = MockWebSocket.instances[1]
      act(() => next.open("xagent-session-v1"))
      await waitFor(() => expect(chatFrames(next)).toHaveLength(1))
      expect(chatFrames(next)[0].file_ids).toEqual([FILE_B])
      accept(next, "cid-r1")
      await act(async () => { await sending })

      expect(uploadCalls()).toHaveLength(2)
      expect(uploadCalls()[1][1].headers.Authorization).toBe("Bearer st_renewed_token")
      expect(statusOf(a)).toBeUndefined()
      expect(toastError).not.toHaveBeenCalled()
      expect(reconnectRequests(parentPostMessage)).toHaveLength(1)
      expect(chatFrames(socket)).toHaveLength(0)
      expect(surfaces.start?.attachmentsBusy).toBe(false)
    })

    it.each(replacements)(
      "fails the send before any upload when the refreshed session %s",
      async (_name, update, message, filesAfter) => {
        vi.useFakeTimers({ toFake: ["Date"] })
        const parentPostMessage = vi.spyOn(window, "postMessage").mockImplementation(() => {})
        const socket = await mountActive({
          capabilities,
          session_token_expires_at: new Date(Date.now() + 61_000).toISOString(),
        })
        vi.setSystemTime(Date.now() + 31_000)
        fetchMock.mockImplementation(async () => created(FILE_A))
        const [a, b] = [pdf("a.pdf"), pdf("b.pdf")]
        let failure: unknown
        act(() => { surfaces.start?.onFilesChange([a, b]) })
        act(() => {
          void surfaces.start?.onSend("hello", [a, b], { clientMessageId: "cid-cap-upload" })
            .then(() => null, (error: unknown) => { failure = error })
        })
        await waitFor(() => expect(reconnectRequests(parentPostMessage)).toHaveLength(1))

        dispatchSession({ session_delivery_id: "delivery-b", session_token: "st_renewed_token", ...update })
        await waitFor(() => expect(failure).toBeDefined())

        expect(failure).toMatchObject({ message, userFacing: true })
        expect(uploadCalls()).toHaveLength(0)
        await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
        act(() => MockWebSocket.instances[1].open("xagent-session-v1"))
        expect(chatFrames(socket)).toHaveLength(0)
        expect(chatFrames(MockWebSocket.instances[1])).toHaveLength(0)
        expect(surfaces.start?.files).toEqual(filesAfter(a, b))
        await waitFor(() => expect(surfaces.start?.isSending).toBe(false))

        // Without a capability the page hides the cards, so their state is only
        // visible once a session advertises uploads again.
        dispatchSession({ session_delivery_id: "delivery-c", capabilities })
        await waitFor(() => expect(surfaces.start?.attachmentStatuses).toBeDefined())
        expect(statusOf(a)).toBeUndefined()
        expect(statusOf(b)).toBeUndefined()
      },
    )

    it.each(replacements)(
      "fails the send after its uploads when the session whose socket opens %s",
      async (_name, update, message, filesAfter) => {
        render(<SessionAgentChatPage />)
        dispatchSession({ capabilities })
        await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
        fetchMock
          .mockImplementationOnce(async () => created(FILE_A, "a.pdf"))
          .mockImplementationOnce(async () => created(FILE_B, "b.pdf"))
        const [a, b] = [pdf("a.pdf"), pdf("b.pdf")]
        let failure: unknown
        act(() => { surfaces.start?.onFilesChange([a, b]) })
        act(() => {
          void surfaces.start?.onSend("hello", [a, b], { clientMessageId: "cid-cap-frame" })
            .then(() => null, (error: unknown) => { failure = error })
        })
        await waitFor(() => expect(uploadCalls()).toHaveLength(2))
        await waitFor(() => expect(statusOf(b)).toBe("uploaded"))

        dispatchSession({ session_delivery_id: "delivery-b", ...update })
        await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
        const next = MockWebSocket.instances[1]
        act(() => next.open("xagent-session-v1"))
        await waitFor(() => expect(failure).toBeDefined())

        expect(failure).toMatchObject({ message, userFacing: true })
        expect(chatFrames(MockWebSocket.instances[0])).toHaveLength(0)
        expect(chatFrames(next)).toHaveLength(0)
        expect(surfaces.start?.files).toEqual(filesAfter(a, b))
        await waitFor(() => expect(surfaces.start?.isSending).toBe(false))
      },
    )

    it("uploads again after the server reports an attached file as unavailable", async () => {
      const socket = await mountActive()
      fetchMock
        .mockImplementationOnce(async () => created(FILE_A))
        .mockImplementationOnce(async () => created(FILE_B))
      const a = pdf("a.pdf")
      let rejected: unknown
      act(() => {
        void surfaces.start?.onSend("hello", [a], { clientMessageId: "cid-gone" })
          .then(() => null, (error: unknown) => { rejected = error })
      })
      await waitFor(() => expect(chatFrames(socket)).toHaveLength(1))

      receive(socket, {
        type: "message_rejected",
        client_message_id: "cid-gone",
        rejection_outcome: "not_accepted",
        error_code: "message_attachment_unavailable",
      })
      await waitFor(() => expect(rejected).toBeDefined())
      expect(rejected).toMatchObject({ errorCode: "message_attachment_unavailable" })
      expect(statusOf(a)).toBeUndefined()

      let sending!: Promise<unknown>
      act(() => {
        sending = surfaces.start?.onSend("hello", [a], { clientMessageId: "cid-gone" })
      })
      await waitFor(() => expect(chatFrames(socket)).toHaveLength(2))
      expect(uploadCalls()).toHaveLength(2)
      expect(chatFrames(socket)[1].file_ids).toEqual([FILE_B])
      accept(socket, "cid-gone")
      await act(async () => { await sending })
    })

    it("refuses a send that cannot succeed before uploading anything", async () => {
      const socket = await mountActive()
      const failures: unknown[] = []

      await act(async () => {
        failures.push(await settle(surfaces.start?.onSend("hi", [pdf("a.exe")], {})))
        failures.push(await settle(surfaces.start?.onSend("hi", [
          new File(["x".repeat(1001)], "big.pdf", { type: "application/pdf" }),
        ], {})))
        failures.push(await settle(surfaces.start?.onSend("hi", [
          pdf("1.pdf"), pdf("2.pdf"), pdf("3.pdf"), pdf("4.pdf"),
        ], {})))
      })

      expect(failures[0]).toMatchObject({ message: "widgetSession.attachments.typeNotSupported", userFacing: true })
      expect(failures[1]).toMatchObject({ errorCode: "upload_too_large" })
      expect(failures[2]).toMatchObject({ message: "widgetSession.attachments.tooManyFiles", userFacing: true })
      expect(uploadCalls()).toHaveLength(0)
      expect(chatFrames(socket)).toHaveLength(0)
    })

    it("reports the connection as not ready when the socket never opens after the upload", async () => {
      render(<SessionAgentChatPage />)
      dispatchSession({ capabilities })
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
      vi.useFakeTimers()
      fetchMock.mockImplementation(async () => created(FILE_A))
      const a = pdf("a.pdf")
      let failure: unknown

      act(() => {
        void surfaces.start?.onSend("hello", [a], { clientMessageId: "cid-slow" })
          .then(() => null, (error: unknown) => { failure = error })
      })
      await act(async () => { await vi.advanceTimersByTimeAsync(95_000) })

      expect(failure).toMatchObject({ errorCode: "connection_not_ready" })
      expect(statusOf(a)).toBe("uploaded")
    })
  })

  describe("sending files from the conversation panel", () => {
    it("uploads the panel's files in order and sends their ids", async () => {
      const socket = await mountActive()
      bindTask(socket)
      await waitFor(() => expect(screen.getByTestId("session-conversation")).toBeInTheDocument())
      const first = deferred<Response>()
      const second = deferred<Response>()
      fetchMock.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise)
      const [a, b] = [pdf("a.pdf"), pdf("b.pdf")]
      let sending!: Promise<unknown>

      act(() => {
        surfaces.panel?.onAttachmentFilesChange([a, b])
        sending = surfaces.panel?.onSend("from panel", { clientMessageId: "cid-panel" }, [a, b])
      })
      await waitFor(() => expect(uploadCalls()).toHaveLength(1))
      expect(uploadCalls()).toHaveLength(1)
      await act(async () => { first.resolve(created(FILE_A, "a.pdf")) })
      await waitFor(() => expect(uploadCalls()).toHaveLength(2))
      await act(async () => { second.resolve(created(FILE_B, "b.pdf")) })
      await waitFor(() => expect(chatFrames(socket)).toHaveLength(1))

      expect(chatFrames(socket)[0]).toMatchObject({ message: "from panel", file_ids: [FILE_A, FILE_B] })
      accept(socket, "cid-panel")
      await act(async () => { await sending })
    })

    it("does not upload or send a file attached to an answer", async () => {
      const socket = await mountActive()
      bindTask(socket)
      await waitFor(() => expect(screen.getByTestId("session-conversation")).toBeInTheDocument())
      let failure: unknown

      await act(async () => {
        failure = await settle(surfaces.panel?.onSend(
          "my answer",
          { clientMessageId: "cid-answer", metadata: { request_id: "inputreq_1" } },
          [pdf("a.pdf")],
        ))
      })

      expect(failure).toMatchObject({
        message: "widgetSession.attachments.noFilesOnAnswer",
        userFacing: true,
      })
      expect(uploadCalls()).toHaveLength(0)
      expect(chatFrames(socket)).toHaveLength(0)
    })

    it("sends a message with no files exactly as before", async () => {
      const socket = await mountActive()
      bindTask(socket)
      await waitFor(() => expect(screen.getByTestId("session-conversation")).toBeInTheDocument())
      let sending!: Promise<unknown>

      act(() => {
        sending = surfaces.panel?.onSend("just text", { clientMessageId: "cid-text" }, [])
      })
      await waitFor(() => expect(chatFrames(socket)).toHaveLength(1))
      accept(socket, "cid-text")
      await act(async () => { await sending })

      expect(chatFrames(socket)[0]).not.toHaveProperty("file_ids")
      expect(uploadCalls()).toHaveLength(0)
    })

    it("ends a send whose file was removed without cancelling a retry running beside it", async () => {
      const socket = await mountActive()
      bindTask(socket)
      await waitFor(() => expect(screen.getByTestId("session-conversation")).toBeInTheDocument())
      const retryReply = deferred<Response>()
      fetchMock
        .mockReturnValueOnce(retryReply.promise)
        .mockImplementationOnce(async () => created(FILE_A, "a.pdf"))
        .mockImplementationOnce(pendingUntilAborted)
      const [r, a, b] = [pdf("r.pdf"), pdf("a.pdf"), pdf("b.pdf")]
      act(() => { surfaces.panel?.onAttachmentFilesChange([r, a, b]) })
      let retrying!: Promise<unknown>
      act(() => { retrying = surfaces.panel?.onRetryAttachment(r) })
      await waitFor(() => expect(uploadCalls()).toHaveLength(1))
      let failure: unknown
      act(() => {
        void surfaces.panel?.onSend("from panel", { clientMessageId: "cid-beside" }, [a, b])
          .then(() => null, (error: unknown) => { failure = error })
      })
      await waitFor(() => expect(uploadCalls()).toHaveLength(3))
      const retrySignal = uploadCalls()[0][1].signal as AbortSignal
      const signalB = uploadCalls()[2][1].signal as AbortSignal

      act(() => { surfaces.panel?.onAttachmentFilesChange([r, b]) })
      await waitFor(() => expect(failure).toBeDefined())

      expect(signalB.aborted).toBe(true)
      expect(retrySignal.aborted).toBe(false)
      await act(async () => {
        retryReply.resolve(created(FILE_B, "r.pdf"))
        await retrying
      })
      expect(surfaces.panel?.attachmentStatuses?.get(r)).toBe("uploaded")
      expect(chatFrames(socket)).toHaveLength(0)
    })

    it("keeps the panel's files while the session is unavailable and sends them once it is back", async () => {
      const socket = await mountActive()
      bindTask(socket)
      await waitFor(() => expect(screen.getByTestId("session-conversation")).toBeInTheDocument())
      fetchMock.mockImplementationOnce(pendingUntilAborted)
      const a = pdf("a.pdf")
      let failure: unknown
      act(() => { surfaces.panel?.onAttachmentFilesChange([a]) })
      act(() => {
        void surfaces.panel?.onSend("hello", { clientMessageId: "cid-panel-away" }, [a])
          .then(() => null, (error: unknown) => { failure = error })
      })
      await waitFor(() => expect(uploadCalls()).toHaveLength(1))

      dispatchDegradedSession()
      await waitFor(() => expect(failure).toBeDefined())
      expect(screen.queryByTestId("session-conversation")).toBeNull()
      // Forget the props the panel had before it unmounted, so the checks below
      // see the panel the page mounts again.
      act(() => { surfaces.panel = null })
      dispatchSession({ session_delivery_id: "delivery-back", capabilities })
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
      const next = MockWebSocket.instances[1]
      act(() => next.open("xagent-session-v1"))

      await waitFor(() => expect(surfaces.panel?.attachmentFiles).toEqual([a]))
      expect(surfaces.panel?.attachmentStatuses?.get(a)).toBeUndefined()
      expect(surfaces.panel?.attachmentsBusy).toBe(false)
      fetchMock.mockImplementationOnce(async () => created(FILE_A, "a.pdf"))
      let sending!: Promise<unknown>
      act(() => {
        sending = surfaces.panel?.onSend("hello", { clientMessageId: "cid-panel-back" }, [a])
      })
      await waitFor(() => expect(chatFrames(next)).toHaveLength(1))
      expect(chatFrames(next)[0].file_ids).toEqual([FILE_A])
      accept(next, "cid-panel-back")
      await act(async () => { await sending })
    })

    it("stops tracking a file the visitor removed from the panel", async () => {
      const socket = await mountActive()
      bindTask(socket)
      await waitFor(() => expect(screen.getByTestId("session-conversation")).toBeInTheDocument())
      fetchMock.mockImplementation((_url: string, init: RequestInit) => new Promise((_resolve, reject) => {
        init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))
      }))
      const a = pdf("a.pdf")
      let sending!: Promise<unknown>
      act(() => {
        surfaces.panel?.onAttachmentFilesChange([a])
        sending = surfaces.panel?.onSend("from panel", { clientMessageId: "cid-rm" }, [a])
          .catch(() => undefined)
      })
      await waitFor(() => expect(uploadCalls()).toHaveLength(1))
      const signal = uploadCalls()[0][1].signal as AbortSignal

      act(() => { surfaces.panel?.onAttachmentFilesChange([]) })
      await act(async () => { await sending })

      expect(signal.aborted).toBe(true)
    })
  })

  describe("with the real start screen, panel and composer", () => {
    const bindTaskWithStatus = (socket: MockWebSocket, status: string) =>
      receive(socket, {
        type: "trace_event",
        timestamp: "2026-05-27T05:00:00Z",
        data: {
          event_id: "task-info-42",
          event_type: "task_info",
          data: {
            id: 42,
            title: "Task 42",
            status,
            description: "Session task",
            created_at: "2026-05-27T05:00:00Z",
            updated_at: "2026-05-27T05:00:00Z",
          },
        },
      })
    const pick = (file: File) => {
      const input = document.querySelector('input[type="file"]') as HTMLInputElement
      act(() => { fireEvent.change(input, { target: { files: [file] } }) })
    }
    const editor = () => screen.getByRole("textbox")
    const submit = (text: string) => {
      act(() => { editor().textContent = text; fireEvent.input(editor()) })
      act(() => { fireEvent.submit(editor().closest("form")!) })
    }
    // Whether the composer shows a card for the file. The sent message's
    // bubble also names its files, so only the composer form is searched.
    const selected = (name: string) => {
      const form = document.querySelector("form")
      return !!form && Array.from(form.querySelectorAll("*"))
        .some((element) => element.children.length === 0 && element.textContent === name)
    }
    const acceptLast = (socket: MockWebSocket) =>
      accept(socket, chatFrames(socket).at(-1)!.client_message_id as string)
    // Replaces the session with one that advertises nothing and opens its
    // socket, then returns that socket.
    const withdraw = async () => {
      dispatchSession({ session_delivery_id: "delivery-withdrawn" })
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
      const next = MockWebSocket.instances[1]
      act(() => next.open("xagent-session-v1"))
      return next
    }

    beforeEach(() => {
      surfaces.real = true
      fetchMock.mockImplementation(async () => new Response("{}", { status: 404 }))
    })

    afterEach(() => {
      surfaces.real = false
    })

    it("types a clicked suggested prompt into the focused composer, then sends it as one chat message", async () => {
      const socket = await mountActive({
        agent: { id: 42, name: "Support Agent", suggested_prompts: ["Show my schedule"] },
      })
      await waitFor(() => expect(editor()).toBeInTheDocument())
      // Focus may stay in the composer when the card is tapped on a phone.
      act(() => { editor().focus() })

      fireEvent.click(screen.getByText("Show my schedule"))

      await waitFor(() => {
        const typed = editor().textContent ?? ""
        expect(typed.length).toBeGreaterThan(0)
        expect(typed.length).toBeLessThan("Show my schedule".length)
      })
      expect(chatFrames(socket)).toHaveLength(0)
      await waitFor(() => expect(chatFrames(socket)).toHaveLength(1))
      expect(chatFrames(socket)[0].message).toBe("Show my schedule")
      expect(editor()).toHaveTextContent(/^Show my schedule$/)
      acceptLast(socket)
      await waitFor(() => expect(editor().textContent).toBe(""))
      expect(chatFrames(socket)).toHaveLength(1)
    })

    it("sends the files picked on the start page with a clicked suggested prompt", async () => {
      const socket = await mountActive({
        capabilities,
        agent: { id: 42, name: "Support Agent", suggested_prompts: ["Show my schedule"] },
      })
      await waitFor(() => expect(editor()).toBeInTheDocument())
      fetchMock.mockResolvedValueOnce(created(FILE_A))
      pick(pdf("a.pdf"))

      fireEvent.click(screen.getByText("Show my schedule"))

      await waitFor(() => expect(chatFrames(socket)).toHaveLength(1))
      expect(uploadCalls()).toHaveLength(1)
      expect(chatFrames(socket)[0]).toEqual(expect.objectContaining({
        message: "Show my schedule",
        file_ids: [FILE_A],
      }))
    })

    it.each(["running", "completed"])(
      "keeps a panel file added while a send with files is in flight (task %s)",
      async (status) => {
        const socket = await mountActive()
        bindTaskWithStatus(socket, status)
        await waitFor(() => expect(editor()).toBeInTheDocument())
        const upload = deferred<Response>()
        fetchMock.mockReturnValueOnce(upload.promise)
        pick(pdf("a.pdf"))
        submit("with a")
        await waitFor(() => expect(uploadCalls()).toHaveLength(1))
        pick(pdf("b.pdf"))
        expect(selected("b.pdf")).toBe(true)

        await act(async () => { upload.resolve(created(FILE_A)) })
        await waitFor(() => expect(chatFrames(socket)).toHaveLength(1))
        acceptLast(socket)
        await waitFor(() => expect(selected("a.pdf")).toBe(false))

        expect(chatFrames(socket)[0].file_ids).toEqual([FILE_A])
        expect(selected("b.pdf")).toBe(true)
      },
    )

    it("keeps a panel file added while a text-only send is in flight", async () => {
      const socket = await mountActive()
      bindTaskWithStatus(socket, "running")
      await waitFor(() => expect(editor()).toBeInTheDocument())
      submit("text only")
      await waitFor(() => expect(chatFrames(socket)).toHaveLength(1))
      pick(pdf("b.pdf"))

      acceptLast(socket)
      // The composer clears the sent draft only after the panel's send settles.
      await waitFor(() => expect(editor().textContent).toBe(""))

      expect(selected("b.pdf")).toBe(true)
    })

    it("keeps a start-page file hidden while uploads are not advertised through a text-only send", async () => {
      await mountActive()
      await waitFor(() => expect(editor()).toBeInTheDocument())
      pick(pdf("a.pdf"))
      const next = await withdraw()
      await waitFor(() => expect(selected("a.pdf")).toBe(false))

      submit("text only")
      await waitFor(() => expect(chatFrames(next)).toHaveLength(1))
      acceptLast(next)
      await waitFor(() => expect(editor().textContent).toBe(""))
      dispatchSession({ session_delivery_id: "delivery-back", capabilities })

      expect(chatFrames(next)[0]).not.toHaveProperty("file_ids")
      await waitFor(() => expect(selected("a.pdf")).toBe(true))
    })

    it("keeps a panel file hidden while uploads are not advertised through a text-only send", async () => {
      const socket = await mountActive()
      bindTaskWithStatus(socket, "running")
      await waitFor(() => expect(editor()).toBeInTheDocument())
      pick(pdf("a.pdf"))
      const next = await withdraw()
      await waitFor(() => expect(selected("a.pdf")).toBe(false))

      submit("text only")
      await waitFor(() => expect(chatFrames(next)).toHaveLength(1))
      acceptLast(next)
      await waitFor(() => expect(editor().textContent).toBe(""))
      dispatchSession({ session_delivery_id: "delivery-back", capabilities })

      await waitFor(() => expect(selected("a.pdf")).toBe(true))
    })

    it("keeps a panel file while the session is unavailable and shows it once the session is back", async () => {
      const socket = await mountActive()
      bindTaskWithStatus(socket, "running")
      await waitFor(() => expect(editor()).toBeInTheDocument())
      pick(pdf("a.pdf"))
      expect(selected("a.pdf")).toBe(true)

      dispatchDegradedSession()
      await waitFor(() => expect(screen.getByText("widgetSession.unavailable.title")).toBeInTheDocument())
      expect(screen.queryByRole("textbox")).toBeNull()
      dispatchSession({ session_delivery_id: "delivery-back", capabilities })
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
      act(() => MockWebSocket.instances[1].open("xagent-session-v1"))
      await waitFor(() => expect(editor()).toBeInTheDocument())

      await waitFor(() => expect(selected("a.pdf")).toBe(true))
    })
  })
})

// Shared by the host-page message suites below.
const ATTACH_CAPABILITIES = {
  type: "connection_capabilities",
  capabilities: ["heartbeat", "conversation_attached"],
}
const ATTACHED = { type: "conversation_attached" }

const mountActive = async () => {
  render(<SessionAgentChatPage />)
  dispatchSession()
  await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
  const socket = MockWebSocket.instances[0]
  act(() => socket.open("xagent-session-v1"))
  return socket
}
const receive = (socket: MockWebSocket, frame: Record<string, unknown>) => {
  act(() => socket.onmessage?.({ data: JSON.stringify(frame) } as MessageEvent))
}
const sentFrames = (socket: MockWebSocket) =>
  socket.send.mock.calls.map((call) => JSON.parse(call[0] as string) as Record<string, unknown>)
const sentTypes = (socket: MockWebSocket) => sentFrames(socket).map((frame) => frame.type)
const taskInfo = (taskId: number) => ({
  type: "trace_event",
  timestamp: "2026-10-28T00:00:00Z",
  data: {
    event_id: `task-info-${taskId}`,
    event_type: "task_info",
    data: {
      id: taskId,
      title: `Task ${taskId}`,
      status: "running",
      description: "Session task",
      created_at: "2026-10-28T00:00:00Z",
      updated_at: "2026-10-28T00:00:00Z",
    },
  },
})
const replayEarlierConversation = (socket: MockWebSocket, extra: Array<Record<string, unknown>> = []) => {
  receive(socket, taskInfo(41))
  receive(socket, {
    type: "trace_event",
    task_id: 41,
    timestamp: "2026-10-28T00:00:01Z",
    data: {
      event_id: "earlier-answer",
      event_type: "agent_message",
      data: { message: "Earlier answer", content: "Earlier answer", role: "assistant" },
    },
  })
  receive(socket, { type: "historical_data_complete", task_id: 41, timestamp: "2026-10-28T00:00:02Z" })
  for (const frame of extra) receive(socket, frame)
}

describe("SessionAgentChatPage opens a new conversation from the host page", () => {
  let warn = vi.fn()

  const deliverOpenMessage = (value: string, id = "1") => {
    act(() => {
      window.dispatchEvent(new MessageEvent("message", {
        data: { xagent: true, v: 1, type: "open_message", id, value },
        source: EMBEDDED_PARENT as unknown as MessageEventSource,
      }))
    })
  }
  const openWarnings = () => warn.mock.calls.filter(([text]) => String(text).startsWith("Xagent Widget:"))

  beforeEach(() => {
    MockWebSocket.instances = []
    MockWebSocket.failConstruction = false
    MockWebSocket.constructionCount = 0
    localStorage.clear()
    sessionStorage.clear()
    vi.stubGlobal("WebSocket", MockWebSocket)
    vi.spyOn(console, "log").mockImplementation(() => {})
    warn = vi.fn()
    vi.spyOn(console, "warn").mockImplementation(warn)
    Object.defineProperty(window, "parent", { configurable: true, value: EMBEDDED_PARENT })
  })

  afterEach(() => {
    cleanup()
    surfaces.real = false
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    if (originalParentDescriptor) Object.defineProperty(window, "parent", originalParentDescriptor)
  })

  it("sends nothing until the socket attaches, then sends the message as the first one", async () => {
    const socket = await mountActive()
    receive(socket, ATTACH_CAPABILITIES)
    deliverOpenMessage("Book a shift")
    expect(sentTypes(socket)).toEqual([])

    receive(socket, ATTACHED)

    await waitFor(() => expect(sentTypes(socket)).toEqual(["chat"]))
    expect(sentFrames(socket)[0].message).toBe("Book a shift")
    expect(openWarnings()).toEqual([])
  })

  it.each([
    ["has finished its answer", []],
    ["is waiting for the visitor", [{
      type: "task_waiting_for_user",
      task_id: 41,
      timestamp: "2026-10-28T00:00:03Z",
      question: "Which day?",
      interactions: [],
      request_id: "inputreq_0011223344556677889900aabbccddee",
    }]],
  ] as const)("replaces a replayed conversation that %s", async (_name, extra) => {
    surfaces.real = true
    vi.stubGlobal("fetch", vi.fn(async () => new Response("{}", { status: 404 })))
    const socket = await mountActive()
    receive(socket, ATTACH_CAPABILITIES)
    deliverOpenMessage("Book a shift")
    replayEarlierConversation(socket, [...extra])
    expect(sentTypes(socket)).toEqual([])

    receive(socket, ATTACHED)
    await waitFor(() => expect(sentTypes(socket)).toEqual(["new_conversation"]))
    receive(socket, { type: "conversation_reset" })
    await waitFor(() => expect(sentTypes(socket)).toEqual(["new_conversation", "chat"]))
    const chat = sentFrames(socket)[1]
    expect(chat.message).toBe("Book a shift")
    receive(socket, taskInfo(42))
    receive(socket, { type: "message_accepted", client_message_id: chat.client_message_id })

    await waitFor(() => expect(screen.getByText("Book a shift")).toBeInTheDocument())
    expect(screen.queryByText("Earlier answer")).not.toBeInTheDocument()
    expect(sentTypes(socket)).toEqual(["new_conversation", "chat"])
    expect(openWarnings()).toEqual([])
  })

  it("waits for the next socket to attach after a reconnect", async () => {
    vi.spyOn(window, "postMessage").mockImplementation(() => {})
    const first = await mountActive()
    receive(first, ATTACH_CAPABILITIES)
    receive(first, ATTACHED)
    act(() => first.triggerClose())
    dispatchSession({ session_delivery_id: "delivery-b" })
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2))
    const second = MockWebSocket.instances[1]
    act(() => second.open("xagent-session-v1"))
    receive(second, ATTACH_CAPABILITIES)

    deliverOpenMessage("Book a shift")
    expect(sentTypes(second)).toEqual([])
    receive(second, taskInfo(41))
    expect(sentTypes(second)).toEqual([])

    receive(second, ATTACHED)

    await waitFor(() => expect(sentTypes(second)).toEqual(["new_conversation"]))
    expect(sentTypes(first)).toEqual([])
  })

  it("drops the message when the server does not announce attach completion", async () => {
    const socket = await mountActive()
    receive(socket, { type: "connection_capabilities", capabilities: ["heartbeat"] })

    deliverOpenMessage("Book a shift")

    await waitFor(() => expect(openWarnings()).toEqual([[W_UNSUPPORTED]]))
    expect(sentTypes(socket)).toEqual([])
  })

  it("offers the attach marker alongside the liveness marker", async () => {
    render(<SessionAgentChatPage />)
    dispatchSession()
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))

    expect(MockWebSocket.instances[0].protocols).toEqual([
      "xagent-session-v1",
      `xagent-session-token.${SESSION_TOKEN}`,
      "xagent-session-liveness-v1",
      "xagent-session-attach-v1",
      "xagent-session-binding-v1",
      "xagent-session-task.unbound",
    ])
  })
})

describe("SessionAgentChatPage sends the host page's initial message", () => {
  const INITIAL_MESSAGE = "Hi, I'd like a quote"
  let initialRequests = 0

  // Acts as widget.js after the visitor has already opened the panel: it
  // answers every initial_message_request with the message, as soon as a
  // posted message can arrive.
  const answerInitialMessageRequests = () => {
    const forward = EMBEDDED_PARENT.postMessage
    vi.spyOn(EMBEDDED_PARENT, "postMessage").mockImplementation((...args) => {
      const [data] = args as [Record<string, unknown>]
      if (data?.type !== "initial_message_request") return forward(...args)
      initialRequests += 1
      setTimeout(() => {
        window.dispatchEvent(new MessageEvent("message", {
          data: { xagent: true, v: 1, type: "initial_message", value: INITIAL_MESSAGE },
          source: EMBEDDED_PARENT as unknown as MessageEventSource,
        }))
      }, 0)
    })
  }

  beforeEach(() => {
    MockWebSocket.instances = []
    MockWebSocket.failConstruction = false
    MockWebSocket.constructionCount = 0
    localStorage.clear()
    sessionStorage.clear()
    vi.stubGlobal("WebSocket", MockWebSocket)
    vi.spyOn(console, "log").mockImplementation(() => {})
    Object.defineProperty(window, "parent", { configurable: true, value: EMBEDDED_PARENT })
    initialRequests = 0
    answerInitialMessageRequests()
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    if (originalParentDescriptor) Object.defineProperty(window, "parent", originalParentDescriptor)
  })

  // Long enough for the answer to a request, had one been posted, to arrive.
  const letHostAnswer = () => act(() => new Promise<void>((resolve) => setTimeout(resolve, 20)))

  it("sends nothing into a conversation the server restores", async () => {
    const socket = await mountActive()
    receive(socket, ATTACH_CAPABILITIES)
    // The replay follows the open, so the host page may answer before it.
    await letHostAnswer()
    replayEarlierConversation(socket)
    receive(socket, ATTACHED)
    await letHostAnswer()

    expect(sentTypes(socket)).toEqual([])
    expect(initialRequests).toBe(0)
  })

  it("sends the message to a visitor without a conversation once the socket attaches", async () => {
    const socket = await mountActive()
    receive(socket, ATTACH_CAPABILITIES)
    expect(initialRequests).toBe(0)
    expect(sentTypes(socket)).toEqual([])

    receive(socket, ATTACHED)

    await waitFor(() => expect(sentTypes(socket)).toEqual(["chat"]))
    expect(sentFrames(socket)[0].message).toBe(INITIAL_MESSAGE)
    expect(initialRequests).toBe(1)
  })

  // The page requires a server that sends connection_capabilities. A missing
  // frame is never read as a server without attach completion: on an
  // attach-capable server whose frame is late, that guess would ask before a
  // restored conversation arrives.
  it("never asks while the open socket has sent no capabilities frame, however long it stays silent", async () => {
    // Faked from before the mount, so a timer armed when the socket opens is
    // covered too. Time still flows, so the mount's own waits work as usual.
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      const socket = await mountActive()

      await act(() => vi.advanceTimersByTimeAsync(10 * 60_000))
      await letHostAnswer()

      expect(initialRequests).toBe(0)
      expect(sentTypes(socket)).toEqual([])
    } finally {
      vi.useRealTimers()
    }
  })

  it("never asks while an attach-capable socket has not attached, however long it stays silent", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      const socket = await mountActive()
      // Without heartbeat, so dead-connection detection does not close the
      // silent socket after 30 seconds and hide a longer timer.
      receive(socket, { type: "connection_capabilities", capabilities: ["conversation_attached"] })

      await act(() => vi.advanceTimersByTimeAsync(10 * 60_000))
      await letHostAnswer()

      expect(initialRequests).toBe(0)
      expect(sentTypes(socket)).toEqual([])
    } finally {
      vi.useRealTimers()
    }
  })

  it("sends the message on the open socket when the server does not announce attach completion", async () => {
    const socket = await mountActive()
    expect(initialRequests).toBe(0)

    receive(socket, { type: "connection_capabilities", capabilities: ["heartbeat"] })

    await waitFor(() => expect(sentTypes(socket)).toEqual(["chat"]))
    expect(sentFrames(socket)[0].message).toBe(INITIAL_MESSAGE)
  })
})

describe("SessionAgentChatPage continues from an earlier conversation", () => {
  const CONTINUE = "Continue this"
  const HISTORY = { conversation_history: {} }
  const fetchMock = vi.fn()
  let warn = vi.fn()
  let initialRequests = 0

  const mountWithHistory = async (overrides: Record<string, unknown> = {}) => {
    render(<SessionAgentChatPage />)
    dispatchSession({ capabilities: HISTORY, ...overrides })
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open("xagent-session-v1"))
    receive(socket, ATTACH_CAPABILITIES)
    return socket
  }
  // The visitor opens an earlier conversation from the menu and reaches the
  // point where the view hands a message over.
  const openEarlierConversation = () => {
    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    fireEvent.click(screen.getByRole("menuitem", { name: "widgetSession.history.menuLabel" }))
    fireEvent.click(screen.getByRole("button", { name: "pick-earlier" }))
  }
  const handOver = () => fireEvent.click(screen.getByRole("button", { name: "hand-over" }))
  const deliverOpenMessage = (value: string) => act(() => {
    window.dispatchEvent(new MessageEvent("message", {
      data: { xagent: true, v: 1, type: "open_message", id: "1", value },
      source: EMBEDDED_PARENT as unknown as MessageEventSource,
    }))
  })
  const chats = (socket: MockWebSocket) => sentFrames(socket).filter((frame) => frame.type === "chat")
  const lastChatId = (socket: MockWebSocket) => chats(socket).at(-1)!.client_message_id as string
  const accept = (socket: MockWebSocket, taskId: number) => {
    receive(socket, taskInfo(taskId))
    receive(socket, { type: "message_accepted", client_message_id: lastChatId(socket) })
  }
  const reject = (socket: MockWebSocket, fields: Record<string, unknown>) =>
    receive(socket, { type: "message_rejected", client_message_id: lastChatId(socket), ...fields })
  // Settles pending work, long enough for a frame that was going to be sent to be.
  const letItRun = () => act(() => new Promise<void>((resolve) => setTimeout(resolve, 20)))
  const asParent = () => {
    const forward = EMBEDDED_PARENT.postMessage
    vi.spyOn(EMBEDDED_PARENT, "postMessage").mockImplementation((...args) => {
      const [data] = args as [Record<string, unknown>]
      if (data?.type !== "initial_message_request") return forward(...args)
      initialRequests += 1
      setTimeout(() => {
        window.dispatchEvent(new MessageEvent("message", {
          data: { xagent: true, v: 1, type: "initial_message", value: "Hello from the host page" },
          source: EMBEDDED_PARENT as unknown as MessageEventSource,
        }))
      }, 0)
    })
  }

  beforeEach(() => {
    MockWebSocket.instances = []
    MockWebSocket.failConstruction = false
    MockWebSocket.constructionCount = 0
    historySurfaces.real = false
    surfaces.real = false
    fetchMock.mockReset()
    fetchMock.mockImplementation(async () => new Response("{}", { status: 404 }))
    localStorage.clear()
    sessionStorage.clear()
    vi.stubGlobal("WebSocket", MockWebSocket)
    vi.stubGlobal("fetch", fetchMock)
    vi.spyOn(console, "log").mockImplementation(() => {})
    warn = vi.fn()
    vi.spyOn(console, "warn").mockImplementation(warn)
    initialRequests = 0
    Object.defineProperty(window, "parent", { configurable: true, value: EMBEDDED_PARENT })
  })

  afterEach(() => {
    cleanup()
    historySurfaces.real = false
    surfaces.real = false
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    if (originalParentDescriptor) Object.defineProperty(window, "parent", originalParentDescriptor)
  })

  describe("the first message of the new conversation", () => {
    it("is sent straight away, with the source, when no conversation is open", async () => {
      const socket = await mountWithHistory()
      receive(socket, ATTACHED)
      openEarlierConversation()

      handOver()

      await waitFor(() => expect(sentTypes(socket)).toEqual(["chat"]))
      expect(sentFrames(socket)[0]).toMatchObject({ message: CONTINUE, source_task_id: 41 })
      expect(screen.queryByRole("button", { name: "hand-over" })).not.toBeInTheDocument()
    })

    it("is sent after a reset when a conversation is open", async () => {
      const socket = await mountWithHistory()
      replayEarlierConversation(socket)
      receive(socket, ATTACHED)
      openEarlierConversation()

      handOver()

      await waitFor(() => expect(sentTypes(socket)).toEqual(["new_conversation"]))
      receive(socket, { type: "conversation_reset" })
      await waitFor(() => expect(sentTypes(socket)).toEqual(["new_conversation", "chat"]))
      expect(sentFrames(socket)[1]).toMatchObject({ message: CONTINUE, source_task_id: 41 })
    })

    it("is sent without another reset when the visitor already started a new conversation", async () => {
      const socket = await mountWithHistory()
      replayEarlierConversation(socket)
      receive(socket, ATTACHED)
      fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
      fireEvent.click(screen.getByRole("menuitem", { name: "widgetSession.startNewConversation" }))
      await waitFor(() => expect(sentTypes(socket)).toEqual(["new_conversation"]))
      receive(socket, { type: "conversation_reset" })
      await waitFor(() => expect(screen.queryByText("widgetSession.resetting")).not.toBeInTheDocument())
      openEarlierConversation()

      handOver()

      await waitFor(() => expect(sentTypes(socket)).toEqual(["new_conversation", "chat"]))
      expect(sentFrames(socket)[1]).toMatchObject({ message: CONTINUE, source_task_id: 41 })
    })

    it("waits until the socket has attached", async () => {
      const socket = await mountWithHistory()
      openEarlierConversation()

      handOver()
      await letItRun()
      expect(sentTypes(socket)).toEqual([])
      receive(socket, ATTACHED)

      await waitFor(() => expect(sentTypes(socket)).toEqual(["chat"]))
    })

    it("is not sent, and the visitor is told, when the connection is lost while the new conversation starts", async () => {
      vi.spyOn(window, "postMessage").mockImplementation(() => {})
      const socket = await mountWithHistory()
      replayEarlierConversation(socket)
      receive(socket, ATTACHED)
      openEarlierConversation()
      handOver()
      await waitFor(() => expect(sentTypes(socket)).toEqual(["new_conversation"]))

      act(() => socket.triggerClose())

      expect(await screen.findByText("widgetSession.reloadRequired")).toBeInTheDocument()
      await letItRun()
      expect(sentTypes(socket)).toEqual(["new_conversation"])
      fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
      expect(screen.queryByRole("menuitem", { name: "widgetSession.history.menuLabel" })).not.toBeInTheDocument()
    })
  })

  describe("when the server refuses the message after the source was checked", () => {
    const starts = ["unbound", "bound", "replacement_ready"] as const
    type Start = (typeof starts)[number]
    // Brings the page to the start phase, opens an earlier conversation and
    // hands a message over; returns the socket once the chat frame is out.
    const handOverFrom = async (start: Start) => {
      const socket = await mountWithHistory()
      if (start !== "unbound") replayEarlierConversation(socket)
      receive(socket, ATTACHED)
      if (start === "replacement_ready") {
        fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
        fireEvent.click(screen.getByRole("menuitem", { name: "widgetSession.startNewConversation" }))
        await waitFor(() => expect(sentTypes(socket)).toEqual(["new_conversation"]))
        receive(socket, { type: "conversation_reset" })
        await waitFor(() => expect(screen.queryByText("widgetSession.resetting")).not.toBeInTheDocument())
      }
      openEarlierConversation()
      handOver()
      if (start === "bound") {
        await waitFor(() => expect(sentTypes(socket)).toEqual(["new_conversation"]))
        receive(socket, { type: "conversation_reset" })
      }
      await waitFor(() => expect(chats(socket)).toHaveLength(1))
      return socket
    }

    it.each(starts.flatMap((start) => [
      ["task_unavailable", "widgetSession.history.continueUnavailable"],
      ["invalid_message", "clientErrors.invalidMessage"],
      // A refusal whose code the page does not know keeps the line the page
      // already shows for any refusal without a usable code.
      ["not_a_code", "clientErrors.messageNotAccepted"],
    ].map(([code, text]) => [start, code, text] as const)))(
      "starting from %s, a refusal with %s shows the right line on the start screen",
      async (start, code, text) => {
        const socket = await handOverFrom(start)

        reject(socket, { rejection_outcome: "not_accepted", error_code: code })

        expect(await screen.findByRole("alert")).toHaveTextContent(text)
        expect(screen.queryByRole("button", { name: "hand-over" })).not.toBeInTheDocument()
        expect(screen.getByTestId("session-start")).toBeInTheDocument()
        // The visitor's words are gone and nothing is sent again by itself.
        await letItRun()
        expect(chats(socket)).toHaveLength(1)
      },
    )

    it.each(starts.map((start) => [start] as const))(
      "starting from %s, a storage failure behind the source is shown as the right outcome",
      async (start) => {
        const socket = await handOverFrom(start)

        reject(socket, { error_code: "message_processing_failed" })

        if (start === "unbound") {
          expect(await screen.findByRole("alert")).toHaveTextContent("clientErrors.messageProcessingFailed")
        } else {
          expect(await screen.findByRole("alert")).toHaveTextContent("widgetSession.reloadRequired")
          expect(screen.queryByText("clientErrors.messageProcessingFailed")).not.toBeInTheDocument()
        }
      },
    )
  })

  describe("with a host page message", () => {
    // Where in the hand-over the host page's open message arrives.
    const moments = [
      ["the message is still waiting for the replay", "slot"],
      ["the reset has been asked for", "reset"],
      ["the reset is confirmed in the same render", "confirmed"],
      ["the message has been sent", "sent"],
    ] as const

    it.each(moments)("sends the hand-over first and the host message after it when %s", async (_label, moment) => {
      const socket = await mountWithHistory()
      replayEarlierConversation(socket)
      if (moment !== "slot") receive(socket, ATTACHED)
      openEarlierConversation()
      handOver()
      if (moment === "slot") {
        deliverOpenMessage("hi")
        receive(socket, ATTACHED)
      }
      await waitFor(() => expect(sentTypes(socket)).toEqual(["new_conversation"]))
      if (moment === "reset") deliverOpenMessage("hi")
      if (moment === "confirmed") {
        act(() => {
          socket.onmessage?.({ data: JSON.stringify({ type: "conversation_reset" }) } as MessageEvent)
          window.dispatchEvent(new MessageEvent("message", {
            data: { xagent: true, v: 1, type: "open_message", id: "1", value: "hi" },
            source: EMBEDDED_PARENT as unknown as MessageEventSource,
          }))
        })
      } else {
        receive(socket, { type: "conversation_reset" })
      }
      await waitFor(() => expect(chats(socket)).toHaveLength(1))
      if (moment === "sent") deliverOpenMessage("hi")
      expect(sentTypes(socket)).toEqual(["new_conversation", "chat"])
      expect(chats(socket)[0]).toMatchObject({ message: CONTINUE, source_task_id: 41 })

      accept(socket, 42)

      await waitFor(() => expect(sentTypes(socket)).toEqual(["new_conversation", "chat", "new_conversation"]))
      receive(socket, { type: "conversation_reset" })
      await waitFor(() => expect(chats(socket)).toHaveLength(2))
      expect(chats(socket)[1].message).toBe("hi")
      expect(chats(socket)[1]).not.toHaveProperty("source_task_id")
      accept(socket, 43)
      await letItRun()
      // Both messages are done: no frame follows, and history can be opened again.
      expect(sentTypes(socket)).toEqual(["new_conversation", "chat", "new_conversation", "chat"])
      fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
      expect(screen.getByRole("menuitem", { name: "widgetSession.history.menuLabel" })).toBeInTheDocument()
      expect(warn.mock.calls.filter(([text]) => String(text).startsWith("Xagent Widget:"))).toEqual([])
    })

    it("lets neither driver send while both arrive in the same render", async () => {
      const socket = await mountWithHistory()
      replayEarlierConversation(socket)
      receive(socket, ATTACHED)
      openEarlierConversation()

      act(() => {
        handOver()
        window.dispatchEvent(new MessageEvent("message", {
          data: { xagent: true, v: 1, type: "open_message", id: "1", value: "hi" },
          source: EMBEDDED_PARENT as unknown as MessageEventSource,
        }))
      })

      await waitFor(() => expect(sentTypes(socket)).toEqual(["new_conversation"]))
      await letItRun()
      expect(sentTypes(socket)).toEqual(["new_conversation"])
    })

    it("takes the visitor back to the chat when the message arrives", async () => {
      const socket = await mountWithHistory()
      receive(socket, ATTACHED)
      openEarlierConversation()
      expect(screen.getByRole("button", { name: "hand-over" })).toBeInTheDocument()

      deliverOpenMessage("hi")

      expect(screen.queryByRole("button", { name: "hand-over" })).not.toBeInTheDocument()
      await waitFor(() => expect(sentTypes(socket)).toEqual(["chat"]))
      expect(sentFrames(socket)[0]).toMatchObject({ message: "hi" })
      expect(sentFrames(socket)[0]).not.toHaveProperty("source_task_id")
    })

    it("takes a visitor reading an earlier conversation back to the chat and starts a new conversation when the message arrives", async () => {
      const socket = await mountWithHistory()
      replayEarlierConversation(socket)
      receive(socket, ATTACHED)
      openEarlierConversation()
      expect(screen.getByRole("button", { name: "hand-over" })).toBeInTheDocument()

      deliverOpenMessage("hi")

      expect(screen.queryByRole("button", { name: "hand-over" })).not.toBeInTheDocument()
      await waitFor(() => expect(sentTypes(socket)).toEqual(["new_conversation"]))
      receive(socket, { type: "conversation_reset" })
      await waitFor(() => expect(sentTypes(socket)).toEqual(["new_conversation", "chat"]))
      expect(sentFrames(socket)[1]).toMatchObject({ message: "hi" })
      expect(sentFrames(socket)[1]).not.toHaveProperty("source_task_id")
      expect(screen.queryByRole("button", { name: "hand-over" })).not.toBeInTheDocument()
      expect(screen.getByTestId("session-start")).toBeInTheDocument()
    })

    it("takes a visitor looking at the list of earlier conversations back to the chat when the message arrives", async () => {
      const socket = await mountWithHistory()
      receive(socket, ATTACHED)
      fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
      fireEvent.click(screen.getByRole("menuitem", { name: "widgetSession.history.menuLabel" }))
      expect(screen.getByRole("button", { name: "pick-earlier" })).toBeInTheDocument()

      deliverOpenMessage("hi")

      expect(screen.queryByRole("button", { name: "pick-earlier" })).not.toBeInTheDocument()
      await waitFor(() => expect(sentTypes(socket)).toEqual(["chat"]))
      expect(sentFrames(socket)[0]).toMatchObject({ message: "hi" })
    })
  })

  describe("with the host page's initial message", () => {
    it("takes a visitor looking at the list of earlier conversations back to the chat when it arrives", async () => {
      asParent()
      const socket = await mountWithHistory()
      fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
      fireEvent.click(screen.getByRole("menuitem", { name: "widgetSession.history.menuLabel" }))
      expect(screen.getByRole("button", { name: "pick-earlier" })).toBeInTheDocument()

      receive(socket, ATTACHED)

      await waitFor(() => expect(sentFrames(socket).map((frame) => frame.message)).toEqual(["Hello from the host page"]))
      expect(screen.queryByRole("button", { name: "pick-earlier" })).not.toBeInTheDocument()
    })
    it.each([
      ["accepted", true],
      ["refused", false],
    ] as const)("is never requested once a hand-over has begun, even when it is %s", async (_label, accepted) => {
      asParent()
      const socket = await mountWithHistory()
      openEarlierConversation()
      handOver()
      expect(initialRequests).toBe(0)

      receive(socket, ATTACHED)
      await waitFor(() => expect(chats(socket)).toHaveLength(1))
      if (accepted) accept(socket, 42)
      else reject(socket, { rejection_outcome: "not_accepted", error_code: "task_unavailable" })
      await letItRun()

      expect(initialRequests).toBe(0)
      expect(chats(socket)).toHaveLength(1)
    })

    // The page promises that no message reaches the wrong conversation, not
    // that every message is delivered. When the host page's reply lands in the
    // very render in which the hand-over is placed, both frames go out in
    // order; the second carries a source while the first has already opened a
    // conversation, so the session server refuses it as invalid and the
    // visitor's follow-up is lost without a notice.
    it("loses the hand-over, quietly, when the host page's reply lands in the render that places it", async () => {
      let answer: (() => void) | null = null
      vi.spyOn(EMBEDDED_PARENT, "postMessage").mockImplementation((...args) => {
        const [data] = args as [Record<string, unknown>]
        if (data?.type !== "initial_message_request") return
        answer = () => window.dispatchEvent(new MessageEvent("message", {
          data: { xagent: true, v: 1, type: "initial_message", value: "Hello from the host page" },
          source: EMBEDDED_PARENT as unknown as MessageEventSource,
        }))
      })
      const socket = await mountWithHistory()
      receive(socket, ATTACHED)
      await waitFor(() => expect(answer).not.toBeNull())
      openEarlierConversation()

      act(() => {
        handOver()
        answer?.()
      })

      await waitFor(() => expect(chats(socket)).toHaveLength(2))
      expect(chats(socket)[0]).toMatchObject({ message: "Hello from the host page" })
      expect(chats(socket)[0]).not.toHaveProperty("source_task_id")
      expect(chats(socket)[1]).toMatchObject({ message: CONTINUE, source_task_id: 41 })
      receive(socket, taskInfo(42))
      receive(socket, { type: "message_accepted", client_message_id: chats(socket)[0].client_message_id })
      receive(socket, {
        type: "message_rejected",
        client_message_id: chats(socket)[1].client_message_id,
        rejection_outcome: "not_accepted",
        error_code: "invalid_message",
      })
      await letItRun()

      expect(screen.queryByRole("alert")).not.toBeInTheDocument()
      expect(chats(socket)).toHaveLength(2)
      expect(sentTypes(socket)).toEqual(["chat", "chat"])
    })

    it("takes the visitor back to the chat when it arrives", async () => {
      asParent()
      const socket = await mountWithHistory()
      openEarlierConversation()

      receive(socket, ATTACHED)

      await waitFor(() => expect(sentFrames(socket).map((frame) => frame.message)).toEqual(["Hello from the host page"]))
      expect(screen.queryByRole("button", { name: "hand-over" })).not.toBeInTheDocument()
    })
  })

  describe("reading history beside the conversation", () => {
    const json = (body: unknown) => new Response(JSON.stringify(body), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })

    it("leaves the conversation, the connection and the token alone while the visitor reads", async () => {
      historySurfaces.real = true
      surfaces.real = true
      const postMessage = vi.spyOn(window, "postMessage").mockImplementation(() => {})
      fetchMock.mockImplementation(async (url: string) => {
        if (String(url).includes("/conversations/40/messages")) {
          return json({
            messages: [{ id: 1, role: "user", content: "An old question", created_at: "2026-01-01T00:00:00Z", attachments: [] }],
            next_cursor: null,
          })
        }
        return json({
          conversations: [
            { task_id: 41, title: "Current chat", created_at: "2026-01-02T00:00:00Z" },
            { task_id: 40, title: "Older chat", created_at: "2026-01-01T00:00:00Z" },
          ],
          next_cursor: null,
        })
      })
      const socket = await mountWithHistory()
      replayEarlierConversation(socket)
      receive(socket, ATTACHED)
      await screen.findByText("Earlier answer")

      fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
      fireEvent.click(screen.getByRole("menuitem", { name: "widgetSession.history.menuLabel" }))
      fireEvent.click(await screen.findByRole("button", { name: /Older chat/ }))
      await screen.findByText("An old question")
      fireEvent.click(screen.getByRole("button", { name: "widgetSession.history.back" }))
      fireEvent.click(await screen.findByRole("button", { name: /Current chat/ }))

      expect(await screen.findByText("Earlier answer")).toBeVisible()
      const reads = fetchMock.mock.calls.filter(([url]) => String(url).includes("/v1/external/chat/sessions/conversations"))
      expect(reads.length).toBeGreaterThanOrEqual(2)
      for (const [url, init] of reads) {
        expect((init as RequestInit).headers).toEqual({ Authorization: `Bearer ${SESSION_TOKEN}` })
        expect((init as RequestInit).credentials).toBe("omit")
        expect(String(url)).not.toContain(SESSION_TOKEN)
      }
      expect(sentFrames(socket)).toEqual([])
      expect(MockWebSocket.instances).toHaveLength(1)
      expect(socket.close).not.toHaveBeenCalled()
      expect(postMessage.mock.calls.filter(([message]) => (message as { type?: string })?.type === "reconnect_request")).toEqual([])
    })

    it("keeps the draft in the same composer node across a visit to history", async () => {
      surfaces.real = true
      const socket = await mountWithHistory()
      replayEarlierConversation(socket)
      receive(socket, ATTACHED)
      await waitFor(() => expect(screen.getByRole("textbox")).toBeInTheDocument())
      const composer = screen.getByRole("textbox")
      act(() => { composer.textContent = "half typed"; fireEvent.input(composer) })

      fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
      fireEvent.click(screen.getByRole("menuitem", { name: "widgetSession.history.menuLabel" }))

      expect(screen.getByRole("textbox", { hidden: true })).toBe(composer)
      const layer = composer.closest("[aria-hidden='true']") as HTMLElement
      expect(layer.style.visibility).toBe("hidden")
      expect(layer).not.toBeVisible()
      fireEvent.click(screen.getByRole("button", { name: "pick-earlier" }))
      fireEvent.click(screen.getByRole("button", { name: "back-to-list" }))
      fireEvent.click(screen.getByRole("button", { name: "back-to-chat" }))

      expect(screen.getByRole("textbox")).toBe(composer)
      expect(composer).toHaveTextContent("half typed")
      expect(layer).toBeVisible()
    })

    it("keeps the draft on the start page across a visit to history", async () => {
      surfaces.real = true
      const socket = await mountWithHistory()
      receive(socket, ATTACHED)
      await waitFor(() => expect(screen.getByRole("textbox")).toBeInTheDocument())
      const composer = screen.getByRole("textbox")
      act(() => { composer.textContent = "half typed"; fireEvent.input(composer) })

      fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
      fireEvent.click(screen.getByRole("menuitem", { name: "widgetSession.history.menuLabel" }))
      fireEvent.click(screen.getByRole("button", { name: "back-to-chat" }))

      expect(screen.getByRole("textbox")).toBe(composer)
      expect(composer).toHaveTextContent("half typed")
    })

    it("keeps a file picked on the start page selected across a visit to history", async () => {
      surfaces.real = true
      const socket = await mountWithHistory({
        capabilities: {
          ...HISTORY,
          file_uploads: { allowed_extensions: [".pdf"], max_file_bytes: 1000, max_files_per_message: 3 },
        },
      })
      receive(socket, ATTACHED)
      await waitFor(() => expect(screen.getByRole("textbox")).toBeInTheDocument())
      const input = document.querySelector('input[type="file"]') as HTMLInputElement
      const file = new File(["abc"], "a.pdf", { type: "application/pdf" })
      act(() => { fireEvent.change(input, { target: { files: [file] } }) })
      const selected = () => {
        const form = document.querySelector("form")
        return !!form && Array.from(form.querySelectorAll("*"))
          .some((element) => element.children.length === 0 && element.textContent === "a.pdf")
      }
      expect(selected()).toBe(true)

      fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
      fireEvent.click(screen.getByRole("menuitem", { name: "widgetSession.history.menuLabel" }))
      fireEvent.click(screen.getByRole("button", { name: "back-to-chat" }))

      expect(selected()).toBe(true)
    })

    it("keeps taking replies for the conversation while history is open", async () => {
      surfaces.real = true
      const socket = await mountWithHistory()
      replayEarlierConversation(socket)
      receive(socket, ATTACHED)
      await screen.findByText("Earlier answer")
      fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
      fireEvent.click(screen.getByRole("menuitem", { name: "widgetSession.history.menuLabel" }))

      receive(socket, {
        type: "trace_event",
        task_id: 41,
        timestamp: "2026-10-28T00:00:05Z",
        data: {
          event_id: "late-answer",
          event_type: "agent_message",
          data: { message: "A late answer", content: "A late answer", role: "assistant" },
        },
      })
      fireEvent.click(screen.getByRole("button", { name: "back-to-chat" }))

      expect(await screen.findByText("A late answer")).toBeVisible()
    })

    it("does not move focus into a composer that appears while history is open, and focuses it on return", async () => {
      surfaces.real = true
      const socket = await mountWithHistory()
      receive(socket, ATTACHED)
      await waitFor(() => expect(screen.getByRole("textbox")).toBeInTheDocument())
      fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
      fireEvent.click(screen.getByRole("menuitem", { name: "widgetSession.history.menuLabel" }))
      act(() => (document.activeElement as HTMLElement | null)?.blur())

      receive(socket, taskInfo(42))
      await letItRun()
      expect(screen.getByRole("textbox", { hidden: true })).not.toHaveFocus()

      fireEvent.click(screen.getByRole("button", { name: "back-to-chat" }))

      await waitFor(() => expect(screen.getByRole("textbox")).toHaveFocus())
    })
  })
})

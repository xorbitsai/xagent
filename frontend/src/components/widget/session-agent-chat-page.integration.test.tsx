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

import { SessionAgentChatPage } from "./session-agent-chat-page"

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

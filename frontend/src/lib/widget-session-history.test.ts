import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  HISTORY_REQUEST_TIMEOUT_MS,
  WIDGET_SESSION_CONVERSATIONS_PATH,
  WidgetSessionHistoryError,
  listWidgetConversations,
  readWidgetConversation,
  type WidgetSessionHistoryAuth,
  type WidgetSessionHistoryErrorKind,
} from "@/lib/widget-session-history"

const ORIGIN = "https://chat.example.test"
const CURRENT_TOKEN = "st_current_token"
const REFRESHED_TOKEN = "st_refreshed_token"

const fetchMock = vi.fn<(input: string, init?: RequestInit) => Promise<Response>>()

const jsonResponse = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })

const LIST_BODY = {
  conversations: [
    { task_id: 101, title: "Support Agent chat", created_at: "2026-01-02T03:04:05Z" },
    { task_id: 99, title: "", created_at: "2026-01-01T00:00:00Z" },
  ],
  next_cursor: "cursor-2",
}
const LIST_RESULT = {
  conversations: [
    { taskId: 101, title: "Support Agent chat", createdAt: "2026-01-02T03:04:05Z" },
    { taskId: 99, title: "", createdAt: "2026-01-01T00:00:00Z" },
  ],
  nextCursor: "cursor-2",
}

const MESSAGES_BODY = {
  messages: [
    {
      id: 1,
      role: "user",
      content: "Where is my order?",
      created_at: "2026-01-02T03:04:05Z",
      attachments: [
        {
          file_id: "0b1e5c9e-3f1a-4a63-9b0e-6c1d2f3a4b5c",
          filename: "receipt.pdf",
          file_size: 11,
          mime_type: "application/pdf",
        },
        { file_id: "7d2f0c1a-8b3e-4c55-a1f0-2e9d4b6c8a10", filename: "note.txt", file_size: null, mime_type: null },
      ],
    },
    { id: 2, role: "assistant", content: "It ships today.", created_at: "2026-01-02T03:04:06Z", attachments: [] },
  ],
  next_cursor: null,
}
// The file ids stay behind: nothing that reads a conversation needs them.
const MESSAGES_RESULT = {
  messages: [
    {
      id: 1,
      role: "user",
      content: "Where is my order?",
      created_at: "2026-01-02T03:04:05Z",
      attachments: [
        { filename: "receipt.pdf", file_size: 11, mime_type: "application/pdf" },
        { filename: "note.txt", file_size: null, mime_type: null },
      ],
    },
    { id: 2, role: "assistant", content: "It ships today.", created_at: "2026-01-02T03:04:06Z", attachments: [] },
  ],
  nextCursor: null,
}

interface Endpoint {
  call: (auth: WidgetSessionHistoryAuth, signal?: AbortSignal) => Promise<unknown>
  path: string
  okBody: unknown
  badBody: unknown
  result: unknown
}

const ENDPOINTS: Record<"list" | "messages", Endpoint> = {
  list: {
    call: (auth, signal) => listWidgetConversations(ORIGIN, auth, { limit: 20 }, signal),
    path: WIDGET_SESSION_CONVERSATIONS_PATH,
    okBody: LIST_BODY,
    badBody: { conversations: [{ task_id: "101", title: "x", created_at: "y" }], next_cursor: null },
    result: LIST_RESULT,
  },
  messages: {
    call: (auth, signal) => readWidgetConversation(ORIGIN, auth, 101, { limit: 20 }, signal),
    path: `${WIDGET_SESSION_CONVERSATIONS_PATH}/101/messages`,
    okBody: MESSAGES_BODY,
    badBody: { messages: [{ id: 1, role: "system", content: "x", created_at: "y", attachments: [] }], next_cursor: null },
    result: MESSAGES_RESULT,
  },
}

const makeAuth = (overrides: Partial<WidgetSessionHistoryAuth> = {}, withRefresh = true) => {
  const auth: WidgetSessionHistoryAuth = {
    getToken: vi.fn(async () => CURRENT_TOKEN),
    ...(withRefresh ? { refreshToken: vi.fn(async () => REFRESHED_TOKEN) } : {}),
    ...overrides,
  }
  return auth
}

const outcomeOf = (promise: Promise<unknown>): Promise<unknown> =>
  promise.then((value) => ({ value }), (error: unknown) => ({ error }))

const expectKind = (outcome: unknown, kind: WidgetSessionHistoryErrorKind) => {
  const error = (outcome as { error?: unknown }).error
  expect(error).toBeInstanceOf(WidgetSessionHistoryError)
  expect((error as WidgetSessionHistoryError).kind).toBe(kind)
}

const expectEveryRequestIsTokenSafe = (path: string) => {
  expect(fetchMock).toHaveBeenCalled()
  for (const [input, init] of fetchMock.mock.calls) {
    const url = new URL(input)
    expect(url.origin).toBe(ORIGIN)
    expect(url.pathname).toBe(path)
    expect(input).not.toContain(CURRENT_TOKEN)
    expect(input).not.toContain(REFRESHED_TOKEN)
    expect(init?.method).toBe("GET")
    expect(init?.credentials).toBe("omit")
    expect(new Headers(init?.headers).get("Authorization")).toMatch(/^Bearer st_/)
  }
}

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal("fetch", fetchMock)
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe.each(Object.keys(ENDPOINTS) as Array<keyof typeof ENDPOINTS>)("the %s request", (name) => {
  const endpoint = ENDPOINTS[name]

  interface StatusCase {
    respond: () => void
    auth?: () => WidgetSessionHistoryAuth
    expected: WidgetSessionHistoryErrorKind | "result"
    requests: number
    timesOut?: boolean
  }
  const STATUS_CASES: Record<string, StatusCase> = {
    "a 200 with a valid body": {
      respond: () => { fetchMock.mockResolvedValueOnce(jsonResponse(200, endpoint.okBody)) },
      expected: "result",
      requests: 1,
    },
    "a 200 with a malformed body": {
      respond: () => { fetchMock.mockResolvedValueOnce(jsonResponse(200, endpoint.badBody)) },
      expected: "load_failed",
      requests: 1,
    },
    "a 401, a refresh, then a 200": {
      respond: () => {
        fetchMock.mockResolvedValueOnce(jsonResponse(401, {}))
        fetchMock.mockResolvedValueOnce(jsonResponse(200, endpoint.okBody))
      },
      expected: "result",
      requests: 2,
    },
    "a 401, a refresh, then another 401": {
      respond: () => {
        fetchMock.mockResolvedValueOnce(jsonResponse(401, {}))
        fetchMock.mockResolvedValueOnce(jsonResponse(401, {}))
      },
      expected: "load_failed",
      requests: 2,
    },
    "a 401 with no refresh": {
      respond: () => { fetchMock.mockResolvedValueOnce(jsonResponse(401, {})) },
      auth: () => makeAuth({}, false),
      expected: "unauthorized",
      requests: 1,
    },
    "a 403": {
      respond: () => { fetchMock.mockResolvedValueOnce(jsonResponse(403, { error: { code: "any_code" } })) },
      expected: "disabled",
      requests: 1,
    },
    "a 404": {
      respond: () => { fetchMock.mockResolvedValueOnce(jsonResponse(404, {})) },
      expected: "not_found",
      requests: 1,
    },
    "a 422": {
      respond: () => { fetchMock.mockResolvedValueOnce(jsonResponse(422, { detail: [] })) },
      expected: "load_failed",
      requests: 1,
    },
    "a 429": {
      respond: () => { fetchMock.mockResolvedValueOnce(jsonResponse(429, {})) },
      expected: "load_failed",
      requests: 1,
    },
    "a 500": {
      respond: () => { fetchMock.mockResolvedValueOnce(jsonResponse(500, {})) },
      expected: "load_failed",
      requests: 1,
    },
    "a network error": {
      respond: () => { fetchMock.mockRejectedValueOnce(new TypeError("Failed to fetch")) },
      expected: "load_failed",
      requests: 1,
    },
    "no response before the deadline": {
      respond: () => { fetchMock.mockImplementationOnce(() => new Promise<Response>(() => {})) },
      expected: "load_failed",
      requests: 1,
      timesOut: true,
    },
    "a token that cannot be obtained": {
      respond: () => {},
      auth: () => makeAuth({ getToken: vi.fn(async () => { throw new Error("refresh failed") }) }),
      expected: "load_failed",
      requests: 0,
    },
    "a refresh that fails": {
      respond: () => { fetchMock.mockResolvedValueOnce(jsonResponse(401, {})) },
      auth: () => makeAuth({ refreshToken: vi.fn(async () => { throw new Error("refresh failed") }) }),
      expected: "load_failed",
      requests: 1,
    },
  }

  it.each(Object.keys(STATUS_CASES))("maps %s", async (label) => {
    const scenario = STATUS_CASES[label]
    if (scenario.timesOut) vi.useFakeTimers()
    scenario.respond()
    const auth = scenario.auth?.() ?? makeAuth()

    const pending = outcomeOf(endpoint.call(auth))
    if (scenario.timesOut) await vi.advanceTimersByTimeAsync(HISTORY_REQUEST_TIMEOUT_MS)
    const outcome = await pending

    if (scenario.expected === "result") {
      expect(outcome).toEqual({ value: endpoint.result })
    } else {
      expectKind(outcome, scenario.expected)
    }
    expect(fetchMock).toHaveBeenCalledTimes(scenario.requests)
    if (scenario.requests > 0) expectEveryRequestIsTokenSafe(endpoint.path)
  })

  it("retries a 401 once with the refreshed token, passing the rejected one to the refresh", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(401, {}))
    fetchMock.mockResolvedValueOnce(jsonResponse(200, endpoint.okBody))
    const auth = makeAuth()

    await endpoint.call(auth)

    expect(auth.refreshToken).toHaveBeenCalledTimes(1)
    expect(auth.refreshToken).toHaveBeenCalledWith(CURRENT_TOKEN)
    const tokens = fetchMock.mock.calls.map(([, init]) => new Headers(init?.headers).get("Authorization"))
    expect(tokens).toEqual([`Bearer ${CURRENT_TOKEN}`, `Bearer ${REFRESHED_TOKEN}`])
  })

  it("keeps the token byte for byte", async () => {
    const token = "  st_spaced.token+/=  "
    fetchMock.mockResolvedValueOnce(jsonResponse(200, endpoint.okBody))

    await endpoint.call(makeAuth({ getToken: vi.fn(async () => token) }))

    expect((fetchMock.mock.calls[0][1]?.headers as Record<string, string>).Authorization).toBe(`Bearer ${token}`)
  })

  it.each([
    ["an unreadable body", () => new Response("{not json", { status: 200 })],
    ["a body that is not an object", () => jsonResponse(200, [endpoint.okBody])],
    ["a null body", () => jsonResponse(200, null)],
  ])("fails a 200 with %s as load_failed", async (_label, response) => {
    fetchMock.mockResolvedValueOnce(response())
    expectKind(await outcomeOf(endpoint.call(makeAuth())), "load_failed")
  })

  it.each([401, 403, 404, 500])("does not depend on the body of a %s", async (status) => {
    fetchMock.mockResolvedValueOnce(new Response("{not json", { status }))
    const expected: Record<number, WidgetSessionHistoryErrorKind> = {
      401: "unauthorized", 403: "disabled", 404: "not_found", 500: "load_failed",
    }
    expectKind(await outcomeOf(endpoint.call(makeAuth({}, false))), expected[status])
  })

  describe("the deadline", () => {
    beforeEach(() => { vi.useFakeTimers() })

    it("covers reading the body", async () => {
      const response = jsonResponse(200, endpoint.okBody)
      vi.spyOn(response, "json").mockImplementation(() => new Promise(() => {}))
      fetchMock.mockResolvedValueOnce(response)
      let outcome: unknown = "pending"
      void outcomeOf(endpoint.call(makeAuth())).then((value) => { outcome = value })

      await vi.advanceTimersByTimeAsync(HISTORY_REQUEST_TIMEOUT_MS - 1)
      expect(outcome).toBe("pending")
      await vi.advanceTimersByTimeAsync(1)
      expectKind(outcome, "load_failed")
    })

    it("aborts the request it gave up on", async () => {
      fetchMock.mockImplementationOnce(() => new Promise<Response>(() => {}))
      const pending = outcomeOf(endpoint.call(makeAuth()))
      await vi.advanceTimersByTimeAsync(HISTORY_REQUEST_TIMEOUT_MS)
      await pending

      expect(fetchMock.mock.calls[0][1]?.signal?.aborted).toBe(true)
    })

    it("starts again for the retry after a refresh", async () => {
      const late = (response: Response) => () => new Promise<Response>((resolve) => {
        setTimeout(() => resolve(response), HISTORY_REQUEST_TIMEOUT_MS - 5_000)
      })
      fetchMock.mockImplementationOnce(late(jsonResponse(401, {})))
      fetchMock.mockImplementationOnce(late(jsonResponse(200, endpoint.okBody)))
      const pending = outcomeOf(endpoint.call(makeAuth()))

      await vi.advanceTimersByTimeAsync(2 * (HISTORY_REQUEST_TIMEOUT_MS - 5_000))
      expect(await pending).toEqual({ value: endpoint.result })
    })
  })

  describe("a caller that aborts", () => {
    it("fails as load_failed without a request when aborted before it starts", async () => {
      const controller = new AbortController()
      controller.abort()

      expectKind(await outcomeOf(endpoint.call(makeAuth(), controller.signal)), "load_failed")
      expect(fetchMock).not.toHaveBeenCalled()
    })

    it("fails as load_failed when aborted while the request is in flight", async () => {
      const controller = new AbortController()
      fetchMock.mockImplementationOnce((_input, init) => new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))
      }))
      const pending = outcomeOf(endpoint.call(makeAuth(), controller.signal))
      await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

      controller.abort()
      expectKind(await pending, "load_failed")
    })

    it("does not refresh the token after an abort", async () => {
      const controller = new AbortController()
      const auth = makeAuth()
      fetchMock.mockImplementationOnce(async () => {
        controller.abort()
        return jsonResponse(401, {})
      })

      expectKind(await outcomeOf(endpoint.call(auth, controller.signal)), "load_failed")
      expect(auth.refreshToken).not.toHaveBeenCalled()
      expect(fetchMock).toHaveBeenCalledTimes(1)
    })

    it("fails as load_failed when aborted before the retry after a refresh answers", async () => {
      const controller = new AbortController()
      const auth = makeAuth()
      fetchMock.mockResolvedValueOnce(jsonResponse(401, {}))
      fetchMock.mockImplementationOnce(async () => {
        controller.abort()
        return jsonResponse(200, endpoint.okBody)
      })

      expectKind(await outcomeOf(endpoint.call(auth, controller.signal)), "load_failed")
      expect(auth.refreshToken).toHaveBeenCalledTimes(1)
      expect(fetchMock).toHaveBeenCalledTimes(2)
    })

    it("does not request with a token obtained after an abort", async () => {
      const controller = new AbortController()
      const auth = makeAuth({
        getToken: vi.fn(async () => {
          controller.abort()
          return CURRENT_TOKEN
        }),
      })

      expectKind(await outcomeOf(endpoint.call(auth, controller.signal)), "load_failed")
      expect(fetchMock).not.toHaveBeenCalled()
    })
  })
})

describe("the query string", () => {
  const queryOf = (index = 0) => new URL(fetchMock.mock.calls[index][0]).searchParams

  it.each([
    ["no page options", {}, {}],
    ["a limit", { limit: 20 }, { limit: "20" }],
    ["a limit and a cursor", { limit: 5, cursor: "abc+/=" }, { limit: "5", cursor: "abc+/=" }],
    ["a null cursor", { cursor: null }, {}],
  ] as const)("carries only the page options for %s", async (_label, page, expected) => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, LIST_BODY))
    fetchMock.mockResolvedValueOnce(jsonResponse(200, MESSAGES_BODY))

    await listWidgetConversations(ORIGIN, makeAuth(), page)
    await readWidgetConversation(ORIGIN, makeAuth(), 7, page)

    expect(Object.fromEntries(queryOf(0))).toEqual(expected)
    expect(Object.fromEntries(queryOf(1))).toEqual(expected)
    expect(new URL(fetchMock.mock.calls[1][0]).pathname).toBe(`${WIDGET_SESSION_CONVERSATIONS_PATH}/7/messages`)
  })
})

describe("the response shape", () => {
  const conversation = { task_id: 101, title: "Support Agent chat", created_at: "2026-01-02T03:04:05Z" }
  const message = MESSAGES_BODY.messages[0]
  const attachment = message.attachments[0]

  it.each([
    ["conversations that are not a list", { conversations: {}, next_cursor: null }],
    ["a missing next cursor", { conversations: [] }],
    ["a numeric next cursor", { conversations: [], next_cursor: 2 }],
    ["an entry that is not an object", { conversations: [null], next_cursor: null }],
    ["a zero task id", { conversations: [{ ...conversation, task_id: 0 }], next_cursor: null }],
    ["a negative task id", { conversations: [{ ...conversation, task_id: -1 }], next_cursor: null }],
    ["a fractional task id", { conversations: [{ ...conversation, task_id: 1.5 }], next_cursor: null }],
    ["a string task id", { conversations: [{ ...conversation, task_id: "101" }], next_cursor: null }],
    ["a missing title", { conversations: [{ task_id: 101, created_at: "x" }], next_cursor: null }],
    ["a numeric created time", { conversations: [{ ...conversation, created_at: 1 }], next_cursor: null }],
  ])("refuses a conversation list with %s", async (_label, body) => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, body))
    expectKind(await outcomeOf(listWidgetConversations(ORIGIN, makeAuth(), {})), "load_failed")
  })

  it("accepts an empty conversation list", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, { conversations: [], next_cursor: null }))
    expect(await listWidgetConversations(ORIGIN, makeAuth(), {})).toEqual({ conversations: [], nextCursor: null })
  })

  it.each([
    ["messages that are not a list", { messages: "x", next_cursor: null }],
    ["a missing next cursor", { messages: [] }],
    ["a message that is not an object", { messages: [1], next_cursor: null }],
    ["a zero id", { messages: [{ ...message, id: 0 }], next_cursor: null }],
    ["a string id", { messages: [{ ...message, id: "1" }], next_cursor: null }],
    ["another role", { messages: [{ ...message, role: "system" }], next_cursor: null }],
    ["missing content", { messages: [{ ...message, content: null }], next_cursor: null }],
    ["a missing created time", { messages: [{ ...message, created_at: undefined }], next_cursor: null }],
    ["attachments that are not a list", { messages: [{ ...message, attachments: null }], next_cursor: null }],
    ["an attachment that is not an object", { messages: [{ ...message, attachments: ["a"] }], next_cursor: null }],
    ["an attachment without a file name", {
      messages: [{ ...message, attachments: [{ ...attachment, filename: undefined }] }], next_cursor: null,
    }],
    ["a string file size", {
      messages: [{ ...message, attachments: [{ ...attachment, file_size: "11" }] }], next_cursor: null,
    }],
    ["a missing file size", {
      messages: [{ ...message, attachments: [{ ...attachment, file_size: undefined }] }], next_cursor: null,
    }],
    ["a numeric media type", {
      messages: [{ ...message, attachments: [{ ...attachment, mime_type: 5 }] }], next_cursor: null,
    }],
  ])("refuses a message page with %s", async (_label, body) => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, body))
    expectKind(await outcomeOf(readWidgetConversation(ORIGIN, makeAuth(), 101, {})), "load_failed")
  })

  it("accepts an empty message page", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, { messages: [], next_cursor: "more" }))
    expect(await readWidgetConversation(ORIGIN, makeAuth(), 101, {})).toEqual({ messages: [], nextCursor: "more" })
  })
})

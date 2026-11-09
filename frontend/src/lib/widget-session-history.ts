// Reads the end user's earlier conversations from the session server with the
// session token. Like the session upload, a request never goes through the
// shared API wrapper and never touches the browser's stored login token: the
// only credential is the session token, sent in a header.

export const WIDGET_SESSION_CONVERSATIONS_PATH = "/v1/external/chat/sessions/conversations"

// The upper bound on one attempt, from sending the request until its body has
// been read and has passed the shape check. It keeps a visitor from waiting
// without end; it is not a performance target.
export const HISTORY_REQUEST_TIMEOUT_MS = 20_000

// "disabled" is the server's 403 and "not_found" its 404; the response body is
// never read to tell them apart. "unauthorized" is a 401 that was not
// refreshed. Everything else is "load_failed", except a caller abort, which
// fails as an AbortError like the session upload's.
export type WidgetSessionHistoryErrorKind = "not_found" | "disabled" | "unauthorized" | "load_failed"

export class WidgetSessionHistoryError extends Error {
  readonly kind: WidgetSessionHistoryErrorKind

  constructor(kind: WidgetSessionHistoryErrorKind) {
    super(kind)
    this.name = "WidgetSessionHistoryError"
    this.kind = kind
  }
}

export interface WidgetSessionHistoryAuth {
  getToken: () => Promise<string>
  // Absent: a 401 is not refreshed and fails as "unauthorized".
  refreshToken?: (rejectedToken: string) => Promise<string>
}

export interface WidgetConversationSummary {
  taskId: number
  title: string
  createdAt: string
}

export interface WidgetConversationAttachment {
  filename: string
  fileSize: number | null
  mimeType: string | null
}

export interface WidgetConversationMessage {
  id: number
  role: "user" | "assistant"
  content: string
  createdAt: string
  attachments: WidgetConversationAttachment[]
}

export interface WidgetSessionHistoryPage {
  limit?: number
  cursor?: string | null
}

const loadFailed = () => new WidgetSessionHistoryError("load_failed")

const abortError = (): DOMException => new DOMException("The read was aborted.", "AbortError")

// Waits for `promise` unless `signal` aborts first. The abort settles only
// this caller's wait; the promise itself, which may be shared, keeps running.
const raceAbort = <T,>(promise: Promise<T>, signal?: AbortSignal): Promise<T> => {
  if (!signal) return promise
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(abortError())
    signal.addEventListener("abort", onAbort, { once: true })
    // Attached before the check below, so a promise that rejects after this
    // wait was aborted is still handled.
    promise.then(
      (value) => { signal.removeEventListener("abort", onAbort); resolve(value) },
      (error) => { signal.removeEventListener("abort", onAbort); reject(error) },
    )
    // The source may have aborted the signal while it was being called, which
    // is before this listener existed.
    if (signal.aborted) onAbort()
  })
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const isPositiveInteger = (value: unknown): value is number =>
  typeof value === "number" && Number.isInteger(value) && value > 0

const isCursor = (value: unknown): value is string | null =>
  value === null || typeof value === "string"

// Each parser returns only what it has checked, or null for any other shape.
const parseConversationPage = (
  body: unknown,
): { conversations: WidgetConversationSummary[]; nextCursor: string | null } | null => {
  if (!isRecord(body) || !Array.isArray(body.conversations) || !isCursor(body.next_cursor)) return null
  const conversations: WidgetConversationSummary[] = []
  for (const entry of body.conversations) {
    if (
      !isRecord(entry)
      || !isPositiveInteger(entry.task_id)
      || typeof entry.title !== "string"
      || typeof entry.created_at !== "string"
    ) {
      return null
    }
    conversations.push({ taskId: entry.task_id, title: entry.title, createdAt: entry.created_at })
  }
  return { conversations, nextCursor: body.next_cursor }
}

const parseAttachment = (entry: unknown): WidgetConversationAttachment | null => {
  if (
    !isRecord(entry)
    || typeof entry.filename !== "string"
    || !(entry.file_size === null || typeof entry.file_size === "number")
    || !(entry.mime_type === null || typeof entry.mime_type === "string")
  ) {
    return null
  }
  return { filename: entry.filename, fileSize: entry.file_size, mimeType: entry.mime_type }
}

const parseMessagePage = (
  body: unknown,
): { messages: WidgetConversationMessage[]; nextCursor: string | null } | null => {
  if (!isRecord(body) || !Array.isArray(body.messages) || !isCursor(body.next_cursor)) return null
  const messages: WidgetConversationMessage[] = []
  for (const entry of body.messages) {
    if (
      !isRecord(entry)
      || !isPositiveInteger(entry.id)
      || (entry.role !== "user" && entry.role !== "assistant")
      || typeof entry.content !== "string"
      || typeof entry.created_at !== "string"
      || !Array.isArray(entry.attachments)
    ) {
      return null
    }
    const attachments: WidgetConversationAttachment[] = []
    for (const item of entry.attachments) {
      const attachment = parseAttachment(item)
      if (!attachment) return null
      attachments.push(attachment)
    }
    messages.push({
      id: entry.id,
      role: entry.role,
      content: entry.content,
      createdAt: entry.created_at,
      attachments,
    })
  }
  return { messages, nextCursor: body.next_cursor }
}

type AttemptResult<T> = { kind: "ok"; value: T } | { kind: "unauthorized" }

// One request under its own deadline, which covers the body read and the shape
// check. A caller abort or the deadline aborts the request.
const attempt = async <T,>(
  url: string,
  token: string,
  parse: (body: unknown) => T | null,
  signal: AbortSignal | undefined,
): Promise<AttemptResult<T>> => {
  const controller = new AbortController()
  const abort = () => controller.abort()
  signal?.addEventListener("abort", abort, { once: true })
  let timer: ReturnType<typeof setTimeout> | undefined
  const deadline = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => {
      controller.abort()
      reject(loadFailed())
    }, HISTORY_REQUEST_TIMEOUT_MS)
  })
  const run = async (): Promise<AttemptResult<T>> => {
    let response: Response
    try {
      response = await fetch(url, {
        method: "GET",
        headers: { Authorization: `Bearer ${token}` },
        credentials: "omit",
        signal: controller.signal,
      })
    } catch {
      throw loadFailed()
    }
    if (response.status === 401) return { kind: "unauthorized" }
    if (response.status === 403) throw new WidgetSessionHistoryError("disabled")
    if (response.status === 404) throw new WidgetSessionHistoryError("not_found")
    if (response.status !== 200) throw loadFailed()
    let body: unknown
    try {
      body = await response.json()
    } catch {
      throw loadFailed()
    }
    const value = parse(body)
    if (value === null) throw loadFailed()
    return { kind: "ok", value }
  }
  try {
    return await Promise.race([run(), deadline])
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener("abort", abort)
  }
}

const request = async <T,>(
  origin: string,
  path: string,
  auth: WidgetSessionHistoryAuth,
  page: WidgetSessionHistoryPage,
  parse: (body: unknown) => T | null,
  signal: AbortSignal | undefined,
): Promise<T> => {
  // Only the page options go into the query string; the token never does.
  const query = new URLSearchParams()
  if (page.limit !== undefined) query.set("limit", String(page.limit))
  if (page.cursor !== undefined && page.cursor !== null) query.set("cursor", page.cursor)
  const search = query.toString()
  const url = `${origin}${path}${search ? `?${search}` : ""}`

  // Checked after every wait, so an aborted read neither sends a request nor
  // asks for a refreshed token.
  const ensureActive = () => {
    if (signal?.aborted) throw abortError()
  }
  // A token wait is not covered by the request deadline, so it is raced
  // against the caller's signal: an abort ends the read even while a token is
  // still pending.
  const obtain = async (source: () => Promise<string>): Promise<string> => {
    let token: string
    try {
      token = await raceAbort(source(), signal)
    } catch {
      throw loadFailed()
    }
    ensureActive()
    return token
  }

  try {
    ensureActive()
    const token = await obtain(auth.getToken)
    const first = await attempt(url, token, parse, signal)
    ensureActive()
    if (first.kind === "ok") return first.value
    const refreshToken = auth.refreshToken
    if (!refreshToken) throw new WidgetSessionHistoryError("unauthorized")
    const refreshed = await obtain(() => refreshToken(token))
    const retry = await attempt(url, refreshed, parse, signal)
    ensureActive()
    if (retry.kind === "ok") return retry.value
    throw loadFailed()
  } catch (error) {
    // Whatever the request ran into, a read the caller aborted fails as
    // aborted.
    ensureActive()
    throw error
  }
}

export function listWidgetConversations(
  origin: string,
  auth: WidgetSessionHistoryAuth,
  page: WidgetSessionHistoryPage,
  signal?: AbortSignal,
): Promise<{ conversations: WidgetConversationSummary[]; nextCursor: string | null }> {
  return request(origin, WIDGET_SESSION_CONVERSATIONS_PATH, auth, page, parseConversationPage, signal)
}

export function readWidgetConversation(
  origin: string,
  auth: WidgetSessionHistoryAuth,
  taskId: number,
  page: WidgetSessionHistoryPage,
  signal?: AbortSignal,
): Promise<{ messages: WidgetConversationMessage[]; nextCursor: string | null }> {
  return request(
    origin,
    `${WIDGET_SESSION_CONVERSATIONS_PATH}/${taskId}/messages`,
    auth,
    page,
    parseMessagePage,
    signal,
  )
}

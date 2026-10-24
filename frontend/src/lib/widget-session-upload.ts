import type { ClientErrorCode } from "@/lib/client-errors"

export const WIDGET_SESSION_UPLOAD_PATH = "/v1/external/chat/sessions/files"

// A 429 with code rate_limited is retried this many times before giving up.
const MAX_RATE_LIMIT_RETRIES = 3
const MAX_BACKOFF_SECONDS = 60
const JITTER_MS = 500
const CANONICAL_UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/

export const isCanonicalUuid = (value: unknown): value is string =>
  typeof value === "string" && CANONICAL_UUID.test(value)

// Why an advertised capability refuses one file, or null when it accepts it.
// `accept` lists lowercase extensions with a leading dot, separated by commas;
// a file with no extension never matches. The per-message count is left to
// the caller, which knows how many files the message already holds.
export const capabilityRefusal = (
  file: File,
  limits: { accept: string; maxFileBytes: number },
): "type" | "size" | null => {
  const dot = file.name.lastIndexOf(".")
  const extension = dot >= 0 ? file.name.slice(dot).toLowerCase() : ""
  if (!limits.accept.split(",").includes(extension)) return "type"
  if (file.size > limits.maxFileBytes) return "size"
  return null
}

export class WidgetSessionUploadError extends Error {
  readonly errorCode: ClientErrorCode

  constructor(errorCode: ClientErrorCode, message?: string) {
    super(message ?? errorCode)
    this.name = "WidgetSessionUploadError"
    this.errorCode = errorCode
  }
}

// The result of one successful upload. The server-assigned id lives here and
// never on the File object, so a File can be reused as a map key without being
// mutated.
export interface SessionUploadedFile {
  kind: "uploaded"
  fileId: string
  name: string
  size: number
  type: string
}

export interface UploadWidgetSessionFileOptions {
  origin: string
  // Called before every request, so a long backoff never sends a token that
  // expired while waiting.
  getToken: () => Promise<string>
  // Called at most once per upload, after a 401, with the token that was
  // rejected; resolves with a different token. Each refresh closes the chat
  // socket, and the widget script ends the session on the fourth reconnect
  // before a connection has stayed open for 60 s, so a refresh is worth
  // asking for only while the rejected token is still the current one.
  refreshToken: (rejectedToken: string) => Promise<string>
  file: File
  signal?: AbortSignal
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>
  random?: () => number
}

const abortError = (): DOMException => new DOMException("The upload was aborted.", "AbortError")

const isAbortError = (error: unknown): boolean =>
  typeof error === "object" && error !== null && (error as { name?: unknown }).name === "AbortError"

export const abortableSleep = (ms: number, signal?: AbortSignal): Promise<void> =>
  new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError())
      return
    }
    const onAbort = () => {
      clearTimeout(timer)
      reject(abortError())
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort)
      resolve()
    }, ms)
    signal?.addEventListener("abort", onAbort, { once: true })
  })

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
    // The provider may have aborted the signal while it was being called, which
    // is before this listener existed.
    if (signal.aborted) onAbort()
  })
}

const readJsonBody = async (response: Response, signal?: AbortSignal): Promise<unknown> => {
  try {
    return await response.json()
  } catch (error) {
    if (isAbortError(error) || signal?.aborted) throw abortError()
    throw new WidgetSessionUploadError("upload_failed")
  }
}

const readErrorCode = async (response: Response, signal?: AbortSignal): Promise<string | null> => {
  let body: unknown
  try {
    body = await readJsonBody(response, signal)
  } catch (error) {
    if (isAbortError(error)) throw error
    return null
  }
  if (typeof body !== "object" || body === null) return null
  const error = (body as { error?: unknown }).error
  if (typeof error !== "object" || error === null) return null
  const code = (error as { code?: unknown }).code
  return typeof code === "string" ? code : null
}

// Retry-After is either a number of seconds or an HTTP date. A date becomes the
// seconds left until it, rounded up so the retry never comes before it. Zero, a
// date already past, and an unreadable value all count as no header.
const readRetryAfterSeconds = (response: Response): number | null => {
  const raw = response.headers.get("Retry-After")?.trim()
  if (!raw) return null
  const seconds = /^\d+$/.test(raw)
    ? Number.parseInt(raw, 10)
    : Math.ceil((Date.parse(raw) - Date.now()) / 1000)
  return seconds > 0 ? seconds : null
}

const backoffMs = (attempt: number, retryAfterSeconds: number | null, random: () => number): number => {
  const exponential = 2 ** attempt
  const seconds = Math.min(Math.max(retryAfterSeconds ?? exponential, exponential), MAX_BACKOFF_SECONDS)
  return seconds * 1000 + Math.floor(random() * JITTER_MS)
}

const parseUploaded = async (response: Response, file: File, signal?: AbortSignal): Promise<SessionUploadedFile> => {
  const body = await readJsonBody(response, signal)
  if (typeof body !== "object" || body === null) throw new WidgetSessionUploadError("upload_failed")
  const info = body as Record<string, unknown>
  if (!isCanonicalUuid(info.file_id)) throw new WidgetSessionUploadError("upload_failed")
  return {
    kind: "uploaded",
    fileId: info.file_id,
    name: typeof info.filename === "string" && info.filename ? info.filename : file.name,
    size: typeof info.file_size === "number" && Number.isFinite(info.file_size) ? info.file_size : file.size,
    type: typeof info.mime_type === "string" && info.mime_type ? info.mime_type : file.type,
  }
}

// Uploads one file to the session upload endpoint with the session token. The
// request never goes through the shared API wrapper and never touches the
// browser's stored login token: the only credential is the session token.
export async function uploadWidgetSessionFile(
  options: UploadWidgetSessionFileOptions,
): Promise<SessionUploadedFile> {
  const { origin, file, signal } = options
  const sleep = options.sleep ?? abortableSleep
  const random = options.random ?? Math.random

  const obtainToken = async (source: () => Promise<string>): Promise<string> => {
    if (signal?.aborted) throw abortError()
    try {
      return await raceAbort(source(), signal)
    } catch (error) {
      if (isAbortError(error) || error instanceof WidgetSessionUploadError) throw error
      throw new WidgetSessionUploadError("upload_failed")
    }
  }

  let rateLimitRetries = 0
  let refreshed = false
  let token = await obtainToken(options.getToken)

  for (;;) {
    if (signal?.aborted) throw abortError()

    const formData = new FormData()
    formData.append("file", file)
    let response: Response
    try {
      response = await fetch(`${origin}${WIDGET_SESSION_UPLOAD_PATH}`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
        credentials: "omit",
        signal,
      })
    } catch (error) {
      if (isAbortError(error) || signal?.aborted) throw abortError()
      throw new WidgetSessionUploadError("upload_failed")
    }

    if (response.status === 201) return parseUploaded(response, file, signal)

    if (response.status === 429) {
      const code = await readErrorCode(response, signal)
      if (code === "quota_exceeded") throw new WidgetSessionUploadError("upload_quota_exceeded")
      if (code !== "rate_limited") throw new WidgetSessionUploadError("upload_failed")
      if (rateLimitRetries >= MAX_RATE_LIMIT_RETRIES) {
        throw new WidgetSessionUploadError("upload_rate_limited")
      }
      rateLimitRetries += 1
      await sleep(backoffMs(rateLimitRetries, readRetryAfterSeconds(response), random), signal)
      token = await obtainToken(options.getToken)
      continue
    }

    if (response.status === 401) {
      if (refreshed) throw new WidgetSessionUploadError("upload_failed")
      refreshed = true
      const rejected = token
      token = await obtainToken(() => options.refreshToken(rejected))
      continue
    }

    if (response.status === 413) throw new WidgetSessionUploadError("upload_too_large")
    if (response.status === 422) throw new WidgetSessionUploadError("upload_rejected")
    throw new WidgetSessionUploadError("upload_failed")
  }
}

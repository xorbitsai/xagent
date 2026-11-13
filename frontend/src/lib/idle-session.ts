/**
 * Browser view of the server-enforced idle-session policy (xagent#3771).
 *
 * The server is the only authority on whether a login has expired. The
 * browser keeps the last `idle_session` object it received for the current
 * login so it can report activity, warn before the deadline and ask the
 * server again when the deadline seems to have passed. It never signs a
 * user out because of its own clock.
 */

export const IDLE_SESSION_EXPIRED_ERROR = "SessionIdleExpired"
export const IDLE_SESSION_STORAGE_KEY = "auth_idle_session"
export const SESSION_END_NOTICE_KEY = "auth_session_end_notice"
export const IDLE_SESSION_CLOSE_CODE = 4010
/** Same-tab signal that the server may have ended the session (for example a socket closed with 4010). */
export const AUTH_REVALIDATE_EVENT = "xagent:auth-revalidate"
/** How long a sign-in-again notice stays relevant for a tab that reaches the login page. */
export const SESSION_END_NOTICE_TTL_MS = 10 * 60 * 1000

const IDLE_SESSION_SCHEMA_VERSION = 1

export interface IdleSessionPolicy {
  enforced: boolean
  sid: string | null
  idleTimeoutSeconds: number
  /** Seconds left on the server clock when the response was produced. */
  idleExpiresIn: number
  absoluteExpiresIn: number | null
  activityMinIntervalSeconds: number
  /** Server clock when the response was produced (ms), used only to order responses. */
  serverTime: number | null
}

export interface IdleSessionRecord extends IdleSessionPolicy {
  schemaVersion: typeof IDLE_SESSION_SCHEMA_VERSION
  /** Browser login lineage (auth cache `sessionId`) the policy belongs to. */
  sessionId: string
  /** Wall-clock time the response was observed; only used to measure elapsed time. */
  observedAt: number
  /** Wall-clock time this login last sent an activity report from any tab. */
  lastReportAt: number | null
}

type Storage = Pick<globalThis.Storage, "getItem" | "setItem" | "removeItem">

function isPositiveInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0
}
function isNonNegativeNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
}
function isTimestamp(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0
}

function parseServerTime(value: unknown): number | null {
  if (typeof value === "number") return isTimestamp(value) ? value : null
  if (typeof value !== "string") return null
  const parsed = Date.parse(value)
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : null
}

function parsePolicy(value: unknown): IdleSessionPolicy | null {
  if (typeof value !== "object" || value === null) return null
  const data = value as Record<string, unknown>
  const sid = data.sid
  const absolute = data.absolute_expires_in ?? null
  if (typeof data.enforced !== "boolean") return null
  if (sid !== null && sid !== undefined && (typeof sid !== "string" || !sid)) return null
  if (!isPositiveInteger(data.idle_timeout_seconds) || !isNonNegativeNumber(data.idle_expires_in)) return null
  if (!isPositiveInteger(data.activity_min_interval_seconds)) return null
  if (absolute !== null && !isNonNegativeNumber(absolute)) return null
  return {
    enforced: data.enforced,
    sid: typeof sid === "string" ? sid : null,
    idleTimeoutSeconds: data.idle_timeout_seconds,
    idleExpiresIn: data.idle_expires_in,
    absoluteExpiresIn: absolute,
    activityMinIntervalSeconds: data.activity_min_interval_seconds,
    serverTime: parseServerTime(data.server_time),
  }
}

/**
 * Reads the top-level `idle_session` object of an auth response body.
 * Absent or malformed means "this server does not support idle expiry",
 * which keeps the existing browser behavior.
 */
export function parseIdleSession(body: unknown): IdleSessionPolicy | null {
  if (typeof body !== "object" || body === null) return null
  return parsePolicy((body as Record<string, unknown>).idle_session)
}

function parseRecord(raw: string | null): IdleSessionRecord | null {
  if (raw === null) return null
  try {
    const value: unknown = JSON.parse(raw)
    if (typeof value !== "object" || value === null) return null
    const data = value as Partial<IdleSessionRecord>
    if (data.schemaVersion !== IDLE_SESSION_SCHEMA_VERSION || typeof data.sessionId !== "string" || !data.sessionId) return null
    if (!isTimestamp(data.observedAt) || (data.lastReportAt !== null && !isTimestamp(data.lastReportAt))) return null
    const policy = parsePolicy({
      enforced: data.enforced, sid: data.sid, idle_timeout_seconds: data.idleTimeoutSeconds,
      idle_expires_in: data.idleExpiresIn, absolute_expires_in: data.absoluteExpiresIn,
      activity_min_interval_seconds: data.activityMinIntervalSeconds, server_time: data.serverTime,
    })
    if (!policy) return null
    return { ...policy, schemaVersion: IDLE_SESSION_SCHEMA_VERSION, sessionId: data.sessionId, observedAt: data.observedAt, lastReportAt: data.lastReportAt ?? null }
  } catch { return null }
}

/** Returns the stored policy only when it belongs to the given browser login. */
export function readIdleSessionRecord(storage: Storage, sessionId: string | null): IdleSessionRecord | null {
  if (!sessionId) return null
  const record = parseRecord(storage.getItem(IDLE_SESSION_STORAGE_KEY))
  return record?.sessionId === sessionId ? record : null
}

/**
 * Stores a freshly observed policy for a browser login. A report time recorded
 * by another tab survives as long as the server login (`sid`) is unchanged.
 */
export function writeIdleSessionRecord(storage: Storage, sessionId: string, policy: IdleSessionPolicy, now = Date.now()): IdleSessionRecord {
  const previous = readIdleSessionRecord(storage, sessionId)
  if (previous && isOutOfOrder(previous, policy)) return previous
  const record: IdleSessionRecord = {
    ...policy,
    schemaVersion: IDLE_SESSION_SCHEMA_VERSION,
    sessionId,
    observedAt: now,
    lastReportAt: previous && previous.sid === policy.sid ? previous.lastReportAt : null,
  }
  storage.setItem(IDLE_SESSION_STORAGE_KEY, JSON.stringify(record))
  return record
}

/**
 * Responses for one login can arrive out of order (a verify sent before an
 * activity report may land after it). One that the server produced earlier
 * than the stored one, and that implies an earlier deadline, is stale and is
 * ignored. Ordering uses the server's clock, never the browser's; without a
 * server time the newest arrival wins.
 */
function isOutOfOrder(previous: IdleSessionRecord, policy: IdleSessionPolicy): boolean {
  if (previous.sid !== policy.sid || previous.enforced !== policy.enforced) return false
  if (previous.serverTime === null || policy.serverTime === null || policy.serverTime >= previous.serverTime) return false
  return policy.serverTime + policy.idleExpiresIn * 1000 < previous.serverTime + previous.idleExpiresIn * 1000
}

export function markIdleActivityReported(storage: Storage, sessionId: string, now = Date.now()): void {
  const record = readIdleSessionRecord(storage, sessionId)
  if (!record) return
  storage.setItem(IDLE_SESSION_STORAGE_KEY, JSON.stringify({ ...record, lastReportAt: now }))
}

/** Milliseconds left before the server deadline, measured from when the policy was observed. */
export function idleRemainingMs(record: IdleSessionPolicy & { observedAt: number }, now = Date.now()): number {
  const elapsed = Math.max(0, now - record.observedAt)
  return record.idleExpiresIn * 1000 - elapsed
}

export function isIdleSessionExpiredResponse(response: Pick<Response, "status" | "headers">): boolean {
  return response.status === 401 && response.headers.get("Error-Type") === IDLE_SESSION_EXPIRED_ERROR
}

export type SessionEndReason = "session_expired"

export function writeSessionEndNotice(storage: Storage, reason: SessionEndReason, now = Date.now()): void {
  storage.setItem(SESSION_END_NOTICE_KEY, JSON.stringify({ reason, at: now }))
}

export function readSessionEndNotice(storage: Storage, now = Date.now()): SessionEndReason | null {
  try {
    const value: unknown = JSON.parse(storage.getItem(SESSION_END_NOTICE_KEY) ?? "null")
    if (typeof value !== "object" || value === null) return null
    const notice = value as { reason?: unknown; at?: unknown }
    if (notice.reason !== "session_expired" || !isTimestamp(notice.at)) return null
    return now - notice.at <= SESSION_END_NOTICE_TTL_MS && now >= notice.at ? notice.reason : null
  } catch { return null }
}

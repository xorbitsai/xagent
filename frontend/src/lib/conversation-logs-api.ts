import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"

export type ConversationLogSource =
  | "all"
  | "widget"
  | "rest_api"
  | "shared_link"
  | "webhook"

export interface ConversationLogSummary {
  task_id: number
  title?: string | null
  description?: string | null
  status: string
  source: Exclude<ConversationLogSource, "all">
  source_label: string
  stored_source?: string | null
  agent_id?: number | null
  agent_name?: string | null
  agent_logo_url?: string | null
  created_at?: string | null
  updated_at?: string | null
  last_activity_at?: string | null
  input_tokens?: number
  output_tokens?: number
  total_tokens?: number
  llm_calls?: number
  message_count?: number
}

export interface ConversationLogAgentOption {
  agent_id: number
  agent_name: string
  agent_logo_url?: string | null
}

export interface ConversationLogPagination {
  page: number
  per_page: number
  total: number
  total_pages: number
}

export interface ConversationLogListResponse {
  logs: ConversationLogSummary[]
  source_counts: Record<ConversationLogSource, number>
  agents: ConversationLogAgentOption[]
  pagination: ConversationLogPagination
}

export interface ConversationLogTranscriptMessage {
  id: number | string
  role: string
  content: string
  message_type?: string | null
  interactions?: unknown
  turn_id?: string | null
  attachments?: unknown[]
  created_at?: string | null
  compaction?: {
    original_tokens?: number | null
    compacted_tokens?: number | null
    compression_ratio?: string | null
  } | null
}

export interface ConversationLogTraceEvent {
  event_id?: string
  event_type?: string
  step_id?: string | null
  timestamp?: number | null
  data?: Record<string, unknown>
  parent_event_id?: string | null
}

export interface ConversationLogDetailResponse {
  log: ConversationLogSummary
  transcript: ConversationLogTranscriptMessage[]
  trace_events?: ConversationLogTraceEvent[]
  // When the retention policy last removed this task's execution trace. Later
  // turns add new events, so trace_events may be non-empty but incomplete.
  trace_events_expired_at?: string | null
  metadata: {
    task?: {
      task_id: number
      input?: string | null
      output?: string | null
      error_message?: string | null
      description?: string | null
      agent_config?: Record<string, unknown>
    }
    trigger?: Record<string, unknown> | null
    public_context?: Record<string, unknown> | null
  }
  read_only: boolean
}

export interface ConversationLogFilters {
  source: ConversationLogSource
  agentId?: number | null
  search?: string
  // UTC ISO timestamps; the window is [activeFrom, activeTo).
  activeFrom?: string | null
  activeTo?: string | null
}

/**
 * The filter part of the list and export requests. Both endpoints take the
 * same filters, so they are built in one place and cannot drift apart.
 */
export function buildConversationLogFilterQuery(
  filters: ConversationLogFilters
): URLSearchParams {
  const query = new URLSearchParams({ source: filters.source })
  if (filters.agentId) query.set("agent_id", String(filters.agentId))
  if (filters.search?.trim()) query.set("search", filters.search.trim())
  if (filters.activeFrom) query.set("active_from", filters.activeFrom)
  if (filters.activeTo) query.set("active_to", filters.activeTo)
  return query
}

export async function fetchConversationLogs(
  params: ConversationLogFilters & {
    page: number
    perPage: number
  }
): Promise<ConversationLogListResponse> {
  const query = buildConversationLogFilterQuery(params)
  query.set("page", String(params.page))
  query.set("per_page", String(params.perPage))

  const response = await apiRequest(`${getApiUrl()}/api/conversation-logs?${query}`)
  if (!response.ok) {
    throw new Error("Failed to load conversation logs")
  }
  return response.json()
}

export type ConversationLogExportLimitReason = "rows" | "bytes"

/**
 * The export endpoint refuses, with no file, a result that has too many rows
 * or is too large; `limit` is the row count or the byte size, per `reason`.
 */
export class ConversationLogExportTooLargeError extends Error {
  readonly reason: ConversationLogExportLimitReason
  readonly limit: number

  constructor(reason: ConversationLogExportLimitReason, limit: number) {
    super("Conversation log export is too large")
    this.name = "ConversationLogExportTooLargeError"
    this.reason = reason
    this.limit = limit
  }
}

function exportTooLargeFromBody(body: unknown): ConversationLogExportTooLargeError | null {
  const detail =
    body && typeof body === "object" ? (body as { detail?: unknown }).detail : undefined
  if (!detail || typeof detail !== "object") return null
  const { code, reason, limit } = detail as {
    code?: unknown
    reason?: unknown
    limit?: unknown
  }
  if (code !== "conversation_log_export_too_large") return null
  if (reason !== "rows" && reason !== "bytes") return null
  if (typeof limit !== "number" || !Number.isFinite(limit)) return null
  return new ConversationLogExportTooLargeError(reason, limit)
}

function pad2(value: number): string {
  return String(value).padStart(2, "0")
}

function exportFilename(now: Date): string {
  const date = `${now.getFullYear()}${pad2(now.getMonth() + 1)}${pad2(now.getDate())}`
  const time = `${pad2(now.getHours())}${pad2(now.getMinutes())}${pad2(now.getSeconds())}`
  return `conversation-logs-${date}-${time}.csv`
}

/**
 * Downloads the CSV for the given filters. The file name is generated here
 * from the local time rather than read from the response headers.
 */
export async function downloadConversationLogsCsv(
  filters: ConversationLogFilters,
  now: Date = new Date()
): Promise<void> {
  const query = buildConversationLogFilterQuery(filters)
  const response = await apiRequest(`${getApiUrl()}/api/conversation-logs/export?${query}`)
  if (!response.ok) {
    if (response.status === 422) {
      const tooLarge = exportTooLargeFromBody(await response.json().catch(() => null))
      if (tooLarge) throw tooLarge
    }
    throw new Error("Failed to export conversation logs")
  }

  const blob = await response.blob()
  const url = window.URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = exportFilename(now)
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  window.URL.revokeObjectURL(url)
}

/**
 * The detail endpoint answers 410 `task_expired` for a conversation the
 * retention policy expired, and only to callers who could have seen it live
 * (everyone else gets the usual 404). A log can be listed and then expire
 * before its detail is fetched, so the page tells that apart from a failure.
 */
export class ConversationLogExpiredError extends Error {
  readonly expiredAt: string | null

  constructor(expiredAt: string | null) {
    super("Conversation log expired")
    this.name = "ConversationLogExpiredError"
    this.expiredAt = expiredAt
  }
}

async function expiredAtFromGoneResponse(response: Response): Promise<string | null | undefined> {
  const body: unknown = await response.json().catch(() => null)
  const detail =
    body && typeof body === "object" ? (body as { detail?: unknown }).detail : undefined
  if (!detail || typeof detail !== "object") return undefined
  const { code, expired_at: expiredAt } = detail as { code?: unknown; expired_at?: unknown }
  if (code !== "task_expired") return undefined
  return typeof expiredAt === "string" ? expiredAt : null
}

export async function fetchConversationLogDetail(
  taskId: number
): Promise<ConversationLogDetailResponse> {
  const response = await apiRequest(`${getApiUrl()}/api/conversation-logs/${taskId}`)
  if (!response.ok) {
    if (response.status === 410) {
      const expiredAt = await expiredAtFromGoneResponse(response)
      if (expiredAt !== undefined) throw new ConversationLogExpiredError(expiredAt)
    }
    throw new Error("Failed to load conversation detail")
  }
  return response.json()
}

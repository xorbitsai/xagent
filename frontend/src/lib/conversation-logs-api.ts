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
  id: number
  role: string
  content: string
  message_type?: string | null
  interactions?: unknown
  turn_id?: string | null
  attachments?: unknown[]
  created_at?: string | null
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

export async function fetchConversationLogs(params: {
  source: ConversationLogSource
  agentId?: number | null
  search?: string
  page: number
  perPage: number
}): Promise<ConversationLogListResponse> {
  const query = new URLSearchParams({
    page: String(params.page),
    per_page: String(params.perPage),
    source: params.source,
  })
  if (params.agentId) query.set("agent_id", String(params.agentId))
  if (params.search?.trim()) query.set("search", params.search.trim())

  const response = await apiRequest(`${getApiUrl()}/api/conversation-logs?${query}`)
  if (!response.ok) {
    throw new Error("Failed to load conversation logs")
  }
  return response.json()
}

export async function fetchConversationLogDetail(
  taskId: number
): Promise<ConversationLogDetailResponse> {
  const response = await apiRequest(`${getApiUrl()}/api/conversation-logs/${taskId}`)
  if (!response.ok) {
    throw new Error("Failed to load conversation detail")
  }
  return response.json()
}

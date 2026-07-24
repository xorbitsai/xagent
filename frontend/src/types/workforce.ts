export type WorkforceStatus = "draft" | "active" | "archived"
export type WorkforceWorkerSourceType = "existing"

export interface WorkforceAgentSummary {
  id: number
  name: string
  description: string | null
  logo_url: string | null
  status: string
  access?: string
  readonly?: boolean
  can_edit?: boolean
  can_publish?: boolean
  can_delete?: boolean
}

export interface WorkforceWorker {
  id: number
  agent: WorkforceAgentSummary
  alias: string | null
  assignment_instructions: string
  source_type: WorkforceWorkerSourceType
  template_id: string | null
  enabled: boolean
  sort_order: number | null
  canvas_position: Record<string, unknown> | null
  created_at: string | null
  updated_at: string | null
}

export interface WorkforceManagerListItem {
  id: number
  name: string
  logo_url: string | null
}

export interface WorkforceRunListItem {
  id: number
  task_id: number | null
  status: string
  created_at: string | null
  completed_at?: string | null
  task?: {
    title: string
    description: string | null
    status: string
  } | null
}

export interface WorkforceRunHistoryItem {
  id: number
  task_id: number | null
  status: string
  is_preview: boolean
  task_title: string | null
  message: string | null
  created_at: string | null
  completed_at: string | null
}

export interface WorkforceRunHistoryResponse {
  items: WorkforceRunHistoryItem[]
  total: number
  page: number
  size: number
  pages: number
}

export interface WorkforceListItem {
  id: number
  name: string
  description: string | null
  status: WorkforceStatus
  manager: WorkforceManagerListItem
  worker_count: number
  last_run: WorkforceRunListItem | null
  created_at: string | null
  updated_at: string | null
}

export interface WorkforceDetail {
  id: number
  name: string
  description: string | null
  status: WorkforceStatus
  manager: WorkforceAgentSummary
  workers: WorkforceWorker[]
  canvas_layout: Record<string, unknown> | null
  scope_type: string
  scope_id: string
  owner_user_id: number
  created_at: string | null
  updated_at: string | null
}

export interface WorkforceListResponse {
  items: WorkforceListItem[]
  total: number
  page: number
  size: number
  pages: number
}

export interface WorkforceAgentExecutionTraceEvent {
  event_id?: string
  event_type?: string
  step_id?: string | null
  timestamp?: number | string | null
  data?: Record<string, unknown>
  parent_event_id?: string | null
}

export interface WorkforceAgentExecution {
  task_id: number
  worker_task_id: string
  agent_id?: number
  agent_name?: string
  worker_member_id?: number
  worker_alias?: string
  status: string
  trace_events: WorkforceAgentExecutionTraceEvent[]
}

export interface WorkforceAgentOption {
  id: number
  name: string
  description: string | null
  logo_url: string | null
  status: string
  access?: string
  readonly?: boolean
  can_edit?: boolean
  can_publish?: boolean
  can_delete?: boolean
}

export interface WorkforceWorkerDraft {
  source_type: WorkforceWorkerSourceType
  agent_id: number
  alias: string
  assignment_instructions: string
  enabled: boolean
  sort_order: number
  canvas_position?: Record<string, unknown> | null
}

export interface WorkforceWorkerPayload {
  source_type: WorkforceWorkerSourceType
  agent_id: number
  alias?: string
  assignment_instructions: string
  enabled?: boolean
  sort_order?: number
  canvas_position?: Record<string, unknown> | null
}

export interface WorkforceCreatePayload {
  name: string
  description?: string
  manager_agent_id: number
  canvas_layout?: Record<string, unknown> | null
  workers?: WorkforceWorkerPayload[]
}

export interface WorkforcePromptCreatePayload {
  prompt: string
}

export interface WorkforceUpdatePayload {
  name?: string
  description?: string | null
  manager_agent_id?: number
  canvas_layout?: Record<string, unknown> | null
}

export interface WorkforceWorkerUpdatePayload {
  alias?: string | null
  assignment_instructions?: string
  enabled?: boolean
  sort_order?: number
  canvas_position?: Record<string, unknown> | null
}

export interface WorkforceArchiveResponse {
  id: number
  status: "archived"
}

export interface WorkforceRunPayload {
  message: string
  files?: string[]
  execution_mode?: string | null
  is_preview?: boolean
  is_visible?: boolean
}

export interface WorkforceRunResponse {
  workforce_run_id: number
  task_id: number
  status: string
  redirect_url: string
}

export interface WorkforceShareLink {
  workforce_id: number
  share_enabled: boolean
  share_token: string | null
  share_updated_at: string | null
}

export interface WorkforceWidgetConfig {
  workforce_id: number
  widget_enabled: boolean
  widget_key: string | null
  allowed_domains: string[]
}

export interface WorkforceBuilderOperation {
  op: string
  [key: string]: unknown
}

export interface WorkforceBuilderPatch {
  summary: string
  operations: WorkforceBuilderOperation[]
  warnings: string[]
  clarification?: string | null
}

export interface WorkforceBuilderMessage {
  id: number
  role: string
  content: string
  status: string
  proposed_patch: WorkforceBuilderPatch | null
  created_at: string | null
}

export interface WorkforceCanvasNode {
  id: string
  type: "human" | "manager" | "worker" | string
  agent_id?: number
  label: string
  position?: Record<string, unknown> | null
  enabled?: boolean
}

export interface WorkforceCanvasEdge {
  id: string
  source: string
  target: string
}

export interface WorkforceCanvasResponse {
  nodes: WorkforceCanvasNode[]
  edges: WorkforceCanvasEdge[]
  layout: Record<string, unknown>
}

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
  manager_instructions: string | null
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
  manager_instructions?: string
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
  manager_instructions?: string | null
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
  is_visible?: boolean
}

export interface WorkforceRunResponse {
  workforce_run_id: number
  task_id: number
  status: string
  redirect_url: string
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

export interface WorkforceBuilderMessagesResponse {
  items: WorkforceBuilderMessage[]
}

export interface WorkforceBuilderProposePayload {
  message: string
}

export interface WorkforceBuilderProposeResponse {
  message_id: number
  user_message: WorkforceBuilderMessage
  assistant_message: string
  message: WorkforceBuilderMessage
  proposed_patch: WorkforceBuilderPatch
  requires_confirmation: boolean
}

export interface WorkforceBuilderApplyPayload {
  message_id: number
  proposed_patch: WorkforceBuilderPatch
}

export interface WorkforceBuilderApplyResponse {
  status: "applied"
  message_id: number
  message: WorkforceBuilderMessage
  workforce: WorkforceDetail
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

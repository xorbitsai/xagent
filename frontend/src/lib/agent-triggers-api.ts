"use client"

import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"

export type AgentTriggerType = "webhook" | "scheduled" | "gmail"
export type AgentTriggerRunStatus = "pending" | "running" | "completed" | "failed"

// Triggers can be owned by an agent or a workforce (issue #950). Workforce
// triggers live under /api/workforces/{id}/triggers and have agent_id null.
export type TriggerOwnerKind = "agent" | "workforce"

export interface TriggerOwnerRef {
  kind: TriggerOwnerKind
  id: number | string
}

export interface AgentTrigger {
  id: number
  user_id: number
  agent_id: number | null
  workforce_id?: number | null
  type: AgentTriggerType
  name: string
  enabled: boolean
  config: Record<string, unknown>
  prompt_template: string | null
  webhook_token: string | null
  webhook_secret?: string | null
  callback_id?: string | null
  provisioning_status?: "pending" | "active" | "failed" | null
  provisioning_error?: string | null
  next_run_at: string | null
  last_run_at: string | null
  last_error: string | null
  created_at: string | null
  updated_at: string | null
}

export interface AgentTriggerRun {
  id: number
  trigger_id: number
  task_id: number | null
  background_job_id: string | null
  status: AgentTriggerRunStatus
  source_event_id: string | null
  payload_snapshot: Record<string, unknown> | null
  idempotency_key: string
  error_message: string | null
  started_at: string | null
  finished_at: string | null
  created_at: string | null
  updated_at: string | null
}

export interface AgentTriggerPayload {
  type?: AgentTriggerType
  name?: string | null
  enabled?: boolean
  config?: Record<string, unknown>
  prompt_template?: string | null
  secret?: string | null
  rotate_secret?: boolean
}

// A trigger configured while the agent itself does not exist yet (issue #928).
// Staged triggers live only in builder state; on agent creation each one is
// posted through createAgentTrigger. clientId is a stable NEGATIVE number so
// it can double as a pseudo AgentTrigger.id without ever colliding with (or
// being mistaken for) a real server id.
export interface StagedTrigger {
  clientId: number
  type: AgentTriggerType
  name: string
  enabled: boolean
  config: Record<string, unknown>
  prompt_template: string | null
  secret: string | null
}

export function stagedToPseudoTrigger(staged: StagedTrigger): AgentTrigger {
  return {
    id: staged.clientId,
    user_id: 0,
    agent_id: 0,
    type: staged.type,
    name: staged.name,
    enabled: staged.enabled,
    config: staged.config,
    prompt_template: staged.prompt_template,
    webhook_token: null,
    callback_id: null,
    next_run_at: null,
    last_run_at: null,
    last_error: null,
    created_at: null,
    updated_at: null,
  }
}

export function stagedToCreatePayload(
  staged: StagedTrigger,
): AgentTriggerPayload & { type: AgentTriggerType } {
  return {
    type: staged.type,
    name: staged.name,
    enabled: staged.enabled,
    config: staged.config,
    prompt_template: staged.prompt_template,
    secret: staged.secret,
  }
}

export interface FailedStagedTrigger {
  staged: StagedTrigger
  error: string
}

export interface StagedTriggerCreateOutcome {
  failed: FailedStagedTrigger[]
  // Auto-generated webhook secrets (returned only once by the create API);
  // omitted for triggers where the user supplied their own secret.
  generatedSecrets: { name: string; secret: string }[]
}

// Post staged triggers against a freshly created agent. Failures never throw:
// each failed trigger is returned with its config intact so the caller can
// offer retry instead of discarding the user's input.
export async function createStagedTriggers(
  agentId: number | string,
  staged: StagedTrigger[],
): Promise<StagedTriggerCreateOutcome> {
  const results = await Promise.allSettled(
    staged.map((item) => createAgentTrigger(agentId, stagedToCreatePayload(item))),
  )
  const outcome: StagedTriggerCreateOutcome = { failed: [], generatedSecrets: [] }
  results.forEach((result, index) => {
    const item = staged[index]
    if (result.status === "rejected") {
      const error =
        result.reason instanceof Error ? result.reason.message : String(result.reason)
      outcome.failed.push({ staged: item, error })
      return
    }
    const secret = result.value.webhook_secret
    if (secret && !item.secret) {
      outcome.generatedSecrets.push({ name: item.name, secret })
    }
  })
  return outcome
}

export interface AgentTriggerTestPayload {
  payload: Record<string, unknown>
  source_event_id?: string | null
}

export interface AgentTriggerTestResponse {
  trigger_run: AgentTriggerRun
  duplicate: boolean
}

export interface GmailAccount {
  id: number
  provider: string
  email: string | null
}

function jsonHeaders(): HeadersInit {
  return {
    "Content-Type": "application/json",
  }
}

function ownerTriggerUrl(
  owner: TriggerOwnerRef,
  triggerId?: number | string,
): string {
  const base =
    owner.kind === "workforce"
      ? `${getApiUrl()}/api/workforces/${owner.id}/triggers`
      : `${getApiUrl()}/api/agents/${owner.id}/triggers`
  return triggerId === undefined ? base : `${base}/${triggerId}`
}

function formatApiDetail(detail: unknown, fallback: string): string {
  if (typeof detail === "string" && detail.trim()) {
    return detail
  }
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) {
          return String(item.msg)
        }
        return null
      })
      .filter(Boolean)
    if (messages.length > 0) {
      return messages.join("; ")
    }
  }
  return fallback
}

async function parseApiError(response: Response, fallback: string): Promise<Error> {
  try {
    const data = await response.json()
    return new Error(formatApiDetail(data?.detail, fallback))
  } catch {
    return new Error(fallback)
  }
}

export async function listGmailAccounts(): Promise<GmailAccount[]> {
  const response = await apiRequest(
    `${getApiUrl()}/api/cloud/accounts?provider=gmail`,
  )
  if (!response.ok) {
    throw await parseApiError(response, "Failed to load Gmail accounts")
  }
  return response.json()
}

export async function listOwnerTriggers(
  owner: TriggerOwnerRef,
): Promise<AgentTrigger[]> {
  const response = await apiRequest(ownerTriggerUrl(owner))
  if (!response.ok) {
    throw await parseApiError(response, "Failed to load triggers")
  }
  return response.json()
}

export async function createOwnerTrigger(
  owner: TriggerOwnerRef,
  payload: AgentTriggerPayload & { type: AgentTriggerType },
): Promise<AgentTrigger> {
  const response = await apiRequest(ownerTriggerUrl(owner), {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    throw await parseApiError(response, "Failed to create trigger")
  }
  return response.json()
}

export async function updateOwnerTrigger(
  owner: TriggerOwnerRef,
  triggerId: number | string,
  payload: AgentTriggerPayload,
): Promise<AgentTrigger> {
  const response = await apiRequest(ownerTriggerUrl(owner, triggerId), {
    method: "PATCH",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    throw await parseApiError(response, "Failed to update trigger")
  }
  return response.json()
}

export async function deleteOwnerTrigger(
  owner: TriggerOwnerRef,
  triggerId: number | string,
): Promise<void> {
  const response = await apiRequest(ownerTriggerUrl(owner, triggerId), {
    method: "DELETE",
  })
  if (!response.ok) {
    throw await parseApiError(response, "Failed to delete trigger")
  }
}

export async function listOwnerTriggerRuns(
  owner: TriggerOwnerRef,
  triggerId: number | string,
): Promise<AgentTriggerRun[]> {
  const response = await apiRequest(`${ownerTriggerUrl(owner, triggerId)}/runs`)
  if (!response.ok) {
    throw await parseApiError(response, "Failed to load trigger runs")
  }
  return response.json()
}

export async function testOwnerTrigger(
  owner: TriggerOwnerRef,
  triggerId: number | string,
  payload: AgentTriggerTestPayload,
): Promise<AgentTriggerTestResponse> {
  const response = await apiRequest(`${ownerTriggerUrl(owner, triggerId)}/test`, {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    throw await parseApiError(response, "Failed to test trigger")
  }
  return response.json()
}

export async function listAgentTriggers(
  agentId: number | string,
): Promise<AgentTrigger[]> {
  return listOwnerTriggers({ kind: "agent", id: agentId })
}

export async function createAgentTrigger(
  agentId: number | string,
  payload: AgentTriggerPayload & { type: AgentTriggerType },
): Promise<AgentTrigger> {
  return createOwnerTrigger({ kind: "agent", id: agentId }, payload)
}

export async function updateAgentTrigger(
  agentId: number | string,
  triggerId: number | string,
  payload: AgentTriggerPayload,
): Promise<AgentTrigger> {
  return updateOwnerTrigger({ kind: "agent", id: agentId }, triggerId, payload)
}

export async function deleteAgentTrigger(
  agentId: number | string,
  triggerId: number | string,
): Promise<void> {
  return deleteOwnerTrigger({ kind: "agent", id: agentId }, triggerId)
}

export async function listAgentTriggerRuns(
  agentId: number | string,
  triggerId: number | string,
): Promise<AgentTriggerRun[]> {
  return listOwnerTriggerRuns({ kind: "agent", id: agentId }, triggerId)
}

export async function testAgentTrigger(
  agentId: number | string,
  triggerId: number | string,
  payload: AgentTriggerTestPayload,
): Promise<AgentTriggerTestResponse> {
  return testOwnerTrigger({ kind: "agent", id: agentId }, triggerId, payload)
}

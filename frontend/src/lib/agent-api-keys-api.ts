import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"

// Mirrors the backend ``AgentApiKeyListItem``. A key is bound to exactly
// one owner: ``owner_type`` says which identity pair is populated
// (agent_id/agent_name for "agent", workforce_id/workforce_name for
// "workforce"); the unset pair is null.
export interface AgentApiKeyListItem {
  id: number
  owner_type: "agent" | "workforce"
  agent_id: number | null
  agent_name: string | null
  workforce_id: number | null
  workforce_name: string | null
  label: string | null
  key_prefix: string
  masked_key: string
  status: "active" | "paused" | "revoked"
  last_used_at: string | null
  created_at: string
}

// Mirrors the backend ``AgentApiKeyStats``.
export interface AgentApiKeyStats {
  total_keys: number
  active_keys: number
  calls_this_month: number
  last_api_call: string | null
}

// Mirrors the backend ``APIKeyGenerateResponse``. ``full_key`` is the
// plaintext secret returned exactly once -- the server only stores a hash.
export interface AgentApiKeyCreated {
  full_key: string
  key_prefix: string
  created_at: string
}

const BASE_URL = `${getApiUrl()}/api/agent-api-keys`

export async function listAgentApiKeys(
  filter?: { agentId?: number; workforceId?: number }
): Promise<AgentApiKeyListItem[]> {
  const params = new URLSearchParams()
  if (filter?.agentId != null) params.set("agent_id", String(filter.agentId))
  if (filter?.workforceId != null) {
    params.set("workforce_id", String(filter.workforceId))
  }
  const query = params.toString()
  const url = query ? `${BASE_URL}?${query}` : BASE_URL
  const res = await apiRequest(url, { method: "GET" })
  if (!res.ok) throw new Error(`Failed to load API keys (${res.status})`)
  return (await res.json()) as AgentApiKeyListItem[]
}

export async function getAgentApiKeyStats(): Promise<AgentApiKeyStats> {
  const res = await apiRequest(`${BASE_URL}/stats`, { method: "GET" })
  if (!res.ok) throw new Error(`Failed to load API key stats (${res.status})`)
  return (await res.json()) as AgentApiKeyStats
}

/** Add a new key for an agent. Does not touch that agent's other keys. */
export async function createAgentApiKey(
  agentId: number,
  label: string | null
): Promise<AgentApiKeyCreated> {
  const res = await apiRequest(BASE_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ agent_id: agentId, label }),
  })
  if (!res.ok) throw new Error(`Failed to create API key (${res.status})`)
  return (await res.json()) as AgentApiKeyCreated
}

/** Add a new key for a workforce. Does not touch its other keys. */
export async function createWorkforceApiKey(
  workforceId: number,
  label: string | null
): Promise<AgentApiKeyCreated> {
  const res = await apiRequest(BASE_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ workforce_id: workforceId, label }),
  })
  if (!res.ok) throw new Error(`Failed to create API key (${res.status})`)
  return (await res.json()) as AgentApiKeyCreated
}

export async function pauseAgentApiKey(
  keyId: number
): Promise<AgentApiKeyListItem> {
  const res = await apiRequest(`${BASE_URL}/${keyId}/pause`, { method: "POST" })
  if (!res.ok) throw new Error(`Failed to pause API key (${res.status})`)
  return (await res.json()) as AgentApiKeyListItem
}

export async function resumeAgentApiKey(
  keyId: number
): Promise<AgentApiKeyListItem> {
  const res = await apiRequest(`${BASE_URL}/${keyId}/resume`, { method: "POST" })
  if (!res.ok) throw new Error(`Failed to resume API key (${res.status})`)
  return (await res.json()) as AgentApiKeyListItem
}

/** Issue a new secret for an existing key, keeping its id/label/status. */
export async function regenerateAgentApiKey(
  keyId: number
): Promise<AgentApiKeyCreated> {
  const res = await apiRequest(`${BASE_URL}/${keyId}/regenerate`, {
    method: "POST",
  })
  if (!res.ok) throw new Error(`Failed to regenerate API key (${res.status})`)
  return (await res.json()) as AgentApiKeyCreated
}

export async function deleteAgentApiKey(keyId: number): Promise<void> {
  const res = await apiRequest(`${BASE_URL}/${keyId}`, { method: "DELETE" })
  if (!res.ok) throw new Error(`Failed to delete API key (${res.status})`)
}

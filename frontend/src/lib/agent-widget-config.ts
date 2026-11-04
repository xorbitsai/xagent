import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"

export interface AgentWidgetConfigUpdates {
  widget_enabled?: boolean
  allowed_domains?: string[]
}

export function buildWidgetSnippet(widgetKey: string, appOrigin: string): string {
  if (!widgetKey || !appOrigin) return ""
  return `<script
  src="${appOrigin}/widget.js"
  data-widget-key="${widgetKey}"
  data-button-size="60px"
  data-button-color="#000"
  data-icon-color="#fff"
  data-panel-bg-color="#fff">
</script>`
}

export function normalizeAllowedDomain(value: string): string {
  return value.trim().toLowerCase()
}

// The backend compares Origin/Referer netloc against stored entries, so only a
// bare host[:port] (or the "*" catch-all) can ever match. Entries with a
// scheme, path, or wildcard label would be stored but silently never allow
// anything, so reject them up front.
const HOST_PATTERN =
  /^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*(:\d{1,5})?$/

export function isValidAllowedDomain(domain: string): boolean {
  return domain === "*" || HOST_PATTERN.test(domain)
}

export async function parseAgentUpdateError(
  response: Response,
  fallbackMessage: string,
): Promise<Error> {
  try {
    const data = await response.json()
    if (typeof data?.detail === "string" && data.detail.trim()) {
      return new Error(data.detail)
    }
  } catch {
    // Use the fallback below.
  }
  return new Error(fallbackMessage)
}

export async function updateAgentWidgetConfig(
  agentId: number | string,
  updates: AgentWidgetConfigUpdates,
  fallbackErrorMessage: string,
): Promise<Record<string, unknown>> {
  const response = await apiRequest(`${getApiUrl()}/api/agents/${agentId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  })
  if (!response.ok) {
    throw await parseAgentUpdateError(response, fallbackErrorMessage)
  }
  return response.json()
}

export interface AgentWidgetKeyState {
  agent_id: number
  widget_enabled: boolean
  widget_key: string
}

export async function fetchAgentWidgetKey(
  agentId: number | string,
  fallbackErrorMessage: string,
): Promise<AgentWidgetKeyState> {
  const response = await apiRequest(`${getApiUrl()}/api/agents/${agentId}/widget-key`)
  if (!response.ok) {
    throw await parseAgentUpdateError(response, fallbackErrorMessage)
  }
  return response.json()
}

export async function rotateAgentWidgetKey(
  agentId: number | string,
  fallbackErrorMessage: string,
): Promise<AgentWidgetKeyState> {
  const response = await apiRequest(
    `${getApiUrl()}/api/agents/${agentId}/widget-key/rotate`,
    { method: "POST" },
  )
  if (!response.ok) {
    throw await parseAgentUpdateError(response, fallbackErrorMessage)
  }
  return response.json()
}

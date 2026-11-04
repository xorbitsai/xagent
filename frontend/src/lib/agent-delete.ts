"use client"

import {
  apiRequest,
  isJsonRecord,
  parseApiResponse,
} from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import type { WorkforceStatus } from "@/types/workforce"

export type AgentDeleteWorkforceRole = "manager" | "worker"

export interface AgentDeleteWorkforceReference {
  workforce_id: number
  name: string
  status: WorkforceStatus
  roles: AgentDeleteWorkforceRole[]
  can_edit: boolean
  can_discard: boolean
}

export interface AgentDeleteConflictDetail {
  code: "agent_in_use_by_workforce"
  message: string
  references: AgentDeleteWorkforceReference[]
  has_hidden_references: boolean
}

export type AgentDeleteResult =
  | { kind: "deleted" }
  | { kind: "blocked"; conflict: AgentDeleteConflictDetail }

const WORKFORCE_STATUSES: WorkforceStatus[] = ["draft", "active", "archived"]
const WORKFORCE_ROLES: AgentDeleteWorkforceRole[] = ["manager", "worker"]

function isWorkforceReference(
  value: unknown,
): value is AgentDeleteWorkforceReference {
  if (!isJsonRecord(value)) return false

  const roles = value.roles
  if (
    !Number.isInteger(value.workforce_id) ||
    Number(value.workforce_id) <= 0 ||
    typeof value.name !== "string" ||
    !value.name.trim() ||
    !WORKFORCE_STATUSES.includes(value.status as WorkforceStatus) ||
    !Array.isArray(roles) ||
    roles.length === 0 ||
    !roles.every((role) => WORKFORCE_ROLES.includes(role as AgentDeleteWorkforceRole)) ||
    typeof value.can_edit !== "boolean" ||
    typeof value.can_discard !== "boolean"
  ) {
    return false
  }

  return !value.can_discard || (value.status === "draft" && value.can_edit)
}

export function parseAgentDeleteConflict(
  value: unknown,
): AgentDeleteConflictDetail | null {
  if (!isJsonRecord(value) || !isJsonRecord(value.detail)) return null

  const detail = value.detail
  if (
    detail.code !== "agent_in_use_by_workforce" ||
    typeof detail.message !== "string" ||
    !detail.message.trim() ||
    !Array.isArray(detail.references) ||
    !detail.references.every(isWorkforceReference) ||
    typeof detail.has_hidden_references !== "boolean" ||
    (detail.references.length === 0 && !detail.has_hidden_references)
  ) {
    return null
  }

  return {
    code: detail.code,
    message: detail.message,
    references: detail.references,
    has_hidden_references: detail.has_hidden_references,
  }
}

export async function requestAgentDeletion(
  agentId: number | string,
  fallbackMessage: string,
): Promise<AgentDeleteResult> {
  let response: Response
  try {
    response = await apiRequest(`${getApiUrl()}/api/agents/${agentId}`, {
      method: "DELETE",
    })
  } catch {
    throw new Error(fallbackMessage)
  }

  if (response.ok) {
    return { kind: "deleted" }
  }

  let parsed: Awaited<ReturnType<typeof parseApiResponse>>
  try {
    parsed = await parseApiResponse(response)
  } catch {
    throw new Error(fallbackMessage)
  }
  if (response.status === 409) {
    const conflict = parseAgentDeleteConflict(parsed.data)
    if (conflict) {
      return { kind: "blocked", conflict }
    }
  }

  throw new Error(fallbackMessage)
}

"use client";

import { apiRequest } from "@/lib/api-wrapper";
import { getApiUrl } from "@/lib/utils";
import type { AgentCard } from "@/components/chat/ChatStartScreen";

export const toAgentId = (
  agent: { id?: number | string | null } | null | undefined
): number | null => {
  const id = Number(agent?.id);
  // Number(null) and Number("") both coerce to 0, not NaN - reject those
  // (and negatives/non-integers) explicitly rather than treating them as a
  // valid agent id.
  return Number.isInteger(id) && id > 0 ? id : null;
};

// Agents created via a template persist that template's id
// (Agent.template_id, set server-side), so reuse is keyed off that stable id
// rather than the user-editable display name - renaming the agent doesn't
// break the correlation.
export const findExistingAgentForTemplate = (
  templateId: string,
  agentList: AgentCard[]
): AgentCard | null => agentList.find((agent) => agent.template_id === templateId) ?? null;

/**
 * Resolve the agent to send this template's prompt to: reuse a locally-known
 * agent when possible, otherwise defer to the server's atomic get-or-create
 * (POST /api/agents/from-template/resolve). The server owns all reuse,
 * ownership, name-disambiguation and publish semantics - repeat calls
 * converge on the same agent, and another user's agents are never touched.
 */
export async function resolveAgentForTemplate(
  templateId: string,
  knownAgents: AgentCard[]
): Promise<{ agent: AgentCard; created: boolean }> {
  const knownExisting = findExistingAgentForTemplate(templateId, knownAgents);
  if (knownExisting) {
    return { agent: knownExisting, created: false };
  }

  const response = await apiRequest(`${getApiUrl()}/api/agents/from-template/resolve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ template_id: templateId }),
  });
  if (!response.ok) {
    throw new Error(`Failed to resolve agent from template (${response.status})`);
  }

  const body = await response.json();
  if (!body?.agent || toAgentId(body.agent) === null) {
    throw new Error("Malformed resolve response: missing agent");
  }
  return { agent: body.agent as AgentCard, created: Boolean(body.created) };
}

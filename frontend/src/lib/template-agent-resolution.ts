"use client";

import { apiRequest } from "@/lib/api-wrapper";
import { getApiUrl } from "@/lib/utils";
import { isPublishedAgent } from "@/lib/agent-ui-access";
import type { AgentCard } from "@/components/chat/ChatStartScreen";

// The backend's DuplicateAgentNameError maps to exactly this 400 detail
// (see POST /api/agents/from-template) - matching on it keeps the
// bare-400-means-collision assumption below from silently misfiring if a
// different validation error is ever added to that endpoint.
export const DUPLICATE_AGENT_NAME_DETAIL = "Agent with this name already exists";

export const toAgentId = (
  agent: { id?: number | string | null } | null | undefined
): number | null => {
  const id = Number(agent?.id);
  return Number.isNaN(id) ? null : id;
};

// Agents created via a template persist that template's id
// (Agent.template_id, set server-side in create_agent_from_template), so
// reuse is keyed off that stable id rather than the user-editable display
// name - renaming the agent no longer breaks the correlation.
export const findExistingAgentForTemplate = (
  templateId: string,
  agentList: AgentCard[]
): AgentCard | null => agentList.find((agent) => agent.template_id === templateId) ?? null;

async function deleteAgentBestEffort(agentId: number): Promise<void> {
  try {
    await apiRequest(`${getApiUrl()}/api/agents/${agentId}`, { method: "DELETE" });
  } catch (error) {
    // Best-effort cleanup: surfacing this would mask the original publish
    // failure the caller is already about to throw. Worst case the agent is
    // reclaimed later via the 400-retry lookup in resolveAgentForTemplate.
    console.error(`Failed to roll back orphaned draft agent ${agentId}:`, error);
  }
}

async function publishAgent(agentId: number): Promise<void> {
  const response = await apiRequest(`${getApiUrl()}/api/agents/${agentId}/publish`, {
    method: "POST",
  });
  if (!response.ok) {
    // Don't leave an unpublished, unreferenced draft behind - a later send
    // from this template would otherwise orphan a new agent every time
    // publish happens to fail, instead of retrying cleanly.
    await deleteAgentBestEffort(agentId);
    throw new Error(`Failed to publish agent (${response.status})`);
  }
}

/**
 * Reuse an already-created agent for this template if one exists; only
 * create + publish a new one the first time a template is used. Returns the
 * full agent record (not just an id) so callers can keep local state
 * accurate without hand-building a stub.
 */
export async function resolveAgentForTemplate(
  templateId: string,
  templateName: string,
  knownAgents: AgentCard[]
): Promise<{ agent: AgentCard; created: boolean }> {
  const knownExisting = findExistingAgentForTemplate(templateId, knownAgents);
  if (knownExisting) {
    return { agent: knownExisting, created: false };
  }

  const createResponse = await apiRequest(`${getApiUrl()}/api/agents/from-template`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ template_id: templateId }),
  });

  if (createResponse.status === 400) {
    const body = await createResponse.json().catch(() => null);
    if (body?.detail !== DUPLICATE_AGENT_NAME_DETAIL) {
      throw new Error(`Failed to create agent from template (${createResponse.status})`);
    }

    // The default template name collided with an existing agent our local
    // (published-only) list didn't know about. Look it up by template_id
    // first - that's the same template, created elsewhere (another tab,
    // session, or a still-draft agent) - and reuse it rather than minting
    // a duplicate.
    const listResponse = await apiRequest(`${getApiUrl()}/api/agents`);
    if (!listResponse.ok) {
      throw new Error(`Failed to create agent from template (${createResponse.status})`);
    }
    const allAgents = await listResponse.json();
    const agentList = Array.isArray(allAgents) ? allAgents : [];
    const templateMatch = agentList.find(
      (agent) => agent && agent.template_id === templateId
    );

    if (templateMatch && toAgentId(templateMatch) !== null) {
      if (!isPublishedAgent(templateMatch)) {
        await publishAgent(templateMatch.id);
        templateMatch.status = "published";
      }
      return { agent: templateMatch, created: false };
    }

    // No agent from this template exists - the name collision is with an
    // unrelated agent that merely shares the template's default name.
    // Retry once with a disambiguated name so this template can still be
    // used.
    const retryResponse = await apiRequest(`${getApiUrl()}/api/agents/from-template`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ template_id: templateId, name: `${templateName} (${Date.now()})` }),
    });
    if (!retryResponse.ok) {
      throw new Error(`Failed to create agent from template (${retryResponse.status})`);
    }
    const retryAgent = await retryResponse.json();
    await publishAgent(retryAgent.id);
    return { agent: { ...retryAgent, status: "published" }, created: true };
  }

  if (!createResponse.ok) {
    throw new Error(`Failed to create agent from template (${createResponse.status})`);
  }
  const createdAgent = await createResponse.json();

  await publishAgent(createdAgent.id);

  return { agent: { ...createdAgent, status: "published" }, created: true };
}

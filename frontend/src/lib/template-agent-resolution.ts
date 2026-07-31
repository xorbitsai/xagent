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

/**
 * Resolve the agent to send this template's prompt to via the server's
 * atomic get-or-create (POST /api/agents/from-template/resolve). The server
 * owns all reuse, ownership, name-disambiguation and publish semantics -
 * repeat calls converge on the same agent, and another user's agents are
 * never touched.
 *
 * There used to be a client-side fast path here (a `.find()` over a locally
 * cached agent list, keyed on Agent.template_id) that skipped the server
 * call when a match was already known. It was removed (PR review finding
 * B5): that cached list comes from GET /api/agents, which under a
 * team-scope hook includes teammates' published, team-visible agents with
 * no ownership check - reintroducing exactly the cross-user exposure this
 * server-side resolve exists to prevent. The server round-trip is the
 * correctness boundary; there is no client-side shortcut that preserves it.
 */
export async function resolveAgentForTemplate(
  templateId: string
): Promise<{ agent: AgentCard; created: boolean }> {
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

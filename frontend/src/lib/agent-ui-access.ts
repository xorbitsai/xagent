export interface AgentUiAccess {
  id: number | string;
  status?: string | null;
  readonly?: boolean;
  can_edit?: boolean;
  can_publish?: boolean;
  can_delete?: boolean;
}

export const isPublishedAgent = (agent: AgentUiAccess) => agent.status === "published";

export const canEditAgent = (agent: AgentUiAccess) =>
  agent.readonly !== true && agent.can_edit !== false;

export const canPublishAgent = (agent: AgentUiAccess) =>
  canEditAgent(agent) && agent.can_publish !== false;

export const canDeleteAgent = (agent: AgentUiAccess) =>
  canEditAgent(agent) && agent.can_delete !== false;

export const canRunAgent = (agent: AgentUiAccess) => isPublishedAgent(agent);

export const getAgentTaskHref = (agentId: number | string) =>
  `/task?agent=${encodeURIComponent(String(agentId))}`;

export const getAgentChatHref = (agent: AgentUiAccess) =>
  getAgentTaskHref(agent.id);

export const findRunnableAgentById = <T extends AgentUiAccess>(
  agents: T[],
  agentId?: string | null
) => {
  if (!agentId) return undefined;
  return agents.find((agent) => String(agent.id) === agentId && canRunAgent(agent));
};

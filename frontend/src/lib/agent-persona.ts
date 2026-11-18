import type { AvatarPersona } from "@/lib/page-extension-contracts"

/**
 * The persona of a real Agent, for `PersonaAvatar` and the conversation. Build
 * it here instead of by hand so the Agent's id always travels with its name and
 * image: a distribution that styles an Agent's avatar finds the style by it.
 * Ids arrive as numbers or numeric strings depending on the API; anything that
 * is not a positive integer becomes no id, never a wrong one.
 */
export function toAgentPersona(
  agent: { id?: number | string | null; name: string },
  avatar: string | null | undefined,
): AvatarPersona {
  const id = Number(agent.id)
  return {
    name: agent.name,
    avatar: avatar ?? null,
    agentId: Number.isInteger(id) && id > 0 ? id : null,
  }
}

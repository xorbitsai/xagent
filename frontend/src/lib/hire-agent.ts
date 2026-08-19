"use client";

import { apiRequest } from "@/lib/api-wrapper";
import { getApiUrl } from "@/lib/utils";
import type { PersonaInfo } from "@/types/template";
import { resolveAgentForTemplate, toAgentId } from "@/lib/template-agent-resolution";

/** Strings the caller supplies (already localized via useI18n) to compose
 * the seeded opening message around persona.intro/kickoff_questions,
 * which are authored per-template and not themselves i18n keys. */
export interface HireMessageStrings {
  beforeWeStart: string;
  closingNote: string;
}

/**
 * Build the plain-text opening chat message a Hire flow seeds: the
 * persona's intro, its kickoff questions as a bullet list (if any), and a
 * closing note. Markdown bullets ("- ") render the same way any other
 * assistant message's list does - no interactions/structured form here,
 * this is purely informational text sent before the user has said anything.
 */
export function buildSeedAssistantMessage(
  persona: PersonaInfo,
  strings: HireMessageStrings
): string {
  const parts = [persona.intro.trim()].filter(Boolean);

  if (persona.kickoff_questions.length > 0) {
    const bulletList = persona.kickoff_questions.map((question) => `- ${question}`).join("\n");
    parts.push(`${strings.beforeWeStart}\n\n${bulletList}`);
  }

  if (strings.closingNote) {
    parts.push(strings.closingNote);
  }

  return parts.join("\n\n");
}

export interface HireAgentResult {
  taskId: number;
  agentId: number;
  created: boolean;
}

/**
 * Hire a marketplace persona: resolve (get-or-create) the user's
 * quick-access agent for this template under the persona's display name,
 * then open a fresh task seeded with the persona's opening message so the
 * agent "speaks first" - the zero-configuration pitch of the AI Team
 * Marketplace. Reuses the same server-side resolve as the older template
 * quick-access flow (template-agent-resolution.ts); this only adds the
 * seeded first message and a persona-derived agent name/task title on top.
 *
 * Calling this again for an already-hired template would mint a second
 * seeded task on top of the existing agent (resolve reuses the agent, but
 * task/create always creates a new task) - callers must only invoke this
 * from the "not yet hired" path and route straight to the agent's chat
 * (e.g. `/agent/{hired_agent_id}`) once `template.hired` is true.
 */
export async function hireAgentFromTemplate(
  templateId: string,
  persona: PersonaInfo,
  strings: HireMessageStrings
): Promise<HireAgentResult> {
  const { agent, created } = await resolveAgentForTemplate(templateId, persona.name);
  const agentId = toAgentId(agent);
  if (agentId === null) {
    throw new Error("Malformed resolve response: missing agent id");
  }

  const title = persona.role ? `${persona.name} — ${persona.role}` : persona.name;
  const response = await apiRequest(`${getApiUrl()}/api/chat/task/create`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title,
      agent_id: agentId,
      seed_assistant_message: buildSeedAssistantMessage(persona, strings),
    }),
  });
  if (!response.ok) {
    throw new Error(`Failed to create task for hired agent (${response.status})`);
  }

  const body = await response.json();
  const taskId = Number(body?.task_id);
  if (!Number.isInteger(taskId) || taskId <= 0) {
    throw new Error("Malformed task create response: missing task_id");
  }

  return { taskId, agentId, created };
}

"use client";

import { apiRequest } from "@/lib/api-wrapper";
import { getApiUrl } from "@/lib/utils";
import type { ConnectionInfo, PersonaInfo } from "@/types/template";
import { resolveAgentForTemplate, toAgentId } from "@/lib/template-agent-resolution";

/** Strings the caller supplies (already localized via useI18n) to compose
 * the seeded opening message around persona.intro/kickoff_questions,
 * which are authored per-template and not themselves i18n keys. */
export interface HireMessageStrings {
  beforeWeStart: string;
  closingNote: string;
  /** Label for the "connect your apps" card, seeded alongside the message
   * when the template has any connections. See buildConnectAppsInteraction. */
  connectAppsLabel: string;
}

/**
 * Build the "connect_apps" interaction seeded alongside the opening
 * message, from the template's own `connections` list - just their display
 * names, matched against useMcpApps() by ConnectAppsField at render time
 * (grouping by OAuth provider so e.g. Gmail + Calendar share one Google
 * sign-in). `null` when the template has no connections, so callers can
 * skip attaching seed_interactions entirely rather than sending an empty
 * card.
 */
export function buildConnectAppsInteraction(
  connections: ConnectionInfo[],
  label: string
): { type: "connect_apps"; field: string; label: string; apps: string[] } | null {
  // .trim() before the filter: a whitespace-only name (a template authoring
  // slip - blank strings pass most "did they fill this in" checks) would
  // otherwise pass Boolean() and reach ConnectAppsField as an app name
  // nothing in the catalog can ever match, silently dropped by
  // resolveRows/findMatchingMcpApp there instead of being caught here.
  const appNames = connections.map((connection) => connection.name.trim()).filter(Boolean);
  if (appNames.length === 0) return null;

  return { type: "connect_apps", field: "connect_apps", label, apps: appNames };
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
  const hasKickoffQuestions = persona.kickoff_questions.length > 0;

  if (hasKickoffQuestions) {
    const bulletList = persona.kickoff_questions.map((question) => `- ${question}`).join("\n");
    parts.push(`${strings.beforeWeStart}\n\n${bulletList}`);
  }

  // closingNote's copy ("Answer what you can...") refers back to the
  // kickoff questions above it - dangling and contextless without them.
  if (hasKickoffQuestions && strings.closingNote) {
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
export async function hireAgentFromTemplate({
  templateId,
  persona,
  strings,
  connections = [],
  abortIfIdentityChanged,
}: {
  templateId: string;
  persona: PersonaInfo;
  strings: HireMessageStrings;
  connections?: ConnectionInfo[];
  /** Checked once, between resolve and task/create - a PR review finding
   * caught that this function itself makes 2 independent network calls,
   * each authenticated with whatever session apiRequest finds live at the
   * moment IT fires, not one pinned at the start of this function. A
   * caller whose own identity-swap guard only checked before invoking
   * this function (the onboarding wizard's case) can't otherwise prevent
   * resolve and task/create running under two DIFFERENT identities if a
   * swap lands in the gap between them. Optional and unused by other
   * callers (e.g. the templates marketplace quick-access flow), which
   * don't accumulate identity-bound answers over time the same way. */
  abortIfIdentityChanged?: () => boolean;
}): Promise<HireAgentResult> {
  const { agent, created } = await resolveAgentForTemplate(templateId, persona.name);
  const agentId = toAgentId(agent);
  if (agentId === null) {
    throw new Error("Malformed resolve response: missing agent id");
  }
  if (abortIfIdentityChanged?.()) {
    throw new Error("Aborted: authenticated identity changed mid-hire");
  }

  const title = persona.role ? `${persona.name} — ${persona.role}` : persona.name;
  const connectAppsInteraction = buildConnectAppsInteraction(connections, strings.connectAppsLabel);
  const response = await apiRequest(`${getApiUrl()}/api/chat/task/create`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title,
      agent_id: agentId,
      seed_assistant_message: buildSeedAssistantMessage(persona, strings),
      ...(connectAppsInteraction ? { seed_interactions: [connectAppsInteraction] } : {}),
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

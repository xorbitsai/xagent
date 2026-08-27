import type { TranslationKey } from "@/i18n/translations";

/** The 8 "what does your team do?" options from the reference onboarding UI
 * (onboarding.html's WORK array) - `id` doubles as the value persisted to
 * `users.preferences.department` via PATCH /api/auth/me/preferences. */
export type OnboardingWorkId =
  | "marketing"
  | "sales"
  | "support"
  | "ops"
  | "finance"
  | "people"
  | "product"
  | "other";

export interface OnboardingWorkOption {
  id: OnboardingWorkId;
  labelKey: TranslationKey;
}

export const ONBOARDING_WORK: OnboardingWorkOption[] = [
  { id: "marketing", labelKey: "onboarding.work.marketing" },
  { id: "sales", labelKey: "onboarding.work.sales" },
  { id: "support", labelKey: "onboarding.work.support" },
  { id: "ops", labelKey: "onboarding.work.ops" },
  { id: "finance", labelKey: "onboarding.work.finance" },
  { id: "people", labelKey: "onboarding.work.people" },
  { id: "product", labelKey: "onboarding.work.product" },
  { id: "other", labelKey: "onboarding.work.other" },
];

/** A goal is a sentence someone would actually say about their week, tied to
 * the template that answers it - mirrors onboarding.html's GOALS array
 * exactly (id/label/tpl/fn), with `tpl` translated from a display name into
 * our real template id. `fn` is the work type it belongs to, used to bring
 * the goals matching what the user just said they do to the front of the
 * list on the Goals step. */
export interface OnboardingGoal {
  id: string;
  labelKey: TranslationKey;
  templateId: string;
  fn: OnboardingWorkId;
}

export const ONBOARDING_GOALS: OnboardingGoal[] = [
  { id: "inbox", labelKey: "onboarding.goal.inbox", templateId: "support-inbox-manager", fn: "support" },
  { id: "social", labelKey: "onboarding.goal.social", templateId: "marketing-social-media-content-manager", fn: "marketing" },
  { id: "meetings", labelKey: "onboarding.goal.meetings", templateId: "sales-meeting-agent", fn: "sales" },
  { id: "support", labelKey: "onboarding.goal.support", templateId: "support-ai-chatbot-agent", fn: "support" },
  { id: "leads", labelKey: "onboarding.goal.leads", templateId: "sales-email-lead-response-agent", fn: "sales" },
  { id: "research", labelKey: "onboarding.goal.research", templateId: "sales-research-enricher", fn: "sales" },
  { id: "docs", labelKey: "onboarding.goal.docs", templateId: "general-doc-summarizer-action-extractor", fn: "ops" },
  { id: "decks", labelKey: "onboarding.goal.decks", templateId: "marketing-collateral-agent", fn: "marketing" },
  { id: "linkedin", labelKey: "onboarding.goal.linkedin", templateId: "marketing-linkedIn-content-manager", fn: "marketing" },
  { id: "ads", labelKey: "onboarding.goal.ads", templateId: "marketing-google-ads-recommendation", fn: "marketing" },
  { id: "kb", labelKey: "onboarding.goal.kb", templateId: "support-kb-writer", fn: "support" },
  { id: "blog", labelKey: "onboarding.goal.blog", templateId: "marketing-content-agent", fn: "marketing" },
];

/** Shown on the My team step when nobody picked a goal (the reference UI's
 * "Not sure yet — show me everyone" path still recommends a starting three,
 * rather than showing nothing) - Maya, Ellie and Kevin, matching
 * onboarding.html's own fallback exactly. */
export const ONBOARDING_FALLBACK_TEMPLATE_IDS = [
  "marketing-social-media-content-manager",
  "support-inbox-manager",
  "sales-meeting-agent",
];

/** The 5 voice options the Launch step offers - ids match
 * VALID_USER_VOICES/apply_user_voice on the backend (src/xagent/web/api/auth.py,
 * src/xagent/web/api/agents.py) exactly; a mismatch here would let someone
 * pick a voice this frontend can display but the backend would 422 on save. */
export type OnboardingVoiceId = "professional" | "friendly" | "concise" | "warm" | "playful";

export interface OnboardingVoiceOption {
  id: OnboardingVoiceId;
  nameKey: TranslationKey;
  sayKey: TranslationKey;
}

export const ONBOARDING_VOICES: OnboardingVoiceOption[] = [
  { id: "professional", nameKey: "onboarding.voice.professional.name", sayKey: "onboarding.voice.professional.say" },
  { id: "friendly", nameKey: "onboarding.voice.friendly.name", sayKey: "onboarding.voice.friendly.say" },
  { id: "concise", nameKey: "onboarding.voice.concise.name", sayKey: "onboarding.voice.concise.say" },
  { id: "warm", nameKey: "onboarding.voice.warm.name", sayKey: "onboarding.voice.warm.say" },
  { id: "playful", nameKey: "onboarding.voice.playful.name", sayKey: "onboarding.voice.playful.say" },
];

export const ONBOARDING_DEFAULT_VOICE: OnboardingVoiceId = "professional";

/** Goals reordered so the ones matching the selected work type come first -
 * the same twelve options, just resorted (mirrors onboarding.html's
 * renderGoals). A stable sort: goals keep their relative order within each
 * of the two buckets. */
export function reorderGoalsByWork(work: string): OnboardingGoal[] {
  return [...ONBOARDING_GOALS].sort((a, b) => {
    const aMatches = a.fn === work ? 0 : 1;
    const bMatches = b.fn === work ? 0 : 1;
    return aMatches - bMatches;
  });
}

export interface RecommendedTemplate {
  templateId: string;
  /** The goal that recommended this template, for the card's "why" footer -
   * null when nothing was picked and this came from the fallback list. */
  goalId: string | null;
}

/** Which templates to show on the My team step, and why - a recommendation
 * that cannot explain itself is just a default with better manners.
 * Deduplicates (one card per template even if multiple goals point to it)
 * and preserves ONBOARDING_GOALS' own order (not selection order).
 *
 * Deliberately NOT capped at 3 here - the caller (page.tsx's
 * validRecommended) filters this down to templates that actually loaded
 * with a persona first, THEN caps at 3. Capping here, before that filter,
 * used to mean a 4th-ranked match with a real persona could never fill a
 * slot vacated by a top-3 match that turned out to have none. */
export function recommendedTemplates(selectedGoalIds: string[]): RecommendedTemplate[] {
  const picked = ONBOARDING_GOALS.filter((goal) => selectedGoalIds.includes(goal.id));
  const seen = new Set<string>();
  const out: RecommendedTemplate[] = [];
  for (const goal of picked) {
    if (seen.has(goal.templateId)) continue;
    seen.add(goal.templateId);
    out.push({ templateId: goal.templateId, goalId: goal.id });
  }
  if (out.length === 0) {
    for (const templateId of ONBOARDING_FALLBACK_TEMPLATE_IDS) {
      out.push({ templateId, goalId: null });
    }
  }
  return out;
}

/** "Gmail and Outlook", not "Gmail, Outlook" - the agent is talking, not
 * printing a CSV. Matches onboarding.html's andList exactly for English
 * (no Oxford comma - Intl.ListFormat's "long" style adds one, which would
 * drift from the pixel/copy-matched reference). `and` and `separator` are
 * caller-supplied, already-localized strings so this doesn't hardcode
 * English/Western punctuation (", ") into otherwise-translated copy - e.g.
 * Chinese conventionally uses "、" between list items, not a Latin comma. */
export function joinWithAnd(items: string[], and = "and", separator = ", "): string {
  if (items.length === 0) return "";
  if (items.length === 1) return items[0];
  return `${items.slice(0, -1).join(separator)} ${and} ${items[items.length - 1]}`;
}

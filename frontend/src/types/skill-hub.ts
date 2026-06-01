/**
 * Skill Hub types — mirror the Pydantic models in
 * ``xagent_saas/api/skill_hub.py``. SaaS-closed-source surface; the
 * frontend never talks to the open-source ``/api/skills/*`` endpoints
 * directly so the Hub contract stays consistent end-to-end.
 */

export type SkillSource = "builtin" | "user" | "external";

/** Trust badge. ``null`` is "not yet scanned" — most skills on
 * ClawHub today fall into this bucket, so the UI shouldn't treat it
 * as a warning. */
export type ScanStatus = "clean" | "suspicious" | "malicious" | null;

// ──────────────────────────────────────────────────────────────────
// Local skills (already installed)
// ──────────────────────────────────────────────────────────────────

export interface SkillSummary {
  name: string;
  description: string;
  when_to_use: string;
  tags: string[];
  source: SkillSource;
}

export interface SkillDetail extends SkillSummary {
  content: string;        // raw SKILL.md
  execution_flow: string;
  files: string[];
  path: string;
}

// ──────────────────────────────────────────────────────────────────
// ClawHub registry (browse / install)
// ──────────────────────────────────────────────────────────────────

export interface RegistrySkillSummary {
  slug: string;
  displayName: string;
  summary: string;
  version: string | null;
  ownerHandle: string | null;
  installs: number | null;
  updatedAt: number | null;     // unix ms
  scanStatus: ScanStatus;
  /** Set to the local skill name when this slug is already installed. */
  installedAs: string | null;
}

export interface RegistrySkillDetail {
  slug: string;
  displayName: string;
  summary: string;
  version: string | null;
  ownerHandle: string | null;
  homepage: string | null;
  readme: string | null;
  scanStatus: ScanStatus;
  moderation: Record<string, unknown> | null;
  installedAs: string | null;
  raw: Record<string, unknown>;
}

export interface RegistryListResponse {
  items: RegistrySkillSummary[];
  nextCursor: string | null;
}

/** Hand-picked skill surfaced on the Discover tab's Featured rail.
 * Same shape as RegistrySkillSummary plus our editorial pitch. */
export interface FeaturedSkill extends RegistrySkillSummary {
  featuredReason: string;
}

/** Pagination metadata. ClawHub has no count endpoint; backend walks
 * the cursor pages and caches. ``truncated`` means the backend hit a
 * safety cap before exhausting cursors (so the real total is ≥ this). */
export interface RegistryStats {
  sort: string;
  total: number;
  walked_pages: number;
  truncated: boolean;
}

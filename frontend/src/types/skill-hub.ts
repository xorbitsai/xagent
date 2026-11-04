/**
 * Skill Hub types — mirror the Pydantic models in
 * ``xagent_saas/api/skill_hub.py``. SaaS-closed-source surface; the
 * frontend never talks to the open-source ``/api/skills/*`` endpoints
 * directly so the Hub contract stays consistent end-to-end.
 */

export type SkillSource = "builtin" | "user" | "team" | "external";

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
  scope?: string | null;
  effective?: boolean;
  shadowed_by?: string | null;
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
  registrySource: string;
  raw: Record<string, unknown>;
}

export interface RegistryListResponse {
  items: RegistrySkillSummary[];
  nextCursor: string | null;
}



/** A supported skill registry (returned by GET /registries). */
export interface RegistryInfo {
  id: string;
  displayName: string;
  description: string;
}

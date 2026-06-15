"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ChevronLeft, Loader2, Plus } from "lucide-react";

import { MarkdownEditor } from "@/components/skill-hub/markdown-editor";
import { useI18n } from "@/contexts/i18n-context";
import { apiRequest } from "@/lib/api-wrapper";
import { getApiUrl } from "@/lib/utils";

/**
 * Create-new-skill page.
 *
 * Two inputs: ``name`` (used verbatim as the on-disk directory name
 * and as the skill's identifier in the SkillManager — the parser
 * ignores the frontmatter ``name`` field, so the dir name is the
 * source of truth) and the SKILL.md body.
 *
 * On success we redirect to ``/skill-hub/<name>`` so the user can
 * immediately see the parsed detail view.
 */

const STARTER_TEMPLATE = `---
description: One-line summary of what this skill does.
when_to_use: "Use this skill when the user wants to ..."
tags:
  - example
---

# My Skill

## Overview

Describe what this skill is for.

## When to Use

Spell out the scenarios where the agent should pick this skill.

## Execution Flow

1. First step
2. Second step
3. Final output
`;

export default function NewSkillPage() {
  const apiBase = getApiUrl();
  const router = useRouter();
  const { t } = useI18n();

  const [name, setName] = useState("");
  const [skillMd, setSkillMd] = useState(STARTER_TEMPLATE);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // The backend regex matches client-side validation here so we can
  // disable the Create button before the user wastes a round trip.
  const nameValid = /^[A-Za-z0-9_-]+$/.test(name);
  const canSubmit = nameValid && skillMd.trim().length > 0 && !saving;

  const handleCreate = async () => {
    setSaving(true);
    setError(null);
    try {
      const res = await apiRequest(`${apiBase}/api/skill-hub/create`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, skill_md: skillMd }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setError(body.detail || `Create failed (HTTP ${res.status})`);
        return;
      }
      // Skip back to detail page — the create response is summary-only
      // and the detail page will fetch the parsed content.
      router.push(`/skill-hub/${encodeURIComponent(name)}`);
    } catch (e) {
      console.error(e);
      setError("Network error while creating.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex h-full flex-col overflow-y-auto bg-background">
      <div className="mx-auto w-full flex-1 px-6 py-10">
        <Link
          href="/skill-hub"
          className="mb-6 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
        >
          <ChevronLeft className="h-4 w-4" /> {t("skillHub.newSkill.back")}
        </Link>

        <div className="mb-6 flex items-center justify-between gap-3">
          <h1 className="text-2xl font-bold tracking-tight">{t("skillHub.newSkill.title")}</h1>
          <button
            type="button"
            onClick={handleCreate}
            disabled={!canSubmit}
            className="inline-flex h-9 items-center gap-1.5 rounded-md bg-primary px-4 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
            {saving ? t("skillHub.newSkill.creating") : t("skillHub.newSkill.create")}
          </button>
        </div>

        <div className="mb-4">
          <label className="mb-1 block text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            {t("skillHub.newSkill.skillName")}
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="my-skill"
            className="h-10 w-full rounded-md border bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
          />
          <p className="mt-1 text-[11px] text-muted-foreground">
            {t("skillHub.newSkill.nameHint", { path: "~/.xagent/skills/" })}
          </p>
          {name && !nameValid && (
            <p className="mt-1 text-[11px] text-destructive">
              {t("skillHub.newSkill.nameInvalid", { pattern: "[A-Za-z0-9_-]+" })}
            </p>
          )}
        </div>

        <MarkdownEditor
          value={skillMd}
          onChange={setSkillMd}
          rows={26}
          placeholder={t("skillHub.newSkill.placeholder")}
        />

        {error && (
          <div className="mt-4 rounded-md border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}

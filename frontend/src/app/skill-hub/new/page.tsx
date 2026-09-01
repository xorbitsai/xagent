"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ChevronLeft, Loader2, Plus, Upload } from "lucide-react";

import { MarkdownEditor } from "@/components/skill-hub/markdown-editor";
import { useI18n } from "@/contexts/i18n-context";
import {
  apiRequest,
  getUploadErrorMessage,
  parseApiResponse,
} from "@/lib/api-wrapper";
import { getApiUrl } from "@/lib/utils";

/**
 * New-skill page. Two ways in, sharing one name field.
 *
 * Authoring: ``name`` plus the SKILL.md body. The name is used verbatim
 * as the on-disk directory name and as the skill's identifier in the
 * SkillManager; on this path the parser ignores the frontmatter ``name``
 * field, so the directory name is the source of truth.
 *
 * Import: a ``.zip`` skill folder or a bare SKILL.md. Identity is *not*
 * the directory name here — the backend resolves it as typed name → zip
 * root → frontmatter ``name`` → filename stem, so leaving the field
 * empty takes the name from the upload. A typed name is honoured or
 * refused, never rewritten.
 *
 * Both flows redirect to ``/skill-hub/<name>`` using the name the
 * backend reports, so the user sees the parsed detail view.
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
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // The backend regex and its 64-character bound, mirrored so the Create
  // button can be disabled before the user wastes a round trip. Both halves
  // are shared with Upload: the typed name is sent as the override, and the
  // backend refuses one it would have to rewrite.
  const NAME_MAX = 64;
  const namePatternOk = /^[A-Za-z0-9_-]+$/.test(name);
  const nameTooLong = name.length > NAME_MAX;
  const nameValid = namePatternOk && !nameTooLong;
  // One busy gate for both flows. They were independent, so a slow upload and
  // a Create submit could run at once and their completion-order redirects
  // could land on the other operation's page.
  const busy = saving || uploading;
  const canSubmit = nameValid && skillMd.trim().length > 0 && !busy;

  // Upload a .zip skill bundle or a bare SKILL.md. The backend resolves the
  // name (typed override → zip root dir → frontmatter name → filename stem)
  // and we redirect to whatever name it reports back.
  const handleUpload = async (file: File) => {
    if (busy) return;
    setUploading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      // Let the typed name win over whatever the archive happens to be
      // called; the backend refuses one it would have to rewrite.
      const typedName = name;
      if (typedName.length > 0) form.append("name", typedName);
      // No Content-Type header: the browser must set the multipart boundary
      // itself, so passing FormData to apiRequest directly is deliberate.
      // The backend reconciles an identical owner/name/content replay, so a
      // response lost after commit can be retried without turning success
      // into a duplicate-name failure.
      const res = await apiRequest(`${apiBase}/api/skill-hub/upload`, {
        method: "POST",
        body: form,
      });
      const parsed = await parseApiResponse(res);
      if (!res.ok) {
        // Shared helper rather than reading `detail` by hand: it also covers
        // the 422 validation array (rendering it as a React child throws) and
        // an HTML 413 from a proxy that rejected the body before it reached us.
        setError(
          getUploadErrorMessage(res, parsed, {
            generic: t("skillHub.newSkill.uploadFailed", { status: res.status }),
            tooLarge: t("skillHub.newSkill.uploadTooLarge"),
            proxy: t("skillHub.newSkill.uploadProxyError"),
          }),
        );
        return;
      }
      // Not `name`: that is the component's state, in scope in this block.
      const createdName =
        parsed.data && typeof parsed.data === "object" && "name" in parsed.data
          ? (parsed.data as { name?: string }).name
          : undefined;
      router.push(
        createdName ? `/skill-hub/${encodeURIComponent(createdName)}` : "/skill-hub",
      );
    } catch (e) {
      console.error(e);
      setError(t("skillHub.newSkill.uploadNetworkError"));
    } finally {
      setUploading(false);
    }
  };

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
      setError(t("skillHub.newSkill.createNetworkError"));
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

        <div
          role="button"
          tabIndex={0}
          onClick={() => !busy && fileInputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              // Space scrolls the page by default while a role="button" is
              // focused, so the dropzone would jump the view as it opened.
              e.preventDefault();
              if (!busy) fileInputRef.current?.click();
            }
          }}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            // Exactly one file: taking files[0] silently discarded the rest,
            // so a multi-file drop imported one skill with no explanation.
            const dropped = e.dataTransfer.files;
            if (!dropped || dropped.length === 0) return;
            if (dropped.length > 1) {
              setError(t("skillHub.newSkill.importOneFileOnly"));
              return;
            }
            void handleUpload(dropped[0]);
          }}
          className={`mb-6 flex cursor-pointer items-center gap-3 rounded-lg border border-dashed p-4 transition-colors ${
            dragOver ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
          } ${busy ? "pointer-events-none opacity-60" : ""}`}
        >
          {uploading ? (
            <Loader2 className="h-5 w-5 shrink-0 animate-spin text-muted-foreground" />
          ) : (
            <Upload className="h-5 w-5 shrink-0 text-muted-foreground" />
          )}
          <div>
            <p className="text-sm font-medium">
              {uploading ? t("skillHub.newSkill.importing") : t("skillHub.newSkill.importTitle")}
            </p>
            <p className="text-xs text-muted-foreground">{t("skillHub.newSkill.importHint")}</p>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".zip,.md"
            className="hidden"
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void handleUpload(f);
              e.target.value = "";
            }}
          />
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
            disabled={busy}
            className="h-10 w-full rounded-md border bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40 disabled:opacity-60"
          />
          <p className="mt-1 text-[11px] text-muted-foreground">
            {t("skillHub.newSkill.nameHint", { path: "~/.xagent/skills/" })}
          </p>
          {name && !namePatternOk && (
            <p className="mt-1 text-[11px] text-destructive">
              {t("skillHub.newSkill.nameInvalid", { pattern: "[A-Za-z0-9_-]+" })}
            </p>
          )}
          {nameTooLong && (
            <p className="mt-1 text-[11px] text-destructive">
              {t("skillHub.newSkill.nameTooLong", { max: NAME_MAX })}
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

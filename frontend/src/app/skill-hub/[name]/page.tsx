"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
  ChevronLeft,
  FileText,
  Loader2,
  Pencil,
  Save,
  Trash2,
  X,
} from "lucide-react";

import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { MarkdownRenderer } from "@/components/ui/markdown-renderer";
import { MarkdownEditor } from "@/components/skill-hub/markdown-editor";
import { apiRequest } from "@/lib/api-wrapper";
import { cn, getApiUrl } from "@/lib/utils";
import type { SkillDetail, SkillSource } from "@/types/skill-hub";

/**
 * Skill detail page. Two modes:
 *   - **view**: SKILL.md rendered as markdown + file listing
 *   - **edit** (user-source only): split-pane MarkdownEditor with
 *     Save / Cancel. Save PUTs the new SKILL.md, refreshes the
 *     parsed detail, and drops back to view mode.
 *
 * Builtin and external skills can be viewed but not edited or
 * deleted — the backend enforces both checks too, but we hide the
 * buttons up front so the affordance matches the actual capability.
 */

function badgeForSource(source: SkillSource) {
  switch (source) {
    case "builtin":
      return { label: "Built-in", classes: "bg-violet-500/10 text-violet-600 border-violet-500/30" };
    case "user":
      return { label: "Installed", classes: "bg-emerald-500/10 text-emerald-600 border-emerald-500/30" };
    default:
      return { label: "External", classes: "bg-amber-500/10 text-amber-600 border-amber-500/30" };
  }
}

export default function SkillDetailPage() {
  const params = useParams<{ name: string }>();
  const router = useRouter();
  const apiBase = getApiUrl();

  const [skill, setSkill] = useState<SkillDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // Edit-mode state
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const [deleting, setDeleting] = useState(false);
  // Drives the ConfirmDialog. ``true`` = dialog open.
  const [confirmRemove, setConfirmRemove] = useState(false);

  const loadSkill = async () => {
    if (!params?.name) return;
    try {
      const res = await apiRequest(
        `${apiBase}/api/skill-hub/installed/${encodeURIComponent(params.name)}`,
      );
      if (!res.ok) {
        setError(`Failed to load skill (HTTP ${res.status})`);
        return;
      }
      const data = (await res.json()) as SkillDetail;
      setSkill(data);
    } catch (e) {
      console.error(e);
      setError("Network error.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSkill();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params?.name, apiBase]);

  const startEdit = () => {
    if (!skill) return;
    setDraft(skill.content);
    setSaveError(null);
    setEditing(true);
  };

  const cancelEdit = () => {
    setEditing(false);
    setDraft("");
    setSaveError(null);
  };

  const handleSave = async () => {
    if (!skill) return;
    setSaving(true);
    setSaveError(null);
    try {
      const res = await apiRequest(
        `${apiBase}/api/skill-hub/installed/${encodeURIComponent(skill.name)}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ skill_md: draft }),
        },
      );
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setSaveError(body.detail || `Save failed (HTTP ${res.status})`);
        return;
      }
      // Re-fetch detail so files / tags / description reflect the
      // saved content. The PUT response is a summary, not the full
      // detail.
      await loadSkill();
      setEditing(false);
      setDraft("");
    } catch (e) {
      console.error(e);
      setSaveError("Network error while saving.");
    } finally {
      setSaving(false);
    }
  };

  // Click the Remove button → open the dialog. Actual DELETE runs
  // from ``performDelete`` once the user confirms.
  const handleDelete = () => {
    if (!skill) return;
    setConfirmRemove(true);
  };

  const performDelete = async () => {
    if (!skill) return;
    setDeleting(true);
    try {
      const res = await apiRequest(
        `${apiBase}/api/skill-hub/installed/${encodeURIComponent(skill.name)}`,
        { method: "DELETE" },
      );
      if (!res.ok && res.status !== 204) {
        const body = await res.json().catch(() => ({}));
        alert(body.detail || `Delete failed (HTTP ${res.status})`);
        setDeleting(false);
        setConfirmRemove(false);
        return;
      }
      router.push("/skill-hub");
    } catch (e) {
      console.error(e);
      alert("Network error while deleting.");
      setDeleting(false);
      setConfirmRemove(false);
    }
  };

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }
  if (error || !skill) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {error || "Skill not found."}
        </div>
      </div>
    );
  }

  const badge = badgeForSource(skill.source);
  const editable = skill.source === "user";

  return (
    <div className="flex h-full flex-col overflow-y-auto bg-background">
      <div className={cn(
        "mx-auto w-full flex-1 px-6 py-10",
        editing ? "max-w-6xl" : "max-w-4xl",
      )}>
        <Link
          href="/skill-hub"
          className="mb-6 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
        >
          <ChevronLeft className="h-4 w-4" /> Back to Skill Hub
        </Link>

        {/* Header */}
        <div className="mb-6 flex items-start gap-4">
          <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-400 to-teal-600 text-white shadow-sm">
            <FileText className="h-6 w-6" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <h1 className="truncate text-2xl font-bold tracking-tight">
                {skill.name}
              </h1>
              <span
                className={cn(
                  "shrink-0 rounded-full border px-2 py-0.5 text-[11px] font-medium",
                  badge.classes,
                )}
              >
                {badge.label}
              </span>
            </div>
            {skill.description && (
              <p className="mt-1 text-sm text-muted-foreground">{skill.description}</p>
            )}
            {skill.tags.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1">
                {skill.tags.map((t) => (
                  <span
                    key={t}
                    className="rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground"
                  >
                    {t}
                  </span>
                ))}
              </div>
            )}
          </div>
          {!editing && editable && (
            <div className="flex shrink-0 gap-2">
              <button
                type="button"
                onClick={startEdit}
                className="inline-flex h-9 items-center gap-1.5 rounded-md border bg-card px-3 text-xs font-medium hover:bg-muted"
              >
                <Pencil className="h-3.5 w-3.5" />
                Edit
              </button>
              <button
                type="button"
                onClick={handleDelete}
                disabled={deleting}
                className="inline-flex h-9 items-center gap-1.5 rounded-md border border-destructive/40 bg-destructive/10 px-3 text-xs font-medium text-destructive hover:bg-destructive/20 disabled:opacity-50"
              >
                {deleting ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Trash2 className="h-3.5 w-3.5" />
                )}
                Remove
              </button>
            </div>
          )}
          {editing && (
            <div className="flex shrink-0 gap-2">
              <button
                type="button"
                onClick={cancelEdit}
                disabled={saving}
                className="inline-flex h-9 items-center gap-1.5 rounded-md border bg-card px-3 text-xs font-medium hover:bg-muted disabled:opacity-50"
              >
                <X className="h-3.5 w-3.5" />
                Cancel
              </button>
              <button
                type="button"
                onClick={handleSave}
                disabled={saving || draft === skill.content}
                className="inline-flex h-9 items-center gap-1.5 rounded-md bg-primary px-3 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
              >
                {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
                {saving ? "Saving…" : "Save"}
              </button>
            </div>
          )}
        </div>

        {editing ? (
          <>
            <MarkdownEditor value={draft} onChange={setDraft} rows={26} />
            {saveError && (
              <div className="mt-3 rounded-md border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive">
                {saveError}
              </div>
            )}
          </>
        ) : (
          <>
            <section className="mb-6 rounded-xl border bg-card p-6">
              <div className="mb-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                SKILL.md
              </div>
              {skill.content ? (
                <MarkdownRenderer
                  content={skill.content}
                  className="prose-sm text-foreground prose-headings:text-foreground prose-strong:text-foreground prose-code:text-foreground"
                />
              ) : (
                <div className="text-sm italic text-muted-foreground">SKILL.md is empty.</div>
              )}
            </section>
            {skill.files.length > 0 && (
              <section className="mb-6 rounded-xl border bg-card p-6">
                <div className="mb-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                  Files · {skill.files.length}
                </div>
                <ul className="space-y-1">
                  {skill.files.map((f) => (
                    <li key={f} className="flex items-center gap-2 text-xs text-foreground/80">
                      <FileText className="h-3 w-3 text-muted-foreground" />
                      <span className="truncate">{f}</span>
                    </li>
                  ))}
                </ul>
              </section>
            )}
            <div className="text-[11px] text-muted-foreground">
              Installed at: <code className="rounded bg-muted px-1 py-0.5">{skill.path}</code>
            </div>
          </>
        )}
      </div>
      <ConfirmDialog
        isOpen={confirmRemove}
        onOpenChange={(open) => {
          // Keep the dialog open while the request is in flight so the
          // spinner stays visible; only allow close when idle.
          if (!open && !deleting) setConfirmRemove(false);
        }}
        onConfirm={performDelete}
        title="Remove skill"
        description={
          skill
            ? `Remove "${skill.name}" from your installed skills? This deletes the files under ${skill.path}.`
            : ""
        }
        confirmText="Remove"
        isLoading={deleting}
      />
    </div>
  );
}

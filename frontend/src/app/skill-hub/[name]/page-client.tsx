"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
  ChevronLeft,
  ExternalLink,
  FileText,
  Loader2,
  Pencil,
  Plus,
  Save,
  Trash2,
  X,
} from "lucide-react";

import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { MarkdownRenderer } from "@/components/ui/markdown-renderer";
import { MarkdownEditor } from "@/components/skill-hub/markdown-editor";
import { badgeForSource, ScanBadge } from "@/components/skill-hub/badges";
import { useI18n } from "@/contexts/i18n-context";
import { apiRequest } from "@/lib/api-wrapper";
import { cn, getApiUrl } from "@/lib/utils";
import type {
  RegistrySkillDetail,
  SkillDetail,
} from "@/types/skill-hub";

/**
 * Skill detail page.
 *
 * Two entry points:
 *   - **Installed**  → ``/skill-hub/<name>`` loads from
 *     ``/api/skill-hub/installed/{name}``. Shows SKILL.md, file
 *     listing, and Edit/Remove for user-owned skills.
 *   - **Registry**   → same URL when the skill isn't installed yet.
 *     Falls back to ``/api/skill-hub/registry/{name}`` and renders
 *     the upstream README with an Install button.
 */

type ViewMode = "installed" | "registry";

export default function SkillDetailPage() {
  const params = useParams<{ name: string }>();
  const router = useRouter();
  const apiBase = getApiUrl();
  const { t } = useI18n();

  const [mode, setMode] = useState<ViewMode>("installed");

  // ── installed state ────────────────────────────────────────
  const [skill, setSkill] = useState<SkillDetail | null>(null);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [confirmRemove, setConfirmRemove] = useState(false);

  // ── registry state ─────────────────────────────────────────
  const [regSkill, setRegSkill] = useState<RegistrySkillDetail | null>(null);
  const [installing, setInstalling] = useState(false);
  const [installError, setInstallError] = useState<string | null>(null);

  // ── shared ─────────────────────────────────────────────────
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const loadSkill = useCallback(async () => {
    const name = params?.name;
    if (!name) return;
    setLoading(true);
    setError(null);

    // 1) Try installed first.
    try {
      const res = await apiRequest(
        `${apiBase}/api/skill-hub/installed/${encodeURIComponent(name)}`,
      );
      if (res.ok) {
        setSkill((await res.json()) as SkillDetail);
        setMode("installed");
        setLoading(false);
        return;
      }
    } catch {
      // Fall through to registry.
    }

    // 2) Not installed → try registry.
    try {
      const res = await apiRequest(
        `${apiBase}/api/skill-hub/registry/${encodeURIComponent(name)}`,
      );
      if (res.ok) {
        const data = (await res.json()) as RegistrySkillDetail;
        setRegSkill(data);
        setMode("registry");
        setLoading(false);
        return;
      }
      setError(`Failed to load skill (HTTP ${res.status})`);
    } catch (e) {
      console.error(e);
      setError("Network error.");
    } finally {
      setLoading(false);
    }
  }, [params?.name, apiBase]);

  useEffect(() => {
    loadSkill();
  }, [loadSkill]);

  // ── installed-edit handlers ────────────────────────────────

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

  // ── registry-install handler ───────────────────────────────

  const handleInstall = async () => {
    if (!regSkill) return;
    setInstalling(true);
    setInstallError(null);
    try {
      const registrySource = regSkill.registrySource || "clawhub";
      const res = await apiRequest(`${apiBase}/api/skill-hub/install/${encodeURIComponent(registrySource)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ slug: regSkill.slug }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setInstallError(body.detail || `Install failed (HTTP ${res.status})`);
        return;
      }
      // Reload — will pick up the now-installed skill.
      await loadSkill();
    } catch (e) {
      console.error(e);
      setInstallError("Network error while installing.");
    } finally {
      setInstalling(false);
    }
  };

  // ── render ─────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }
  if (error || (mode === "installed" && !skill) || (mode === "registry" && !regSkill)) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {error || t("skillHub.detail.notFound")}
        </div>
      </div>
    );
  }

  // ── Installed view ─────────────────────────────────────────

  if (mode === "installed" && skill) {
    const badge = badgeForSource(skill.source);
    const editable = skill.source === "user";

    return (
      <div className="flex h-full flex-col overflow-y-auto bg-background">
        <div className="mx-auto w-full flex-1 px-6 py-10">
          <Link
            href="/skill-hub"
            className="mb-6 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
          >
            <ChevronLeft className="h-4 w-4" /> {t("skillHub.detail.back")}
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
                  {t(`skillHub.sourceLabel.${badge.label}`)}
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
                  {t("skillHub.detail.edit")}
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
                  {t("skillHub.detail.remove")}
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
                  {t("skillHub.detail.cancel")}
                </button>
                <button
                  type="button"
                  onClick={handleSave}
                  disabled={saving || draft === skill.content}
                  className="inline-flex h-9 items-center gap-1.5 rounded-md bg-primary px-3 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                >
                  {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
                  {saving ? t("skillHub.detail.saving") : t("skillHub.detail.save")}
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
              {skill.source === "team" && (
                <div className="mb-4 rounded-md border border-blue-500/30 bg-blue-500/10 p-3 text-xs text-blue-700">
                  {t("skillHub.detail.teamNote")}
                </div>
              )}
              <section className="mb-6 rounded-xl border bg-card p-6">
                <div className="mb-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                  {t("skillHub.detail.skillMd")}
                </div>
                {skill.content ? (
                  <MarkdownRenderer
                    content={skill.content}
                    className="prose-sm text-foreground prose-headings:text-foreground prose-strong:text-foreground prose-code:text-foreground"
                  />
                ) : (
                  <div className="text-sm italic text-muted-foreground">{t("skillHub.detail.skillMdEmpty")}</div>
                )}
              </section>
              {skill.files.length > 0 && (
                <section className="mb-6 rounded-xl border bg-card p-6">
                  <div className="mb-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                    {t("skillHub.detail.files")} · {skill.files.length}
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
                {t("skillHub.detail.installedAt")} <code className="rounded bg-muted px-1 py-0.5">{skill.path}</code>
              </div>
            </>
          )}
        </div>
        <ConfirmDialog
          isOpen={confirmRemove}
          onOpenChange={(open) => {
            if (!open && !deleting) setConfirmRemove(false);
          }}
          onConfirm={performDelete}
          title={t("skillHub.detail.removeTitle")}
          description={
            skill
              ? t("skillHub.detail.removeDescription", { name: skill.name, path: skill.path })
              : ""
          }
          confirmText={t("skillHub.detail.remove")}
          isLoading={deleting}
        />
      </div>
    );
  }

  // ── Registry view ──────────────────────────────────────────

  if (mode === "registry" && regSkill) {
    return (
      <div className="flex h-full flex-col overflow-y-auto bg-background">
        <div className="mx-auto w-full flex-1 px-6 py-10">
          <Link
            href="/skill-hub"
            className="mb-6 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
          >
            <ChevronLeft className="h-4 w-4" /> {t("skillHub.detail.back")}
          </Link>

          {/* Header */}
          <div className="mb-6 flex items-start gap-4">
            <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-400 to-indigo-600 text-white shadow-sm">
              <ExternalLink className="h-6 w-6" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <h1 className="truncate text-2xl font-bold tracking-tight">
                  {regSkill.displayName || regSkill.slug}
                </h1>
                <ScanBadge status={regSkill.scanStatus} />
              </div>
              {regSkill.ownerHandle && (
                <div className="mt-1 text-sm text-muted-foreground">
                  {t("skillHub.discover.by", { owner: regSkill.ownerHandle })}
                </div>
              )}
              {regSkill.version && (
                <div className="mt-1 text-[11px] text-muted-foreground">
                  v{regSkill.version}
                </div>
              )}
              {regSkill.summary && (
                <p className="mt-2 text-sm text-muted-foreground">{regSkill.summary}</p>
              )}
            </div>
            <div className="flex shrink-0 gap-2">
              <button
                type="button"
                onClick={handleInstall}
                disabled={installing}
                className="inline-flex h-9 items-center gap-1.5 rounded-md bg-primary px-3 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
              >
                {installing ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Plus className="h-3.5 w-3.5" />
                )}
                {installing ? t("skillHub.discover.installing") : t("skillHub.discover.install")}
              </button>
              {regSkill.homepage && (
                <a
                  href={regSkill.homepage}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex h-9 items-center gap-1.5 rounded-md border bg-card px-3 text-xs font-medium hover:bg-muted"
                >
                  <ExternalLink className="h-3.5 w-3.5" />
                  Homepage
                </a>
              )}
            </div>
          </div>

          {installError && (
            <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive">
              {installError}
            </div>
          )}

          {/* Readme */}
          <section className="rounded-xl border bg-card p-6">
            <div className="mb-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              README
            </div>
            {regSkill.readme ? (
              <MarkdownRenderer
                content={regSkill.readme}
                className="prose-sm text-foreground prose-headings:text-foreground prose-strong:text-foreground prose-code:text-foreground"
              />
            ) : (
              <div className="text-sm italic text-muted-foreground">
                No README available for this skill.
              </div>
            )}
          </section>
        </div>
      </div>
    );
  }

  return null;
}

"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Compass,
  Library,
  Loader2,
  Plus,
  Search,
  Trash2,
} from "lucide-react";

import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { Select, type SelectOption } from "@/components/ui/select";
import { badgeForSource, ScanBadge } from "@/components/skill-hub/badges";
import { useI18n } from "@/contexts/i18n-context";
import { apiRequest } from "@/lib/api-wrapper";
import { cn, getApiUrl } from "@/lib/utils";
import type {
  RegistryListResponse,
  RegistrySkillSummary,
  SkillSummary,
} from "@/types/skill-hub";

const PAGE_SIZE = 30;
const SEARCH_LIMIT = 100;  // /search has no pagination — fetch as many as upstream allows

/**
 * Skill Hub — manage agent skills. Two tabs:
 *   - Discover  → browse / search ClawHub registry, click Install
 *   - My Skills → list locally installed skills + create new + delete
 *
 * Install / create both round-trip through the SkillManager singleton
 * the chat runtime uses, so newly added skills become available to
 * agents on the next task without a process restart.
 */

type Tab = "discover" | "mine";

function formatInstalls(n: number | null): string {
  if (n == null) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

export default function SkillHubPage() {
  const apiBase = getApiUrl();
  const { t } = useI18n();
  const [tab, setTab] = useState<Tab>("mine");

  // ── installed (mine) ──────────────────────────────────────────────
  const [installed, setInstalled] = useState<SkillSummary[]>([]);
  const [installedLoading, setInstalledLoading] = useState(true);
  const [installedError, setInstalledError] = useState<string | null>(null);
  const [installedQuery, setInstalledQuery] = useState("");
  const [installedSort, setInstalledSort] = useState<"source" | "name">("source");
  const [deleting, setDeleting] = useState<string | null>(null);
  // Held by the ConfirmDialog flow — the skill name pending a
  // user-confirmed delete. Null when the dialog is closed.
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  // ── registry (discover) ───────────────────────────────────────────
  const [registry, setRegistry] = useState<RegistrySkillSummary[]>([]);
  const [registryLoading, setRegistryLoading] = useState(false);
  const [registryError, setRegistryError] = useState<string | null>(null);
  const [registryQuery, setRegistryQuery] = useState("");
  // Default to ``installsCurrent`` rather than ``trending`` because
  // ClawHub's trending list caps at ~100 (it's a top-N ranking,
  // not a full list), which renders "Page 1 of 1" and gives the
  // misleading impression there are very few skills. Most-installed
  // paginates over the full ~2,000 corpus and gives a richer first
  // impression with 30+ pages to browse.
  const [registrySort, setRegistrySort] = useState("installsCurrent");
  const [registrySource, setRegistrySource] = useState("");
  const [sourceOptions, setSourceOptions] = useState<SelectOption[]>([]);

  // Load available registries on mount so we know the default source.
  useEffect(() => {
    (async () => {
      try {
        const res = await apiRequest(`${getApiUrl()}/api/skill-hub/registries`);
        if (!res.ok) return;
        const data = (await res.json()) as { id: string; displayName: string; description: string }[];
        const opts: SelectOption[] = data.map((r) => ({
          value: r.id,
          label: r.displayName,
          description: r.description,
        }));
        setSourceOptions(opts);
        if (opts.length > 0) {
          setRegistrySource(opts[0].value);
        }
      } catch {
        // Non-critical.
      }
    })();
  }, []);

  // Cursor-based pagination. ClawHub uses opaque cursors (not offsets),
  // so we keep a history stack to support Prev/Next:
  //   cursorHistory[0] = null  (page 1 cursor)
  //   cursorHistory[1] = <cursor returned by page 1>
  //   ...
  const [cursorHistory, setCursorHistory] = useState<(string | null)[]>([null]);
  const [historyIndex, setHistoryIndex] = useState(0);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const discoverInitializedRef = useRef(false);
  const [installingSlug, setInstallingSlug] = useState<string | null>(null);
  const [installError, setInstallError] = useState<{ slug: string; msg: string } | null>(null);

  // Debounce search input so we don't hammer the proxy on every keystroke.
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const reloadInstalled = useCallback(async () => {
    try {
      const res = await apiRequest(`${apiBase}/api/skill-hub/installed`);
      if (!res.ok) {
        setInstalledError(`Failed to load installed skills (HTTP ${res.status})`);
        return;
      }
      setInstalled((await res.json()) as SkillSummary[]);
      setInstalledError(null);
    } catch (e) {
      console.error(e);
      setInstalledError("Could not reach the skill hub.");
    } finally {
      setInstalledLoading(false);
    }
  }, [apiBase]);

  /**
   * Load one page of registry results — REPLACE-style (not append).
   *
   * Two modes:
   *   - `q` non-empty → ``/registry/search`` (single page, no cursor)
   *   - `q` empty     → ``/registry/list`` with the cursor from
   *     ``pageCursors[targetIndex]`` (null for page 1)
   *
   * The fetch always replaces the visible list; pagination cursor
   * bookkeeping happens in ``goToNextPage`` / ``goToPrevPage`` and in
   * the sort/query-reset effect.
   */
  const loadRegistry = useCallback(
    async (opts: { q: string; sort: string; source: string; cursor: string | null }) => {
      const { q, sort, source, cursor } = opts;
      setRegistryLoading(true);
      setRegistryError(null);
      try {
        const trimmed = q.trim();
        let url: string;
        if (trimmed) {
          url = `${apiBase}/api/skill-hub/registry/search?q=${encodeURIComponent(trimmed)}&source=${encodeURIComponent(source)}&limit=${SEARCH_LIMIT}`;
        } else {
          const cursorPart = cursor ? `&cursor=${encodeURIComponent(cursor)}` : "";
          url = `${apiBase}/api/skill-hub/registry/list?sort=${encodeURIComponent(sort)}&source=${encodeURIComponent(source)}&limit=${PAGE_SIZE}${cursorPart}`;
        }
        const res = await apiRequest(url);
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          setRegistryError(body.detail || `Registry load failed (HTTP ${res.status})`);
          setRegistry([]);
          setNextCursor(null);
          return;
        }
        const data = (await res.json()) as RegistryListResponse;
        setRegistry(data.items);
        setNextCursor(trimmed ? null : data.nextCursor);
      } catch (e) {
        console.error(e);
        setRegistryError("Network error while loading registry.");
        setRegistry([]);
        setNextCursor(null);
      } finally {
        setRegistryLoading(false);
      }
    },
    [apiBase],
  );

  const goNext = useCallback(() => {
    if (!nextCursor || registryLoading) return;
    setCursorHistory((prev) => {
      if (prev.length > historyIndex + 1) return prev;
      return [...prev, nextCursor];
    });
    setHistoryIndex((i) => i + 1);
  }, [nextCursor, historyIndex, registryLoading]);

  const goPrev = useCallback(() => {
    if (!registryLoading && historyIndex > 0) {
      setHistoryIndex((i) => i - 1);
    }
  }, [historyIndex, registryLoading]);

  // Initial loads.
  useEffect(() => {
    reloadInstalled();
  }, [reloadInstalled]);

  // When sort or search query changes, reset to page 1.
  useEffect(() => {
    if (tab !== "discover") return;
    setCursorHistory([null]);
    setHistoryIndex(0);
    setNextCursor(null);
  }, [tab, registryQuery, registrySort, registrySource]);

  // Pagination / sort change — immediate fetch (no debounce).
  useEffect(() => {
    if (tab !== "discover") return;
    if (registryQuery.trim()) return;
    const cursor = cursorHistory[historyIndex] ?? null;
    loadRegistry({ q: "", sort: registrySort, source: registrySource, cursor });
  }, [tab, registrySort, registrySource, historyIndex, cursorHistory, loadRegistry, registryQuery]);

  // Search — debounced 250ms.
  useEffect(() => {
    if (tab !== "discover" || !registryQuery.trim()) return;
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    searchTimerRef.current = setTimeout(
      () => loadRegistry({ q: registryQuery, sort: registrySort, source: registrySource, cursor: null }),
      250,
    );
    return () => {
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    };
  }, [tab, registryQuery, registrySort, registrySource, loadRegistry]);

  // Discover initial load — fires once when user first opens Discover.
  useEffect(() => {
    if (tab !== "discover" || discoverInitializedRef.current) return;
    discoverInitializedRef.current = true;
    loadRegistry({ q: "", sort: registrySort, source: registrySource, cursor: null });
  }, [tab, registrySort, registrySource, loadRegistry]);

  // ── handlers ──────────────────────────────────────────────────────

  const handleInstall = async (slug: string) => {
    setInstallingSlug(slug);
    setInstallError(null);
    try {
      const res = await apiRequest(`${apiBase}/api/skill-hub/install/${registrySource}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ slug }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setInstallError({
          slug,
          msg: body.detail || `Install failed (HTTP ${res.status})`,
        });
        return;
      }
      // Optimistically flip the Installed badge in every rail that
      // can carry this slug — registry grid. The
      // update avoids a visible round-trip.
      setRegistry((prev) =>
        prev.map((p) => (p.slug === slug ? { ...p, installedAs: slug } : p)),
      );
      // Refresh My Skills so the badge / count updates.
      await reloadInstalled();
    } catch (e) {
      console.error(e);
      setInstallError({ slug, msg: "Network error while installing." });
    } finally {
      setInstallingSlug(null);
    }
  };

  // Click → arm the ConfirmDialog. The actual DELETE fires from
  // ``performDelete`` once the user clicks Confirm.
  const handleDelete = (name: string) => {
    setDeleteTarget(name);
  };

  const performDelete = async () => {
    const name = deleteTarget;
    if (!name) return;
    setDeleting(name);
    try {
      const res = await apiRequest(
        `${apiBase}/api/skill-hub/installed/${encodeURIComponent(name)}`,
        { method: "DELETE" },
      );
      if (!res.ok && res.status !== 204) {
        const body = await res.json().catch(() => ({}));
        alert(body.detail || `Delete failed (HTTP ${res.status})`);
        return;
      }
      await reloadInstalled();
    } catch (e) {
      console.error(e);
      alert("Network error while deleting.");
    } finally {
      setDeleting(null);
      setDeleteTarget(null);
    }
  };

  // ── derived ───────────────────────────────────────────────────────

  const installedFiltered = useMemo(() => {
    const q = installedQuery.trim().toLowerCase();
    const base = q
      ? installed.filter(
        (s) =>
          s.name.toLowerCase().includes(q) ||
          s.description.toLowerCase().includes(q) ||
          s.tags.some((t) => t.toLowerCase().includes(q)),
      )
      : installed;
    // ``source`` mode preserves the original "user → team → builtin → external"
    // grouping (already applied server-side); just sort by name within.
    // ``name`` mode collapses all sources into one alphabetical list.
    const sorted = [...base];
    if (installedSort === "name") {
      sorted.sort((a, b) => a.name.localeCompare(b.name));
    } else {
      sorted.sort((a, b) => {
        const rank = { user: 0, team: 1, builtin: 2, external: 3 } as const;
        const r = rank[a.source] - rank[b.source];
        return r !== 0 ? r : a.name.localeCompare(b.name);
      });
    }
    return sorted;
  }, [installed, installedQuery, installedSort]);

  return (
    <div className="flex h-full flex-col overflow-y-auto bg-background">
      <div className="mx-auto w-full flex-1 px-6 py-10">
        {/* Header */}
        <div className="mb-6 flex flex-col items-center gap-3 text-center">
          <div className="rounded-2xl bg-emerald-500/10 p-3 text-emerald-500">
            <Library className="h-8 w-8" />
          </div>
          <h1 className="text-3xl font-bold tracking-tight">{t("skillHub.page.title")}</h1>
          <p className="max-w-2xl text-sm text-muted-foreground">
            {t("skillHub.page.subtitle")}
          </p>
        </div>

        {/* Tabs + create */}
        <div className="mb-6 flex items-center justify-between gap-2">
          <div className="inline-flex items-center gap-1.5 rounded-2xl bg-muted p-1">
            <button
              type="button"
              onClick={() => setTab("discover")}
              className={cn(
                "inline-flex items-center gap-2 rounded-xl px-5 py-2.5 text-sm font-bold transition-all",
                tab === "discover"
                  ? "bg-white text-foreground shadow-sm ring-1 ring-border/50"
                  : "text-muted-foreground hover:bg-white/70 hover:text-foreground",
              )}
            >
              <Compass className="h-4 w-4" />
              {t("skillHub.tabs.discover")}
            </button>
            <button
              type="button"
              onClick={() => setTab("mine")}
              className={cn(
                "inline-flex items-center gap-2 rounded-xl px-5 py-2.5 text-sm font-bold transition-all",
                tab === "mine"
                  ? "bg-white text-foreground shadow-sm ring-1 ring-border/50"
                  : "text-muted-foreground hover:bg-white/70 hover:text-foreground",
              )}
            >
              <Library className="h-4 w-4" />
              {t("skillHub.tabs.mySkills")}
              {installed.length > 0 && (
                <span className={cn(
                  "ml-0.5 rounded-full px-2 py-px text-[11px] font-semibold",
                  tab === "mine"
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted-foreground/15 text-muted-foreground",
                )}>
                  {installed.length}
                </span>
              )}
            </button>
          </div>
          <Link
            href="/skill-hub/new"
            className="inline-flex items-center gap-1.5 rounded-md border bg-card px-3 py-1.5 text-xs font-medium hover:bg-muted"
          >
            <Plus className="h-3.5 w-3.5" />
            {t("skillHub.page.createNew")}
          </Link>
        </div>

        {tab === "discover" ? (
          <DiscoverTab
            registry={registry}
            loading={registryLoading}
            error={registryError}
            query={registryQuery}
            setQuery={setRegistryQuery}
            sort={registrySort}
            setSort={setRegistrySort}
            source={registrySource}
            onSourceChange={setRegistrySource}
            sourceOptions={sourceOptions}
            pageIndex={historyIndex}
            hasNext={!!nextCursor && !registryQuery.trim()}
            hasPrev={historyIndex > 0 && !registryQuery.trim()}
            onPrev={goPrev}
            onNext={goNext}
            installingSlug={installingSlug}
            installError={installError}
            onInstall={handleInstall}
          />
        ) : (
          <MyTab
            installed={installedFiltered}
            totalCount={installed.length}
            loading={installedLoading}
            error={installedError}
            query={installedQuery}
            setQuery={setInstalledQuery}
            sort={installedSort}
            setSort={setInstalledSort}
            deleting={deleting}
            onDelete={handleDelete}
          />
        )}
      </div>

      {/* Project-style delete confirmation — matches dialogs used
       * elsewhere (Agents tab, KB delete). The dialog is mounted
       * once at the root rather than per-card so it doesn't fight
       * with grid hover state. */}
      <ConfirmDialog
        isOpen={!!deleteTarget}
        onOpenChange={(open) => {
          if (!open && deleting !== deleteTarget) setDeleteTarget(null);
        }}
        onConfirm={performDelete}
        title={t("skillHub.mySkills.removeTitle")}
        description={
          deleteTarget
            ? t("skillHub.mySkills.removeDescription", { name: deleteTarget, path: `~/.xagent/skills/${deleteTarget}/` })
            : ""
        }
        confirmText={t("skillHub.mySkills.remove")}
        isLoading={!!deleting && deleting === deleteTarget}
      />
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────
// Discover tab
// ────────────────────────────────────────────────────────────────────

function DiscoverTab({
  registry, loading, error, query, setQuery, sort, setSort,
  source, onSourceChange, sourceOptions,
  pageIndex, hasNext, hasPrev, onPrev, onNext,
  installingSlug, installError, onInstall,
}: {
  registry: RegistrySkillSummary[];
  loading: boolean;
  error: string | null;
  query: string;
  setQuery: (s: string) => void;
  sort: string;
  setSort: (s: string) => void;
  source: string;
  onSourceChange: (s: string) => void;
  sourceOptions: SelectOption[];
  pageIndex: number;
  hasNext: boolean;
  hasPrev: boolean;
  onPrev: () => void;
  onNext: () => void;
  installingSlug: string | null;
  installError: { slug: string; msg: string } | null;
  onInstall: (slug: string) => void;
}) {
  const { t } = useI18n();

  /** Sort dimensions ClawHub honors today. */
  const SORT_OPTIONS: SelectOption[] = [
    { value: "trending", label: t("skillHub.discover.sort.trending") },
    { value: "newest", label: t("skillHub.discover.sort.newest") },
    { value: "installsCurrent", label: t("skillHub.discover.sort.mostInstalled") },
    { value: "stars", label: t("skillHub.discover.sort.mostStarred") },
    { value: "updated", label: t("skillHub.discover.sort.recentlyUpdated") },
  ];

  const inSearchMode = !!query.trim();

  return (
    <>
      {/* "All" header — the section below this is the full sortable
       * browse grid. The label is what flips Discover from
       * "showcase" into "catalog" mode visually. */}
      <div className="mb-2 flex items-center gap-1.5 text-sm font-semibold">
        <Compass className="h-4 w-4 text-blue-500" />
        <span>{t("skillHub.discover.browseAll")}</span>
      </div>

      <div className="mb-3 flex flex-col gap-3 sm:flex-row sm:items-stretch">
        <div className="sm:w-48">
          <Select
            value={source}
            onValueChange={onSourceChange}
            options={sourceOptions}
          />
        </div>
        <div className="sm:w-48">
          {/* Sort doesn't apply in search mode — upstream /search has no
           * sort param. Disable rather than hide so the affordance stays
           * predictable. */}
          <Select
            value={sort}
            onValueChange={setSort}
            options={SORT_OPTIONS}
            disabled={inSearchMode}
          />
        </div>
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={t("skillHub.discover.searchPlaceholder")}
            className="h-10 w-full rounded-md border bg-background pl-10 pr-4 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
          />
        </div>
      </div>
      {/* Count line. */}
      {!loading && !error && registry.length > 0 && (
        <div className="mb-4 text-xs text-muted-foreground">
          {inSearchMode ? (
            <>
              {registry.length === 1
                ? t("skillHub.discover.pagination.resultFor", { count: registry.length, query: query.trim() })
                : t("skillHub.discover.pagination.resultsFor", { count: registry.length, query: query.trim() })}
            </>
          ) : (
            t("skillHub.discover.pagination.page", { page: pageIndex + 1 })
          )}
        </div>
      )}

      {loading ? (
        <div className="flex justify-center py-20">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      ) : error ? (
        <div className="mx-auto max-w-md rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      ) : registry.length === 0 ? (
        <div className="mt-10 rounded-xl border bg-card p-10 text-center text-sm text-muted-foreground">
          {query ? t("skillHub.discover.noMatchSearch") : t("skillHub.discover.registryEmpty")}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {registry.map((s) => {
              const installing = installingSlug === s.slug;
              const isInstalled = !!s.installedAs;
              const showError = installError?.slug === s.slug;
              return (
                <Link
                  key={s.slug}
                  href={`/skill-hub/${encodeURIComponent(s.slug)}`}
                  className="flex flex-col gap-2 rounded-xl border bg-card p-4 transition-all hover:border-primary/40 hover:shadow-sm"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="block truncate text-sm font-semibold leading-tight">
                          {s.displayName || s.slug}
                        </span>
                        <ScanBadge status={s.scanStatus} />
                      </div>
                    </div>
                    {s.ownerHandle && (
                      <span className="shrink-0 text-[11px] text-muted-foreground">
                        @{s.ownerHandle}
                      </span>
                    )}
                  </div>
                  <p className="flex-1 text-xs text-muted-foreground line-clamp-3">
                    {s.summary || t("skillHub.discover.noDescription")}
                  </p>
                  <div className="flex items-center justify-between gap-2 pt-1">
                    <div className="text-[11px] text-muted-foreground">
                      {s.version ? `v${s.version}` : ""}
                      {s.installs != null && (
                        <span className="ml-2">{t("skillHub.discover.installs", { count: formatInstalls(s.installs) })}</span>
                      )}
                    </div>
                    {isInstalled ? (
                      <span className="inline-flex items-center gap-1 rounded-md border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-1 text-[11px] font-medium text-emerald-600">
                        <Check className="h-3 w-3" /> {t("skillHub.discover.installed")}
                      </span>
                    ) : (
                      <button
                        type="button"
                        disabled={installing}
                        onClick={(e) => { e.preventDefault(); e.stopPropagation(); onInstall(s.slug); }}
                        className="inline-flex items-center gap-1 rounded-md bg-primary px-2.5 py-1 text-[11px] font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                      >
                        {installing ? (
                          <>
                            <Loader2 className="h-3 w-3 animate-spin" /> {t("skillHub.discover.installing")}
                          </>
                        ) : (
                          <>
                            <Plus className="h-3 w-3" /> {t("skillHub.discover.install")}
                          </>
                        )}
                      </button>
                    )}
                  </div>
                  {showError && (
                    <div className="rounded border border-destructive/40 bg-destructive/10 p-2 text-[11px] leading-snug text-destructive">
                      {installError!.msg}
                    </div>
                  )}
                </Link>
              );
            })}
          </div>
          {/* Prev / Next pagination. Only meaningful in browse mode
         * (search results don't paginate). Always rendered in browse
         * mode so the affordance is predictable — buttons disable
         * themselves when there's nowhere to go (e.g. sort=trending
         * only has ~100 items so Next disables on page 2). Hiding the
         * whole control on edge sorts made users think "the feature
         * disappeared". */}
          {!inSearchMode && (
            <div className="mt-5 flex items-center justify-end gap-3">
              <button
                type="button"
                onClick={onPrev}
                disabled={!hasPrev}
                className="inline-flex items-center gap-1 rounded-md border bg-card px-4 py-2 text-xs font-medium hover:bg-muted disabled:opacity-30"
              >
                <ChevronLeft className="h-3.5 w-3.5" />
                {t("skillHub.discover.pagination.prev")}
              </button>
              <span className="text-xs text-muted-foreground">
                {t("skillHub.discover.pagination.page", { page: pageIndex + 1 })}
              </span>
              <button
                type="button"
                onClick={onNext}
                disabled={!hasNext}
                className="inline-flex items-center gap-1 rounded-md border bg-card px-4 py-2 text-xs font-medium hover:bg-muted disabled:opacity-30"
              >
                {t("skillHub.discover.pagination.next")}
                <ChevronRight className="h-3.5 w-3.5" />
              </button>
            </div>
          )}
        </>
      )}
    </>
  );
}

// ────────────────────────────────────────────────────────────────────
// My Skills tab
// ────────────────────────────────────────────────────────────────────

function MyTab({
  installed, totalCount, loading, error, query, setQuery, sort, setSort, deleting, onDelete,
}: {
  installed: SkillSummary[];
  totalCount: number;
  loading: boolean;
  error: string | null;
  query: string;
  setQuery: (s: string) => void;
  sort: "source" | "name";
  setSort: (s: "source" | "name") => void;
  deleting: string | null;
  onDelete: (name: string) => void;
}) {
  const { t } = useI18n();

  const INSTALLED_SORT_OPTIONS: SelectOption[] = [
    { value: "source", label: t("skillHub.mySkills.sort.groupBySource"), description: t("skillHub.mySkills.sort.groupBySourceDesc") },
    { value: "name", label: t("skillHub.mySkills.sort.byName"), description: t("skillHub.mySkills.sort.byNameDesc") },
  ];

  return (
    <>
      <div className="mb-3 flex flex-col gap-3 sm:flex-row sm:items-stretch">
        <div className="sm:w-56">
          <Select
            value={sort}
            onValueChange={(v) => setSort(v as "source" | "name")}
            options={INSTALLED_SORT_OPTIONS}
          />
        </div>
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={t("skillHub.mySkills.searchPlaceholder")}
            className="h-10 w-full rounded-md border bg-background pl-10 pr-4 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
          />
        </div>
      </div>
      {!loading && !error && totalCount > 0 && (
        <div className="mb-4 text-[11px] text-muted-foreground">
          {query
            ? t("skillHub.mySkills.matchSummary", { shown: installed.length, total: totalCount })
            : t("skillHub.mySkills.skillsInstalled", { count: totalCount })}
        </div>
      )}

      {loading ? (
        <div className="flex justify-center py-20">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      ) : error ? (
        <div className="mx-auto max-w-md rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      ) : installed.length === 0 ? (
        <div className="mt-10 rounded-xl border bg-card p-10 text-center text-sm text-muted-foreground">
          {query ? (
            t("skillHub.mySkills.noFilterMatch")
          ) : (
            <>
              {t("skillHub.mySkills.noInstalled", { discover: t("skillHub.mySkills.discover") })}
            </>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {installed.map((s) => {
            const badge = badgeForSource(s.source);
            const removable = s.source === "user";
            return (
              <div
                key={s.name}
                className="group flex flex-col gap-2 rounded-xl border bg-card p-4 transition-all hover:border-primary/40 hover:shadow-sm"
              >
                <div className="flex items-start justify-between gap-2">
                  <Link
                    href={`/skill-hub/${encodeURIComponent(s.name)}`}
                    className="min-w-0 flex-1"
                  >
                    <div className="truncate text-sm font-semibold leading-tight">
                      {s.name}
                    </div>
                  </Link>
                  <span
                    className={cn(
                      "shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-medium",
                      badge.classes,
                    )}
                  >
                    {t(`skillHub.sourceLabel.${badge.label}`)}
                  </span>
                </div>
                <Link
                  href={`/skill-hub/${encodeURIComponent(s.name)}`}
                  className="flex-1 text-xs text-muted-foreground line-clamp-3"
                >
                  {s.description || t("skillHub.mySkills.noDescription")}
                </Link>
                {s.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {s.tags.slice(0, 4).map((t) => (
                      <span
                        key={t}
                        className="rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground"
                      >
                        {t}
                      </span>
                    ))}
                  </div>
                )}
                {removable && (
                  <button
                    type="button"
                    onClick={() => onDelete(s.name)}
                    disabled={deleting === s.name}
                    className="mt-1 inline-flex items-center gap-1 self-start text-[11px] text-destructive opacity-0 transition-opacity hover:underline group-hover:opacity-100 disabled:opacity-30"
                  >
                    {deleting === s.name ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <Trash2 className="h-3 w-3" />
                    )}
                    {t("skillHub.mySkills.remove")}
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}
    </>
  );
}

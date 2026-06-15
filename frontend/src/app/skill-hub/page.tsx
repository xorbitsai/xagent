"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Compass,
  Flame,
  Library,
  Loader2,
  Plus,
  Search,
  ShieldAlert,
  ShieldCheck,
  ShieldQuestion,
  Sparkles,
  Trash2,
} from "lucide-react";

import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { Select, type SelectOption } from "@/components/ui/select";
import { apiRequest } from "@/lib/api-wrapper";
import { cn, getApiUrl } from "@/lib/utils";
import type {
  FeaturedSkill,
  RegistryListResponse,
  RegistrySkillSummary,
  RegistryStats,
  ScanStatus,
  SkillSource,
  SkillSummary,
} from "@/types/skill-hub";

const PAGE_SIZE = 30;

// ──────────────────────────────────────────────────────────────────
// sessionStorage cache — stale-while-revalidate
//
// First-time visit to Discover: fetch normally + write cache.
// Re-visit within TTL: paint cached data instantly, then refetch
// in background and quietly update if the response changed.
//
// Per-tab not per-user (sessionStorage scope), so signing out
// doesn't need to clear it. Cleared on tab close. Keys are
// namespaced with ``sh:`` so they don't collide with anything
// else the app writes.
// ──────────────────────────────────────────────────────────────────

const CACHE_TTL_MS = 5 * 60 * 1000;  // 5 min, matches backend cache TTL

function readCache<T>(key: string): T | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem(`sh:${key}`);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { data: T; savedAt: number };
    if (Date.now() - parsed.savedAt > CACHE_TTL_MS) {
      window.sessionStorage.removeItem(`sh:${key}`);
      return null;
    }
    return parsed.data;
  } catch {
    return null;
  }
}

function writeCache(key: string, data: unknown): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.setItem(
      `sh:${key}`,
      JSON.stringify({ data, savedAt: Date.now() }),
    );
  } catch {
    // sessionStorage can throw on quota / private-mode — silent
    // failure is fine, the next fetch will repopulate.
  }
}

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

function badgeForSource(source: SkillSource) {
  switch (source) {
    case "builtin":
      return { label: "Built-in", classes: "bg-violet-500/10 text-violet-600 border-violet-500/30" };
    case "user":
      return { label: "Installed", classes: "bg-emerald-500/10 text-emerald-600 border-emerald-500/30" };
    case "team":
      return { label: "Team", classes: "bg-blue-500/10 text-blue-600 border-blue-500/30" };
    default:
      return { label: "External", classes: "bg-amber-500/10 text-amber-600 border-amber-500/30" };
  }
}

function ScanBadge({ status }: { status: ScanStatus }) {
  if (status === "clean") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-600">
        <ShieldCheck className="h-3 w-3" /> Scanned clean
      </span>
    );
  }
  if (status === "suspicious") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium text-amber-600">
        <ShieldAlert className="h-3 w-3" /> Flagged
      </span>
    );
  }
  if (status === "malicious") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full border border-rose-500/30 bg-rose-500/10 px-2 py-0.5 text-[10px] font-medium text-rose-600">
        <ShieldAlert className="h-3 w-3" /> Malicious
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-full border border-muted-foreground/20 bg-muted/40 px-2 py-0.5 text-[10px] font-medium text-muted-foreground">
      <ShieldQuestion className="h-3 w-3" /> Not scanned
    </span>
  );
}

function formatInstalls(n: number | null): string {
  if (n == null) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

export default function SkillHubPage() {
  const apiBase = getApiUrl();
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
  // Cursor-based pagination state.
  //
  // ClawHub uses opaque cursors, not offsets — we can only walk
  // forward / back one page at a time, and we don't know the total
  // page count (no count endpoint). To support Prev/Next we keep a
  // stack of the cursors we've seen:
  //
  //   pageCursors[0] === null    (the implicit "page 1" cursor)
  //   pageCursors[1] === <cursor returned by page-1 fetch>
  //   pageCursors[2] === <cursor returned by page-2 fetch>
  //   ...
  //
  // `pageIndex` points at the page currently rendered (0-based).
  // `nextCursorFromLast` is whatever the most recent fetch returned —
  // null when we're on the last page (Next button hides).
  const [pageCursors, setPageCursors] = useState<(string | null)[]>([null]);
  const [pageIndex, setPageIndex] = useState(0);
  const [nextCursorFromLast, setNextCursorFromLast] = useState<string | null>(null);
  // Pagination "total pages" comes from /registry/stats — the backend
  // walks ClawHub once per (sort, TTL) and caches. ``null`` means
  // "not yet known"; we render "Page N" without the total in that
  // case rather than blocking the grid.
  const [registryStats, setRegistryStats] = useState<RegistryStats | null>(null);
  // Featured rail — editorial pick fetched once on first Discover
  // open, then refreshed only when the user installs something
  // (so badges flip).
  const [featured, setFeatured] = useState<FeaturedSkill[]>([]);
  const [featuredLoading, setFeaturedLoading] = useState(false);
  const featuredLoadedRef = useRef(false);
  // "Popular this week" rail — ClawHub's installsCurrent sort. Gives
  // a second discovery angle that's complementary to trending (which
  // weights recency more heavily). Reuses /registry/list with a
  // different sort + smaller limit so we don't need a new endpoint.
  const [popular, setPopular] = useState<RegistrySkillSummary[]>([]);
  const [popularLoading, setPopularLoading] = useState(false);
  const popularLoadedRef = useRef(false);
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
    async (opts: { q: string; sort: string; cursor: string | null }) => {
      const { q, sort, cursor } = opts;
      setRegistryLoading(true);
      setRegistryError(null);
      try {
        const trimmed = q.trim();
        let url: string;
        if (trimmed) {
          url = `${apiBase}/api/skill-hub/registry/search?q=${encodeURIComponent(trimmed)}&limit=60`;
        } else {
          const cursorPart = cursor ? `&cursor=${encodeURIComponent(cursor)}` : "";
          url = `${apiBase}/api/skill-hub/registry/list?sort=${encodeURIComponent(sort)}&limit=60${cursorPart}`;
        }
        const res = await apiRequest(url);
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          setRegistryError(body.detail || `Registry load failed (HTTP ${res.status})`);
          setRegistry([]);
          setNextCursorFromLast(null);
          return;
        }
        const data = (await res.json()) as RegistryListResponse;
        setRegistry(data.items);
        // Search responses don't paginate; only browse mode advances.
        setNextCursorFromLast(trimmed ? null : data.nextCursor);
      } catch (e) {
        console.error(e);
        setRegistryError("Network error while loading registry.");
        setRegistry([]);
        setNextCursorFromLast(null);
      } finally {
        setRegistryLoading(false);
      }
    },
    [apiBase],
  );

  /** Advance to next page. If we've never visited this page before,
   * grow the cursor stack — otherwise just bump the index and rely
   * on the effect below to refetch. */
  const goToNextPage = useCallback(() => {
    if (!nextCursorFromLast || registryLoading) return;
    setPageCursors((prev) => {
      // Already have the cursor for the next page → don't duplicate.
      if (prev.length > pageIndex + 1) return prev;
      return [...prev, nextCursorFromLast];
    });
    setPageIndex(pageIndex + 1);
  }, [nextCursorFromLast, pageIndex, registryLoading]);

  const goToPrevPage = useCallback(() => {
    if (pageIndex <= 0 || registryLoading) return;
    setPageIndex(pageIndex - 1);
  }, [pageIndex, registryLoading]);

  /** Reset paging state — used whenever the user changes sort or
   * issues a new search, both of which invalidate the cursor stack. */
  const resetPagination = useCallback(() => {
    setPageCursors([null]);
    setPageIndex(0);
    setNextCursorFromLast(null);
  }, []);

  // Stale-while-revalidate: paint cached data instantly (if any),
  // then refetch in the background to upgrade. All three rails
  // (featured, popular, stats) follow the same pattern — only the
  // URL + cache key + setter differ.

  const loadFeatured = useCallback(async () => {
    const cached = readCache<FeaturedSkill[]>("featured");
    if (cached) {
      setFeatured(cached);
      setFeaturedLoading(false);
    } else {
      setFeaturedLoading(true);
    }
    try {
      const res = await apiRequest(`${apiBase}/api/skill-hub/featured`);
      if (!res.ok) {
        // Non-fatal — the rail is editorial polish, not a contract.
        if (!cached) setFeatured([]);
        return;
      }
      const data = (await res.json()) as FeaturedSkill[];
      setFeatured(data);
      writeCache("featured", data);
    } catch (e) {
      console.error(e);
      if (!cached) setFeatured([]);
    } finally {
      setFeaturedLoading(false);
    }
  }, [apiBase]);

  const loadStats = useCallback(
    async (sort: string) => {
      const cacheKey = `stats:${sort}`;
      const cached = readCache<RegistryStats>(cacheKey);
      if (cached) setRegistryStats(cached);
      try {
        const res = await apiRequest(
          `${apiBase}/api/skill-hub/registry/stats?sort=${encodeURIComponent(sort)}`,
        );
        if (!res.ok) {
          if (!cached) setRegistryStats(null);
          return;
        }
        const data = (await res.json()) as RegistryStats;
        setRegistryStats(data);
        writeCache(cacheKey, data);
      } catch (e) {
        console.error(e);
        if (!cached) setRegistryStats(null);
      }
    },
    [apiBase],
  );

  const loadPopular = useCallback(async () => {
    const cached = readCache<RegistrySkillSummary[]>("popular");
    if (cached) {
      setPopular(cached);
      setPopularLoading(false);
    } else {
      setPopularLoading(true);
    }
    try {
      const res = await apiRequest(
        `${apiBase}/api/skill-hub/registry/list?sort=installsCurrent&limit=8`,
      );
      if (!res.ok) {
        if (!cached) setPopular([]);
        return;
      }
      const data = (await res.json()) as RegistryListResponse;
      setPopular(data.items);
      writeCache("popular", data.items);
    } catch (e) {
      console.error(e);
      if (!cached) setPopular([]);
    } finally {
      setPopularLoading(false);
    }
  }, [apiBase]);

  // Initial loads.
  useEffect(() => {
    reloadInstalled();
  }, [reloadInstalled]);

  // Reset pagination on sort change / new query. Triggers *before*
  // the fetch effect below so the fetch picks up the cleared cursor
  // state. Effect runs only on the inputs that should bump us back
  // to page 1; ``pageIndex`` changes (Prev/Next clicks) intentionally
  // don't reset.
  useEffect(() => {
    if (tab !== "discover") return;
    resetPagination();
  }, [tab, registryQuery, registrySort, resetPagination]);

  // Refresh total-page count whenever the sort dimension changes
  // (totals differ across sorts because trending caps at ~100, etc).
  // Stats is independent of the grid fetch so it can resolve later;
  // the UI shows "Page N" until it arrives, then upgrades to "Page N of M".
  //
  // Don't clear stats unconditionally before loadStats — loadStats
  // reads the SWR cache and may have an instant hit. Clearing here
  // would flash "Page N of —" between sort change and cache read.
  useEffect(() => {
    if (tab !== "discover" || registryQuery.trim()) return;
    loadStats(registrySort);
  }, [tab, registrySort, registryQuery, loadStats]);

  // Fetch the page indicated by ``pageIndex``. Debounced when query
  // changes so we don't fire on every keystroke; immediate on sort
  // change or pagination click.
  useEffect(() => {
    if (tab !== "discover") return;
    const cursor = pageCursors[pageIndex] ?? null;
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    searchTimerRef.current = setTimeout(
      () => loadRegistry({ q: registryQuery, sort: registrySort, cursor }),
      250,
    );
    return () => {
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    };
  }, [tab, registryQuery, registrySort, pageIndex, pageCursors, loadRegistry]);

  // Featured rail: fetch once when the user first opens Discover,
  // then never again on tab toggles (backend caches it anyway).
  // Install handler updates installedAs state directly so badges flip
  // without a refetch.
  useEffect(() => {
    if (tab !== "discover") return;
    if (featuredLoadedRef.current) return;
    featuredLoadedRef.current = true;
    loadFeatured();
  }, [tab, loadFeatured]);

  // Popular rail: same one-shot lazy load pattern as featured.
  useEffect(() => {
    if (tab !== "discover") return;
    if (popularLoadedRef.current) return;
    popularLoadedRef.current = true;
    loadPopular();
  }, [tab, loadPopular]);

  // ── handlers ──────────────────────────────────────────────────────

  const handleInstall = async (slug: string) => {
    setInstallingSlug(slug);
    setInstallError(null);
    try {
      const res = await apiRequest(`${apiBase}/api/skill-hub/install/clawhub`, {
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
      // can carry this slug — registry grid, Featured, Popular. The
      // backend caches both Featured and Popular, but the optimistic
      // update avoids a visible round-trip.
      setRegistry((prev) =>
        prev.map((p) => (p.slug === slug ? { ...p, installedAs: slug } : p)),
      );
      setFeatured((prev) =>
        prev.map((p) => (p.slug === slug ? { ...p, installedAs: slug } : p)),
      );
      setPopular((prev) =>
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
      <div className="mx-auto w-full max-w-6xl flex-1 px-6 py-10">
        {/* Header */}
        <div className="mb-6 flex flex-col items-center gap-3 text-center">
          <div className="rounded-2xl bg-emerald-500/10 p-3 text-emerald-500">
            <Library className="h-8 w-8" />
          </div>
          <h1 className="text-3xl font-bold tracking-tight">Skill Hub</h1>
          <p className="max-w-2xl text-sm text-muted-foreground">
            Install skills from ClawHub or write your own. Agents
            automatically pick the right skill for each task.
          </p>
        </div>

        {/* Tabs + create */}
        <div className="mb-5 flex items-center justify-between gap-2">
          <div className="flex gap-1 rounded-full border bg-card p-1">
            <button
              type="button"
              onClick={() => setTab("discover")}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full px-4 py-1.5 text-sm font-medium transition-colors",
                tab === "discover"
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              <Compass className="h-4 w-4" />
              Discover
            </button>
            <button
              type="button"
              onClick={() => setTab("mine")}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full px-4 py-1.5 text-sm font-medium transition-colors",
                tab === "mine"
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              <Library className="h-4 w-4" />
              My Skills · {installed.length}
            </button>
          </div>
          <Link
            href="/skill-hub/new"
            className="inline-flex items-center gap-1.5 rounded-md border bg-card px-3 py-1.5 text-xs font-medium hover:bg-muted"
          >
            <Plus className="h-3.5 w-3.5" />
            Create new
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
            pageIndex={pageIndex}
            hasNext={!!nextCursorFromLast && !registryQuery.trim()}
            hasPrev={pageIndex > 0 && !registryQuery.trim()}
            onPrev={goToPrevPage}
            onNext={goToNextPage}
            stats={registryStats}
            featured={featured}
            featuredLoading={featuredLoading}
            popular={popular}
            popularLoading={popularLoading}
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
        title="Remove skill"
        description={
          deleteTarget
            ? `Remove "${deleteTarget}" from your installed skills? This deletes the files under ~/.xagent/skills/${deleteTarget}/.`
            : ""
        }
        confirmText="Remove"
        isLoading={!!deleting && deleting === deleteTarget}
      />
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────
// Discover tab
// ────────────────────────────────────────────────────────────────────

/** Roadmap surface: only the first is wired today. Adding a new
 * registry later means a new entry here plus a backend ``/registry/*``
 * implementation; the rest of DiscoverTab stays unchanged. */
const SOURCE_OPTIONS: SelectOption[] = [
  { value: "clawhub", label: "OpenClaw", description: "ClawHub public registry" },
  {
    value: "huggingface",
    label: "Hugging Face (coming soon)",
    description: "Browse Claude-format skills on HF Hub",
  },
  {
    value: "anthropic",
    label: "Anthropic Skills (coming soon)",
    description: "Official Anthropic skill catalog",
  },
];

/** Sort dimensions ClawHub honors today (audited 2026-05). The
 * ``value`` strings are passed verbatim to the registry's ``sort``
 * query param. */
const SORT_OPTIONS: SelectOption[] = [
  { value: "trending", label: "Trending" },
  { value: "newest", label: "Newest" },
  { value: "installsCurrent", label: "Most installed" },
  { value: "stars", label: "Most starred" },
  { value: "updated", label: "Recently updated" },
];

function DiscoverTab({
  registry, loading, error, query, setQuery, sort, setSort,
  pageIndex, hasNext, hasPrev, onPrev, onNext,
  stats,
  featured, featuredLoading,
  popular, popularLoading,
  installingSlug, installError, onInstall,
}: {
  registry: RegistrySkillSummary[];
  loading: boolean;
  error: string | null;
  query: string;
  setQuery: (s: string) => void;
  sort: string;
  setSort: (s: string) => void;
  pageIndex: number;             // 0-based; UI shows page = pageIndex + 1
  hasNext: boolean;
  hasPrev: boolean;
  onPrev: () => void;
  onNext: () => void;
  /** Total-page metadata. null while loading on first fetch. */
  stats: RegistryStats | null;
  featured: FeaturedSkill[];
  featuredLoading: boolean;
  popular: RegistrySkillSummary[];
  popularLoading: boolean;
  installingSlug: string | null;
  installError: { slug: string; msg: string } | null;
  onInstall: (slug: string) => void;
}) {
  // Derive total pages from total skills + the page size the grid
  // uses. Stats may be null (still loading) or truncated (we hit the
  // safety cap before exhausting cursors — show "≥M" then).
  const totalPages = stats ? Math.max(1, Math.ceil(stats.total / PAGE_SIZE)) : null;
  // Toast for the "coming soon" sources — keeps the dropdown
  // meaningful as a roadmap surface while making it obvious nothing
  // else is wired yet.
  const [comingSoon, setComingSoon] = useState<string | null>(null);
  const handleSourceChange = (v: string) => {
    if (v === "clawhub") return;
    const label = SOURCE_OPTIONS.find((o) => o.value === v)?.label || v;
    setComingSoon(`${label.replace(" (coming soon)", "")} isn't wired yet — for now only OpenClaw is supported.`);
    setTimeout(() => setComingSoon(null), 3500);
  };

  const inSearchMode = !!query.trim();

  return (
    <>
      {/* Featured rail — editorial, only in browse mode. Hides in
       * search so search-result space isn't cluttered. */}
      {!inSearchMode && (featuredLoading || featured.length > 0) && (
        <section className="mb-6">
          <div className="mb-2 flex items-center gap-1.5 text-sm font-semibold">
            <Sparkles className="h-4 w-4 text-amber-500" />
            <span>Featured by xagent</span>
          </div>
          {featuredLoading ? (
            <div className="flex justify-center py-6">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {featured.map((s) => {
                const installing = installingSlug === s.slug;
                const isInstalled = !!s.installedAs;
                return (
                  <div
                    key={s.slug}
                    className="flex flex-col gap-1.5 rounded-xl border border-amber-500/30 bg-amber-500/5 p-3 transition-all hover:border-amber-500/50 hover:shadow-sm"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-sm font-semibold">
                          {s.displayName || s.slug}
                        </div>
                        {s.ownerHandle && (
                          <div className="text-[10px] text-muted-foreground">
                            by {s.ownerHandle}
                          </div>
                        )}
                      </div>
                      {isInstalled ? (
                        <span className="inline-flex shrink-0 items-center gap-1 rounded-md border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-600">
                          <Check className="h-3 w-3" /> Installed
                        </span>
                      ) : (
                        <button
                          type="button"
                          disabled={installing}
                          onClick={() => onInstall(s.slug)}
                          className="inline-flex shrink-0 items-center gap-1 rounded-md bg-primary px-2 py-0.5 text-[10px] font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                        >
                          {installing ? (
                            <Loader2 className="h-3 w-3 animate-spin" />
                          ) : (
                            <Plus className="h-3 w-3" />
                          )}
                          Install
                        </button>
                      )}
                    </div>
                    <p className="text-[11px] italic text-amber-800/80 dark:text-amber-200/80 line-clamp-2">
                      {s.featuredReason}
                    </p>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      )}

      {/* Popular rail — ClawHub's installsCurrent angle. Distinct
       * from Trending (which weights recency), so the same skill
       * may appear in both Trending and Popular but the lists rarely
       * match exactly. Hidden in search mode (same reasoning as
       * Featured). */}
      {!inSearchMode && (popularLoading || popular.length > 0) && (
        <section className="mb-6">
          <div className="mb-2 flex items-center gap-1.5 text-sm font-semibold">
            <Flame className="h-4 w-4 text-rose-500" />
            <span>Popular this week</span>
          </div>
          {popularLoading ? (
            <div className="flex justify-center py-6">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
              {popular.map((s) => {
                const installing = installingSlug === s.slug;
                const isInstalled = !!s.installedAs;
                return (
                  <div
                    key={s.slug}
                    className="flex flex-col gap-1.5 rounded-xl border bg-card p-3 transition-all hover:border-primary/40 hover:shadow-sm"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-sm font-semibold">
                          {s.displayName || s.slug}
                        </div>
                        {s.installs != null && (
                          <div className="text-[10px] text-muted-foreground">
                            {formatInstalls(s.installs)} installs
                          </div>
                        )}
                      </div>
                      {isInstalled ? (
                        <span className="inline-flex shrink-0 items-center gap-1 rounded-md border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-600">
                          <Check className="h-3 w-3" /> Installed
                        </span>
                      ) : (
                        <button
                          type="button"
                          disabled={installing}
                          onClick={() => onInstall(s.slug)}
                          className="inline-flex shrink-0 items-center gap-1 rounded-md bg-primary px-2 py-0.5 text-[10px] font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                        >
                          {installing ? (
                            <Loader2 className="h-3 w-3 animate-spin" />
                          ) : (
                            <Plus className="h-3 w-3" />
                          )}
                          Install
                        </button>
                      )}
                    </div>
                    <p className="text-[11px] text-muted-foreground line-clamp-2">
                      {s.summary || "No description."}
                    </p>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      )}

      {/* "All" header — the section below this is the full sortable
       * browse grid. The label is what flips Discover from
       * "showcase" into "catalog" mode visually. */}
      {!inSearchMode && (
        <div className="mb-2 flex items-center gap-1.5 text-sm font-semibold">
          <Compass className="h-4 w-4 text-blue-500" />
          <span>Browse all</span>
        </div>
      )}

      <div className="mb-3 flex flex-col gap-3 sm:flex-row sm:items-stretch">
        <div className="sm:w-48">
          <Select
            value="clawhub"
            onValueChange={handleSourceChange}
            options={SOURCE_OPTIONS}
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
            placeholder="Search ClawHub…"
            className="h-10 w-full rounded-md border bg-background pl-10 pr-4 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
          />
        </div>
      </div>
      {comingSoon && (
        <div className="mb-4 rounded-md border border-amber-500/40 bg-amber-500/10 p-2.5 text-xs text-amber-700 dark:text-amber-300">
          {comingSoon}
        </div>
      )}

      {/* Count line.
       *   - search: "N results for X"
       *   - browse + stats unknown: "Page N · 60 shown"
       *   - browse + stats known:   "Page N of M · 1,234 skills total"
       *   - browse + truncated:     "Page N of ≥M · ≥1,234 skills"
       */}
      {!loading && !error && registry.length > 0 && (
        <div className="mb-4 text-xs text-muted-foreground">
          {inSearchMode ? (
            <>
              {registry.length} result{registry.length === 1 ? "" : "s"} for
              {" "}<span className="font-semibold text-foreground">&quot;{query.trim()}&quot;</span>
            </>
          ) : (
            <>
              Page <span className="font-semibold text-foreground">{pageIndex + 1}</span>
              {totalPages && (
                <>
                  {" "}of <span className="font-semibold text-foreground">
                    {stats?.truncated ? `≥${totalPages}` : totalPages}
                  </span>
                </>
              )}
              {stats && (
                <>
                  {" "}· {stats.truncated ? "≥" : ""}{stats.total.toLocaleString()} skills in {sort}
                </>
              )}
              {!stats && <> · {registry.length} shown</>}
            </>
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
          {query ? "No skills match your search." : "Registry is empty."}
        </div>
      ) : (
        <>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {registry.map((s) => {
            const installing = installingSlug === s.slug;
            const isInstalled = !!s.installedAs;
            const showError = installError?.slug === s.slug;
            return (
              <div
                key={s.slug}
                className="flex flex-col gap-2 rounded-xl border bg-card p-4 transition-all hover:border-primary/40 hover:shadow-sm"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-semibold leading-tight">
                      {s.displayName || s.slug}
                    </div>
                    {s.ownerHandle && (
                      <div className="text-[11px] text-muted-foreground">
                        by {s.ownerHandle}
                      </div>
                    )}
                  </div>
                  <ScanBadge status={s.scanStatus} />
                </div>
                <p className="text-xs text-muted-foreground line-clamp-3">
                  {s.summary || "No description."}
                </p>
                <div className="flex items-center justify-between gap-2 pt-1">
                  <div className="text-[11px] text-muted-foreground">
                    {s.version ? `v${s.version}` : ""}
                    {s.installs != null && (
                      <span className="ml-2">{formatInstalls(s.installs)} installs</span>
                    )}
                  </div>
                  {isInstalled ? (
                    <span className="inline-flex items-center gap-1 rounded-md border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-1 text-[11px] font-medium text-emerald-600">
                      <Check className="h-3 w-3" /> Installed
                    </span>
                  ) : (
                    <button
                      type="button"
                      disabled={installing}
                      onClick={() => onInstall(s.slug)}
                      className="inline-flex items-center gap-1 rounded-md bg-primary px-2.5 py-1 text-[11px] font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                    >
                      {installing ? (
                        <>
                          <Loader2 className="h-3 w-3 animate-spin" /> Installing…
                        </>
                      ) : (
                        <>
                          <Plus className="h-3 w-3" /> Install
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
              </div>
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
              Prev
            </button>
            <span className="text-xs text-muted-foreground">
              Page <span className="font-semibold text-foreground">{pageIndex + 1}</span>
              {totalPages && (
                <>
                  {" "}/ <span className="font-semibold text-foreground">
                    {stats?.truncated ? `≥${totalPages}` : totalPages}
                  </span>
                </>
              )}
            </span>
            <button
              type="button"
              onClick={onNext}
              disabled={!hasNext}
              className="inline-flex items-center gap-1 rounded-md border bg-card px-4 py-2 text-xs font-medium hover:bg-muted disabled:opacity-30"
            >
              Next
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

const INSTALLED_SORT_OPTIONS: SelectOption[] = [
  { value: "source", label: "Group by source", description: "User → Built-in → External" },
  { value: "name", label: "Name (A-Z)", description: "Flat alphabetical" },
];

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
            placeholder="Search installed skills…"
            className="h-10 w-full rounded-md border bg-background pl-10 pr-4 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
          />
        </div>
      </div>
      {!loading && !error && totalCount > 0 && (
        <div className="mb-4 text-[11px] text-muted-foreground">
          {query
            ? `${installed.length} of ${totalCount} match`
            : `${totalCount} skill${totalCount === 1 ? "" : "s"} installed`}
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
            "No skills match your filter."
          ) : (
            <>
              No installed skills yet. Browse the{" "}
              <span className="font-medium text-foreground">Discover</span> tab or
              create one from scratch.
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
                    {badge.label}
                  </span>
                </div>
                <Link
                  href={`/skill-hub/${encodeURIComponent(s.name)}`}
                  className="text-xs text-muted-foreground line-clamp-3"
                >
                  {s.description || "No description."}
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
                    Remove
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

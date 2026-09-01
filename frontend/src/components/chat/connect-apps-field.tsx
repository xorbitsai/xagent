"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/auth-context";
import { useI18n } from "@/contexts/i18n-context";
import { useMcpApps, type McpApp } from "@/contexts/mcp-apps-context";
import { openBuiltinOAuthPopup, openMcpOAuthPopup } from "@/lib/oauth-connect";
import { toast } from "@/components/ui/sonner";
import { getBrandingFromEnv } from "@/lib/branding";
import { apiRequest } from "@/lib/api-wrapper";
import { cn, getApiUrl } from "@/lib/utils";
import type { Interaction } from "@/contexts/app-context-chat";
import { capitalize } from "@/lib/tool-category-labels";
import { findMatchingMcpApp } from "@/lib/mcp-lookup";
import { ApiKeyConnectDialog, CONNECT_TIMEOUT_MS, handleIconLoadError } from "./api-key-connect-dialog";

// The 13 builtin-OAuth providers (see src/xagent/web/builtin_mcp_registry.py's
// get_builtin_oauth_provider_rows) - brand names, left untranslated in both
// locales like every other connector name in the app. Falling back to
// capitalize() below still runs for any provider missing here, but gets the
// casing wrong for a multi-cap brand name (e.g. "Github" instead of
// "GitHub"), so every provider in the registry should have an entry here.
export const PROVIDER_DISPLAY_NAMES: Record<string, string> = {
  google: "Google",
  linkedin: "LinkedIn",
  microsoft: "Microsoft",
  hubspot: "HubSpot",
  meta: "Meta",
  slack: "Slack",
  zoom: "Zoom",
  intercom: "Intercom",
  github: "GitHub",
  linear: "Linear",
  jira: "Jira",
  salesforce: "Salesforce",
  deputy: "Deputy",
};

// connectingKeys/connectingKeysRef below hold both of these kinds of key in
// the same Set - an OAuth provider name (shared across every app in that
// provider's group) and an individual app id (mcp_oauth/keyless rows, which
// don't share state with anything). Prefixing keeps a catalog app whose id
// ever happened to collide with a provider name (e.g. an app literally
// id'd "google") from being reported as "connecting" merely because that
// provider's OAuth group is mid-login, or vice versa.
function providerConnectingKey(provider: string): string {
  return `provider:${provider}`;
}

function appConnectingKey(appId: string): string {
  return `app:${appId}`;
}

interface OAuthProviderGroup {
  provider: string;
  /** Every requested app under this provider, not just the one this row
   * renders - handleConnect needs the whole group to decide between a bare
   * batch login and an app-scoped one (see its own comment). */
  apps: McpApp[];
}

type ConnectAppsRow =
  | { kind: "oauth"; app: McpApp; group: OAuthProviderGroup }
  | { kind: "mcp_oauth"; app: McpApp }
  // api_key (e.g. AWS) - a small in-card dialog collects the key(s), see
  // ApiKeyConnectDialog; keyless (e.g. Chrome) - a one-click connect
  // straight to the endpoint, no dialog needed; or unconnectable/anything
  // else this card has no flow for, shown with just a hint and no action.
  // Either way the app isn't silently missing from a card that claims to
  // list everything
  // the persona needs.
  | { kind: "manual"; app: McpApp };

/**
 * Resolves an interaction's requested app names against the connector
 * catalog, in the interaction's own order (deduped by app id), and
 * classifies each into the row it needs: one shared OAuth-provider group
 * per builtin_oauth app (so a bare login can still cover every app under
 * that provider even though each now renders its own row - see
 * ConnectAppsField's "one row per app, not per provider" note below), its
 * own row for a remote-MCP OAuth app (mcp_oauth, e.g. Granola), or a
 * manual-setup row for anything else. A name that doesn't resolve to a
 * catalog entry at all is silently dropped - nothing this card could do
 * about it either way.
 */
export function resolveRows(appNames: string[] | undefined, allApps: McpApp[]): ConnectAppsRow[] {
  const wanted = appNames || [];
  if (wanted.length === 0) return [];

  // findMatchingMcpApp (lib/mcp-lookup.ts) matches by name OR id, tolerating
  // a hyphen-for-space variant either way - the same lenient matching
  // tests/templates/test_manager.py's test_builtin_template_connections_
  // resolve_to_a_registered_mcp_app already assumes this card does. A plain
  // lowercase/trim name-only lookup silently drops any template that names
  // its connection by app_id (e.g. "facebook-pages") instead of display name.
  const seen = new Set<string>();
  const resolved: McpApp[] = [];
  for (const name of wanted) {
    const app = findMatchingMcpApp(allApps, name);
    if (!app || seen.has(app.id)) continue;
    seen.add(app.id);
    resolved.push(app);
  }

  // Tolerate a missing auth_type (older catalog rows / test fixtures) the
  // same way this card always has: an app carrying a provider is treated as
  // builtin-OAuth unless the backend explicitly classified it as something
  // else. Checking auth_type explicitly here (not just "has a provider")
  // matters once a custom catalog app can carry a provider_name alongside a
  // non-oauth transport - classify_app_auth would call that api_key/keyless/
  // mcp_oauth, and without this check it would still render as an "oauth"
  // row and never be connectable from this card.
  const isOAuthApp = (app: McpApp) =>
    !!app.provider && (app.auth_type === undefined || app.auth_type === "builtin_oauth");

  const groupsByProvider = new Map<string, OAuthProviderGroup>();
  for (const app of resolved) {
    if (!isOAuthApp(app)) continue;
    let group = groupsByProvider.get(app.provider);
    if (!group) {
      group = { provider: app.provider, apps: [] };
      groupsByProvider.set(app.provider, group);
    }
    group.apps.push(app);
  }

  return resolved.map((app): ConnectAppsRow => {
    if (app.auth_type === "mcp_oauth") return { kind: "mcp_oauth", app };
    if (isOAuthApp(app)) return { kind: "oauth", app, group: groupsByProvider.get(app.provider)! };
    return { kind: "manual", app };
  });
}

function RowIcon({
  url,
  fallbackName,
  connected,
}: {
  url: string | undefined;
  fallbackName: string;
  connected: boolean;
}) {
  return (
    <div
      className={cn(
        "flex h-8 w-8 flex-shrink-0 items-center justify-center overflow-hidden rounded-[9px] border",
        connected
          ? "border-green-200 bg-white dark:border-green-800 dark:bg-background"
          : "border-border bg-muted"
      )}
    >
      {url ? (
        <img
          src={url}
          alt=""
          className="h-full w-full object-contain p-1"
          onError={handleIconLoadError(fallbackName)}
        />
      ) : (
        <span className="text-[11.5px] font-bold text-muted-foreground">
          {fallbackName.slice(0, 1)}
        </span>
      )}
    </div>
  );
}

function ConnectedBadge({ label }: { label: string }) {
  return (
    <span className="flex flex-shrink-0 items-center gap-1.5 text-xs font-semibold text-green-700 dark:text-green-400">
      {/* Same glyph connect-mcp-dialog.tsx uses for "connected/verified/
          selected" everywhere in Settings -> Tools, so a connected state
          reads the same whether the user sees it there or in this card. */}
      <CheckCircle2 className="h-3.5 w-3.5" />
      {label}
    </span>
  );
}

export function ConnectAppsField({
  interaction,
  onSkip,
  onContinue,
}: {
  interaction: Interaction;
  /** Optional (and omitted) when this card isn't the live, active pause -
   * see onContinue's own doc comment. Skip sends an ordinary chat message
   * just like Continue does, with the exact same resume/interrupt side
   * effects on whatever task state exists when it's received - so a stale
   * historical card must not offer it either. */
  onSkip?: () => Promise<void> | void;
  /** Called once every requested app is connected, in place of onSkip -
   * distinct so the message it sends can say "connected" rather than
   * "I'll do this later" (see clarification-form.tsx's
   * handleContinueConnectApps). Optional so an older/mixed-list caller
   * that doesn't pass it just keeps the allConnectedNote-only footer this
   * card had before, rather than crashing. */
  onContinue?: () => Promise<void> | void;
}) {
  const { apps, refresh, isLoading, error } = useMcpApps();
  const { token } = useAuth();
  const { t } = useI18n();
  const branding = getBrandingFromEnv();
  // A Set, not a single key - each row connects independently, so a click
  // on one must not clear or overwrite another's in-flight state. Keyed by
  // provider for an "oauth" row (shared with its sibling rows under the
  // same provider, since one bare login can finish several at once) or by
  // app id for an "mcp_oauth" row (no such sharing exists there).
  const [connectingKeys, setConnectingKeys] = useState<Set<string>>(new Set());
  const [skipped, setSkipped] = useState(false);
  const [continued, setContinued] = useState(false);
  const [keyConnectApp, setKeyConnectApp] = useState<McpApp | null>(null);
  const isMountedRef = useRef(true);
  // Synchronous shadow of connectingKeys, same reason connect-mcp-dialog.tsx
  // keeps loadingAppsRef alongside loadingApps (#1330 there): setState-based
  // `disabled={isConnecting}` lags a commit cycle behind two clicks landing
  // in the same tick, and for the mcp_oauth (DCR) path a double-click
  // reaching openMcpOAuthPopup twice really does register two OAuth clients
  // at the third-party authorization server - a side effect nothing here can
  // withdraw, unlike the bare-OAuth-popup or keyless paths that merely open a
  // redundant popup or hit an idempotent backend.
  const connectingKeysRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  const rows = useMemo(() => resolveRows(interaction.apps, apps), [interaction.apps, apps]);
  // Recomputed on every render (not memoized against rows, which doesn't
  // change when just an app's is_connected flips) - the footer's "not
  // connected yet" copy plus a Skip button, alongside a card whose every row
  // already shows Connected, would read as an outright contradiction.
  const allConnected = rows.length > 0 && rows.every((row) => !!row.app.is_connected);

  if (rows.length === 0) {
    // ClarificationForm's Collapsible card (title bar + chevron) is already
    // showing above this by the time useMcpApps() is still fetching or has
    // failed - returning null unconditionally here left it sitting over a
    // blank void with nothing telling the user why.
    if (isLoading) {
      return (
        <p className="text-xs text-muted-foreground">
          {t("chatPage.clarification.connectApps.loading")}
        </p>
      );
    }
    if (error) {
      return (
        <div className="flex items-center gap-3">
          <p className="flex-1 text-xs text-destructive">{error}</p>
          <button
            type="button"
            className="flex-shrink-0 rounded-md border border-border px-2 py-1 text-xs font-medium text-foreground hover:bg-muted"
            onClick={() => void refresh()}
          >
            {t("chatPage.clarification.connectApps.retry")}
          </button>
        </div>
      );
    }
    // The catalog loaded fine, but none of the requested app names resolved
    // against it - the backend names apps without filtering by
    // is_visible_in_connector, while the frontend catalog fetch strong-hides
    // any app with that flag off (e.g. a hidden-rollout gate), so a real
    // pause can legitimately name an app this card can never render a row
    // for. Retry (in case the catalog just hadn't loaded that app yet) plus
    // Skip, matching the still-genuinely-paused task's own escape hatch -
    // returning null here left a live pause with no visible action at all.
    return (
      <div className="flex items-center gap-3">
        <p className="flex-1 text-xs text-muted-foreground">
          {skipped
            ? t("chatPage.clarification.connectApps.skippedNote")
            : t("chatPage.clarification.connectApps.noneMatched")}
        </p>
        {!skipped && (
          <>
            <button
              type="button"
              className="flex-shrink-0 rounded-md border border-border px-2 py-1 text-xs font-medium text-foreground hover:bg-muted"
              onClick={() => void refresh()}
            >
              {t("chatPage.clarification.connectApps.retry")}
            </button>
            {onSkip && (
              <button
                type="button"
                className="flex-shrink-0 rounded-md px-2 py-1 text-xs font-medium text-muted-foreground hover:bg-muted hover:text-foreground"
                onClick={async () => {
                  setSkipped(true);
                  try {
                    await onSkip();
                  } catch {
                    if (isMountedRef.current) {
                      setSkipped(false);
                    }
                  }
                }}
              >
                {t("chatPage.clarification.connectApps.skip")}
              </button>
            )}
          </>
        )}
      </div>
    );
  }

  const withConnectingKey = async (key: string, run: () => Promise<void>) => {
    if (connectingKeysRef.current.has(key)) return;
    connectingKeysRef.current.add(key);
    setConnectingKeys((prev) => new Set(prev).add(key));
    try {
      await run();
    } finally {
      connectingKeysRef.current.delete(key);
      if (isMountedRef.current) {
        setConnectingKeys((prev) => {
          const next = new Set(prev);
          next.delete(key);
          return next;
        });
        await refresh();
      }
    }
  };

  const handleConnectOAuth = (group: OAuthProviderGroup) =>
    withConnectingKey(providerConnectingKey(group.provider), async () => {
      const unconnectedApps = group.apps.filter((app) => !app.is_connected);
      if (unconnectedApps.length === 0) return;
      // More than one app still needs connecting: run the bare, app_id-less
      // login so one consent screen covers every app under this provider
      // (see openBuiltinOAuthPopup's doc comment) instead of only the
      // clicked row's own app. Once that narrows it down to exactly one
      // remaining app - either because the group only ever had one, or
      // because the bare login couldn't cover an app requiring its own
      // app-scoped grant (e.g. Facebook Pages under Meta) - fall back to
      // that app's own id so a second click actually finishes the group
      // instead of repeating the same bare login forever.
      const appId = unconnectedApps.length === 1 ? unconnectedApps[0].id : undefined;
      // Names whichever the popup call actually targets: the group's
      // provider for a real bare batch login (more than one app still
      // unconnected - the toast should describe what was actually
      // attempted), or the one remaining app's own name once appId above
      // narrows it to an app-scoped grant - matching the row's own button,
      // which already reads "Continue with {app.name}" for this same case
      // (see continueWith below), not the provider's name.
      const failureTarget = appId
        ? unconnectedApps[0].name
        : PROVIDER_DISPLAY_NAMES[group.provider] || capitalize(group.provider);

      // Both this and openMcpOAuthPopup below are documented to resolve
      // rather than reject on every path they control - but window.open()
      // inside openBuiltinOAuthPopup's Promise executor can still throw
      // synchronously in some sandboxed/policy-restricted browser contexts,
      // which the Promise constructor turns into a rejection. Catching it
      // here (matching handleConnectKeyless's own try/catch around its raw
      // apiRequest call, the one call in this file already documented to be
      // able to throw) means that edge case surfaces as the same error toast
      // instead of an unhandled promise rejection with connectingKeys stuck
      // showing "Connecting...".
      try {
        const result = await openBuiltinOAuthPopup({ provider: group.provider, appId, token });
        if (!isMountedRef.current) return;
        if (!result.success) {
          toast.error(
            t(
              result.popupBlocked
                ? "chatPage.clarification.connectApps.popupBlocked"
                : "chatPage.clarification.connectApps.connectFailed",
              { provider: failureTarget }
            )
          );
        }
      } catch {
        if (isMountedRef.current) {
          toast.error(
            t("chatPage.clarification.connectApps.connectFailed", { provider: failureTarget })
          );
        }
      }
    });

  const handleConnectMcpOAuth = (app: McpApp) =>
    withConnectingKey(appConnectingKey(app.id), async () => {
      // Same reasoning as handleConnectOAuth above.
      try {
        const result = await openMcpOAuthPopup({ appId: app.id });
        if (!isMountedRef.current) return;
        if (!result.connected) {
          toast.error(
            result.message ||
              t(
                result.popupBlocked
                  ? "chatPage.clarification.connectApps.popupBlocked"
                  : "chatPage.clarification.connectApps.connectFailed",
                { provider: app.name }
              )
          );
        }
      } catch {
        if (isMountedRef.current) {
          toast.error(t("chatPage.clarification.connectApps.connectFailed", { provider: app.name }));
        }
      }
    });

  // Keyless catalog app (e.g. Chrome): classify_app_auth only assigns
  // "keyless" when required_env is empty, so it can never take the
  // hasKeyForm branch below - it needs its own one-click path straight to
  // the connect endpoint, mirroring connect-mcp-dialog.tsx's
  // submitKeylessConnect (is_active sent explicitly so re-connecting a
  // dormant association reactivates it).
  const handleConnectKeyless = (app: McpApp) =>
    withConnectingKey(appConnectingKey(app.id), async () => {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/mcp/apps/${encodeURIComponent(app.id)}/connect`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ is_active: true }),
          signal: AbortSignal.timeout(CONNECT_TIMEOUT_MS),
        });
        if (!isMountedRef.current) return;
        if (!response.ok) {
          toast.error(t("chatPage.clarification.connectApps.connectFailed", { provider: app.name }));
        }
      } catch {
        if (isMountedRef.current) {
          toast.error(t("chatPage.clarification.connectApps.connectFailed", { provider: app.name }));
        }
      }
    });

  return (
    <div>
      <p className="text-xs text-muted-foreground">
        {t("chatPage.clarification.connectApps.subtitle")}
      </p>

      {/* Rows already resolved from a still-good, if now stale, catalog
          fetch, so this only ever hides the row list on a *fresh* failure -
          a later refresh (e.g. after connecting something) failing doesn't
          reach the early return above (rows.length is already non-zero) and
          would otherwise fail completely silently. */}
      {error && (
        <div className="mt-3 flex items-center gap-3 rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2">
          <p className="flex-1 text-xs text-destructive">{error}</p>
          <button
            type="button"
            className="flex-shrink-0 rounded-md border border-border px-2 py-1 text-xs font-medium text-foreground hover:bg-muted"
            onClick={() => void refresh()}
          >
            {t("chatPage.clarification.connectApps.retry")}
          </button>
        </div>
      )}

      {/* One row per app, not per provider - a "Google" row that also
          covers Calendar used to fold both into one merged row; that read
          as the two being one connector rather than two, so each app gets
          its own row/icon/name now. Clicking either one's button still
          finishes both in a single sign-in via handleConnectOAuth's bare
          login above; only the display stopped merging, not the mechanism. */}
      <div className="mt-3 overflow-hidden rounded-xl border">
        {rows.map((row, index) => {
          const { app } = row;
          const rowClassName = cn(
            "flex items-center gap-3 p-3",
            index > 0 && "border-t",
            app.is_connected && "bg-green-50 dark:bg-green-950/20"
          );

          if (row.kind === "manual") {
            const hasKeyForm = (app.launch_config?.required_env?.length ?? 0) > 0;
            const isKeyless = app.auth_type === "keyless";
            const isConnectingKeyless = connectingKeys.has(appConnectingKey(app.id));
            return (
              <div key={app.id} className={rowClassName}>
                <RowIcon url={app.icon} fallbackName={app.name} connected={!!app.is_connected} />
                <div className="min-w-0 flex-1">
                  <div className="text-[13px] font-semibold text-foreground">{app.name}</div>
                  {!app.is_connected && !isKeyless && (
                    <div className="truncate text-[11.5px] text-muted-foreground">
                      {t("chatPage.clarification.connectApps.manualHint")}
                    </div>
                  )}
                </div>
                {app.is_connected ? (
                  <ConnectedBadge label={t("chatPage.clarification.connectApps.connected")} />
                ) : hasKeyForm ? (
                  <Button
                    size="sm"
                    onClick={() => setKeyConnectApp(app)}
                    className="flex-shrink-0 rounded-[9px]"
                  >
                    {t("tools.mcp.dialog.connect")}
                  </Button>
                ) : isKeyless ? (
                  <Button
                    size="sm"
                    disabled={isConnectingKeyless}
                    onClick={() => handleConnectKeyless(app)}
                    className="flex-shrink-0 rounded-[9px]"
                  >
                    {isConnectingKeyless
                      ? t("chatPage.clarification.connectApps.connecting")
                      : t("tools.mcp.dialog.connect")}
                  </Button>
                ) : null}
              </div>
            );
          }

          const isConnecting =
            row.kind === "oauth"
              ? connectingKeys.has(providerConnectingKey(row.group.provider))
              : connectingKeys.has(appConnectingKey(app.id));
          const onConnect =
            row.kind === "oauth"
              ? () => handleConnectOAuth(row.group)
              : () => handleConnectMcpOAuth(app);

          return (
            <div key={app.id} className={rowClassName}>
              <RowIcon url={app.icon} fallbackName={app.name} connected={!!app.is_connected} />
              <div className="min-w-0 flex-1">
                <div className="text-[13px] font-semibold text-foreground">{app.name}</div>
              </div>
              {app.is_connected ? (
                <ConnectedBadge label={t("chatPage.clarification.connectApps.connected")} />
              ) : (
                <Button
                  size="sm"
                  disabled={isConnecting}
                  onClick={onConnect}
                  className="flex-shrink-0 rounded-[9px]"
                >
                  {isConnecting
                    ? t("chatPage.clarification.connectApps.connecting")
                    : t("chatPage.clarification.connectApps.continueWith", { provider: app.name })}
                </Button>
              )}
            </div>
          );
        })}

        <div className="flex items-center gap-3 border-t bg-muted/40 px-3 py-2.5">
          <span className="flex-1 text-[11.5px] text-muted-foreground">
            {skipped
              ? t("chatPage.clarification.connectApps.skippedNote")
              : allConnected
                ? t("chatPage.clarification.connectApps.allConnectedNote")
                : t("chatPage.clarification.connectApps.privacyNote", { appName: branding.appName })}
          </span>
          {/* Swaps for a Continue button once every row is Connected (see
              onContinue's doc comment) - a Skip button next to an "I'll do
              this later" note would contradict a card with nothing left to
              do, and without any button at all a card seeded onto a task
              that's genuinely paused waiting for this connection (not just
              the Hire-flow seed message, which was never actually waiting)
              would have no way to tell the task to resume. Gated on !skipped
              too - Skip already sent its own acknowledgement message, and
              nothing here disables the per-row Connect buttons, so without
              this a user who skips and then connects the remaining apps
              anyway would see Continue reappear and could send a second,
              contradictory message on top of the one Skip already sent. */}
          {allConnected ? (
            !skipped &&
            onContinue &&
            !continued && (
              <button
                type="button"
                className="flex-shrink-0 rounded-md bg-primary px-2.5 py-1 text-xs font-medium text-primary-foreground hover:bg-primary/90"
                onClick={async () => {
                  setContinued(true);
                  try {
                    await onContinue();
                  } catch {
                    // A failed send must roll the optimistic "continued"
                    // state back, or the button disappears while the
                    // acknowledgement never actually went through, leaving
                    // the user with no way to retry.
                    if (isMountedRef.current) {
                      setContinued(false);
                    }
                  }
                }}
              >
                {t("chatPage.clarification.connectApps.continue")}
              </button>
            )
          ) : (
            !skipped &&
            onSkip && (
              <button
                type="button"
                className="flex-shrink-0 rounded-md px-2 py-1 text-xs font-medium text-muted-foreground hover:bg-muted hover:text-foreground"
                onClick={async () => {
                  setSkipped(true);
                  try {
                    await onSkip();
                  } catch {
                    // Same rationale as the Continue button's rollback above.
                    if (isMountedRef.current) {
                      setSkipped(false);
                    }
                  }
                }}
              >
                {t("chatPage.clarification.connectApps.skip")}
              </button>
            )
          )}
        </div>
      </div>

      <ApiKeyConnectDialog
        app={keyConnectApp}
        onOpenChange={(open) => {
          if (!open) setKeyConnectApp(null);
        }}
        onConnected={refresh}
      />
    </div>
  );
}

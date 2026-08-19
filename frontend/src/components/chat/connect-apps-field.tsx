"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/auth-context";
import { useI18n } from "@/contexts/i18n-context";
import { useMcpApps, type McpApp } from "@/contexts/mcp-apps-context";
import { openBuiltinOAuthPopup } from "@/lib/oauth-connect";
import { toast } from "@/components/ui/sonner";
import type { Interaction } from "@/contexts/app-context-chat";
import { capitalize } from "@/lib/tool-category-labels";

// The 8 builtin-OAuth providers (see src/xagent/web/builtin_mcp_registry.py's
// get_builtin_oauth_provider_rows) - brand names, left untranslated in both
// locales like every other connector name in the app.
const PROVIDER_DISPLAY_NAMES: Record<string, string> = {
  google: "Google",
  linkedin: "LinkedIn",
  microsoft: "Microsoft",
  hubspot: "HubSpot",
  meta: "Meta",
  slack: "Slack",
  zoom: "Zoom",
  intercom: "Intercom",
};

interface ConnectAppsGroup {
  provider: string;
  apps: McpApp[];
  allConnected: boolean;
}

/**
 * Groups an interaction's requested app names by OAuth provider (one
 * sign-in covers every app under it, e.g. Google covers Gmail + Calendar)
 * and drops any name that doesn't resolve to a connected-apps-catalog entry
 * with a provider - apps with no provider (Granola, Notion, AWS, ...) use a
 * different connect flow than the builtin-OAuth popup this field offers, so
 * they're intentionally left out of this card rather than half-supported.
 */
function groupByProvider(appNames: string[] | undefined, allApps: McpApp[]): ConnectAppsGroup[] {
  const wanted = new Set((appNames || []).map((name) => name.toLowerCase().trim()));
  if (wanted.size === 0) return [];

  const byProvider = new Map<string, McpApp[]>();
  for (const app of allApps) {
    if (!app.provider) continue;
    if (!wanted.has(app.name.toLowerCase().trim())) continue;
    const list = byProvider.get(app.provider) || [];
    list.push(app);
    byProvider.set(app.provider, list);
  }

  return Array.from(byProvider.entries()).map(([provider, apps]) => ({
    provider,
    apps,
    allConnected: apps.every((app) => app.is_connected),
  }));
}

export function ConnectAppsField({
  interaction,
  onSkip,
}: {
  interaction: Interaction;
  onSkip: () => void;
}) {
  const { apps, refresh } = useMcpApps();
  const { token } = useAuth();
  const { t } = useI18n();
  // A Set, not a single provider - each group connects independently, so a
  // click on one must not clear or overwrite another's in-flight state.
  const [connectingProviders, setConnectingProviders] = useState<Set<string>>(new Set());
  const [skipped, setSkipped] = useState(false);
  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  const groups = useMemo(() => groupByProvider(interaction.apps, apps), [interaction.apps, apps]);

  if (groups.length === 0) return null;

  const handleConnect = async (group: ConnectAppsGroup) => {
    const primaryApp = group.apps[0];
    if (!primaryApp) return;

    setConnectingProviders((prev) => new Set(prev).add(group.provider));
    try {
      const result = await openBuiltinOAuthPopup({
        provider: group.provider,
        appId: primaryApp.id,
        token,
      });
      if (!isMountedRef.current) return;
      if (!result.success) {
        toast.error(
          t(
            result.popupBlocked
              ? "chatPage.clarification.connectApps.popupBlocked"
              : "chatPage.clarification.connectApps.connectFailed",
            { provider: PROVIDER_DISPLAY_NAMES[group.provider] || capitalize(group.provider) }
          )
        );
      }
    } finally {
      if (isMountedRef.current) {
        setConnectingProviders((prev) => {
          const next = new Set(prev);
          next.delete(group.provider);
          return next;
        });
        await refresh();
      }
    }
  };

  return (
    <div className="w-full rounded-lg border bg-card p-4">
      {/* No heading here: the enclosing ClarificationForm's Collapsible
          header already shows interaction.label as its title (see
          isConnectAppsOnly in clarification-form.tsx) - repeating it here
          would just duplicate the same text twice on screen. */}
      <p className="mb-3 text-xs text-muted-foreground">
        {t("chatPage.clarification.connectApps.subtitle")}
      </p>

      <div className="flex flex-col gap-2">
        {groups.map((group) => {
          const displayName = PROVIDER_DISPLAY_NAMES[group.provider] || capitalize(group.provider);
          const isConnecting = connectingProviders.has(group.provider);
          return (
            <div
              key={group.provider}
              className="flex items-center justify-between gap-3 rounded-md border p-3"
            >
              <div className="flex min-w-0 items-center gap-2.5">
                {group.apps[0]?.icon ? (
                  <img src={group.apps[0].icon} alt="" className="h-6 w-6 flex-shrink-0 rounded" />
                ) : null}
                <div className="min-w-0">
                  <div className="text-sm font-medium text-foreground">{displayName}</div>
                  <div className="truncate text-xs text-muted-foreground">
                    {group.apps.map((app) => app.name).join(" · ")}
                  </div>
                </div>
              </div>
              <Button
                size="sm"
                variant={group.allConnected ? "outline" : "default"}
                disabled={group.allConnected || isConnecting}
                onClick={() => handleConnect(group)}
              >
                {group.allConnected
                  ? t("chatPage.clarification.connectApps.connected")
                  : isConnecting
                    ? t("chatPage.clarification.connectApps.connecting")
                    : t("chatPage.clarification.connectApps.connect")}
              </Button>
            </div>
          );
        })}
      </div>

      {!skipped && (
        <button
          type="button"
          className="mt-3 text-xs font-medium text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
          onClick={() => {
            setSkipped(true);
            onSkip();
          }}
        >
          {t("chatPage.clarification.connectApps.skip")}
        </button>
      )}
    </div>
  );
}

"use client";

import React, { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { useI18n } from "@/contexts/i18n-context";
import { apiRequest } from "@/lib/api-wrapper";
import { getApiUrl } from "@/lib/utils";
import { toast } from "@/components/ui/sonner";
import type { McpApp } from "@/contexts/mcp-apps-context";

const CONNECT_TIMEOUT_MS = 30_000;
// A same-key retention token: submitting it back unchanged tells the backend
// (_merge_masked_env, src/xagent/web/api/mcp.py) to keep the stored secret
// rather than overwrite it. Matches connect-mcp-dialog.tsx's own constant of
// the same name/value - duplicated per this file's established narrow-slice
// pattern rather than imported, since neither file exports it.
const MASKED_SECRET_VALUE = "********";

// Same load-failure fallback as connect-apps-field.tsx's RowIcon: a real
// icon URL can still 404, and this dialog reuses the exact same app.icon
// value that row already has to guard for the same reason.
function iconFallbackUrl(name: string): string {
  return `https://ui-avatars.com/api/?name=${encodeURIComponent(name)}&background=random&color=fff&size=128`;
}

// FastAPI's RequestValidationError handler (src/xagent/web/app.py) returns
// `detail` as an array of {msg, loc, ...} objects for a 422 on this endpoint,
// not the plain string every other failure mode here returns it as - render
// that shape's first message instead of stringifying the whole array.
function readableErrorDetail(detail: unknown): string | undefined {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length > 0) {
    const first = detail[0];
    if (typeof first === "string") return first;
    if (first && typeof first === "object" && typeof first.msg === "string") return first.msg;
  }
  return undefined;
}

/**
 * A small, self-contained "connect with your own API key" dialog for one
 * catalog app. Posts to the same `/api/mcp/apps/{id}/connect` endpoint as
 * connect-mcp-dialog.tsx's key-entry form, minus that form's team-sharing,
 * select-mode, and shared/platform-key options - a caller from a chat
 * message only ever needs "the current user connects with their own key,"
 * the same narrowing openBuiltinOAuthPopup/openMcpOAuthPopup in
 * lib/oauth-connect.ts already apply to their own slice of that dialog.
 */
export function ApiKeyConnectDialog({
  app,
  onOpenChange,
  onConnected,
}: {
  /** null closes the dialog. */
  app: McpApp | null;
  onOpenChange: (open: boolean) => void;
  onConnected: () => void;
}) {
  const { t } = useI18n();
  const [values, setValues] = useState<Record<string, string>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const requiredEnv = app?.launch_config?.required_env || [];

  // Re-seed on every app change (including re-opening for the same app after
  // a prior close), not just once on mount - a bare `useState({})` initial
  // value only ever runs for this component instance's first render, so a
  // later app would inherit the previous app's stale values otherwise.
  useEffect(() => {
    if (!app) return;
    const required = app.launch_config?.required_env || [];
    const configured = new Set(app.configured_env_keys || []);
    const initial: Record<string, string> = {};
    // Pre-fill masked per key that's already configured, so submitting
    // without retyping it preserves it instead of silently wiping it -
    // _merge_masked_env only restores a key whose incoming value is exactly
    // this sentinel; any other value (including "") is written as the new
    // secret, dropping whatever was stored. This has to be per key, not
    // app.user_env_configured's single all-required-keys-set flag: an app
    // with more than one required key (e.g. AWS's 3) can be configured
    // key-by-key over time, and treating "not fully configured yet" as
    // "blank every field" would wipe the ones that *are* already set the
    // moment the user only means to add the missing one.
    required.forEach((key) => {
      initial[key] = configured.has(key) ? MASKED_SECRET_VALUE : "";
    });
    setValues(initial);
  }, [app]);

  const handleOpenChange = (open: boolean) => {
    if (open || isSubmitting) return;
    setValues({});
    onOpenChange(false);
  };

  const handleSubmit = async () => {
    if (!app) return;
    setIsSubmitting(true);
    try {
      // Send every required key explicitly, even ones left untouched, rather
      // than only the ones the user actually typed into - matching
      // connect-mcp-dialog.tsx's openKeyConnect, whose POST body always
      // carries the full required_env set (each entry either the masked
      // sentinel seeded above, the user's edit, or "" for a key that was
      // never configured and is still untouched).
      const env = Object.fromEntries(requiredEnv.map((key) => [key, values[key] ?? ""]));
      const response = await apiRequest(`${getApiUrl()}/api/mcp/apps/${app.id}/connect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ env, env_source: "own" }),
        signal: AbortSignal.timeout(CONNECT_TIMEOUT_MS),
      });
      if (response.ok) {
        toast.success(t("tools.mcp.dialog.connectSuccess", { name: app.name }));
        setValues({});
        onConnected();
        onOpenChange(false);
      } else {
        const error = await response.json().catch(() => ({}));
        toast.error(readableErrorDetail(error.detail) || t("tools.mcp.alerts.saveFailed"));
      }
    } catch (error) {
      const timedOut = error instanceof DOMException && error.name === "TimeoutError";
      toast.error(
        timedOut ? t("tools.mcp.alerts.connectTimedOut") : t("tools.mcp.alerts.saveFailed")
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Dialog open={!!app} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {app?.icon && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={app.icon}
                alt=""
                className="h-5 w-5"
                onError={(event) => {
                  event.currentTarget.onerror = null;
                  event.currentTarget.src = iconFallbackUrl(app.name);
                }}
              />
            )}
            {t("tools.mcp.dialog.connect")} {app?.name}
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-4 py-2">
          {requiredEnv.map((key) => (
            <div key={key} className="space-y-1.5">
              <Label htmlFor={`connect-apps-key-${key}`}>{key}</Label>
              <Input
                id={`connect-apps-key-${key}`}
                type="password"
                autoComplete="off"
                value={values[key] || ""}
                onChange={(event) =>
                  setValues((prev) => ({ ...prev, [key]: event.target.value }))
                }
              />
            </div>
          ))}
        </div>
        <div className="flex justify-end gap-2">
          <Button variant="outline" onClick={() => handleOpenChange(false)} disabled={isSubmitting}>
            {t("common.cancel")}
          </Button>
          <Button onClick={handleSubmit} disabled={isSubmitting}>
            {isSubmitting ? t("tools.mcp.dialog.connecting") : t("tools.mcp.dialog.connect")}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

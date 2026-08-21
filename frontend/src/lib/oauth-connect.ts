import { apiRequest } from "@/lib/api-wrapper";
import { getApiUrl } from "@/lib/utils";
import { MCP_OAUTH_POPUP_WINDOW_NAME, parseMcpOAuthErrorMessage } from "@/lib/mcp-utils";

/**
 * Opens the builtin-OAuth login popup for a connector app and resolves once
 * the popup reports success (via postMessage) or is closed/times out without
 * one. Intentionally a small, self-contained duplicate of the equivalent
 * flow in components/mcp/connect-mcp-dialog.tsx's `handleConnectApp` rather
 * than a shared extraction - that file is large and stateful (selection,
 * sharing, multiple auth types), and this call site only ever needs the
 * "open popup, wait for one postMessage" case for a builtin_oauth app.
 */
export interface OpenBuiltinOAuthPopupResult {
  /** True if the popup reported an "oauth-success" postMessage before
   * closing/timing out. False on popup-blocked, close-without-success, or
   * timeout - the caller should re-check connection state via useMcpApps()
   * either way, since a same-tick success message can race this promise on
   * some browsers. */
  success: boolean;
  /** True when window.open() itself returned null (the browser's popup
   * blocker intervened) - lets the caller show a more actionable message
   * than the generic connect-failed one. */
  popupBlocked?: boolean;
}

const POPUP_WIDTH = 600;
const POPUP_HEIGHT = 700;
const MAX_WAIT_MS = 5 * 60 * 1000;
const POLL_INTERVAL_MS = 500;

export function openBuiltinOAuthPopup(options: {
  provider: string;
  /** Omit to run the backend's bare, app_id-less login instead: it grants
   * every visible OAuth app under this provider in one consent screen
   * (see auth.py's app_id-less branch), rather than just one app. Not
   * every app can be covered this way (`APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT`
   * skips itself during that batch and needs its own app-scoped call), so
   * this is an opt-in per call, not a default. */
  appId?: string;
  token: string | null | undefined;
}): Promise<OpenBuiltinOAuthPopupResult> {
  const { provider, appId, token } = options;

  return new Promise((resolve) => {
    const left = window.screenX + (window.outerWidth - POPUP_WIDTH) / 2;
    const top = window.screenY + (window.outerHeight - POPUP_HEIGHT) / 2;
    const appIdParam = appId ? `&app_id=${appId}` : "";
    const authUrl = `${getApiUrl()}/api/auth/${provider}/login?token=${token || ""}${appIdParam}&redirect=${encodeURIComponent(window.location.href)}`;
    const popup = window.open(
      authUrl,
      `${provider}OAuth`,
      `width=${POPUP_WIDTH},height=${POPUP_HEIGHT},left=${left},top=${top},scrollbars=yes`
    );

    if (!popup) {
      resolve({ success: false, popupBlocked: true });
      return;
    }

    let settled = false;
    const finish = (success: boolean) => {
      if (settled) return;
      settled = true;
      window.clearInterval(pollTimer);
      window.removeEventListener("message", handleMessage);
      resolve({ success });
    };

    const handleMessage = (event: MessageEvent) => {
      if (event.data?.type === "oauth-success") {
        finish(true);
      }
    };
    window.addEventListener("message", handleMessage);

    const startedAt = Date.now();
    const pollTimer = window.setInterval(() => {
      const expired = Date.now() - startedAt >= MAX_WAIT_MS;
      if (!popup.closed && !expired) return;
      finish(false);
    }, POLL_INTERVAL_MS);
  });
}

export interface OpenMcpOAuthPopupResult {
  /** Whether the app is actually connected after the popup closed. Unlike
   * the builtin flow there is no postMessage channel here (the popup's
   * opener is severed, matching connect-mcp-dialog.tsx's handling of the
   * same remote-MCP OAuth type) - a closed popup can mean success OR a
   * cancelled/denied/failed authorization, so this is only known by asking
   * the backend which one actually happened once the popup closes. */
  connected: boolean;
  popupBlocked?: boolean;
  /** The backend's own reported reason the connect POST failed (rate limit,
   * DCR failure, etc.), via the same parseMcpOAuthErrorMessage helper
   * connect-mcp-dialog.tsx uses - so a caller can show it instead of a
   * generic failure message. Absent when the failure happened before or
   * after that POST (network error, still-unconnected on recheck). */
  message?: string;
}

/**
 * Opens the remote-MCP OAuth ("mcp_oauth" auth_type, e.g. Granola) connect
 * popup for one app and resolves once the popup closes (or times out). A
 * small, self-contained duplicate of connect-mcp-dialog.tsx's
 * `handleConnectMcpOAuthApp` for the same reason `openBuiltinOAuthPopup`
 * above duplicates that file's builtin_oauth branch: this call site only
 * ever needs the single-app "open popup, wait for it to close, recheck"
 * case, not that dialog's full selection/sharing/multi-auth-type state.
 */
export async function openMcpOAuthPopup(options: {
  appId: string;
}): Promise<OpenMcpOAuthPopupResult> {
  const { appId } = options;
  const left = window.screenX + (window.outerWidth - POPUP_WIDTH) / 2;
  const top = window.screenY + (window.outerHeight - POPUP_HEIGHT) / 2;
  // Opened on "about:blank" and navigated only after the connect POST
  // resolves (rather than opening straight to the authorization URL, as
  // openBuiltinOAuthPopup does) because this flow has to mint the OAuth
  // client via Dynamic Client Registration server-side first - there is no
  // fixed login URL to open synchronously the way the builtin flow has.
  const popup = window.open(
    "about:blank",
    MCP_OAUTH_POPUP_WINDOW_NAME,
    `width=${POPUP_WIDTH},height=${POPUP_HEIGHT},left=${left},top=${top},scrollbars=yes`
  );
  if (!popup) {
    return { connected: false, popupBlocked: true };
  }
  popup.opener = null;

  try {
    const response = await apiRequest(`${getApiUrl()}/api/mcp/apps/${appId}/oauth/connect`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ redirect_after: "/tools?tab=mcp" }),
    });
    if (!response.ok) {
      popup.close();
      const message = await parseMcpOAuthErrorMessage(response, "");
      return { connected: false, message: message || undefined };
    }
    const data = (await response.json()) as { authorization_url?: string };
    if (!data.authorization_url) {
      popup.close();
      return { connected: false };
    }
    popup.location.href = data.authorization_url;
  } catch {
    popup.close();
    return { connected: false };
  }

  await new Promise<void>((resolve) => {
    const startedAt = Date.now();
    const pollTimer = window.setInterval(() => {
      const expired = Date.now() - startedAt >= MAX_WAIT_MS;
      if (!popup.closed && !expired) return;
      window.clearInterval(pollTimer);
      resolve();
    }, POLL_INTERVAL_MS);
  });

  try {
    const response = await apiRequest(`${getApiUrl()}/api/mcp/apps?location=remote`);
    if (!response.ok) return { connected: false };
    const apps = (await response.json()) as Array<{ id: string; is_connected?: boolean }>;
    return { connected: apps.some((app) => app.id === appId && app.is_connected === true) };
  } catch {
    return { connected: false };
  }
}

import { getApiUrl } from "@/lib/utils";

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
  appId: string;
  token: string | null | undefined;
}): Promise<OpenBuiltinOAuthPopupResult> {
  const { provider, appId, token } = options;

  return new Promise((resolve) => {
    const left = window.screenX + (window.outerWidth - POPUP_WIDTH) / 2;
    const top = window.screenY + (window.outerHeight - POPUP_HEIGHT) / 2;
    const authUrl = `${getApiUrl()}/api/auth/${provider}/login?token=${token || ""}&app_id=${appId}&redirect=${encodeURIComponent(window.location.href)}`;
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

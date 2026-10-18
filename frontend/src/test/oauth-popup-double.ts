import type { openOAuthAuthorizationPopup } from "@/lib/oauth-launch"

/** Fixed-origin message observer for dialog unit tests; real launch metadata
 * and cross-origin admission are exercised in oauth-launch/connect tests. */
export function isOAuthSuccessMessageDouble(event: MessageEvent, popup: Window | null, provider: string) {
  return event.source === popup && event.origin === "http://api.local"
    && event.data?.type === "oauth-success" && event.data?.provider === provider
}

/** Dialog-unit observer; the real HTTP/ticket contract is tested separately. */
export function openOAuthPopupDouble(options: Parameters<typeof openOAuthAuthorizationPopup>[0]) {
  const url = new URL(`http://api.local/api/auth/${options.provider}/login`)
  if (options.appId !== undefined) url.searchParams.set("app_id", options.appId)
  if (options.account !== undefined) url.searchParams.set("account", options.account)
  return window.open(url.href, options.name, options.features)
}

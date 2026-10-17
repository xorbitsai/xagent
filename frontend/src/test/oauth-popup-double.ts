import type { openOAuthAuthorizationPopup } from "@/lib/oauth-launch"

/** Dialog-unit observer; the real HTTP/ticket contract is tested separately. */
export function openOAuthPopupDouble(options: Parameters<typeof openOAuthAuthorizationPopup>[0]) {
  const url = new URL(`http://api.local/api/auth/${options.provider}/login`)
  if (options.appId !== undefined) url.searchParams.set("app_id", options.appId)
  if (options.account !== undefined) url.searchParams.set("account", options.account)
  return window.open(url.href, options.name, options.features)
}

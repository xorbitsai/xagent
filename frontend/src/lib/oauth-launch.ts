import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import { toast } from "@/components/ui/sonner"

const callbackOrigins = new WeakMap<Window, string>()

/** Trust only the configured callback origin returned by the authenticated POST. */
export function isOAuthSuccessMessage(event: MessageEvent, popup: Window | null, provider: string): boolean {
  return popup !== null && event.source === popup
    && event.origin === callbackOrigins.get(popup)
    && event.data?.type === "oauth-success" && event.data?.provider === provider
}

/** Open on the click stack; send the login credential only in the POST header. */
export function openOAuthAuthorizationPopup(options: {
  provider: string
  appId?: string
  account?: string
  name: string
  features: string
  onError?: () => void
}): Window | null {
  const popup = window.open("about:blank", options.name, options.features)
  if (!popup) return null
  // Named popups can reuse a WindowProxy from an older attempt. Do not trust
  // its old callback origin while the new authenticated POST is pending.
  callbackOrigins.delete(popup)

  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), 30_000)
  void (async () => {
    try {
      const response = await apiRequest(
        `${getApiUrl()}/api/auth/${encodeURIComponent(options.provider)}/login`,
        {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json", Accept: "application/json" },
          body: JSON.stringify({
            app_id: options.appId,
            account: options.account,
            redirect: window.location.origin,
          }),
          signal: controller.signal,
        },
      )
      if (!response.ok) throw new Error("Authorization could not be started")
      const data: unknown = await response.json()
      const raw = data && typeof data === "object" && "authorization_url" in data
        ? data.authorization_url : null
      if (typeof raw !== "string") throw new Error("Invalid authorization response")
      const callbackOrigin = data && typeof data === "object" && "callback_origin" in data ? data.callback_origin : null
      if (typeof callbackOrigin !== "string") throw new Error("Missing callback origin")
      const callback = new URL(callbackOrigin)
      if (!["https:", "http:"].includes(callback.protocol) || callback.username || callback.password
        || callback.pathname !== "/" || callback.search || callback.hash) {
        throw new Error("Invalid callback origin")
      }
      const endpoint = new URL(`${getApiUrl()}/api/auth/${encodeURIComponent(options.provider)}/login`, window.location.href)
      if (!raw.startsWith(`/api/auth/${encodeURIComponent(options.provider)}/login?`)) {
        throw new Error("Invalid authorization destination")
      }
      const url = new URL(`${getApiUrl()}${raw}`, window.location.href)
      if (url.origin !== endpoint.origin || url.username || url.password
        || url.pathname !== endpoint.pathname
        || !url.searchParams.get("ticket") || url.searchParams.has("token")
        || url.searchParams.has("access_token")) {
        throw new Error("Invalid authorization destination")
      }
      callbackOrigins.set(popup, callback.origin)
      if (!popup.closed) popup.location.replace(url.href)
    } catch {
      callbackOrigins.delete(popup)
      // Never log a response body or URL: OAuth state can contain sensitive data.
      if (!popup.closed) popup.close()
      if (options.onError) options.onError()
      else toast.error("Authorization could not be started. Please try again.")
    } finally {
      window.clearTimeout(timeout)
    }
  })()
  return popup
}

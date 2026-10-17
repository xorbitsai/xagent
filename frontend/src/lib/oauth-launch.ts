import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"

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
            redirect: window.location.href,
          }),
          signal: controller.signal,
        },
      )
      if (!response.ok) throw new Error("Authorization could not be started")
      const data: unknown = await response.json()
      const raw = data && typeof data === "object" && "authorization_url" in data
        ? data.authorization_url : null
      if (typeof raw !== "string") throw new Error("Invalid authorization response")
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
      if (!popup.closed) popup.location.replace(url.href)
    } catch {
      // Never log a response body or URL: OAuth state can contain sensitive data.
      if (!popup.closed) popup.close()
      options.onError?.()
    } finally {
      window.clearTimeout(timeout)
    }
  })()
  return popup
}

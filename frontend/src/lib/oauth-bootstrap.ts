/** The allowlist comes only from same-origin deployment configuration. */
export function validateRegionalBootstrap(raw: string, origins: readonly string[]): string {
  const url = new URL(raw)
  if (!origins.includes(url.origin) || !["https:", "http:"].includes(url.protocol)
    || url.username || url.password || url.hash || raw.includes("#")
    || !/^\/api\/oauth\/launch\/[a-z0-9-]+$/.test(url.pathname)
    || [...url.searchParams.keys()].length !== 1
    || !url.searchParams.get("ticket") || [...url.searchParams.keys()][0] !== "ticket") {
    throw new Error("Invalid regional authorization destination")
  }
  return url.href
}

export async function regionalBootstrapURL(data: { authorization_url?: string; launch_mode?: string }): Promise<string | null> {
  if (data.launch_mode === undefined) return null
  if (data.launch_mode !== "regional_bootstrap" || typeof data.authorization_url !== "string") {
    throw new Error("Unknown authorization launch mode")
  }
  const response = await fetch("/api/deployment-config", { credentials: "same-origin", cache: "no-store", redirect: "error" })
  if (!response.ok) throw new Error("Authorization configuration unavailable")
  const config = await response.json()
  if (!Array.isArray(config.oauth_authorization_origins)
    || !config.oauth_authorization_origins.every((origin: unknown) => typeof origin === "string")) {
    throw new Error("Authorization configuration unavailable")
  }
  return validateRegionalBootstrap(data.authorization_url, config.oauth_authorization_origins)
}

export async function mcpAuthorizationURL(data: { authorization_url?: string; launch_mode?: string }): Promise<string> {
  const regional = await regionalBootstrapURL(data)
  if (regional) return regional
  if (!data.authorization_url) throw new Error("Missing authorization destination")
  return data.authorization_url
}

export interface McpLookupReference {
  name?: unknown
  id?: unknown
  app_id?: unknown
}

const normalizeMcpLookupValue = (value: unknown): string => {
  return String(value ?? "").trim().toLowerCase()
}

const getMcpLookupKeys = (...values: unknown[]): Set<string> => {
  const keys = new Set<string>()

  values.forEach((value) => {
    const normalized = normalizeMcpLookupValue(value)
    if (!normalized) return

    keys.add(normalized)
    keys.add(normalized.replace(/\s+/g, "-"))
  })

  return keys
}

const hasSharedMcpLookupKey = (left: Set<string>, right: Set<string>): boolean => {
  for (const key of left) {
    if (right.has(key)) return true
  }
  return false
}

export const mcpNameMatches = (left: unknown, right: unknown): boolean => {
  return hasSharedMcpLookupKey(getMcpLookupKeys(left), getMcpLookupKeys(right))
}

export const findMatchingMcpServer = <T extends McpLookupReference>(
  servers: T[],
  serverName: string
): T | undefined => {
  const targetKeys = getMcpLookupKeys(serverName)
  return servers.find((server) =>
    hasSharedMcpLookupKey(getMcpLookupKeys(server.name, server.app_id), targetKeys)
  )
}

export const findMatchingMcpApp = <T extends McpLookupReference>(
  apps: T[],
  serverName: string
): T | undefined => {
  const targetKeys = getMcpLookupKeys(serverName)
  return apps.find((app) =>
    hasSharedMcpLookupKey(getMcpLookupKeys(app.name, app.id), targetKeys)
  )
}

// Turns a selected connector (identified by whatever string the UI is
// currently holding for it -- historically a mix of display names and ids)
// into the string that should follow "mcp:" when building a tool-category
// selector for the backend.
//
// The backend has two contradictory conventions for what a catalog app's
// MCPServer row is actually named:
//   - _ensure_catalog_app_server / _ensure_catalog_mcp_oauth_server
//     (src/xagent/web/api/mcp.py) name the row after the app_id.
//   - _ensure_user_mcp_server (src/xagent/web/api/auth.py), used for
//     builtin_oauth apps, names the row after the display name.
// Guessing a single fallback (id, or name) can only ever satisfy one
// convention -- e.g. "chrome-devtools" (app_id) vs. "Chrome" (name) diverge
// one way, "facebook" (app_id) vs. "Facebook Pages" (name) diverge the
// other way, and a hard-coded id-first or name-first fallback silently
// breaks whichever app uses the other convention.
//
// So this resolves the *real* server row instead of guessing at its name:
// find the catalog app the selector refers to, then look up an actual
// MCPServer row by that app's id, then by its name -- whichever convention
// created the row, this finds it and returns its authoritative real name.
// Only when no server row exists at all (not yet connected) does this fall
// back to the app's id, then to the raw selector unchanged.
export const resolveMcpToolSelector = <T extends McpLookupReference>(
  server: string,
  mcpServers: T[],
  officialApps: T[]
): string => {
  const connectedApp = findMatchingMcpApp(officialApps, server)

  const connectedServer =
    findMatchingMcpServer(mcpServers, server) ||
    (typeof connectedApp?.id === "string" && connectedApp.id
      ? findMatchingMcpServer(mcpServers, connectedApp.id)
      : undefined) ||
    (typeof connectedApp?.name === "string" && connectedApp.name
      ? findMatchingMcpServer(mcpServers, connectedApp.name)
      : undefined)

  if (typeof connectedServer?.name === "string" && connectedServer.name) {
    return connectedServer.name
  }
  if (typeof connectedApp?.id === "string" && connectedApp.id) {
    return connectedApp.id
  }
  return server
}

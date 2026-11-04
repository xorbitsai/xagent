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
// selector for the backend. The backend matches that selector by name only,
// with no id fallback (src/xagent/core/tools/adapters/vibe/mcp_tools.py),
// so an unresolved or wrongly-guessed selector silently loads zero tools
// for that connector -- no error, just an agent that can't use it.
//
// Why this can't be "prefer id" or "prefer name": the backend has two
// different conventions for what a catalog app's MCPServer row is actually
// named -- _ensure_catalog_app_server / _ensure_catalog_mcp_oauth_server
// (src/xagent/web/api/mcp.py) name shared-server rows after the app_id,
// while _ensure_user_mcp_server (src/xagent/web/api/auth.py), used for
// builtin_oauth apps, names the row after the display name. A single
// hard-coded fallback direction can only ever match one convention, and
// guessing wrong doesn't just fail to help -- it can *overwrite* a selector
// that was already correct (e.g. because the app just isn't visible in
// mcpServers yet: a stale fetch, or a row this viewer genuinely can't
// query) with a wrong one, corrupting a previously-working selection.
//
// So: resolve the actual connected MCPServer row and use its real name --
// find the catalog app the selector refers to, then look up a server row
// by that app's id, then by its name, covering both conventions regardless
// of which one applies to this app. This falls through to returning the
// incoming selector completely unchanged (never a guessed id or name) in
// two cases: no server row can be found at all, or a matched row's `name`
// is empty/whitespace -- narrow (every known write path already requires a
// non-blank name), but a row *was* found there, it just had nothing usable
// to return. Either way, if the incoming selector was already correct, it
// stays correct; if the app genuinely isn't connected, there's nothing
// better to substitute.
//
// A caller that gets the unresolved input back has no signal of *why* --
// only that this selector may load zero tools once persisted. Logged here
// rather than surfaced as a user-facing warning: deciding the right UX for
// that (a toast? which flows?) is a separate, larger call than this helper
// should make on its own.
export const resolveMcpToolSelector = <
  S extends McpLookupReference,
  A extends McpLookupReference
>(
  server: string,
  mcpServers: S[],
  officialApps: A[]
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

  if (typeof connectedServer?.name === "string" && connectedServer.name.trim()) {
    return connectedServer.name
  }
  if (server) {
    console.warn(
      `resolveMcpToolSelector: could not resolve "${server}" to a connected MCP server row; persisting it unchanged, which may load zero tools for this connector.`
    )
  }
  return server
}

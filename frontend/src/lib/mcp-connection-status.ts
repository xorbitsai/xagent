/**
 * Persisted OAuth grant state that GET /api/mcp/servers reports per builtin
 * OAuth row (`MCPServerResponse.connection_status` in
 * src/xagent/web/api/mcp.py). "needs_reconnect" means the grant's token was
 * cleared or the only grant is one the app-scoped policy rejects; null or an
 * absent field means no persisted grant, which callers keep rendering as
 * before. The row stays listed in every case, so its presence alone does not
 * mean the connector can run.
 */
export type McpConnectionStatus = "connected" | "needs_reconnect"

export function needsReconnect(
  entry: { connection_status?: McpConnectionStatus | null } | null | undefined,
): boolean {
  return entry?.connection_status === "needs_reconnect"
}

/**
 * Whether a selected connector counts as connected, given its server row (if
 * any) and its catalog entry (if any). A row that needs a reconnect wins over
 * the catalog: it is the grant the runtime resolves.
 */
export function isMcpSelectionConnected(
  server: { connection_status?: McpConnectionStatus | null } | null | undefined,
  app: { is_connected?: boolean } | null | undefined,
): boolean {
  return !needsReconnect(server) && Boolean(server || app?.is_connected)
}

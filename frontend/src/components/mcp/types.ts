// Shared shape for a connector catalog entry / connected MCP app. Kept in one
// place so the connector dialog and the settings dialog can't drift apart.
export interface AppIntegration {
  id: string
  name: string
  description: string
  icon: string
  is_connected?: boolean
  users?: string
  provider?: string
  category?: string
  is_local?: boolean
  server_id?: number
  transport?: string
  connected_account?: string
  is_custom?: boolean
  // Canonical connect classification derived on the catalog entry by the
  // backend (mcp_apps.classify_app_auth). Read this instead of re-deriving
  // from provider/required_env so the dialogs can't drift from the backend.
  auth_type?: "builtin_oauth" | "api_key" | "keyless" | "mcp_oauth" | "unconnectable"
  launch_config?: {
    command?: string
    args?: string[]
    required_env?: string[]
  }
  // Key-based apps: a shared key (injected by a deployment hook) already
  // covers required_env, so the user can connect without their own.
  shared_env_available?: boolean
  // Key-based apps: the platform-global key on the server row covers required_env.
  platform_env_available?: boolean
  // Key-based apps: this user has set their own per-user key.
  user_env_configured?: boolean
  // Key-based apps: the user's current env-source pick, if any.
  env_source?: "own" | "shared" | "platform" | null
  // Whether this entry may be selected into an agent, decided by the backend
  // (list_mcp_apps._local_mcp_can_attach) rather than re-derived here from
  // is_connected/is_custom/auth_type. It answers "the runtime will see this
  // connector and its credentials will plausibly resolve", which depends on
  // backend-only facts — grant state, team links, and whether the deployment
  // installed an OAuth token resolver hook (#1347).
  can_attach?: boolean
  // Whether starting the per-server MCP OAuth flow is meaningful for this
  // entry. False for catalog entries (they connect through
  // /apps/{id}/oauth/connect, dispatched on auth_type), for a connector with
  // no active personal association (the per-server route would 404), and for
  // a deployment whose tokens arrive through the resolver hook, where no
  // interactive consent exists at all.
  can_authorize?: boolean
  // Whether this connector reached the viewer through a team link rather than
  // a connection of their own (#1387). A third state, not a synonym for
  // is_connected: the connector is usable without the viewer owning it, so the
  // card labels it as the team's and — for the shapes whose credentials are
  // per-user — still offers the Connect route. Absent entirely on deployments
  // without a team-visibility hook (standalone payloads stay pre-#1387);
  // present-false means teams exist and nothing is shared with this viewer.
  // Read absence as false, which Boolean() already does.
  is_team_shared?: boolean
  // Team-sharing status (from POST /api/connectors/status), merged in after list load.
  shared?: boolean
  is_owner?: boolean
  needs_config?: boolean
}

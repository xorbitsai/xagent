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
  // Key-based apps: which of launch_config.required_env this user already
  // has a stored value for - a per-key breakdown of user_env_configured's
  // all-or-nothing flag (see _app_configured_env_keys in
  // src/xagent/web/api/mcp.py), so a reconnect for an app with more than
  // one required key doesn't have to blank a key it can't tell is already
  // set (submitting blank for a key clears it - see connect_mcp_app).
  configured_env_keys?: string[]
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
  // Whether this viewer's Configure route would resolve, decided by the
  // backend (_local_mcp_can_configure in src/xagent/web/api/mcp.py for
  // local entries; the connection state for a catalog entry, whose
  // Configure equivalent is
  // "manage my key" or "re-run OAuth" and only exists once connected) rather
  // than re-derived here from is_connected. A connector whose tokens arrive
  // through a deployment-installed resolver hook is never "connected" for
  // its own creator -- no personal grant row is ever written -- so the
  // connected gate hid the edit route from the one person entitled to it.
  can_configure?: boolean
  // Team-sharing status (from POST /api/connectors/status), merged in after list load.
  shared?: boolean
  is_owner?: boolean
  needs_config?: boolean
}

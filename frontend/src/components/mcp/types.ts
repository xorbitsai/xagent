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
  // Team-sharing status (from POST /api/connectors/status), merged in after list load.
  shared?: boolean
  is_owner?: boolean
  needs_config?: boolean
}

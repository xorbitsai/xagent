import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { OfficialMcpSettingsDialog } from "./official-mcp-settings-dialog"
import type { AppIntegration } from "./types"

vi.mock("@/lib/api-wrapper", () => ({ apiRequest: vi.fn() }))

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return { ...actual, getApiUrl: () => "http://api.local" }
})

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ token: "token", inTeam: true }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, params?: Record<string, string>) =>
      params?.host ? `${key}:${params.host}` : key,
  }),
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: { error: vi.fn(), success: vi.fn() },
}))

vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ open, children }: { open: boolean; children: React.ReactNode }) =>
    open ? <div role="dialog">{children}</div> : null,
  DialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogTitle: ({ children }: { children: React.ReactNode }) => <h1>{children}</h1>,
}))

const DESTINATION = "tools.mcp.memberOAuth.destination"

const teamMcpOAuth: Partial<AppIntegration> = {
  id: "records",
  name: "records",
  is_custom: true,
  auth_type: "mcp_oauth",
  shared: true,
  is_owner: false,
  server_id: 7,
  url_host: "mcp.example.com",
}

const CASES: Array<{
  id: string
  app: Partial<AppIntegration>
  connected: boolean
  shown: boolean
}> = [
  { id: "team_mcp_oauth_unconnected", app: teamMcpOAuth, connected: false, shown: true },
  { id: "team_mcp_oauth_connected", app: teamMcpOAuth, connected: true, shown: false },
  {
    id: "owned_custom",
    app: { ...teamMcpOAuth, is_owner: true },
    connected: false,
    shown: false,
  },
  {
    id: "team_mcp_oauth_not_open",
    app: { ...teamMcpOAuth, auth_type: undefined },
    connected: false,
    shown: false,
  },
  {
    id: "catalog_app",
    app: { ...teamMcpOAuth, is_custom: false },
    connected: false,
    shown: false,
  },
]

describe("OfficialMcpSettingsDialog member sign-in destination", () => {
  afterEach(cleanup)

  it.each(CASES)("$id", ({ app, connected, shown }) => {
    render(
      <OfficialMcpSettingsDialog
        open
        onOpenChange={vi.fn()}
        app={{
          description: "d",
          icon: "",
          is_connected: connected,
          transport: "streamable_http",
          ...app,
        } as AppIntegration}
        isGloballyConnected={connected}
      />,
    )

    const hint = screen.queryByText(`${DESTINATION}:mcp.example.com`)
    if (shown) {
      expect(hint).not.toBeNull()
      expect(screen.getByText("tools.mcp.dialog.connect")).not.toBeNull()
    } else {
      expect(hint).toBeNull()
    }
  })
})

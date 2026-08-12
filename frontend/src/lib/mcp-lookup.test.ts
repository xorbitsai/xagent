import { describe, expect, it } from "vitest"

import {
  findMatchingMcpApp,
  findMatchingMcpServer,
  mcpNameMatches,
  resolveMcpToolSelector,
} from "./mcp-lookup"

describe("MCP lookup helpers", () => {
  it("matches saved app ids to connected server names", () => {
    const server = findMatchingMcpServer(
      [{ name: "Outlook", app_id: "outlook" }],
      "outlook"
    )

    expect(server?.name).toBe("Outlook")
  })

  it("matches slug ids to display names", () => {
    const app = findMatchingMcpApp(
      [{ id: "google-drive", name: "Google Drive" }],
      "Google Drive"
    )

    expect(app?.id).toBe("google-drive")
    expect(mcpNameMatches("google-drive", "Google Drive")).toBe(true)
  })
})

describe("resolveMcpToolSelector", () => {
  // The backend has two contradictory conventions for a catalog app's real
  // MCPServer row name: app_id (_ensure_catalog_app_server /
  // _ensure_catalog_mcp_oauth_server) for shared-server apps, or the
  // display name (_ensure_user_mcp_server) for builtin_oauth apps. A single
  // hard-coded fallback can only satisfy one -- these two apps diverge in
  // opposite directions and must both resolve correctly.
  const chromeApp = { id: "chrome-devtools", name: "Chrome" }
  const facebookApp = { id: "facebook", name: "Facebook Pages" }

  it("resolves a connected app-id-named server row (chrome-devtools convention)", () => {
    const mcpServers = [{ name: "chrome-devtools" }]
    expect(resolveMcpToolSelector("Chrome", mcpServers, [chromeApp])).toBe(
      "chrome-devtools"
    )
  })

  it("resolves a connected name-named server row (facebook builtin_oauth convention)", () => {
    const mcpServers = [{ name: "Facebook Pages" }]
    expect(resolveMcpToolSelector("facebook", mcpServers, [facebookApp])).toBe(
      "Facebook Pages"
    )
  })

  it("falls back to the app id when the app is known but no server row is connected yet", () => {
    expect(resolveMcpToolSelector("Chrome", [], [chromeApp])).toBe(
      "chrome-devtools"
    )
  })

  it("falls back to the raw selector when nothing matches at all", () => {
    expect(resolveMcpToolSelector("some-custom-server", [], [])).toBe(
      "some-custom-server"
    )
  })
})

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
  //
  // Chrome is stdio transport: _enrich_oauth_server_info short-circuits on
  // non-oauth transports, so its real connected row never carries an
  // app_id -- just { name: "chrome-devtools" }, as used below.
  const chromeApp = { id: "chrome-devtools", name: "Chrome" }
  // Facebook is oauth transport: GET /api/mcp/servers enriches oauth rows
  // with app_id via _enrich_oauth_server_info, so its real row shape is
  // { name: "Facebook Pages", app_id: "facebook" } -- carrying both.
  const facebookApp = { id: "facebook", name: "Facebook Pages" }

  it("resolves a connected app-id-named server row (chrome-devtools convention, no app_id on the row)", () => {
    const mcpServers = [{ name: "chrome-devtools" }]
    expect(resolveMcpToolSelector("Chrome", mcpServers, [chromeApp])).toBe(
      "chrome-devtools"
    )
  })

  it("resolves a real oauth row carrying app_id (facebook, via step 1's direct app_id match)", () => {
    const mcpServers = [{ name: "Facebook Pages", app_id: "facebook" }]
    expect(resolveMcpToolSelector("facebook", mcpServers, [facebookApp])).toBe(
      "Facebook Pages"
    )
  })

  it("resolves a name-named row with no id at all (isolates the builtin_oauth name-fallback step)", () => {
    // A row shaped like _ensure_user_mcp_server's convention with no app_id
    // present at all -- unlike the case above, this can only resolve via
    // resolveMcpToolSelector's app-name fallback step, not a direct id
    // match, so it exercises that step independently.
    const mcpServers = [{ name: "Facebook Pages" }]
    expect(resolveMcpToolSelector("facebook", mcpServers, [facebookApp])).toBe(
      "Facebook Pages"
    )
  })

  it("picks the one matching row out of several connected servers", () => {
    const mcpServers = [
      { name: "Zoom" },
      { name: "chrome-devtools" },
      { name: "Facebook Pages", app_id: "facebook" },
    ]
    expect(
      resolveMcpToolSelector("Chrome", mcpServers, [chromeApp, facebookApp])
    ).toBe("chrome-devtools")
    expect(
      resolveMcpToolSelector("facebook", mcpServers, [chromeApp, facebookApp])
    ).toBe("Facebook Pages")
  })

  it("round-trips an already-resolved real server name unchanged (loadAgent -> re-save)", () => {
    // loadAgent seeds selectedMcpServers straight from a saved
    // tool_categories entry, which after any save already holds the
    // resolved real name -- the very next save must not perturb it.
    const mcpServers = [{ name: "chrome-devtools" }]
    expect(
      resolveMcpToolSelector("chrome-devtools", mcpServers, [chromeApp])
    ).toBe("chrome-devtools")
  })

  it("preserves the raw selector unchanged -- never a guessed id or name -- when the app is known but no server row is connected yet", () => {
    // Locks in the invariant: whatever the app type or naming convention,
    // an unresolvable selector always comes back exactly as given, never
    // substituted with a guess. That's what makes it safe to call on a
    // selector that may already be correct for a row this viewer simply
    // can't currently see.
    expect(resolveMcpToolSelector("Chrome", [], [chromeApp])).toBe("Chrome")
    expect(resolveMcpToolSelector("Facebook Pages", [], [facebookApp])).toBe(
      "Facebook Pages"
    )
  })

  it("preserves the raw selector unchanged when a matched row's name is blank", () => {
    // A row *is* found here (by app_id) -- unlike the no-row-at-all case
    // above, this exercises the other unresolved path: a found row with
    // nothing usable to return. Whitespace-only is deliberately included:
    // "   " is truthy in JS, so this also pins that it's rejected by an
    // explicit trim-and-check, not just a falsy check.
    const mcpServers = [{ name: "   ", app_id: "facebook" }]
    expect(resolveMcpToolSelector("facebook", mcpServers, [facebookApp])).toBe(
      "facebook"
    )
  })

  it("preserves the raw selector unchanged when nothing matches at all", () => {
    expect(resolveMcpToolSelector("some-custom-server", [], [])).toBe(
      "some-custom-server"
    )
  })

  it("preserves an empty selector unchanged rather than guessing", () => {
    // Documents current behavior rather than asserting it's the ideal
    // outcome: an empty selector never resolves to anything, so it passes
    // through as "" (becoming the tool category "mcp:", handled downstream
    // by selection_spec.py as an unmatchable scope entry).
    expect(resolveMcpToolSelector("", [], [])).toBe("")
  })
})

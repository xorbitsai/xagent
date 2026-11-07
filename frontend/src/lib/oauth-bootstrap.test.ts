import { describe, it, expect, vi, afterEach } from "vitest"
import { validateRegionalBootstrap, regionalBootstrapURL } from "./oauth-bootstrap"

const origin = "https://cloud.xagent.co"
const good = `${origin}/api/oauth/launch/au?ticket=proof`
afterEach(() => vi.unstubAllGlobals())
describe("regional OAuth navigation", () => {
  it("accepts only the configured exact bootstrap", () => {
    expect(validateRegionalBootstrap(good, [origin])).toBe(good)
  })
  it.each([
    good.replace(origin, "https://evil.test"), good + "#fragment", good + "#",
    good + "&ticket=duplicate", good + "&token=secret", good + "&access_token=secret",
    good.replace("/au?", "/au/other?"), good.replace("/au?", "/%61u?"),
    good.replace("https://", "https://user@"), good.replace("ticket=proof", "ticket="),
  ])("rejects %s", raw => expect(() => validateRegionalBootstrap(raw, [origin])).toThrow())
  it("retains explicit legacy handling", async () => {
    expect(await regionalBootstrapURL({ authorization_url: "https://provider.test" })).toBeNull()
  })
  it("fails closed on unknown mode or unavailable deployment config", async () => {
    await expect(regionalBootstrapURL({ launch_mode: "future" })).rejects.toThrow()
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false }))
    await expect(regionalBootstrapURL({ launch_mode: "regional_bootstrap", authorization_url: good })).rejects.toThrow()
  })
  it("uses same-origin deployment config, not a launch response allowlist", async () => {
    const fetcher = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ oauth_authorization_origins: [origin] }) })
    vi.stubGlobal("fetch", fetcher)
    expect(await regionalBootstrapURL({ launch_mode: "regional_bootstrap", authorization_url: good })).toBe(good)
    expect(fetcher.mock.calls[0][0]).toBe("/api/deployment-config")
  })
})

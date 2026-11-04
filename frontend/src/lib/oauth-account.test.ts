import { describe, expect, it } from "vitest"

import { normalizeOAuthAccount } from "./oauth-account"

const SUFFIX = ".zendesk.com"

describe("normalizeOAuthAccount", () => {
  it("accepts a bare account label", () => {
    expect(normalizeOAuthAccount("acme", SUFFIX)).toBe("acme")
  })

  it("trims and lowercases, like the backend's normalize_oauth_account", () => {
    expect(normalizeOAuthAccount("  Acme-Support \n", SUFFIX)).toBe("acme-support")
  })

  it("accepts labels at the DNS length bounds", () => {
    expect(normalizeOAuthAccount("a", SUFFIX)).toBe("a")
    expect(normalizeOAuthAccount("a".repeat(63), SUFFIX)).toBe("a".repeat(63))
  })

  it.each([
    ["empty", ""],
    ["whitespace only", "   "],
    ["too long", "a".repeat(64)],
    ["leading hyphen", "-acme"],
    ["trailing hyphen", "acme-"],
    ["underscore", "ac_me"],
    ["inner space", "ac me"],
    ["non-ASCII", "acmé"],
  ])("rejects a malformed label (%s)", (_name, raw) => {
    expect(normalizeOAuthAccount(raw, SUFFIX)).toBeNull()
  })

  it.each([
    ["host only", "acme.zendesk.com"],
    ["https URL", "https://acme.zendesk.com"],
    ["URL with path and query", "https://ACME.zendesk.com/agent/tickets?x=1#top"],
    ["trailing slash", "http://acme.zendesk.com/"],
  ])("extracts the label from a pasted address on the declared host (%s)", (_name, raw) => {
    expect(normalizeOAuthAccount(raw, SUFFIX)).toBe("acme")
  })

  it.each([
    ["another host", "acme.example.com"],
    ["suffix not at a label boundary", "acmezendesk.com"],
    ["declared host nested under another domain", "acme.zendesk.com.evil.test"],
    ["multi-label account", "a.b.zendesk.com"],
    ["port", "acme.zendesk.com:443"],
    ["credentials", "https://user@acme.zendesk.com"],
    ["bare suffix", "zendesk.com"],
  ])("rejects an address that is not exactly one label on the declared host (%s)", (_name, raw) => {
    expect(normalizeOAuthAccount(raw, SUFFIX)).toBeNull()
  })

  it("matches the declared suffix case-insensitively", () => {
    expect(normalizeOAuthAccount("acme.ZENDESK.com", ".Zendesk.COM")).toBe("acme")
  })

  it("does not strip anything when the provider declares no suffix", () => {
    expect(normalizeOAuthAccount("acme.zendesk.com", "")).toBeNull()
    expect(normalizeOAuthAccount("acme", "")).toBe("acme")
  })
})

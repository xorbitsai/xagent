import { describe, expect, it } from "vitest"
import { resolveAgentLogoUrl } from "./utils"

const bigintValue = (globalThis as { BigInt: (value: number) => bigint }).BigInt(1)

const logoPathCases = [
  ["no leading slash", "logos/agent.png"],
  ["one leading slash", "/logos/agent.png"],
] as const

const logoBaseCases = [
  ["same-origin empty", ""],
  ["same-origin one slash", "/"],
  ["same-origin multiple slashes", "///"],
  ["root prefix no trailing slash", "/api"],
  ["root prefix one trailing slash", "/api/"],
  ["root prefix multiple trailing slashes", "/api///"],
  ["absolute prefix no trailing slash", "https://api.example/v1"],
  ["absolute prefix one trailing slash", "https://api.example/v1/"],
  ["absolute prefix multiple trailing slashes", "https://api.example/v1///"],
] as const

const joinedLogoCases = logoBaseCases.flatMap(([baseLabel, base]) =>
  logoPathCases.map(([pathLabel, path]) => {
    const normalizedBase = /^\/+$/u.test(base) ? "" : base.replace(/\/+$/u, "")
    return [baseLabel, base, pathLabel, path, `${normalizedBase}/${path.replace(/^\//u, "")}`] as const
  }),
)

const internalSpaceCases = [
  ["relative logo path", "logos/agent image.png", "/api", "/api/logos/agent image.png"],
  ["root-relative API base", "logos/agent.png", "/api internal", "/api internal/logos/agent.png"],
  ["absolute HTTP logo path", "https://assets.example/agent image.png", "/ignored", "https://assets.example/agent image.png"],
  ["absolute API base path prefix", "logos/agent.png", "https://api.example/v1 internal", "https://api.example/v1 internal/logos/agent.png"],
] as const

describe("resolveAgentLogoUrl", () => {
  it.each([
    ["HTTPS://assets.example/logo.png", "not consulted", "HTTPS://assets.example/logo.png"],
    ["http://assets.example:8443/logo.png?size=1#top", "\\u0000bad", "http://assets.example:8443/logo.png?size=1#top"],
  ])("accepts %s with base %s", (value, base, expected) => {
    expect(resolveAgentLogoUrl(value, base)).toBe(expected)
  })

  it.each(joinedLogoCases)(
    "joins %s base %s with a %s logo path %s using one boundary slash",
    (_baseLabel, base, _pathLabel, path, expected) => {
      expect(resolveAgentLogoUrl(path, base)).toBe(expected)
    },
  )

  it.each(internalSpaceCases)(
    "preserves ordinary internal spaces in a %s byte-for-byte",
    (_label, value, base, expected) => {
      expect(resolveAgentLogoUrl(value, base)).toBe(expected)
    },
  )

  it("accepts a colon after a slash in a relative logo path", () => {
    expect(resolveAgentLogoUrl("icons/tenant:42/logo.png", "/api:v1")).toBe(
      "/api:v1/icons/tenant:42/logo.png",
    )
    expect(resolveAgentLogoUrl("/icons/tenant:42/logo.png", "https://api.example/v1/")).toBe(
      "https://api.example/v1/icons/tenant:42/logo.png",
    )
    expect(resolveAgentLogoUrl("1tenant:42/logo.png", "/api")).toBe(
      "/api/1tenant:42/logo.png",
    )
    expect(resolveAgentLogoUrl("custom_name:payload", "/api")).toBe(
      "/api/custom_name:payload",
    )
  })

  it.each([
    undefined,
    null,
    1,
    true,
    {},
    [],
    new String("logo.png"),
    bigintValue,
    Symbol("logo"),
    () => "logo.png",
    "",
    "   ",
    " logo.png",
    "logo.png ",
    " logo.png ",
    "\u00a0logo.png",
    "\tlogo.png",
    "logo.png\n",
    "logo\u0000.png",
    "logo\u001f.png",
    "logo\u007f.png",
    "logo\\path.png",
    "logo\\\\path.png",
    "logo/mixed\\path.png",
    "logo\\mixed/path.png",
    "https://assets.example\\logo.png",
    "/\t/host",
    "/\n/host",
    "/\r/host",
    "/\u0000/host",
    "/\u007f/host",
    "//assets.example/logo.png",
    "///assets.example/logo.png",
    "/",
    "////",
    "data:image/png;base64,x",
    "DATA:image/png;base64,x",
    "Custom+V1.Test-Scheme:payload",
    "javascript:alert(1)",
    "httpx://assets.example/logo.png",
    "http://",
    "https://",
    "http:/assets.example/logo.png",
    "https:///host/path",
    "https:////host/path",
    "https://?query",
    "https://#fragment",
    "https://:443/logo.png",
  ])("rejects unsupported logo input %#", (value) => {
    expect(() => resolveAgentLogoUrl(value, "/api")).not.toThrow()
    expect(resolveAgentLogoUrl(value, "/api")).toBeNull()
  })

  it.each([
    "relative-base",
    "//api.example",
    "///api.example",
    "ftp://api.example",
    "httpx://api.example",
    "http://",
    "https://",
    "http:/api.example",
    "https:///api",
    "https:////host/path",
    "https://?query",
    "https://#fragment",
    "/api?query=1",
    "/api#fragment",
    "/api\\path",
    " /api",
    "/api ",
    "/api\u2003",
    "/api\tsegment",
    "/api\nsegment",
    "/api\rsegment",
    "/api\u0000segment",
    "/api\u001fsegment",
    "/api\u007fsegment",
    "https://api\t.example",
    "https://api\n.example",
    "https://api\r.example",
    "https://api\u0000.example",
    "https://api\u001f.example",
    "https://api\u007f.example",
    "https://api.example\\v1",
    "https://:443/api",
  ])("rejects invalid relative-path API base %s", (base) => {
    expect(resolveAgentLogoUrl("logos/agent.png", base)).toBeNull()
  })

  it.each(["\t", "\n", "\r", "\u0000", "\u001f", "\u007f"])(
    "rejects controls in an absolute HTTP(S) logo authority: %j",
    (control) => {
      expect(resolveAgentLogoUrl(`https://assets${control}.example/logo.png`, "/api")).toBeNull()
    },
  )

  it("does not consult an invalid base for an already-valid absolute logo", () => {
    expect(resolveAgentLogoUrl("https://assets.example/logo.png", "\u0000invalid")).toBe(
      "https://assets.example/logo.png",
    )
  })
})

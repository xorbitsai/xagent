import { spawnSync } from "node:child_process"
import { describe, expect, it } from "vitest"
import { browserSecurityHeaders } from "../../browser-security.mjs"
import { createRequire } from "node:module"

const { pathToRegexp } = createRequire(import.meta.url)("next/dist/compiled/path-to-regexp") as {
  pathToRegexp: (source: string) => RegExp
}

describe("standalone browser security policy", () => {
  it("enforces baseline CSP without permitting production eval", () => {
    const headers = Object.fromEntries(browserSecurityHeaders().map(({ key, value }) => [key, value]))
    const policy = headers["Content-Security-Policy"]
    for (const directive of ["base-uri 'none'", "object-src 'none'", "form-action 'self'", "frame-ancestors 'none'"]) {
      expect(policy).toContain(directive)
    }
    expect(policy).not.toContain("unsafe-eval")
    expect(headers["X-Frame-Options"]).toBe("DENY")
    expect(policy).toContain("script-src 'self' 'unsafe-inline' https:")
    expect(policy).toContain("frame-src 'self' blob: https://docs.google.com")
  })

  it("permits local API media only in development", () => {
    for (const development of [false, true]) {
      const policy = browserSecurityHeaders({ development })[0].value
      for (const directive of ["img-src", "media-src", "connect-src"]) {
        const value = policy.split("; ").find((part) => part.startsWith(directive))!
        expect(value.split(" ").includes("http:")).toBe(development)
      }
      expect(policy.includes("unsafe-eval")).toBe(development)
    }
  })

  it("keeps widgets embeddable without lifting other restrictions", () => {
    const headers = Object.fromEntries(browserSecurityHeaders({ embeddable: true }).map(({ key, value }) => [key, value]))
    expect(headers["X-Frame-Options"]).toBeUndefined()
    expect(headers["Content-Security-Policy"]).not.toContain("frame-ancestors")
    expect(headers["Content-Security-Policy"]).toContain("object-src 'none'")
  })

  it.each(["export", "standalone"])("wires the %s deployment without relying on ignored export headers", (output) => {
    const child = spawnSync(process.execPath, ["--input-type=module", "-e", `
      const { default: config } = await import('./next.config.mjs');
      console.log(JSON.stringify({output: config.output, headers: config.headers ? await config.headers() : null}));
    `], { cwd: process.cwd(), env: { ...process.env, NODE_ENV: "production", NEXT_OUTPUT: output }, encoding: "utf8" })
    expect(child.status, child.stderr).toBe(0)
    const config = JSON.parse(child.stdout)
    expect(config.output).toBe(output)
    if (output === "export") expect(config.headers).toBeNull()
    else {
      expect(config.headers).toHaveLength(2)
      const route = pathToRegexp(config.headers[0].source)
      for (const path of ["/", "/login", "/widgetfoo"]) expect(route.test(path)).toBe(true)
      for (const path of ["/widget", "/widget/x"]) expect(route.test(path)).toBe(false)
      expect(config.headers[0].headers.find(({ key }: { key: string }) => key === "X-Frame-Options")?.value).toBe("DENY")
      expect(config.headers[1].source).toBe("/widget/:path*")
      expect(config.headers[1].headers.some(({ key }: { key: string }) => key === "X-Frame-Options")).toBe(false)
    }
  })
})

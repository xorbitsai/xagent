import { spawnSync } from "node:child_process"
import { readFileSync } from "node:fs"
import { afterEach, describe, expect, it, vi } from "vitest"
import YAML from "yaml"
import { browserSecurityHeaders } from "../../browser-security.mjs"
import { createRequire } from "node:module"

const { pathToRegexp } = createRequire(import.meta.url)("next/dist/compiled/path-to-regexp") as {
  pathToRegexp: (source: string) => RegExp
}

describe("standalone browser security policy", () => {
  afterEach(() => vi.unstubAllEnvs())

  it.each<[string, string[]]>([
    ["", []],
    ["invalid", []],
    ["https://preview.example.test;evil", []],
    ["https://preview%2eexample.test", []],
    ["https://host\\evil", []],
    ["http://127.1:8787", []],
    ["https://preview.example.test:invalid", []],
    ["https://user:pass@preview.example.test", []],
    ["http://preview.example.test", []],
    ["https://preview.example.test/path?q=1", ["https://preview.example.test"]],
    ["http://127.0.0.1:8787", ["http://127.0.0.1:8787"]],
    ["http://[::1]:8787", ["http://[::1]:8787"]],
    ["auto", ["https://preview.xagent.co", "https://preview-stage.xagent.co", "https://preview-dev.xagent.co"]],
  ])("allows only the configured runner origins for %s", (previewOrigin, origins) => {
    for (const embeddable of [false, true]) {
      const policy = browserSecurityHeaders({ previewOrigin, embeddable })[0].value
      expect(policy.split("; ").find(part => part.startsWith("frame-src "))).toBe(
        ["frame-src 'self' blob: https://docs.google.com", ...origins].join(" "),
      )
    }
  })

  it.each<[string, string[]]>([
    ["https://widget.example.com", ["https://widget.example.com"]],
    ["https://*.widget.example.com", ["https://*.widget.example.com"]],
    ["https://intranet", ["https://intranet"]],
    ["https://xn--bcher-kva.example", ["https://xn--bcher-kva.example"]],
    [
      " https://a.example.com\thttps://*.b.example.com\nhttps://a.example.com ",
      ["https://a.example.com", "https://*.b.example.com"],
    ],
    [
      "https://a.example.com https://evil.example.com;script-src https://*.b.example.com",
      ["https://a.example.com", "https://*.b.example.com"],
    ],
  ])("appends valid extra frame-src origins for %j", (extraFrameSrc, origins) => {
    for (const embeddable of [false, true]) {
      const policy = browserSecurityHeaders({ extraFrameSrc, embeddable })[0].value
      expect(policy.split("; ").find(part => part.startsWith("frame-src "))).toBe(
        ["frame-src 'self' blob: https://docs.google.com", ...origins].join(" "),
      )
    }
  })

  it.each([
    "widget.example.com",
    "http://widget.example.com",
    "wss://widget.example.com",
    "https://widget.example.com:443",
    "https://widget.example.com/",
    "https://widget.example.com/chat",
    "https://widget.example.com?x=1",
    "https://widget.example.com#x",
    "https://user@widget.example.com",
    "https://widget%2eexample.com",
    "https://widget.example.com\\evil",
    "https://Widget.example.com",
    "https://wídget.example.com",
    "https://widget.example.com.",
    "https://-widget.example.com",
    "https://widget-.example.com",
    `https://${"a".repeat(64)}.example.com`,
    "https://10.0.0.1",
    "https://[::1]",
    "https://*",
    "*",
    "https://*.com",
    "https://*.*.example.com",
    "https://a.*.example.com",
    "https://widget.example.com;script-src",
    "https://widget.example.com,https://evil.example.com",
    "https://a.example.com\u00a0https://b.example.com",
    "'self'",
    "'unsafe-inline'",
    "data:",
  ])("refuses the invalid extra frame-src entry %j", (extraFrameSrc) => {
    expect(browserSecurityHeaders({ extraFrameSrc })).toEqual(browserSecurityHeaders({ extraFrameSrc: "" }))
  })

  it("leaves the policy byte-for-byte unchanged without extra frame-src origins", () => {
    vi.stubEnv("XAGENT_CSP_EXTRA_FRAME_SRC", "")
    for (const embeddable of [false, true]) {
      const baseline = browserSecurityHeaders({ embeddable, extraFrameSrc: undefined })
      expect(baseline[0].value).toContain("; frame-src 'self' blob: https://docs.google.com; worker-src")
      for (const extraFrameSrc of ["", " \t\n"]) {
        expect(browserSecurityHeaders({ embeddable, extraFrameSrc })).toEqual(baseline)
      }
      expect(browserSecurityHeaders({ embeddable })).toEqual(baseline)
    }
  })

  it("reads extra frame-src origins from the build environment and the frontend image", () => {
    vi.stubEnv("XAGENT_CSP_EXTRA_FRAME_SRC", "https://*.widget.example.com")
    expect(browserSecurityHeaders()[0].value).toContain(" https://*.widget.example.com;")
    const child = spawnSync(process.execPath, ["--input-type=module", "-e", `
      const { default: config } = await import('./next.config.mjs');
      console.log(JSON.stringify(await config.headers()));
    `], { env: { ...process.env, NODE_ENV: "production", NEXT_OUTPUT: "standalone" }, encoding: "utf8" })
    expect(child.status, child.stderr).toBe(0)
    for (const route of JSON.parse(child.stdout)) {
      expect(route.headers.find((h: { key: string }) => h.key === "Content-Security-Policy").value)
        .toContain(" https://*.widget.example.com;")
    }
    const dockerfile = readFileSync("../docker/Dockerfile.frontend", "utf8")
    expect(dockerfile).toContain("ARG XAGENT_CSP_EXTRA_FRAME_SRC=")
    expect(dockerfile).toContain("ENV XAGENT_CSP_EXTRA_FRAME_SRC=${XAGENT_CSP_EXTRA_FRAME_SRC}")
  })

  it("passes the public build setting to CSP and every official frontend image build", async () => {
    vi.stubEnv("NEXT_PUBLIC_XAGENT_HTML_PREVIEW_ORIGIN", "https://preview.example.test/path")
    const policy = browserSecurityHeaders()[0].value
    expect(policy).toContain("https://preview.example.test;")
    const child = spawnSync(process.execPath, ["--input-type=module", "-e", `
      const { default: config } = await import('./next.config.mjs');
      console.log(JSON.stringify(await config.headers()));
    `], { env: { ...process.env, NODE_ENV: "production", NEXT_OUTPUT: "standalone" }, encoding: "utf8" })
    expect(child.status, child.stderr).toBe(0)
    for (const route of JSON.parse(child.stdout)) {
      expect(route.headers.find((h: { key: string }) => h.key === "Content-Security-Policy").value).toContain("https://preview.example.test;")
    }
    const dockerfile = readFileSync("../docker/Dockerfile.frontend", "utf8")
    expect(dockerfile).toContain("ARG NEXT_PUBLIC_XAGENT_HTML_PREVIEW_ORIGIN=")
    expect(dockerfile).toContain("ENV NEXT_PUBLIC_XAGENT_HTML_PREVIEW_ORIGIN=${NEXT_PUBLIC_XAGENT_HTML_PREVIEW_ORIGIN}")
    for (const workflow of ["docker-publish.yml", "nightly-build.yml"]) {
      const parsed = YAML.parse(readFileSync(`../.github/workflows/${workflow}`, "utf8"))
      const steps = Object.values(parsed.jobs).flatMap((job) => (job as { steps: Record<string, unknown>[] }).steps)
      const builds = steps.filter(step => (step.with as Record<string, string> | undefined)?.file === "docker/Dockerfile.frontend")
      expect(builds).toHaveLength(2)
      for (const step of builds) expect((step.with as Record<string, string>)["build-args"])
        .toContain("NEXT_PUBLIC_XAGENT_HTML_PREVIEW_ORIGIN=${{ vars.NEXT_PUBLIC_XAGENT_HTML_PREVIEW_ORIGIN }}")
    }
  })
  it("enforces baseline CSP without permitting production eval", () => {
    const headers = Object.fromEntries(browserSecurityHeaders().map(({ key, value }) => [key, value]))
    const policy = headers["Content-Security-Policy"]
    for (const directive of ["base-uri 'none'", "object-src 'none'", "form-action 'self'", "frame-ancestors 'none'"]) {
      expect(policy).toContain(directive)
    }
    expect(policy.split(/[;\s]+/)).not.toContain("'unsafe-eval'")
    expect(policy.split(/[;\s]+/)).toContain("'wasm-unsafe-eval'")
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
      expect(policy.split(/[;\s]+/).includes("'unsafe-eval'")).toBe(development)
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

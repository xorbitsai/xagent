/**
 * Behavioral tests for the standalone embed script (frontend/public/widget.js).
 *
 * The file is vanilla JS, not a module -- it's an IIFE meant to run once at
 * <script> load time, so it's exercised here by stubbing the minimal DOM
 * surface it touches (document.currentScript, createElement, head/body) and
 * evaluating its source directly, rather than importing it.
 */
import { readFileSync } from "node:fs"
import path from "node:path"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const widgetJsSource = readFileSync(
  path.resolve(__dirname, "../../public/widget.js"),
  "utf8",
)

interface FakeScriptTag {
  attrs: Record<string, string>
  getAttribute(name: string): string | null
  src: string
}

function makeScriptTag(attrs: Record<string, string>): FakeScriptTag {
  return {
    attrs,
    getAttribute(name) {
      return Object.prototype.hasOwnProperty.call(this.attrs, name) ? this.attrs[name] : null
    },
    src: "http://localhost:3000/widget.js",
  }
}

async function runWidget(attrs: Record<string, string>) {
  const store: Record<string, string> = {}
  const fakeLocalStorage = {
    getItem: (k: string) => (k in store ? store[k] : null),
    setItem: (k: string, v: string) => {
      store[k] = v
    },
  }
  const consoleErrors: string[] = []
  let iframeSrc: string | null = null

  const fakeDocument = {
    currentScript: makeScriptTag(attrs),
    createElement: () => {
      const el: Record<string, unknown> = {
        style: {},
        classList: { add() {}, remove() {} },
        appendChild() {},
      }
      Object.defineProperty(el, "src", {
        set(v: string) {
          iframeSrc = v
        },
        get() {
          return iframeSrc
        },
      })
      return el
    },
    head: { appendChild() {} },
    body: { appendChild() {} },
  }

  vi.stubGlobal("document", fakeDocument)
  vi.stubGlobal("localStorage", fakeLocalStorage)
  vi.stubGlobal("console", { ...console, error: (...args: unknown[]) => consoleErrors.push(args.join(" ")) })
  vi.stubGlobal(
    "fetch",
    vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ ticket: "test-ticket", agent_id: 1 }),
      }),
    ),
  )

  // eslint-disable-next-line no-eval
  eval(widgetJsSource)

  // Flush the fetch().then().then() microtask chain that calls loadIframe.
  await Promise.resolve()
  await Promise.resolve()
  await Promise.resolve()
  await Promise.resolve()

  return { iframeSrc, consoleErrors, storedDirectGuestId: store["xagent_guest_id"] ?? null }
}

describe("widget.js embed script", () => {
  beforeEach(() => {
    vi.resetModules()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("uses the signed end-user identity when both id and signature are provided", async () => {
    const { iframeSrc, consoleErrors, storedDirectGuestId } = await runWidget({
      "data-widget-key": "wk-test",
      "data-end-user-id": "tenant_42:user_007",
      "data-end-user-signature": "deadbeefdeadbeef",
    })

    expect(iframeSrc).toContain("end_user_id=tenant_42%3Auser_007")
    expect(iframeSrc).toContain("end_user_signature=deadbeefdeadbeef")
    expect(iframeSrc).not.toContain("guest_id=")
    expect(consoleErrors).toHaveLength(0)
    expect(storedDirectGuestId).toBeNull()
  })

  it("falls back to an anonymous per-browser id when the signature is missing", async () => {
    const { iframeSrc, consoleErrors, storedDirectGuestId } = await runWidget({
      "data-widget-key": "wk-test",
      "data-end-user-id": "tenant_42:user_007",
    })

    expect(iframeSrc).toContain("guest_id=guest_")
    expect(iframeSrc).not.toContain("end_user_id=")
    expect(consoleErrors.some((msg) => msg.includes("without data-end-user-signature"))).toBe(true)
    expect(storedDirectGuestId).toMatch(/^guest_/)
  })

  it("rejects an end-user id over 256 characters instead of silently truncating it", async () => {
    const { iframeSrc, consoleErrors } = await runWidget({
      "data-widget-key": "wk-test",
      "data-end-user-id": "x".repeat(300),
      "data-end-user-signature": "deadbeefdeadbeef",
    })

    expect(iframeSrc).toContain("guest_id=guest_")
    expect(iframeSrc).not.toContain("end_user_id=")
    expect(consoleErrors.some((msg) => msg.includes("exceeds 256 characters"))).toBe(true)
  })

  it("generates and caches a random anonymous guest id when no identity is provided at all", async () => {
    const first = await runWidget({ "data-widget-key": "wk-test" })
    expect(first.iframeSrc).toContain("guest_id=guest_")
    expect(first.consoleErrors).toHaveLength(0)
  })

  it("url-encodes special characters in the signed end-user id", async () => {
    const { iframeSrc } = await runWidget({
      "data-widget-key": "wk-test",
      "data-end-user-id": "tenant&42#user@007 space",
      "data-end-user-signature": "deadbeefdeadbeef",
    })

    expect(iframeSrc).toContain("end_user_id=tenant%2642%23user%40007%20space")
  })
})

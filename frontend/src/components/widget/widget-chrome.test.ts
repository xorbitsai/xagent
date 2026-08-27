import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { pathToFileURL } from "node:url"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const widgetScriptPath = resolve(process.cwd(), "public/widget.js")
const widgetScript = readFileSync(widgetScriptPath, "utf8")
const widgetScriptUrl = pathToFileURL(widgetScriptPath).href

const HOST = "https://chat.example"
const CLOSED_KEY = "xagent_widget_closed"

function runWidget(attributes: Record<string, string> = { "data-widget-key": "widget-secret" }) {
  const script = document.createElement("script")
  script.src = `${HOST}/widget.js`
  for (const [name, value] of Object.entries(attributes)) {
    script.setAttribute(name, value)
  }
  document.body.appendChild(script)
  Object.defineProperty(document, "currentScript", { configurable: true, value: script })
  window.eval(`${widgetScript}\n//# sourceURL=${widgetScriptUrl}`)
}

function panelEl() {
  return document.querySelector(".xagent-widget-panel")
}

function fabEl() {
  return document.querySelector<HTMLButtonElement>(".xagent-widget-fab")
}

function iframeEl() {
  return document.querySelector<HTMLIFrameElement>(".xagent-widget-iframe")
}

function fromIframe(type: string) {
  window.dispatchEvent(new MessageEvent("message", {
    data: { xagent: true, v: 1, type },
    origin: HOST,
    source: iframeEl()?.contentWindow as Window,
  }))
}

describe("widget close chrome", () => {
  let currentScriptDescriptor: PropertyDescriptor | undefined
  const fetchMock = vi.fn()

  beforeEach(() => {
    currentScriptDescriptor = Object.getOwnPropertyDescriptor(document, "currentScript")
    document.head.innerHTML = ""
    document.body.innerHTML = ""
    localStorage.clear()
    localStorage.setItem("xagent_guest_id", "guest-fixed")
    vi.stubGlobal("fetch", fetchMock)
    fetchMock.mockReset()
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ ticket: "t", agent_id: 1 }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }))
  })

  afterEach(() => {
    if (currentScriptDescriptor) {
      Object.defineProperty(document, "currentScript", currentScriptDescriptor)
    } else {
      Reflect.deleteProperty(document, "currentScript")
    }
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it("starts closed with no persisted preference for a first-time visitor", () => {
    runWidget()

    expect(panelEl()).not.toHaveClass("open")
    expect(localStorage.getItem(CLOSED_KEY)).toBeNull()
  })

  it("opens on FAB click and persists the open (not-closed) preference", () => {
    runWidget()

    fabEl()?.click()

    expect(panelEl()).toHaveClass("open")
    expect(localStorage.getItem(CLOSED_KEY)).toBe("false")
  })

  it("closes on a second FAB click and persists the closed preference", () => {
    runWidget()

    fabEl()?.click()
    fabEl()?.click()

    expect(panelEl()).not.toHaveClass("open")
    expect(localStorage.getItem(CLOSED_KEY)).toBe("true")
  })

  it("reopens automatically once the embed ticket resolves, for a visitor who last left it open", async () => {
    localStorage.setItem(CLOSED_KEY, "false")

    runWidget()

    // Guest mode's auto-open is deferred to iframe.src actually being set,
    // which only happens after this fetch's promise chain resolves.
    await vi.waitFor(() => {
      expect(panelEl()).toHaveClass("open")
    })
  })

  it("does not re-persist the preference when auto-restoring on load", async () => {
    localStorage.setItem(CLOSED_KEY, "false")
    const setItemSpy = vi.spyOn(localStorage, "setItem")

    runWidget()

    await vi.waitFor(() => {
      expect(panelEl()).toHaveClass("open")
    })
    expect(setItemSpy).not.toHaveBeenCalledWith(CLOSED_KEY, expect.anything())
    setItemSpy.mockRestore()
  })

  it("does not auto-open when the embed-ticket request fails, even for a visitor who left it open", async () => {
    // A stale allowlist, a rate limit, or a network error all resolve this
    // fetch chain without ever setting iframe.src (see createGuestMode) --
    // auto-opening ahead of that would show a blank panel on every reload.
    fetchMock.mockReset()
    fetchMock.mockResolvedValueOnce(new Response("forbidden", { status: 403 }))
    localStorage.setItem(CLOSED_KEY, "false")
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)

    runWidget()

    await vi.waitFor(() => {
      expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("embed authorization failed"))
    })
    expect(panelEl()).not.toHaveClass("open")
    expect(iframeEl()?.getAttribute("src")).toBeNull()
    errorSpy.mockRestore()
  })

  it("stays closed on load once the visitor has closed it before", () => {
    localStorage.setItem(CLOSED_KEY, "true")

    runWidget()

    expect(panelEl()).not.toHaveClass("open")
  })

  it("closes the panel and persists closed when the iframe posts widget_close", () => {
    runWidget()
    fabEl()?.click()
    expect(panelEl()).toHaveClass("open")

    fromIframe("widget_close")

    expect(panelEl()).not.toHaveClass("open")
    expect(localStorage.getItem(CLOSED_KEY)).toBe("true")
  })

  it("ignores an unrecognized chrome message type", () => {
    runWidget()
    fabEl()?.click()

    fromIframe("widget_minimize")

    expect(panelEl()).toHaveClass("open")
  })

  it("ignores a chrome message from a different origin", () => {
    runWidget()
    fabEl()?.click()

    window.dispatchEvent(new MessageEvent("message", {
      data: { xagent: true, v: 1, type: "widget_close" },
      origin: "https://evil.example",
      source: iframeEl()?.contentWindow as Window,
    }))

    expect(panelEl()).toHaveClass("open")
  })

  it("ignores a chrome message not sourced from this widget's own iframe", () => {
    runWidget()
    fabEl()?.click()

    window.dispatchEvent(new MessageEvent("message", {
      data: { xagent: true, v: 1, type: "widget_close" },
      origin: HOST,
      source: window,
    }))

    expect(panelEl()).toHaveClass("open")
  })

  it("ignores a same-origin, same-source message missing the xagent envelope", () => {
    runWidget()
    fabEl()?.click()

    window.dispatchEvent(new MessageEvent("message", {
      data: { type: "widget_close" },
      origin: HOST,
      source: iframeEl()?.contentWindow as Window,
    }))

    expect(panelEl()).toHaveClass("open")
  })

  it("ignores a same-origin, same-source message with a mismatched protocol version", () => {
    runWidget()
    fabEl()?.click()

    window.dispatchEvent(new MessageEvent("message", {
      data: { xagent: true, v: 2, type: "widget_close" },
      origin: HOST,
      source: iframeEl()?.contentWindow as Window,
    }))

    expect(panelEl()).toHaveClass("open")
  })

  it("ignores a chrome message when the iframe has been detached (contentWindow is null)", () => {
    runWidget()
    fabEl()?.click()
    const frame = iframeEl()
    if (!frame) throw new Error("iframe not mounted")
    const contentWindowGetter = vi.spyOn(frame, "contentWindow", "get").mockReturnValue(null)

    // A detached iframe's own contentWindow is null; a same-origin message
    // whose event.source also happens to be null must not be misread as a
    // match against it.
    window.dispatchEvent(new MessageEvent("message", {
      data: { xagent: true, v: 1, type: "widget_close" },
      origin: HOST,
      source: null,
    }))

    expect(panelEl()).toHaveClass("open")
    contentWindowGetter.mockRestore()
  })

  it("falls back to closed when reading the persisted preference throws", async () => {
    // Scoped to this one key: a blanket throw would also hit the pre-existing,
    // unrelated xagent_guest_id lookup earlier in the same bootstrap and blow
    // up before ever reaching readStoredClosed().
    const realGetItem = localStorage.getItem.bind(localStorage)
    const getItemSpy = vi.spyOn(localStorage, "getItem").mockImplementation((key) => {
      if (key === CLOSED_KEY) throw new Error("blocked")
      return realGetItem(key)
    })
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)

    runWidget()

    await vi.waitFor(() => {
      expect(iframeEl()?.getAttribute("src")).not.toBeNull()
    })
    expect(panelEl()).not.toHaveClass("open")
    // "panel stays closed" alone doesn't distinguish a graceful fallback from
    // readStoredClosed() throwing and the exception just happening to be
    // swallowed one level up, by the embed-ticket fetch chain's own catch --
    // which logs this different message. Assert it's never reached.
    expect(errorSpy).not.toHaveBeenCalledWith(
      expect.stringContaining("embed authorization request failed"),
    )
    getItemSpy.mockRestore()
    errorSpy.mockRestore()
  })

  it("does not surface an uncaught error when persisting the preference is blocked", () => {
    // A synchronous throw inside a native onclick handler doesn't propagate
    // back through .click() in jsdom (matches real browser behavior) --
    // expect(() => ...).not.toThrow() can't observe it either way. The
    // window "error" event is how jsdom actually surfaces it.
    runWidget()
    const setItemSpy = vi.spyOn(localStorage, "setItem").mockImplementation(() => {
      throw new Error("blocked")
    })
    const onError = vi.fn()
    window.addEventListener("error", onError)

    fabEl()?.click()

    expect(onError).not.toHaveBeenCalled()
    expect(panelEl()).toHaveClass("open")
    window.removeEventListener("error", onError)
    setItemSpy.mockRestore()
  })
})

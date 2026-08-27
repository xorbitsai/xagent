import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { pathToFileURL } from "node:url"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const widgetScriptPath = resolve(process.cwd(), "public/widget.js")
const widgetScript = readFileSync(widgetScriptPath, "utf8")
const widgetScriptUrl = pathToFileURL(widgetScriptPath).href

const HOST = "https://chat.example"

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

  it("starts closed for a first-time visitor", () => {
    runWidget()

    expect(panelEl()).not.toHaveClass("open")
  })

  it("opens on FAB click", () => {
    runWidget()

    fabEl()?.click()

    expect(panelEl()).toHaveClass("open")
  })

  it("closes on a second FAB click", () => {
    runWidget()

    fabEl()?.click()
    fabEl()?.click()

    expect(panelEl()).not.toHaveClass("open")
  })

  it("closes the panel when the iframe posts widget_close", () => {
    runWidget()
    fabEl()?.click()
    expect(panelEl()).toHaveClass("open")

    fromIframe("widget_close")

    expect(panelEl()).not.toHaveClass("open")
  })

  it("stops reacting to widget_close once the widget is torn down (removed from the DOM)", async () => {
    // Regression coverage: this listener used to be an inline anonymous
    // function, structurally impossible to removeEventListener, and wasn't
    // wired into the file's established torndown-flag/teardown-observer
    // pattern that every other listener here uses. A stray message arriving
    // after removal (a host SPA swap, not a full page reload) would
    // otherwise still mutate a detached panel.
    runWidget()
    fabEl()?.click()
    expect(panelEl()).toHaveClass("open")
    // Both captured before removal: querying by class after the container is
    // detached returns null (the element still exists, just outside the
    // document), and jsdom may separately null out a detached iframe's own
    // contentWindow, which would make the pre-existing origin/source check
    // reject the message on its own -- this proves the *teardown* path
    // specifically, independent of that check, by keeping the source a
    // match regardless.
    const capturedPanel = panelEl()
    const capturedSource = iframeEl()?.contentWindow as Window

    document.querySelector(".xagent-widget-container")?.remove()
    // The teardown observer's callback fires as a microtask.
    await Promise.resolve()

    window.dispatchEvent(new MessageEvent("message", {
      data: { xagent: true, v: 1, type: "widget_close" },
      origin: HOST,
      source: capturedSource,
    }))

    // Unchanged from the open-click above: a stray post-teardown widget_close
    // must not have run closePanel().
    expect(capturedPanel).toHaveClass("open")
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
})

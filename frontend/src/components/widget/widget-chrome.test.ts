import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { pathToFileURL } from "node:url"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { WidgetParentMessageType } from "@/lib/widget-parent-message"

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

  afterEach(async () => {
    for (const container of document.querySelectorAll(".xagent-widget-container")) {
      container.remove()
    }
    // Let panelRemovalObserver's callback run its production teardown (which
    // touches `window`) before Vitest tears down the jsdom globals -- a
    // container left in the DOM until then fires that MutationObserver
    // callback after `window` is already gone, surfacing as an uncaught
    // "window is not defined" even though every test still passes.
    await Promise.resolve()

    if (currentScriptDescriptor) {
      Object.defineProperty(document, "currentScript", currentScriptDescriptor)
    } else {
      Reflect.deleteProperty(document, "currentScript")
    }
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it("keeps the widget_close message type literal in sync with the host script", () => {
    // widget.js is a hand-authored static asset with no build-time import
    // from TS source, so nothing else ties these two literals together --
    // a rename on one side without the other would silently break the close
    // button (a same-origin, same-source, correctly-enveloped message the
    // host would now just ignore) with no compiler error to catch it.
    const messageType: WidgetParentMessageType = "widget_close"
    expect(widgetScript).toContain(`data.type === '${messageType}'`)
  })

  it("keeps the chrome-readiness message type literals in sync with the host script", () => {
    const ready: WidgetParentMessageType = "widget_chrome_ready"
    const notReady: WidgetParentMessageType = "widget_chrome_not_ready"
    expect(widgetScript).toContain(`data.type === '${ready}'`)
    expect(widgetScript).toContain(`data.type === '${notReady}'`)
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

  it("reopens on FAB click after a widget_close message closed it", () => {
    // openPanel()/closePanel() share one `isOpen` variable with fab.onclick;
    // a widget_close message must leave that variable in the same state a
    // manual FAB close would, not just the panel's CSS class, or the FAB
    // toggle desyncs from the panel on the very next click.
    runWidget()
    fabEl()?.click()
    fromIframe("widget_close")
    expect(panelEl()).not.toHaveClass("open")

    fabEl()?.click()

    expect(panelEl()).toHaveClass("open")
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

  it("marks the panel chrome-ready once the iframe's own close control confirms it's mounted", () => {
    runWidget()
    fabEl()?.click()
    expect(panelEl()).not.toHaveClass("xagent-chrome-ready")

    fromIframe("widget_chrome_ready")

    expect(panelEl()).toHaveClass("xagent-chrome-ready")
  })

  it("revokes chrome-ready if the child's close control unmounts while the panel stays open", () => {
    // e.g. an active Session degrading mid-conversation: WidgetChromeControls
    // unmounts, but nothing closes the panel itself. The mobile FAB-hiding
    // CSS rule keys off this class specifically so the parent's fallback
    // close control reappears instead of leaving the visitor stuck behind a
    // full-screen overlay with no dismiss action at all.
    runWidget()
    fabEl()?.click()
    fromIframe("widget_chrome_ready")
    expect(panelEl()).toHaveClass("xagent-chrome-ready")

    fromIframe("widget_chrome_not_ready")

    expect(panelEl()).not.toHaveClass("xagent-chrome-ready")
    expect(panelEl()).toHaveClass("open")
  })

  it("resets chrome-ready on an iframe reload, since a fresh document has confirmed nothing yet", () => {
    // widget_chrome_not_ready is only ever sent from WidgetChromeControls's
    // React unmount cleanup -- a real navigation of the iframe's document
    // (as opposed to the app's own client-side routing, which never fires
    // `load`) replaces that whole document without running it, which would
    // otherwise leave a stale .xagent-chrome-ready from the previous
    // document hiding the FAB with nothing behind it to close the panel.
    runWidget()
    fabEl()?.click()
    fromIframe("widget_chrome_ready")
    expect(panelEl()).toHaveClass("xagent-chrome-ready")

    iframeEl()?.dispatchEvent(new Event("load"))

    expect(panelEl()).not.toHaveClass("xagent-chrome-ready")
  })

  it("stops resetting chrome-ready on iframe load once the widget is torn down", async () => {
    runWidget()
    fabEl()?.click()
    fromIframe("widget_chrome_ready")
    const capturedPanel = panelEl()
    const capturedIframe = iframeEl()

    document.querySelector(".xagent-widget-container")?.remove()
    await Promise.resolve()

    capturedIframe?.dispatchEvent(new Event("load"))

    expect(capturedPanel).toHaveClass("xagent-chrome-ready")
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

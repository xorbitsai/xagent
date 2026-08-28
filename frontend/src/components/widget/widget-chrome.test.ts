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

  it("labels the FAB for screen readers, and keeps the label in sync with open/close state", () => {
    // In the no-ready mobile states this is the only dismiss control the
    // panel has at all -- an unlabeled icon-only button leaves a screen
    // reader user with no way to know what it does.
    runWidget()
    expect(fabEl()).toHaveAttribute("aria-label", "Open chat")

    fabEl()?.click()
    expect(fabEl()).toHaveAttribute("aria-label", "Close chat")

    fabEl()?.click()
    expect(fabEl()).toHaveAttribute("aria-label", "Open chat")
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
    // All captured before removal: querying by class after the container is
    // detached returns null (the element still exists, just outside the
    // document), and jsdom may separately null out a detached iframe's own
    // contentWindow, which would make the pre-existing origin/source check
    // reject the message on its own -- this proves the *teardown* path
    // specifically, independent of that check, by keeping the source a
    // match regardless.
    const capturedSource = iframeEl()?.contentWindow as Window
    const capturedFab = fabEl()!
    // Teardown itself now clears the panel's 'open' class directly (so a
    // reinserted stale node can't render an unclosable full-screen overlay
    // -- see widget-bootstrap.test.ts), so the panel being closed afterward
    // no longer distinguishes "the message was ignored" from "teardown
    // already did this." Teardown never touches the FAB's icon markup
    // though (only its display), so that staying exactly what it was
    // immediately after teardown -- not reset back to the closed-state icon
    // by a live closePanel() call -- is what actually proves the stray
    // message never ran anything.
    const fabIconAfterTeardown = () => capturedFab.innerHTML

    document.querySelector(".xagent-widget-container")?.remove()
    // The teardown observer's callback fires as a microtask.
    await Promise.resolve()
    const iconRightAfterTeardown = fabIconAfterTeardown()

    window.dispatchEvent(new MessageEvent("message", {
      data: { xagent: true, v: 1, type: "widget_close" },
      origin: HOST,
      source: capturedSource,
    }))

    expect(fabIconAfterTeardown()).toBe(iconRightAfterTeardown)
  })

  it("marks the panel chrome-ready once the iframe's own close control confirms it's mounted", () => {
    runWidget()
    fabEl()?.click()
    expect(panelEl()).not.toHaveClass("xagent-widget-chrome-ready")

    fromIframe("widget_chrome_ready")

    expect(panelEl()).toHaveClass("xagent-widget-chrome-ready")
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
    expect(panelEl()).toHaveClass("xagent-widget-chrome-ready")

    fromIframe("widget_chrome_not_ready")

    expect(panelEl()).not.toHaveClass("xagent-widget-chrome-ready")
    expect(panelEl()).toHaveClass("open")
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

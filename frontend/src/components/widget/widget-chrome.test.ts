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

function styleText() {
  return document.head.querySelector("style")!.innerHTML
}

// Slices from the desktop-scoped @media marker up to the next @media marker
// (the mobile block that follows it) -- narrow enough that a rule can't leak
// in from a later, unrelated block the way slicing to end-of-string would.
function expandedBlock() {
  const text = styleText()
  const start = text.indexOf("@media not all and (max-width: 480px)")
  if (start === -1) throw new Error("expanded media query block not found in generated <style>")
  const end = text.indexOf("@media", start + 1)
  return end === -1 ? text.slice(start) : text.slice(start, end)
}

function mobileBlock() {
  const text = styleText()
  const start = text.indexOf("@media (max-width: 480px)")
  if (start === -1) throw new Error("mobile media query block not found in generated <style>")
  return text.slice(start)
}

describe("widget chrome", () => {
  let currentScriptDescriptor: PropertyDescriptor | undefined
  let originalInnerWidth: number
  const fetchMock = vi.fn()

  beforeEach(() => {
    currentScriptDescriptor = Object.getOwnPropertyDescriptor(document, "currentScript")
    originalInnerWidth = window.innerWidth
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
    Object.defineProperty(window, "innerWidth", { configurable: true, value: originalInnerWidth })
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

  it("keeps every WidgetParentMessageType literal in sync with the host script", () => {
    // widget.js is a hand-authored static asset with no build-time import
    // from TS source, so nothing else ties these two literals together --
    // a rename on one side without the other would silently break the
    // corresponding control (a same-origin, same-source, correctly-enveloped
    // message the host would now just ignore) with no compiler error to
    // catch it.
    const messageTypes: WidgetParentMessageType[] = ["widget_close", "widget_expand", "widget_collapse"]
    for (const messageType of messageTypes) {
      expect(widgetScript).toContain(`data.type === '${messageType}'`)
    }
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

  it("ignores an unrecognized chrome message type", () => {
    runWidget()
    fabEl()?.click()

    fromIframe("widget_minimize")

    expect(panelEl()).toHaveClass("open")
  })

  it("expands the panel when the iframe posts widget_expand", () => {
    runWidget()
    fabEl()?.click()

    fromIframe("widget_expand")

    expect(panelEl()).toHaveClass("expanded")
  })

  it("tells the iframe to correct itself when the mobile guard rejects widget_expand", () => {
    // Without this, the iframe's own optimistic setIsExpanded(true) is never
    // corrected: the menu keeps reading "Collapse window" with nothing
    // actually expanded, and the visitor's next click (a no-op collapse from
    // the iframe's perspective) needs a second click after an eventual
    // desktop-width resize to actually expand.
    Object.defineProperty(window, "innerWidth", { configurable: true, value: 400 })
    runWidget()
    fabEl()?.click()
    const postToIframe = vi.spyOn(iframeEl()!.contentWindow!, "postMessage")

    fromIframe("widget_expand")

    expect(panelEl()).not.toHaveClass("expanded")
    expect(postToIframe).toHaveBeenCalledWith(
      { xagent: true, v: 1, type: "widget_expand_rejected" },
      HOST,
    )
  })

  it("does not tell the iframe to correct itself when widget_expand actually succeeds", () => {
    runWidget()
    fabEl()?.click()
    const postToIframe = vi.spyOn(iframeEl()!.contentWindow!, "postMessage")

    fromIframe("widget_expand")

    expect(panelEl()).toHaveClass("expanded")
    expect(postToIframe).not.toHaveBeenCalled()
  })

  it("keeps the .expanded rule (and its resize-handle override) scoped above the mobile breakpoint", () => {
    // jsdom never evaluates @media against getComputedStyle (confirmed
    // empirically), so this asserts against the generated CSS text instead --
    // the same pattern the sibling widget-bootstrap.test.ts file already uses
    // for the mobile block. Without this, a future edit that moves .expanded
    // into the mobile block, or duplicates it there, would pass every other
    // test here while silently breaking "can't expand on mobile."
    runWidget({ "data-widget-key": "widget-secret" })

    const block = expandedBlock()
    expect(block).toMatch(/\.xagent-widget-panel\.expanded\s*\{[^}]*width:\s*min\(720px, 100vw - 40px\);/)
    expect(block).toMatch(/\.xagent-widget-panel\.expanded\s+\.xagent-widget-resize-handle\s*\{[^}]*display:\s*none;/)
    expect(mobileBlock()).not.toMatch(/\.xagent-widget-panel\.expanded/)
  })

  it("collapses the panel, restoring its normal width, when the iframe posts widget_collapse", () => {
    localStorage.setItem("xagent_widget_width", "500")
    runWidget()
    fabEl()?.click()
    fromIframe("widget_expand")
    // The .expanded rule's own width would otherwise lose to a lingering
    // inline one -- expanding must clear whatever the resize handle left.
    // toHaveStyle checks computed style, which jsdom never resolves for
    // anything behind a media query (confirmed empirically), so this reads
    // the raw inline style directly like the resize-handle tests already do.
    expect((panelEl() as HTMLElement).style.width).toBe("")

    fromIframe("widget_collapse")

    expect(panelEl()).not.toHaveClass("expanded")
    expect((panelEl() as HTMLElement).style.width).toBe("500px")
  })

  it("does not let a same-side window resize clobber the expanded width", () => {
    // applyPanelWidth() (called from onWindowResize on every horizontal
    // resize) used to unconditionally set an inline width, which outranks
    // the .expanded CSS rule's own width by specificity regardless of
    // viewport -- desyncing the panel's width from its still-expanded
    // height on any resize that stays well above the mobile breakpoint.
    runWidget()
    fabEl()?.click()
    fromIframe("widget_expand")
    expect((panelEl() as HTMLElement).style.width).toBe("")

    Object.defineProperty(window, "innerWidth", { configurable: true, value: window.innerWidth - 50 })
    window.dispatchEvent(new Event("resize"))

    expect(panelEl()).toHaveClass("expanded")
    expect((panelEl() as HTMLElement).style.width).toBe("")
  })

  it("stays expanded across a close/reopen within the same page view", () => {
    // No persistence for this (unlike width): a JS variable that's simply
    // never reset by closePanel(), so it only resets on an actual reload.
    runWidget()
    fabEl()?.click()
    fromIframe("widget_expand")

    fabEl()?.click()
    fabEl()?.click()

    expect(panelEl()).toHaveClass("expanded")
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

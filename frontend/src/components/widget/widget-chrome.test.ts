import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { MAX_INITIAL_MESSAGE_LENGTH, type WidgetParentMessageType } from "@/lib/widget-parent-message"

import { evaluateWidgetScript, widgetScript } from "@/test-utils/widget-script"

const HOST = "https://chat.example"

function runWidget(attributes: Record<string, string> = { "data-widget-key": "widget-secret" }) {
  const script = document.createElement("script")
  script.src = `${HOST}/widget.js`
  for (const [name, value] of Object.entries(attributes)) {
    script.setAttribute(name, value)
  }
  document.body.appendChild(script)
  Object.defineProperty(document, "currentScript", { configurable: true, value: script })
  evaluateWidgetScript()
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
    const messageTypes: WidgetParentMessageType[] = ["widget_close", "widget_expand", "widget_collapse", "prompt_context_request", "initial_message_request", "initial_message_ack", "open_message_request", "open_message_ack"]
    for (const messageType of messageTypes) {
      expect(widgetScript).toContain(`data.type === '${messageType}'`)
    }
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

  describe("data-initial-message", () => {
    const initialMessage = (value: string) => (
      { xagent: true, v: 1, type: "initial_message", value }
    )

    function runWithInitialMessage(value = "  hello agent  ") {
      runWidget({ "data-widget-key": "widget-secret", "data-initial-message": value })
      return vi.spyOn(iframeEl()!.contentWindow!, "postMessage")
    }

    it("holds the message until the panel is first opened", () => {
      const postToIframe = runWithInitialMessage()

      fromIframe("initial_message_request")
      expect(postToIframe).not.toHaveBeenCalled()

      fabEl()?.click()
      expect(postToIframe).toHaveBeenCalledTimes(1)
      expect(postToIframe).toHaveBeenCalledWith(initialMessage("hello agent"), HOST)
    })

    it("replies at once when the panel was opened before the request", () => {
      const postToIframe = runWithInitialMessage()
      fabEl()?.click()

      fromIframe("initial_message_request")

      expect(postToIframe).toHaveBeenCalledWith(initialMessage("hello agent"), HOST)
    })

    it("answers a repeated request until the iframe acknowledges, then never again", () => {
      const postToIframe = runWithInitialMessage()
      fromIframe("initial_message_request")
      // The iframe's listener went away (not ready) before the first open.
      fabEl()?.click()
      expect(postToIframe).toHaveBeenCalledTimes(1)

      // Closing and reopening does not repost without a new request.
      fabEl()?.click()
      fabEl()?.click()
      expect(postToIframe).toHaveBeenCalledTimes(1)

      fromIframe("initial_message_request")
      expect(postToIframe).toHaveBeenCalledTimes(2)

      fromIframe("initial_message_ack")
      fromIframe("initial_message_request")
      fabEl()?.click()
      fabEl()?.click()
      expect(postToIframe).toHaveBeenCalledTimes(2)
    })

    it("accepts a message of exactly the length cap", () => {
      const message = "x".repeat(2000)
      const postToIframe = runWithInitialMessage(message)
      fabEl()?.click()

      fromIframe("initial_message_request")

      expect(postToIframe).toHaveBeenCalledWith(initialMessage(message), HOST)
    })

    it("keeps the length cap in step with the iframe side", () => {
      expect(widgetScript).toContain(`var MAX_INITIAL_MESSAGE_LENGTH = ${MAX_INITIAL_MESSAGE_LENGTH};`)
    })

    it("reads the attribute at delivery, not at load", () => {
      const postToIframe = runWithInitialMessage("early")
      fromIframe("initial_message_request")

      document.currentScript!.setAttribute("data-initial-message", "late")
      fabEl()?.click()

      expect(postToIframe).toHaveBeenCalledWith(initialMessage("late"), HOST)
    })

    it("sends nothing without the attribute or with a blank one", () => {
      runWidget()
      const postWithout = vi.spyOn(iframeEl()!.contentWindow!, "postMessage")
      fabEl()?.click()
      fromIframe("initial_message_request")
      expect(postWithout).not.toHaveBeenCalled()

      document.currentScript!.setAttribute("data-initial-message", "   ")
      fromIframe("initial_message_request")
      expect(postWithout).not.toHaveBeenCalled()
    })

    it("warns once and sends nothing for an over-long message", () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {})
      const postToIframe = runWithInitialMessage("x".repeat(2001))
      fabEl()?.click()

      fromIframe("initial_message_request")
      fromIframe("initial_message_request")

      expect(postToIframe).not.toHaveBeenCalled()
      expect(warn).toHaveBeenCalledTimes(1)
    })

    it("ignores a request from a different source or origin", () => {
      const postToIframe = runWithInitialMessage()
      fabEl()?.click()

      window.dispatchEvent(new MessageEvent("message", {
        data: { xagent: true, v: 1, type: "initial_message_request" },
        origin: HOST,
        source: window,
      }))
      window.dispatchEvent(new MessageEvent("message", {
        data: { xagent: true, v: 1, type: "initial_message_request" },
        origin: "https://evil.example",
        source: iframeEl()?.contentWindow as Window,
      }))

      expect(postToIframe).not.toHaveBeenCalled()
    })
  })

  it("replies to a prompt_context_request with the parsed attribute values", () => {
    runWidget({
      "data-widget-key": "widget-secret",
      "data-prompt-context": '{"product.sku":"A-100","customer.tier":"vip"}',
    })
    const postToIframe = vi.spyOn(iframeEl()!.contentWindow!, "postMessage")

    fromIframe("prompt_context_request")

    expect(postToIframe).toHaveBeenCalledWith(
      {
        xagent: true,
        v: 1,
        type: "prompt_context",
        values: { "product.sku": "A-100", "customer.tier": "vip" },
      },
      HOST,
    )
  })

  it("reads data-prompt-context when the request arrives, not at load", () => {
    runWidget()
    const postToIframe = vi.spyOn(iframeEl()!.contentWindow!, "postMessage")

    document.currentScript!.setAttribute("data-prompt-context", '{"late":"value"}')
    fromIframe("prompt_context_request")

    expect(postToIframe).toHaveBeenCalledWith(
      { xagent: true, v: 1, type: "prompt_context", values: { late: "value" } },
      HOST,
    )
  })

  it("warns once and replies values: null for invalid JSON or an array", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {})
    runWidget({ "data-widget-key": "widget-secret", "data-prompt-context": "not json" })
    const postToIframe = vi.spyOn(iframeEl()!.contentWindow!, "postMessage")

    fromIframe("prompt_context_request")
    document.currentScript!.setAttribute("data-prompt-context", '["A-100"]')
    fromIframe("prompt_context_request")

    expect(postToIframe).toHaveBeenCalledTimes(2)
    expect(postToIframe.mock.calls[0][0]).toMatchObject({ values: null })
    expect(postToIframe.mock.calls[1][0]).toMatchObject({ values: null })
    expect(warn).toHaveBeenCalledTimes(1)
    expect(warn.mock.calls[0][0]).toContain("Xagent Widget:")
  })

  it("replies values: null for a missing data-prompt-context attribute", () => {
    runWidget()
    const postToIframe = vi.spyOn(iframeEl()!.contentWindow!, "postMessage")

    fromIframe("prompt_context_request")

    expect(postToIframe).toHaveBeenCalledWith(
      { xagent: true, v: 1, type: "prompt_context", values: null },
      HOST,
    )
  })

  it("never replies in session mode", () => {
    runWidget({
      "data-encrypted-context": "grant",
      "data-prompt-context": '{"product.sku":"A-100"}',
    })
    const postToIframe = vi.spyOn(iframeEl()!.contentWindow!, "postMessage")

    fromIframe("prompt_context_request")

    expect(postToIframe).not.toHaveBeenCalled()
  })

  it("ignores a prompt_context_request from a different source or origin", () => {
    runWidget({ "data-widget-key": "widget-secret", "data-prompt-context": '{"a":"b"}' })
    const postToIframe = vi.spyOn(iframeEl()!.contentWindow!, "postMessage")

    window.dispatchEvent(new MessageEvent("message", {
      data: { xagent: true, v: 1, type: "prompt_context_request" },
      origin: HOST,
      source: window,
    }))
    window.dispatchEvent(new MessageEvent("message", {
      data: { xagent: true, v: 1, type: "prompt_context_request" },
      origin: "https://evil.example",
      source: iframeEl()!.contentWindow,
    }))

    expect(postToIframe).not.toHaveBeenCalled()
  })

  it("never puts prompt context values in the iframe URL", async () => {
    runWidget({
      "data-widget-key": "widget-secret",
      "data-prompt-context": '{"product.sku":"SKU-IN-URL-SENTINEL"}',
    })

    await vi.waitFor(() => {
      expect(iframeEl()!.src).toContain("embed_ticket=t")
    })
    expect(iframeEl()!.src).not.toContain("SKU-IN-URL-SENTINEL")
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
    //
    // A non-default data-button-size is used deliberately: both the base
    // max-height and the expanded height are button-size-derived, and the
    // default 60px happens to make the old hardcoded "100px" offset produce
    // the same result (60px + 40px = 100px) -- so only a distinguishing size
    // can catch a regression back to that hardcode.
    runWidget({ "data-widget-key": "widget-secret", "data-button-size": "100px" })

    const block = expandedBlock()
    expect(block).toMatch(/\.xagent-widget-panel\.expanded\s*\{[^}]*width:\s*min\(720px, 100vw - 40px\);/)
    expect(block).toMatch(
      /\.xagent-widget-panel\.expanded\s*\{[^}]*height:\s*calc\(100vh - 100px - 40px\);/,
    )
    expect(block).toMatch(/\.xagent-widget-panel\.expanded\s+\.xagent-widget-resize-handle\s*\{[^}]*display:\s*none;/)
    expect(mobileBlock()).not.toMatch(/\.xagent-widget-panel\.expanded/)
    expect(styleText()).toMatch(/max-height:\s*calc\(100vh - 100px - 40px\);/)
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
    const postToIframe = vi.spyOn(iframeEl()!.contentWindow!, "postMessage")

    Object.defineProperty(window, "innerWidth", { configurable: true, value: window.innerWidth - 50 })
    window.dispatchEvent(new Event("resize"))

    expect(panelEl()).toHaveClass("expanded")
    expect((panelEl() as HTMLElement).style.width).toBe("")
    // Still on the desktop side of the breakpoint -- nothing to correct.
    expect(postToIframe).not.toHaveBeenCalled()
  })

  it("force-collapses an already-expanded panel when a resize carries it across the mobile breakpoint", () => {
    // The .expanded CSS rule is scoped above the mobile breakpoint, so
    // nothing previously re-validated isExpanded against a resize the way
    // expandPanel() already does for a fresh click -- the panel would
    // render mobile-sized while isExpanded stayed true on both sides, then
    // silently snap back to full size if the window later widened past the
    // breakpoint again with no further click from the visitor.
    runWidget()
    fabEl()?.click()
    fromIframe("widget_expand")
    expect(panelEl()).toHaveClass("expanded")
    const postToIframe = vi.spyOn(iframeEl()!.contentWindow!, "postMessage")

    Object.defineProperty(window, "innerWidth", { configurable: true, value: 400 })
    window.dispatchEvent(new Event("resize"))

    expect(panelEl()).not.toHaveClass("expanded")
    expect(postToIframe).toHaveBeenCalledWith(
      { xagent: true, v: 1, type: "widget_expand_rejected" },
      HOST,
    )

    // Widening back past the breakpoint afterward must not silently
    // re-expand the panel with no further click.
    Object.defineProperty(window, "innerWidth", { configurable: true, value: 1024 })
    window.dispatchEvent(new Event("resize"))

    expect(panelEl()).not.toHaveClass("expanded")
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

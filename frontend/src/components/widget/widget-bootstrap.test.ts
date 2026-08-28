import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { pathToFileURL } from "node:url"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const widgetScriptPath = resolve(process.cwd(), "public/widget.js")
const widgetScript = readFileSync(
  widgetScriptPath,
  "utf8",
)
const widgetScriptUrl = pathToFileURL(widgetScriptPath).href

function runWidget(attributes: Record<string, string>) {
  const script = document.createElement("script")
  script.src = "https://chat.example/widget.js"
  for (const [name, value] of Object.entries(attributes)) {
    script.setAttribute(name, value)
  }
  document.body.appendChild(script)
  Object.defineProperty(document, "currentScript", {
    configurable: true,
    value: script,
  })

  window.eval(`${widgetScript}\n//# sourceURL=${widgetScriptUrl}`)
}

describe("widget bootstrap", () => {
  const fetchMock = vi.fn()
  let currentScriptDescriptor: PropertyDescriptor | undefined

  beforeEach(() => {
    currentScriptDescriptor = Object.getOwnPropertyDescriptor(document, "currentScript")
    document.head.innerHTML = ""
    document.body.innerHTML = ""
    // innerHTML resets don't touch inline styles on body itself, and it's
    // shared across every test in this file (not just the panel-resize
    // describe below) -- reset here, not just in a nested beforeEach, so any
    // test anywhere in the file that touches these can't leak into another.
    document.body.style.userSelect = ""
    document.body.style.cursor = ""
    localStorage.setItem("xagent_guest_id", "guest-fixed")
    vi.stubGlobal("fetch", fetchMock)
    fetchMock.mockReset()
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

  it("fails closed without a widget key on the default token channel", () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)

    runWidget({})

    expect(fetchMock).not.toHaveBeenCalled()
    expect(document.querySelector(".xagent-widget-container")).toBeNull()
    expect(document.head.querySelector("style")).toBeNull()
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("Missing data-widget-key"))
  })

  it("generates and persists a guest id for a first-time visitor", async () => {
    localStorage.removeItem("xagent_guest_id")
    vi.spyOn(Math, "random")
      .mockReturnValueOnce(0.123456)
      .mockReturnValueOnce(0.654321)
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({
      ticket: "ticket/one",
      agent_id: 17,
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }))

    runWidget({ "data-widget-key": "widget-secret" })

    const guestId = localStorage.getItem("xagent_guest_id")
    expect(guestId).toBe("guest_4fzyo82mvyqnk000qvgin")
    await vi.waitFor(() => {
      expect(document.querySelector<HTMLIFrameElement>(".xagent-widget-iframe")?.src).toBe(
        `https://chat.example/widget/chat/default?guest_id=${guestId}&agent_id=17&embed_ticket=ticket%2Fone`,
      )
    })
  })

  it("loads an embed-ticket iframe URL without exposing the widget key", async () => {
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({
      ticket: "ticket/one",
      agent_id: 17,
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }))

    runWidget({ "data-widget-key": "widget-secret" })

    await vi.waitFor(() => {
      expect(document.querySelector<HTMLIFrameElement>(".xagent-widget-iframe")?.src).toBe(
        "https://chat.example/widget/chat/default?guest_id=guest-fixed&agent_id=17&embed_ticket=ticket%2Fone",
      )
    })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "https://chat.example/api/widget/embed-ticket",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ widget_key: "widget-secret" }),
      },
    )
    expect(document.querySelector<HTMLIFrameElement>(".xagent-widget-iframe")?.src)
      .not.toContain("widget-secret")
  })

  it("does not navigate the iframe on a non-OK embed-ticket response", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockResolvedValueOnce(new Response("forbidden", { status: 403 }))

    runWidget({ "data-widget-key": "widget-secret" })

    await vi.waitFor(() => {
      expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("embed authorization failed"))
    })
    expect(document.querySelector<HTMLIFrameElement>(".xagent-widget-iframe")?.getAttribute("src")).toBeNull()
  })

  it("does not navigate the iframe on an embed-ticket network failure", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockRejectedValueOnce(new Error("network unavailable"))

    runWidget({ "data-widget-key": "widget-secret" })

    await vi.waitFor(() => {
      expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("embed authorization request failed"))
    })
    expect(document.querySelector<HTMLIFrameElement>(".xagent-widget-iframe")?.getAttribute("src")).toBeNull()
  })

  it("keeps the deprecated non-default token channel available without a ticket", () => {
    runWidget({ "data-token": "legacy-token" })

    expect(fetchMock).not.toHaveBeenCalled()
    expect(document.querySelector<HTMLIFrameElement>(".xagent-widget-iframe")?.src).toBe(
      "https://chat.example/widget/chat/legacy-token?guest_id=guest-fixed",
    )
  })

  describe("panel resize", () => {
    let originalInnerWidth: number

    beforeEach(() => {
      originalInnerWidth = window.innerWidth
      // userSelect/cursor are reset in the outer beforeEach above.
      fetchMock.mockResolvedValue(new Response(JSON.stringify({
        ticket: "ticket/one",
        agent_id: 17,
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }))
    })

    afterEach(() => {
      setInnerWidth(originalInnerWidth)
    })

    function setInnerWidth(value: number) {
      Object.defineProperty(window, "innerWidth", { configurable: true, value })
    }

    function panel() {
      return document.querySelector<HTMLDivElement>(".xagent-widget-panel")!
    }

    function handle() {
      return document.querySelector<HTMLDivElement>(".xagent-widget-resize-handle")!
    }

    function openPanel() {
      // The handle is only draggable while the panel carries the 'open'
      // class (see the pointerdown guard in widget.js) -- click the FAB,
      // the same real-world path a user takes, rather than poking the class
      // directly.
      document.querySelector<HTMLButtonElement>(".xagent-widget-fab")!.click()
    }

    function widgetIframe() {
      return document.querySelector<HTMLIFrameElement>(".xagent-widget-iframe")!
    }

    function firePointerEvent(
      target: EventTarget,
      type: string,
      init: { pointerId: number; clientX: number; button?: number },
    ) {
      // jsdom has no PointerEvent constructor; a plain Event with the fields
      // the widget reads is enough since addEventListener matches by type.
      const event = new MouseEvent(type, {
        bubbles: true,
        cancelable: true,
        clientX: init.clientX,
        button: init.button ?? 0,
      })
      Object.defineProperty(event, "pointerId", { value: init.pointerId })
      target.dispatchEvent(event)
      return event
    }

    it("applies the default width on a fresh visit", () => {
      runWidget({ "data-widget-key": "widget-secret" })
      expect(panel().style.width).toBe("380px")
    })

    it("applies a persisted width from localStorage", () => {
      localStorage.setItem("xagent_widget_width", "500")
      runWidget({ "data-widget-key": "widget-secret" })
      expect(panel().style.width).toBe("500px")
    })

    it("falls back to the default width for a corrupt stored value", () => {
      localStorage.setItem("xagent_widget_width", "not-a-number")
      runWidget({ "data-widget-key": "widget-secret" })
      expect(panel().style.width).toBe("380px")
    })

    it("falls back to the default width for an out-of-range stored value", () => {
      localStorage.setItem("xagent_widget_width", "99999")
      runWidget({ "data-widget-key": "widget-secret" })
      expect(panel().style.width).toBe("380px")
    })

    it("falls back to the default width for a stored value below the minimum", () => {
      // Only the upper bound was covered above; MIN_PANEL_WIDTH is a
      // separate comparison in the same guard.
      localStorage.setItem("xagent_widget_width", "100")
      runWidget({ "data-widget-key": "widget-secret" })
      expect(panel().style.width).toBe("380px")
    })

    it("accepts a stored value at exactly the minimum boundary", () => {
      // readStoredWidth rejects strictly below MIN_PANEL_WIDTH (320) -- pin
      // that the boundary value itself is accepted, not folded into the
      // rejected range by an off-by-one.
      localStorage.setItem("xagent_widget_width", "320")
      runWidget({ "data-widget-key": "widget-secret" })
      expect(panel().style.width).toBe("320px")
    })

    it("accepts a stored value at exactly the maximum boundary", () => {
      // Same as above for the upper MAX_PANEL_WIDTH (720) boundary.
      localStorage.setItem("xagent_widget_width", "720")
      runWidget({ "data-widget-key": "widget-secret" })
      expect(panel().style.width).toBe("720px")
    })

    it("falls back to the default width when reading storage throws", () => {
      const realGetItem = window.localStorage.getItem.bind(window.localStorage)
      vi.spyOn(window.localStorage, "getItem").mockImplementation((key) => {
        if (key === "xagent_widget_width") throw new Error("blocked")
        return realGetItem(key)
      })
      runWidget({ "data-widget-key": "widget-secret" })
      expect(panel().style.width).toBe("380px")
    })

    it("round-trips a dragged width through storage into a fresh load", () => {
      // Every other persistence test either seeds storage directly or checks
      // it after a drag -- neither proves a value a drag itself wrote is
      // actually what the *next* page load reads back. Simulate that next
      // load by tearing down this instance's DOM and re-running the
      // bootstrap script, the same way a real page refresh would.
      // The describe-level mock resolves every call with the same Response
      // instance, whose body a single fetch already consumes -- this test
      // fetches twice (once per runWidget), so it needs a fresh Response per
      // call instead.
      fetchMock.mockImplementation(() => Promise.resolve(new Response(JSON.stringify({
        ticket: "ticket/one",
        agent_id: 17,
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })))
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()

      firePointerEvent(handle(), "pointerdown", { pointerId: 10, clientX: 300 })
      firePointerEvent(handle(), "pointermove", { pointerId: 10, clientX: 200 })
      expect(panel().style.width).toBe("480px")
      firePointerEvent(handle(), "pointerup", { pointerId: 10, clientX: 200 })
      expect(localStorage.getItem("xagent_widget_width")).toBe("480")

      document.head.innerHTML = ""
      document.body.innerHTML = ""
      runWidget({ "data-widget-key": "widget-secret" })
      expect(panel().style.width).toBe("480px")
    })

    it("re-clamps to the viewport on resize without losing the stored preference", () => {
      localStorage.setItem("xagent_widget_width", "600")
      runWidget({ "data-widget-key": "widget-secret" })
      expect(panel().style.width).toBe("600px")

      setInnerWidth(500)
      window.dispatchEvent(new Event("resize"))
      expect(panel().style.width).toBe("460px")

      setInnerWidth(1024)
      window.dispatchEvent(new Event("resize"))
      expect(panel().style.width).toBe("600px")
    })

    it("does not corrupt a stored preference wider than the load-time viewport", () => {
      localStorage.setItem("xagent_widget_width", "700")
      setInnerWidth(600) // viewportMax = 560, so the initial render clamps the display
      runWidget({ "data-widget-key": "widget-secret" })
      expect(panel().style.width).toBe("560px")

      // A click-and-release with no movement must not persist the render
      // clamp over the raw stored preference.
      openPanel()
      firePointerEvent(handle(), "pointerdown", { pointerId: 6, clientX: 300 })
      firePointerEvent(handle(), "pointerup", { pointerId: 6, clientX: 300 })
      expect(localStorage.getItem("xagent_widget_width")).toBe("700")

      // Widening the viewport back must recover the full preference, not the
      // narrower value that was ever rendered.
      setInnerWidth(1024)
      window.dispatchEvent(new Event("resize"))
      expect(panel().style.width).toBe("700px")
    })

    it("does not lose a wider preference to a drag that never escapes the viewport ceiling", () => {
      localStorage.setItem("xagent_widget_width", "700")
      setInnerWidth(600) // viewportMax = 560
      runWidget({ "data-widget-key": "widget-secret" })
      expect(panel().style.width).toBe("560px")

      // A 1px nudge attempting to widen further is fully absorbed by the
      // ceiling -- the rendered width never actually changes, unlike a true
      // zero-movement release.
      openPanel()
      firePointerEvent(handle(), "pointerdown", { pointerId: 11, clientX: 300 })
      firePointerEvent(handle(), "pointermove", { pointerId: 11, clientX: 299 })
      expect(panel().style.width).toBe("560px")
      firePointerEvent(handle(), "pointerup", { pointerId: 11, clientX: 299 })

      // The 700px preference must survive this drag, not get silently
      // replaced by the 560px ceiling it never actually moved past.
      expect(localStorage.getItem("xagent_widget_width")).toBe("700")
      setInnerWidth(1024)
      window.dispatchEvent(new Event("resize"))
      expect(panel().style.width).toBe("700px")
    })

    it("does adopt a genuinely smaller width when a drag visibly shrinks past the ceiling", () => {
      localStorage.setItem("xagent_widget_width", "700")
      setInnerWidth(600) // viewportMax = 560
      runWidget({ "data-widget-key": "widget-secret" })

      // Drag right (shrink) past the ceiling: a real, visible size change,
      // as opposed to the previous test's absorbed widening attempt.
      openPanel()
      firePointerEvent(handle(), "pointerdown", { pointerId: 12, clientX: 300 })
      firePointerEvent(handle(), "pointermove", { pointerId: 12, clientX: 350 })
      expect(panel().style.width).toBe("510px") // 560 - 50, a real shrink
      firePointerEvent(handle(), "pointerup", { pointerId: 12, clientX: 350 })

      expect(localStorage.getItem("xagent_widget_width")).toBe("510")
    })

    it("hides the inline width below the mobile breakpoint and restores it above", () => {
      runWidget({ "data-widget-key": "widget-secret" })

      setInnerWidth(400)
      window.dispatchEvent(new Event("resize"))
      expect(panel().style.width).toBe("")

      setInnerWidth(1024)
      window.dispatchEvent(new Event("resize"))
      expect(panel().style.width).toBe("380px")
    })

    it("declares the mobile media rule that actually hides the resize handle", () => {
      // jsdom doesn't evaluate CSS, so the inline-width behavior above can't
      // prove the stylesheet itself still hides the handle below the
      // breakpoint -- assert the injected rule text directly, so weakening
      // or dropping that declaration fails here even though it wouldn't
      // change any inline style the other tests observe.
      runWidget({ "data-widget-key": "widget-secret" })
      const css = document.head.querySelector("style")!.textContent!

      expect(css).toMatch(/@media\s*\(max-width:\s*480px\)/)
      const mobileBlock = css.slice(css.indexOf("@media"))
      expect(mobileBlock).toMatch(/\.xagent-widget-resize-handle\s*{[^}]*display:\s*none/)
    })

    it("declares the closed-panel rule that hides the resize handle without JS", () => {
      // Same rationale as the mobile-rule test above: the JS-level
      // panel.classList.contains('open') check in the pointerdown guard is
      // defense in depth, not the primary mechanism -- this stylesheet rule
      // is what actually keeps the handle un-hit-testable while the panel is
      // closed on a real page.
      runWidget({ "data-widget-key": "widget-secret" })
      const css = document.head.querySelector("style")!.textContent!

      expect(css).toMatch(
        /\.xagent-widget-panel:not\(\.open\)\s*\.xagent-widget-resize-handle\s*{[^}]*display:\s*none/,
      )
    })

    it("ignores a drag start below the mobile breakpoint", () => {
      setInnerWidth(400)
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()

      // Isolate the mobile-viewport guard specifically: with the panel open,
      // only isMobileViewport() can be blocking this pointerdown.
      firePointerEvent(handle(), "pointerdown", { pointerId: 1, clientX: 100 })

      expect(document.body.style.userSelect).toBe("")
      expect(document.body.style.cursor).toBe("")
      expect(widgetIframe().style.pointerEvents).toBe("")
      // Proves dragState was never created, not just that these three
      // particular side effects happen to be unset.
      firePointerEvent(handle(), "pointermove", { pointerId: 1, clientX: 250 })
      expect(panel().style.width).toBe("")
    })

    it("boundary: applies the mobile CSS at exactly the breakpoint width, not just below it", () => {
      // isMobileViewport() uses <=, matching the CSS media query's own
      // max-width: 480px (also <= in CSS terms) -- off by one here would
      // mean inline width wins over the media query by specificity right at
      // this exact width, which is the one case the whole mobile branch
      // exists to prevent (see the comment above applyPanelWidth). 480 is
      // MOBILE_BREAKPOINT's value, not importable from widget.js's IIFE
      // closure -- the CSS-rule-content test above already pins that the
      // constant itself is 480, so hardcoding it here is safe.
      setInnerWidth(480)
      runWidget({ "data-widget-key": "widget-secret" })

      expect(panel().style.width).toBe("")
    })

    it("resizes the panel while dragging and persists the final width on release", () => {
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()
      document.body.style.userSelect = "text"

      firePointerEvent(handle(), "pointerdown", { pointerId: 7, clientX: 300 })
      expect(document.body.style.userSelect).toBe("none")
      expect(widgetIframe().style.pointerEvents).toBe("none")

      // startWidth is read from computed style, which reflects the panel's
      // width at drag start (380px, the default applied on load).
      firePointerEvent(handle(), "pointermove", { pointerId: 7, clientX: 400 })
      expect(panel().style.width).toBe("320px") // 380 - 100 = 280, clamped to MIN_PANEL_WIDTH

      firePointerEvent(handle(), "pointermove", { pointerId: 7, clientX: 180 })
      expect(panel().style.width).toBe("500px") // 380 + 120 = 500, within bounds

      firePointerEvent(handle(), "pointerup", { pointerId: 7, clientX: 180 })
      expect(document.body.style.userSelect).toBe("text")
      expect(widgetIframe().style.pointerEvents).toBe("")
      expect(localStorage.getItem("xagent_widget_width")).toBe("500")
    })

    it("clamps a drag to the static MAX_PANEL_WIDTH, not just the viewport ceiling", () => {
      // The other ceiling-clamp tests all clamp against a narrowed viewport
      // (viewportMax below 720) -- this one keeps the default wide viewport
      // (viewportMax = 984) so the static 720px cap itself is what stops the
      // drag, not window width.
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()

      firePointerEvent(handle(), "pointerdown", { pointerId: 9, clientX: 300 })
      firePointerEvent(handle(), "pointermove", { pointerId: 9, clientX: -100 }) // 380 + 400 = 780, capped to 720
      expect(panel().style.width).toBe("720px")

      firePointerEvent(handle(), "pointerup", { pointerId: 9, clientX: -100 })
      expect(localStorage.getItem("xagent_widget_width")).toBe("720")
    })

    it("commits the width the drag actually ends at, not the widest point it passed through", () => {
      // A very ordinary gesture: overshoot while dragging, then correct back
      // -- while still wider than the start, i.e. never triggering the old
      // "shrunk past the ceiling" escape hatch. The committed width must
      // match what's actually rendered when the pointer is released.
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()

      firePointerEvent(handle(), "pointerdown", { pointerId: 8, clientX: 300 })
      firePointerEvent(handle(), "pointermove", { pointerId: 8, clientX: 180 }) // overshoot to 500px
      expect(panel().style.width).toBe("500px")
      firePointerEvent(handle(), "pointermove", { pointerId: 8, clientX: 280 }) // correct back to 400px
      expect(panel().style.width).toBe("400px")

      firePointerEvent(handle(), "pointerup", { pointerId: 8, clientX: 280 })
      expect(panel().style.width).toBe("400px") // unchanged by the commit itself
      expect(localStorage.getItem("xagent_widget_width")).toBe("400") // not the passed-through 500
    })

    it("continues tracking the drag when pointermove/pointerup arrive on document rather than the handle", () => {
      // jsdom has no PointerEvent/setPointerCapture at all, so every other
      // test in this file dispatching pointermove/pointerup directly on
      // handle() can't distinguish "capture is retargeting these events"
      // from "these listeners just happen to live on the same element".
      // Real Pointer Capture would retarget events to the handle
      // regardless of what's under the cursor; bypass that entirely here by
      // dispatching where an event would actually land if capture were
      // unavailable and the cursor were, say, over the iframe.
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()

      firePointerEvent(handle(), "pointerdown", { pointerId: 27, clientX: 300 })
      firePointerEvent(document, "pointermove", { pointerId: 27, clientX: 200 })
      expect(panel().style.width).toBe("480px")

      firePointerEvent(document, "pointerup", { pointerId: 27, clientX: 200 })
      expect(localStorage.getItem("xagent_widget_width")).toBe("480")
    })

    it("releases pointer capture when a drag is cancelled, not just on a real release", () => {
      // Browsers auto-release capture on pointerup/pointercancel, but not on
      // blur or a plain resize -- restoreDragSideEffects releases it
      // explicitly for those paths. jsdom doesn't implement (release)
      // PointerCapture at all, so stub it directly on the handle instance to
      // observe the call (production code already guards the call in a
      // try/catch for exactly this kind of absence).
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()
      const releaseSpy = vi.fn()
      handle().releasePointerCapture = releaseSpy

      firePointerEvent(handle(), "pointerdown", { pointerId: 28, clientX: 300 })
      window.dispatchEvent(new Event("blur")) // cancels, no pointerup/pointercancel ever fires

      expect(releaseSpy).toHaveBeenCalledWith(28)
    })

    it("cancels rather than commits on pointercancel, reverting the in-progress width", () => {
      // pointercancel is an involuntary interruption (a touch gesture
      // reinterpreted as a scroll, an OS-level interrupt), not the user
      // confirming a release -- it must not persist wherever the drag had
      // gotten to.
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()

      firePointerEvent(handle(), "pointerdown", { pointerId: 2, clientX: 300 })
      firePointerEvent(handle(), "pointermove", { pointerId: 2, clientX: 250 }) // mid-drag, unconfirmed
      expect(panel().style.width).toBe("430px")

      firePointerEvent(handle(), "pointercancel", { pointerId: 2, clientX: 250 })

      expect(document.body.style.userSelect).toBe("")
      expect(widgetIframe().style.pointerEvents).toBe("")
      expect(panel().style.width).toBe("380px") // reverted, not left at the abandoned 430px
      expect(localStorage.getItem("xagent_widget_width")).toBeNull()
    })

    it("does not let a cancelled drag's abandoned width survive to be committed by a later no-op drag", () => {
      // A cancelled drag must never have touched the committed panelWidth in
      // the first place -- otherwise the abandoned in-progress value stays
      // the module's current width, and an unrelated zero-movement drag
      // later on would commit it via a normal release.
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()

      firePointerEvent(handle(), "pointerdown", { pointerId: 21, clientX: 300 })
      firePointerEvent(handle(), "pointermove", { pointerId: 21, clientX: 250 }) // -> 430px, abandoned below
      window.dispatchEvent(new Event("blur")) // cancels: 430 must never reach panelWidth
      expect(panel().style.width).toBe("380px")

      // An unrelated click-and-release with zero movement -- a true no-op,
      // so it persists nothing at all rather than re-affirming 380.
      firePointerEvent(handle(), "pointerdown", { pointerId: 22, clientX: 300 })
      firePointerEvent(handle(), "pointerup", { pointerId: 22, clientX: 300 })

      expect(localStorage.getItem("xagent_widget_width")).not.toBe("430")
      expect(localStorage.getItem("xagent_widget_width")).toBeNull()
    })

    it("restores the host page's own cursor rather than clearing it", () => {
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()
      document.body.style.cursor = "help"

      firePointerEvent(handle(), "pointerdown", { pointerId: 23, clientX: 300 })
      expect(document.body.style.cursor).toBe("ew-resize")

      firePointerEvent(handle(), "pointerup", { pointerId: 23, clientX: 300 })
      expect(document.body.style.cursor).toBe("help")
    })

    it("ignores a non-primary pointer button", () => {
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()

      // Isolate the button guard: with the panel open, only a non-primary
      // button can be blocking this pointerdown.
      firePointerEvent(handle(), "pointerdown", { pointerId: 24, clientX: 300, button: 2 })

      expect(document.body.style.userSelect).toBe("")
      expect(document.body.style.cursor).toBe("")
      expect(widgetIframe().style.pointerEvents).toBe("")
      // Proves dragState itself was never created, not just that these three
      // particular side effects happen to be unset: if the button guard were
      // moved after dragState's assignment, this assertion alone wouldn't
      // catch it, but a subsequent move actually changing the width would.
      firePointerEvent(handle(), "pointermove", { pointerId: 24, clientX: 100 })
      expect(panel().style.width).toBe("380px")
    })

    it("ignores a second concurrent pointer without disrupting the first pointer's drag", () => {
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()

      firePointerEvent(handle(), "pointerdown", { pointerId: 1, clientX: 300 })
      firePointerEvent(handle(), "pointerdown", { pointerId: 2, clientX: 100 })
      firePointerEvent(handle(), "pointermove", { pointerId: 2, clientX: 0 })
      expect(panel().style.width).toBe("380px") // untouched: pointer 2 was never tracked

      firePointerEvent(handle(), "pointerup", { pointerId: 2, clientX: 0 })
      expect(document.body.style.userSelect).toBe("none") // pointer 1's drag is still active

      firePointerEvent(handle(), "pointermove", { pointerId: 1, clientX: 250 })
      expect(panel().style.width).toBe("430px") // 380 + 50, from pointer 1's own startWidth

      firePointerEvent(handle(), "pointerup", { pointerId: 1, clientX: 250 })
      expect(document.body.style.userSelect).toBe("")
      expect(localStorage.getItem("xagent_widget_width")).toBe("430")
    })

    it("cancels (without persisting) an interrupted drag on window blur", () => {
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()

      firePointerEvent(handle(), "pointerdown", { pointerId: 3, clientX: 300 })
      firePointerEvent(handle(), "pointermove", { pointerId: 3, clientX: 250 }) // mid-drag, unconfirmed
      expect(document.body.style.cursor).toBe("ew-resize")

      window.dispatchEvent(new Event("blur"))

      expect(document.body.style.userSelect).toBe("")
      expect(document.body.style.cursor).toBe("")
      expect(widgetIframe().style.pointerEvents).toBe("")
      // A blur interruption is not the user confirming a width via release --
      // nothing was ever persisted in this test.
      expect(localStorage.getItem("xagent_widget_width")).toBeNull()
      // Reverted, not left at the abandoned 430px (380 + 50 from the move above).
      expect(panel().style.width).toBe("380px")

      // The cancelled drag is really over: further movement on the same
      // pointer does nothing, regardless of where it goes.
      firePointerEvent(handle(), "pointermove", { pointerId: 3, clientX: 100 })
      expect(panel().style.width).toBe("380px")
    })

    it("ignores a blur caused by focus moving into the widget's own iframe", () => {
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()
      vi.spyOn(document, "hasFocus").mockReturnValue(true)

      firePointerEvent(handle(), "pointerdown", { pointerId: 17, clientX: 300 })
      expect(document.body.style.userSelect).toBe("none")

      window.dispatchEvent(new Event("blur"))

      // document.hasFocus() still true means the window itself never lost
      // focus -- this blur doesn't interrupt the drag.
      expect(document.body.style.userSelect).toBe("none")

      firePointerEvent(handle(), "pointerup", { pointerId: 17, clientX: 300 })
      expect(document.body.style.userSelect).toBe("")
    })

    it("cancels an active drag on a genuine viewport change and re-renders for the new viewport", () => {
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()

      firePointerEvent(handle(), "pointerdown", { pointerId: 18, clientX: 300 })
      firePointerEvent(handle(), "pointermove", { pointerId: 18, clientX: 100 }) // -> 580px, abandoned below
      expect(panel().style.width).toBe("580px")
      expect(document.body.style.userSelect).toBe("none")

      // A real viewport change (not just a bare resize event) invalidates
      // this drag's frozen startX/startWidth anchors -- rather than let the
      // next pointermove misread the viewport's own movement as the user
      // shrinking the panel, the drag is cancelled outright, and the panel
      // re-renders for the *new* viewport off the untouched preference
      // (380), not a stale snapshot of the old render or the abandoned 580.
      setInnerWidth(350) // now below the mobile breakpoint
      window.dispatchEvent(new Event("resize"))

      expect(document.body.style.userSelect).toBe("")
      expect(widgetIframe().style.pointerEvents).toBe("")
      expect(localStorage.getItem("xagent_widget_width")).toBeNull()
      expect(panel().style.width).toBe("") // mobile now: inline width cleared, not "380px" or "580px"

      firePointerEvent(handle(), "pointermove", { pointerId: 18, clientX: 50 })
      expect(panel().style.width).toBe("") // still unaffected: the drag is over
    })

    it("does not cancel an active drag on a resize event that leaves the viewport width unchanged", () => {
      // Mobile browsers fire a bare `resize` event when the on-screen
      // keyboard opens/closes (innerHeight changes, innerWidth doesn't) --
      // startX/startWidth depend only on innerWidth, so this must not abort
      // an in-progress drag the way the genuine-width-change test above does.
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()

      firePointerEvent(handle(), "pointerdown", { pointerId: 26, clientX: 300 })
      firePointerEvent(handle(), "pointermove", { pointerId: 26, clientX: 250 })
      expect(panel().style.width).toBe("430px")
      expect(document.body.style.userSelect).toBe("none")

      window.dispatchEvent(new Event("resize")) // innerWidth unchanged

      expect(document.body.style.userSelect).toBe("none") // drag still active
      expect(panel().style.width).toBe("430px")

      firePointerEvent(handle(), "pointermove", { pointerId: 26, clientX: 200 })
      expect(panel().style.width).toBe("480px")
      firePointerEvent(handle(), "pointerup", { pointerId: 26, clientX: 200 })
      expect(localStorage.getItem("xagent_widget_width")).toBe("480")
    })

    it("keeps a wider preference across a drag that shrinks past the ceiling and returns to it", () => {
      localStorage.setItem("xagent_widget_width", "700")
      setInnerWidth(600) // viewportMax = 560
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()
      expect(panel().style.width).toBe("560px")

      firePointerEvent(handle(), "pointerdown", { pointerId: 19, clientX: 300 })
      // Genuinely shrink past the ceiling first...
      firePointerEvent(handle(), "pointermove", { pointerId: 19, clientX: 350 })
      expect(panel().style.width).toBe("510px")
      // ...then drag back up to exactly the ceiling value. Since this drag
      // already demonstrated a real shrink, this is now a deliberate choice
      // of 560, not an unmoved starting position -- it must NOT snap back to
      // the untouched 700px preference.
      firePointerEvent(handle(), "pointermove", { pointerId: 19, clientX: 300 })
      expect(panel().style.width).toBe("560px")
      firePointerEvent(handle(), "pointerup", { pointerId: 19, clientX: 300 })

      expect(localStorage.getItem("xagent_widget_width")).toBe("560")
    })

    it("does not restore userSelect if the host page changed it since drag start", () => {
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()

      firePointerEvent(handle(), "pointerdown", { pointerId: 13, clientX: 300 })
      expect(document.body.style.userSelect).toBe("none")

      // Host page sets its own value mid-drag, e.g. while rebuilding its own UI.
      document.body.style.userSelect = "all"

      firePointerEvent(handle(), "pointerup", { pointerId: 13, clientX: 300 })
      // Must NOT be clobbered back to the pre-drag snapshot.
      expect(document.body.style.userSelect).toBe("all")
    })

    it("does not restore cursor if the host page changed it since drag start", () => {
      // Same guard as userSelect above, applied to cursor -- restoreDragSideEffects
      // has two symmetric checks and this one had no dedicated negative-space
      // coverage of its own.
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()

      firePointerEvent(handle(), "pointerdown", { pointerId: 14, clientX: 300 })
      expect(document.body.style.cursor).toBe("ew-resize")

      document.body.style.cursor = "wait"

      firePointerEvent(handle(), "pointerup", { pointerId: 14, clientX: 300 })
      expect(document.body.style.cursor).toBe("wait")
    })

    it("does not throw when persisting the width fails", () => {
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()
      vi.spyOn(window.localStorage, "setItem").mockImplementation(() => {
        throw new Error("quota exceeded")
      })

      firePointerEvent(handle(), "pointerdown", { pointerId: 4, clientX: 300 })
      expect(() => firePointerEvent(handle(), "pointerup", { pointerId: 4, clientX: 300 }))
        .not.toThrow()
    })

    it("falls back to the default width if computed style is unreadable", () => {
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()
      vi.spyOn(window, "getComputedStyle").mockReturnValue({ width: "auto" } as CSSStyleDeclaration)

      firePointerEvent(handle(), "pointerdown", { pointerId: 5, clientX: 300 })
      firePointerEvent(handle(), "pointermove", { pointerId: 5, clientX: 250 })

      expect(panel().style.width).toBe("430px") // DEFAULT_PANEL_WIDTH (380) + 50
    })

    it("tears down cleanup immediately on DOM removal, even mid-drag, without waiting for blur", async () => {
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()
      document.body.style.userSelect = "text"

      firePointerEvent(handle(), "pointerdown", { pointerId: 14, clientX: 300 })
      expect(document.body.style.userSelect).toBe("none")

      document.querySelector(".xagent-widget-container")?.remove()
      // The teardown observer's callback fires as a microtask.
      await Promise.resolve()

      // Cleanup already ran on removal itself -- no blur or resize needed.
      expect(document.body.style.userSelect).toBe("text")
    })

    it("skips persisting the width for a drag interrupted by DOM removal", async () => {
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()

      firePointerEvent(handle(), "pointerdown", { pointerId: 15, clientX: 300 })
      firePointerEvent(handle(), "pointermove", { pointerId: 15, clientX: 200 }) // mid-drag, unconfirmed
      expect(panel().style.width).not.toBe("380px")

      document.querySelector(".xagent-widget-container")?.remove()
      await Promise.resolve()

      // The unconfirmed mid-drag width must not have been written to
      // storage -- nothing was ever persisted in this test at all.
      expect(localStorage.getItem("xagent_widget_width")).toBeNull()
    })

    it("tears down when the panel is removed directly, without container itself leaving the DOM", async () => {
      // Regression coverage for subtree: true. A remove() on .xagent-widget-
      // container (used by every other teardown test in this file) is
      // already a direct childList change on document.body, so it would
      // pass even with the narrower { childList: true } this diff replaced.
      // Removing panel out from under container, while container stays put,
      // only shows up as a *subtree* mutation of document.body -- this is
      // the one scenario that actually distinguishes the two.
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()
      document.body.style.userSelect = "text"

      firePointerEvent(handle(), "pointerdown", { pointerId: 25, clientX: 300 })
      expect(document.body.style.userSelect).toBe("none")

      panel().remove()
      await Promise.resolve()

      expect(document.body.style.userSelect).toBe("text")
    })

    it("never starts a new drag if the same panel node is later re-inserted into the DOM", async () => {
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()
      const container = document.querySelector(".xagent-widget-container")!
      const detachedHandle = handle()

      container.remove()
      await Promise.resolve() // teardown observer fires and disconnects

      // Re-inserting the SAME node (e.g. a host SPA's portal/keep-alive
      // remounting it) flips isConnected back to true without re-arming the
      // (already-disconnected) observer or any removed listener.
      document.body.appendChild(container)
      expect(container.isConnected).toBe(true)

      document.body.style.userSelect = "text"
      firePointerEvent(detachedHandle, "pointerdown", { pointerId: 20, clientX: 300 })
      // The torndown flag blocks this permanently, regardless of isConnected
      // -- the panel was opened above, so the open-state guard can't be
      // masking this as the actual reason.
      expect(document.body.style.userSelect).toBe("text")
    })

    it("stops reacting to window/document listeners once the panel leaves the DOM", async () => {
      // Spy without mockImplementation so these still call through -- we
      // want the real listeners attached, just also recorded.
      const winAddSpy = vi.spyOn(window, "addEventListener")
      const docAddSpy = vi.spyOn(document, "addEventListener")
      const winRemoveSpy = vi.spyOn(window, "removeEventListener")
      const docRemoveSpy = vi.spyOn(document, "removeEventListener")
      runWidget({ "data-widget-key": "widget-secret" })
      openPanel()
      const detachedHandle = handle()
      const detachedPanel = panel()

      function addedListener(spy: typeof winAddSpy, type: string) {
        const matches = spy.mock.calls.filter(([calledType]) => calledType === type)
        // Fail loudly on zero or multiple matches rather than silently
        // returning undefined (a legal, if useless, arg to
        // toHaveBeenCalledWith) or picking an arbitrary one of several.
        expect(matches).toHaveLength(1)
        return matches[0][1]
      }
      // Captured before removal so removeEventListener can be checked
      // against the *exact* function reference that was added -- not just
      // "any function", which would also pass if the wrong handler (or a
      // fresh unrelated one) were removed instead.
      const resizeListener = addedListener(winAddSpy, "resize")
      const blurListener = addedListener(winAddSpy, "blur")
      const pointermoveListener = addedListener(docAddSpy, "pointermove")
      const pointerupListener = addedListener(docAddSpy, "pointerup")
      const pointercancelListener = addedListener(docAddSpy, "pointercancel")

      document.querySelector(".xagent-widget-container")?.remove()
      await Promise.resolve()

      expect(winRemoveSpy).toHaveBeenCalledWith("resize", resizeListener)
      expect(winRemoveSpy).toHaveBeenCalledWith("blur", blurListener)
      expect(docRemoveSpy).toHaveBeenCalledWith("pointermove", pointermoveListener)
      expect(docRemoveSpy).toHaveBeenCalledWith("pointerup", pointerupListener)
      expect(docRemoveSpy).toHaveBeenCalledWith("pointercancel", pointercancelListener)

      // Behavioral proof for the resize listener specifically: below the
      // mobile breakpoint this would normally clear the inline width, so an
      // unchanged value after dispatch proves the listener is truly gone,
      // not just idle.
      const widthBeforeResize = detachedPanel.style.width
      setInnerWidth(400)
      window.dispatchEvent(new Event("resize"))
      expect(detachedPanel.style.width).toBe(widthBeforeResize)

      // Functional proof, not just a spy check that could pass even if the
      // handler removed were the wrong one: the torndown flag permanently
      // blocks pointerdown on this instance's handle -- covering a host SPA
      // re-inserting this same panel node later, which flips
      // panel.isConnected back to true without re-arming anything -- so a
      // fresh drag never starts and userSelect is never touched at all.
      document.body.style.userSelect = "text"
      firePointerEvent(detachedHandle, "pointerdown", { pointerId: 16, clientX: 300 })
      expect(document.body.style.userSelect).toBe("text")

      window.dispatchEvent(new Event("blur"))
      expect(document.body.style.userSelect).toBe("text")
    })
  })

  describe("mobile full screen", () => {
    function styleText() {
      return document.head.querySelector("style")!.innerHTML
    }

    function fab() {
      return document.querySelector<HTMLButtonElement>(".xagent-widget-fab")!
    }

    // Slices from the @media marker to the end of the stylesheet text (it's
    // the last block in the template) rather than trying to match a precise
    // closing brace -- formatting drift inside the block (added/reordered
    // declarations) then can't make this fragile.
    function mobileBlock() {
      const text = styleText()
      const index = text.indexOf('@media (max-width: 480px)')
      if (index === -1) throw new Error("mobile media query block not found in generated <style>")
      return text.slice(index)
    }

    beforeEach(() => {
      fetchMock.mockResolvedValue(new Response(JSON.stringify({
        ticket: "ticket/one",
        agent_id: 17,
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }))
    })

    it("makes the panel a true edge-to-edge overlay under the 480px breakpoint", () => {
      runWidget({ "data-widget-key": "widget-secret" })

      const block = mobileBlock()
      expect(block).toMatch(/\.xagent-widget-panel\s*\{[^}]*position:\s*fixed;/)
      expect(block).toMatch(/\.xagent-widget-panel\s*\{[^}]*inset:\s*0;/)
      expect(block).toMatch(/\.xagent-widget-panel\s*\{[^}]*border-radius:\s*0;/)
    })

    it("respects device safe areas instead of drawing under a notch or home indicator", () => {
      runWidget({ "data-widget-key": "widget-secret" })

      const block = mobileBlock()
      expect(block).toMatch(/\.xagent-widget-panel\s*\{[^}]*padding-top:\s*env\(safe-area-inset-top/)
      expect(block).toMatch(/\.xagent-widget-panel\s*\{[^}]*padding-bottom:\s*env\(safe-area-inset-bottom/)
    })

    it("moves the close control off the composer by repositioning the open FAB to the top corner", () => {
      runWidget({ "data-widget-key": "widget-secret" })

      const block = mobileBlock()
      expect(block).toMatch(/\.xagent-widget-fab\.xagent-widget-fab-panel-open\s*\{[^}]*position:\s*fixed;/)
      expect(block).toMatch(/\.xagent-widget-fab\.xagent-widget-fab-panel-open\s*\{[^}]*bottom:\s*auto;/)
    })

    it("marks the FAB open only while the panel is open, on the real open/close path", () => {
      runWidget({ "data-widget-key": "widget-secret" })

      expect(fab().classList.contains("xagent-widget-fab-panel-open")).toBe(false)

      fab().click()
      expect(fab().classList.contains("xagent-widget-fab-panel-open")).toBe(true)

      fab().click()
      expect(fab().classList.contains("xagent-widget-fab-panel-open")).toBe(false)
    })
  })
})

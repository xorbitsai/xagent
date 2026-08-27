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
      // innerHTML resets don't touch inline styles on body itself, and it's
      // shared across every test in this file.
      document.body.style.userSelect = ""
      document.body.style.cursor = ""
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

    it("falls back to the default width when reading storage throws", () => {
      const realGetItem = window.localStorage.getItem.bind(window.localStorage)
      vi.spyOn(window.localStorage, "getItem").mockImplementation((key) => {
        if (key === "xagent_widget_width") throw new Error("blocked")
        return realGetItem(key)
      })
      runWidget({ "data-widget-key": "widget-secret" })
      expect(panel().style.width).toBe("380px")
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

    it("ignores a drag start below the mobile breakpoint", () => {
      setInnerWidth(400)
      runWidget({ "data-widget-key": "widget-secret" })

      firePointerEvent(handle(), "pointerdown", { pointerId: 1, clientX: 100 })

      expect(document.body.style.userSelect).toBe("")
      expect(widgetIframe().style.pointerEvents).toBe("")
    })

    it("resizes the panel while dragging and persists the final width on release", () => {
      runWidget({ "data-widget-key": "widget-secret" })
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

    it("cancels rather than commits on pointercancel, reverting the in-progress width", () => {
      // pointercancel is an involuntary interruption (a touch gesture
      // reinterpreted as a scroll, an OS-level interrupt), not the user
      // confirming a release -- it must not persist wherever the drag had
      // gotten to.
      runWidget({ "data-widget-key": "widget-secret" })

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
      // A drag that gets cancelled must roll panelWidth itself back, not
      // just skip persisting once -- otherwise the abandoned in-progress
      // value stays the module's current width, and an unrelated
      // zero-movement drag later on would commit it via a normal release.
      runWidget({ "data-widget-key": "widget-secret" })

      firePointerEvent(handle(), "pointerdown", { pointerId: 21, clientX: 300 })
      firePointerEvent(handle(), "pointermove", { pointerId: 21, clientX: 250 }) // -> 430px, abandoned below
      window.dispatchEvent(new Event("blur")) // cancels; must revert to 380px, not just skip persisting

      // An unrelated click-and-release with zero movement.
      firePointerEvent(handle(), "pointerdown", { pointerId: 22, clientX: 300 })
      firePointerEvent(handle(), "pointerup", { pointerId: 22, clientX: 300 })

      expect(localStorage.getItem("xagent_widget_width")).toBe("380") // not the abandoned 430
    })

    it("restores the host page's own cursor rather than clearing it", () => {
      runWidget({ "data-widget-key": "widget-secret" })
      document.body.style.cursor = "help"

      firePointerEvent(handle(), "pointerdown", { pointerId: 23, clientX: 300 })
      expect(document.body.style.cursor).toBe("ew-resize")

      firePointerEvent(handle(), "pointerup", { pointerId: 23, clientX: 300 })
      expect(document.body.style.cursor).toBe("help")
    })

    it("ignores a non-primary pointer button", () => {
      runWidget({ "data-widget-key": "widget-secret" })

      firePointerEvent(handle(), "pointerdown", { pointerId: 24, clientX: 300, button: 2 })

      expect(document.body.style.userSelect).toBe("")
      expect(widgetIframe().style.pointerEvents).toBe("")
    })

    it("ignores a second concurrent pointer without disrupting the first pointer's drag", () => {
      runWidget({ "data-widget-key": "widget-secret" })

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

    it("cancels (without persisting) an active drag on window resize", () => {
      runWidget({ "data-widget-key": "widget-secret" })

      firePointerEvent(handle(), "pointerdown", { pointerId: 18, clientX: 300 })
      firePointerEvent(handle(), "pointermove", { pointerId: 18, clientX: 250 }) // mid-drag, unconfirmed
      expect(document.body.style.userSelect).toBe("none")

      // A viewport change invalidates this drag's frozen startX/startWidth --
      // rather than let the next pointermove misread it as a shrink, the
      // drag is cancelled outright.
      window.dispatchEvent(new Event("resize"))

      expect(document.body.style.userSelect).toBe("")
      expect(widgetIframe().style.pointerEvents).toBe("")
      expect(localStorage.getItem("xagent_widget_width")).toBeNull()
      // Reverted, not left at the abandoned 430px (380 + 50 from the move above).
      expect(panel().style.width).toBe("380px")

      firePointerEvent(handle(), "pointermove", { pointerId: 18, clientX: 100 })
      expect(panel().style.width).toBe("380px") // still unaffected: the drag is over
    })

    it("keeps a wider preference across a drag that shrinks past the ceiling and returns to it", () => {
      localStorage.setItem("xagent_widget_width", "700")
      setInnerWidth(600) // viewportMax = 560
      runWidget({ "data-widget-key": "widget-secret" })
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

      firePointerEvent(handle(), "pointerdown", { pointerId: 13, clientX: 300 })
      expect(document.body.style.userSelect).toBe("none")

      // Host page sets its own value mid-drag, e.g. while rebuilding its own UI.
      document.body.style.userSelect = "all"

      firePointerEvent(handle(), "pointerup", { pointerId: 13, clientX: 300 })
      // Must NOT be clobbered back to the pre-drag snapshot.
      expect(document.body.style.userSelect).toBe("all")
    })

    it("does not throw when persisting the width fails", () => {
      runWidget({ "data-widget-key": "widget-secret" })
      vi.spyOn(window.localStorage, "setItem").mockImplementation(() => {
        throw new Error("quota exceeded")
      })

      firePointerEvent(handle(), "pointerdown", { pointerId: 4, clientX: 300 })
      expect(() => firePointerEvent(handle(), "pointerup", { pointerId: 4, clientX: 300 }))
        .not.toThrow()
    })

    it("falls back to the default width if computed style is unreadable", () => {
      runWidget({ "data-widget-key": "widget-secret" })
      vi.spyOn(window, "getComputedStyle").mockReturnValue({ width: "auto" } as CSSStyleDeclaration)

      firePointerEvent(handle(), "pointerdown", { pointerId: 5, clientX: 300 })
      firePointerEvent(handle(), "pointermove", { pointerId: 5, clientX: 250 })

      expect(panel().style.width).toBe("430px") // DEFAULT_PANEL_WIDTH (380) + 50
    })

    it("tears down cleanup immediately on DOM removal, even mid-drag, without waiting for blur", async () => {
      runWidget({ "data-widget-key": "widget-secret" })
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

      firePointerEvent(handle(), "pointerdown", { pointerId: 15, clientX: 300 })
      firePointerEvent(handle(), "pointermove", { pointerId: 15, clientX: 200 }) // mid-drag, unconfirmed
      expect(panel().style.width).not.toBe("380px")

      document.querySelector(".xagent-widget-container")?.remove()
      await Promise.resolve()

      // The unconfirmed mid-drag width must not have been written to
      // storage -- nothing was ever persisted in this test at all.
      expect(localStorage.getItem("xagent_widget_width")).toBeNull()
    })

    it("never starts a new drag if the same panel node is later re-inserted into the DOM", async () => {
      runWidget({ "data-widget-key": "widget-secret" })
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
      // The torndown flag blocks this permanently, regardless of isConnected.
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
      const detachedHandle = handle()
      const detachedPanel = panel()

      function addedListener(spy: typeof winAddSpy, type: string) {
        return spy.mock.calls.find(([calledType]) => calledType === type)?.[1]
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
})

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
      init: { pointerId: number; clientX: number },
    ) {
      // jsdom has no PointerEvent constructor; a plain Event with the fields
      // the widget reads is enough since addEventListener matches by type.
      const event = new MouseEvent(type, { bubbles: true, cancelable: true, clientX: init.clientX })
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

    it("hides the inline width below the mobile breakpoint and restores it above", () => {
      runWidget({ "data-widget-key": "widget-secret" })

      setInnerWidth(400)
      window.dispatchEvent(new Event("resize"))
      expect(panel().style.width).toBe("")

      setInnerWidth(1024)
      window.dispatchEvent(new Event("resize"))
      expect(panel().style.width).toBe("380px")
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

    it("ends the drag on pointercancel", () => {
      runWidget({ "data-widget-key": "widget-secret" })

      firePointerEvent(handle(), "pointerdown", { pointerId: 2, clientX: 300 })
      firePointerEvent(handle(), "pointercancel", { pointerId: 2, clientX: 300 })

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

    it("cleans up an interrupted drag on window blur", () => {
      runWidget({ "data-widget-key": "widget-secret" })

      firePointerEvent(handle(), "pointerdown", { pointerId: 3, clientX: 300 })
      expect(document.body.style.userSelect).toBe("none")

      window.dispatchEvent(new Event("blur"))

      expect(document.body.style.userSelect).toBe("")
      expect(widgetIframe().style.pointerEvents).toBe("")
      expect(localStorage.getItem("xagent_widget_width")).toBe("380")
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

    it("stops reacting to window resize once the panel leaves the DOM", () => {
      const removeSpy = vi.spyOn(window, "removeEventListener")
      localStorage.setItem("xagent_widget_width", "600")
      runWidget({ "data-widget-key": "widget-secret" })
      const panelEl = panel()
      expect(panelEl.style.width).toBe("600px")

      document.querySelector(".xagent-widget-container")?.remove()

      // Below the mobile breakpoint would normally clear the inline width;
      // an unchanged value proves the listener self-removed instead of
      // reapplying the clamp to a detached panel.
      setInnerWidth(400)
      window.dispatchEvent(new Event("resize"))

      expect(panelEl.style.width).toBe("600px")
      expect(removeSpy).toHaveBeenCalledWith("resize", expect.any(Function))
    })

    it("still cleans up an interrupted drag on blur after the panel leaves the DOM", () => {
      const removeSpy = vi.spyOn(window, "removeEventListener")
      runWidget({ "data-widget-key": "widget-secret" })
      document.body.style.userSelect = "text"

      firePointerEvent(handle(), "pointerdown", { pointerId: 9, clientX: 300 })
      expect(document.body.style.userSelect).toBe("none")

      document.querySelector(".xagent-widget-container")?.remove()
      window.dispatchEvent(new Event("blur"))

      // The drag's cleanup obligation (restoring the host page's userSelect)
      // must run even though the panel is no longer connected -- unlike the
      // resize listener, self-unsubscribing here must not skip it.
      expect(document.body.style.userSelect).toBe("text")
      expect(removeSpy).toHaveBeenCalledWith("blur", expect.any(Function))

      // And the listener is now actually gone: a second blur is a no-op,
      // not just a repeat no-drag-active early return.
      document.body.style.userSelect = "text"
      window.dispatchEvent(new Event("blur"))
      expect(document.body.style.userSelect).toBe("text")
    })
  })
})

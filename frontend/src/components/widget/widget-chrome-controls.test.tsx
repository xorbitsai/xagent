import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { WidgetChromeControls } from "./widget-chrome-controls"

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

describe("WidgetChromeControls", () => {
  const postMessageSpy = vi.fn()

  beforeEach(() => {
    postMessageSpy.mockReset()
    vi.stubGlobal("parent", { postMessage: postMessageSpy })
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it("renders a close button and a collapsed menu trigger", () => {
    render(<WidgetChromeControls />)

    expect(screen.getByRole("button", { name: "widgetChat.close" })).toBeInTheDocument()
    const menuTrigger = screen.getByRole("button", { name: "widgetChat.moreOptions" })
    expect(menuTrigger).toHaveAttribute("aria-expanded", "false")
    expect(screen.queryByRole("menu")).toBeNull()
  })

  it("posts widget_close to the parent window when the close button is clicked", () => {
    render(<WidgetChromeControls />)

    fireEvent.click(screen.getByRole("button", { name: "widgetChat.close" }))

    expect(postMessageSpy).toHaveBeenCalledWith(
      { xagent: true, v: 1, type: "widget_close" },
      "*",
    )
  })

  it("opens the menu on trigger click and posts widget_minimize, closing the menu", () => {
    render(<WidgetChromeControls />)

    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    expect(screen.getByRole("menu")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("menuitem", { name: "widgetChat.minimize" }))

    expect(postMessageSpy).toHaveBeenCalledWith(
      { xagent: true, v: 1, type: "widget_minimize" },
      "*",
    )
    expect(screen.queryByRole("menu")).toBeNull()
  })

  it("closes the menu on Escape", () => {
    render(<WidgetChromeControls />)

    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    expect(screen.getByRole("menu")).toBeInTheDocument()

    fireEvent.keyDown(document, { key: "Escape" })

    expect(screen.queryByRole("menu")).toBeNull()
  })

  it("closes the menu when the iframe window loses focus", () => {
    // e.g. the host page's own FAB, entirely outside this document, hiding
    // the panel -- neither the pointerdown nor the Escape listener fires.
    render(<WidgetChromeControls />)

    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    expect(screen.getByRole("menu")).toBeInTheDocument()

    fireEvent(window, new Event("blur"))

    expect(screen.queryByRole("menu")).toBeNull()
  })

  it("closes the menu on an outside pointer press", () => {
    render(
      <div>
        <div data-testid="outside">outside</div>
        <WidgetChromeControls />
      </div>,
    )

    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    expect(screen.getByRole("menu")).toBeInTheDocument()

    fireEvent.pointerDown(screen.getByTestId("outside"))

    expect(screen.queryByRole("menu")).toBeNull()
  })
})

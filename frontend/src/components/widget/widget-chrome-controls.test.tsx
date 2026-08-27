import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { WidgetChromeControls } from "./widget-chrome-controls"

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

describe("WidgetChromeControls", () => {
  const postMessageSpy = vi.fn()
  const onNewConversation = vi.fn()

  beforeEach(() => {
    postMessageSpy.mockReset()
    onNewConversation.mockReset()
    vi.stubGlobal("parent", { postMessage: postMessageSpy })
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it("renders only the close button when there is no new-conversation action", () => {
    render(<WidgetChromeControls />)

    expect(screen.getByRole("button", { name: "widgetChat.close" })).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "widgetChat.moreOptions" })).toBeNull()
  })

  it("renders a collapsed menu trigger when a new-conversation action is given", () => {
    render(<WidgetChromeControls newConversation={{ label: "Start over", onClick: onNewConversation }} />)

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

  it("opens the menu on trigger click, shows the given label, and calls onClick then closes the menu", () => {
    render(<WidgetChromeControls newConversation={{ label: "Start over", onClick: onNewConversation }} />)

    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    const item = screen.getByRole("menuitem", { name: "Start over" })
    expect(item).toBeInTheDocument()

    fireEvent.click(item)

    expect(onNewConversation).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole("menu")).toBeNull()
  })

  it("disables the new-conversation menu item when told to", () => {
    render(
      <WidgetChromeControls
        newConversation={{ label: "Resetting...", onClick: onNewConversation, disabled: true }}
      />,
    )

    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))

    expect(screen.getByRole("menuitem", { name: "Resetting..." })).toBeDisabled()
  })

  it("closes the menu on Escape", () => {
    render(<WidgetChromeControls newConversation={{ label: "Start over", onClick: onNewConversation }} />)

    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    expect(screen.getByRole("menu")).toBeInTheDocument()

    fireEvent.keyDown(document, { key: "Escape" })

    expect(screen.queryByRole("menu")).toBeNull()
  })

  it("closes the menu when the iframe window loses focus", () => {
    // e.g. the host page's own FAB, entirely outside this document, hiding
    // the panel -- neither the pointerdown nor the Escape listener fires.
    render(<WidgetChromeControls newConversation={{ label: "Start over", onClick: onNewConversation }} />)

    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    expect(screen.getByRole("menu")).toBeInTheDocument()

    fireEvent(window, new Event("blur"))

    expect(screen.queryByRole("menu")).toBeNull()
  })

  it("closes the menu on an outside pointer press", () => {
    render(
      <div>
        <div data-testid="outside">outside</div>
        <WidgetChromeControls newConversation={{ label: "Start over", onClick: onNewConversation }} />
      </div>,
    )

    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    expect(screen.getByRole("menu")).toBeInTheDocument()

    fireEvent.pointerDown(screen.getByTestId("outside"))

    expect(screen.queryByRole("menu")).toBeNull()
  })
})

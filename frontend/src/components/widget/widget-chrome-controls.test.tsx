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

  it("renders the close button and a collapsed menu trigger even with no new-conversation action", () => {
    // Expand/collapse is always offered regardless of newConversation, so
    // the "..." trigger has something to open onto even without it.
    render(<WidgetChromeControls />)

    expect(screen.getByRole("button", { name: "widgetChat.close" })).toBeInTheDocument()
    const menuTrigger = screen.getByRole("button", { name: "widgetChat.moreOptions" })
    expect(menuTrigger).toHaveAttribute("aria-expanded", "false")
    expect(screen.queryByRole("menu")).toBeNull()
  })

  it("omits the new-conversation menu item when no action is given, but still offers expand", () => {
    render(<WidgetChromeControls />)

    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    const items = screen.getAllByRole("menuitem")

    expect(items).toHaveLength(1)
    expect(items[0]).toHaveTextContent("widgetChat.expandWindow")
  })

  it("posts widget_close to the parent window when the close button is clicked", () => {
    render(<WidgetChromeControls />)

    fireEvent.click(screen.getByRole("button", { name: "widgetChat.close" }))

    expect(postMessageSpy).toHaveBeenCalledWith(
      { xagent: true, v: 1, type: "widget_close" },
      "*",
    )
  })

  it("does not post a message when the page is visited directly (window.parent === window)", () => {
    // window.parent === window here means the guard's else-branch calls the
    // real window.postMessage, not the postMessageSpy stub above -- spy on
    // the real one directly so a missing guard would actually be caught.
    vi.stubGlobal("parent", window)
    const realPostMessageSpy = vi.spyOn(window, "postMessage")

    render(<WidgetChromeControls />)
    fireEvent.click(screen.getByRole("button", { name: "widgetChat.close" }))

    expect(realPostMessageSpy).not.toHaveBeenCalled()
    realPostMessageSpy.mockRestore()
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

  it("keeps a pending action visible on the trigger after the menu closes on click", () => {
    // The menu closes immediately on menuitem click (standard menu UX), so a
    // slow round-trip needs feedback that survives that -- the trigger itself
    // switches to a spinner rather than depending on the visitor reopening
    // the (by-then pending) menu to see anything changed.
    const { rerender } = render(
      <WidgetChromeControls
        newConversation={{ label: "Start over", onClick: onNewConversation }}
      />,
    )
    const trigger = screen.getByRole("button", { name: "widgetChat.moreOptions" })
    fireEvent.click(trigger)
    fireEvent.click(screen.getByRole("menuitem", { name: "Start over" }))
    expect(screen.queryByRole("menu")).toBeNull()
    expect(onNewConversation).toHaveBeenCalledTimes(1)

    rerender(
      <WidgetChromeControls
        newConversation={{
          label: "Resetting...",
          onClick: onNewConversation,
          disabled: true,
          pending: true,
        }}
      />,
    )

    expect(trigger).toBeDisabled()
    expect(screen.queryByRole("menu")).toBeNull()
    // Not just "disabled" -- the whole point of this prop is a visible
    // in-progress indicator on the trigger once the menu itself has closed.
    expect(trigger.querySelector("svg.animate-spin")).not.toBeNull()
  })

  it("toggles expand/collapse: posts the right message each way and flips the item's icon and label", () => {
    render(<WidgetChromeControls />)

    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    fireEvent.click(screen.getByRole("menuitem", { name: "widgetChat.expandWindow" }))

    expect(postMessageSpy).toHaveBeenCalledWith(
      { xagent: true, v: 1, type: "widget_expand" },
      "*",
    )
    // The menu closes on click, same as every other menu item.
    expect(screen.queryByRole("menu")).toBeNull()

    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    expect(screen.getByRole("menuitem", { name: "widgetChat.collapseWindow" })).toBeInTheDocument()

    fireEvent.click(screen.getByRole("menuitem", { name: "widgetChat.collapseWindow" }))

    expect(postMessageSpy).toHaveBeenCalledWith(
      { xagent: true, v: 1, type: "widget_collapse" },
      "*",
    )
  })

  it("keeps the new-conversation item and the expand toggle as two independent menu items", () => {
    render(<WidgetChromeControls newConversation={{ label: "Start over", onClick: onNewConversation }} />)

    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    const items = screen.getAllByRole("menuitem")

    expect(items).toHaveLength(2)
    fireEvent.click(screen.getByRole("menuitem", { name: "widgetChat.expandWindow" }))

    expect(onNewConversation).not.toHaveBeenCalled()
    expect(postMessageSpy).toHaveBeenCalledWith(
      { xagent: true, v: 1, type: "widget_expand" },
      "*",
    )
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

  it("ignores a blur caused by focus moving within this same document, not the panel being hidden", () => {
    // e.g. focus moving into the chat composer -- document.hasFocus() stays
    // true because the iframe itself was never actually hidden; only a blur
    // where it goes false (focus left the iframe's document entirely) means
    // the host page hid the panel out from under an open menu.
    const hasFocusSpy = vi.spyOn(document, "hasFocus").mockReturnValue(true)
    render(<WidgetChromeControls newConversation={{ label: "Start over", onClick: onNewConversation }} />)

    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    expect(screen.getByRole("menu")).toBeInTheDocument()

    fireEvent(window, new Event("blur"))

    expect(screen.getByRole("menu")).toBeInTheDocument()
    hasFocusSpy.mockRestore()
  })

  it("ignores a pointerdown whose target was removed from the DOM during dispatch", () => {
    // A sibling handler on the target itself can synchronously detach it
    // before the event finishes bubbling to this component's document
    // listener; Node.contains() on a now-disconnected node would otherwise
    // read the same as a genuine outside click and close the menu.
    render(<WidgetChromeControls newConversation={{ label: "Start over", onClick: onNewConversation }} />)

    fireEvent.click(screen.getByRole("button", { name: "widgetChat.moreOptions" }))
    expect(screen.getByRole("menu")).toBeInTheDocument()

    const ephemeral = document.createElement("button")
    document.body.appendChild(ephemeral)
    ephemeral.addEventListener("pointerdown", () => ephemeral.remove())

    fireEvent.pointerDown(ephemeral)

    expect(screen.getByRole("menu")).toBeInTheDocument()
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

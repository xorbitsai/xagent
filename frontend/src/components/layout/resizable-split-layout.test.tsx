import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"

import { ResizableSplitLayout } from "./resizable-split-layout"

const leftPanel = <div>Left panel</div>
const rightPanel = <div>Right panel</div>
const otherRightPanel = <div>Other right panel</div>

function stubContainerWidth(container: HTMLElement) {
  const layout = container.firstElementChild as HTMLDivElement
  Object.defineProperty(layout, "getBoundingClientRect", {
    configurable: true,
    value: () => ({
      left: 0,
      width: 1000,
      top: 0,
      right: 1000,
      bottom: 500,
      height: 500,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    }),
  })
  return layout
}

function dragTo(clientX: number) {
  fireEvent.mouseDown(screen.getByRole("separator", { name: "Resize panels" }))
  fireEvent.mouseMove(document, { clientX })
  fireEvent.mouseUp(document)
}

describe("ResizableSplitLayout", () => {
  afterEach(() => {
    cleanup()
  })

  it("preserves a manual resize when the right panel's content changes but initialLeftWidth does not", () => {
    const { container, rerender } = render(
      <ResizableSplitLayout
        leftPanel={leftPanel}
        rightPanel={rightPanel}
        initialLeftWidth={65}
      />,
    )
    const layout = stubContainerWidth(container)

    dragTo(600)
    expect(layout.firstElementChild).toHaveStyle({ width: "60%" })

    // Same initialLeftWidth (e.g. switching between two inspector modes that
    // share the same preset, like flow <-> agent) -- the manual drag stands.
    rerender(
      <ResizableSplitLayout
        leftPanel={leftPanel}
        rightPanel={otherRightPanel}
        initialLeftWidth={65}
      />,
    )
    expect(layout.firstElementChild).toHaveStyle({ width: "60%" })
  })

  it("resets to the new initialLeftWidth when it changes while the right panel stays open", () => {
    const { container, rerender } = render(
      <ResizableSplitLayout
        leftPanel={leftPanel}
        rightPanel={rightPanel}
        initialLeftWidth={65}
      />,
    )
    const layout = stubContainerWidth(container)

    dragTo(600)
    expect(layout.firstElementChild).toHaveStyle({ width: "60%" })

    // A caller switching to a mode with a different desired split (e.g. a
    // file preview wanting 50/50 instead of 65/35) must not need to remount
    // either panel to get it -- this reactivity is what lets it snap to the
    // new preset in place.
    rerender(
      <ResizableSplitLayout
        leftPanel={leftPanel}
        rightPanel={otherRightPanel}
        initialLeftWidth={50}
      />,
    )
    expect(layout.firstElementChild).toHaveStyle({ width: "50%" })
  })

  it("resets to initialLeftWidth on the closed-to-open transition", () => {
    const { container, rerender } = render(
      <ResizableSplitLayout
        leftPanel={leftPanel}
        rightPanel={rightPanel}
        initialLeftWidth={65}
      />,
    )
    const layout = stubContainerWidth(container)

    dragTo(600)
    expect(layout.firstElementChild).toHaveStyle({ width: "60%" })

    rerender(<ResizableSplitLayout leftPanel={leftPanel} initialLeftWidth={50} />)
    expect(layout.firstElementChild).toHaveStyle({ width: "100%" })

    rerender(
      <ResizableSplitLayout
        leftPanel={leftPanel}
        rightPanel={rightPanel}
        initialLeftWidth={50}
      />,
    )
    expect(layout.firstElementChild).toHaveStyle({ width: "50%" })
  })
})

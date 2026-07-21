import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { ResizableSplitLayout } from "./resizable-split-layout"

const leftPanel = <div>Left panel</div>
const rightPanel = <div>Right panel</div>

describe("ResizableSplitLayout", () => {
  it("preserves a manual resize when the open inspector mode changes", () => {
    const { container, rerender } = render(
      <ResizableSplitLayout
        leftPanel={leftPanel}
        rightPanel={rightPanel}
        initialLeftWidth={65}
      />,
    )
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

    fireEvent.mouseDown(screen.getByRole("separator", { name: "Resize panels" }))
    fireEvent.mouseMove(document, { clientX: 600 })
    fireEvent.mouseUp(document)

    expect(layout.firstElementChild).toHaveStyle({ width: "60%" })

    rerender(
      <ResizableSplitLayout
        leftPanel={leftPanel}
        rightPanel={rightPanel}
        initialLeftWidth={50}
      />,
    )

    expect(layout.firstElementChild).toHaveStyle({ width: "60%" })

    rerender(
      <ResizableSplitLayout
        leftPanel={leftPanel}
        initialLeftWidth={50}
      />,
    )
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

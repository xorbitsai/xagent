/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { render } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { ScrollArea } from "./scroll-area"

describe("ScrollArea", () => {
  it("enables horizontal and vertical scrolling", () => {
    render(
      <ScrollArea>
        <div>wide preview content</div>
      </ScrollArea>,
    )

    const viewport = document.querySelector("[data-slot='scroll-area-viewport']")
    expect(viewport).toHaveStyle({
      overflowX: "scroll",
      overflowY: "scroll",
    })
  })
})

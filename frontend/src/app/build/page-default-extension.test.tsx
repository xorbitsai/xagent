import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"

import {
  BuildAgentCardExtension,
  BuildPageExtensionProvider,
} from "@/lib/build-page-extension"

describe("Build page shipped defaults", () => {
  afterEach(() => cleanup())

  it("renders Provider children without an extra wrapper", () => {
    const { container } = render(
      <BuildPageExtensionProvider>
        <span data-testid="provider-child" />
      </BuildPageExtensionProvider>,
    )
    const child = screen.getByTestId("provider-child")

    expect(container.childElementCount).toBe(1)
    expect(container.firstElementChild).toBe(child)
  })

  it("contributes no card DOM", () => {
    const { container } = render(<BuildAgentCardExtension agentId={42} />)

    expect(container).toBeEmptyDOMElement()
  })
})

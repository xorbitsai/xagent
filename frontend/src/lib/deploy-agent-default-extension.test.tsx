import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"

import { DeployAgentEmbedExtension } from "@/lib/deploy-agent-extension"

describe("Deploy dialog shipped default", () => {
  afterEach(() => cleanup())

  it("renders only the built-in embed snippet", () => {
    const { container } = render(
      <DeployAgentEmbedExtension agentId={7} widgetOrigin="https://example.test">
        <span data-testid="embed-snippet" />
      </DeployAgentEmbedExtension>,
    )

    expect(container.childElementCount).toBe(1)
    expect(container.firstElementChild).toBe(screen.getByTestId("embed-snippet"))
  })
})

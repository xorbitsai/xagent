/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { render } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import {
  AGENT_CHANNEL_ENABLED,
  AgentChannelHeaderAction,
  AgentChannelOption,
  AgentChannelView,
} from "./agent-channel-extension"

const agent = { id: 1, name: "Cesar", description: "", logo_url: null, status: "published" }

describe("agent channel extension default", () => {
  it("is disabled and renders nothing", () => {
    expect(AGENT_CHANNEL_ENABLED).toBe(false)
    const { container } = render(
      <>
        <AgentChannelHeaderAction agent={agent} onOpen={vi.fn()} />
        <AgentChannelOption agent={agent} onOpen={vi.fn()} />
        <AgentChannelView agent={agent} onBack={vi.fn()} onClose={vi.fn()} />
      </>,
    )
    expect(container).toBeEmptyDOMElement()
  })
})

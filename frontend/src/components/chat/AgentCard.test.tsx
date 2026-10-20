import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { AgentCard, AgentChip } from "./AgentCard"

const push = vi.hoisted(() => vi.fn())
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push }),
  usePathname: () => "/task",
}))
afterEach(() => { cleanup(); push.mockReset() })

describe.each([["card", AgentCard], ["chip", AgentChip]] as const)("%s navigation", (_name, Component) => {
  it.each(["draft", "published"] as const)("routes a %s agent to the appropriate page", (status) => {
    render(<Component agentId="7" agentName="Ben" status={status} />)
    fireEvent.click(screen.getByText("Ben"))
    expect(push).toHaveBeenCalledWith(status === "draft" ? "/build/7" : "/task?agent=7")
  })
})

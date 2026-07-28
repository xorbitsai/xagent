import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
  }),
}))

vi.mock("@/components/chat/ChatInput", () => ({
  ChatInput: () => <div data-testid="chat-input" />,
}))

vi.mock("@/components/ui/tooltip", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  TooltipContent: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  TooltipProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  TooltipTrigger: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}))

import { ChatStartScreen } from "@/components/chat/ChatStartScreen"

const AGENTS_SECTION = "chatPage.sections.chatWithAgents"
// t() is mocked as identity, so the deleted demo cards would surface as their
// literal keys — assert on those directly rather than only on the header they
// happened to share a branch with.
const MOCKED_AGENT_KEYS = /chatPage\.agents\./

// The agents block only renders inside the prompts branch, so every case needs
// a non-empty prompts list to reach it at all.
const renderScreen = (agents?: React.ComponentProps<typeof ChatStartScreen>["agents"]) =>
  render(
    <ChatStartScreen
      title="Support Agent"
      prompts={["Summarize this page"]}
      agents={agents}
      onSend={vi.fn()}
    />
  )

afterEach(() => {
  cleanup()
})

describe("ChatStartScreen agents section", () => {
  it("renders nothing when no agent list is passed", () => {
    renderScreen(undefined)

    expect(screen.queryByText(AGENTS_SECTION)).toBeNull()
    expect(screen.queryByText(MOCKED_AGENT_KEYS)).toBeNull()
  })

  it("renders nothing for an empty agent list", () => {
    renderScreen([])

    expect(screen.queryByText(AGENTS_SECTION)).toBeNull()
    expect(screen.queryByText(MOCKED_AGENT_KEYS)).toBeNull()
  })

  it("renders the agents that were passed", () => {
    renderScreen([{ id: 7, name: "Billing Agent" }])

    expect(screen.getByText(AGENTS_SECTION)).toBeTruthy()
    expect(screen.getByText("Billing Agent")).toBeTruthy()
  })
})

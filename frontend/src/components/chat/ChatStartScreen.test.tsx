import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

const chatInputProps = vi.hoisted(() => ({
  current: null as null | {
    files?: File[]
    filesDisabled?: boolean
    hideFileUpload?: boolean
    onSend: (message: string, config?: unknown) => void
  },
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
  }),
}))

vi.mock("@/lib/utils", () => ({
  getApiUrl: () => "http://api.local",
}))

vi.mock("@/components/chat/ChatInput", () => ({
  ChatInput: (props: NonNullable<typeof chatInputProps.current>) => {
    chatInputProps.current = props
    return <div data-testid="chat-input" />
  },
}))

vi.mock("@/components/ui/tooltip", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  TooltipContent: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  TooltipProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  TooltipTrigger: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}))

import { ChatStartScreen } from "@/components/chat/ChatStartScreen"

const AGENTS_SECTION = "chatPage.sections.chatWithAgents"

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
  chatInputProps.current = null
})

describe("ChatStartScreen agents section", () => {
  it("renders nothing when no agent list is passed", () => {
    renderScreen(undefined)

    expect(screen.queryByText(AGENTS_SECTION)).toBeNull()
  })

  it("renders nothing for an empty agent list", () => {
    renderScreen([])

    expect(screen.queryByText(AGENTS_SECTION)).toBeNull()
  })

  it("renders the agents that were passed", () => {
    renderScreen([{ id: 7, name: "Billing Agent" }])

    expect(screen.getByText(AGENTS_SECTION)).toBeTruthy()
    expect(screen.getByText("Billing Agent")).toBeTruthy()
  })
})

describe("ChatStartScreen file capability", () => {
  it("forwards the disabled file capability to ChatInput", () => {
    const onSend = vi.fn()
    const file = new File(["secret"], "secret.txt", { type: "text/plain" })
    render(
      <ChatStartScreen
        files={[file]}
        filesDisabled
        onSend={onSend}
        title="Session Agent"
      />
    )

    expect(screen.getByTestId("chat-input")).toBeInTheDocument()
    expect(chatInputProps.current?.hideFileUpload).toBe(true)
    expect(chatInputProps.current?.filesDisabled).toBe(true)
    expect(chatInputProps.current?.files).toEqual([])
    chatInputProps.current?.onSend("hello", { mode: "balanced" })
    expect(onSend).toHaveBeenCalledWith("hello", [], { mode: "balanced" })
  })

  it("preserves file input by default", () => {
    const onSend = vi.fn()
    const file = new File(["legacy"], "legacy.txt", { type: "text/plain" })
    render(
      <ChatStartScreen
        files={[file]}
        onSend={onSend}
        title="Legacy Agent"
      />
    )

    expect(chatInputProps.current?.hideFileUpload).toBe(false)
    expect(chatInputProps.current?.filesDisabled).toBe(false)
    expect(chatInputProps.current?.files).toEqual([file])
    chatInputProps.current?.onSend("hello", { mode: "balanced" })
    expect(onSend).toHaveBeenCalledWith("hello", [file], { mode: "balanced" })
  })
})

import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

const chatInputProps = vi.hoisted(() => ({
  current: null as null | {
    files?: File[]
    filesDisabled?: boolean
    hideFileUpload?: boolean
    selectedAgents?: Array<{ id: number | string; name: string }>
    onSend: (message: string, config?: unknown) => void
  },
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
  }),
}))

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return { ...actual, getApiUrl: () => "http://api.local" }
})

vi.mock("@/lib/branding", () => ({
  getBrandingFromEnv: () => ({ appName: "Xagent" }),
}))

vi.mock("@/components/chat/ChatInput", () => ({
  ChatInput: (props: NonNullable<typeof chatInputProps.current>) => {
    chatInputProps.current = props
    return <div data-testid="chat-input" />
  },
}))

import { ChatStartScreen } from "@/components/chat/ChatStartScreen"

const AGENTS_SECTION = "chatPage.sections.assignToTeammate"

// The agents block renders independently of `prompts` - task/page.tsx (its
// only real caller) never passes prompts at all.
const renderScreen = (agents?: React.ComponentProps<typeof ChatStartScreen>["agents"]) =>
  render(
    <ChatStartScreen
      title="Support Agent"
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

  it("shows a pill's specialty label and prefers its persona photo over logo_url", () => {
    renderScreen([
      {
        id: 7,
        name: "Maya",
        logo_url: "/uploads/should-not-be-used.png",
        persona_avatar: "/marketplace/avatars/maya.png",
        specialty: "Marketing",
      },
    ])

    expect(screen.getByText("Marketing")).toBeTruthy()
    expect(screen.getByRole("img", { name: "Maya" })).toHaveAttribute(
      "src",
      "/marketplace/avatars/maya.png"
    )
  })

  it("swaps the header for the selected teammate's portrait and a ready-to-lead subline", () => {
    const maya = { id: 7, name: "Maya", persona_avatar: "/marketplace/avatars/maya.png" }
    render(
      <ChatStartScreen
        title="Describe the goal"
        description="Some description"
        agents={[maya]}
        selectedAgents={[maya]}
        onSend={vi.fn()}
      />
    )

    expect(screen.getByText("chatPage.sections.leadReady")).toBeTruthy()
    // One portrait in the swapped header, one in her still-visible picker pill.
    const portraits = screen.getAllByRole("img", { name: "Maya" })
    expect(portraits).toHaveLength(2)
    portraits.forEach((portrait) =>
      expect(portrait).toHaveAttribute("src", "/marketplace/avatars/maya.png")
    )
    // The plain (no-lead) description only renders in the fallback header.
    expect(screen.queryByText("Some description")).toBeNull()
  })

  it("keeps the plain header when nobody is selected", () => {
    // Give Maya a real avatar so the assertion below is actually load-bearing -
    // with no avatar at all, "no portrait in the header" would pass trivially
    // even if the hero-swap logic were broken, since PersonaAvatar always
    // falls back to text initials regardless of selection state.
    renderScreen([{ id: 7, name: "Maya", persona_avatar: "/marketplace/avatars/maya.png" }])

    expect(screen.queryByText("chatPage.sections.leadReady")).toBeNull()
    // Her picker pill still renders its own portrait either way - only a
    // second (hero) portrait would indicate the header wrongly swapped.
    expect(screen.getAllByRole("img", { name: "Maya" })).toHaveLength(1)
  })

  it("does not show a right-edge fade when the pill row does not overflow", () => {
    renderScreen([{ id: 7, name: "Maya" }])

    expect(screen.queryByTestId("team-strip")?.nextSibling).toBeNull()
  })

  it("shows the right-edge fade once the pill row actually has scrollable overflow", () => {
    renderScreen([{ id: 7, name: "Maya" }])

    const strip = screen.getByTestId("team-strip")
    Object.defineProperty(strip, "scrollWidth", { configurable: true, value: 800 })
    Object.defineProperty(strip, "clientWidth", { configurable: true, value: 400 })
    Object.defineProperty(strip, "scrollLeft", { configurable: true, value: 0 })
    fireEvent.scroll(strip)

    expect(strip.nextSibling).not.toBeNull()
  })

  it("still forwards selectedAgents to ChatInput (it drives ChatInput's own no-model-selected guard)", () => {
    const maya = { id: 7, name: "Maya" }
    render(
      <ChatStartScreen
        title="Describe the goal"
        agents={[maya]}
        selectedAgents={[maya]}
        onSend={vi.fn()}
      />
    )

    expect(chatInputProps.current?.selectedAgents).toEqual([maya])
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

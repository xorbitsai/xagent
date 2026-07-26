import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({ openFilePreview: vi.fn() }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
    tDynamic: (_key: string, fallback: string) => fallback,
  }),
}))

vi.mock("./TraceEventRenderer", () => ({
  TraceEventRenderer: () => null,
}))

vi.mock("@/components/ui/markdown-renderer", () => ({
  MarkdownRenderer: ({ content }: { content: string }) => <div>{content}</div>,
}))

import { ChatMessage } from "./ChatMessage"

describe("ChatMessage computer access metadata", () => {
  afterEach(() => {
    cleanup()
  })

  it("shows browser computer use on a user message", () => {
    render(
      <ChatMessage
        role="user"
        content="Inspect this page"
        computerRuntimeKind="extension_relay"
      />,
    )

    expect(
      screen.getByText("computerRuntime.messageBadge.browser"),
    ).toBeInTheDocument()
  })

  it("does not add computer metadata to ordinary user messages", () => {
    render(<ChatMessage role="user" content="Summarize this text" />)

    expect(
      screen.queryByText("computerRuntime.messageBadge.browser"),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByText("computerRuntime.messageBadge.desktop"),
    ).not.toBeInTheDocument()
  })
})

import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

const navigation = vi.hoisted(() => ({
  params: { token: "session" },
  searchParamsGet: vi.fn(),
  useSearchParams: vi.fn(),
}))

const pages = vi.hoisted(() => ({
  publicProps: null as Record<string, unknown> | null,
}))

vi.mock("next/navigation", () => ({
  useParams: () => navigation.params,
  useSearchParams: () => {
    navigation.useSearchParams()
    return { get: navigation.searchParamsGet }
  },
}))

vi.mock("@/components/widget/public-agent-chat-page", () => ({
  PublicAgentChatPage: (props: Record<string, unknown>) => {
    pages.publicProps = props
    return <div data-testid="legacy-widget-chat" />
  },
}))

vi.mock("@/components/widget/session-agent-chat-page", () => ({
  SessionAgentChatPage: () => <div data-testid="session-widget-chat" />,
}))

import WidgetChatPage from "./page-client"

afterEach(() => {
  cleanup()
  navigation.params = { token: "session" }
  navigation.searchParamsGet.mockReset()
  navigation.useSearchParams.mockReset()
  pages.publicProps = null
})

describe("WidgetChatPage", () => {
  it("selects Session before reading or mounting any legacy query identity", () => {
    navigation.params = { token: "session" }
    navigation.searchParamsGet.mockImplementation(() => {
      throw new Error("legacy query identity must not be read")
    })

    render(<WidgetChatPage />)

    expect(screen.getByTestId("session-widget-chat")).toBeInTheDocument()
    expect(screen.queryByTestId("legacy-widget-chat")).not.toBeInTheDocument()
    expect(navigation.useSearchParams).not.toHaveBeenCalled()
    expect(navigation.searchParamsGet).not.toHaveBeenCalled()
  })

  it("preserves the legacy Widget route and its exact query inputs", () => {
    navigation.params = { token: "legacy-token" }
    const values: Record<string, string | null> = {
      guest_id: "guest-7",
      agent_id: "42",
      embed_ticket: "embed-ticket",
      widget_key: "widget-key",
    }
    navigation.searchParamsGet.mockImplementation((key: string) => values[key] ?? null)

    render(<WidgetChatPage />)

    expect(screen.getByTestId("legacy-widget-chat")).toBeInTheDocument()
    expect(screen.queryByTestId("session-widget-chat")).not.toBeInTheDocument()
    expect(pages.publicProps).toEqual({
      authMode: "widget",
      routeToken: "legacy-token",
      guestId: "guest-7",
      searchAgentId: 42,
      embedTicket: "embed-ticket",
      widgetKey: "widget-key",
    })
  })
})

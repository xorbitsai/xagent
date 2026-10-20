import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { resolveTranslation, type TranslationKey } from "@/i18n/translations"
import type { Interaction } from "@/contexts/app-context-chat"
import { ClarificationForm } from "./clarification-form"

const language = vi.hoisted(() => ({ value: "en" as "en" | "zh" }))
const sendMessage = vi.hoisted(() => vi.fn())
vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string | number>) => resolveTranslation(language.value, key as TranslationKey, vars),
  }),
}))
vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({ sendMessage, dispatch: vi.fn() }),
}))
vi.mock("./connect-apps-field", () => ({
  ConnectAppsField: ({ interaction }: { interaction: Interaction }) => <button>{interaction.apps?.join(", ")}</button>,
}))

const summary: Interaction = {
  type: "teammate_summary", field: "teammate_summary", label: "Social posts",
  description: "Draft platform-native posts.", placeholder: "Share your topic.", apps: ["Instagram"],
}
afterEach(() => { cleanup(); language.value = "en"; sendMessage.mockClear() })

describe("first-chat summary", () => {
  it("renders a restored summary before a live app card without a clarification submit button", () => {
    render(<ClarificationForm active={false} interactions={[
      summary, { type: "connect_apps", field: "connect_apps", label: "Connect", apps: ["Instagram"] },
    ]} />)
    const card = screen.getByRole("region", { name: "How your teammate works" })
    const connect = screen.getByRole("button", { name: "Instagram" })
    expect(card.compareDocumentPosition(connect) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    for (const text of ["Working from", "You get", "Data access", "Runs as", "Share your topic."]) {
      expect(screen.getByText(text)).toBeInTheDocument()
    }
    expect(connect).toBeEnabled()
    expect(screen.queryByRole("button", { name: /submit/i })).not.toBeInTheDocument()
    expect(sendMessage).not.toHaveBeenCalled()
  })

  it("shows a summary for templates with no apps and never fabricates a connected state", () => {
    render(<ClarificationForm active={false} interactions={[{ ...summary, apps: [] }]} />)
    expect(screen.getByText(/requires no app connections/)).toBeInTheDocument()
    expect(screen.queryByText("Connect your apps")).not.toBeInTheDocument()
    expect(screen.queryByText("Connected")).not.toBeInTheDocument()
  })

  it("escapes template content and relocalizes labels after a locale change", () => {
    const interaction = { ...summary, description: "<script>alert('bad')</script>" }
    const { container, rerender } = render(<ClarificationForm interactions={[interaction]} />)
    expect(container.querySelector("script")).toBeNull()
    expect(screen.getByText(interaction.description)).toBeInTheDocument()
    language.value = "zh"
    rerender(<ClarificationForm interactions={[interaction]} />)
    expect(screen.getByText("工作依据")).toBeInTheDocument()
    expect(screen.getByText("数据访问")).toBeInTheDocument()
  })
})

import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const navigation = vi.hoisted(() => ({
  search: "",
  searchParams: new URLSearchParams(),
  replace: vi.fn(),
}))
const listAgentApiKeysMock = vi.hoisted(() => vi.fn())
const getAgentApiKeyStatsMock = vi.hoisted(() => vi.fn())
const translateMock = vi.hoisted(() => vi.fn((key: string) => key))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: navigation.replace }),
  useSearchParams: () => navigation.searchParams,
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: translateMock }),
}))

vi.mock("@/lib/agent-api-keys-api", () => ({
  createAgentApiKey: vi.fn(),
  deleteAgentApiKey: vi.fn(),
  getAgentApiKeyStats: getAgentApiKeyStatsMock,
  listAgentApiKeys: listAgentApiKeysMock,
  pauseAgentApiKey: vi.fn(),
  regenerateAgentApiKey: vi.fn(),
  resumeAgentApiKey: vi.fn(),
}))

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: vi.fn().mockResolvedValue({ ok: true, json: async () => [] }),
}))

vi.mock("@/components/pages/personal-api-keys-panel", () => ({
  PersonalApiKeysPanel: () => <div>personal-api-keys-panel</div>,
}))

import { ApiKeysPage } from "./api-keys"

function setSearch(search: string) {
  navigation.search = search
  navigation.searchParams = new URLSearchParams(search)
}

describe("ApiKeysPage tabs", () => {
  beforeEach(() => {
    setSearch("")
    navigation.replace.mockReset()
    listAgentApiKeysMock.mockReset()
    getAgentApiKeyStatsMock.mockReset()
    listAgentApiKeysMock.mockResolvedValue([])
    getAgentApiKeyStatsMock.mockResolvedValue({
      total_keys: 0,
      active_keys: 0,
      calls_this_month: 0,
      last_api_call: null,
    })
  })

  afterEach(cleanup)

  it("selects Personal Keys from the tab query parameter", async () => {
    setSearch("tab=personal")

    render(<ApiKeysPage />)

    expect(await screen.findByText("personal-api-keys-panel")).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "apiKeysPage.tabs.personal" })).toHaveAttribute(
      "data-state",
      "active",
    )
    const personalTab = screen.getByRole("tab", { name: "apiKeysPage.tabs.personal" })
    const personalPanel = screen.getByRole("tabpanel")
    expect(personalPanel).toHaveAttribute("aria-labelledby", personalTab.id)
    expect(personalTab).toHaveAttribute("aria-controls", personalPanel.id)
  })

  it("keeps an agent deep link on Agent Keys even when the Personal tab is requested", async () => {
    setSearch("agent=12&tab=personal")

    render(<ApiKeysPage />)

    expect(await screen.findByRole("tab", { name: "apiKeysPage.tabs.agent" })).toHaveAttribute(
      "data-state",
      "active",
    )
    const inactivePersonalPanel = screen.getByText("personal-api-keys-panel").closest('[role="tabpanel"]')
    expect(inactivePersonalPanel).toHaveAttribute("data-state", "inactive")
    const activePanel = screen.getAllByRole("tabpanel").find(
      (panel) => panel.getAttribute("data-state") === "active",
    )
    expect(activePanel).toHaveAttribute(
      "aria-labelledby",
      screen.getByRole("tab", { name: "apiKeysPage.tabs.agent" }).id,
    )
  })

  it("switches an agent deep link to Personal Keys without retaining the agent filter", async () => {
    setSearch("agent=12")

    render(<ApiKeysPage />)

    fireEvent.mouseDown(await screen.findByRole("tab", { name: "apiKeysPage.tabs.personal" }), {
      button: 0,
    })

    expect(navigation.replace).toHaveBeenCalledWith("/api-keys?tab=personal")
  })

  it("clears the agent filter after a Personal Keys round trip", async () => {
    setSearch("agent=12")
    const { rerender } = render(<ApiKeysPage />)

    expect(await screen.findByText("apiKeysPage.filteredByAgent")).toBeInTheDocument()

    fireEvent.mouseDown(screen.getByRole("tab", { name: "apiKeysPage.tabs.personal" }), { button: 0 })
    expect(navigation.replace).toHaveBeenCalledWith("/api-keys?tab=personal")

    setSearch("tab=personal")
    rerender(<ApiKeysPage />)
    fireEvent.mouseDown(screen.getByRole("tab", { name: "apiKeysPage.tabs.agent" }), { button: 0 })
    expect(navigation.replace).toHaveBeenLastCalledWith("/api-keys")

    setSearch("")
    rerender(<ApiKeysPage />)

    expect(screen.queryByText("apiKeysPage.filteredByAgent")).not.toBeInTheDocument()
  })
})

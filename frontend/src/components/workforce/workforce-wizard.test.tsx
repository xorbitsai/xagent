/// <reference types="@testing-library/jest-dom/vitest" />

import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const createWorkforceMock = vi.hoisted(() => vi.fn())
const listAgentOptionsMock = vi.hoisted(() => vi.fn())
const translateMock = vi.hoisted(() => (key: string) => key)

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: translateMock }),
}))

vi.mock("@/lib/workforces-api", () => ({
  createWorkforce: createWorkforceMock,
  listAgentOptions: listAgentOptionsMock,
}))

vi.mock("sonner", () => ({
  toast: { error: vi.fn() },
}))

import { WorkforceWizard } from "./workforce-wizard"

describe("WorkforceWizard", () => {
  beforeEach(() => {
    createWorkforceMock.mockReset()
    listAgentOptionsMock.mockReset()
    listAgentOptionsMock.mockResolvedValue([
      {
        id: 7,
        name: "Manager Agent",
        description: "Coordinates the workforce",
        logo_url: null,
        status: "published",
      },
    ])
  })

  it("keeps the manager dropdown visible outside the step content scroller", async () => {
    render(<WorkforceWizard onCreated={vi.fn()} />)

    await waitFor(() => {
      expect(listAgentOptionsMock).toHaveBeenCalledOnce()
    })

    const panel = screen.getByRole("tabpanel")
    expect(panel).toHaveClass("overflow-visible")
    expect(panel).not.toHaveClass("overflow-y-auto")

    fireEvent.click(screen.getByText("workforces.create.manager.placeholder"))
    fireEvent.click(screen.getByRole("button", { name: /Manager Agent/ }))

    expect(screen.getByText("Manager Agent")).toBeInTheDocument()

    fireEvent.change(screen.getByPlaceholderText("workforces.create.placeholders.name"), {
      target: { value: "Launch Team" },
    })
    fireEvent.click(screen.getByText("common.next"))

    expect(panel).toHaveClass("overflow-y-auto")
    expect(panel).not.toHaveClass("overflow-visible")
  })
})

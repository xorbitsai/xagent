import { fireEvent, render, screen } from "@testing-library/react"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { MyTeamTabs } from "./my-team-tabs"

const routerPush = vi.hoisted(() => vi.fn())
vi.mock("next/navigation", () => ({ useRouter: () => ({ push: routerPush }) }))
vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key === "teamPage.tabs.teammates" ? "Teammates" : "Workforces",
  }),
}))

describe("MyTeamTabs", () => {
  beforeEach(() => {
    routerPush.mockReset()
  })

  it("shows both team types with counts and navigates between their views", () => {
    render(<MyTeamTabs active="teammates" teammateCount={7} workforceCount={5} />)

    expect(screen.getByRole("button", { name: "Teammates 7" })).toHaveAttribute("aria-pressed", "true")
    fireEvent.click(screen.getByRole("button", { name: "Workforces 5" }))
    expect(routerPush).toHaveBeenCalledWith("/workforces")
  })

  it("keeps the teammate list under the shared workforces route", () => {
    render(<MyTeamTabs active="workforces" />)

    fireEvent.click(screen.getByRole("button", { name: "Teammates" }))
    expect(routerPush).toHaveBeenCalledWith("/workforces?view=teammates")
  })
})

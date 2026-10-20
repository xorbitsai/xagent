import { render, waitFor } from "@testing-library/react"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import AgentChatRedirectPage from "./page-client"

const navigation = vi.hoisted(() => ({
  id: "42" as string | string[],
  replace: vi.fn(),
}))

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: navigation.id }),
  useRouter: () => ({ replace: navigation.replace }),
}))

describe("legacy agent chat route", () => {
  beforeEach(() => {
    navigation.id = "42"
    navigation.replace.mockReset()
  })

  it("redirects an old /agent/:id URL to the shared task composer", async () => {
    render(<AgentChatRedirectPage />)

    await waitFor(() => expect(navigation.replace).toHaveBeenCalledWith("/task?agent=42"))
  })

  it("redirects the static export shell to the unassigned task composer", async () => {
    navigation.id = "__shell__"
    render(<AgentChatRedirectPage />)

    await waitFor(() => expect(navigation.replace).toHaveBeenCalledWith("/task"))
  })
})

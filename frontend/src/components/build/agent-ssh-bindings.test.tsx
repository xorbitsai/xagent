import React from "react"
import { cleanup, render, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

// The OSS build ships no SSH routes, so the bindings fetch 404s and used to
// toast "loadFailed" on every builder open: the `!inTeam` guard sat in the
// render path, which hooks run before.

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const inTeamMock = vi.hoisted(() => ({ value: false }))

vi.mock("@/lib/api-wrapper", () => ({ apiRequest: apiRequestMock }))
vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return { ...actual, getApiUrl: () => "http://api.local" }
})
vi.mock("@/components/ui/sonner", () => ({
  toast: { error: toastErrorMock, success: vi.fn() },
}))
vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ inTeam: inTeamMock.value }),
}))
vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

import { AgentSshBindings } from "./agent-ssh-bindings"

// The component compiles with the classic JSX runtime under vitest.
beforeEach(() => {
  vi.stubGlobal("React", React)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe("AgentSshBindings", () => {
  it("skips the bindings fetch when the user is not in a team", () => {
    inTeamMock.value = false
    render(<AgentSshBindings agentId="agent-1" />)
    // render flushes the effect synchronously and the guard returns before any
    // await, so the absence of a call is observable without waitFor.
    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(toastErrorMock).not.toHaveBeenCalled()
  })

  it("loads bindings when the user is in a team", async () => {
    inTeamMock.value = true
    apiRequestMock.mockResolvedValue({ ok: true, json: async () => [] })
    const onCount = vi.fn()
    render(<AgentSshBindings agentId="agent-1" onCount={onCount} />)
    await waitFor(() => expect(onCount).toHaveBeenCalledWith(0))
    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/agents/agent-1/ssh-targets",
    )
  })
})

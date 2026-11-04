import React from "react"
import { act, cleanup, render, waitFor } from "@testing-library/react"
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
// The i18n return value must be referentially stable: `load` is keyed on `t`,
// so a per-render `t` identity turns the mount effect into an unbounded
// fetch/render loop (same hazard documented in agent-builder-publish-errors).
vi.mock("@/contexts/i18n-context", () => {
  const i18n = { locale: "en", t: (key: string) => key }
  return { useI18n: () => i18n }
})

import { AgentSshBindings } from "./agent-ssh-bindings"

// The component compiles with the classic JSX runtime under vitest.
beforeEach(() => {
  vi.stubGlobal("React", React)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

/** Let the load() promise chain settle so its catch branch can run. */
async function flushLoad() {
  await act(async () => {
    await Promise.resolve()
    await Promise.resolve()
  })
}

describe("AgentSshBindings", () => {
  it("skips the bindings fetch when the user is not in a team", async () => {
    inTeamMock.value = false
    const { container } = render(<AgentSshBindings agentId="agent-1" />)

    // toast.error only fires from load()'s catch, two awaits in, so the
    // absence of a toast is only meaningful once those microtasks have run.
    await flushLoad()
    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(toastErrorMock).not.toHaveBeenCalled()
    // The render-level guard still renders nothing — the premise for leaving
    // the targets-dropdown effect keyed on dialogOpen alone.
    expect(container).toBeEmptyDOMElement()
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
    expect(apiRequestMock).toHaveBeenCalledTimes(1)
  })

  it("loads once the user joins a team after mount", async () => {
    inTeamMock.value = false
    apiRequestMock.mockResolvedValue({ ok: true, json: async () => [] })
    const { rerender } = render(<AgentSshBindings agentId="agent-1" />)
    await flushLoad()
    expect(apiRequestMock).not.toHaveBeenCalled()

    // `inTeam` belongs in load()'s deps, not just the render guard: dropping it
    // from the array leaves the fetch stuck on its mount-time value.
    inTeamMock.value = true
    rerender(<AgentSshBindings agentId="agent-1" />)
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledTimes(1))
  })
})

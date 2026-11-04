import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { ApiExample } from "./api-example"

const translateMock = vi.hoisted(() => vi.fn((key: string) => key))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({
    user: {
      id: "1",
      username: "acct_0123456789abcdef0123456789abcdef",
      email: "current@example.com",
    },
    token: "access-token",
    refreshToken: "refresh-token",
  }),
}))
vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: translateMock }),
}))
vi.mock("@/hooks/use-api", () => ({
  apiHooks: {
    useGet: () => ({ data: [], loading: false, error: null, refetch: vi.fn() }),
  },
}))
vi.mock("@/hooks/use-websocket", () => ({
  useWebSocket: () => ({ isConnected: false, connectionError: null }),
}))
vi.mock("@/lib/api-wrapper", () => ({ apiRequest: vi.fn() }))
vi.mock("@/lib/utils", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/utils")>()),
  getApiUrl: () => "http://api.test",
}))

describe("ApiExample current-user label", () => {
  beforeEach(() => {
    vi.stubGlobal("React", React)
    translateMock.mockClear()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    cleanup()
  })

  it("shows email without exposing the opaque username", () => {
    render(<ApiExample />)

    expect(screen.getByText(/current@example\.com/)).toBeInTheDocument()
    expect(screen.queryByText(/acct_0123456789abcdef/)).not.toBeInTheDocument()
  })

  it("uses an example-owned fallback label", () => {
    render(<ApiExample />)

    expect(translateMock).toHaveBeenCalledWith(
      "agent.vibeMode.descriptions.think.examples.apiExample.labels.defaultUser",
    )
    expect(translateMock).not.toHaveBeenCalledWith("sidebar.user.defaultName")
  })
})

import React, { useEffect } from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

const toolCredentialsPanelMock = vi.hoisted(() => vi.fn())

vi.mock("next/navigation", () => ({
  useSearchParams: () => ({
    get: () => null,
  }),
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({
    user: { is_admin: true },
  }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
  }),
}))

vi.mock("@/components/tools/tool-credentials-panel", () => ({
  ToolCredentialsPanel: ({
    scope,
    onAvailabilityChange,
  }: {
    scope: "user" | "instance"
    onAvailabilityChange?: (available: boolean) => void
  }) => {
    toolCredentialsPanelMock(scope)
    useEffect(() => {
      if (scope === "instance") onAvailabilityChange?.(false)
    }, [onAvailabilityChange, scope])
    return <div data-testid={`credentials-panel-${scope}`} />
  },
}))

import ToolCredentialsPage from "./page"

describe("ToolCredentialsPage", () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it("omits the instance-unavailable message when instance credentials are disabled", async () => {
    render(<ToolCredentialsPage />)

    expect(await screen.findByTestId("credentials-panel-user")).toBeInTheDocument()
    expect(toolCredentialsPanelMock).toHaveBeenCalledWith("instance")
    expect(screen.queryByText("tools.credentials.instanceUnavailable")).not.toBeInTheDocument()
  })
})

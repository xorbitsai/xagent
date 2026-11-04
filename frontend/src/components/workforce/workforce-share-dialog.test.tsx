/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { WorkforceDetail } from "@/types/workforce"

const apiRequestMock = vi.hoisted(() => vi.fn())
const copyToClipboardMock = vi.hoisted(() => vi.fn())
const getWorkforceShareLinkMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/clipboard", () => ({
  copyToClipboard: copyToClipboardMock,
}))

vi.mock("@/lib/utils", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/utils")>()),
  getApiUrl: () => "",
}))

vi.mock("@/lib/workforces-api", () => ({
  disableWorkforceShareLink: vi.fn(),
  enableWorkforceShareLink: vi.fn(),
  getWorkforceShareLink: getWorkforceShareLinkMock,
  rotateWorkforceShareLink: vi.fn(),
}))

vi.mock("@/lib/browser-location", () => ({
  getBrowserLocationOrigin: () => "https://cloud.example.test",
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: {
    error: toastErrorMock,
    success: vi.fn(),
  },
}))

import { __resetDeploymentConfigCache } from "@/lib/deployment-config"
import { WorkforceShareDialog } from "./workforce-share-dialog"

const WORKFORCE = {
  id: 42,
  name: "Regional Workforce",
  status: "active",
} as WorkforceDetail

describe("WorkforceShareDialog", () => {
  beforeEach(() => {
    __resetDeploymentConfigCache()
    apiRequestMock.mockReset()
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          deployment_origin: "https://sg-origin.cloud.example.test",
          app_origin: "https://cloud.example.test",
          region: "sg",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )
    getWorkforceShareLinkMock.mockReset()
    getWorkforceShareLinkMock.mockResolvedValue({
      workforce_id: 42,
      share_enabled: true,
      share_token: "regional-share",
      share_updated_at: "2026-07-24T00:00:00Z",
    })
    copyToClipboardMock.mockReset()
    copyToClipboardMock.mockResolvedValue(true)
    toastErrorMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it("keeps share copy disabled until config retry succeeds", async () => {
    apiRequestMock.mockRejectedValueOnce(new Error("deployment config unavailable"))

    render(
      <WorkforceShareDialog
        workforce={WORKFORCE}
        open
        onClose={vi.fn()}
      />,
    )

    expect(await screen.findByText("deployment_config.messages.load_failed")).toBeInTheDocument()
    const copyButton = screen.getByRole("button", { name: "common.copy" })
    expect(screen.getByRole("textbox")).toHaveValue("")
    expect(copyButton).toBeDisabled()
    expect(toastErrorMock).toHaveBeenCalledWith(
      "deployment_config.messages.load_failed",
    )

    expect(screen.getByText("deployment_config.messages.load_failed")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", {
      name: "deployment_config.actions.retry",
    }))

    expect(
      await screen.findByDisplayValue(
        "https://cloud.example.test/change-region?region=sg&next=%2Fshare%2Fregional-share",
      ),
    ).toBeInTheDocument()
    expect(
      screen.queryByText("deployment_config.messages.load_failed"),
    ).not.toBeInTheDocument()
    expect(copyButton).toBeEnabled()
  })

  it("builds a canonical share link that bootstraps the owning region", async () => {
    render(
      <WorkforceShareDialog
        workforce={WORKFORCE}
        open
        onClose={vi.fn()}
      />,
    )

    expect(
      await screen.findByDisplayValue(
        "https://cloud.example.test/change-region?region=sg&next=%2Fshare%2Fregional-share",
      ),
    ).toBeInTheDocument()
  })

  it("reports clipboard failures", async () => {
    copyToClipboardMock.mockResolvedValue(false)

    render(
      <WorkforceShareDialog
        workforce={WORKFORCE}
        open
        onClose={vi.fn()}
      />,
    )

    await screen.findByDisplayValue(
      "https://cloud.example.test/change-region?region=sg&next=%2Fshare%2Fregional-share",
    )
    screen.getByText("common.copy").click()

    await vi.waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        "workforces.share_link.messages.copy_failed",
      )
    })
  })
})

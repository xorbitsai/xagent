/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { WorkforceDetail } from "@/types/workforce"

const apiRequestMock = vi.hoisted(() => vi.fn())
const copyToClipboardMock = vi.hoisted(() => vi.fn())
const getWorkforceWidgetConfigMock = vi.hoisted(() => vi.fn())
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
  getWorkforceWidgetConfig: getWorkforceWidgetConfigMock,
  rotateWorkforceWidgetKey: vi.fn(),
  updateWorkforceWidgetConfig: vi.fn(),
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
import { WorkforceWidgetDialog } from "./workforce-widget-dialog"

const WORKFORCE = {
  id: 42,
  name: "Regional Workforce",
  status: "active",
} as WorkforceDetail

describe("WorkforceWidgetDialog", () => {
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
    getWorkforceWidgetConfigMock.mockReset()
    getWorkforceWidgetConfigMock.mockResolvedValue({
      workforce_id: 42,
      widget_enabled: true,
      widget_key: "wk-regional",
      allowed_domains: ["*"],
    })
    copyToClipboardMock.mockReset()
    copyToClipboardMock.mockResolvedValue(true)
    toastErrorMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it("keeps widget copy disabled until config retry succeeds", async () => {
    apiRequestMock.mockRejectedValueOnce(new Error("deployment config unavailable"))

    render(
      <WorkforceWidgetDialog
        workforce={WORKFORCE}
        open
        onClose={vi.fn()}
      />,
    )

    expect(await screen.findByText("deployment_config.messages.load_failed")).toBeInTheDocument()
    const copyButton = screen.getByTitle("workforces.widget.copy_btn")
    expect(screen.getByText("…")).toBeInTheDocument()
    expect(copyButton).toBeDisabled()
    expect(toastErrorMock).toHaveBeenCalledWith(
      "deployment_config.messages.load_failed",
    )

    expect(screen.getByText("deployment_config.messages.load_failed")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", {
      name: "deployment_config.actions.retry",
    }))

    expect(
      await screen.findByText((content) =>
        content.includes(
          'src="https://sg-origin.cloud.example.test/widget.js"',
        ),
      ),
    ).toBeInTheDocument()
    expect(
      screen.queryByText("deployment_config.messages.load_failed"),
    ).not.toBeInTheDocument()
    expect(copyButton).toBeEnabled()
  })

  it("builds the embed snippet from the advertised deployment origin", async () => {
    render(
      <WorkforceWidgetDialog
        workforce={WORKFORCE}
        open
        onClose={vi.fn()}
      />,
    )

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledOnce()
    })
    expect(
      await screen.findByText((content) =>
        content.includes(
          'src="https://sg-origin.cloud.example.test/widget.js"',
        ),
      ),
    ).toBeInTheDocument()
  })

  it("reports clipboard failures", async () => {
    copyToClipboardMock.mockResolvedValue(false)

    render(
      <WorkforceWidgetDialog
        workforce={WORKFORCE}
        open
        onClose={vi.fn()}
      />,
    )

    await screen.findByText((content) =>
      content.includes(
        'src="https://sg-origin.cloud.example.test/widget.js"',
      ),
    )
    screen.getByTitle("workforces.widget.copy_btn").click()

    await vi.waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        "workforces.widget.messages.copy_failed",
      )
    })
  })
})

/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const createWorkforceApiKeyMock = vi.hoisted(() => vi.fn())
const apiRequestMock = vi.hoisted(() => vi.fn())
const copyToClipboardMock = vi.hoisted(() => vi.fn())
const listAgentApiKeysMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const translateMock = vi.hoisted(() => (key: string) => key)

vi.mock("@/lib/agent-api-keys-api", () => ({
  createWorkforceApiKey: createWorkforceApiKeyMock,
  listAgentApiKeys: listAgentApiKeysMock,
}))

vi.mock("@/lib/clipboard", () => ({
  copyToClipboard: copyToClipboardMock,
}))

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/utils")>()),
  getApiUrl: () => "https://configured-api.example.test",
}))

vi.mock("@/lib/browser-location", () => ({
  getBrowserLocationOrigin: () => "https://cloud.example.test",
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: {
    error: toastErrorMock,
  },
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: translateMock }),
}))

import { __resetDeploymentConfigCache } from "@/lib/deployment-config"
import { DeployWorkforceDialog } from "./deploy-workforce-dialog"

describe("DeployWorkforceDialog", () => {
  beforeEach(() => {
    __resetDeploymentConfigCache()
    createWorkforceApiKeyMock.mockReset()
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
    copyToClipboardMock.mockReset()
    copyToClipboardMock.mockResolvedValue(true)
    listAgentApiKeysMock.mockReset()
    toastErrorMock.mockReset()
  })

  it("builds API and SDK snippets from the advertised deployment origin", async () => {
    listAgentApiKeysMock.mockResolvedValue([])

    render(
      <DeployWorkforceDialog
        open
        workforceId={42}
        workforceName="Regional Workforce"
        onClose={vi.fn()}
      />,
    )

    expect(
      await screen.findByText((content) =>
        content.includes(
          "https://sg-origin.cloud.example.test/v1/workforces/42/runs",
        ),
      ),
    ).toBeInTheDocument()
  })

  afterEach(() => {
    cleanup()
  })

  it("keeps API copy disabled until config retry succeeds", async () => {
    apiRequestMock.mockRejectedValueOnce(
      new Error("deployment config unavailable"),
    )
    listAgentApiKeysMock.mockResolvedValue([])

    render(
      <DeployWorkforceDialog
        open
        workforceId={42}
        workforceName="Regional Workforce"
        onClose={vi.fn()}
      />,
    )

    expect(await screen.findByText("deployment_config.messages.load_failed")).toBeInTheDocument()
    const copyButton = screen.getByTitle("deploy_workforce.copy")
    expect(copyButton).toBeDisabled()
    expect(
      screen.queryByText((content) =>
        content.includes("https://cloud.example.test/v1/workforces/42/runs"),
      ),
    ).not.toBeInTheDocument()
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
          "https://sg-origin.cloud.example.test/v1/workforces/42/runs",
        ),
      ),
    ).toBeInTheDocument()
    expect(
      screen.queryByText("deployment_config.messages.load_failed"),
    ).not.toBeInTheDocument()
    expect(copyButton).toBeEnabled()
  })

  it("reports snippet clipboard failures", async () => {
    copyToClipboardMock.mockResolvedValue(false)
    listAgentApiKeysMock.mockResolvedValue([])

    render(
      <DeployWorkforceDialog
        open
        workforceId={42}
        workforceName="Regional Workforce"
        onClose={vi.fn()}
      />,
    )

    const copyButton = screen.getByTitle("deploy_workforce.copy")
    await waitFor(() => expect(copyButton).toBeEnabled())
    copyButton.click()

    await vi.waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        "deploy_workforce.copy_failed",
      )
    })
  })

  it("keeps a newly created secret visible when refreshing the key list fails", async () => {
    listAgentApiKeysMock
      .mockResolvedValueOnce([])
      .mockRejectedValueOnce(new Error("refresh failed"))
    createWorkforceApiKeyMock.mockResolvedValue({
      full_key: "xag_test_one_shot_secret",
      key_prefix: "test",
      created_at: "2026-07-23T00:00:00Z",
    })

    render(
      <DeployWorkforceDialog
        open
        workforceId={42}
        workforceName="Review Workforce"
        onClose={vi.fn()}
      />,
    )

    await waitFor(() => {
      expect(listAgentApiKeysMock).toHaveBeenCalledWith({ workforceId: 42 })
    })

    fireEvent.change(
      screen.getByPlaceholderText("deploy_workforce.label_placeholder"),
      { target: { value: "CI" } },
    )
    fireEvent.click(screen.getByRole("button", { name: "deploy_workforce.create_key" }))

    expect(await screen.findByText("xag_test_one_shot_secret")).toBeInTheDocument()
    expect(createWorkforceApiKeyMock).toHaveBeenCalledWith(42, "CI")
    expect(toastErrorMock).toHaveBeenCalledWith("apiKeysPage.messages.loadFailed")
    expect(toastErrorMock).not.toHaveBeenCalledWith(
      "apiKeysPage.messages.createFailed",
    )
  })
})

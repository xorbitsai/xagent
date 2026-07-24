import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const createWorkforceApiKeyMock = vi.hoisted(() => vi.fn())
const listAgentApiKeysMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const translateMock = vi.hoisted(() => (key: string) => key)

vi.mock("@/lib/agent-api-keys-api", () => ({
  createWorkforceApiKey: createWorkforceApiKeyMock,
  listAgentApiKeys: listAgentApiKeysMock,
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: {
    error: toastErrorMock,
  },
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: translateMock }),
}))

import { DeployWorkforceDialog } from "./deploy-workforce-dialog"

describe("DeployWorkforceDialog", () => {
  beforeEach(() => {
    createWorkforceApiKeyMock.mockReset()
    listAgentApiKeysMock.mockReset()
    toastErrorMock.mockReset()
  })

  afterEach(() => {
    cleanup()
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

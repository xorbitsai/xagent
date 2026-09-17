/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

import { DeploymentConfigErrorAlert } from "./deployment-config-error-alert"

describe("DeploymentConfigErrorAlert", () => {
  it("remains retryable when another configuration request fails", async () => {
    let rejectRetry: ((error: Error) => void) | undefined
    const onRetry = vi.fn(
      () => new Promise<void>((_resolve, reject) => {
        rejectRetry = reject
      }),
    )

    render(<DeploymentConfigErrorAlert onRetry={onRetry} />)

    const retryButton = screen.getByRole("button", {
      name: "deployment_config.actions.retry",
    })
    fireEvent.click(retryButton)

    expect(retryButton).toBeDisabled()
    expect(onRetry).toHaveBeenCalledOnce()

    await act(async () => {
      rejectRetry?.(new Error("deployment config unavailable"))
    })

    await waitFor(() => expect(retryButton).toBeEnabled())
    expect(screen.getByRole("alert")).toHaveTextContent(
      "deployment_config.messages.load_failed",
    )
  })
})

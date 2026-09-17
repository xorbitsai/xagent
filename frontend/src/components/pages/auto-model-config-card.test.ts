import { render, waitFor } from "@testing-library/react"
import { createElement } from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import type { Model } from "./models"
import { AutoModelConfigCard, guessProfile } from "./auto-model-config-card"

const { apiRequestMock, toastErrorMock } = vi.hoisted(() => ({
  apiRequestMock: vi.fn(),
  toastErrorMock: vi.fn(),
}))

vi.mock("@/lib/api-wrapper", () => ({ apiRequest: apiRequestMock }))
vi.mock("@/components/ui/sonner", () => ({
  toast: { error: toastErrorMock, success: vi.fn() },
}))
vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

const model: Model = {
  id: 1,
  model_id: "saved-gpt",
  category: "llm",
  model_provider: "openai",
  model_name: "gpt-5.5",
  is_active: true,
  is_owner: true,
  can_edit: true,
  can_delete: true,
  is_shared: false,
}

describe("guessProfile", () => {
  it("matches a profile when aliases are missing or null", () => {
    expect(
      guessProfile(model, [
        { id: "other/model", aliases: null, input_modalities: ["text"] },
        { id: "openai/gpt-5.5", input_modalities: ["text"] },
      ]),
    ).toBe("openai/gpt-5.5")
  })
})

describe("AutoModelConfigCard", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
  })

  it("stays hidden without a router installation and does not show an error", async () => {
    apiRequestMock
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({ configured: false, candidates: [] }),
          { status: 200 },
        ),
      )
      .mockResolvedValueOnce(new Response(null, { status: 503 }))

    const view = render(
      createElement(AutoModelConfigCard, {
        models: [],
        onSuccess: vi.fn(),
      }),
    )

    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(view.container).toBeEmptyDOMElement())
    expect(toastErrorMock).not.toHaveBeenCalled()
  })
})

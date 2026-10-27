import { fireEvent, render, screen, waitFor } from "@testing-library/react"
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
vi.mock("@/components/ui/multi-select", () => ({ MultiSelect: () => null }))
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

  it("keeps repeated profiles valid in sequential order", async () => {
    const secondModel = {
      ...model,
      id: 2,
      model_id: "official-gpt",
      model_name: "gpt-5.5-official",
    }
    apiRequestMock
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            configured: true,
            mode: "sequential_failover",
            candidates: [
              {
                routing_model_id: "openai/gpt-5.5",
                target_model_id: 1,
                target_model: model,
              },
              {
                routing_model_id: "openai/gpt-5.5",
                target_model_id: 2,
                target_model: secondModel,
              },
            ],
          }),
          { status: 200 },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify([
            { id: "openai/gpt-5.5", input_modalities: ["text"] },
          ]),
          { status: 200 },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            configured: true,
            mode: "sequential_failover",
            auto_model: model,
            candidates: [
              {
                routing_model_id: "openai/gpt-5.5",
                target_model_id: 2,
                target_model: secondModel,
              },
              {
                routing_model_id: "openai/gpt-5.5",
                target_model_id: 1,
                target_model: model,
              },
            ],
          }),
          { status: 200 },
        ),
      )

    const onSuccess = vi.fn().mockResolvedValue(undefined)

    render(
      createElement(AutoModelConfigCard, {
        models: [model, secondModel],
        onSuccess,
      }),
    )

    await waitFor(() => expect(screen.getByText("models.auto.edit")).toBeEnabled())
    fireEvent.click(screen.getByText("models.auto.edit"))

    expect(await screen.findByText("1. gpt-5.5")).toBeInTheDocument()
    expect(screen.getByText("2. gpt-5.5-official")).toBeInTheDocument()
    expect(screen.queryByText("models.auto.duplicateProfile")).not.toBeInTheDocument()
    expect(screen.queryByText("models.auto.fallback")).not.toBeInTheDocument()

    fireEvent.click(screen.getAllByLabelText("models.auto.moveUp")[1])
    expect(screen.getByText("1. gpt-5.5-official")).toBeInTheDocument()
    expect(screen.getByText("2. gpt-5.5")).toBeInTheDocument()

    fireEvent.click(screen.getByText("models.auto.save"))
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledTimes(3))
    const request = apiRequestMock.mock.calls[2][1] as RequestInit
    expect(JSON.parse(String(request.body))).toMatchObject({
      mode: "sequential_failover",
      fallback_model_id: null,
      candidates: [
        { target_model_id: 2, routing_model_id: "openai/gpt-5.5" },
        { target_model_id: 1, routing_model_id: "openai/gpt-5.5" },
      ],
    })
    await waitFor(() => expect(onSuccess).toHaveBeenCalledOnce())
  })
})

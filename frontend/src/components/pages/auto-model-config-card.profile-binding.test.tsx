import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react"
import React, { createContext, useContext } from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { resolveTranslation, type TranslationKey } from "@/i18n/translations"

import type { Model } from "./models"
import { AutoModelConfigCard } from "./auto-model-config-card"

const { apiRequestMock } = vi.hoisted(() => ({ apiRequestMock: vi.fn() }))

vi.mock("@/lib/api-wrapper", () => ({ apiRequest: apiRequestMock }))
vi.mock("@/components/ui/sonner", () => ({
  toast: { error: vi.fn(), success: vi.fn() },
}))
vi.mock("@/components/ui/multi-select", () => ({ MultiSelect: () => null }))
vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: TranslationKey, vars?: { name: string }) =>
      key === "models.auto.moveUp" || key === "models.auto.moveDown"
        ? resolveTranslation("en", key, vars)
        : key,
  }),
}))
// Radix Select needs pointer APIs jsdom lacks; render each item as a button so
// a row can be rebound.
vi.mock("@/components/ui/select-radix", () => {
  const Choose = createContext<(value: string) => void>(() => {})
  return {
    Select: ({
      onValueChange,
      children,
    }: {
      onValueChange?: (value: string) => void
      children: React.ReactNode
    }) => (
      <Choose.Provider value={onValueChange ?? (() => {})}>
        <div>{children}</div>
      </Choose.Provider>
    ),
    SelectTrigger: ({
      className,
      children,
    }: {
      className?: string
      children: React.ReactNode
    }) => (
      <div data-testid="select-trigger" className={className}>
        {children}
      </div>
    ),
    SelectValue: () => null,
    SelectContent: ({ children }: { children: React.ReactNode }) => (
      <div>{children}</div>
    ),
    SelectItem: ({
      value,
      children,
    }: {
      value: string
      children: React.ReactNode
    }) => {
      const choose = useContext(Choose)
      return (
        <button type="button" onClick={() => choose(value)}>
          {children}
        </button>
      )
    },
  }
})

// Several tests render in this file, and Testing Library only unmounts
// automatically when the test globals are enabled, which this project does not.
afterEach(() => {
  cleanup()
})

const base = {
  category: "llm",
  model_provider: "openai",
  is_active: true,
  is_owner: true,
  can_edit: true,
  can_delete: true,
  is_shared: false,
}
const stale: Model = { ...base, id: 1, model_id: "stale", model_name: "stale-model" }
const healthy: Model = {
  ...base,
  id: 2,
  model_id: "healthy",
  model_name: "healthy-model",
}

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200 })
}

function row(modelName: string) {
  // The fallback select also lists the model name, outside any binding row.
  const element = screen
    .getAllByText(modelName)
    .map((match) => match.closest("div.rounded-lg"))
    .find(Boolean)
  if (!(element instanceof HTMLElement)) throw new Error(`no row for ${modelName}`)
  return element
}

describe("AutoModelConfigCard with a profile missing from the catalog", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
  })

  it("marks the stale row and keeps Save disabled until it is rebound", async () => {
    apiRequestMock
      .mockResolvedValueOnce(
        jsonResponse({
          configured: true,
          mode: "routing",
          fallback_model_id: 1,
          candidates: [
            {
              routing_model_id: "removed/profile",
              target_model_id: 1,
              target_model: stale,
            },
            {
              routing_model_id: "openai/gpt-5.5",
              target_model_id: 2,
              target_model: healthy,
            },
          ],
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse([
          { id: "openai/gpt-5.5", input_modalities: ["text"] },
          { id: "openai/gpt-5.5-mini", input_modalities: ["text"] },
        ]),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          configured: true,
          mode: "routing",
          fallback_model_id: 1,
          auto_model: healthy,
          candidates: [],
        }),
      )

    render(
      <AutoModelConfigCard
        models={[stale, healthy]}
        onSuccess={vi.fn().mockResolvedValue(undefined)}
      />,
    )

    await waitFor(() => expect(screen.getByText("models.auto.edit")).toBeEnabled())
    fireEvent.click(screen.getByText("models.auto.edit"))

    const staleRow = row("stale-model")
    expect(within(staleRow).getByText("models.auto.unknownProfile")).toBeInTheDocument()
    expect(within(staleRow).getByTestId("select-trigger")).toHaveClass(
      "border-destructive",
    )
    expect(
      within(row("healthy-model")).queryByText("models.auto.unknownProfile"),
    ).not.toBeInTheDocument()
    expect(screen.getByText("models.auto.save")).toBeDisabled()

    fireEvent.click(within(staleRow).getByText("openai/gpt-5.5-mini"))

    expect(screen.queryByText("models.auto.unknownProfile")).not.toBeInTheDocument()
    expect(screen.getByText("models.auto.save")).toBeEnabled()
    fireEvent.click(screen.getByText("models.auto.save"))
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledTimes(3))
    const request = apiRequestMock.mock.calls[2][1] as RequestInit
    expect(JSON.parse(String(request.body))).toMatchObject({
      candidates: [
        { target_model_id: 1, routing_model_id: "openai/gpt-5.5-mini" },
        { target_model_id: 2, routing_model_id: "openai/gpt-5.5" },
      ],
    })
  })
})

describe("AutoModelConfigCard with a profile whose inputs differ from the model", () => {
  const visionModel: Model = {
    ...base,
    id: 3,
    model_id: "vision",
    model_name: "vision-model",
    abilities: ["chat", "vision"],
  }
  const textModel: Model = {
    ...base,
    id: 4,
    model_id: "text",
    model_name: "text-model",
    abilities: ["chat"],
  }
  const profiles = [
    { id: "openai/gpt-5.5", input_modalities: ["text"] },
    { id: "openai/gpt-5.5-mini", input_modalities: ["text"] },
    { id: "openai/gpt-5.5-vision", input_modalities: ["text", "image"] },
  ]
  const savedConfig = (mode: "routing" | "sequential_failover") => ({
    configured: true,
    mode,
    ...(mode === "routing" ? { fallback_model_id: 4 } : {}),
    candidates: [
      {
        routing_model_id: "openai/gpt-5.5",
        target_model_id: 3,
        target_model: visionModel,
      },
      {
        routing_model_id: "openai/gpt-5.5-mini",
        target_model_id: 4,
        target_model: textModel,
      },
    ],
  })

  beforeEach(() => {
    apiRequestMock.mockReset()
  })

  it("marks a mismatched row in routing mode and keeps Save disabled until it is rebound", async () => {
    apiRequestMock
      .mockResolvedValueOnce(jsonResponse(savedConfig("routing")))
      .mockResolvedValueOnce(jsonResponse(profiles))
      .mockResolvedValueOnce(
        jsonResponse({
          configured: true,
          mode: "routing",
          fallback_model_id: 4,
          auto_model: textModel,
          candidates: [],
        }),
      )
    render(
      <AutoModelConfigCard
        models={[visionModel, textModel]}
        onSuccess={vi.fn().mockResolvedValue(undefined)}
      />,
    )
    await waitFor(() => expect(screen.getByText("models.auto.edit")).toBeEnabled())
    fireEvent.click(screen.getByText("models.auto.edit"))

    const mismatchedRow = row("vision-model")
    expect(
      within(mismatchedRow).getByText("models.auto.profileModalityMismatch"),
    ).toBeInTheDocument()
    expect(
      within(mismatchedRow).queryByText("models.auto.unknownProfile"),
    ).not.toBeInTheDocument()
    expect(within(mismatchedRow).getByTestId("select-trigger")).toHaveClass(
      "border-destructive",
    )
    const textRow = row("text-model")
    expect(
      within(textRow).queryByText("models.auto.profileModalityMismatch"),
    ).not.toBeInTheDocument()
    expect(within(textRow).getByTestId("select-trigger")).not.toHaveClass(
      "border-destructive",
    )
    expect(screen.getByText("models.auto.save")).toBeDisabled()

    fireEvent.click(within(mismatchedRow).getByText("openai/gpt-5.5-vision"))

    expect(
      screen.queryByText("models.auto.profileModalityMismatch"),
    ).not.toBeInTheDocument()
    expect(screen.getByText("models.auto.save")).toBeEnabled()
    fireEvent.click(screen.getByText("models.auto.save"))
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledTimes(3))
    const request = apiRequestMock.mock.calls[2][1] as RequestInit
    expect(JSON.parse(String(request.body))).toMatchObject({
      mode: "routing",
      fallback_model_id: 4,
      candidates: [
        { target_model_id: 3, routing_model_id: "openai/gpt-5.5-vision" },
        { target_model_id: 4, routing_model_id: "openai/gpt-5.5-mini" },
      ],
    })
  })

  it("does not mark the same binding in sequential failover, which saves it", async () => {
    apiRequestMock
      .mockResolvedValueOnce(jsonResponse(savedConfig("sequential_failover")))
      .mockResolvedValueOnce(jsonResponse(profiles))
    render(
      <AutoModelConfigCard
        models={[visionModel, textModel]}
        onSuccess={vi.fn().mockResolvedValue(undefined)}
      />,
    )
    await waitFor(() => expect(screen.getByText("models.auto.edit")).toBeEnabled())
    fireEvent.click(screen.getByText("models.auto.edit"))

    // Sequential rows prefix the position, as in "1. vision-model".
    expect(
      screen.queryByText("models.auto.profileModalityMismatch"),
    ).not.toBeInTheDocument()
    expect(
      within(row("1. vision-model")).getByTestId("select-trigger"),
    ).not.toHaveClass("border-destructive")
    expect(screen.getByText("models.auto.save")).toBeEnabled()
  })
})


describe("AutoModelConfigCard with an inactive saved candidate", () => {
  const inactive: Model = {
    ...base,
    id: 3,
    model_id: "inactive",
    model_name: "inactive-model",
    is_active: false,
  }
  const profiles = [
    { id: "openai/gpt-5.5", input_modalities: ["text"] },
    { id: "openai/gpt-5.5-mini", input_modalities: ["text"] },
  ]
  const savedConfig = {
    configured: true,
    mode: "sequential_failover",
    fallback_model_id: null,
    candidates: [
      { routing_model_id: profiles[0].id, target_model_id: stale.id, target_model: stale },
      { routing_model_id: profiles[0].id, target_model_id: inactive.id, target_model: inactive },
      { routing_model_id: profiles[1].id, target_model_id: healthy.id, target_model: healthy },
    ],
  }

  beforeEach(() => {
    apiRequestMock.mockReset()
    apiRequestMock
      .mockResolvedValueOnce(jsonResponse(savedConfig))
      .mockResolvedValueOnce(jsonResponse(profiles))
      .mockResolvedValueOnce(jsonResponse({ ...savedConfig, auto_model: healthy }))
  })

  async function openConfig() {
    render(
      <AutoModelConfigCard
        models={[stale, inactive, healthy]}
        onSuccess={vi.fn().mockResolvedValue(undefined)}
      />,
    )
    await waitFor(() => expect(screen.getByText("models.auto.edit")).toBeEnabled())
    fireEvent.click(screen.getByText("models.auto.edit"))
  }

  async function saveConfig() {
    fireEvent.click(screen.getByText("models.auto.save"))
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledTimes(3))
    const request = apiRequestMock.mock.calls[2][1] as RequestInit
    return JSON.parse(String(request.body))
  }

  it("moves visible candidates in either direction and saves the displayed order", async () => {
    await openConfig()

    expect(row("1. stale-model")).toBeInTheDocument()
    expect(row("2. healthy-model")).toBeInTheDocument()
    const moveUp = screen.getByRole("button", { name: "Move healthy-model up" })
    const moveDown = screen.getByRole("button", { name: "Move healthy-model down" })
    expect(screen.getByRole("button", { name: "Move stale-model up" })).toBeDisabled()
    expect(moveDown).toBeDisabled()

    fireEvent.click(moveUp)
    expect(row("1. healthy-model")).toBeInTheDocument()
    expect(row("2. stale-model")).toBeInTheDocument()
    expect(moveUp).toBeDisabled()
    expect(moveDown).toBeEnabled()
    expect(screen.getByRole("button", { name: "Move stale-model down" })).toBeDisabled()

    fireEvent.click(moveDown)
    expect(row("1. stale-model")).toBeInTheDocument()
    expect(row("2. healthy-model")).toBeInTheDocument()
    fireEvent.click(moveUp)

    expect(await saveConfig()).toMatchObject({
      mode: "sequential_failover",
      fallback_model_id: null,
      candidates: [
        { target_model_id: healthy.id, routing_model_id: profiles[1].id },
        { target_model_id: stale.id, routing_model_id: profiles[0].id },
      ],
    })
  })

  it("defaults to the first visible candidate when switching to routing", async () => {
    await openConfig()
    fireEvent.click(screen.getByRole("button", { name: "models.auto.modeRouting" }))

    expect(screen.getByText("models.auto.save")).toBeEnabled()
    expect(await saveConfig()).toMatchObject({
      mode: "routing",
      fallback_model_id: stale.id,
      candidates: [
        { target_model_id: stale.id, routing_model_id: profiles[0].id },
        { target_model_id: healthy.id, routing_model_id: profiles[1].id },
      ],
    })
  })
})

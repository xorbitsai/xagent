import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const i18nMock = vi.hoisted(() => ({ locale: "en" as "en" | "zh" }))

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", () => ({
  cn: (...classes: Array<string | false | null | undefined>) =>
    classes.filter(Boolean).join(" "),
  getApiUrl: () => "http://api.local",
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    locale: i18nMock.locale,
    t: (key: string, vars?: Record<string, string | number>) => {
      const message = {
        "chatPage.tokenUsage.input": "Input tokens",
        "chatPage.tokenUsage.output": "Output tokens",
        "chatPage.tokenUsage.cached": "Cached input tokens",
        "chatPage.tokenUsage.inputShort": "Input",
        "chatPage.tokenUsage.outputShort": "Output",
        "chatPage.tokenUsage.cachedShort": "Cached",
        "chatPage.tokenUsage.cachedShare": "{pct}% cached",
        "chatPage.tokenUsage.oneModel": "{count} model",
        "chatPage.tokenUsage.models": "{count} models",
        "chatPage.tokenUsage.oneModelWithUnattributed": "{count} model + {unattributed} unattributed",
        "chatPage.tokenUsage.modelsWithUnattributed": "{count} models + {unattributed} unattributed",
        "chatPage.tokenUsage.unattributedCount": "{count} unattributed",
        "chatPage.tokenUsage.byModel": "Usage by model",
        "chatPage.tokenUsage.model": "Model",
        "chatPage.tokenUsage.unknownModel": "Unknown model",
        "chatPage.tokenUsage.unattributed": "Unattributed",
      }[key] ?? key
      return Object.entries(vars ?? {}).reduce(
        (result, [name, value]) => result.replaceAll(`{${name}}`, String(value)),
        message,
      )
    },
  }),
}))

import {
  formatExactTokenCount,
  formatTokenCount,
  TokenUsageDisplay,
} from "./TokenUsageDisplay"

describe("TokenUsageDisplay", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    i18nMock.locale = "en"
  })

  afterEach(() => {
    cleanup()
  })

  it("formats large token counts with compact lowercase suffixes", () => {
    expect(formatTokenCount(999)).toBe("999")
    expect(formatTokenCount(37_499)).toBe("37.5k")
    expect(formatTokenCount(2_755_525)).toBe("2.76m")
    expect(formatExactTokenCount(2_755_525)).toBe("2,755,525")
    expect(formatTokenCount(2_755_525, "zh")).toBe("275.55万")
  })

  it.each([-1, Number.NaN, Number.POSITIVE_INFINITY])(
    "normalizes invalid token count %s to zero",
    (value) => {
      expect(formatTokenCount(value)).toBe("0")
      expect(formatExactTokenCount(value)).toBe("0")
    },
  )

  it("uses the active locale when rendering token counts", async () => {
    i18nMock.locale = "zh"
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 2_755_525,
          output_tokens: 0,
          total_tokens: 2_755_525,
          llm_calls: 1,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={6} isRunning={false} />)

    expect(await screen.findByText("275.55万")).toHaveAttribute("title", "2,755,525")
  })

  it("shows aggregate counts and exposes each model in a popover", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 2_755_525,
          output_tokens: 37_499,
          total_tokens: 2_793_024,
          llm_calls: 3,
          model_usage: [
            {
              model_id: "main",
              model_name: "deepseek/deepseek-v4-pro",
              input_tokens: 2_700_000,
              output_tokens: 35_000,
            },
            {
              model_id: "compact",
              model_name: "deepseek/deepseek-v4-flash",
              input_tokens: 55_525,
              output_tokens: 2_499,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={7} isRunning={false} />)

    await waitFor(() => {
      expect(screen.getByText("2.76m")).toHaveAttribute("title", "2,755,525")
    })
    expect(screen.getByText("37.5k")).toHaveAttribute("title", "37,499")
    expect(screen.getByText("Input")).toHaveAttribute("title", "Input tokens")
    expect(screen.getByText("Output")).toHaveAttribute("title", "Output tokens")

    fireEvent.click(screen.getByRole("button", { name: /2 models/ }))

    const modelUsageDialog = await screen.findByRole("dialog")
    expect(modelUsageDialog).toHaveClass("w-[32rem]")
    expect(screen.getAllByText("Input")).toHaveLength(2)
    expect(screen.getAllByText("Output")).toHaveLength(2)
    expect(screen.getByText("deepseek/deepseek-v4-pro")).toBeInTheDocument()
    expect(screen.getByText("deepseek/deepseek-v4-flash")).toBeInTheDocument()
    expect(screen.getByText("main")).toBeInTheDocument()
    expect(screen.getByText("compact")).toBeInTheDocument()
    expect(screen.getByText("2.7m")).toHaveAttribute("title", "2,700,000")
    expect(screen.getByText("55.53k")).toHaveAttribute("title", "55,525")
  })

  it("uses the singular label and renders an id-only model without a sub-label", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 12,
          output_tokens: 3,
          total_tokens: 15,
          llm_calls: 1,
          model_usage: [
            {
              model_id: "router:model-only",
              model_name: "",
              input_tokens: 12,
              output_tokens: 3,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={8} isRunning={false} />)

    const modelsButton = await screen.findByRole("button", { name: /^1 model$/ })
    expect(modelsButton).toHaveAccessibleName("1 model")
    expect(screen.queryByRole("button", { name: "1 models" })).not.toBeInTheDocument()
    fireEvent.click(modelsButton)

    expect(await screen.findByText("router:model-only")).toBeInTheDocument()
    expect(screen.queryByText("Unattributed")).not.toBeInTheDocument()
  })

  it("counts and labels unknown model usage as unattributed", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 12,
          output_tokens: 3,
          total_tokens: 15,
          llm_calls: 1,
          model_usage: [
            {
              model_id: "",
              model_name: "",
              input_tokens: 12,
              output_tokens: 3,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={10} isRunning={false} />)

    fireEvent.click(await screen.findByRole("button", { name: /^1 unattributed$/ }))
    expect(await screen.findByText("Unknown model")).toBeInTheDocument()
  })

  it("separates attributed models from name-only usage in the trigger count", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 50,
          output_tokens: 0,
          total_tokens: 50,
          llm_calls: 2,
          model_usage: [
            {
              model_id: "main",
              model_name: "shared-name",
              input_tokens: 20,
              output_tokens: 0,
            },
            {
              model_id: "",
              model_name: "shared-name",
              input_tokens: 30,
              output_tokens: 0,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={11} isRunning={false} />)

    fireEvent.click(
      await screen.findByRole("button", { name: /^1 model \+ 1 unattributed$/ }),
    )
    expect(await screen.findAllByText("shared-name")).toHaveLength(2)
    expect(screen.getByText("Unattributed")).toBeInTheDocument()
  })

  it.each([undefined, []])(
    "does not render a model popover without model usage (%s)",
    async (modelUsage) => {
      apiRequestMock.mockResolvedValue(
        new Response(
          JSON.stringify({
            input_tokens: 12,
            output_tokens: 3,
            total_tokens: 15,
            llm_calls: 1,
            model_usage: modelUsage,
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      )

      render(<TokenUsageDisplay taskId={9} isRunning={false} />)

      await screen.findByText("12")
      expect(screen.queryByRole("button")).not.toBeInTheDocument()
    },
  )
})

describe("TokenUsageDisplay cached tokens", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    i18nMock.locale = "en"
  })

  afterEach(() => {
    cleanup()
  })

  it("shows the cached share and a per-model cached column", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 100_000,
          output_tokens: 5_000,
          total_tokens: 105_000,
          llm_calls: 2,
          cached_input_tokens: 75_000,
          model_usage: [
            {
              model_id: "main",
              model_name: "claude-sonnet-5",
              input_tokens: 100_000,
              output_tokens: 5_000,
              cached_input_tokens: 75_000,
              cache_write_input_tokens: 1_000,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={11} isRunning={false} />)

    const share = await screen.findByText("75% cached")
    expect(share).toHaveAttribute("title", "Cached input tokens: 75,000")

    fireEvent.click(screen.getByRole("button", { name: /^1 model$/ }))
    await screen.findByRole("dialog")
    expect(screen.getByText("Cached")).toHaveAttribute(
      "title",
      "Cached input tokens",
    )
    expect(screen.getByText("75k")).toHaveAttribute("title", "75,000")
  })

  it("suppresses the cached share when input tokens are zero", async () => {
    // Malformed/partial backend data: cached > 0 with input == 0 must not
    // render a NaN/Infinity percentage.
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 0,
          output_tokens: 5,
          total_tokens: 5,
          llm_calls: 1,
          cached_input_tokens: 75_000,
          model_usage: [],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={13} isRunning={false} />)

    await screen.findByText("Input")
    expect(screen.queryByText(/% cached/)).not.toBeInTheDocument()
    expect(screen.queryByText(/NaN|Infinity/)).not.toBeInTheDocument()
  })

  it("hides the cached share when the backend reports no cache usage", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 100,
          output_tokens: 5,
          total_tokens: 105,
          llm_calls: 1,
          model_usage: [
            {
              model_id: "main",
              model_name: "gpt-4.1",
              input_tokens: 100,
              output_tokens: 5,
              cached_input_tokens: 0,
              cache_write_input_tokens: 0,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={12} isRunning={false} />)

    await screen.findByText("Input")
    expect(screen.queryByText(/% cached/)).not.toBeInTheDocument()
  })
})

import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())

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
    t: (key: string, vars?: Record<string, string | number>) => {
      const message = {
        "chatPage.tokenUsage.input": "Input tokens",
        "chatPage.tokenUsage.output": "Output tokens",
        "chatPage.tokenUsage.inputShort": "Input",
        "chatPage.tokenUsage.outputShort": "Output",
        "chatPage.tokenUsage.oneModel": "{count} model",
        "chatPage.tokenUsage.models": "{count} models",
        "chatPage.tokenUsage.byModel": "Usage by model",
        "chatPage.tokenUsage.model": "Model",
        "chatPage.tokenUsage.unknownModel": "Unknown model",
      }[key] ?? key
      return vars?.count !== undefined
        ? message.replace("{count}", String(vars.count))
        : message
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
  })

  afterEach(() => {
    cleanup()
  })

  it("formats large token counts with compact lowercase suffixes", () => {
    expect(formatTokenCount(999)).toBe("999")
    expect(formatTokenCount(37_499)).toBe("37.5k")
    expect(formatTokenCount(2_755_525)).toBe("2.76m")
    expect(formatExactTokenCount(2_755_525)).toBe("2,755,525")
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
              total_tokens: 2_735_000,
            },
            {
              model_id: "compact",
              model_name: "deepseek/deepseek-v4-flash",
              input_tokens: 55_525,
              output_tokens: 2_499,
              total_tokens: 58_024,
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
    expect(modelUsageDialog).toHaveClass("w-[28rem]")
    expect(screen.getAllByText("Input")).toHaveLength(2)
    expect(screen.getAllByText("Output")).toHaveLength(2)
    expect(screen.getByText("deepseek/deepseek-v4-pro")).toBeInTheDocument()
    expect(screen.getByText("deepseek/deepseek-v4-flash")).toBeInTheDocument()
    expect(screen.getByText("main")).toBeInTheDocument()
    expect(screen.getByText("compact")).toBeInTheDocument()
    expect(screen.getByText("2.7m")).toHaveAttribute("title", "2,700,000")
    expect(screen.getByText("55.53k")).toHaveAttribute("title", "55,525")
  })

  it("uses the singular label and renders the unknown-model fallback", async () => {
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
              total_tokens: 15,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={8} isRunning={false} />)

    const modelsButton = await screen.findByRole("button", { name: /1 model/ })
    expect(modelsButton).toHaveTextContent("1 model")
    fireEvent.click(modelsButton)

    expect(await screen.findByText("Unknown model")).toBeInTheDocument()
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

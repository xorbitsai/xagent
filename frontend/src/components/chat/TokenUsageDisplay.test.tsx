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
      if (vars?.count !== undefined) return `${vars.count} models`
      return {
        "chatPage.tokenUsage.input": "Input tokens",
        "chatPage.tokenUsage.output": "Output tokens",
        "chatPage.tokenUsage.inputShort": "Input",
        "chatPage.tokenUsage.outputShort": "Output",
        "chatPage.tokenUsage.inputColumn": "Input",
        "chatPage.tokenUsage.outputColumn": "Output",
      }[key] ?? key
    },
  }),
}))

import { formatTokenCount, TokenUsageDisplay } from "./TokenUsageDisplay"

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
    expect(screen.getByText("2.7m")).toHaveAttribute("title", "2,700,000")
    expect(screen.getByText("55.53k")).toHaveAttribute("title", "55,525")
  })
})

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

const interpolate = (
  message: string,
  vars?: Record<string, string | number>,
) =>
  Object.entries(vars ?? {}).reduce(
    (result, [name, value]) => result.replaceAll(`{${name}}`, String(value)),
    message,
  )

const messages: Record<string, string> = {
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
  "chatPage.tokenUsage.mediaByModel": "Media usage",
  "chatPage.tokenUsage.mediaCall": "{count} media call",
  "chatPage.tokenUsage.mediaCalls": "{count} media calls",
  "chatPage.tokenUsage.quantity": "Amount",
  "chatPage.tokenUsage.callType": "Type",
  // Mirrors the shipped locale shape: one/other per unit. The component picks
  // the branch, so a regression that drops pluralisation shows up here.
  "chatPage.tokenUsage.unit.images.one": "image",
  "chatPage.tokenUsage.unit.images.other": "images",
  "chatPage.tokenUsage.unit.seconds.one": "sec",
  "chatPage.tokenUsage.unit.seconds.other": "sec",
  "chatPage.tokenUsage.unit.characters.one": "char",
  "chatPage.tokenUsage.unit.characters.other": "chars",
  "chatPage.tokenUsage.unit.requests.one": "request",
  "chatPage.tokenUsage.unit.requests.other": "requests",
  "chatPage.tokenUsage.unit.texts.one": "text",
  "chatPage.tokenUsage.unit.texts.other": "texts",
  // Present so a missing/renamed key fails instead of silently falling back to
  // the key string (the mock's tDynamic returns the fallback on a miss).
  "chatPage.tokenUsage.unmeasured": "not yet measured",
  "chatPage.tokenUsage.tokensShort": "tokens",
  "chatPage.tokenUsage.mediaType.generate_image": "Image generation",
  "chatPage.tokenUsage.mediaType.tts": "Text-to-speech",
  "chatPage.tokenUsage.mediaType.video": "Video",
  "chatPage.tokenUsage.mediaType.asr": "Speech-to-text",
  "chatPage.tokenUsage.mediaType.embedding": "Embedding",
  "chatPage.tokenUsage.mediaType.rerank": "Rerank",
}

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    locale: i18nMock.locale,
    t: (key: string, vars?: Record<string, string | number>) =>
      interpolate(messages[key] ?? key, vars),
    tDynamic: (
      key: string,
      fallback: string,
      vars?: Record<string, string | number>,
    ) => interpolate(messages[key] ?? fallback, vars),
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

describe("TokenUsageDisplay media usage", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    i18nMock.locale = "en"
  })

  afterEach(() => {
    cleanup()
  })

  it("exposes media usage in its own popover with unit-formatted amounts", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 100,
          output_tokens: 20,
          total_tokens: 120,
          llm_calls: 1,
          model_usage: [],
          media_usage: [
            {
              model_id: "sd",
              model_name: "stable-diffusion-xl",
              unit: "images",
              call_type: "generate_image",
              resolution: "1K",
              quantity: 3,
              calls: 2,
              tokens: 0,
            },
            {
              model_id: "tts-1",
              model_name: "elevenlabs-tts",
              // MediaCallType.TTS derives MediaUnit.CHARACTERS, so a
              // tts/seconds pair is unrepresentable upstream; asserting it
              // would pin a state the producer can never emit.
              unit: "characters",
              call_type: "tts",
              quantity: 480,
              calls: 1,
              tokens: 0,
            },
            {
              model_id: "whisper-1",
              model_name: "whisper-large",
              unit: "seconds",
              call_type: "asr",
              quantity: 12.5,
              calls: 1,
              tokens: 0,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={20} isRunning={false} />)

    // 4 media calls total (2 image + 1 tts + 1 asr).
    fireEvent.click(await screen.findByRole("button", { name: /4 media calls/ }))

    expect(await screen.findByText("Media usage")).toBeInTheDocument()
    expect(screen.getByText("stable-diffusion-xl")).toBeInTheDocument()
    expect(screen.getByText("Image generation")).toBeInTheDocument()
    expect(screen.getByText("3 images")).toBeInTheDocument()
    expect(screen.getByText("1K")).toBeInTheDocument()
    expect(screen.getByText("elevenlabs-tts")).toBeInTheDocument()
    expect(screen.getByText("Text-to-speech")).toBeInTheDocument()
    expect(screen.getByText("480 chars")).toBeInTheDocument()
    expect(screen.getByText("whisper-large")).toBeInTheDocument()
    expect(screen.getByText("Speech-to-text")).toBeInTheDocument()
    expect(screen.getByText("12.5 sec")).toBeInTheDocument()
  })

  it("renders the texts unit label used by embedding rows", async () => {
    // The embedding modality bills in `texts`, and that key was missing from
    // this mock while a stale `unit.tokens` lingered. With no assertion on it,
    // the desync was structurally uncatchable: the suite passed whether or not
    // the real locale files defined the key.
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0,
          llm_calls: 0,
          model_usage: [],
          media_usage: [
            {
              model_id: "emb-1",
              model_name: "text-embedding-v4",
              unit: "texts",
              call_type: "embedding",
              quantity: 32,
              calls: 1,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={31} isRunning={false} />)

    fireEvent.click(await screen.findByRole("button", { name: /1 media call/ }))

    expect(await screen.findByText("32 texts")).toBeInTheDocument()
  })

  it("labels a zero-quantity row as unmeasured rather than as a zero cost", async () => {
    // The duration-billed tools record quantity=0 to mean "this provider call
    // happened but its size is not known yet" (an async video with no duration).
    // Rendering that as "0 sec" reads as "this cost nothing", which is the
    // opposite of what it means.
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0,
          llm_calls: 0,
          model_usage: [],
          media_usage: [
            {
              model_id: "veo-1",
              model_name: "veo-3",
              unit: "seconds",
              call_type: "video",
              quantity: 0,
              calls: 2,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={33} isRunning={false} />)

    // calls > 0 with quantity 0: the row must still be counted and shown.
    fireEvent.click(await screen.findByRole("button", { name: /2 media calls/ }))

    expect(await screen.findByText("not yet measured")).toBeInTheDocument()
    expect(screen.getByText("veo-3")).toBeInTheDocument()
    expect(screen.queryByText("0 sec")).not.toBeInTheDocument()
  })

  it("renders a positive sub-rounding quantity as a bound, not as zero", async () => {
    // A 0.02s ASR clip is measured, so it must not collapse to "0 sec" and
    // claim the call was free. maximumFractionDigits: 1 alone rounds it to 0.
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0,
          llm_calls: 0,
          model_usage: [],
          media_usage: [
            {
              model_id: "whisper-1",
              model_name: "whisper-large",
              unit: "seconds",
              call_type: "asr",
              quantity: 0.02,
              calls: 1,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={34} isRunning={false} />)

    fireEvent.click(await screen.findByRole("button", { name: /1 media call/ }))

    expect(await screen.findByText("<0.1 sec")).toBeInTheDocument()
    expect(screen.queryByText("0 sec")).not.toBeInTheDocument()
    expect(screen.queryByText("not yet measured")).not.toBeInTheDocument()
  })

  it("uses singular unit labels when the quantity is exactly one", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0,
          llm_calls: 0,
          model_usage: [],
          media_usage: [
            {
              model_id: "sd",
              model_name: "stable-diffusion-xl",
              unit: "images",
              call_type: "generate_image",
              quantity: 1,
              calls: 1,
            },
            {
              model_id: "rr",
              model_name: "bge-reranker",
              unit: "requests",
              call_type: "rerank",
              quantity: 1,
              calls: 1,
            },
            {
              model_id: "emb-1",
              model_name: "text-embedding-v4",
              unit: "texts",
              call_type: "embedding",
              quantity: 1,
              calls: 1,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={35} isRunning={false} />)

    fireEvent.click(await screen.findByRole("button", { name: /3 media calls/ }))

    expect(await screen.findByText("1 image")).toBeInTheDocument()
    expect(screen.getByText("1 request")).toBeInTheDocument()
    expect(screen.getByText("1 text")).toBeInTheDocument()
    // The plural forms must not leak into the singular case.
    expect(screen.queryByText("1 images")).not.toBeInTheDocument()
    expect(screen.queryByText("1 requests")).not.toBeInTheDocument()
    expect(screen.queryByText("1 texts")).not.toBeInTheDocument()
  })

  it("distinguishes media rows that share a model name but differ by id", async () => {
    // The backend groups media by `model_id or model_name`, so these are two
    // separate billable rows. Showing only the name renders them identically.
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0,
          llm_calls: 0,
          model_usage: [],
          media_usage: [
            {
              model_id: "sd-primary",
              model_name: "stable-diffusion-xl",
              unit: "images",
              call_type: "generate_image",
              quantity: 4,
              calls: 2,
            },
            {
              model_id: "sd-fallback",
              model_name: "stable-diffusion-xl",
              unit: "images",
              call_type: "generate_image",
              quantity: 3,
              calls: 1,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={36} isRunning={false} />)

    fireEvent.click(await screen.findByRole("button", { name: /3 media calls/ }))

    expect(await screen.findByText("sd-primary")).toBeInTheDocument()
    expect(screen.getByText("sd-fallback")).toBeInTheDocument()
    expect(screen.getAllByText("stable-diffusion-xl")).toHaveLength(2)
    expect(screen.getByText("4 images")).toBeInTheDocument()
    expect(screen.getByText("3 images")).toBeInTheDocument()
  })

  it("omits the secondary id when it merely repeats the model name", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0,
          llm_calls: 0,
          model_usage: [],
          media_usage: [
            {
              model_id: "stable-diffusion-xl",
              model_name: "stable-diffusion-xl",
              unit: "images",
              call_type: "generate_image",
              quantity: 2,
              calls: 1,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={37} isRunning={false} />)

    fireEvent.click(await screen.findByRole("button", { name: /1 media call/ }))

    await screen.findByText("2 images")
    expect(screen.getAllByText("stable-diffusion-xl")).toHaveLength(1)
  })

  it("survives null and non-object media rows", async () => {
    // token_usage_details is free-form legacy JSON. Reducing over a null row
    // throws on property access, which would blank the whole component.
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0,
          llm_calls: 0,
          model_usage: [],
          media_usage: [
            null,
            "not an object",
            42,
            // Arrays are typeof 'object', so without the Array.isArray guard
            // this normalises into an extra all-zero "Unknown model" row.
            [],
            {
              model_id: "sd",
              model_name: "stable-diffusion-xl",
              unit: "images",
              call_type: "generate_image",
              quantity: 2,
              calls: 1,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={32} isRunning={false} />)

    // The junk rows are dropped; the real one still renders and is counted.
    fireEvent.click(await screen.findByRole("button", { name: /1 media call/ }))
    expect(await screen.findByText("2 images")).toBeInTheDocument()
    // Assert absence too: the count and quantity above both still pass if a
    // dropped row leaks back in as an extra unknown/unmeasured row, so those
    // assertions alone do not pin the guards.
    expect(screen.queryByText("Unknown model")).not.toBeInTheDocument()
    expect(screen.queryByText("not yet measured")).not.toBeInTheDocument()
  })

  it("renders a string quantity as its real value, not zero", async () => {
    // "4" passes a `> 0` check but fails Number.isFinite in the formatter, so
    // before normalisation this rendered the "unmeasured" placeholder and read
    // as if the call had cost nothing.
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0,
          llm_calls: 0,
          model_usage: [],
          media_usage: [
            {
              model_id: "sd",
              model_name: "stable-diffusion-xl",
              unit: "images",
              call_type: "generate_image",
              quantity: "4",
              calls: "2",
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={33} isRunning={false} />)

    // The string calls count is coerced too, so the label reads 2.
    fireEvent.click(await screen.findByRole("button", { name: /2 media calls/ }))
    expect(await screen.findByText("4 images")).toBeInTheDocument()
  })

  it("uses the singular label for a single media call", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0,
          llm_calls: 0,
          model_usage: [],
          media_usage: [
            {
              model_id: "sd",
              model_name: "sd",
              unit: "images",
              call_type: "generate_image",
              quantity: 1,
              calls: 1,
              tokens: 0,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={21} isRunning={false} />)

    expect(
      await screen.findByRole("button", { name: /^1 media call$/ }),
    ).toBeInTheDocument()
  })

  it("does not render a media popover without media usage", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 10,
          output_tokens: 2,
          total_tokens: 12,
          llm_calls: 1,
          model_usage: [],
          media_usage: [],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={22} isRunning={false} />)

    await screen.findByText("Input")
    expect(
      screen.queryByRole("button", { name: /media call/ }),
    ).not.toBeInTheDocument()
  })

  it("falls back to the raw call type and unit when no translation exists", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0,
          llm_calls: 0,
          model_usage: [],
          media_usage: [
            {
              model_id: "x",
              model_name: "custom-model",
              unit: "widgets",
              call_type: "custom_op",
              quantity: 4,
              calls: 1,
              tokens: 0,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={23} isRunning={false} />)

    fireEvent.click(await screen.findByRole("button", { name: /1 media call/ }))
    expect(await screen.findByText("custom_op")).toBeInTheDocument()
    expect(screen.getByText("4 widgets")).toBeInTheDocument()
  })

  it("renders an empty unit without a dangling trailing space", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0,
          llm_calls: 0,
          model_usage: [],
          media_usage: [
            {
              model_id: "x",
              model_name: "no-unit-model",
              unit: "",
              call_type: "video",
              quantity: 4,
              calls: 1,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={24} isRunning={false} />)

    fireEvent.click(await screen.findByRole("button", { name: /1 media call/ }))
    // Exact match: "4 " with a trailing space would not satisfy this.
    expect(await screen.findByText("4")).toBeInTheDocument()
  })

  it("renders provider tokens and marks estimated counts", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0,
          llm_calls: 0,
          model_usage: [],
          media_usage: [
            {
              model_id: "g",
              model_name: "gemini-image",
              unit: "images",
              call_type: "generate_image",
              quantity: 1,
              calls: 1,
              provider_tokens: 1120,
              tokens_estimated: false,
            },
            {
              model_id: "e",
              model_name: "text-embed",
              unit: "texts",
              call_type: "embedding",
              quantity: 3,
              calls: 1,
              provider_tokens: 40,
              tokens_estimated: true,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={25} isRunning={false} />)

    fireEvent.click(await screen.findByRole("button", { name: /2 media calls/ }))
    // Assert the whole string, label included: matching only the number passes
    // even if `tokensShort` is missing or resolved from the wrong key, which is
    // what makes the count meaningful rather than a bare number in the popover.
    // Normalise whitespace — the count and label are separate JSX children.
    const tokenText = (pattern: RegExp) =>
      screen.getByText((_content, element) => {
        if (!element) return false
        return pattern.test((element.textContent ?? "").replace(/\s+/g, " ").trim())
      })

    // Real Gemini image tokens are surfaced, not silently dropped.
    expect(await screen.findByText(/1\.12k/)).toBeInTheDocument()
    expect(tokenText(/^1\.12k tokens$/)).toBeInTheDocument()
    // The estimate is visibly marked so it is not mistaken for a measurement.
    expect(tokenText(/^40~ tokens$/)).toBeInTheDocument()
  })
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

import { describe, expect, it } from "vitest"

import {
  getProviderDisplayCapabilities,
  isContextWindowUnset,
} from "./model-display-capabilities"

describe("getProviderDisplayCapabilities", () => {
  it("keeps sound effect and music distinct on the audio tab", () => {
    expect(
      getProviderDisplayCapabilities(
        [
          { category: "speech", abilities: ["tts"] },
          { category: "speech", abilities: ["asr"] },
          { category: "sound_effect", abilities: ["generate"] },
          { category: "music", abilities: ["generate"] },
        ],
        "audio",
      ),
    ).toEqual(["tts", "asr", "sound_effect", "music"])
  })

  it("keeps generic generate for non-audio provider cards", () => {
    expect(
      getProviderDisplayCapabilities(
        [{ category: "image", abilities: ["generate"] }],
        "image",
      ),
    ).toEqual(["generate"])
  })

  it("keeps non-generate abilities on sound effect and music models", () => {
    expect(
      getProviderDisplayCapabilities(
        [
          { category: "sound_effect", abilities: ["generate", "edit"] },
          { category: "music", abilities: ["generate", "tts"] },
        ],
        "audio",
      ),
    ).toEqual(["sound_effect", "edit", "music", "tts"])
  })

  it("deduplicates repeated provider capabilities", () => {
    expect(
      getProviderDisplayCapabilities(
        [
          { category: "speech", abilities: ["tts"] },
          { category: "speech", abilities: ["tts"] },
        ],
        "audio",
      ),
    ).toEqual(["tts"])
  })
})

describe("isContextWindowUnset", () => {
  it("flags LLM rows whose context window is missing or non-positive", () => {
    expect(isContextWindowUnset({ category: "llm" })).toBe(true)
    expect(isContextWindowUnset({ category: "llm", context_window: 0 })).toBe(true)
    expect(isContextWindowUnset({ category: "llm", context_window: -1 })).toBe(true)
  })

  it("does not flag LLM rows with a window or non-LLM rows", () => {
    expect(isContextWindowUnset({ category: "llm", context_window: 256000 })).toBe(false)
    expect(isContextWindowUnset({ category: "embedding" })).toBe(false)
    expect(isContextWindowUnset({ category: "image", context_window: 0 })).toBe(false)
  })
})

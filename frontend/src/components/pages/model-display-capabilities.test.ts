import { describe, expect, it } from "vitest"

import { getProviderDisplayCapabilities } from "./model-display-capabilities"

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

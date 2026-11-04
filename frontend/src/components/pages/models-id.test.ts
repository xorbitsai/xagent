import { afterEach, describe, expect, it, vi } from "vitest"

import { generateModelId } from "./models"

describe("generateModelId", () => {
  afterEach(() => vi.restoreAllMocks())

  it("bounds generated IDs without truncating the model name used by the request", () => {
    vi.spyOn(crypto, "getRandomValues").mockImplementation((array) => {
      const bytes = array as Uint8Array
      bytes.set([0x01, 0x23, 0x45, 0x67])
      return array
    })
    const modelName = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/anthropic.claude-3-7-sonnet-20250219-v1:0"

    const modelId = generateModelId(modelName, "bedrock", "12345678")

    expect(modelId.length).toBeLessThanOrEqual(100)
    expect(modelId).toMatch(/^anthropic\.claude-3-7-sonnet-20250219-v1:0-bedrock-12345678-01234567$/)
    expect(modelName).toBe("arn:aws:bedrock:us-east-1:123456789012:inference-profile/anthropic.claude-3-7-sonnet-20250219-v1:0")
  })

  it("keeps the random suffix when every descriptive segment is oversized", () => {
    vi.spyOn(crypto, "getRandomValues").mockImplementation((array) => {
      const bytes = array as Uint8Array
      bytes.set([0x89, 0xab, 0xcd, 0xef])
      return array
    })

    const modelId = generateModelId("m".repeat(300), "p".repeat(100), "u".repeat(100))

    expect(modelId).toHaveLength(100)
    expect(modelId).toMatch(/-89abcdef$/)
  })
})

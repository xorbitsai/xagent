import { describe, expect, it, vi } from "vitest"

import {
  resolveDynamicTranslation,
  resolveTranslation,
  translations,
} from "./translations"

function isTranslationBranch(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value)
}

function assertTranslationTreeParity(
  left: unknown,
  right: unknown,
  path = "translations",
): void {
  expect(isTranslationBranch(left), `${path} must be an object`).toBe(true)
  expect(isTranslationBranch(right), `${path} must be an object`).toBe(true)
  if (!isTranslationBranch(left) || !isTranslationBranch(right)) return

  const leftKeys = Object.keys(left).sort()
  const rightKeys = Object.keys(right).sort()
  expect(rightKeys, `${path} must expose the same keys`).toEqual(leftKeys)

  for (const key of leftKeys) {
    const childPath = `${path}.${key}`
    const leftValue = left[key]
    const rightValue = right[key]
    const leftIsBranch = isTranslationBranch(leftValue)
    const rightIsBranch = isTranslationBranch(rightValue)

    expect(rightIsBranch, `${childPath} must have the same node type`).toBe(
      leftIsBranch,
    )
    if (leftIsBranch && rightIsBranch) {
      assertTranslationTreeParity(leftValue, rightValue, childPath)
      continue
    }

    expect(typeof leftValue, `${childPath} must be a string in en`).toBe("string")
    expect(typeof rightValue, `${childPath} must be a string in zh`).toBe("string")
  }
}

function assertTranslationLeavesNonEmpty(
  value: unknown,
  path: string,
): void {
  expect(isTranslationBranch(value), `${path} must be an object`).toBe(true)
  if (!isTranslationBranch(value)) return

  for (const [key, child] of Object.entries(value)) {
    const childPath = `${path}.${key}`
    if (isTranslationBranch(child)) {
      assertTranslationLeavesNonEmpty(child, childPath)
      continue
    }
    expect(typeof child, `${childPath} must be a string`).toBe("string")
    expect((child as string).trim(), `${childPath} must be non-empty`).not.toBe("")
  }
}

describe("translations", () => {
  it("keeps locale trees structurally identical", () => {
    assertTranslationTreeParity(translations.en, translations.zh)
  })

  it("keeps MCP runtime translations non-empty", () => {
    assertTranslationLeavesNonEmpty(
      translations.en.tools.mcp.runtime,
      "translations.en.tools.mcp.runtime",
    )
    assertTranslationLeavesNonEmpty(
      translations.zh.tools.mcp.runtime,
      "translations.zh.tools.mcp.runtime",
    )
  })

  it("resolves a typed translation key", () => {
    expect(resolveTranslation("en", "tools.mcp.runtime.title")).toBe(
      "Runtime Inputs",
    )
  })

  it("provides localized Agent delete dependency copy", () => {
    const english = (translations.en.builds.list as Record<string, unknown>).deleteDialog
    const chinese = (translations.zh.builds.list as Record<string, unknown>).deleteDialog

    expect(english).toEqual(expect.objectContaining({
      blockedTitle: expect.any(String),
      hiddenReferences: expect.any(String),
      readyToRetry: expect.any(String),
      retryDelete: expect.any(String),
      discardNotAllowed: expect.any(String),
      discardHasRuns: expect.any(String),
    }))
    expect(chinese).toEqual(expect.objectContaining({
      blockedTitle: expect.any(String),
      hiddenReferences: expect.any(String),
      readyToRetry: expect.any(String),
      retryDelete: expect.any(String),
      discardNotAllowed: expect.any(String),
      discardHasRuns: expect.any(String),
    }))
  })

  it("reports a missing dynamic key and uses its explicit fallback", () => {
    const onMissing = vi.fn()

    expect(
      resolveDynamicTranslation(
        "en",
        "tools.mcp.runtime.missing",
        "Unavailable",
        undefined,
        {
          onMissing,
        },
      ),
    ).toBe("Unavailable")
    expect(onMissing).toHaveBeenCalledWith("tools.mcp.runtime.missing")
  })
})

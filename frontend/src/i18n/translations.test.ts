import { describe, expect, it, vi } from "vitest"
import { readFileSync } from "node:fs"

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

  it("describes the admin account label and searchable identities", () => {
    expect(translations.en.userManagement.list).toEqual(
      expect.objectContaining({
        search_placeholder: "Search email or username...",
        table: expect.objectContaining({ account: "Account" }),
      }),
    )
    expect(translations.zh.userManagement.list).toEqual(
      expect.objectContaining({
        search_placeholder: "搜索邮箱或用户名...",
        table: expect.objectContaining({ account: "账户" }),
      }),
    )
  })

  it("describes deployment configuration failure without a browser fallback", () => {
    expect(translations.zh.deployment_config.messages.load_failed).toBe(
      "部署配置加载失败。复制部署信息前请重试。",
    )
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

  it("keeps publication failure copy neutral, exact, and fully migrated", () => {
    expect(translations.en.builds.publication).toEqual({
      publishFailed: "Failed to publish agent",
      unpublishFailed: "Failed to unpublish agent",
    })
    expect(translations.zh.builds.publication).toEqual({
      publishFailed: "发布 Agent 失败",
      unpublishFailed: "取消发布 Agent 失败",
    })
    for (const editorError of [
      translations.en.builds.editor.error,
      translations.zh.builds.editor.error,
    ]) {
      expect(editorError as Record<string, unknown>).not.toHaveProperty("publishFailed")
      expect(editorError as Record<string, unknown>).not.toHaveProperty("unpublishFailed")
    }

    const buildPage = readFileSync(`${process.cwd()}/src/app/build/page.tsx`, "utf8")
    const agentBuilder = readFileSync(
      `${process.cwd()}/src/components/build/agent-builder.tsx`,
      "utf8",
    )
    const localeSources = [
      readFileSync(`${process.cwd()}/src/i18n/locales/en.ts`, "utf8"),
      readFileSync(`${process.cwd()}/src/i18n/locales/zh.ts`, "utf8"),
    ]

    expect(buildPage.match(/builds\.publication\.publishFailed/g)).toHaveLength(1)
    expect(buildPage.match(/builds\.publication\.unpublishFailed/g)).toHaveLength(1)
    expect(agentBuilder.match(/builds\.publication\.publishFailed/g)).toHaveLength(2)
    expect(agentBuilder.match(/builds\.publication\.unpublishFailed/g)).toHaveLength(1)

    for (const source of [buildPage, agentBuilder, ...localeSources]) {
      expect(source).not.toContain("builds.editor.error.publishFailed")
      expect(source).not.toContain("builds.editor.error.unpublishFailed")
    }
  })

  it("provides localized Widget Session lifecycle copy", () => {
    assertTranslationLeavesNonEmpty(
      translations.en.widgetSession,
      "translations.en.widgetSession",
    )
    assertTranslationLeavesNonEmpty(
      translations.zh.widgetSession,
      "translations.zh.widgetSession",
    )
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

  // Pins a PR review finding: the onboarding Done step composes these
  // prefix/<b>value</b>/suffix strings in JSX (page.tsx) with no space of
  // its own between them - each locale must carry its OWN spacing, since
  // English needs a space between words where Chinese's equivalents (which
  // sit directly against the bold value, or against nothing at all for an
  // empty suffix) don't. A hardcoded JSX-level space used to leak into
  // Chinese copy as visible gaps like "所在领域： 市场营销".
  it("gives the onboarding Done step's composed summary lines English-only inter-word spacing", () => {
    const enPrefixes = [
      resolveTranslation("en", "onboarding.done.workingInPrefix"),
      resolveTranslation("en", "onboarding.done.briefedPrefix"),
      resolveTranslation("en", "onboarding.done.writingInPrefix"),
      resolveTranslation("en", "onboarding.done.willConnectPrefix"),
    ]
    for (const prefix of enPrefixes) {
      expect(prefix.endsWith(" ")).toBe(true)
    }
    const enSuffixes = [
      resolveTranslation("en", "onboarding.done.briefedSuffixOne"),
      resolveTranslation("en", "onboarding.done.briefedSuffixOther"),
      resolveTranslation("en", "onboarding.done.writingInSuffix"),
      resolveTranslation("en", "onboarding.done.willConnectSuffix"),
    ]
    for (const suffix of enSuffixes) {
      expect(suffix.startsWith(" ")).toBe(true)
    }

    const zhPrefixes = [
      resolveTranslation("zh", "onboarding.done.workingInPrefix"),
      resolveTranslation("zh", "onboarding.done.briefedPrefix"),
      resolveTranslation("zh", "onboarding.done.writingInPrefix"),
      resolveTranslation("zh", "onboarding.done.willConnectPrefix"),
    ]
    for (const prefix of zhPrefixes) {
      expect(prefix.endsWith(" ")).toBe(false)
    }
    const zhSuffixes = [
      resolveTranslation("zh", "onboarding.done.briefedSuffixOne"),
      resolveTranslation("zh", "onboarding.done.briefedSuffixOther"),
    ]
    for (const suffix of zhSuffixes) {
      expect(suffix.startsWith(" ")).toBe(false)
    }
  })

  // Pins the same finding for the team step's "N other matches" subtitle:
  // the em-dash-led extra-matches phrase is appended directly after the
  // base sentence in JSX with no space of its own - English needs one
  // before the em dash, Chinese's own em dash must sit directly against
  // the preceding sentence with no space.
  it("gives the team step's 'other matches' subtitle English-only leading spacing before the em dash", () => {
    expect(resolveTranslation("en", "onboarding.team.subtitleExtraOne").startsWith(" ")).toBe(true)
    expect(resolveTranslation("en", "onboarding.team.subtitleExtraMany").startsWith(" ")).toBe(true)
    expect(resolveTranslation("zh", "onboarding.team.subtitleExtraOne").startsWith(" ")).toBe(false)
    expect(resolveTranslation("zh", "onboarding.team.subtitleExtraMany").startsWith(" ")).toBe(false)
  })
})

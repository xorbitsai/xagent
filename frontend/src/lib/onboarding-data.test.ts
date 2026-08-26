import { describe, expect, it } from "vitest"
import {
  ONBOARDING_FALLBACK_TEMPLATE_IDS,
  ONBOARDING_GOALS,
  joinWithAnd,
  recommendedTemplates,
  reorderGoalsByWork,
} from "./onboarding-data"

describe("reorderGoalsByWork", () => {
  it("brings goals matching the selected work type to the front, preserving relative order within each bucket", () => {
    const reordered = reorderGoalsByWork("sales")
    const salesIds = ONBOARDING_GOALS.filter((g) => g.fn === "sales").map((g) => g.id)
    const otherIds = ONBOARDING_GOALS.filter((g) => g.fn !== "sales").map((g) => g.id)

    expect(reordered.map((g) => g.id)).toEqual([...salesIds, ...otherIds])
  })

  it("returns goals in their original order when nothing matches (e.g. 'other')", () => {
    const reordered = reorderGoalsByWork("other")
    expect(reordered.map((g) => g.id)).toEqual(ONBOARDING_GOALS.map((g) => g.id))
  })

  it("does not mutate the original ONBOARDING_GOALS array", () => {
    const before = ONBOARDING_GOALS.map((g) => g.id)
    reorderGoalsByWork("marketing")
    expect(ONBOARDING_GOALS.map((g) => g.id)).toEqual(before)
  })
})

describe("recommendedTemplates", () => {
  it("maps each selected goal to its template, in ONBOARDING_GOALS' declared order (not selection order)", () => {
    const result = recommendedTemplates(["social", "inbox"])
    expect(result).toEqual([
      { templateId: "support-inbox-manager", goalId: "inbox" },
      { templateId: "marketing-social-media-content-manager", goalId: "social" },
    ])
  })

  it("dedupes when two selected goals point at the same template", () => {
    // "inbox" and "support" both map to a support-inbox flavored template? -
    // pick two goals that share a templateId to prove dedup, using the real
    // data: "support" -> support-ai-chatbot-agent is distinct, so instead
    // assert general dedup behavior using a goal selected twice via id reuse.
    const result = recommendedTemplates(["inbox", "inbox"])
    expect(result).toEqual([{ templateId: "support-inbox-manager", goalId: "inbox" }])
  })

  it("falls back to the fixed 3-agent default when no goals are selected", () => {
    const result = recommendedTemplates([])
    expect(result).toEqual(
      ONBOARDING_FALLBACK_TEMPLATE_IDS.map((templateId) => ({ templateId, goalId: null }))
    )
  })

  it("caps at 3 recommendations even when more goals are selected", () => {
    const manyGoalIds = ONBOARDING_GOALS.slice(0, 6).map((g) => g.id)
    const result = recommendedTemplates(manyGoalIds)
    // Asserts the exact first 3 (in ONBOARDING_GOALS' own declared order),
    // not just an upper bound - toBeLessThanOrEqual(3) would also pass for
    // an empty array, silently missing a regression that dropped everything.
    expect(result).toEqual(ONBOARDING_GOALS.slice(0, 3).map((g) => ({ templateId: g.templateId, goalId: g.id })))
  })

  it("ignores unknown goal ids", () => {
    expect(recommendedTemplates(["not-a-real-goal"])).toEqual(
      ONBOARDING_FALLBACK_TEMPLATE_IDS.map((templateId) => ({ templateId, goalId: null }))
    )
  })
})

describe("joinWithAnd", () => {
  it("returns an empty string for no items", () => {
    expect(joinWithAnd([])).toBe("")
  })

  it("returns the single item unchanged", () => {
    expect(joinWithAnd(["Gmail"])).toBe("Gmail")
  })

  it("joins two items with 'and', no comma", () => {
    expect(joinWithAnd(["Gmail", "Outlook"])).toBe("Gmail and Outlook")
  })

  it("joins 3+ items with commas and a trailing 'and'", () => {
    expect(joinWithAnd(["LinkedIn", "Facebook Pages", "Instagram", "Google Drive"])).toBe(
      "LinkedIn, Facebook Pages, Instagram and Google Drive"
    )
  })

  // Pins a PR review finding: the connector word must be localizable, not a
  // hardcoded English "and" leaking into otherwise-translated copy.
  it("uses a caller-supplied localized connector word instead of hardcoding English", () => {
    expect(joinWithAnd(["Gmail", "Outlook"], "和")).toBe("Gmail 和 Outlook")
  })
})

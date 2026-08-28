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

  // Pins a PR review finding: this only proves duplicate SELECTED ids don't
  // produce duplicate output - `ONBOARDING_GOALS.filter(...includes...)`
  // already collapses a repeated id to one matched goal before the
  // templateId-based `seen` dedup loop ever runs, so deleting that loop
  // entirely would leave this test passing. No two goals in today's real
  // catalog share a templateId, so proving the `seen` loop itself would
  // need a synthetic catalog; not worth mocking the whole module (every
  // other test here relies on the real one) for this one case.
  it("collapses a goal id selected more than once to a single result", () => {
    const result = recommendedTemplates(["inbox", "inbox"])
    expect(result).toEqual([{ templateId: "support-inbox-manager", goalId: "inbox" }])
  })

  it("falls back to the fixed 3-agent default when no goals are selected", () => {
    const result = recommendedTemplates([])
    expect(result).toEqual(
      ONBOARDING_FALLBACK_TEMPLATE_IDS.map((templateId) => ({ templateId, goalId: null }))
    )
  })

  // Pins a PR review finding: capping at 3 here (before the caller filters
  // by persona-availability) meant a 4th-ranked match with a real persona
  // could never fill a slot vacated by a top-3 match that had none. The cap
  // now belongs to the caller (page.tsx's validRecommended), applied AFTER
  // that filter - this function returns every match, uncapped.
  it("does not cap the result - returns every distinct match, leaving capping to the caller", () => {
    const manyGoalIds = ONBOARDING_GOALS.slice(0, 6).map((g) => g.id)
    const result = recommendedTemplates(manyGoalIds)
    expect(result).toEqual(ONBOARDING_GOALS.slice(0, 6).map((g) => ({ templateId: g.templateId, goalId: g.id })))
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

  // Pins a PR review finding: the ", " separator was hardcoded Western
  // punctuation, same class of bug as the "and" word before it was
  // localized - Chinese conventionally uses "、" between list items.
  it("uses a caller-supplied separator instead of a hardcoded comma", () => {
    expect(joinWithAnd(["LinkedIn", "Facebook Pages", "Instagram"], "和", "、")).toBe(
      "LinkedIn、Facebook Pages 和 Instagram"
    )
  })

  // Pins a PR review finding: the connector word must be localizable, not a
  // hardcoded English "and" leaking into otherwise-translated copy.
  it("uses a caller-supplied localized connector word instead of hardcoding English", () => {
    expect(joinWithAnd(["Gmail", "Outlook"], "和")).toBe("Gmail 和 Outlook")
  })
})

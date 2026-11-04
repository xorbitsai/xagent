import { describe, expect, it } from "vitest"

import {
  normalizeCategoryKey,
  orderCategoriesWithPreferred,
} from "./template-categories"

describe("orderCategoriesWithPreferred", () => {
  it("prepends preferred categories in the caller-supplied order, then the rest in first-seen order", () => {
    const dynamic = ["General", "Support", "Sales", "Marketing"]
    expect(orderCategoriesWithPreferred(dynamic, ["Marketing", "Sales", "Support"])).toEqual([
      "Marketing",
      "Sales",
      "Support",
      "General",
    ])
    // Same input, a different caller-supplied preferred order (e.g. the
    // /templates library page's own order) - shared algorithm, independent
    // results, exactly the point of extracting this instead of each page
    // reimplementing its own copy.
    expect(orderCategoriesWithPreferred(dynamic, ["Sales", "Marketing", "Support"])).toEqual([
      "Sales",
      "Marketing",
      "Support",
      "General",
    ])
  })

  it("drops preferred entries that aren't actually present", () => {
    expect(orderCategoriesWithPreferred(["Sales"], ["Marketing", "Sales"])).toEqual(["Sales"])
  })
})

describe("normalizeCategoryKey", () => {
  it("lowercases, collapses \"&\"-joined words, and underscores remaining whitespace", () => {
    expect(normalizeCategoryKey("Healthcare & Fitness")).toBe("healthcare_fitness")
    expect(normalizeCategoryKey("General Productivity")).toBe("general_productivity")
    expect(normalizeCategoryKey("Sales")).toBe("sales")
  })
})

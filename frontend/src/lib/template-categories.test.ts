import { describe, expect, it } from "vitest"

import type { Template } from "@/types/template"
import {
  FEATURED_CATEGORY_ID,
  getOrderedCategoriesWithCounts,
  getTemplatesForCategory,
  normalizeCategoryKey,
  orderCategoriesWithPreferred,
} from "./template-categories"

function makeTemplate(overrides: Partial<Template> & Pick<Template, "id" | "category">): Template {
  return {
    name: overrides.id,
    featured: false,
    description: "",
    features: [],
    connections: [],
    setup_time: "5 min setup",
    tags: [],
    author: "Xagent",
    version: "1.0",
    views: 0,
    likes: 0,
    used_count: 0,
    ...overrides,
  }
}

describe("getOrderedCategoriesWithCounts", () => {
  it("omits the Featured tab entirely when no template is featured", () => {
    const templates = [makeTemplate({ id: "a", category: "Sales" })]
    const tabs = getOrderedCategoriesWithCounts(templates)

    expect(tabs.find((tab) => tab.id === FEATURED_CATEGORY_ID)).toBeUndefined()
  })

  it("prepends Featured with the count of featured templates when any exist", () => {
    const templates = [
      makeTemplate({ id: "a", category: "Sales", featured: true }),
      makeTemplate({ id: "b", category: "Sales" }),
      makeTemplate({ id: "c", category: "Marketing", featured: true }),
    ]
    const tabs = getOrderedCategoriesWithCounts(templates)

    expect(tabs[0]).toEqual({ id: FEATURED_CATEGORY_ID, count: 2 })
  })

  it("orders known categories Marketing, Sales, Support before any others", () => {
    const templates = [
      makeTemplate({ id: "a", category: "Support" }),
      makeTemplate({ id: "b", category: "Sales" }),
      makeTemplate({ id: "c", category: "Marketing" }),
      makeTemplate({ id: "d", category: "General" }),
    ]
    const tabs = getOrderedCategoriesWithCounts(templates)

    expect(tabs.map((tab) => tab.id)).toEqual(["Marketing", "Sales", "Support", "General"])
  })

  it("counts each category independent of the featured count", () => {
    const templates = [
      makeTemplate({ id: "a", category: "Marketing" }),
      makeTemplate({ id: "b", category: "Marketing" }),
      makeTemplate({ id: "c", category: "Sales" }),
    ]
    const tabs = getOrderedCategoriesWithCounts(templates)

    expect(tabs.find((tab) => tab.id === "Marketing")?.count).toBe(2)
    expect(tabs.find((tab) => tab.id === "Sales")?.count).toBe(1)
  })
})

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

describe("getTemplatesForCategory", () => {
  it("filters by the featured flag for the Featured tab", () => {
    const featured = makeTemplate({ id: "a", category: "Sales", featured: true })
    const notFeatured = makeTemplate({ id: "b", category: "Sales" })
    const result = getTemplatesForCategory([featured, notFeatured], FEATURED_CATEGORY_ID)

    expect(result).toEqual([featured])
  })

  it("filters by category for a non-Featured tab", () => {
    const marketing = makeTemplate({ id: "a", category: "Marketing" })
    const sales = makeTemplate({ id: "b", category: "Sales" })
    const result = getTemplatesForCategory([marketing, sales], "Sales")

    expect(result).toEqual([sales])
  })

  it("caps results to the given limit, defaulting to 4", () => {
    const templates = Array.from({ length: 6 }, (_, index) =>
      makeTemplate({ id: `t${index}`, category: "Marketing" })
    )

    expect(getTemplatesForCategory(templates, "Marketing")).toHaveLength(4)
    expect(getTemplatesForCategory(templates, "Marketing", 2)).toHaveLength(2)
  })
})

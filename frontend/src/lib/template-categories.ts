import type { Template } from "@/types/template";

export const FEATURED_CATEGORY_ID = "Featured";

const PREFERRED_ORDER = ["Marketing", "Sales", "Support", "Research", "Productivity"];

export interface CategoryTab {
  id: string;
  count: number;
}

/**
 * Shared by every page that renders a category filter (this quick-access
 * pill row and the /templates library page): a known-first subset in
 * caller-supplied preferred order, then any remaining categories in
 * first-seen order. Each caller keeps its own preferred order (the two
 * pages intentionally differ here - Featured/Marketing/Sales/Support for
 * this compact panel vs Sales/Marketing/Support for the full library) while
 * sharing the actual ordering algorithm instead of each reimplementing it.
 */
export function orderCategoriesWithPreferred(
  dynamicCategories: string[],
  preferredOrder: string[]
): string[] {
  return [
    ...preferredOrder.filter((category) => dynamicCategories.includes(category)),
    ...dynamicCategories.filter((category) => !preferredOrder.includes(category)),
  ];
}

/**
 * Normalizes a category name into a lookup/section-id key: lowercase,
 * "&"-joined words collapsed to a single underscore, remaining whitespace
 * underscored. Shared so a category like "Healthcare & Fitness" resolves to
 * the same key ("healthcare_fitness") everywhere it's looked up.
 */
export function normalizeCategoryKey(category: string): string {
  return category.toLowerCase().replace(/\s*&\s*/g, "_").replace(/\s+/g, "_");
}

/**
 * Ordered category tabs with counts: Featured first (when any template is
 * featured), then known categories in PREFERRED_ORDER, then any remaining
 * categories in first-seen order.
 */
export function getOrderedCategoriesWithCounts(templates: Template[]): CategoryTab[] {
  const featuredCount = templates.filter((template) => template.featured).length;
  const dynamicCategories = Array.from(
    new Set(templates.map((template) => template.category).filter(Boolean))
  );
  const orderedCategories = orderCategoriesWithPreferred(dynamicCategories, PREFERRED_ORDER);

  const tabs: CategoryTab[] = [];
  if (featuredCount > 0) {
    tabs.push({ id: FEATURED_CATEGORY_ID, count: featuredCount });
  }
  orderedCategories.forEach((category) => {
    tabs.push({
      id: category,
      count: templates.filter((template) => template.category === category).length,
    });
  });

  return tabs;
}

/** Top `limit` templates for a given category tab, in natural API order. */
export function getTemplatesForCategory(
  templates: Template[],
  categoryId: string,
  limit = 4
): Template[] {
  const filtered =
    categoryId === FEATURED_CATEGORY_ID
      ? templates.filter((template) => template.featured)
      : templates.filter((template) => template.category === categoryId);
  return filtered.slice(0, limit);
}

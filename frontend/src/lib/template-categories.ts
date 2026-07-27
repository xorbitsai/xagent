import type { Template } from "@/types/template";

export const FEATURED_CATEGORY_ID = "Featured";

const PREFERRED_ORDER = ["Marketing", "Sales", "Support", "Research", "Productivity"];

export interface CategoryTab {
  id: string;
  count: number;
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
  const orderedCategories = [
    ...PREFERRED_ORDER.filter((category) => dynamicCategories.includes(category)),
    ...dynamicCategories.filter((category) => !PREFERRED_ORDER.includes(category)),
  ];

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

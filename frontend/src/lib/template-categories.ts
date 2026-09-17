import type { Translate } from "@/contexts/i18n-context";
import type { TranslationKey } from "@/i18n/translations";

const CATEGORY_LABEL_KEYS: Record<string, TranslationKey> = {
  general: "templates.categoryTitles.general",
  sales: "templates.categoryTitles.sales",
  marketing: "templates.categoryTitles.marketing",
  support: "templates.categoryTitles.support",
  research: "templates.sections.knowledge",
  productivity: "templates.categoryTitles.general_productivity",
  healthcare_fitness: "templates.categoryTitles.healthcare_fitness",
  general_productivity: "templates.categoryTitles.general_productivity",
  customer_service: "templates.categoryTitles.customer_service",
  finance_lms_ops: "templates.categoryTitles.finance_lms_ops",
  security: "templates.categoryTitles.security",
  operations: "templates.categoryTitles.operations",
};

const formatFallbackLabel = (category: string) =>
  category
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());

/**
 * A template category's display label: the known i18n key for it when one
 * exists, else its raw slug title-cased - shared so a category badge reads
 * the same everywhere it's shown (the /templates library page and any
 * agent card that traces back to a template) instead of some places
 * showing a translated label and others an untranslated raw slug.
 */
export function categoryLabel(t: Translate, category: string | undefined): string {
  const c = category || "Others";
  const key = CATEGORY_LABEL_KEYS[normalizeCategoryKey(c)];
  return key ? t(key) : formatFallbackLabel(c);
}

/**
 * Shared by every page that renders a category filter: a known-first
 * subset in caller-supplied preferred order, then any remaining categories
 * in first-seen order - each caller keeps its own preferred order while
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

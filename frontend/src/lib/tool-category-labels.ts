/** Maps a raw `tool_categories` key (as stored on Agent/AgentConfig) to the
 * i18n key for its friendly display label. Shared between the AI Team
 * Marketplace detail page's "Tools" panel and its card's capability tags,
 * so the two surfaces can't drift out of sync with each other. */
export const TOOL_CATEGORY_I18N_KEYS: Record<string, string> = {
  basic: "builds.configForm.tools.categories.basic",
  web_search: "builds.configForm.tools.categories.webSearch",
  file: "builds.configForm.tools.categories.file",
  vision: "builds.configForm.tools.categories.vision",
  image: "builds.configForm.tools.categories.image",
  video: "builds.configForm.tools.categories.video",
  audio: "builds.configForm.tools.categories.audio",
  knowledge: "builds.configForm.tools.categories.knowledge",
  browser: "builds.configForm.tools.categories.browser",
  ppt: "builds.configForm.tools.categories.ppt",
  office: "builds.configForm.tools.categories.office",
  database: "builds.configForm.tools.categories.database",
  skill: "builds.configForm.tools.categories.skill",
  ssh: "builds.configForm.tools.categories.ssh",
};

export function capitalize(value: string): string {
  return value.length > 0 ? value[0].toUpperCase() + value.slice(1) : value;
}

/**
 * A curated, on-brand subset of `tool_categories` + `skills` for a
 * marketplace card's highlight-tags row: drops the categories that are
 * either redundant with the connector icons (`mcp:*`), assumed present on
 * every agent (`basic`), or not interesting to show as a capability
 * headline (`browser`, `knowledge`) - then appends skill names verbatim
 * (they're author-chosen slugs, not translated categories).
 */
const CARD_TAG_EXCLUDED_CATEGORIES = new Set(["basic", "browser", "knowledge"]);

export function getCardCapabilityTags(
  toolCategories: string[] | undefined,
  skills: string[] | undefined,
  formatCategory: (category: string) => string
): string[] {
  const categoryTags = (toolCategories || [])
    .filter(
      (category) =>
        !category.startsWith("mcp:") && !CARD_TAG_EXCLUDED_CATEGORIES.has(category)
    )
    .map(formatCategory);
  return [...categoryTags, ...(skills || [])];
}

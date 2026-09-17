import { describe, expect, it } from "vitest";
import { capitalize, getCardCapabilityTags, TOOL_CATEGORY_I18N_KEYS } from "./tool-category-labels";

describe("capitalize", () => {
  it("uppercases only the first character", () => {
    expect(capitalize("ssh")).toBe("Ssh");
    expect(capitalize("")).toBe("");
  });
});

describe("TOOL_CATEGORY_I18N_KEYS", () => {
  it("has an entry for the plain 'mcp' category (all-MCP-tools selection), distinct from the mcp:ServerName-prefixed entries filtered elsewhere", () => {
    // Without this, both consumers of this map (the card's capability tags
    // and the detail page's Tools panel) fall through to the capitalize()
    // fallback and render the raw string as "Mcp".
    expect(TOOL_CATEGORY_I18N_KEYS.mcp).toBe("builds.configForm.tools.categories.mcp");
  });
});

describe("getCardCapabilityTags", () => {
  const format = (category: string) => `[${category}]`;

  it("drops basic/browser/knowledge and mcp: entries, keeps the rest formatted, appends skills verbatim", () => {
    // Reproduces Maya's real tool_categories/skills from the marketing
    // template - this exact case is what the reference design shows.
    const tags = getCardCapabilityTags(
      ["basic", "web_search", "browser", "file", "image", "knowledge", "mcp:LinkedIn", "mcp:Facebook"],
      ["static-visual-design"],
      format
    );

    expect(tags).toEqual(["[web_search]", "[file]", "[image]", "static-visual-design"]);
  });

  it("returns an empty list for a workforce template with no capabilities", () => {
    expect(getCardCapabilityTags([], [], format)).toEqual([]);
    expect(getCardCapabilityTags(undefined, undefined, format)).toEqual([]);
  });
});

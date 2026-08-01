import React from "react";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("next/link", () => ({
  default: ({
    children,
    href,
    ...props
  }: React.AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }) => (
    <a href={href} {...props}>{children}</a>
  ),
}));

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string | number>) =>
      vars ? `${key}:${JSON.stringify(vars)}` : key,
  }),
}));

import { TemplateQuickAccess } from "./TemplateQuickAccess";
import { FEATURED_CATEGORY_ID } from "@/lib/template-categories";
import type { Template } from "@/types/template";

function makeTemplate(overrides: Partial<Template> = {}): Template {
  return {
    id: "tmpl-1",
    name: "Test Template",
    category: "Marketing",
    featured: true,
    description: "A test template",
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
  };
}

afterEach(cleanup);

describe("TemplateQuickAccess", () => {
  it("renders at most 2 sample prompts per card, even when the template has more", () => {
    const template = makeTemplate({
      sample_prompts: [
        { title: "Prompt one", prompt: "Do the first thing." },
        { title: "Prompt two", prompt: "Do the second thing." },
        { title: "Prompt three", prompt: "Do the third thing." },
      ],
    });

    render(
      <TemplateQuickAccess
        templates={[template]}
        selectedCategory={FEATURED_CATEGORY_ID}
        onCategoryChange={vi.fn()}
        selectedPromptKey={null}
        onPromptSelect={vi.fn()}
      />
    );

    expect(screen.getByRole("button", { name: /Prompt one/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Prompt two/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Prompt three/ })).not.toBeInTheDocument();
  });

  it("renders nothing when there are no templates", () => {
    const { container } = render(
      <TemplateQuickAccess
        templates={[]}
        selectedCategory={FEATURED_CATEGORY_ID}
        onCategoryChange={vi.fn()}
        selectedPromptKey={null}
        onPromptSelect={vi.fn()}
      />
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("survives the loading -> loaded transition without a hooks-order crash", () => {
    // Regression test for PR review finding C1: the real /task mount order is
    // (isLoading=false, templates=[]) on the very first render - falling
    // through to the full render path - then a mount effect synchronously
    // flips isLoading to true before the fetch resolves, forcing a second
    // render down the early-return path. Both useMemo calls must run on
    // every render regardless of which path is taken, or React throws
    // "Rendered fewer hooks than expected."
    const { rerender } = render(
      <TemplateQuickAccess
        templates={[]}
        selectedCategory={FEATURED_CATEGORY_ID}
        onCategoryChange={vi.fn()}
        selectedPromptKey={null}
        onPromptSelect={vi.fn()}
        isLoading={false}
      />
    );

    expect(() =>
      rerender(
        <TemplateQuickAccess
          templates={[]}
          selectedCategory={FEATURED_CATEGORY_ID}
          onCategoryChange={vi.fn()}
          selectedPromptKey={null}
          onPromptSelect={vi.fn()}
          isLoading={true}
        />
      )
    ).not.toThrow();

    expect(
      screen.getByText("chatPage.templateQuickAccess.loading")
    ).toBeInTheDocument();
  });

  it.each(["General", "Operations", "Marketing", "Sales", "Support"])(
    "resolves a real label key for the shipped %s category, not the raw-id fallback",
    (category) => {
      // Regression test for PR review finding m7: CATEGORY_LABEL_KEYS used to
      // be missing entries for two of the five shipped categories (General,
      // Operations), so selecting them fell through to categoryLabel's raw
      // categoryId fallback instead of a translated label. The mocked t()
      // above echoes the key it's given, so a resolved key renders as
      // "templates.categoryTitles.<x>" while a missed one renders as the
      // untranslated category string itself.
      render(
        <TemplateQuickAccess
          templates={[makeTemplate({ category, featured: false })]}
          selectedCategory={category}
          onCategoryChange={vi.fn()}
          selectedPromptKey={null}
          onPromptSelect={vi.fn()}
        />
      );

      expect(screen.queryByText(category)).not.toBeInTheDocument();
    }
  );

  it('links "All templates" back to the active category, not always the unscoped default', () => {
    // Regression test for PR review finding FE-11/m11: this escape hatch
    // used to be a bare /templates link with no category, so following it
    // from e.g. the Support tab always landed back on "All" instead of
    // Support.
    render(
      <TemplateQuickAccess
        templates={[makeTemplate({ category: "Support", featured: false })]}
        selectedCategory="Support"
        onCategoryChange={vi.fn()}
        selectedPromptKey={null}
        onPromptSelect={vi.fn()}
      />
    );

    expect(
      screen.getByText("chatPage.templateQuickAccess.allTemplates")
    ).toHaveAttribute("href", "/templates?category=Support");
  });

  it('links "All templates" to the unscoped /templates from the Featured tab', () => {
    // /templates has no selectable "Featured" tab of its own - selecting
    // "All" there is what surfaces its Featured section - so the Featured
    // tab must not carry a category=Featured param /templates wouldn't
    // recognize.
    render(
      <TemplateQuickAccess
        templates={[makeTemplate({ category: "Support", featured: true })]}
        selectedCategory={FEATURED_CATEGORY_ID}
        onCategoryChange={vi.fn()}
        selectedPromptKey={null}
        onPromptSelect={vi.fn()}
      />
    );

    expect(
      screen.getByText("chatPage.templateQuickAccess.allTemplates")
    ).toHaveAttribute("href", "/templates");
  });
});

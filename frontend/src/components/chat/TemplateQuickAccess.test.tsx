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
});

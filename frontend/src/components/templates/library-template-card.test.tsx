import React from "react";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { LibraryTemplateCard } from "./library-template-card";
import type { Template } from "@/types/template";

function makeTemplate(overrides: Partial<Template> = {}): Template {
  return {
    id: "tmpl-1",
    name: "Test Template",
    category: "Marketing",
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

describe("LibraryTemplateCard", () => {
  it("does not show the workforce badge for a plain agent template", () => {
    render(
      <LibraryTemplateCard
        template={makeTemplate({ type: "agent" })}
        useLabel="Use Template"
        defaultSetupTime="5 min setup"
        workforceBadgeLabel="Workforce"
        onUse={vi.fn()}
      />
    );

    expect(screen.queryByText("Workforce")).not.toBeInTheDocument();
  });

  it("shows the workforce badge with the total agent count (manager + workers)", () => {
    // The badge label reads "N agents" - the count must include the
    // manager, or it undercounts against what "Use" actually creates
    // (PR #1127 re-review, F6a).
    render(
      <LibraryTemplateCard
        template={makeTemplate({
          type: "workforce",
          agent_count: 3,
        })}
        useLabel="Use Template"
        defaultSetupTime="5 min setup"
        workforceBadgeLabel="Workforce"
        formatAgentsCount={(count) => `${count} agents`}
        onUse={vi.fn()}
      />
    );

    expect(screen.getByText(/Workforce/)).toBeInTheDocument();
    expect(screen.getByText(/3 agents/)).toBeInTheDocument();
  });

  it("calls onUse when the card is clicked", () => {
    const onUse = vi.fn();
    render(
      <LibraryTemplateCard
        template={makeTemplate({ id: "tmpl-42" })}
        useLabel="Use Template"
        defaultSetupTime="5 min setup"
        onUse={onUse}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Use Template" }));

    expect(onUse).toHaveBeenCalledWith("tmpl-42");
  });

  it("disables activation and shows the busy label while isBusy is true", () => {
    const onUse = vi.fn();
    const { container } = render(
      <LibraryTemplateCard
        template={makeTemplate({ id: "tmpl-42", type: "workforce" })}
        useLabel="Use Template"
        defaultSetupTime="5 min setup"
        isBusy
        busyLabel="Creating..."
        onUse={onUse}
      />
    );

    expect(screen.getByText("Creating...")).toBeInTheDocument();
    // The card also has a "like" button; the "Use" button is the last of
    // the two native <button>s. The outer clickable wrapper is a
    // <div role="button">, whose computed accessible name also contains
    // "Creating...", so querying by role/name here would be ambiguous.
    const buttons = container.querySelectorAll("button");
    const useButton = buttons[buttons.length - 1];
    expect(useButton).toBeDisabled();

    fireEvent.click(useButton!);
    // The whole card is also clickable; clicking anywhere on it while busy
    // must not fire onUse either.
    fireEvent.click(container.querySelector('[role="button"]')!);

    expect(onUse).not.toHaveBeenCalled();
  });
});

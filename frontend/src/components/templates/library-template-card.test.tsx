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

  it("renders the persona's name/role instead of the template name when present", () => {
    render(
      <LibraryTemplateCard
        template={makeTemplate({
          name: "Email Lead Response Agent",
          persona: {
            name: "Leo",
            role: "Email Lead Response Agent",
            avatar: "/marketplace/avatars/leo.png",
            intro: "Hi — I'm Leo.",
            kickoff_questions: [],
          },
        })}
        useLabel="Use Template"
        defaultSetupTime="5 min setup"
        onOpenPersona={{ onOpen: vi.fn(), formatMeetLabel: (name) => `Meet ${name}` }}
        onUse={vi.fn()}
      />
    );

    expect(screen.getByText("Leo")).toBeInTheDocument();
    expect(screen.getByText("Email Lead Response Agent")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Meet Leo" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Leo" })).toHaveAttribute(
      "src",
      "/marketplace/avatars/leo.png"
    );
  });

  it("calls onOpen (not onUse) when a persona-bearing card is activated", () => {
    const onUse = vi.fn();
    const onOpen = vi.fn();
    render(
      <LibraryTemplateCard
        template={makeTemplate({
          id: "sales-email-lead-response-agent",
          persona: {
            name: "Leo",
            role: "Email Lead Response Agent",
            avatar: null,
            intro: "Hi — I'm Leo.",
            kickoff_questions: [],
          },
        })}
        useLabel="Use Template"
        defaultSetupTime="5 min setup"
        onOpenPersona={{ onOpen, formatMeetLabel: (name) => `Meet ${name}` }}
        onUse={onUse}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Meet Leo" }));

    expect(onOpen).toHaveBeenCalledWith("sales-email-lead-response-agent");
    expect(onUse).not.toHaveBeenCalled();
  });

  it("falls back entirely to onUse's label and behavior when onOpenPersona is omitted", () => {
    // Regression test: onOpen and formatMeetLabel used to be independent
    // optional props, so a caller could wire one without the other and get
    // a "Meet {name}" button that silently called onUse. Bundling them into
    // one onOpenPersona prop means omitting it falls back to useLabel too,
    // not just onUse - there is no way to render "Meet Leo" without also
    // getting onOpen behavior.
    const onUse = vi.fn();
    render(
      <LibraryTemplateCard
        template={makeTemplate({
          id: "sales-email-lead-response-agent",
          persona: {
            name: "Leo",
            role: "Email Lead Response Agent",
            avatar: null,
            intro: "Hi — I'm Leo.",
            kickoff_questions: [],
          },
        })}
        useLabel="Use Template"
        defaultSetupTime="5 min setup"
        onUse={onUse}
      />
    );

    expect(screen.queryByRole("button", { name: "Meet Leo" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Use Template" }));

    expect(onUse).toHaveBeenCalledWith("sales-email-lead-response-agent");
  });

  it("ignores persona and keeps the workforce click behavior for a workforce-type template", () => {
    const onUse = vi.fn();
    const onOpen = vi.fn();
    render(
      <LibraryTemplateCard
        template={makeTemplate({
          id: "growth-workforce",
          type: "workforce",
          agent_count: 3,
          persona: {
            name: "Leo",
            role: "Email Lead Response Agent",
            avatar: null,
            intro: "Hi — I'm Leo.",
            kickoff_questions: [],
          },
        })}
        useLabel="Use Template"
        defaultSetupTime="5 min setup"
        workforceBadgeLabel="Workforce"
        onOpenPersona={{ onOpen, formatMeetLabel: (name) => `Meet ${name}` }}
        onUse={onUse}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Use Template" }));

    expect(onUse).toHaveBeenCalledWith("growth-workforce");
    expect(onOpen).not.toHaveBeenCalled();
  });

  it("does not lock a persona card's onOpenPersona navigation while a sibling workforce is being created", () => {
    // Regression test: the cross-card `disabled` lock exists to stop a
    // slow workforce-creation request from racing a click elsewhere, but
    // onOpenPersona is pure client-side navigation with nothing to race -
    // it must stay active even while `disabled` is true for that reason.
    const onOpen = vi.fn();
    render(
      <LibraryTemplateCard
        template={makeTemplate({
          id: "sales-email-lead-response-agent",
          persona: {
            name: "Leo",
            role: "Email Lead Response Agent",
            avatar: null,
            intro: "Hi — I'm Leo.",
            kickoff_questions: [],
          },
        })}
        useLabel="Use Template"
        defaultSetupTime="5 min setup"
        onOpenPersona={{ onOpen, formatMeetLabel: (name) => `Meet ${name}` }}
        onUse={vi.fn()}
        disabled
      />
    );

    const button = screen.getByRole("button", { name: "Meet Leo" });
    expect(button).not.toBeDisabled();

    fireEvent.click(button);

    expect(onOpen).toHaveBeenCalledWith("sales-email-lead-response-agent");
  });

  describe("hero variant", () => {
    const MAYA_TEMPLATE = makeTemplate({
      id: "marketing-social-media-content-manager",
      description: "Turn one brief into ready-to-publish posts and visuals.",
      features: [
        "Transforms topics or briefs into platform-native copy",
        "Creates matching visuals at correct dimensions",
        "Applies brand tone, voice, and visual guidelines",
      ],
      tool_categories: ["basic", "web_search", "browser", "file", "image", "knowledge", "mcp:LinkedIn"],
      skills: ["static-visual-design"],
      persona: {
        name: "Maya",
        role: "Social Media Content Manager",
        avatar: null,
        intro: "Hi — I'm Maya.",
        kickoff_questions: [],
      },
    });

    it("shows the full description, feature bullets, capability tags, and a 'most used' badge", () => {
      render(
        <LibraryTemplateCard
          template={MAYA_TEMPLATE}
          useLabel="Use Template"
          defaultSetupTime="5 min setup"
          variant="hero"
          heroBadgeLabel="Most used"
          formatToolLabel={(category) => `[${category}]`}
          onOpenPersona={{ onOpen: vi.fn(), formatMeetLabel: (name) => `Meet ${name}` }}
          onUse={vi.fn()}
        />
      );

      expect(screen.getByText("Most used")).toBeInTheDocument();
      expect(
        screen.getByText("Turn one brief into ready-to-publish posts and visuals.")
      ).toBeInTheDocument();
      expect(
        screen.getByText("Transforms topics or briefs into platform-native copy")
      ).toBeInTheDocument();
      // basic/browser/knowledge/mcp: are excluded from the tag row; skills
      // are appended verbatim - see tool-category-labels.test.ts for the
      // filtering behavior itself.
      expect(screen.getByText("[web_search]")).toBeInTheDocument();
      expect(screen.getByText("[file]")).toBeInTheDocument();
      expect(screen.getByText("[image]")).toBeInTheDocument();
      expect(screen.getByText("static-visual-design")).toBeInTheDocument();
      expect(screen.queryByText("[basic]")).not.toBeInTheDocument();
      expect(screen.queryByText("[browser]")).not.toBeInTheDocument();
    });

    it("does not show the badge, bullets, or tags for the default variant, and truncates the description", () => {
      render(
        <LibraryTemplateCard
          template={MAYA_TEMPLATE}
          useLabel="Use Template"
          defaultSetupTime="5 min setup"
          heroBadgeLabel="Most used"
          formatToolLabel={(category) => `[${category}]`}
          onOpenPersona={{ onOpen: vi.fn(), formatMeetLabel: (name) => `Meet ${name}` }}
          onUse={vi.fn()}
        />
      );

      expect(screen.queryByText("Most used")).not.toBeInTheDocument();
      expect(
        screen.queryByText("Transforms topics or briefs into platform-native copy")
      ).not.toBeInTheDocument();
      expect(screen.queryByText("[web_search]")).not.toBeInTheDocument();
      // The description still renders (truncated via CSS line-clamp, not
      // removed from the DOM).
      expect(
        screen.getByText("Turn one brief into ready-to-publish posts and visuals.")
      ).toBeInTheDocument();
    });

    it("ignores the hero variant for a template with no persona", () => {
      render(
        <LibraryTemplateCard
          template={makeTemplate({ id: "growth-workforce", type: "workforce", agent_count: 2 })}
          useLabel="Use Template"
          defaultSetupTime="5 min setup"
          workforceBadgeLabel="Workforce"
          variant="hero"
          heroBadgeLabel="Most used"
          onUse={vi.fn()}
        />
      );

      expect(screen.queryByText("Most used")).not.toBeInTheDocument();
    });
  });
});

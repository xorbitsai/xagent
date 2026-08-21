import React from "react";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiRequestMock = vi.hoisted(() => vi.fn());
const toastErrorMock = vi.hoisted(() => vi.fn());
const routerPushMock = vi.hoisted(() => vi.fn());
const searchParamsMock = vi.hoisted(() => ({ value: new URLSearchParams() }));

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper"
  );
  return { ...actual, apiRequest: apiRequestMock };
});

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils");
  return { ...actual, getApiUrl: () => "http://api.local" };
});

vi.mock("sonner", () => ({
  toast: { error: toastErrorMock },
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: routerPushMock, replace: vi.fn() }),
  useSearchParams: () => searchParamsMock.value,
}));

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string | number>) =>
      vars ? `${key}:${JSON.stringify(vars)}` : key,
    tDynamic: (_key: string, fallback: string) => fallback,
    locale: "en",
  }),
}));

import TemplatesPage from "./page";
import type { Template } from "@/types/template";

function makeTemplate(overrides: Partial<Template> = {}): Template {
  return {
    id: "sales_assistant",
    name: "Sales Assistant",
    category: "Sales",
    description: "A sales agent template",
    features: [],
    connections: [],
    setup_time: "5 min setup",
    tags: [],
    author: "Xagent",
    version: "1.0",
    views: 0,
    likes: 0,
    used_count: 0,
    type: "agent",
    ...overrides,
  };
}

const LEO_PERSONA = {
  name: "Leo",
  role: "Email Lead Response Agent",
  avatar: null,
  intro: "Hi — I'm Leo.",
  kickoff_questions: [],
};

const AGENT_TEMPLATE = makeTemplate();
const PERSONA_TEMPLATE = makeTemplate({
  id: "sales-email-lead-response-agent",
  name: "Email Lead Response Agent",
  persona: LEO_PERSONA,
});
const WORKFORCE_TEMPLATE = makeTemplate({
  id: "growth_workforce",
  name: "Growth Marketing Workforce",
  category: "Marketing",
  description: "A workforce template",
  type: "workforce",
  agent_count: 3,
});

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

/** Dispatches the page's requests by URL; POST behavior is injectable. */
function installApiMock(
  onUseAsWorkforce: () => Promise<Response> | Response = () =>
    jsonResponse({ workforce_id: 7 })
) {
  apiRequestMock.mockImplementation(async (url: string, options?: RequestInit) => {
    if (url.includes("/use-as-workforce")) return onUseAsWorkforce();
    if (url.includes("/api/templates/?lang=")) {
      return jsonResponse([AGENT_TEMPLATE, WORKFORCE_TEMPLATE]);
    }
    // Legacy /use analytics ping and anything else.
    void options;
    return jsonResponse({});
  });
}

function cardOf(name: string): HTMLElement {
  const card = screen.getByText(name).closest('[role="button"]');
  expect(card).not.toBeNull();
  return card as HTMLElement;
}

async function renderPage() {
  render(<TemplatesPage />);
  await waitFor(() => {
    expect(screen.getByText("Sales Assistant")).toBeInTheDocument();
  });
}

beforeEach(() => {
  apiRequestMock.mockReset();
  toastErrorMock.mockReset();
  routerPushMock.mockReset();
  searchParamsMock.value = new URLSearchParams();
});

afterEach(cleanup);

describe("TemplatesPage type filtering", () => {
  it("shows all templates by default and filters to workforce-only via the type tab", async () => {
    installApiMock();
    await renderPage();

    expect(screen.getByText("Growth Marketing Workforce")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "templates.typeFilter.workforce" })
    );

    expect(screen.getByText("Growth Marketing Workforce")).toBeInTheDocument();
    expect(screen.queryByText("Sales Assistant")).not.toBeInTheDocument();
  });

  it("seeds the type filter from the ?type= query param", async () => {
    searchParamsMock.value = new URLSearchParams("type=workforce");
    installApiMock();
    render(<TemplatesPage />);

    await waitFor(() => {
      expect(screen.getByText("Growth Marketing Workforce")).toBeInTheDocument();
    });
    expect(screen.queryByText("Sales Assistant")).not.toBeInTheDocument();
  });
});

describe("TemplatesPage persona routing", () => {
  function installPersonaApiMock() {
    apiRequestMock.mockImplementation(async (url: string) => {
      if (url.includes("/api/templates/?lang=")) {
        return jsonResponse([AGENT_TEMPLATE, PERSONA_TEMPLATE]);
      }
      return jsonResponse({});
    });
  }

  it("opens the marketplace detail page for a persona-bearing agent card instead of instantiating it", async () => {
    installPersonaApiMock();
    render(<TemplatesPage />);
    await waitFor(() => {
      expect(screen.getByText("Leo")).toBeInTheDocument();
    });

    fireEvent.click(
      screen.getByRole("button", {
        name: 'templates.marketplace.meet:{"name":"Leo"}',
      })
    );

    expect(routerPushMock).toHaveBeenCalledWith(
      "/templates/sales-email-lead-response-agent"
    );
    // Unlike the old flow, opening the detail page never pings /use or
    // touches /build/new - "Meet" is purely navigational.
    expect(apiRequestMock).not.toHaveBeenCalledWith(
      expect.stringContaining("/use"),
      expect.anything()
    );
  });

  it("routes an already-hired persona card straight to its agent's chat, not through the detail page", async () => {
    // The card shows "Chat" (not "Meet {name}") once hired - activating it
    // must actually deliver that, matching page-client.tsx's own
    // hired-aware primary-button behavior one navigation earlier.
    apiRequestMock.mockImplementation(async (url: string) => {
      if (url.includes("/api/templates/?lang=")) {
        return jsonResponse([
          AGENT_TEMPLATE,
          { ...PERSONA_TEMPLATE, hired: true, hired_agent_id: 42 },
        ]);
      }
      return jsonResponse({});
    });
    render(<TemplatesPage />);
    await waitFor(() => {
      expect(screen.getByText("Leo")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: "templates.marketplace.chat" }));

    expect(routerPushMock).toHaveBeenCalledWith("/agent/42");
    expect(routerPushMock).not.toHaveBeenCalledWith(
      expect.stringContaining("/templates/sales-email-lead-response-agent")
    );
  });

  it("falls back to the old builder-prefill flow for an agent template with no persona", async () => {
    installPersonaApiMock();
    render(<TemplatesPage />);
    await waitFor(() => {
      expect(screen.getByText("Sales Assistant")).toBeInTheDocument();
    });

    fireEvent.click(cardOf("Sales Assistant"));

    await waitFor(() => {
      expect(routerPushMock).toHaveBeenCalledWith("/build/new?template=sales_assistant");
    });
  });
});

describe("TemplatesPage featured section", () => {
  const MAYA = makeTemplate({
    id: "marketing-social-media-content-manager",
    name: "Social Media Content Manager",
    category: "Marketing",
    featured: true,
    used_count: 84,
    tool_categories: ["basic", "web_search", "image"],
    skills: ["static-visual-design"],
    persona: {
      name: "Maya",
      role: "Social Media Content Manager",
      avatar: null,
      intro: "Hi — I'm Maya.",
      kickoff_questions: [],
    },
  });
  const SOPHIE = makeTemplate({
    id: "support-ai-chatbot-agent",
    name: "AI Chatbot Agent",
    category: "Support",
    featured: true,
    used_count: 62,
    persona: {
      name: "Sophie",
      role: "AI Chatbot Agent",
      avatar: null,
      intro: "Hi — I'm Sophie.",
      kickoff_questions: [],
    },
  });
  const ELLIE = makeTemplate({
    id: "support-inbox-manager",
    name: "Inbox Manager",
    category: "Support",
    featured: true,
    used_count: 18,
    persona: {
      name: "Ellie",
      role: "Inbox Manager",
      avatar: null,
      intro: "Hi — I'm Ellie.",
      kickoff_questions: [],
    },
  });

  const FEATURED_WORKFORCE = makeTemplate({
    id: "growth_workforce",
    name: "Growth Workforce",
    category: "Marketing",
    type: "workforce",
    featured: true,
    used_count: 999,
    persona: null,
  });

  function installFeaturedApiMock(featured: Template[]) {
    apiRequestMock.mockImplementation(async (url: string) => {
      if (url.includes("/api/templates/?lang=")) {
        return jsonResponse([...featured, AGENT_TEMPLATE]);
      }
      return jsonResponse({});
    });
  }

  it("gives the hero treatment to the most-used featured template, regardless of API order", async () => {
    // Sophie has the higher used_count but is listed first - the hero slot
    // must go to Maya (used_count 84) either way.
    installFeaturedApiMock([SOPHIE, MAYA, ELLIE]);
    render(<TemplatesPage />);

    await waitFor(() => {
      expect(screen.getByText("Maya")).toBeInTheDocument();
    });

    // Only the hero card renders feature-bullet-adjacent capability tags
    // (image -> "Image Generation" via the real i18n fallback, or the raw
    // category via this test's tDynamic passthrough mock - either way,
    // the skill name renders verbatim only once, on the hero card).
    expect(screen.getByText("static-visual-design")).toBeInTheDocument();
    // Sophie and Ellie render as compact cards with no capability tags.
    expect(screen.getByText("Sophie")).toBeInTheDocument();
    expect(screen.getByText("Ellie")).toBeInTheDocument();
  });

  it("shows the count of all featured templates in the section header", async () => {
    installFeaturedApiMock([SOPHIE, MAYA, ELLIE]);
    render(<TemplatesPage />);

    await waitFor(() => {
      expect(screen.getByText("Maya")).toBeInTheDocument();
    });

    expect(
      screen.getByText("templates.categoryTitles.featured").parentElement
    ).toHaveTextContent("3");
  });

  it("excludes a featured workforce template (no persona) from the hero slot, even with the highest used_count", async () => {
    // If a workforce template were ever left in the pool, FeaturedSection's
    // [hero, ...rest] destructuring would hand it the hero slot outright -
    // it has no persona, so it'd render as a plain compact card with no
    // "Most used" ribbon in the hero's oversized grid column. Excluding it
    // from the Featured pool entirely just falls it back to its normal
    // category section instead, same as any other non-featured template.
    installFeaturedApiMock([FEATURED_WORKFORCE, SOPHIE, MAYA, ELLIE]);
    render(<TemplatesPage />);

    await waitFor(() => {
      expect(screen.getByText("Maya")).toBeInTheDocument();
    });

    expect(screen.getByText("static-visual-design")).toBeInTheDocument();

    const featuredSection = screen
      .getByText("templates.categoryTitles.featured")
      .closest("section");
    expect(featuredSection).not.toBeNull();
    expect(featuredSection).toHaveTextContent("3");
    expect(
      within(featuredSection as HTMLElement).queryByText("Growth Workforce")
    ).not.toBeInTheDocument();

    // Still browsable via its own category section, just not featured.
    expect(screen.getByText("Growth Workforce")).toBeInTheDocument();
  });
});

describe("TemplatesPage workforce creation", () => {
  it("POSTs to use-as-workforce with the locale and navigates to the new canvas", async () => {
    installApiMock(() => jsonResponse({ workforce_id: 7 }));
    await renderPage();

    fireEvent.click(cardOf("Growth Marketing Workforce"));

    await waitFor(() => {
      expect(routerPushMock).toHaveBeenCalledWith("/workforces/7?view=canvas");
    });
    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/templates/growth_workforce/use-as-workforce?lang=en",
      { method: "POST" }
    );
  });

  it("maps the structured unpublished-agent error to a toast naming the agent", async () => {
    installApiMock(() =>
      jsonResponse(
        {
          detail: {
            code: "workforce_worker_unpublished",
            message: "unused by the frontend",
            params: { agent_name: "GA Analyzer" },
          },
        },
        400
      )
    );
    await renderPage();

    fireEvent.click(cardOf("Growth Marketing Workforce"));

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        'templates.errors.useWorkforceUnpublishedAgent:{"agentName":"GA Analyzer"}'
      );
    });
    expect(routerPushMock).not.toHaveBeenCalled();
  });

  it("maps the structured conflict code and falls back to the generic message otherwise", async () => {
    installApiMock(() =>
      jsonResponse(
        { detail: { code: "workforce_create_conflict", message: "unused" } },
        409
      )
    );
    await renderPage();
    fireEvent.click(cardOf("Growth Marketing Workforce"));
    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith("templates.errors.useWorkforceRetry");
    });

    // Plain-string details (other endpoints' errors) fall back to generic.
    toastErrorMock.mockReset();
    installApiMock(() =>
      jsonResponse({ detail: "Template is not a workforce template" }, 400)
    );
    fireEvent.click(cardOf("Growth Marketing Workforce"));
    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        "templates.errors.useWorkforceFailed"
      );
    });
  });

  it("disables every other card while a workforce is being created", async () => {
    let resolveUse: (response: Response) => void = () => {};
    installApiMock(
      () => new Promise<Response>((resolve) => (resolveUse = resolve))
    );
    await renderPage();

    fireEvent.click(cardOf("Growth Marketing Workforce"));

    await waitFor(() => {
      expect(cardOf("Sales Assistant")).toHaveAttribute("aria-disabled", "true");
    });

    // Clicking the disabled agent card is inert: no /use ping, no navigation.
    const callsBefore = apiRequestMock.mock.calls.length;
    fireEvent.click(cardOf("Sales Assistant"));
    expect(apiRequestMock.mock.calls.length).toBe(callsBefore);
    expect(routerPushMock).not.toHaveBeenCalled();

    resolveUse(jsonResponse({ workforce_id: 7 }));
    await waitFor(() => {
      expect(routerPushMock).toHaveBeenCalledWith("/workforces/7?view=canvas");
    });
  });

  it("does not navigate or toast when the request completes after unmount", async () => {
    let resolveUse: (response: Response) => void = () => {};
    installApiMock(
      () => new Promise<Response>((resolve) => (resolveUse = resolve))
    );
    await renderPage();

    fireEvent.click(cardOf("Growth Marketing Workforce"));
    cleanup();

    resolveUse(jsonResponse({ workforce_id: 7 }));
    // Give the resolved promise chain a tick to (incorrectly) fire if the
    // mounted guard were missing.
    await new Promise((tick) => setTimeout(tick, 0));

    expect(routerPushMock).not.toHaveBeenCalled();
    expect(toastErrorMock).not.toHaveBeenCalled();
  });
});

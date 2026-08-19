import React from "react";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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

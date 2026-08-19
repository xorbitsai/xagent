import React from "react";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiRequestMock = vi.hoisted(() => vi.fn());
const toastErrorMock = vi.hoisted(() => vi.fn());
const routerPushMock = vi.hoisted(() => vi.fn());
const paramsMock = vi.hoisted(() => ({ value: { id: "sales-email-lead-response-agent" } }));
const hireAgentFromTemplateMock = vi.hoisted(() => vi.fn());

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
  useParams: () => paramsMock.value,
}));

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string | number>) =>
      vars ? `${key}:${JSON.stringify(vars)}` : key,
    tDynamic: (_key: string, fallback: string) => fallback,
    locale: "en",
  }),
}));

vi.mock("@/lib/hire-agent", () => ({
  hireAgentFromTemplate: hireAgentFromTemplateMock,
}));

import TemplateDetailPage from "./page-client";
import type { TemplateDetail } from "@/types/template";

function makeTemplate(overrides: Partial<TemplateDetail> = {}): TemplateDetail {
  return {
    id: "sales-email-lead-response-agent",
    name: "Email Lead Response Agent",
    category: "Sales",
    description: "Scores inbound lead emails and books meetings.",
    features: ["Triggers on new inbound lead emails", "Scores leads against your criteria"],
    connections: [{ name: "HubSpot", logo: "https://example.com/hubspot.png" }],
    setup_time: "5 min setup",
    tags: [],
    author: "Xagent",
    version: "1.0",
    views: 0,
    likes: 0,
    used_count: 3,
    type: "agent",
    hired: false,
    hired_agent_id: null,
    persona: {
      name: "Leo",
      role: "Email Lead Response Agent",
      avatar: "/marketplace/avatars/leo.png",
      intro: "Hi — I'm Leo, your Email Lead Response Agent.",
      kickoff_questions: ["What makes a lead worth chasing for you?"],
    },
    agent_config: {
      instructions: "...",
      skills: ["evidence-based-rag"],
      tool_categories: ["basic", "mcp:HubSpot"],
      execution_mode: "balanced",
    },
    sample_prompts: [{ title: "Score a lead", prompt: "Score this lead and draft a reply" }],
    ...overrides,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

beforeEach(() => {
  apiRequestMock.mockReset();
  toastErrorMock.mockReset();
  routerPushMock.mockReset();
  hireAgentFromTemplateMock.mockReset();
  paramsMock.value = { id: "sales-email-lead-response-agent" };
});

afterEach(cleanup);

describe("TemplateDetailPage", () => {
  it("renders the persona header, WHAT IT DOES, and WHAT'S INCLUDED from the fetched template", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse(makeTemplate()));

    render(<TemplateDetailPage />);

    await waitFor(() => {
      expect(screen.getByText("Leo")).toBeInTheDocument();
    });
    expect(screen.getByText("Email Lead Response Agent")).toBeInTheDocument();
    expect(screen.getByText("Triggers on new inbound lead emails")).toBeInTheDocument();
    expect(screen.getByText("Balanced")).toBeInTheDocument();
    expect(screen.getByText("HubSpot")).toBeInTheDocument();
    expect(screen.getByText("evidence-based-rag")).toBeInTheDocument();
    // The mcp:HubSpot tool_categories entry must not double up with the
    // Connected apps section, which already shows HubSpot from `connections`.
    expect(screen.queryByText("mcp:HubSpot")).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: 'templates.marketplace.hire:{"name":"Leo"}' })
    ).toBeInTheDocument();
  });

  it("shows a not-found state and a way back when the template 404s", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse({ detail: "Template not found" }, 404));

    render(<TemplateDetailPage />);

    await waitFor(() => {
      expect(screen.getByText("templates.marketplace.notFound")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /templates.marketplace.back/ }));
    expect(routerPushMock).toHaveBeenCalledWith("/templates");
  });

  it("shows a distinct retryable error (not the not-found message) for a transient load failure, and retries the fetch", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse({ detail: "boom" }, 500));

    render(<TemplateDetailPage />);

    await waitFor(() => {
      expect(screen.getByText("templates.marketplace.loadFailed")).toBeInTheDocument();
    });
    expect(screen.queryByText("templates.marketplace.notFound")).not.toBeInTheDocument();

    apiRequestMock.mockResolvedValueOnce(jsonResponse(makeTemplate()));
    fireEvent.click(screen.getByRole("button", { name: "templates.marketplace.retry" }));

    await waitFor(() => {
      expect(screen.getByText("Leo")).toBeInTheDocument();
    });
    expect(apiRequestMock).toHaveBeenCalledTimes(2);
  });

  it("shows the same retryable error for a network failure (thrown fetch) as for a non-2xx response", async () => {
    apiRequestMock.mockRejectedValueOnce(new Error("network down"));

    render(<TemplateDetailPage />);

    await waitFor(() => {
      expect(screen.getByText("templates.marketplace.loadFailed")).toBeInTheDocument();
    });
  });

  it("hires the agent and navigates to the newly seeded task on success", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse(makeTemplate()));
    hireAgentFromTemplateMock.mockResolvedValueOnce({ taskId: 99, agentId: 5, created: true });

    render(<TemplateDetailPage />);
    await waitFor(() => {
      expect(screen.getByText("Leo")).toBeInTheDocument();
    });

    fireEvent.click(
      screen.getByRole("button", { name: 'templates.marketplace.hire:{"name":"Leo"}' })
    );

    await waitFor(() => {
      expect(routerPushMock).toHaveBeenCalledWith("/task/99");
    });
    expect(hireAgentFromTemplateMock).toHaveBeenCalledWith(
      "sales-email-lead-response-agent",
      expect.objectContaining({ name: "Leo" }),
      expect.objectContaining({
        beforeWeStart: "templates.marketplace.beforeWeStart",
        closingNote: "templates.marketplace.hireClosingNote",
        connectAppsLabel: "chatPage.clarification.connectApps.title",
      }),
      [{ name: "HubSpot", logo: "https://example.com/hubspot.png" }]
    );
  });

  it("routes straight to the agent's chat, without hiring, if a fresh recheck reveals the template was hired elsewhere in the meantime", async () => {
    // Simulates the two-tab race: this tab mounted with hired: false, but
    // by the time the user clicks Hire, another tab/request already
    // completed the hire. hireAgentFromTemplate's resolve step would reuse
    // the same agent either way, but task/create always mints a new task -
    // the recheck must catch this before that happens.
    apiRequestMock.mockResolvedValueOnce(jsonResponse(makeTemplate({ hired: false })));
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse(makeTemplate({ hired: true, hired_agent_id: 77 }))
    );

    render(<TemplateDetailPage />);
    await waitFor(() => {
      expect(screen.getByText("Leo")).toBeInTheDocument();
    });

    fireEvent.click(
      screen.getByRole("button", { name: 'templates.marketplace.hire:{"name":"Leo"}' })
    );

    await waitFor(() => {
      expect(routerPushMock).toHaveBeenCalledWith("/agent/77");
    });
    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled();
  });

  it("proceeds with hiring when the pre-hire freshness recheck itself fails, rather than blocking the action", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse(makeTemplate()));
    apiRequestMock.mockRejectedValueOnce(new Error("network down"));
    hireAgentFromTemplateMock.mockResolvedValueOnce({ taskId: 99, agentId: 5, created: true });

    render(<TemplateDetailPage />);
    await waitFor(() => {
      expect(screen.getByText("Leo")).toBeInTheDocument();
    });

    fireEvent.click(
      screen.getByRole("button", { name: 'templates.marketplace.hire:{"name":"Leo"}' })
    );

    await waitFor(() => {
      expect(routerPushMock).toHaveBeenCalledWith("/task/99");
    });
  });

  it("ignores a second click while a hire is already in flight", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse(makeTemplate()));
    // The pre-hire freshness recheck (fired inside handlePrimaryAction).
    apiRequestMock.mockResolvedValueOnce(jsonResponse(makeTemplate()));
    let resolveHire: (result: { taskId: number; agentId: number; created: boolean }) => void =
      () => {};
    hireAgentFromTemplateMock.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveHire = resolve;
      })
    );

    render(<TemplateDetailPage />);
    await waitFor(() => {
      expect(screen.getByText("Leo")).toBeInTheDocument();
    });

    const button = screen.getByRole("button", {
      name: 'templates.marketplace.hire:{"name":"Leo"}',
    });
    fireEvent.click(button);
    fireEvent.click(button);
    fireEvent.click(button);

    // hiringRef guards synchronously on the first click, but the actual
    // hireAgentFromTemplate call now happens after an awaited freshness
    // recheck - wait for it rather than asserting synchronously.
    await waitFor(() => {
      expect(hireAgentFromTemplateMock).toHaveBeenCalledTimes(1);
    });

    resolveHire({ taskId: 99, agentId: 5, created: true });
    await waitFor(() => {
      expect(routerPushMock).toHaveBeenCalledWith("/task/99");
    });
    expect(hireAgentFromTemplateMock).toHaveBeenCalledTimes(1);
  });

  it("does not navigate or toast when hiring resolves after unmount", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse(makeTemplate()));
    let resolveHire: (result: { taskId: number; agentId: number; created: boolean }) => void =
      () => {};
    hireAgentFromTemplateMock.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveHire = resolve;
      })
    );

    const { unmount } = render(<TemplateDetailPage />);
    await waitFor(() => {
      expect(screen.getByText("Leo")).toBeInTheDocument();
    });

    fireEvent.click(
      screen.getByRole("button", { name: 'templates.marketplace.hire:{"name":"Leo"}' })
    );
    unmount();

    resolveHire({ taskId: 99, agentId: 5, created: true });
    // Give the resolved promise chain a tick to (incorrectly) fire if the
    // mounted guard were missing.
    await new Promise((tick) => setTimeout(tick, 0));

    expect(routerPushMock).not.toHaveBeenCalledWith(expect.stringContaining("/task/"));
    expect(toastErrorMock).not.toHaveBeenCalled();
  });

  it("toasts and re-enables the button when hiring fails", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse(makeTemplate()));
    hireAgentFromTemplateMock.mockRejectedValueOnce(new Error("boom"));

    render(<TemplateDetailPage />);
    await waitFor(() => {
      expect(screen.getByText("Leo")).toBeInTheDocument();
    });

    fireEvent.click(
      screen.getByRole("button", { name: 'templates.marketplace.hire:{"name":"Leo"}' })
    );

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        'templates.marketplace.hireFailed:{"name":"Leo"}'
      );
    });
    expect(routerPushMock).not.toHaveBeenCalledWith(expect.stringContaining("/task/"));
    expect(
      screen.getByRole("button", { name: 'templates.marketplace.hire:{"name":"Leo"}' })
    ).not.toBeDisabled();
  });

  it("routes straight to the agent's chat for an already-hired template, without hiring again", async () => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse(makeTemplate({ hired: true, hired_agent_id: 5 }))
    );

    render(<TemplateDetailPage />);
    await waitFor(() => {
      expect(screen.getByText("Leo")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: "templates.marketplace.chat" }));

    expect(routerPushMock).toHaveBeenCalledWith("/agent/5");
    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled();
  });

  it("links to the old builder-prefill flow only when not yet hired", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse(makeTemplate()));

    render(<TemplateDetailPage />);
    await waitFor(() => {
      expect(screen.getByText("Leo")).toBeInTheDocument();
    });

    fireEvent.click(
      screen.getByRole("button", { name: "templates.marketplace.customizeBeforeHiring" })
    );
    expect(routerPushMock).toHaveBeenCalledWith(
      "/build/new?template=sales-email-lead-response-agent"
    );
  });

  it("does not offer the customize-before-hiring link once hired", async () => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse(makeTemplate({ hired: true, hired_agent_id: 5 }))
    );

    render(<TemplateDetailPage />);
    await waitFor(() => {
      expect(screen.getByText("Leo")).toBeInTheDocument();
    });

    expect(
      screen.queryByRole("button", { name: "templates.marketplace.customizeBeforeHiring" })
    ).not.toBeInTheDocument();
  });
});

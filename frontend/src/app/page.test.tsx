/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiRequestMock = vi.hoisted(() => vi.fn());
const routerPushMock = vi.hoisted(() => vi.fn());
const setPendingMessageMock = vi.hoisted(() => vi.fn());
const setTaskIdMock = vi.hoisted(() => vi.fn());
const translateMock = vi.hoisted(
  () => (key: string, vars?: Record<string, string | number>) => {
    const translations: Record<string, string> = {
      "home.agents.newAgent": "New agent",
      "home.agents.buildTime": "Build in minutes",
      "home.agents.manageAll": "Manage all",
      "home.agents.subtitle": "Click to chat - these are your published, ready-to-run workers",
      "home.agents.title": "Your agents",
      "home.revamp.greeting": "Hello",
      "home.revamp.greetingAfternoon": "Good afternoon",
      "home.revamp.greetingEvening": "Good evening",
      "home.revamp.greetingMorning": "Good morning",
    };
    const value = translations[key] ?? key;
    if (!vars) return value;
    return Object.entries(vars).reduce(
      (current, [name, replacement]) => current.replace(`{${name}}`, String(replacement)),
      value
    );
  }
);

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: routerPushMock }),
}));

vi.mock("next/link", () => ({
  default: ({
    children,
    href,
    ...props
  }: React.AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
}));

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    setPendingMessage: setPendingMessageMock,
    setTaskId: setTaskIdMock,
  }),
}));

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({
    user: { username: "Test User" },
  }),
}));

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    locale: "en",
    t: translateMock,
  }),
}));

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}));

vi.mock("@/components/welcome-modal", () => ({
  WelcomeModal: () => null,
}));

vi.mock("@/components/templates/home-template-card", () => ({
  HomeTemplateCard: ({ template }: { template: { name: string } }) => <div>{template.name}</div>,
}));

import Home from "./page";

const okJson = (data: unknown) =>
  ({
    ok: true,
    json: async () => data,
  }) as Response;

const mockHomeData = (agents: unknown[], templates: unknown[] = []) => {
  apiRequestMock.mockImplementation((url: string) => {
    if (url.includes("/api/templates/")) {
      return Promise.resolve(okJson(templates));
    }
    if (url.includes("/api/chat/tasks")) {
      return Promise.resolve(okJson({ tasks: [] }));
    }
    if (url.endsWith("/api/agents")) {
      return Promise.resolve(okJson(agents));
    }
    return Promise.reject(new Error(`Unexpected request: ${url}`));
  });
};

describe("Home", () => {
  beforeEach(() => {
    localStorage.setItem("hasVisitedXagent", "true");
    apiRequestMock.mockReset();
    routerPushMock.mockReset();
    setPendingMessageMock.mockReset();
    setTaskIdMock.mockReset();
  });

  afterEach(() => {
    cleanup();
    localStorage.clear();
  });

  it("does not fall back to showing draft agents in the published agents section", async () => {
    mockHomeData(
      [
        {
          id: 1,
          name: "Draft-only agent",
          status: "draft",
          updated_at: "2026-06-01T12:00:00Z",
        },
      ],
      [
        {
          id: "template-1",
          name: "Test Template",
          category: "Test Category",
          description: "Template used to wait for async home data.",
          connections: [],
          setup_time: "5 min",
          used_count: 0,
        },
      ]
    );

    render(<Home />);

    expect(await screen.findByText("Test Template")).toBeInTheDocument();
    expect(screen.getByText("New agent")).toBeInTheDocument();
    expect(screen.queryByText("Draft-only agent")).not.toBeInTheDocument();
  });

  it("shows published agents in the published agents section", async () => {
    mockHomeData([
      {
        id: 1,
        name: "Draft agent",
        status: "draft",
        updated_at: "2026-06-01T12:00:00Z",
      },
      {
        id: 2,
        name: "Published agent",
        status: "published",
        updated_at: "2026-06-02T12:00:00Z",
      },
    ]);

    render(<Home />);

    expect(await screen.findByText("Published agent")).toBeInTheDocument();
    expect(screen.queryByText("Draft agent")).not.toBeInTheDocument();
  });
});

/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { McpApp } from "@/contexts/mcp-apps-context";

const mcpAppsMock = vi.hoisted(() => ({
  apps: [] as McpApp[],
  refresh: vi.fn(),
}));
const toastErrorMock = vi.hoisted(() => vi.fn());
const openBuiltinOAuthPopupMock = vi.hoisted(() => vi.fn());

vi.mock("@/contexts/mcp-apps-context", () => ({
  useMcpApps: () => mcpAppsMock,
}));

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ token: "test-token" }),
}));

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string | number>) =>
      vars ? `${key}:${JSON.stringify(vars)}` : key,
  }),
}));

vi.mock("@/components/ui/sonner", () => ({
  toast: { error: toastErrorMock },
}));

vi.mock("@/lib/oauth-connect", () => ({
  openBuiltinOAuthPopup: openBuiltinOAuthPopupMock,
}));

import { ConnectAppsField } from "./connect-apps-field";
import type { Interaction } from "@/contexts/app-context-chat";

function makeApp(overrides: Partial<Record<string, unknown>> = {}) {
  return {
    id: "gmail",
    name: "Gmail",
    description: "",
    icon: "https://example.com/gmail.png",
    users: "",
    transport: "builtin",
    provider: "google",
    category: "Communication",
    is_connected: false,
    ...overrides,
  };
}

const LEO_INTERACTION: Interaction = {
  type: "connect_apps",
  field: "connect_apps",
  label: "Connect your apps",
  apps: ["HubSpot", "Gmail", "Google Calendar"],
};

beforeEach(() => {
  mcpAppsMock.apps = [];
  mcpAppsMock.refresh.mockReset().mockResolvedValue(undefined);
  toastErrorMock.mockReset();
  openBuiltinOAuthPopupMock.mockReset();
});

afterEach(cleanup);

describe("ConnectAppsField", () => {
  it("groups requested apps by provider, one sign-in per provider", () => {
    mcpAppsMock.apps = [
      makeApp({ id: "hubspot", name: "HubSpot", provider: "hubspot" }),
      makeApp({ id: "gmail", name: "Gmail", provider: "google" }),
      makeApp({ id: "google-calendar", name: "Google Calendar", provider: "google" }),
    ];

    render(<ConnectAppsField interaction={LEO_INTERACTION} onSkip={vi.fn()} />);

    // HubSpot is a single-app provider group, so its provider-name line and
    // its app-list line both read "HubSpot".
    expect(screen.getAllByText("HubSpot")).toHaveLength(2);
    expect(screen.getByText("Google")).toBeInTheDocument();
    expect(screen.getByText("Gmail · Google Calendar")).toBeInTheDocument();
    // One provider = one "Connect" action per group, not per app.
    expect(screen.getAllByText("chatPage.clarification.connectApps.connect")).toHaveLength(2);
  });

  it("drops apps with no provider and renders nothing if none of the requested apps resolve", () => {
    mcpAppsMock.apps = [makeApp({ id: "granola", name: "Granola", provider: undefined })];

    const { container } = render(
      <ConnectAppsField
        interaction={{ ...LEO_INTERACTION, apps: ["Granola"] }}
        onSkip={vi.fn()}
      />
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("shows Connected and disables the button when every app in the group is already connected", () => {
    mcpAppsMock.apps = [makeApp({ provider: "google", is_connected: true })];

    render(
      <ConnectAppsField interaction={{ ...LEO_INTERACTION, apps: ["Gmail"] }} onSkip={vi.fn()} />
    );

    const button = screen.getByRole("button", {
      name: "chatPage.clarification.connectApps.connected",
    });
    expect(button).toBeDisabled();
  });

  it("opens the OAuth popup for the group's provider and refreshes afterwards", async () => {
    mcpAppsMock.apps = [makeApp({ provider: "google", is_connected: false })];
    openBuiltinOAuthPopupMock.mockResolvedValue({ success: true });

    render(
      <ConnectAppsField interaction={{ ...LEO_INTERACTION, apps: ["Gmail"] }} onSkip={vi.fn()} />
    );

    fireEvent.click(
      screen.getByRole("button", { name: "chatPage.clarification.connectApps.connect" })
    );

    await waitFor(() => {
      expect(openBuiltinOAuthPopupMock).toHaveBeenCalledWith({
        provider: "google",
        appId: "gmail",
        token: "test-token",
      });
    });
    await waitFor(() => {
      expect(mcpAppsMock.refresh).toHaveBeenCalled();
    });
    expect(toastErrorMock).not.toHaveBeenCalled();
  });

  it("toasts an error when the popup does not succeed", async () => {
    mcpAppsMock.apps = [makeApp({ provider: "google", is_connected: false })];
    openBuiltinOAuthPopupMock.mockResolvedValue({ success: false });

    render(
      <ConnectAppsField interaction={{ ...LEO_INTERACTION, apps: ["Gmail"] }} onSkip={vi.fn()} />
    );

    fireEvent.click(
      screen.getByRole("button", { name: "chatPage.clarification.connectApps.connect" })
    );

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        'chatPage.clarification.connectApps.connectFailed:{"provider":"Google"}'
      );
    });
  });

  it("keeps a second provider's Connect button disabled independently while a first is still in flight", async () => {
    mcpAppsMock.apps = [
      makeApp({ id: "hubspot", name: "HubSpot", provider: "hubspot", is_connected: false }),
      makeApp({ id: "gmail", name: "Gmail", provider: "google", is_connected: false }),
    ];
    let resolveHubspot: (result: { success: boolean }) => void = () => {};
    openBuiltinOAuthPopupMock.mockImplementation((options: { provider: string }) => {
      if (options.provider === "hubspot") {
        return new Promise((resolve) => {
          resolveHubspot = resolve;
        });
      }
      return Promise.resolve({ success: true });
    });

    render(
      <ConnectAppsField
        interaction={{ ...LEO_INTERACTION, apps: ["HubSpot", "Gmail"] }}
        onSkip={vi.fn()}
      />
    );

    const [hubspotButton, googleButton] = screen.getAllByRole("button", {
      name: "chatPage.clarification.connectApps.connect",
    });
    fireEvent.click(hubspotButton);

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "chatPage.clarification.connectApps.connecting" })
      ).toBeDisabled();
    });

    fireEvent.click(googleButton);

    await waitFor(() => {
      expect(mcpAppsMock.refresh).toHaveBeenCalledTimes(1);
    });

    // Google's popup already resolved and refreshed, but HubSpot's is still
    // pending - its button must still read "Connecting..." and stay
    // disabled, not have been cleared by Google's unrelated finally block.
    expect(
      screen.getByRole("button", { name: "chatPage.clarification.connectApps.connecting" })
    ).toBeDisabled();

    resolveHubspot({ success: true });
    await waitFor(() => {
      expect(mcpAppsMock.refresh).toHaveBeenCalledTimes(2);
    });
  });

  it("calls onSkip once when the skip link is clicked, and hides the link afterwards", () => {
    mcpAppsMock.apps = [makeApp({ provider: "google" })];
    const onSkip = vi.fn();

    render(
      <ConnectAppsField interaction={{ ...LEO_INTERACTION, apps: ["Gmail"] }} onSkip={onSkip} />
    );

    const skipLink = screen.getByText("chatPage.clarification.connectApps.skip");
    fireEvent.click(skipLink);

    expect(onSkip).toHaveBeenCalledTimes(1);
    expect(screen.queryByText("chatPage.clarification.connectApps.skip")).not.toBeInTheDocument();
  });
});

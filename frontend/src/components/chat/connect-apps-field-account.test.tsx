/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { McpApp } from "@/contexts/mcp-apps-context";

const mcpAppsMock = vi.hoisted(() => ({
  apps: [] as McpApp[],
  refresh: vi.fn(),
  isLoading: false,
  error: null as string | null,
}));
const toastErrorMock = vi.hoisted(() => vi.fn());
const openBuiltinOAuthPopupMock = vi.hoisted(() => vi.fn());

vi.mock("@/contexts/mcp-apps-context", () => ({
  useMcpApps: () => mcpAppsMock,
}));

vi.mock("@/lib/api-wrapper", () => ({ apiRequest: vi.fn() }));

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
  openMcpOAuthPopup: vi.fn(),
}));

import { ConnectAppsField } from "./connect-apps-field";

const zendeskInput = {
  label: "Zendesk subdomain",
  placeholder: "acme",
  help: "The part before .zendesk.com.",
  suffix: ".zendesk.com",
};

function makeApp(overrides: Partial<McpApp> = {}): McpApp {
  return {
    id: "zendesk",
    name: "Zendesk",
    description: "",
    icon: "",
    users: "",
    transport: "oauth",
    provider: "zendesk",
    category: "Support",
    auth_type: "builtin_oauth",
    is_connected: false,
    oauth_account_input: zendeskInput,
    ...overrides,
  };
}

function renderField(apps: string[]) {
  return render(
    <ConnectAppsField
      interaction={{ type: "connect_apps", field: "connect_apps", label: "Connect", apps }}
      onSkip={vi.fn()}
    />
  );
}

function continueButton(name: string) {
  return screen.getByRole("button", {
    name: `chatPage.clarification.connectApps.continueWith:{"provider":"${name}"}`,
  });
}

function submitAccount(value: string) {
  fireEvent.change(screen.getByLabelText("Zendesk subdomain"), { target: { value } });
  fireEvent.click(screen.getByRole("button", { name: "tools.mcp.accountStep.continue" }));
}

beforeEach(() => {
  mcpAppsMock.apps = [];
  mcpAppsMock.refresh.mockReset().mockResolvedValue(undefined);
  toastErrorMock.mockReset();
  openBuiltinOAuthPopupMock.mockReset();
});

afterEach(cleanup);

describe("ConnectAppsField account step", () => {
  it("asks for the account before opening the popup, then opens it from the same click", async () => {
    mcpAppsMock.apps = [makeApp()];
    openBuiltinOAuthPopupMock.mockResolvedValue({ success: true });
    renderField(["Zendesk"]);

    fireEvent.click(continueButton("Zendesk"));

    expect(openBuiltinOAuthPopupMock).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Zendesk subdomain")).toBeInTheDocument();

    submitAccount("Acme");

    // Called synchronously from the submit click, so the browser still
    // treats window.open as user-initiated.
    expect(openBuiltinOAuthPopupMock).toHaveBeenCalledWith({
      provider: "zendesk",
      appId: "zendesk",
      account: "acme",
    });
    await waitFor(() => expect(mcpAppsMock.refresh).toHaveBeenCalled());
    expect(screen.queryByLabelText("Zendesk subdomain")).not.toBeInTheDocument();
  });

  it("does not open the popup for an invalid account", () => {
    mcpAppsMock.apps = [makeApp()];
    renderField(["Zendesk"]);

    fireEvent.click(continueButton("Zendesk"));
    submitAccount("acme.example.com");

    expect(openBuiltinOAuthPopupMock).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent("tools.mcp.accountStep.invalid");
  });

  it("leaves the row idle when the account step is cancelled", () => {
    mcpAppsMock.apps = [makeApp()];
    renderField(["Zendesk"]);

    fireEvent.click(continueButton("Zendesk"));
    fireEvent.click(screen.getByRole("button", { name: "common.cancel" }));

    expect(openBuiltinOAuthPopupMock).not.toHaveBeenCalled();
    expect(continueButton("Zendesk")).toBeEnabled();
  });

  it("returns to the account step with the last label when the popup closes without connecting", async () => {
    mcpAppsMock.apps = [makeApp()];
    openBuiltinOAuthPopupMock.mockResolvedValue({ success: false });
    renderField(["Zendesk"]);

    fireEvent.click(continueButton("Zendesk"));
    submitAccount("acme");

    expect(await screen.findByText("tools.mcp.accountStep.retryHint")).toBeInTheDocument();
    expect(screen.getByLabelText("Zendesk subdomain")).toHaveValue("acme");
    // The reopened step carries the retry hint; no second message on top.
    await waitFor(() => expect(mcpAppsMock.refresh).toHaveBeenCalled());
    expect(toastErrorMock).not.toHaveBeenCalled();

    // Resubmitting from the reopened step starts a second popup, which shows
    // the connecting key was released after the first attempt.
    submitAccount("acme2");

    await waitFor(() => expect(openBuiltinOAuthPopupMock).toHaveBeenCalledTimes(2));
    expect(openBuiltinOAuthPopupMock.mock.calls[1][0]).toMatchObject({
      provider: "zendesk",
      appId: "zendesk",
      account: "acme2",
    });
  });

  it("does not reopen the account step when the popup was blocked", async () => {
    mcpAppsMock.apps = [makeApp()];
    openBuiltinOAuthPopupMock.mockResolvedValue({ success: false, popupBlocked: true });
    renderField(["Zendesk"]);

    fireEvent.click(continueButton("Zendesk"));
    submitAccount("acme");

    await waitFor(() =>
      expect(toastErrorMock).toHaveBeenCalledWith(
        'chatPage.clarification.connectApps.popupBlocked:{"provider":"Zendesk"}'
      )
    );
    expect(screen.queryByLabelText("Zendesk subdomain")).not.toBeInTheDocument();
  });

  it("asks once and passes the account to the bare batch login when several apps of the provider are unconnected", () => {
    // Only one app in the group carries the field: the bare login needs the
    // account all the same, so it is taken from any app in the group.
    mcpAppsMock.apps = [
      makeApp({ id: "zendesk", name: "Zendesk" }),
      makeApp({ id: "zendesk-chat", name: "Zendesk Chat", oauth_account_input: undefined }),
    ];
    openBuiltinOAuthPopupMock.mockResolvedValue({ success: true });
    renderField(["Zendesk", "Zendesk Chat"]);

    fireEvent.click(continueButton("Zendesk Chat"));
    submitAccount("acme");

    expect(openBuiltinOAuthPopupMock).toHaveBeenCalledTimes(1);
    expect(openBuiltinOAuthPopupMock).toHaveBeenCalledWith({
      provider: "zendesk",
      appId: undefined,
      account: "acme",
    });
  });

  it("keeps the direct popup for apps that do not declare an account input", () => {
    mcpAppsMock.apps = [
      makeApp({ id: "gmail", name: "Gmail", provider: "google", oauth_account_input: undefined }),
    ];
    openBuiltinOAuthPopupMock.mockResolvedValue({ success: true });
    renderField(["Gmail"]);

    fireEvent.click(continueButton("Gmail"));

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(openBuiltinOAuthPopupMock).toHaveBeenCalledTimes(1);
    expect(openBuiltinOAuthPopupMock.mock.calls[0][0].account).toBeUndefined();
  });
});

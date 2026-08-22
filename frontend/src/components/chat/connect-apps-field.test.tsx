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
const openMcpOAuthPopupMock = vi.hoisted(() => vi.fn());
const apiRequestMock = vi.hoisted(() => vi.fn());

vi.mock("@/contexts/mcp-apps-context", () => ({
  useMcpApps: () => mcpAppsMock,
}));

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}));

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils");
  return { ...actual, getApiUrl: () => "http://api.local" };
});

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
  openMcpOAuthPopup: openMcpOAuthPopupMock,
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

function continueButton(appName: string) {
  return screen.getByRole("button", {
    name: `chatPage.clarification.connectApps.continueWith:{"provider":"${appName}"}`,
  });
}

function jsonResponse(data: unknown, init?: ResponseInit) {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

beforeEach(() => {
  mcpAppsMock.apps = [];
  mcpAppsMock.refresh.mockReset().mockResolvedValue(undefined);
  mcpAppsMock.isLoading = false;
  mcpAppsMock.error = null;
  toastErrorMock.mockReset();
  openBuiltinOAuthPopupMock.mockReset();
  openMcpOAuthPopupMock.mockReset();
  apiRequestMock.mockReset();
});

afterEach(cleanup);

describe("ConnectAppsField", () => {
  it("renders one row per app, not merged by provider, in the interaction's own order", () => {
    mcpAppsMock.apps = [
      makeApp({ id: "hubspot", name: "HubSpot", provider: "hubspot" }),
      makeApp({ id: "gmail", name: "Gmail", provider: "google" }),
      makeApp({ id: "google-calendar", name: "Google Calendar", provider: "google" }),
    ];

    render(<ConnectAppsField interaction={LEO_INTERACTION} onSkip={vi.fn()} />);

    expect(screen.getByText("HubSpot")).toBeInTheDocument();
    expect(screen.getByText("Gmail")).toBeInTheDocument();
    expect(screen.getByText("Google Calendar")).toBeInTheDocument();
    // Never merged into one "Gmail · Google Calendar" line.
    expect(screen.queryByText("Gmail · Google Calendar")).not.toBeInTheDocument();
    // Each app gets its own action, addressed by its own name.
    expect(continueButton("HubSpot")).toBeInTheDocument();
    expect(continueButton("Gmail")).toBeInTheDocument();
    expect(continueButton("Google Calendar")).toBeInTheDocument();
  });

  it("shows each app's own real icon, falling back to a generated avatar on load failure", () => {
    mcpAppsMock.apps = [
      makeApp({ provider: "google", icon: "https://example.com/gmail-icon.png" }),
    ];

    const { container } = render(
      <ConnectAppsField interaction={{ ...LEO_INTERACTION, apps: ["Gmail"] }} onSkip={vi.fn()} />
    );

    const img = container.querySelector("img") as HTMLImageElement;
    expect(img).toHaveAttribute("src", "https://example.com/gmail-icon.png");

    fireEvent.error(img);
    expect(img.src).toBe(
      "https://ui-avatars.com/api/?name=Gmail&background=random&color=fff&size=128"
    );
  });

  it("falls back to an app-initial monogram when the app has no icon at all", () => {
    mcpAppsMock.apps = [makeApp({ provider: "google", icon: "" })];

    const { container } = render(
      <ConnectAppsField interaction={{ ...LEO_INTERACTION, apps: ["Gmail"] }} onSkip={vi.fn()} />
    );

    expect(container.querySelector("img")).not.toBeInTheDocument();
    expect(screen.getByText("G")).toBeInTheDocument();
  });

  it("renders nothing if none of the requested app names resolve to a catalog entry", () => {
    mcpAppsMock.apps = [];

    const { container } = render(
      <ConnectAppsField
        interaction={{ ...LEO_INTERACTION, apps: ["Some Unknown App"] }}
        onSkip={vi.fn()}
      />
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("resolves a requested name against an app's id (with a hyphen-for-space variant), not just its display name", () => {
    // Mirrors tests/templates/test_manager.py's
    // test_builtin_template_connections_resolve_to_a_registered_mcp_app,
    // which assumes this exact lenient name/app_id/hyphen matching.
    mcpAppsMock.apps = [
      makeApp({ id: "facebook", name: "Facebook Pages", provider: "meta" }),
    ];

    render(
      <ConnectAppsField interaction={{ ...LEO_INTERACTION, apps: ["facebook"] }} onSkip={vi.fn()} />
    );

    expect(screen.getByText("Facebook Pages")).toBeInTheDocument();
  });

  it("shows a loading message instead of an empty panel while the catalog is still fetching", () => {
    mcpAppsMock.apps = [];
    mcpAppsMock.isLoading = true;

    render(
      <ConnectAppsField interaction={{ ...LEO_INTERACTION, apps: ["Gmail"] }} onSkip={vi.fn()} />
    );

    expect(
      screen.getByText("chatPage.clarification.connectApps.loading")
    ).toBeInTheDocument();
  });

  it("shows the catalog fetch error instead of an empty panel when useMcpApps() fails", () => {
    mcpAppsMock.apps = [];
    mcpAppsMock.error = "tools.mcp.dialog.fetchFailed";

    render(
      <ConnectAppsField interaction={{ ...LEO_INTERACTION, apps: ["Gmail"] }} onSkip={vi.fn()} />
    );

    expect(screen.getByText("tools.mcp.dialog.fetchFailed")).toBeInTheDocument();
  });

  it("shows a Connected badge instead of a button once an app is connected", () => {
    mcpAppsMock.apps = [makeApp({ provider: "google", is_connected: true })];

    render(
      <ConnectAppsField interaction={{ ...LEO_INTERACTION, apps: ["Gmail"] }} onSkip={vi.fn()} />
    );

    expect(screen.getByText("chatPage.clarification.connectApps.connected")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /connectApps\.continueWith/ })
    ).not.toBeInTheDocument();
  });

  it("opens the OAuth popup with this app's own id when it's the only unconnected app under its provider", async () => {
    mcpAppsMock.apps = [makeApp({ provider: "google", is_connected: false })];
    openBuiltinOAuthPopupMock.mockResolvedValue({ success: true });

    render(
      <ConnectAppsField interaction={{ ...LEO_INTERACTION, apps: ["Gmail"] }} onSkip={vi.fn()} />
    );

    fireEvent.click(continueButton("Gmail"));

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

  it("runs the bare app_id-less login when more than one app under the same provider is unconnected, from either row's button", async () => {
    mcpAppsMock.apps = [
      makeApp({ id: "gmail", name: "Gmail", provider: "google", is_connected: false }),
      makeApp({
        id: "google-calendar",
        name: "Google Calendar",
        provider: "google",
        is_connected: false,
      }),
    ];
    openBuiltinOAuthPopupMock.mockResolvedValue({ success: true });

    render(
      <ConnectAppsField
        interaction={{ ...LEO_INTERACTION, apps: ["Gmail", "Google Calendar"] }}
        onSkip={vi.fn()}
      />
    );

    // Clicking Google Calendar's own row - not Gmail's - still runs the bare
    // login that covers both, proving the two rows share one provider group
    // under the hood despite rendering separately.
    fireEvent.click(continueButton("Google Calendar"));

    await waitFor(() => {
      expect(openBuiltinOAuthPopupMock).toHaveBeenCalledWith({
        provider: "google",
        appId: undefined,
        token: "test-token",
      });
    });
  });

  it("targets the one remaining app directly once a bare login has already covered the rest", async () => {
    mcpAppsMock.apps = [
      makeApp({ id: "instagram", name: "Instagram", provider: "meta", is_connected: true }),
      makeApp({
        id: "facebook",
        name: "Facebook Pages",
        provider: "meta",
        is_connected: false,
      }),
    ];
    openBuiltinOAuthPopupMock.mockResolvedValue({ success: true });

    render(
      <ConnectAppsField
        interaction={{ ...LEO_INTERACTION, apps: ["Instagram", "Facebook Pages"] }}
        onSkip={vi.fn()}
      />
    );

    fireEvent.click(continueButton("Facebook Pages"));

    await waitFor(() => {
      // Facebook Pages needs its own app-scoped grant (it can never be
      // covered by the bare batch login), so with exactly one app left
      // unconnected the click must target it by id, not go bare again.
      expect(openBuiltinOAuthPopupMock).toHaveBeenCalledWith({
        provider: "meta",
        appId: "facebook",
        token: "test-token",
      });
    });
  });

  it("toasts an error when the popup does not succeed", async () => {
    mcpAppsMock.apps = [makeApp({ provider: "google", is_connected: false })];
    openBuiltinOAuthPopupMock.mockResolvedValue({ success: false });

    render(
      <ConnectAppsField interaction={{ ...LEO_INTERACTION, apps: ["Gmail"] }} onSkip={vi.fn()} />
    );

    fireEvent.click(continueButton("Gmail"));

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        'chatPage.clarification.connectApps.connectFailed:{"provider":"Google"}'
      );
    });
  });

  it("capitalizes github's failure toast as 'GitHub', not the generic capitalize() fallback's 'Github'", async () => {
    mcpAppsMock.apps = [
      makeApp({ id: "github", name: "GitHub", provider: "github", is_connected: false }),
    ];
    openBuiltinOAuthPopupMock.mockResolvedValue({ success: false });

    render(
      <ConnectAppsField interaction={{ ...LEO_INTERACTION, apps: ["GitHub"] }} onSkip={vi.fn()} />
    );

    fireEvent.click(continueButton("GitHub"));

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        'chatPage.clarification.connectApps.connectFailed:{"provider":"GitHub"}'
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

    fireEvent.click(continueButton("HubSpot"));

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "chatPage.clarification.connectApps.connecting" })
      ).toBeDisabled();
    });

    fireEvent.click(continueButton("Gmail"));

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

  it("connects a remote-MCP OAuth app (mcp_oauth, e.g. Granola) via its own popup flow, independent of provider grouping", async () => {
    mcpAppsMock.apps = [
      makeApp({
        id: "granola",
        name: "Granola",
        provider: undefined,
        auth_type: "mcp_oauth",
        is_connected: false,
      }),
    ];
    openMcpOAuthPopupMock.mockResolvedValue({ connected: true });

    render(
      <ConnectAppsField interaction={{ ...LEO_INTERACTION, apps: ["Granola"] }} onSkip={vi.fn()} />
    );

    fireEvent.click(continueButton("Granola"));

    await waitFor(() => {
      expect(openMcpOAuthPopupMock).toHaveBeenCalledWith({ appId: "granola" });
    });
    await waitFor(() => {
      expect(mcpAppsMock.refresh).toHaveBeenCalled();
    });
    expect(openBuiltinOAuthPopupMock).not.toHaveBeenCalled();
    expect(toastErrorMock).not.toHaveBeenCalled();
  });

  it("toasts an error when the mcp_oauth popup does not end up connected", async () => {
    mcpAppsMock.apps = [
      makeApp({ id: "granola", name: "Granola", provider: undefined, auth_type: "mcp_oauth" }),
    ];
    openMcpOAuthPopupMock.mockResolvedValue({ connected: false });

    render(
      <ConnectAppsField interaction={{ ...LEO_INTERACTION, apps: ["Granola"] }} onSkip={vi.fn()} />
    );

    fireEvent.click(continueButton("Granola"));

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        'chatPage.clarification.connectApps.connectFailed:{"provider":"Granola"}'
      );
    });
  });

  it("prefers the backend's own error message over the generic connectFailed toast for mcp_oauth", async () => {
    mcpAppsMock.apps = [
      makeApp({ id: "granola", name: "Granola", provider: undefined, auth_type: "mcp_oauth" }),
    ];
    openMcpOAuthPopupMock.mockResolvedValue({ connected: false, message: "Rate limited, try later" });

    render(
      <ConnectAppsField interaction={{ ...LEO_INTERACTION, apps: ["Granola"] }} onSkip={vi.fn()} />
    );

    fireEvent.click(continueButton("Granola"));

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith("Rate limited, try later");
    });
  });

  it("shows a manual-setup row with a Connect button that opens the API-key dialog for an app with required env vars", () => {
    mcpAppsMock.apps = [
      makeApp({
        id: "aws",
        name: "AWS",
        provider: undefined,
        auth_type: "api_key",
        icon: "",
        is_connected: false,
        launch_config: { required_env: ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"] },
      }),
    ];

    render(
      <ConnectAppsField interaction={{ ...LEO_INTERACTION, apps: ["AWS"] }} onSkip={vi.fn()} />
    );

    expect(screen.getByText("AWS")).toBeInTheDocument();
    expect(screen.getByText("chatPage.clarification.connectApps.manualHint")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /continueWith/ })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }));

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByLabelText("AWS_ACCESS_KEY_ID")).toBeInTheDocument();
    expect(screen.getByLabelText("AWS_SECRET_ACCESS_KEY")).toBeInTheDocument();
  });

  it("shows only a hint, with no action, for an unconnectable app this card has no flow for", () => {
    // A real "unconnectable" shape per the backend's classify_app_auth (not
    // "api_key" with an empty required_env, which the backend never
    // actually produces - api_key implies a non-empty required_env).
    mcpAppsMock.apps = [
      makeApp({
        id: "some-tool",
        name: "Some Tool",
        provider: undefined,
        auth_type: "unconnectable",
        launch_config: undefined,
        icon: "",
        is_connected: false,
      }),
    ];

    render(
      <ConnectAppsField interaction={{ ...LEO_INTERACTION, apps: ["Some Tool"] }} onSkip={vi.fn()} />
    );

    expect(screen.getByText("chatPage.clarification.connectApps.manualHint")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "tools.mcp.dialog.connect" })
    ).not.toBeInTheDocument();
  });

  it("connects a keyless manual app with one click straight to the connect endpoint, no dialog", async () => {
    mcpAppsMock.apps = [
      makeApp({
        id: "chrome",
        name: "Chrome",
        provider: undefined,
        auth_type: "keyless",
        launch_config: undefined,
        icon: "",
        is_connected: false,
      }),
    ];
    apiRequestMock.mockResolvedValue(jsonResponse({ ok: true }));

    render(
      <ConnectAppsField interaction={{ ...LEO_INTERACTION, apps: ["Chrome"] }} onSkip={vi.fn()} />
    );

    // A keyless app needs zero setup, so it must never show the generic
    // "Needs manual setup" hint the other manual rows show.
    expect(
      screen.queryByText("chatPage.clarification.connectApps.manualHint")
    ).not.toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }));

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/mcp/apps/chrome/connect",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({ is_active: true }),
        })
      );
    });
    await waitFor(() => {
      expect(mcpAppsMock.refresh).toHaveBeenCalled();
    });
    expect(toastErrorMock).not.toHaveBeenCalled();
  });

  it("toasts an error when the keyless one-click connect fails", async () => {
    mcpAppsMock.apps = [
      makeApp({
        id: "chrome",
        name: "Chrome",
        provider: undefined,
        auth_type: "keyless",
        launch_config: undefined,
        is_connected: false,
      }),
    ];
    apiRequestMock.mockResolvedValue(jsonResponse({ detail: "no" }, { status: 500 }));

    render(
      <ConnectAppsField interaction={{ ...LEO_INTERACTION, apps: ["Chrome"] }} onSkip={vi.fn()} />
    );

    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }));

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        'chatPage.clarification.connectApps.connectFailed:{"provider":"Chrome"}'
      );
    });
  });

  it("only treats a provider-carrying app as an oauth row when auth_type agrees (or is absent), not any app with a provider", () => {
    // A hypothetical custom catalog app that carries a provider_name but was
    // classified api_key by the backend (non-oauth transport + required_env)
    // - resolveRows must route it to the manual dialog, not an oauth
    // "Continue with" button it could never actually complete.
    mcpAppsMock.apps = [
      makeApp({
        id: "custom-crm",
        name: "Custom CRM",
        provider: "google",
        auth_type: "api_key",
        icon: "",
        is_connected: false,
        launch_config: { required_env: ["CRM_API_KEY"] },
      }),
    ];

    render(
      <ConnectAppsField
        interaction={{ ...LEO_INTERACTION, apps: ["Custom CRM"] }}
        onSkip={vi.fn()}
      />
    );

    expect(screen.queryByRole("button", { name: /continueWith/ })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }));
    expect(screen.getByLabelText("CRM_API_KEY")).toBeInTheDocument();
  });

  it("shows a Connected badge for an already-connected manual-setup app instead of a Connect button", () => {
    mcpAppsMock.apps = [
      makeApp({
        id: "aws",
        name: "AWS",
        provider: undefined,
        auth_type: "api_key",
        is_connected: true,
        launch_config: { required_env: ["AWS_ACCESS_KEY_ID"] },
      }),
    ];

    render(
      <ConnectAppsField interaction={{ ...LEO_INTERACTION, apps: ["AWS"] }} onSkip={vi.fn()} />
    );

    expect(screen.getByText("chatPage.clarification.connectApps.connected")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "tools.mcp.dialog.connect" })
    ).not.toBeInTheDocument();
  });

  it("calls onSkip once when the skip link is clicked, hides the link, and swaps the footer note", () => {
    mcpAppsMock.apps = [makeApp({ provider: "google" })];
    const onSkip = vi.fn();

    render(
      <ConnectAppsField interaction={{ ...LEO_INTERACTION, apps: ["Gmail"] }} onSkip={onSkip} />
    );

    expect(
      screen.getByText('chatPage.clarification.connectApps.privacyNote:{"appName":"Xagent"}')
    ).toBeInTheDocument();

    const skipLink = screen.getByText("chatPage.clarification.connectApps.skip");
    fireEvent.click(skipLink);

    expect(onSkip).toHaveBeenCalledTimes(1);
    expect(screen.queryByText("chatPage.clarification.connectApps.skip")).not.toBeInTheDocument();
    expect(
      screen.getByText("chatPage.clarification.connectApps.skippedNote")
    ).toBeInTheDocument();
  });
});

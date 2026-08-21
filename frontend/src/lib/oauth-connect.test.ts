import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiRequestMock = vi.hoisted(() => vi.fn());

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils");
  return { ...actual, getApiUrl: () => "http://api.local" };
});

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}));

import { openBuiltinOAuthPopup, openMcpOAuthPopup } from "./oauth-connect";
import { MCP_OAUTH_POPUP_WINDOW_NAME } from "@/lib/mcp-utils";

function fakePopup(overrides: Partial<Window> = {}): Window {
  return { closed: false, ...overrides } as Window;
}

function fakeMcpPopup(overrides: Partial<Window> = {}): Window {
  return { closed: false, opener: null, location: { href: "" }, ...overrides } as unknown as Window;
}

function jsonResponse(data: unknown, init?: ResponseInit) {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

describe("openBuiltinOAuthPopup", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("opens the provider login URL with the app id and current page as redirect", () => {
    const windowOpenSpy = vi.spyOn(window, "open").mockReturnValue(fakePopup());

    void openBuiltinOAuthPopup({ provider: "google", appId: "gmail", token: "tok123" });

    expect(windowOpenSpy).toHaveBeenCalledTimes(1);
    const [url] = windowOpenSpy.mock.calls[0];
    expect(url).toBe(
      `http://api.local/api/auth/google/login?token=tok123&app_id=gmail&redirect=${encodeURIComponent(window.location.href)}`
    );
  });

  it("omits app_id from the login URL when no appId is given, for the bare batch-connect login", () => {
    const windowOpenSpy = vi.spyOn(window, "open").mockReturnValue(fakePopup());

    void openBuiltinOAuthPopup({ provider: "google", token: "tok123" });

    const [url] = windowOpenSpy.mock.calls[0];
    expect(url).toBe(
      `http://api.local/api/auth/google/login?token=tok123&redirect=${encodeURIComponent(window.location.href)}`
    );
    expect(url).not.toContain("app_id");
  });

  it("resolves success:false immediately when the popup is blocked", async () => {
    vi.spyOn(window, "open").mockReturnValue(null);

    const result = await openBuiltinOAuthPopup({ provider: "google", appId: "gmail", token: "t" });

    expect(result).toEqual({ success: false, popupBlocked: true });
  });

  it("resolves success:true on an oauth-success postMessage, and ignores unrelated messages", async () => {
    vi.spyOn(window, "open").mockReturnValue(fakePopup());

    const pending = openBuiltinOAuthPopup({ provider: "google", appId: "gmail", token: "t" });

    window.dispatchEvent(new MessageEvent("message", { data: { type: "something-else" } }));
    window.dispatchEvent(new MessageEvent("message", { data: { type: "oauth-success" } }));

    await expect(pending).resolves.toEqual({ success: true });
  });

  it("resolves success:false once the popup is closed without a success message", async () => {
    const popup = fakePopup();
    vi.spyOn(window, "open").mockReturnValue(popup);

    const pending = openBuiltinOAuthPopup({ provider: "google", appId: "gmail", token: "t" });

    (popup as { closed: boolean }).closed = true;
    await vi.advanceTimersByTimeAsync(500);

    await expect(pending).resolves.toEqual({ success: false });
  });

  it("resolves success:false after the timeout even if the popup never closes", async () => {
    vi.spyOn(window, "open").mockReturnValue(fakePopup());

    const pending = openBuiltinOAuthPopup({ provider: "google", appId: "gmail", token: "t" });

    await vi.advanceTimersByTimeAsync(5 * 60 * 1000);

    await expect(pending).resolves.toEqual({ success: false });
  });

  it("stops listening after resolving, so a late success message is a no-op", async () => {
    const popup = fakePopup();
    vi.spyOn(window, "open").mockReturnValue(popup);
    const removeEventListenerSpy = vi.spyOn(window, "removeEventListener");

    const pending = openBuiltinOAuthPopup({ provider: "google", appId: "gmail", token: "t" });
    window.dispatchEvent(new MessageEvent("message", { data: { type: "oauth-success" } }));
    await pending;

    expect(removeEventListenerSpy).toHaveBeenCalledWith("message", expect.any(Function));
  });
});

describe("openMcpOAuthPopup", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    apiRequestMock.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("resolves connected:false and popupBlocked when window.open returns null", async () => {
    vi.spyOn(window, "open").mockReturnValue(null);

    const result = await openMcpOAuthPopup({ appId: "granola" });

    expect(result).toEqual({ connected: false, popupBlocked: true });
    expect(apiRequestMock).not.toHaveBeenCalled();
  });

  it("opens about:blank first, POSTs the connect endpoint, then navigates the popup to the authorization url", async () => {
    const popup = fakeMcpPopup();
    const windowOpenSpy = vi.spyOn(window, "open").mockReturnValue(popup);
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse({ authorization_url: "https://mcp.granola.ai/authorize?x=1" })
    );
    apiRequestMock.mockResolvedValueOnce(jsonResponse([{ id: "granola", is_connected: true }]));

    const pending = openMcpOAuthPopup({ appId: "granola" });

    expect(windowOpenSpy).toHaveBeenCalledWith(
      "about:blank",
      MCP_OAUTH_POPUP_WINDOW_NAME,
      expect.stringContaining("width=600")
    );

    await vi.waitFor(() => {
      expect((popup as unknown as { location: { href: string } }).location.href).toBe(
        "https://mcp.granola.ai/authorize?x=1"
      );
    });
    expect(apiRequestMock).toHaveBeenNthCalledWith(
      1,
      "http://api.local/api/mcp/apps/granola/oauth/connect",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ redirect_after: "/tools?tab=mcp" }),
      })
    );

    (popup as unknown as { closed: boolean }).closed = true;
    await vi.advanceTimersByTimeAsync(500);

    await expect(pending).resolves.toEqual({ connected: true });
    expect(apiRequestMock).toHaveBeenNthCalledWith(2, "http://api.local/api/mcp/apps?location=remote");
  });

  it("resolves connected:false with the backend's own error detail, without ever opening the popup's location, when the connect POST fails", async () => {
    const popup = fakeMcpPopup();
    const closeSpy = vi.fn();
    vi.spyOn(window, "open").mockReturnValue({ ...popup, close: closeSpy } as unknown as Window);
    apiRequestMock.mockResolvedValueOnce(jsonResponse({ detail: "no" }, { status: 500 }));

    const result = await openMcpOAuthPopup({ appId: "granola" });

    expect(result).toEqual({ connected: false, message: "no" });
    expect(closeSpy).toHaveBeenCalledTimes(1);
  });

  it("omits message when the connect POST fails with a body that carries no readable detail", async () => {
    const popup = fakeMcpPopup();
    vi.spyOn(window, "open").mockReturnValue({ ...popup, close: vi.fn() } as unknown as Window);
    apiRequestMock.mockResolvedValueOnce(jsonResponse({}, { status: 500 }));

    const result = await openMcpOAuthPopup({ appId: "granola" });

    expect(result).toEqual({ connected: false });
  });

  it("resolves connected:false when the popup closes but the app is still not connected on recheck", async () => {
    const popup = fakeMcpPopup();
    vi.spyOn(window, "open").mockReturnValue(popup);
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse({ authorization_url: "https://mcp.granola.ai/authorize" })
    );
    apiRequestMock.mockResolvedValueOnce(jsonResponse([{ id: "granola", is_connected: false }]));

    const pending = openMcpOAuthPopup({ appId: "granola" });
    await vi.waitFor(() => expect(apiRequestMock).toHaveBeenCalledTimes(1));

    (popup as unknown as { closed: boolean }).closed = true;
    await vi.advanceTimersByTimeAsync(500);

    await expect(pending).resolves.toEqual({ connected: false });
  });
});

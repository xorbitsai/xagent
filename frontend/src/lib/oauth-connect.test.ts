import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils");
  return { ...actual, getApiUrl: () => "http://api.local" };
});

import { openBuiltinOAuthPopup } from "./oauth-connect";

function fakePopup(overrides: Partial<Window> = {}): Window {
  return { closed: false, ...overrides } as Window;
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

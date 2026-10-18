import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
vi.mock("@/lib/api-wrapper", () => ({ apiRequest: apiRequestMock }))
vi.mock("@/lib/utils", () => ({ getApiUrl: () => "https://api.example" }))

import { isOAuthSuccessMessage, openOAuthAuthorizationPopup } from "./oauth-launch"

const options = { provider: "custom", appId: "app", account: "tenant", name: "OAuth", features: "width=600" }
const navigation = "/api/auth/custom/login?ticket=short-proof"

function response(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), { status })
}

describe("OAuth navigation launch", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    apiRequestMock.mockReset()
  })
  afterEach(() => {
    vi.restoreAllMocks()
    vi.useRealTimers()
  })

  it("opens on the click stack and navigates only after the authenticated POST", async () => {
    const popup = { closed: false, close: vi.fn(), location: { replace: vi.fn() } }
    const open = vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    let release!: (value: Response) => void
    apiRequestMock.mockReturnValue(new Promise(resolve => { release = resolve }))
    expect(openOAuthAuthorizationPopup(options)).toBe(popup)
    expect(open).toHaveBeenCalledWith("about:blank", "OAuth", "width=600")
    expect(popup.location.replace).not.toHaveBeenCalled()
    expect(apiRequestMock).toHaveBeenCalledWith("https://api.example/api/auth/custom/login", expect.objectContaining({
      method: "POST", headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ app_id: "app", account: "tenant", redirect: window.location.origin }),
      signal: expect.any(AbortSignal),
    }))
    release(response({ authorization_url: navigation, callback_origin: "https://callback.example" }))
    await vi.advanceTimersByTimeAsync(0)
    expect(popup.location.replace).toHaveBeenCalledWith(`https://api.example${navigation}`)
    expect(popup.close).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)
  })

  it("makes no credential request when popups are blocked", () => {
    vi.spyOn(window, "open").mockReturnValue(null)
    expect(openOAuthAuthorizationPopup(options)).toBeNull()
    expect(apiRequestMock).not.toHaveBeenCalled()
  })

  it("accepts only the POST-configured callback origin, source window and app", async () => {
    const popup = { closed: false, close: vi.fn(), location: { replace: vi.fn() } } as unknown as Window
    vi.spyOn(window, "open").mockReturnValue(popup)
    apiRequestMock.mockResolvedValue(response({ authorization_url: navigation, callback_origin: "https://callback.example" }))
    openOAuthAuthorizationPopup(options)
    const valid = { source: popup, origin: "https://callback.example", data: { type: "oauth-success", provider: "app" } }
    expect(isOAuthSuccessMessage(new MessageEvent("message", valid), popup, "app")).toBe(false)
    await vi.advanceTimersByTimeAsync(0)
    expect(isOAuthSuccessMessage(new MessageEvent("message", valid), popup, "app")).toBe(true)
    for (const change of [
      { origin: "https://attacker.example" }, { origin: "https://api.example" },
      { source: window }, { data: { type: "oauth-success", provider: "other" } },
    ]) expect(isOAuthSuccessMessage(new MessageEvent("message", { ...valid, ...change }), popup, "app")).toBe(false)
    apiRequestMock.mockReturnValue(new Promise(() => {}))
    openOAuthAuthorizationPopup(options)
    expect(isOAuthSuccessMessage(new MessageEvent("message", valid), popup, "app")).toBe(false)
  })

  it.each([
    "https://attacker.example/api/auth/custom/login?ticket=proof",
    "//attacker.example/api/auth/custom/login?ticket=proof",
    "javascript:alert(1)",
    "data:text/html,hostile",
    "/api/auth/other/login?ticket=proof",
    "/api/auth/custom/login?token=login-jwt",
    "/api/auth/custom/login?ticket=proof&token=login-jwt",
    "/api/auth/custom/login?ticket=proof&access_token=login-jwt",
    "/api/auth/custom/login?ticket=",
    null,
  ])("rejects an unsafe or malformed navigation response: %s", async (url) => {
    const popup = { closed: false, close: vi.fn(), location: { replace: vi.fn() } }
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    apiRequestMock.mockResolvedValue(response({ authorization_url: url }))
    const onError = vi.fn()
    openOAuthAuthorizationPopup({ ...options, onError })
    await vi.advanceTimersByTimeAsync(0)
    expect(popup.location.replace).not.toHaveBeenCalled()
    expect(popup.close).toHaveBeenCalledOnce()
    expect(onError).toHaveBeenCalledOnce()
    expect(vi.getTimerCount()).toBe(0)
  })

  it("does not navigate a popup the user closed while the POST was pending", async () => {
    const popup = { closed: true, close: vi.fn(), location: { replace: vi.fn() } }
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    apiRequestMock.mockResolvedValue(response({ authorization_url: navigation, callback_origin: "https://callback.example" }))
    openOAuthAuthorizationPopup(options)
    await vi.advanceTimersByTimeAsync(0)
    expect(popup.location.replace).not.toHaveBeenCalled()
    expect(popup.close).not.toHaveBeenCalled()
  })

  it("bounds a stalled launch with an abort, not an authenticated URL fallback", async () => {
    const popup = { closed: false, close: vi.fn(), location: { replace: vi.fn() } }
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    apiRequestMock.mockImplementation((_url, init: RequestInit) => new Promise((_resolve, reject) => {
      init.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")))
    }))
    openOAuthAuthorizationPopup(options)
    await vi.advanceTimersByTimeAsync(30_000)
    expect(popup.close).toHaveBeenCalledOnce()
    expect(popup.location.replace).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)
  })

  it.each([401, 403, 500])("closes the popup on HTTP %s without exposing the body", async status => {
    const popup = { closed: false, close: vi.fn(), location: { replace: vi.fn() } }
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    const log = vi.spyOn(console, "error")
    apiRequestMock.mockResolvedValue(response({ detail: "private credential" }, status))
    openOAuthAuthorizationPopup(options)
    await vi.advanceTimersByTimeAsync(0)
    expect(popup.close).toHaveBeenCalledOnce()
    expect(popup.location.replace).not.toHaveBeenCalled()
    expect(log).not.toHaveBeenCalled()
  })
})

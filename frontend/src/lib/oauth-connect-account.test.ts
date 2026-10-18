import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return { ...actual, getApiUrl: () => "http://api.local" }
})

const apiRequestMock = vi.hoisted(() => vi.fn())
vi.mock("@/lib/api-wrapper", () => ({ apiRequest: apiRequestMock }))

import { openBuiltinOAuthPopup } from "./oauth-connect"

function fakePopup(): Window {
  return { closed: false, close: vi.fn(), location: { replace: vi.fn() } } as unknown as Window
}

async function settle(popup: Window, pending: Promise<unknown>) {
  ;(popup as { closed: boolean }).closed = true
  await vi.advanceTimersByTimeAsync(500)
  await pending
}

describe("openBuiltinOAuthPopup account parameter", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    apiRequestMock.mockReset().mockImplementation(async () => new Response(JSON.stringify({ authorization_url: "/api/auth/zendesk/login?ticket=short-proof", callback_origin: "http://callback.local" })))
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.useRealTimers()
  })

  it("passes the account label in the authenticated POST body", async () => {
    const popup = fakePopup()
    const windowOpenSpy = vi.spyOn(window, "open").mockReturnValue(popup)

    const pending = openBuiltinOAuthPopup({ provider: "zendesk", appId: "zendesk", account: "acme" })

    expect(windowOpenSpy.mock.calls[0][0]).toBe("about:blank")
    expect(apiRequestMock.mock.calls[0][0]).toBe("http://api.local/api/auth/zendesk/login")
    expect(JSON.parse(apiRequestMock.mock.calls[0][1].body)).toEqual({ account: "acme", app_id: "zendesk", redirect: window.location.origin })
    await settle(popup, pending)
  })

  it("passes the account on the bare batch login too", async () => {
    const popup = fakePopup()
    const windowOpenSpy = vi.spyOn(window, "open").mockReturnValue(popup)

    const pending = openBuiltinOAuthPopup({ provider: "zendesk", account: "acme" })

    expect(windowOpenSpy.mock.calls[0][0]).toBe("about:blank")
    expect(JSON.parse(apiRequestMock.mock.calls[0][1].body)).toEqual({ account: "acme", redirect: window.location.origin })
    await settle(popup, pending)
  })

  it("omits the account when none is given", async () => {
    const popup = fakePopup()
    const windowOpenSpy = vi.spyOn(window, "open").mockReturnValue(popup)

    const pending = openBuiltinOAuthPopup({ provider: "google", appId: "gmail" })

    expect(windowOpenSpy.mock.calls[0][0]).toBe("about:blank")
    expect(JSON.parse(apiRequestMock.mock.calls[0][1].body)).toEqual({ app_id: "gmail", redirect: window.location.origin })
    await settle(popup, pending)
  })
})

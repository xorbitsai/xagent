import { beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", () => ({
  getApiUrl: () => "http://api.local",
}))

import {
  createPersonalApiKey,
  listPersonalApiKeys,
  revokePersonalApiKey,
} from "./personal-api-keys-api"

function jsonResponse(body: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  })
}

describe("personal API keys client", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
  })

  it("loads the scoped management list with owner metadata", async () => {
    apiRequestMock.mockResolvedValue(
      jsonResponse({
        items: [{
          id: 7,
          key_prefix: "abc123",
          masked_key: "xag_personal_abc123_••••••••",
          revoked_at: null,
          expires_at: null,
          created_at: "2026-07-22T00:00:00Z",
          owner: { id: 3, username: "alice", email: "alice@example.com" },
        }],
        can_manage_others: true,
      }),
    )

    const result = await listPersonalApiKeys()

    expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/personal-api-keys", {
      method: "GET",
    })
    expect(result.can_manage_others).toBe(true)
    expect(result.items[0].owner.username).toBe("alice")
  })

  it("creates a key through the current-user route", async () => {
    apiRequestMock.mockResolvedValue(
      jsonResponse({
        id: 9,
        full_key: "xag_personal_abc123_secret",
        key_prefix: "abc123",
        created_at: "2026-07-22T00:00:00Z",
        expires_at: null,
      }),
    )

    const result = await createPersonalApiKey()

    expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/me/personal-keys", {
      method: "POST",
    })
    expect(result.full_key).toBe("xag_personal_abc123_secret")
  })

  it("revokes a key through the scoped management route", async () => {
    apiRequestMock.mockResolvedValue(
      jsonResponse({ revoked: true, revoked_at: "2026-07-22T00:00:00Z" }),
    )

    await revokePersonalApiKey(9)

    expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/personal-api-keys/9", {
      method: "DELETE",
    })
  })
})

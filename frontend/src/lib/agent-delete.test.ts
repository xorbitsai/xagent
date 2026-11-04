import { beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper",
  )
  return {
    ...actual,
    apiRequest: apiRequestMock,
  }
})

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return {
    ...actual,
    getApiUrl: () => "http://api.local",
  }
})

import {
  parseAgentDeleteConflict,
  requestAgentDeletion,
} from "./agent-delete"

const conflictPayload = {
  detail: {
    code: "agent_in_use_by_workforce",
    message: "Agent is referenced by workforces",
    references: [
      {
        workforce_id: 7,
        name: "Launch Team",
        status: "draft",
        roles: ["manager", "worker"],
        can_edit: true,
        can_discard: true,
      },
      {
        workforce_id: 8,
        name: "Support Team",
        status: "active",
        roles: ["worker"],
        can_edit: false,
        can_discard: false,
      },
    ],
    has_hidden_references: true,
  },
}

function jsonResponse(data: unknown, init?: ResponseInit) {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  })
}

describe("Agent delete contract", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
  })

  it("parses the exact structured workforce reference contract", () => {
    expect(parseAgentDeleteConflict(conflictPayload)).toEqual(conflictPayload.detail)
  })

  it("accepts a conflict containing only hidden references", () => {
    const payload = {
      detail: {
        ...conflictPayload.detail,
        references: [],
        has_hidden_references: true,
      },
    }

    expect(parseAgentDeleteConflict(payload)).toEqual(payload.detail)
  })

  it.each([
    ["unknown status", { status: "deleted" }],
    ["unknown role", { roles: ["owner"] }],
    ["non-positive id", { workforce_id: 0 }],
    ["invalid discard permission", { status: "active", can_edit: true, can_discard: true }],
  ])("rejects malformed references: %s", (_name, replacement) => {
    const payload = {
      detail: {
        ...conflictPayload.detail,
        references: [
          {
            ...conflictPayload.detail.references[0],
            ...replacement,
          },
        ],
        has_hidden_references: false,
      },
    }

    expect(parseAgentDeleteConflict(payload)).toBeNull()
  })

  it("returns a blocked result for a valid 409 response", async () => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse(conflictPayload, { status: 409 }),
    )

    await expect(requestAgentDeletion(42, "Delete failed")).resolves.toEqual({
      kind: "blocked",
      conflict: conflictPayload.detail,
    })
    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/agents/42",
      { method: "DELETE" },
    )
  })

  it("returns deleted without reading a success body", async () => {
    apiRequestMock.mockResolvedValueOnce(new Response(null, { status: 204 }))

    await expect(requestAgentDeletion(42, "Delete failed")).resolves.toEqual({
      kind: "deleted",
    })
  })

  it("uses the localized fallback when a 409 conflict payload is malformed", async () => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse({
        detail: {
          ...conflictPayload.detail,
          references: [{ ...conflictPayload.detail.references[0], roles: ["owner"] }],
        },
      }, { status: 409 }),
    )

    await expect(requestAgentDeletion(42, "Delete failed")).rejects.toThrow(
      "Delete failed",
    )
  })

  it("uses the localized fallback for the stable delete-failure code", async () => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse({
        detail: {
          code: "agent_delete_failed",
          message: "Failed to delete agent",
        },
      }, { status: 500 }),
    )

    await expect(requestAgentDeletion(42, "Delete failed")).rejects.toThrow(
      "Delete failed",
    )
  })

  it("uses the localized fallback for unexpected structured errors", async () => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse({
        detail: {
          code: "unexpected_delete_error",
          message: "Backend English",
        },
      }, { status: 500 }),
    )

    await expect(requestAgentDeletion(42, "Delete failed")).rejects.toThrow(
      "Delete failed",
    )
  })

  it("uses the localized fallback for network failures", async () => {
    apiRequestMock.mockRejectedValueOnce(new Error("Network connection failed"))

    await expect(requestAgentDeletion(42, "Delete failed")).rejects.toThrow(
      "Delete failed",
    )
  })
})

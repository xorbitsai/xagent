import { beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", () => ({ apiRequest: apiRequestMock }))
vi.mock("@/lib/utils", () => ({ getApiUrl: () => "http://api.local" }))

import {
  ConversationLogExportTooLargeError,
  buildConversationLogFilterQuery,
  downloadConversationLogsCsv,
} from "./conversation-logs-api"

describe("buildConversationLogFilterQuery", () => {
  it("includes only the filters that are set", () => {
    expect(buildConversationLogFilterQuery({ source: "all" }).toString()).toBe("source=all")
    const query = buildConversationLogFilterQuery({
      source: "widget",
      agentId: 7,
      search: "  crm ",
      activeFrom: "2026-06-29T00:00:00.000Z",
      activeTo: null,
    })
    expect(query.get("source")).toBe("widget")
    expect(query.get("agent_id")).toBe("7")
    expect(query.get("search")).toBe("crm")
    expect(query.get("active_from")).toBe("2026-06-29T00:00:00.000Z")
    expect(query.has("active_to")).toBe(false)
  })
})

describe("downloadConversationLogsCsv", () => {
  beforeEach(() => apiRequestMock.mockReset())

  async function failure(status: number, body: unknown): Promise<unknown> {
    apiRequestMock.mockResolvedValue(new Response(JSON.stringify(body), { status }))
    return downloadConversationLogsCsv({ source: "all" }).catch((err: unknown) => err)
  }

  it("maps a size refusal to a typed error carrying the reason and limit", async () => {
    const err = await failure(422, {
      detail: { code: "conversation_log_export_too_large", reason: "bytes", limit: 123 },
    })
    expect(err).toBeInstanceOf(ConversationLogExportTooLargeError)
    expect((err as ConversationLogExportTooLargeError).reason).toBe("bytes")
    expect((err as ConversationLogExportTooLargeError).limit).toBe(123)
  })

  it.each([
    [{ detail: { code: "conversation_log_export_too_large", reason: "x", limit: 1 } }],
    [{ detail: { code: "conversation_log_export_too_large", reason: "rows", limit: "1" } }],
    [{ detail: { code: "other", reason: "rows", limit: 1 } }],
    [{ detail: "nope" }],
    [{ detail: [{ msg: "bad", loc: ["query", "active_from"] }] }],
  ])("treats %j as an ordinary failure", async (body) => {
    const err = await failure(422, body)
    expect(err).toBeInstanceOf(Error)
    expect(err).not.toBeInstanceOf(ConversationLogExportTooLargeError)
  })
})

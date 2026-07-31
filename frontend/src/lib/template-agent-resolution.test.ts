import { beforeEach, describe, expect, it, vi } from "vitest";

const apiRequestMock = vi.hoisted(() => vi.fn());

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper"
  );
  return {
    ...actual,
    apiRequest: apiRequestMock,
  };
});

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils");
  return {
    ...actual,
    getApiUrl: () => "http://api.local",
  };
});

import { resolveAgentForTemplate, toAgentId } from "./template-agent-resolution";

function jsonResponse(data: unknown, init?: ResponseInit) {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

describe("toAgentId", () => {
  it("parses a numeric or numeric-string id", () => {
    expect(toAgentId({ id: 42 })).toBe(42);
    expect(toAgentId({ id: "42" })).toBe(42);
  });

  it("returns null for a missing or non-numeric id", () => {
    expect(toAgentId(null)).toBeNull();
    expect(toAgentId(undefined)).toBeNull();
    expect(toAgentId({ id: "not-a-number" })).toBeNull();
    expect(toAgentId({ id: null })).toBeNull();
    expect(toAgentId({ id: 0 })).toBeNull();
    expect(toAgentId({ id: -3 })).toBeNull();
  });
});

describe("resolveAgentForTemplate", () => {
  beforeEach(() => {
    apiRequestMock.mockReset();
  });

  it("always calls the server's resolve endpoint - no client-side cache shortcut", async () => {
    // Regression test for PR review finding B5: a prior client-side fast
    // path matched against a locally-cached agent list (from GET
    // /api/agents, which can include teammates' agents under a team-scope
    // hook) before ever reaching the server, with no ownership check. The
    // server round-trip must be the only path, every time.
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse({
        agent: { id: 10, name: "Inbox Manager", template_id: "support-inbox-manager", status: "published" },
        created: true,
      })
    );

    const result = await resolveAgentForTemplate("support-inbox-manager");

    expect(result.created).toBe(true);
    expect(result.agent).toMatchObject({ id: 10, status: "published" });
    expect(apiRequestMock).toHaveBeenCalledTimes(1);
    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/agents/from-template/resolve",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ template_id: "support-inbox-manager" }),
      })
    );
  });

  it("reports created=false when the server reused an existing agent", async () => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse({
        agent: { id: 11, name: "Inbox Manager", template_id: "support-inbox-manager", status: "published" },
        created: false,
      })
    );

    const result = await resolveAgentForTemplate("support-inbox-manager");

    expect(result.created).toBe(false);
    expect(result.agent).toMatchObject({ id: 11 });
  });

  it("throws on a non-ok resolve response", async () => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse({ detail: "boom" }, { status: 500 })
    );

    await expect(resolveAgentForTemplate("support-inbox-manager")).rejects.toThrow(
      "Failed to resolve agent from template (500)"
    );
  });

  it("throws on a malformed resolve response body", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse({ created: true }));

    await expect(resolveAgentForTemplate("support-inbox-manager")).rejects.toThrow(
      "Malformed resolve response"
    );
  });
});

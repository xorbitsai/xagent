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

import {
  findExistingAgentForTemplate,
  resolveAgentForTemplate,
  toAgentId,
} from "./template-agent-resolution";

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

describe("findExistingAgentForTemplate", () => {
  it("matches by template_id, not by name", () => {
    const agents = [
      { id: 1, name: "Renamed Agent", template_id: "tmpl-a" },
      { id: 2, name: "tmpl-a", template_id: null },
    ];

    expect(findExistingAgentForTemplate("tmpl-a", agents)?.id).toBe(1);
    expect(findExistingAgentForTemplate("tmpl-missing", agents)).toBeNull();
  });
});

describe("resolveAgentForTemplate", () => {
  beforeEach(() => {
    apiRequestMock.mockReset();
  });

  it("reuses a known agent without any network calls", async () => {
    const known = [{ id: 7, name: "Inbox Manager", template_id: "support-inbox-manager", status: "published" }];

    const result = await resolveAgentForTemplate("support-inbox-manager", known);

    expect(result).toEqual({ agent: known[0], created: false });
    expect(apiRequestMock).not.toHaveBeenCalled();
  });

  it("asks the server's resolve endpoint when no known agent matches", async () => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse({
        agent: { id: 10, name: "Inbox Manager", template_id: "support-inbox-manager", status: "published" },
        created: true,
      })
    );

    const result = await resolveAgentForTemplate("support-inbox-manager", []);

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

    const result = await resolveAgentForTemplate("support-inbox-manager", []);

    expect(result.created).toBe(false);
    expect(result.agent).toMatchObject({ id: 11 });
  });

  it("throws on a non-ok resolve response", async () => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse({ detail: "boom" }, { status: 500 })
    );

    await expect(resolveAgentForTemplate("support-inbox-manager", [])).rejects.toThrow(
      "Failed to resolve agent from template (500)"
    );
  });

  it("throws on a malformed resolve response body", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse({ created: true }));

    await expect(resolveAgentForTemplate("support-inbox-manager", [])).rejects.toThrow(
      "Malformed resolve response"
    );
  });
});

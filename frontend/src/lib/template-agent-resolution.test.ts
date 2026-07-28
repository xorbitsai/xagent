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
  DUPLICATE_AGENT_NAME_DETAIL,
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

    const result = await resolveAgentForTemplate("support-inbox-manager", "Inbox Manager", known);

    expect(result).toEqual({ agent: known[0], created: false });
    expect(apiRequestMock).not.toHaveBeenCalled();
  });

  it("creates and publishes a new agent when none exists locally", async () => {
    apiRequestMock
      .mockResolvedValueOnce(jsonResponse({ id: 10, name: "Inbox Manager", template_id: "support-inbox-manager" }))
      .mockResolvedValueOnce(jsonResponse({ message: "ok" }));

    const result = await resolveAgentForTemplate("support-inbox-manager", "Inbox Manager", []);

    expect(result.created).toBe(true);
    expect(result.agent).toMatchObject({ id: 10, status: "published" });
    expect(apiRequestMock).toHaveBeenNthCalledWith(
      1,
      "http://api.local/api/agents/from-template",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ template_id: "support-inbox-manager" }),
      })
    );
    expect(apiRequestMock).toHaveBeenNthCalledWith(
      2,
      "http://api.local/api/agents/10/publish",
      expect.objectContaining({ method: "POST" })
    );
  });

  it("reuses (and publishes) an agent from this template found only via the 400 lookup", async () => {
    apiRequestMock
      .mockResolvedValueOnce(
        jsonResponse({ detail: DUPLICATE_AGENT_NAME_DETAIL }, { status: 400 })
      )
      .mockResolvedValueOnce(
        jsonResponse([{ id: 11, name: "Inbox Manager", template_id: "support-inbox-manager", status: "draft" }])
      )
      .mockResolvedValueOnce(jsonResponse({ message: "ok" }));

    const result = await resolveAgentForTemplate("support-inbox-manager", "Inbox Manager", []);

    expect(result.created).toBe(false);
    expect(result.agent).toMatchObject({ id: 11, status: "published" });
    expect(apiRequestMock).toHaveBeenNthCalledWith(
      3,
      "http://api.local/api/agents/11/publish",
      expect.objectContaining({ method: "POST" })
    );
  });

  it("retries with a disambiguated name when the 400 collides with an unrelated agent", async () => {
    apiRequestMock
      .mockResolvedValueOnce(
        jsonResponse({ detail: DUPLICATE_AGENT_NAME_DETAIL }, { status: 400 })
      )
      .mockResolvedValueOnce(
        jsonResponse([{ id: 12, name: "Inbox Manager", template_id: null, status: "published" }])
      )
      .mockResolvedValueOnce(jsonResponse({ id: 13, name: "Inbox Manager (12345)" }))
      .mockResolvedValueOnce(jsonResponse({ message: "ok" }));

    const result = await resolveAgentForTemplate("support-inbox-manager", "Inbox Manager", []);

    expect(result.created).toBe(true);
    expect(result.agent).toMatchObject({ id: 13, status: "published" });
    expect(apiRequestMock).toHaveBeenNthCalledWith(
      3,
      "http://api.local/api/agents/from-template",
      expect.objectContaining({
        method: "POST",
        body: expect.stringContaining('"template_id":"support-inbox-manager"'),
      })
    );
  });

  it("throws on a 400 whose detail does not match the duplicate-name contract", async () => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse({ detail: "Something else went wrong" }, { status: 400 })
    );

    await expect(resolveAgentForTemplate("support-inbox-manager", "Inbox Manager", [])).rejects.toThrow();
    expect(apiRequestMock).toHaveBeenCalledTimes(1);
  });

  it("rolls back the draft agent when publish fails after a fresh create", async () => {
    apiRequestMock
      .mockResolvedValueOnce(jsonResponse({ id: 14, name: "Inbox Manager", template_id: "support-inbox-manager" }))
      .mockResolvedValueOnce(jsonResponse({ detail: "publish failed" }, { status: 500 }))
      .mockResolvedValueOnce(jsonResponse({ message: "deleted" }));

    await expect(resolveAgentForTemplate("support-inbox-manager", "Inbox Manager", [])).rejects.toThrow(
      "Failed to publish agent"
    );

    expect(apiRequestMock).toHaveBeenNthCalledWith(
      3,
      "http://api.local/api/agents/14",
      expect.objectContaining({ method: "DELETE" })
    );
  });
});

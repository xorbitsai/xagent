import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ConnectionInfo, PersonaInfo } from "@/types/template";

const apiRequestMock = vi.hoisted(() => vi.fn());
const resolveAgentForTemplateMock = vi.hoisted(() => vi.fn());

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

vi.mock("@/lib/template-agent-resolution", async () => {
  const actual = await vi.importActual<typeof import("@/lib/template-agent-resolution")>(
    "@/lib/template-agent-resolution"
  );
  return {
    ...actual,
    resolveAgentForTemplate: resolveAgentForTemplateMock,
  };
});

import {
  buildConnectAppsInteraction,
  buildSeedAssistantMessage,
  hireAgentFromTemplate,
} from "./hire-agent";

function jsonResponse(data: unknown, init?: ResponseInit) {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

const STRINGS = {
  beforeWeStart: "A few things before I start:",
  closingNote: "Answer what you can - I'll default the rest.",
  connectAppsLabel: "Connect your apps",
};

const LEO_PERSONA: PersonaInfo = {
  name: "Leo",
  role: "Email Lead Response Agent",
  avatar: "/marketplace/avatars/leo.png",
  intro: "Hi — I'm Leo, your Email Lead Response Agent.",
  kickoff_questions: [
    "What makes a lead worth chasing for you?",
    "Which calendar should I offer times from?",
  ],
};

describe("buildSeedAssistantMessage", () => {
  it("joins intro, a kickoff-questions bullet list, and the closing note", () => {
    const message = buildSeedAssistantMessage(LEO_PERSONA, STRINGS);

    expect(message).toBe(
      [
        "Hi — I'm Leo, your Email Lead Response Agent.",
        "A few things before I start:\n\n- What makes a lead worth chasing for you?\n- Which calendar should I offer times from?",
        "Answer what you can - I'll default the rest.",
      ].join("\n\n")
    );
  });

  it("omits the kickoff-questions section, and the closing note that refers back to it, when there are none", () => {
    // closingNote's copy ("Answer what you can...") only makes sense as a
    // reply to the kickoff questions above it - a persona with none must
    // not get a dangling, contextless closing sentence.
    const persona: PersonaInfo = { ...LEO_PERSONA, kickoff_questions: [] };

    const message = buildSeedAssistantMessage(persona, STRINGS);

    expect(message).not.toContain("A few things before I start");
    expect(message).not.toContain(STRINGS.closingNote);
    expect(message).toBe("Hi — I'm Leo, your Email Lead Response Agent.");
  });
});

describe("buildConnectAppsInteraction", () => {
  it("builds a connect_apps interaction from the template's connection names", () => {
    const connections: ConnectionInfo[] = [
      { name: "LinkedIn", logo: "https://example.com/linkedin.png" },
      { name: "Google Drive", logo: "https://example.com/drive.png" },
    ];

    expect(buildConnectAppsInteraction(connections, "Connect your apps")).toEqual({
      type: "connect_apps",
      field: "connect_apps",
      label: "Connect your apps",
      apps: ["LinkedIn", "Google Drive"],
    });
  });

  it("returns null when there are no connections", () => {
    expect(buildConnectAppsInteraction([], "Connect your apps")).toBeNull();
  });
});

describe("hireAgentFromTemplate", () => {
  beforeEach(() => {
    apiRequestMock.mockReset();
    resolveAgentForTemplateMock.mockReset();
  });

  it("resolves under the persona name, then creates a task seeded with the opening message", async () => {
    resolveAgentForTemplateMock.mockResolvedValueOnce({
      agent: { id: 42, name: "Leo", template_id: "sales-email-lead-response-agent" },
      created: true,
    });
    apiRequestMock.mockResolvedValueOnce(jsonResponse({ task_id: 7 }));

    const result = await hireAgentFromTemplate(
      "sales-email-lead-response-agent",
      LEO_PERSONA,
      STRINGS
    );

    expect(result).toEqual({ taskId: 7, agentId: 42, created: true });
    expect(resolveAgentForTemplateMock).toHaveBeenCalledWith(
      "sales-email-lead-response-agent",
      "Leo"
    );

    const [url, init] = apiRequestMock.mock.calls[0];
    expect(url).toBe("http://api.local/api/chat/task/create");
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.agent_id).toBe(42);
    expect(body.title).toBe("Leo — Email Lead Response Agent");
    expect(body.seed_assistant_message).toContain("Hi — I'm Leo");
    expect(body.seed_interactions).toBeUndefined();
  });

  it("attaches a seed_interactions connect_apps card when the template has connections", async () => {
    resolveAgentForTemplateMock.mockResolvedValueOnce({
      agent: { id: 42, name: "Leo", template_id: "sales-email-lead-response-agent" },
      created: true,
    });
    apiRequestMock.mockResolvedValueOnce(jsonResponse({ task_id: 7 }));

    const connections: ConnectionInfo[] = [
      { name: "HubSpot", logo: "https://example.com/hubspot.png" },
    ];

    await hireAgentFromTemplate(
      "sales-email-lead-response-agent",
      LEO_PERSONA,
      STRINGS,
      connections
    );

    const [, init] = apiRequestMock.mock.calls[0];
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.seed_interactions).toEqual([
      {
        type: "connect_apps",
        field: "connect_apps",
        label: STRINGS.connectAppsLabel,
        apps: ["HubSpot"],
      },
    ]);
  });

  it("throws when the resolved agent id is missing", async () => {
    resolveAgentForTemplateMock.mockResolvedValueOnce({
      agent: { id: null, name: "Leo" },
      created: true,
    });

    await expect(
      hireAgentFromTemplate("sales-email-lead-response-agent", LEO_PERSONA, STRINGS)
    ).rejects.toThrow("Malformed resolve response");
    expect(apiRequestMock).not.toHaveBeenCalled();
  });

  it("throws on a non-ok task/create response", async () => {
    resolveAgentForTemplateMock.mockResolvedValueOnce({
      agent: { id: 42, name: "Leo" },
      created: false,
    });
    apiRequestMock.mockResolvedValueOnce(jsonResponse({ detail: "boom" }, { status: 500 }));

    await expect(
      hireAgentFromTemplate("sales-email-lead-response-agent", LEO_PERSONA, STRINGS)
    ).rejects.toThrow("Failed to create task for hired agent (500)");
  });

  it("throws on a malformed task/create response body", async () => {
    resolveAgentForTemplateMock.mockResolvedValueOnce({
      agent: { id: 42, name: "Leo" },
      created: false,
    });
    apiRequestMock.mockResolvedValueOnce(jsonResponse({}));

    await expect(
      hireAgentFromTemplate("sales-email-lead-response-agent", LEO_PERSONA, STRINGS)
    ).rejects.toThrow("Malformed task create response");
  });
});

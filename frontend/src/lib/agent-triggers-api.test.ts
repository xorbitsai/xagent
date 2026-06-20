import { beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", () => ({
  getApiUrl: () => "http://api.local",
}))

import {
  createAgentTrigger,
  listAgentTriggerRuns,
  updateAgentTrigger,
} from "./agent-triggers-api"

function jsonResponse(body: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  })
}

describe("agent trigger API client", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
  })

  it("creates webhook triggers with the expected endpoint and payload", async () => {
    apiRequestMock.mockResolvedValue(
      jsonResponse({
        id: 7,
        user_id: 1,
        agent_id: 42,
        type: "webhook",
        name: "CRM webhook",
        enabled: true,
        config: {},
        prompt_template: "Handle {{payload}}",
        webhook_token: "token-123",
        webhook_secret: "secret-123",
        next_run_at: null,
        last_run_at: null,
        last_error: null,
        created_at: null,
        updated_at: null,
      }),
    )

    const result = await createAgentTrigger(42, {
      type: "webhook",
      name: "CRM webhook",
      enabled: true,
      config: {},
      prompt_template: "Handle {{payload}}",
      secret: null,
    })

    expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/agents/42/triggers", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        type: "webhook",
        name: "CRM webhook",
        enabled: true,
        config: {},
        prompt_template: "Handle {{payload}}",
        secret: null,
      }),
    })
    expect(result.webhook_secret).toBe("secret-123")
  })

  it("updates triggers with PATCH and surfaces backend validation details", async () => {
    apiRequestMock.mockResolvedValue(
      jsonResponse({ detail: "Trigger name must not be empty" }, { status: 400 }),
    )

    await expect(updateAgentTrigger(42, 7, { name: "   " })).rejects.toThrow(
      "Trigger name must not be empty",
    )

    expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/agents/42/triggers/7", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: "   " }),
    })
  })

  it("loads trigger runs from the agent-scoped route", async () => {
    apiRequestMock.mockResolvedValue(
      jsonResponse([
        {
          id: 12,
          trigger_id: 7,
          task_id: 99,
          background_job_id: null,
          status: "completed",
          source_event_id: "event-1",
          payload_snapshot: {},
          idempotency_key: "key",
          error_message: null,
          started_at: null,
          finished_at: null,
          created_at: null,
          updated_at: null,
        },
      ]),
    )

    const runs = await listAgentTriggerRuns(42, 7)

    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/agents/42/triggers/7/runs",
    )
    expect(runs).toHaveLength(1)
    expect(runs[0].task_id).toBe(99)
  })
})

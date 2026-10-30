import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { cancelTaskAutoResume } from "./task-auto-resume-api"

describe("cancelTaskAutoResume", () => {
  const fetchMock = vi.fn()

  beforeEach(() => {
    localStorage.clear()
    vi.stubGlobal("fetch", fetchMock)
  })
  afterEach(() => {
    fetchMock.mockReset()
    vi.unstubAllGlobals()
  })

  it("posts the run fence and parses the authoritative snapshot", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({
      task_id: 42,
      outcome: "cancelled",
      status: "paused",
      control_state: "paused",
      run_id: "run-1",
      state_version: 5,
      auto_recovery: {
        auto_resume: { status: "stopped", stop_reason: "cancelled" },
      },
      notice: null,
    }), { status: 200 }))

    await expect(cancelTaskAutoResume(42, "run-1")).resolves.toEqual({
      ok: true,
      snapshot: {
        taskId: 42,
        outcome: "cancelled",
        status: "paused",
        controlState: "paused",
        runId: "run-1",
        stateVersion: 5,
        autoRecovery: {
          auto_resume: { status: "stopped", stop_reason: "cancelled" },
        },
        notice: null,
      },
    })
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringMatching(/\/api\/chat\/task\/42\/auto-resume\/cancel$/),
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ run_id: "run-1" }),
      }),
    )
  })

  it("returns the server error code without interpreting its retryability", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({
      detail: { code: "auto_resume_not_cancellable" },
    }), { status: 409 }))

    await expect(cancelTaskAutoResume(42, "run-1")).resolves.toEqual({
      ok: false,
      failure: { kind: "http", status: 409, code: "auto_resume_not_cancellable" },
    })
  })

  it("rejects a malformed success response", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ outcome: "cancelled" }), {
      status: 200,
    }))

    await expect(cancelTaskAutoResume(42, "run-1")).resolves.toEqual({
      ok: false,
      failure: { kind: "malformed_response" },
    })
  })

  it("turns a network rejection into a transport failure", async () => {
    fetchMock.mockRejectedValue(new TypeError("offline"))

    await expect(cancelTaskAutoResume(42, "run-1")).resolves.toEqual({
      ok: false,
      failure: { kind: "transport" },
    })
  })
})

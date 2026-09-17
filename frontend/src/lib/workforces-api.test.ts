import { beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const resolveTimezoneMock = vi.hoisted(() => vi.fn<() => string | undefined>())

vi.mock("@/hooks/use-websocket", () => ({
  resolveReportedTimezone: resolveTimezoneMock,
}))

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper",
  )
  return {
    ...actual,
    apiRequest: apiRequestMock,
  }
})

vi.mock("@/lib/utils", () => ({
  getApiUrl: () => "http://api.local",
}))

import {
  archiveWorkforce,
  createWorkforce,
  deleteWorkforcePermanently,
  discardWorkforce,
  getWorkforceAgentExecution,
  listAgentOptions,
  listWorkforces,
  runWorkforce,
  runWorkforcePreview,
  unarchiveWorkforce,
} from "./workforces-api"

function jsonResponse(data: unknown, init?: ResponseInit) {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  })
}

describe("workforces-api", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    resolveTimezoneMock.mockReset()
    resolveTimezoneMock.mockReturnValue(undefined)
  })

  it("uses the PR5 list pagination and visibility contract", async () => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse({ items: [], total: 0, page: 2, size: 10, pages: 0 }),
    )

    const result = await listWorkforces({
      page: 2,
      size: 10,
      search: "launch",
      status: "active",
    })

    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/workforces?page=2&size=10&search=launch&status=active",
    )
    expect(result).toEqual({ items: [], total: 0, page: 2, size: 10, pages: 0 })
  })

  it("creates a draft workforce without sending unsupported status fields", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse({ id: 42, name: "Launch" }))

    await createWorkforce({
      name: "Launch",
      manager_agent_id: 7,
      workers: [
        {
          source_type: "existing",
          agent_id: 8,
          assignment_instructions: "Research competitors",
          sort_order: 1,
        },
      ],
    })

    const [, options] = apiRequestMock.mock.calls[0]
    expect(apiRequestMock.mock.calls[0][0]).toBe("http://api.local/api/workforces")
    expect(options.method).toBe("POST")
    expect(JSON.parse(String(options.body))).toEqual({
      name: "Launch",
      manager_agent_id: 7,
      workers: [
        {
          source_type: "existing",
          agent_id: 8,
          assignment_instructions: "Research competitors",
          sort_order: 1,
        },
      ],
    })
    expect(JSON.parse(String(options.body))).not.toHaveProperty("status")
    expect(JSON.parse(String(options.body))).not.toHaveProperty(
      "manager_instructions",
    )
  })

  it("loads workforce-selectable agents from the workforce options endpoint", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse([]))

    await expect(listAgentOptions()).resolves.toEqual([])

    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/workforces/agent-options",
    )
  })

  it("runs a workforce with the run payload shape", async () => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse({
        workforce_run_id: 9,
        task_id: 10,
        status: "running",
        redirect_url: "/task/10",
      }),
    )

    const result = await runWorkforce(5, {
      message: "Prepare the launch brief",
      files: ["file-1"],
      execution_mode: "react",
    })

    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/workforces/5/runs",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          message: "Prepare the launch brief",
          files: ["file-1"],
          execution_mode: "react",
        }),
      }),
    )
    expect(result.redirect_url).toBe("/task/10")
  })

  it("attaches the reported timezone to a workforce run opener", async () => {
    resolveTimezoneMock.mockReturnValue("Australia/Melbourne")
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse({
        workforce_run_id: 9,
        task_id: 10,
        status: "running",
        redirect_url: "/task/10",
      }),
    )

    await runWorkforce(5, { message: "go", files: ["file-1"] })

    const body = JSON.parse(apiRequestMock.mock.calls[0][1].body)
    expect(body.timezone).toBe("Australia/Melbourne")
  })

  it("omits the timezone from a workforce run opener when none resolves", async () => {
    resolveTimezoneMock.mockReturnValue(undefined)
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse({
        workforce_run_id: 9,
        task_id: 10,
        status: "running",
        redirect_url: "/task/10",
      }),
    )

    await runWorkforce(5, { message: "go" })

    const body = JSON.parse(apiRequestMock.mock.calls[0][1].body)
    expect("timezone" in body).toBe(false)
  })

  it("lets an explicit timezone on the payload win over the resolver", async () => {
    resolveTimezoneMock.mockReturnValue("Australia/Melbourne")
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse({
        workforce_run_id: 9,
        task_id: 10,
        status: "running",
        redirect_url: "/task/10",
      }),
    )

    await runWorkforce(5, { message: "go", timezone: "Asia/Kolkata" })

    const body = JSON.parse(apiRequestMock.mock.calls[0][1].body)
    expect(body.timezone).toBe("Asia/Kolkata")
  })

  it("attaches the reported timezone to a workforce preview opener", async () => {
    resolveTimezoneMock.mockReturnValue("Australia/Melbourne")
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse({
        workforce_run_id: 9,
        task_id: 10,
        status: "running",
        redirect_url: "/task/10",
      }),
    )

    await runWorkforcePreview({
      manager_agent_id: 1,
      workers: [{ agent_id: 2, assignment_instructions: "do it" }],
      message: "go",
    })

    const body = JSON.parse(apiRequestMock.mock.calls[0][1].body)
    expect(body.timezone).toBe("Australia/Melbourne")
  })

  it("loads one delegated Agent execution on demand", async () => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse({
        task_id: 760,
        worker_task_id: "agent_17 run",
        status: "completed",
        trace_events: [],
      }),
    )

    await getWorkforceAgentExecution(5, 760, "agent_17 run")

    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/workforces/5/runs/760/agent-executions/agent_17%20run",
    )
  })

  it("surfaces backend detail strings for archived edit boundaries", async () => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse(
        { detail: "Archived workforce cannot be edited" },
        { status: 409 },
      ),
    )

    await expect(archiveWorkforce(5)).rejects.toThrow(
      "Archived workforce cannot be edited",
    )
  })

  it("restores an archived workforce to draft through the unarchive endpoint", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse({ id: 5, status: "draft" }))

    const result = await unarchiveWorkforce(5)

    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/workforces/5/unarchive",
      { method: "POST" },
    )
    expect(result.status).toBe("draft")
  })

  it("surfaces backend detail strings when unarchive is rejected", async () => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse(
        { detail: "Only archived workforces can be unarchived" },
        { status: 409 },
      ),
    )

    await expect(unarchiveWorkforce(5)).rejects.toThrow(
      "Only archived workforces can be unarchived",
    )
  })

  it("extracts detail.message from the structured error shape when unarchive fails", async () => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse(
        { detail: { code: "workforce_unarchive_failed", message: "Failed to unarchive workforce" } },
        { status: 500 },
      ),
    )

    let caught: unknown
    try {
      await unarchiveWorkforce(5)
    } catch (error) {
      caught = error
    }

    expect(caught).toBeInstanceOf(Error)
    expect((caught as Error).message).toBe("Failed to unarchive workforce")
  })

  it("permanently deletes a workforce with the ?permanent=true DELETE verb", async () => {
    apiRequestMock.mockResolvedValueOnce(new Response(null, { status: 200 }))

    await expect(deleteWorkforcePermanently(5)).resolves.toBeUndefined()

    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/workforces/5?permanent=true",
      { method: "DELETE" },
    )
  })

  it("extracts detail.message from the structured error shape when permanent delete fails", async () => {
    // toThrow(string) does a substring match, which would also pass if the
    // whole raw JSON body leaked through unextracted -- assert the exact
    // message instead to actually prove extraction.
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse(
        { detail: { code: "workforce_delete_failed", message: "Failed to delete workforce" } },
        { status: 500 },
      ),
    )

    let caught: unknown
    try {
      await deleteWorkforcePermanently(5)
    } catch (error) {
      caught = error
    }

    expect(caught).toBeInstanceOf(Error)
    expect((caught as Error).message).toBe("Failed to delete workforce")
  })

  it("discards an eligible draft through the dedicated endpoint", async () => {
    apiRequestMock.mockResolvedValueOnce(new Response(null, { status: 204 }))

    await expect(discardWorkforce(5, "Failed to discard draft")).resolves.toBeUndefined()

    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/workforces/5/discard",
      { method: "POST" },
    )
  })

  it.each([
    "workforce_not_discardable",
    "workforce_has_runs",
  ])("returns a typed discard error for stable code %s", async (code) => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse(
        { detail: { code, message: "Backend English" } },
        { status: 409 },
      ),
    )

    let caught: unknown
    try {
      await discardWorkforce(5, "Localized fallback")
    } catch (error) {
      caught = error
    }

    expect(caught).toBeInstanceOf(Error)
    expect(caught).toMatchObject({
      name: "WorkforceDiscardError",
      code,
      message: "Localized fallback",
    })
  })

  it.each([
    { detail: { message: "This draft already has runs" } },
    { detail: { code: "unknown_discard_error", message: "Backend English" } },
  ])("uses the localized fallback for malformed structured discard errors", async (payload) => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse(payload, { status: 409 }),
    )

    await expect(discardWorkforce(5, "Failed to discard draft")).rejects.toThrow(
      "Failed to discard draft",
    )
  })

  it("keeps unexpected structured discard failures localized", async () => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse(
        {
          detail: {
            code: "workforce_discard_failed",
            message: "Backend English",
          },
        },
        { status: 500 },
      ),
    )

    let caught: unknown
    try {
      await discardWorkforce(5, "Localized fallback")
    } catch (error) {
      caught = error
    }

    expect(caught).toBeInstanceOf(Error)
    expect((caught as Error).message).toBe("Localized fallback")
    expect((caught as Error).message).not.toContain("Backend English")
  })

  it("uses the localized fallback when discard cannot reach the API", async () => {
    apiRequestMock.mockRejectedValueOnce(new Error("Network connection failed"))

    await expect(discardWorkforce(5, "Failed to discard draft")).rejects.toThrow(
      "Failed to discard draft",
    )
  })
})

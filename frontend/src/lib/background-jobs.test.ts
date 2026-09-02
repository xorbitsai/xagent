import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper",
  )
  return { ...actual, apiRequest: apiRequestMock }
})

import { waitForBackgroundJob, type BackgroundJobResponse } from "@/lib/background-jobs"

function job(status: string): BackgroundJobResponse {
  return {
    id: "job-1",
    user_id: 1,
    job_type: "kb_ingest_web",
    queue: "default",
    status,
    attempts: 1,
    max_attempts: 3,
  }
}

function ok(status: string) {
  return { ok: true, json: async () => job(status) } as unknown as Response
}

const badGateway = { ok: false, status: 502, json: async () => ({}) } as unknown as Response

// The loop's own pacing and a poll's duration both spend the same budget, so the clock
// only advances where the code under test would really wait.
let clock = 0
function elapsedSeconds() {
  return clock / 1000
}

beforeEach(() => {
  apiRequestMock.mockReset()
  clock = 0
  vi.spyOn(Date, "now").mockImplementation(() => clock)
  const realSetTimeout = window.setTimeout
  vi.spyOn(window, "setTimeout").mockImplementation(((fn: () => void, ms?: number) => {
    clock += ms ?? 0
    return realSetTimeout(fn, 0)
  }) as unknown as typeof window.setTimeout)
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe("waitForBackgroundJob", () => {
  it("rides out a burst of instant failures and returns the terminal job", async () => {
    for (let i = 0; i < 9; i++) apiRequestMock.mockResolvedValueOnce(badGateway)
    apiRequestMock.mockResolvedValueOnce(ok("succeeded"))

    await expect(waitForBackgroundJob("http://api.local", job("running"))).resolves.toMatchObject({
      status: "succeeded",
    })
  })

  it("rejects once instant failures outlast the window, not before", async () => {
    apiRequestMock.mockResolvedValue(badGateway)

    await expect(waitForBackgroundJob("http://api.local", job("running"))).rejects.toThrow(
      "Failed to fetch background job job-1",
    )
    // First failure at 1s opens a window closing at 11s: the rejection lands on that
    // boundary exactly, so pin it from both sides.
    expect(elapsedSeconds()).toBe(11)
  })

  it("rejects a slow auth-refresh failure without waiting for ten of them", async () => {
    // A 401 whose refresh times out burns AUTH_REFRESH_TIMEOUT_MS before returning.
    apiRequestMock.mockImplementation(async () => {
      clock += 15_000
      return { ok: false, status: 401, json: async () => ({}) } as unknown as Response
    })

    await expect(waitForBackgroundJob("http://api.local", job("running"))).rejects.toThrow(
      "Failed to fetch background job job-1",
    )
    expect(apiRequestMock).toHaveBeenCalledTimes(2)
    expect(elapsedSeconds()).toBeLessThan(60)
  })

  it("reopens the full window after any successful poll", async () => {
    for (let i = 0; i < 9; i++) apiRequestMock.mockResolvedValueOnce(badGateway)
    apiRequestMock.mockResolvedValueOnce(ok("running"))
    for (let i = 0; i < 9; i++) apiRequestMock.mockResolvedValueOnce(badGateway)
    apiRequestMock.mockResolvedValueOnce(ok("succeeded"))

    await expect(waitForBackgroundJob("http://api.local", job("running"))).resolves.toMatchObject({
      status: "succeeded",
    })
    expect(apiRequestMock).toHaveBeenCalledTimes(20)
  })

  it("keeps rejecting a malformed payload immediately", async () => {
    apiRequestMock.mockResolvedValue({ ok: true, json: async () => ({ nope: true }) })

    await expect(waitForBackgroundJob("http://api.local", job("running"))).rejects.toThrow(
      "Invalid background job response",
    )
    expect(apiRequestMock).toHaveBeenCalledTimes(1)
  })
})

import { describe, expect, it } from "vitest"

import { normalizeTaskCompletedMessage } from "./task-completion"

describe("task-completion", () => {
  it("normalizes legacy nested data payloads", () => {
    const payload = normalizeTaskCompletedMessage({
      type: "task_completed",
      data: {
        success: true,
        result: "done",
        file_outputs: [{ file_id: "file-1", filename: "out.xlsx" }],
      },
    })

    expect(payload).toEqual({
      success: true,
      status: "completed",
      task: undefined,
      result: "done",
      output: undefined,
      fileOutputs: [{ file_id: "file-1", filename: "out.xlsx" }],
      chatResponse: undefined,
      metadata: undefined,
    })
  })

  it("normalizes top-level websocket completion payloads", () => {
    const payload = normalizeTaskCompletedMessage({
      type: "task_completed",
      task: {
        id: 3,
        status: "failed",
      },
      success: false,
      result: "failed",
      file_outputs: [],
    })

    expect(payload.status).toBe("failed")
    expect(payload.success).toBe(false)
    expect(payload.task?.id).toBe(3)
  })

  it("accepts enum-style uppercase terminal statuses", () => {
    const payload = normalizeTaskCompletedMessage({
      type: "task_completed",
      task: {
        status: "FAILED",
      },
    })

    expect(payload.status).toBe("failed")
    expect(payload.success).toBe(false)
  })

  it("falls back to task status when success is omitted", () => {
    const payload = normalizeTaskCompletedMessage({
      type: "task_completed",
      task: {
        status: "completed",
      },
    })

    expect(payload.status).toBe("completed")
    expect(payload.success).toBe(true)
  })

  it.each([null, "invalid", ["invalid"]])(
    "rejects malformed error details: %j",
    (errorDetails) => {
      const payload = normalizeTaskCompletedMessage({
        type: "task_completed",
        success: false,
        error_details: errorDetails,
      })

      expect(payload.errorDetails).toBeUndefined()
    }
  )

  it("normalizes a coded failure's error_code and plain-object error_details", () => {
    const payload = normalizeTaskCompletedMessage({
      type: "task_completed",
      task: { status: "failed" },
      success: false,
      output: "quota reason",
      error_code: "quota_exceeded",
      error_details: { code: "quota_exceeded", limit: 0, plan: "free" },
    })

    expect(payload.status).toBe("failed")
    expect(payload.errorCode).toBe("quota_exceeded")
    expect(payload.errorDetails).toEqual({
      code: "quota_exceeded",
      limit: 0,
      plan: "free",
    })
  })
})

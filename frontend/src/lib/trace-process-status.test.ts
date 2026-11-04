import { describe, expect, it } from "vitest"

import {
  normalizeTraceProcessStatus,
  resolveTraceProcessStatus,
} from "./trace-process-status"

describe("trace process status", () => {
  it("normalizes task status values", () => {
    expect(normalizeTraceProcessStatus("FAILED")).toBe("failed")
    expect(normalizeTraceProcessStatus("waiting_for_user")).toBe("waiting_for_user")
    expect(normalizeTraceProcessStatus("unknown")).toBeUndefined()
  })

  it("infers a failed process from terminal trace errors", () => {
    expect(
      resolveTraceProcessStatus({
        traceEvents: [
          {
            event_type: "react_task_start",
            data: {},
          },
          {
            event_type: "llm_call_start",
            data: {},
          },
          {
            event_type: "trace_error",
            data: {
              error_type: "agent_error",
              status: "failed",
            },
          },
        ],
      })
    ).toBe("failed")
  })

  it("infers a failed process from step-local trace errors without a status field", () => {
    expect(
      resolveTraceProcessStatus({
        traceEvents: [
          {
            event_type: "react_task_start",
            data: {},
          },
          {
            event_type: "llm_call_start",
            data: {},
          },
          {
            event_type: "trace_error",
            data: {
              error_type: "agent_pattern_error",
              error_message: "OpenAI bad request",
            },
          },
        ],
      })
    ).toBe("failed")
  })

  it("prefers explicit process status over trace inference", () => {
    expect(
      resolveTraceProcessStatus({
        processStatus: "completed",
        traceEvents: [
          {
            event_type: "trace_error",
            data: {
              error_type: "agent_error",
              status: "failed",
            },
          },
        ],
      })
    ).toBe("completed")
  })

  it("infers a completed process from react_task_end events", () => {
    expect(
      resolveTraceProcessStatus({
        traceEvents: [
          {
            event_type: "react_task_start",
            data: {},
          },
          {
            event_type: "react_task_end",
            data: {},
          },
        ],
      })
    ).toBe("completed")
  })

  it("prefers terminal trace status over a still-running task status", () => {
    expect(
      resolveTraceProcessStatus({
        processStatus: "running",
        traceEvents: [
          {
            event_type: "react_task_start",
            data: {},
          },
          {
            event_type: "react_task_end",
            data: {},
          },
        ],
      })
    ).toBe("completed")
  })

  it("prefers explicit failed status over inferred completed trace status", () => {
    expect(
      resolveTraceProcessStatus({
        processStatus: "failed",
        traceEvents: [
          {
            event_type: "react_task_start",
            data: {},
          },
          {
            event_type: "react_task_end",
            data: {},
          },
        ],
      })
    ).toBe("failed")
  })

  describe("an inferred interrupted end event", () => {
    const interruptedRun = [
      { event_type: "react_task_start", data: {} },
      { event_type: "llm_call_start", data: {} },
      { event_type: "react_task_end", data: { status: "interrupted" } },
      { event_type: "react_task_start", data: {} },
      { event_type: "tool_execution_start", data: {} },
    ]

    it.each(["running", "pending"])(
      "keeps an explicit %s status over it",
      (processStatus) => {
        expect(
          resolveTraceProcessStatus({ processStatus, traceEvents: interruptedRun })
        ).toBe(processStatus)
      }
    )

    it("is used when no status is given", () => {
      expect(resolveTraceProcessStatus({ traceEvents: interruptedRun })).toBe("interrupted")
    })

    it.each([
      ["failed", [{ event_type: "agent_error", data: {} }]],
      ["completed", [{ event_type: "react_task_end", data: {} }]],
      ["paused", [{ event_type: "react_task_end", data: { status: "paused" } }]],
      [
        "waiting_for_user",
        [{ event_type: "react_task_end", data: { status: "waiting_for_user" } }],
      ],
    ])("still lets an inferred %s status override an explicit running status", (inferred, traceEvents) => {
      expect(resolveTraceProcessStatus({ processStatus: "running", traceEvents })).toBe(inferred)
    })
  })
})

/**
 * Inc.7 (frontend) — tool-event attribution by tool_call_id.
 *
 * With in-turn tool concurrency, a single step can have several same-named tool
 * actions in flight. The processor must attribute each tool_execution_end /
 * _failed to the action with the matching tool_call_id, not to the
 * "last running tool" (which mis-attributes output/status the moment completion
 * order differs from reverse-start order).
 */
import { describe, it, expect, vi } from "vitest"

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }))
vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))
vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({ openFilePreview: vi.fn(), dispatch: vi.fn() }),
}))
vi.mock("@/lib/api-wrapper", () => ({ apiRequest: vi.fn() }))
// These pull in heavy/optional deps (e.g. pptxviewjs) not needed for the
// pure reducer under test.
vi.mock("@/components/file/docx-preview-renderer", () => ({
  DocxPreviewRenderer: () => null,
}))
vi.mock("@/components/file/excel-preview-renderer", () => ({
  ExcelPreviewRenderer: () => null,
}))
vi.mock("@/components/file/pptx-preview-renderer", () => ({
  PptxPreviewRenderer: () => null,
}))

import { processTraceEvents } from "./TraceEventRenderer"

const t = (key: string, vars?: Record<string, string | number>) =>
  vars?.tool ? `${key}:${vars.tool}` : key

const ev = (event_type: string, data: Record<string, unknown>) => ({
  event_type,
  step_id: "step-1",
  data,
})

const stepStart = ev("dag_step_start", { step_name: "Search" })

describe("processTraceEvents tool_call_id attribution", () => {
  it("normalizes malformed trace data before nested access", () => {
    const malformedEvents = [
      {
        event_type: "workforce_delegation_start",
        step_id: "worker-null",
        timestamp: 1,
        data: null,
      },
      {
        event_type: "workforce_delegation_end",
        step_id: "worker-string",
        timestamp: 2,
        data: "not-an-object",
      },
      {
        event_type: "tool_execution_start",
        step_id: "worker-array",
        timestamp: 3,
        data: [],
      },
    ]

    expect(() => processTraceEvents(malformedEvents as never, t)).not.toThrow()
  })

  it("attributes concurrent same-name tool results by tool_call_id", () => {
    const events = [
      stepStart,
      ev("tool_execution_start", {
        tool_name: "web_search",
        tool_call_id: "A",
        tool_args: { query: "a" },
      }),
      ev("tool_execution_start", {
        tool_name: "web_search",
        tool_call_id: "B",
        tool_args: { query: "b" },
      }),
      // First-started finishes first: this is the order that breaks LIFO.
      ev("tool_execution_end", {
        tool_name: "web_search",
        tool_call_id: "A",
        result: { output: "RESULT_A" },
      }),
      ev("tool_execution_end", {
        tool_name: "web_search",
        tool_call_id: "B",
        result: { output: "RESULT_B" },
      }),
    ]

    const steps = processTraceEvents(events as never, t)
    const toolActions = steps[0].actions.filter((a) => a.type === "tool")
    const a = toolActions.find((x) => x.data.tool_call_id === "A")
    const b = toolActions.find((x) => x.data.tool_call_id === "B")

    expect(a?.data.output).toBe("RESULT_A")
    expect(b?.data.output).toBe("RESULT_B")
    expect(a?.status).toBe("completed")
    expect(b?.status).toBe("completed")
  })

  it("attributes a concurrent tool failure by tool_call_id", () => {
    const events = [
      stepStart,
      ev("tool_execution_start", {
        tool_name: "web_search",
        tool_call_id: "A",
        tool_args: { query: "a" },
      }),
      ev("tool_execution_start", {
        tool_name: "web_search",
        tool_call_id: "B",
        tool_args: { query: "b" },
      }),
      ev("tool_execution_failed", {
        tool_name: "web_search",
        tool_call_id: "A",
        error: "boom-A",
      }),
      ev("tool_execution_end", {
        tool_name: "web_search",
        tool_call_id: "B",
        result: { output: "RESULT_B" },
      }),
    ]

    const steps = processTraceEvents(events as never, t)
    const toolActions = steps[0].actions.filter((a) => a.type === "tool")
    const a = toolActions.find((x) => x.data.tool_call_id === "A")
    const b = toolActions.find((x) => x.data.tool_call_id === "B")

    expect(a?.status).toBe("failed")
    expect(a?.data.error).toBe("boom-A")
    expect(b?.status).toBe("completed")
    expect(b?.data.output).toBe("RESULT_B")
  })

  it("updates step.output for sequential tools within one step", () => {
    // Two tools run one-after-another (never overlapping). step.output must
    // track the latest tool's output, exactly as it did before concurrency was
    // introduced — counting total tool actions would wrongly freeze it.
    const events = [
      stepStart,
      ev("tool_execution_start", {
        tool_name: "calculator",
        tool_call_id: "A",
        tool_args: { expression: "1+1" },
      }),
      ev("tool_execution_end", {
        tool_name: "calculator",
        tool_call_id: "A",
        result: { output: "RESULT_A" },
      }),
      ev("tool_execution_start", {
        tool_name: "calculator",
        tool_call_id: "B",
        tool_args: { expression: "2+2" },
      }),
      ev("tool_execution_end", {
        tool_name: "calculator",
        tool_call_id: "B",
        result: { output: "RESULT_B" },
      }),
    ]

    const steps = processTraceEvents(events as never, t)
    expect(steps[0].output).toBe("RESULT_B")
  })

  it("does not clobber step.output when tools run concurrently", () => {
    // Both tools are in flight at once, so the step-level scalar is ambiguous;
    // the processor leaves it unset and the per-action outputs carry the data.
    const events = [
      stepStart,
      ev("tool_execution_start", {
        tool_name: "web_search",
        tool_call_id: "A",
        tool_args: { query: "a" },
      }),
      ev("tool_execution_start", {
        tool_name: "web_search",
        tool_call_id: "B",
        tool_args: { query: "b" },
      }),
      ev("tool_execution_end", {
        tool_name: "web_search",
        tool_call_id: "A",
        result: { output: "RESULT_A" },
      }),
      ev("tool_execution_end", {
        tool_name: "web_search",
        tool_call_id: "B",
        result: { output: "RESULT_B" },
      }),
    ]

    const steps = processTraceEvents(events as never, t)
    // step.output is left at its initial value, not clobbered by whichever
    // concurrent tool happened to finish last.
    expect(steps[0].output).not.toBe("RESULT_A")
    expect(steps[0].output).not.toBe("RESULT_B")
    const toolActions = steps[0].actions.filter((a) => a.type === "tool")
    expect(toolActions.find((x) => x.data.tool_call_id === "A")?.data.output).toBe(
      "RESULT_A"
    )
    expect(toolActions.find((x) => x.data.tool_call_id === "B")?.data.output).toBe(
      "RESULT_B"
    )
  })

  it("falls back to last running tool when tool_call_id is absent (legacy)", () => {
    const events = [
      stepStart,
      ev("tool_execution_start", {
        tool_name: "calculator",
        tool_args: { expression: "1+1" },
      }),
      ev("tool_execution_end", {
        tool_name: "calculator",
        result: { output: "2" },
      }),
    ]

    const steps = processTraceEvents(events as never, t)
    const toolActions = steps[0].actions.filter((a) => a.type === "tool")
    expect(toolActions).toHaveLength(1)
    expect(toolActions[0].status).toBe("completed")
    expect(toolActions[0].data.output).toBe("2")
  })
})

/**
 * Inc.7 (frontend) — tool-event attribution by tool_call_id.
 *
 * With in-turn tool concurrency, a single step can have several same-named tool
 * actions in flight. The processor must attribute each tool_execution_end /
 * _failed to the action with the matching tool_call_id, not to the
 * "last running tool" (which mis-attributes output/status the moment completion
 * order differs from reverse-start order).
 */
import { readFileSync } from "node:fs"
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

import {
  getFriendlyToolName,
  getProgressNarrationText,
  getRawToolName,
  processTraceEvents,
} from "./TraceEventRenderer"
import { resolveDynamicTranslation, type Locale } from "@/i18n/translations"
import type { TranslateDynamic } from "@/contexts/i18n-context"
import en from "@/i18n/locales/en"
import zh from "@/i18n/locales/zh"

const t = (key: string, vars?: Record<string, string | number>) =>
  vars?.tool ? `${key}:${vars.tool}` : key

const ev = (event_type: string, data: Record<string, unknown>) => ({
  event_type,
  step_id: "step-1",
  data,
})

const stepStart = ev("dag_step_start", { step_name: "Search" })

describe("processTraceEvents DAG step outcomes", () => {
  it.each([undefined, "failed", "completed"])(
    "keeps a failed DAG step separate from successful siblings when task status is %s",
    (taskStatus) => {
      const steps = processTraceEvents([
        stepStart,
        ev("dag_step_end", { status: "completed", result: "Verified output" }),
        { ...ev("dag_step_start", { step_name: "Verify remaining data" }), step_id: "verify" },
        {
          ...ev("dag_step_end", { status: "failed", error: "Iteration limit reached" }),
          step_id: "verify",
        },
      ], t, taskStatus)

      expect(steps[0]).toMatchObject({ status: "completed", output: "Verified output" })
      expect(steps[1]).toMatchObject({ status: "failed", output: "" })
      expect(steps[1].actions).toEqual([
        expect.objectContaining({
          type: "error",
          status: "failed",
          data: { error: "Iteration limit reached" },
        }),
      ])
    },
  )

  it.each(["dag_step_end", "dag_step_failed"])("ends unfinished actions on %s without changing completed work", (eventType) => {
    const [step] = processTraceEvents([
      stepStart,
      ev("tool_execution_start", { tool_name: "read_file", tool_call_id: "done" }),
      ev("tool_execution_end", { tool_name: "read_file", tool_call_id: "done", result: "Verified rows" }),
      ev("tool_execution_start", { tool_name: "web_search", tool_call_id: "pending-1" }),
      ev("tool_execution_start", { tool_name: "web_search", tool_call_id: "pending-2" }),
      ev(eventType, { status: "failed", error: "Step stopped" }),
    ], t, "completed")

    expect(step.status).toBe("failed")
    expect(step.actions[0]).toMatchObject({ status: "completed", data: { output: "Verified rows" } })
    expect(step.actions.slice(1)).toHaveLength(2)
    step.actions.slice(1).forEach((action) => {
      expect(action).toMatchObject({ status: "failed", data: { error: "Step stopped" } })
    })
  })

  it("uses the existing fallback for failed DAG steps without an error message", () => {
    const [step] = processTraceEvents([
      stepStart,
      ev("dag_step_end", { status: "failed" }),
    ], t)

    expect(step.status).toBe("failed")
    expect(step.actions[0].data.error).toBe("traceEventRenderer.unknownError")
  })

  it.each(["completed", undefined])("preserves successful and legacy DAG ends with status %s", (status) => {
    const [step] = processTraceEvents([
      stepStart,
      ev("dag_step_end", { ...(status ? { status } : {}), result: "Verified output" }),
    ], t)

    expect(step).toMatchObject({ status: "completed", output: "Verified output" })
    expect(step.actions[0]).toMatchObject({ type: "info", status: "completed" })
  })
})

// Every status in the final settling pass crossed with every step shape that
// can reach it. A step only holds pending/running/completed/failed before that
// pass, and a pending step never carries a name, so "unfinished" means running.
describe("processTraceEvents settling of stopped processes", () => {
  type Shape = "running" | "running+action" | "completed" | "completed+action" | "failed" | "failed+action"
  type Settled = {
    step: string
    actions: Array<[status: string, error: string | undefined]>
  }

  const stepEnd = (data: Record<string, unknown>) => ev("dag_step_end", data)
  const shapeEvents: Record<Shape, ReturnType<typeof ev>[]> = {
    running: [stepStart],
    "running+action": [stepStart, ev("llm_call_start", { model_name: "m" })],
    completed: [stepStart, stepEnd({ status: "completed" })],
    "completed+action": [
      stepStart,
      stepEnd({ status: "completed" }),
      ev("llm_call_start", { model_name: "m" }),
    ],
    failed: [stepStart, stepEnd({ status: "failed", error: "boom" })],
    "failed+action": [
      stepStart,
      stepEnd({ status: "failed", error: "boom" }),
      ev("llm_call_start", { model_name: "m" }),
    ],
  }

  const unknownError = "traceEventRenderer.unknownError"
  const reported: Settled["actions"][number] = ["failed", "boom"]
  // Known existing behaviour, pinned as it is today: a running action stays
  // spinning under a step that has already settled, under "completed" when the
  // step failed and under "paused" or "waiting_for_user" when the step completed
  // or failed. A later fix that settles these actions should update the
  // `spinning` entries in those rows; that change is not a regression.
  const spinning: Settled["actions"][number] = ["running", undefined]

  const expectations: Record<string, Record<Shape, Settled>> = {
    interrupted: {
      running: { step: "interrupted", actions: [] },
      "running+action": { step: "interrupted", actions: [["interrupted", undefined]] },
      completed: { step: "completed", actions: [] },
      "completed+action": { step: "completed", actions: [["interrupted", undefined]] },
      failed: { step: "failed", actions: [reported] },
      "failed+action": { step: "failed", actions: [reported, ["interrupted", undefined]] },
    },
    failed: {
      running: { step: "failed", actions: [] },
      "running+action": { step: "failed", actions: [["failed", unknownError]] },
      completed: { step: "completed", actions: [] },
      "completed+action": { step: "failed", actions: [["failed", unknownError]] },
      failed: { step: "failed", actions: [reported] },
      "failed+action": { step: "failed", actions: [reported, ["failed", unknownError]] },
    },
    completed: {
      running: { step: "completed", actions: [] },
      "running+action": { step: "completed", actions: [["completed", undefined]] },
      completed: { step: "completed", actions: [] },
      "completed+action": { step: "completed", actions: [["completed", undefined]] },
      failed: { step: "failed", actions: [reported] },
      "failed+action": { step: "failed", actions: [reported, spinning] },
    },
    paused: {
      running: { step: "paused", actions: [] },
      "running+action": { step: "paused", actions: [["completed", undefined]] },
      completed: { step: "completed", actions: [] },
      "completed+action": { step: "completed", actions: [spinning] },
      failed: { step: "failed", actions: [reported] },
      "failed+action": { step: "failed", actions: [reported, spinning] },
    },
    waiting_for_user: {
      running: { step: "waiting_for_user", actions: [] },
      "running+action": { step: "waiting_for_user", actions: [["completed", undefined]] },
      completed: { step: "completed", actions: [] },
      "completed+action": { step: "completed", actions: [spinning] },
      failed: { step: "failed", actions: [reported] },
      "failed+action": { step: "failed", actions: [reported, spinning] },
    },
    // Not stopped statuses: the settling pass must leave the step untouched.
    running: {
      running: { step: "running", actions: [] },
      "running+action": { step: "running", actions: [spinning] },
      completed: { step: "completed", actions: [] },
      "completed+action": { step: "completed", actions: [spinning] },
      failed: { step: "failed", actions: [reported] },
      "failed+action": { step: "failed", actions: [reported, spinning] },
    },
  }

  const cases = Object.entries(expectations).flatMap(([status, byShape]) =>
    (Object.keys(byShape) as Shape[]).map((shape) => [status, shape, byShape[shape]] as const),
  )

  it.each(cases)("settles a %s process with a %s step", (status, shape, expected) => {
    const [step] = processTraceEvents(shapeEvents[shape] as never, t, status)
    expect({
      step: step.status,
      actions: step.actions.map((action) => [action.status, action.data.error]),
    }).toEqual(expected)
  })

  it("settles nothing when the process status is absent", () => {
    const [step] = processTraceEvents(shapeEvents["running+action"] as never, t)
    expect(step.status).toBe("running")
    expect(step.actions.map((action) => action.status)).toEqual(["running"])
  })
})

describe("processTraceEvents exact settlement projection", () => {
  const toolActions = (events: Parameters<typeof processTraceEvents>[0]) =>
    processTraceEvents(events, t).flatMap(step =>
      step.actions.filter(action => action.type === "tool"),
    )
  const event = (
    event_type: string,
    step_id: string,
    data: Record<string, unknown>,
  ) => ({ event_type, step_id, data })
  const startStep = (stepId: string, name = stepId) =>
    event("react_task_start", stepId, { step_name: name })
  const startTool = (
    stepId: string,
    invocationId: string | undefined,
    result?: string,
    settlement = false,
  ) => [
    event("tool_execution_start", stepId, {
      tool_name: "approval_gate",
      tool_call_id: "tool_call_0",
      ...(invocationId ? { invocation_id: invocationId } : {}),
      ...(settlement ? { settlement_delivery: true } : {}),
    }),
    event("tool_execution_end", stepId, {
      tool_name: "approval_gate",
      tool_call_id: "tool_call_0",
      ...(invocationId ? { invocation_id: invocationId } : {}),
      ...(settlement ? { settlement_delivery: true } : {}),
      result: { output: result ?? "waiting" },
    }),
  ]

  it("settles only the exact invocation when one step reuses a provider id", () => {
    const events = [
      startStep("same-step"),
      ...startTool("same-step", "invocation-a", "first"),
      ...startTool("same-step", "invocation-b", "second"),
      ...startTool("same-step", "invocation-b", "settled-second", true),
    ]

    const steps = processTraceEvents(events, t)
    const actions = steps.flatMap(step => step.actions.filter(action => action.type === "tool"))
    expect(actions).toHaveLength(2)
    expect(actions.find(action => action.data.invocation_id === "invocation-a")?.data.output).toBe("first")
    expect(actions.find(action => action.data.invocation_id === "invocation-b")?.data.output).toBe("settled-second")
  })

  it("settles only the exact invocation across steps that reuse a provider id", () => {
    const events = [
      startStep("step-a"),
      ...startTool("step-a", "invocation-a", "first"),
      startStep("step-b"),
      ...startTool("step-b", "invocation-b", "second"),
      ...startTool("fresh-resume-step", "invocation-a", "settled-first", true),
    ]

    const steps = processTraceEvents(events, t)
    const actions = steps.flatMap(step => step.actions.filter(action => action.type === "tool"))
    expect(actions).toHaveLength(2)
    expect(actions.find(action => action.data.invocation_id === "invocation-a")?.data.output).toBe("settled-first")
    expect(actions.find(action => action.data.invocation_id === "invocation-b")?.data.output).toBe("second")
    expect(steps.some(step => step.stepId === "fresh-resume-step")).toBe(false)
  })

  it("does not let an unknown exact identity hijack a legacy candidate", () => {
    const events = [
      startStep("step-a"),
      ...startTool("step-a", undefined, "legacy"),
      ...startTool("step-a", "unknown-invocation", "orphan-settlement", true),
    ]

    const actions = toolActions(events)
    expect(actions).toHaveLength(2)
    expect(actions.find(action => !action.data.invocation_id)?.data.output).toBe("legacy")
    expect(actions.find(action => action.data.invocation_id === "unknown-invocation")?.data.output).toBe("orphan-settlement")
  })

  it("keeps a paired orphan settlement visible without a task-start event", () => {
    const actions = toolActions([
      ...startTool("truncated-step", "orphan-invocation", "terminal", true),
    ])

    expect(actions).toHaveLength(1)
    expect(actions[0]).toMatchObject({
      status: "completed",
      data: {
        invocation_id: "orphan-invocation",
        output: "terminal",
      },
    })
  })

  it("keeps an orphan terminal event visible when settlement START is truncated", () => {
    const steps = processTraceEvents([
      event("tool_execution_end", "truncated-step", {
        tool_name: "approval_gate",
        tool_call_id: "tool_call_0",
        invocation_id: "orphan-invocation",
        settlement_delivery: true,
        result: { output: "terminal" },
      }),
    ] as never, t)

    expect(steps).toHaveLength(1)
    expect(steps[0].stepName).toBe("traceEventRenderer.taskExecution")
    expect(steps[0].actions).toEqual([
      expect.objectContaining({
        status: "completed",
        data: expect.objectContaining({ output: "terminal" }),
      }),
    ])
  })

  it("uses step identity only for ambiguous legacy settlements", () => {
    const events = [
      startStep("step-a"),
      ...startTool("step-a", undefined, "waiting-a"),
      startStep("step-b"),
      ...startTool("step-b", undefined, "waiting-b"),
      ...startTool("step-a", undefined, "settled-a", true),
      ...startTool("step-b", undefined, "settled-b", true),
    ]

    const actions = toolActions(events)
    expect(actions).toHaveLength(2)
    expect(actions.map(action => action.data.output).sort()).toEqual(["settled-a", "settled-b"])
  })

  it("namespaces exact and legacy settlement cache keys", () => {
    const exactId = "tool_call_0::step-a"
    const events = [
      startStep("step-a"),
      ...startTool("step-a", undefined, "legacy-waiting"),
      startStep("step-b"),
      ...startTool("step-b", exactId, "exact-waiting"),
      ...startTool("step-a", undefined, "legacy-settled", true),
      ...startTool("step-b", exactId, "exact-settled", true),
    ]

    const actions = toolActions(events)
    expect(actions).toHaveLength(2)
    expect(actions.find(action => !action.data.invocation_id)?.data.output).toBe("legacy-settled")
    expect(actions.find(action => action.data.invocation_id === exactId)?.data.output).toBe("exact-settled")
  })

  it("projects a failed settlement onto the exact card and clears stale output", () => {
    const events = [
      startStep("step-a"),
      ...startTool("step-a", "invocation-a", "Approve?"),
      event("tool_execution_start", "resume", {
        tool_name: "approval_gate",
        tool_call_id: "tool_call_0",
        invocation_id: "invocation-a",
        settlement_delivery: true,
      }),
      event("tool_execution_failed", "resume", {
        tool_name: "approval_gate",
        tool_call_id: "tool_call_0",
        invocation_id: "invocation-a",
        settlement_delivery: true,
        error: "Rejected",
      }),
    ]

    const actions = toolActions(events)
    expect(actions).toHaveLength(1)
    expect(actions[0]).toMatchObject({ status: "failed", data: { error: "Rejected" } })
    expect(actions[0].data.output).toBeUndefined()
  })
})

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

  it("does not throw when tool_name in the payload is not a string", () => {
    // tool_name's "string" type is a compile-time annotation, not a runtime
    // guarantee: a malformed/legacy backend payload could send a number,
    // object, or null. getFriendlyToolName must not crash on it.
    const malformedToolNameEvents = [
      ev("tool_execution_start", { tool_name: 42 }),
      ev("tool_execution_start", { tool_name: { unexpected: "object" } }),
      ev("tool_execution_start", { tool_name: null }),
    ]

    expect(() => processTraceEvents(malformedToolNameEvents as never, t)).not.toThrow()
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

describe("processTraceEvents unavailable connector placeholder", () => {
  const tc = (key: string, vars?: Record<string, string | number>) =>
    vars?.connector ? `${key}:${vars.connector}` : t(key, vars)
  const failedCall = (toolName: string, result: unknown) =>
    [
      stepStart,
      ev("tool_execution_start", { tool_name: toolName, tool_call_id: "A" }),
      ev("tool_execution_failed", {
        tool_name: toolName,
        tool_call_id: "A",
        error: "MCP server tools could not be loaded.",
        result,
      }),
    ].map((event, index) => ({ ...event, timestamp: index + 1 }))
  const failedToolAction = (toolName: string, title: string) => ({
    id: "event-1",
    type: "tool",
    title: `traceEventRenderer.executeTool:${title}`,
    status: "failed",
    timestamp: 2000,
    data: {
      tool: toolName,
      args: undefined,
      code: "",
      tool_call_id: "A",
      sandboxed: false,
      error: "MCP server tools could not be loaded.",
    },
  })

  it("renders a call carrying unavailable_server as a connector status line", () => {
    const steps = processTraceEvents(
      failedCall("mcp_google_drive_42_unavailable", {
        success: false,
        is_error: true,
        unavailable_server: "Google Drive",
      }) as never,
      tc,
    )

    expect(steps[0].actions).toEqual([
      {
        id: "event-1",
        type: "info",
        title: "traceEventRenderer.connectorUnavailable:Google Drive",
        status: "completed",
        timestamp: 2000,
        data: { statusLine: true },
      },
    ])
  })

  it.each([
    ["a non-empty error", "MCP server credentials are unavailable.", { statusLine: true, error: "MCP server credentials are unavailable." }],
    ["a blank error", "  ", { statusLine: true }],
    ["a non-string error", 42, { statusLine: true }],
  ])("sets the status line detail from result.error given %s", (_, error, data) => {
    const steps = processTraceEvents(
      failedCall("mcp_google_drive_42_unavailable", {
        success: false,
        is_error: true,
        error,
        unavailable_server: "Google Drive",
      }) as never,
      tc,
    )

    expect(steps[0].actions).toEqual([
      expect.objectContaining({ type: "info", data }),
    ])
  })

  it("only converts tool failures into a connector status line", () => {
    const steps = processTraceEvents(
      [
        stepStart,
        ev("dag_step_failed", {
          error: "step failed",
          result: { unavailable_server: "Google Drive" },
        }),
      ].map((event, index) => ({ ...event, timestamp: index + 1 })) as never,
      tc,
    )

    expect(steps[0].actions).toEqual([
      {
        id: "event-1",
        type: "error",
        title: "traceEventRenderer.executionFailed",
        status: "failed",
        timestamp: 2000,
        data: { error: "step failed" },
      },
    ])
  })

  it("keeps an ordinary MCP tool failure as a failed tool call", () => {
    const steps = processTraceEvents(
      failedCall("mcp_notion_search", {
        success: false,
        is_error: true,
        content: [{ text: "boom" }],
      }) as never,
      tc,
    )

    expect(steps[0].actions).toEqual([
      failedToolAction("mcp_notion_search", "Mcp Notion Search"),
    ])
  })

  it.each([
    ["missing", { success: false, is_error: true }],
    ["empty", { success: false, is_error: true, unavailable_server: " " }],
    ["non-string", { success: false, is_error: true, unavailable_server: 42 }],
    ["non-object result", "MCP server tools could not be loaded."],
  ])("falls back to the tool failure when unavailable_server is %s", (_, result) => {
    const steps = processTraceEvents(
      failedCall("mcp_google_drive_42_unavailable", result) as never,
      tc,
    )

    expect(steps[0].actions).toEqual([
      failedToolAction(
        "mcp_google_drive_42_unavailable",
        "Mcp Google Drive 42 Unavailable",
      ),
    ])
  })

  const placeholderFailure = (toolCallId?: string) =>
    ev("tool_execution_failed", {
      tool_name: "mcp_google_drive_42_unavailable",
      ...(toolCallId ? { tool_call_id: toolCallId } : {}),
      error: "MCP server tools could not be loaded.",
      result: { success: false, is_error: true, unavailable_server: "Google Drive" },
    })
  const statusLineAction = (id: string, timestamp: number) => ({
    id,
    type: "info",
    title: "traceEventRenderer.connectorUnavailable:Google Drive",
    status: "completed",
    timestamp,
    data: { statusLine: true },
  })

  const S = "web_search"
  const P = "mcp_google_drive_42_unavailable"
  const start = (tool: string, id?: string) =>
    ev("tool_execution_start", { tool_name: tool, ...(id ? { tool_call_id: id } : {}) })
  const endS = ev("tool_execution_end", {
    tool_name: S,
    tool_call_id: "A",
    result: { output: "RESULT_S" },
  })
  const run = (...events: Array<ReturnType<typeof ev>>) =>
    processTraceEvents(
      [stepStart, ...events, ev("dag_step_end", {})].map((event, index) => ({
        ...event,
        timestamp: index + 1,
      })) as never,
      tc,
    )[0].actions
  const webSearchCards = (actions: ReturnType<typeof run>) =>
    actions.filter((a) => a.data.tool === S)
  const placeholderCards = (actions: ReturnType<typeof run>) =>
    actions.filter((a) => a.data.tool === P)

  it("replaces the placeholder card when a sibling reuses its id and ends later", () => {
    const actions = run(start(S, "A"), start(P, "A"), placeholderFailure("A"), endS)

    expect(actions).toHaveLength(2)
    expect(webSearchCards(actions)).toEqual([
      expect.objectContaining({ status: "completed", data: expect.objectContaining({ output: "RESULT_S" }) }),
    ])
    expect(placeholderCards(actions)).toEqual([])
    expect(actions[1]).toEqual(statusLineAction("event-2", 3000))
  })

  it("appends a status line without removing cards when a sibling reusing the id ends first", () => {
    const actions = run(start(S, "A"), start(P, "A"), endS, placeholderFailure("A"))

    expect(actions).toHaveLength(3)
    // The end lookup ignores tool names (baseline, out of scope): S's result lands on P's card.
    expect(actions[0]).toMatchObject({ type: "tool", status: "completed", data: { tool: S } })
    expect(actions[0].data.output).toBeUndefined()
    expect(actions[1]).toMatchObject({
      type: "tool",
      status: "completed",
      data: { tool: P, output: "RESULT_S" },
    })
    expect(actions[2]).toEqual(statusLineAction("event-4", 5000))
  })

  it.each([
    ["does not match the placeholder card", "B", "C"],
    ["is missing", undefined, undefined],
  ])("appends a status line and keeps every card when the failure id %s", (_, startId, failedId) => {
    const actions = run(start(S, "A"), start(P, startId), placeholderFailure(failedId), endS)

    expect(actions).toHaveLength(3)
    expect(webSearchCards(actions)).toEqual([
      expect.objectContaining({ status: "completed", data: expect.objectContaining({ output: "RESULT_S" }) }),
    ])
    expect(placeholderCards(actions)).toHaveLength(1)
    expect(actions[2]).toEqual(statusLineAction("event-3", 4000))
  })

  it("replaces only the running placeholder card when a completed card shares its id", () => {
    const endP = ev("tool_execution_end", {
      tool_name: P,
      tool_call_id: "A",
      result: { output: "RESULT_P" },
    })
    const actions = run(start(P, "A"), endP, start(P, "A"), placeholderFailure("A"))

    expect(actions).toHaveLength(2)
    expect(actions[0]).toMatchObject({
      type: "tool",
      status: "completed",
      data: { tool: P, output: "RESULT_P" },
    })
    expect(actions[1]).toEqual(statusLineAction("event-3", 4000))
  })

  it("appends a status line when two running cards share the id and tool name", () => {
    const actions = run(start(P, "A"), start(P, "A"), placeholderFailure("A"))

    expect(actions).toHaveLength(3)
    expect(placeholderCards(actions)).toHaveLength(2)
    expect(actions[2]).toEqual(statusLineAction("event-3", 4000))
  })

  it("renders a failure without a matching start as only a status line", () => {
    const events = [stepStart, placeholderFailure("A")].map((event, index) => ({
      ...event,
      timestamp: index + 1,
    }))

    const steps = processTraceEvents(events as never, tc)

    expect(steps[0].actions).toEqual([statusLineAction("event-1", 2000)])
  })
})

describe("getFriendlyToolName", () => {
  it("prettifies an unmapped snake_case tool name", () => {
    expect(getFriendlyToolName("some_future_tool")).toBe("Some Future Tool")
  })

  it("returns empty string for malformed (non-string) input instead of throwing", () => {
    expect(() => getFriendlyToolName(42 as never)).not.toThrow()
    expect(() => getFriendlyToolName({ unexpected: "object" } as never)).not.toThrow()
    expect(() => getFriendlyToolName(null as never)).not.toThrow()
    expect(getFriendlyToolName(42 as never)).toBe("")
    expect(getFriendlyToolName({ unexpected: "object" } as never)).toBe("")
    expect(getFriendlyToolName(null as never)).toBe("")
  })
})

// The mocked useI18n() above (t: key => key) means every other test in this
// file exercises processTraceEvents' own logic, not what the toolNames map
// actually resolves to — deleting the whole map would leave them green. Use
// the real resolver here so a regression in en.ts/zh.ts's toolNames entries,
// or in getFriendlyToolName's lookup path, actually fails a test.
describe("getFriendlyToolName against the real translation trees", () => {
  const realTDynamic = (locale: Locale): TranslateDynamic => (key, fallback) =>
    resolveDynamicTranslation(locale, key, fallback)

  it("resolves a mapped tool name to its curated phrase in both locales", () => {
    expect(getFriendlyToolName("web_search", realTDynamic("en"))).toBe("Searching the web")
    expect(getFriendlyToolName("web_search", realTDynamic("zh"))).toBe("正在搜索网络")
  })

  it("falls back to the prettified raw name for a tool absent from the map", () => {
    expect(getFriendlyToolName("some_future_tool", realTDynamic("en"))).toBe(
      "Some Future Tool",
    )
    expect(getFriendlyToolName("some_future_tool", realTDynamic("zh"))).toBe(
      "Some Future Tool",
    )
  })
})

describe("getRawToolName", () => {
  it("falls through to data.tool_name when response.tool_name is a truthy non-string", () => {
    // response.tool_name and data.tool_name must be checked independently —
    // `a || b` then type-checking the combined result gives up here instead
    // of falling through, because the truthy (but non-string) 42 already
    // won the `||` before either side was ever checked for being a string.
    const event = {
      data: { response: { tool_name: 42 }, tool_name: "web_search" },
    } as never
    expect(getRawToolName(event)).toBe("web_search")
  })

  it("prefers response.tool_name over data.tool_name when both are valid strings", () => {
    const event = {
      data: { response: { tool_name: "fetch_web_content" }, tool_name: "web_search" },
    } as never
    expect(getRawToolName(event)).toBe("fetch_web_content")
  })

  it("returns an empty string when neither field is a string", () => {
    expect(getRawToolName({ data: {} } as never)).toBe("")
    expect(getRawToolName({} as never)).toBe("")
  })
})

describe("getProgressNarrationText", () => {
  it("falls through to content when message is an empty string, via ||", () => {
    // Deliberately || not ?? — an empty message string is falsy under ||
    // (falls through to content) but not under ?? (only null/undefined
    // fall through). A regression back to ?? would silently re-break this.
    const event = { data: { message: "", content: "text" } } as never
    expect(getProgressNarrationText(event)).toBe("text")
  })

  it("prefers message over content when both are present", () => {
    const event = { data: { message: "the message", content: "the content" } } as never
    expect(getProgressNarrationText(event)).toBe("the message")
  })

  it("returns an empty string for a non-string value and for no data at all", () => {
    expect(getProgressNarrationText({ data: { message: 42 } } as never)).toBe("")
    expect(getProgressNarrationText({} as never)).toBe("")
  })
})

describe("every toolNames entry resolves through getFriendlyToolName", () => {
  // The single-key test above only pins web_search; the other 55 entries
  // per locale were unpinned — a typo'd value, or a key present in one
  // locale's source object but silently dropped en route to the resolver,
  // would go unnoticed. Iterate every key actually in the source file
  // (self-referential, so no per-key value needs hand-copying here) and
  // assert getFriendlyToolName resolves it exactly.
  //
  // Typed with `satisfies Partial<…>` rather than annotated
  // `Record<Locale, …>`: a downstream build can replace @/i18n/translations
  // with one that adds locales, widening Locale beyond the source objects
  // this repo ships (src/lib/time-utils.ts types displayDateLocales as a
  // Partial for the same reason, and its test pins that annotation). An
  // exhaustive Record here would demand an entry for a locale core has no
  // module to import, breaking their type-check on a file only this repo
  // owns. `satisfies` still rejects a wrong value shape or a key that is not
  // a Locale at all.
  const toolNames = {
    en: en.traceEventRenderer.toolNames,
    zh: zh.traceEventRenderer.toolNames,
  } satisfies Partial<Record<Locale, Record<string, string>>>

  for (const locale of Object.keys(toolNames) as (keyof typeof toolNames)[]) {
    it(`resolves all ${Object.keys(toolNames[locale]).length} ${locale} entries to their literal source value`, () => {
      const tDynamic = (key: string, fallback: string) =>
        resolveDynamicTranslation(locale, key, fallback)
      for (const [toolName, expected] of Object.entries(toolNames[locale])) {
        expect(getFriendlyToolName(toolName, tDynamic)).toBe(expected)
      }
    })
  }

  it("keeps the same set of tool names mapped in both locales", () => {
    expect(Object.keys(toolNames.en).sort()).toEqual(Object.keys(toolNames.zh).sort())
  })

  it("keeps the fixture's Locale boundary a Partial", () => {
    // Everything above iterates only the keys the fixture actually has, so it
    // would all still pass if the annotation regressed to an exhaustive
    // `Record<Locale, …>`: while Locale is exactly en | zh the two types are
    // interchangeable here, and no runtime assertion can tell them apart. The
    // difference only surfaces in a build that widens Locale, which this repo
    // never runs -- so pin the boundary in the source text, the way
    // src/lib/time-utils.test.ts pins displayDateLocales' Partial.
    //
    // The needle is assembled from fragments rather than written out whole:
    // this guard sits in the file it guards, so a single literal would match
    // its own occurrence and keep passing after the fixture had regressed.
    const boundary = ["} satisfies Partial<Record<Locale, ", "Record<string, string>>>"].join("")
    const source = readFileSync("src/components/chat/trace-event-processing.test.ts", "utf8")

    expect(source).toContain(boundary)
  })
})

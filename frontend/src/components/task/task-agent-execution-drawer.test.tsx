import React from "react"
import { act, cleanup, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { I18nProvider } from "@/contexts/i18n-context"
import { TaskAgentExecutionDrawer } from "./task-agent-execution-drawer"

const request = vi.hoisted(() => vi.fn())
vi.mock("@/lib/api-wrapper", () => ({ apiRequest: request }))
vi.mock("@/lib/utils", async (original) => ({
  ...await original<typeof import("@/lib/utils")>(), getApiUrl: () => "https://example.test",
}))
const realPanel = vi.hoisted(() => ({ enabled: false }))
vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }))
vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({ dispatch: vi.fn(), openFilePreview: vi.fn(), filesDisabled: false }),
}))
vi.mock("@/components/file/docx-preview-renderer", () => ({ DocxPreviewRenderer: () => null }))
vi.mock("@/components/file/excel-preview-renderer", () => ({ ExcelPreviewRenderer: () => null }))
vi.mock("@/components/file/pptx-preview-renderer", () => ({ PptxPreviewRenderer: () => null }))
vi.mock("./agent-execution-panel", async () => {
  const actual = await vi.importActual<typeof import("./agent-execution-panel")>("./agent-execution-panel")
  return {
    ...actual,
    sanitizeAgentExecutionTraceEvents: (events: unknown) =>
      realPanel.enabled ? actual.sanitizeAgentExecutionTraceEvents(events) : events || [],
    // Tests that assert on the rendered trace opt in through realPanel.
    AgentExecutionPanel: (props: React.ComponentProps<typeof actual.AgentExecutionPanel>) =>
      realPanel.enabled ? <actual.AgentExecutionPanel {...props} /> : (
        <div>{props.loading ? "Loading child" : (props.detail as { output?: string } | null)?.output}</div>
      ),
  }
})
const selection = { workerTaskId: "child/a", agentName: "Researcher", status: "running" as const }
const response = (status: string, output: string) => ({ ok: true, json: async () => ({ status, output, trace_events: [] }) })
const view = (taskId = 1, workerTaskId = selection.workerTaskId) => (
  <I18nProvider initialLocale="en">
    <TaskAgentExecutionDrawer taskId={taskId} selection={{ ...selection, workerTaskId }} onClose={vi.fn()} />
  </I18nProvider>
)

beforeEach(() => { vi.useFakeTimers(); request.mockReset() })
afterEach(() => { cleanup(); vi.useRealTimers(); realPanel.enabled = false })

describe("TaskAgentExecutionDrawer", () => {
  it("loads the selected authorized child scope and refreshes until terminal", async () => {
    request.mockResolvedValueOnce(response("running", "In progress"))
      .mockResolvedValueOnce(response("completed", "Finished child"))
    await act(async () => { render(view()) })
    expect(request.mock.calls[0][0]).toBe("https://example.test/api/chat/task/1/agent-executions/child%2Fa")
    expect(screen.getByText("In progress")).toBeInTheDocument()
    await act(async () => { await vi.advanceTimersByTimeAsync(5000) })
    expect(screen.getByText("Finished child")).toBeInTheDocument()
    await act(async () => { await vi.advanceTimersByTimeAsync(10000) })
    expect(request).toHaveBeenCalledTimes(2)
  })

  it("aborts an old task request and ignores its late response", async () => {
    let resolveOld!: (value: ReturnType<typeof response>) => void
    request.mockReturnValueOnce(new Promise((resolve) => { resolveOld = resolve }))
      .mockResolvedValueOnce(response("completed", "New task child"))
    const rendered = render(view())
    const oldSignal = request.mock.calls[0][1].signal as AbortSignal
    await act(async () => { rendered.rerender(view(2, "child-b")) })
    expect(oldSignal.aborted).toBe(true)
    expect(request.mock.calls[1][0]).toBe("https://example.test/api/chat/task/2/agent-executions/child-b")
    await act(async () => { resolveOld(response("running", "Old task child")) })
    expect(screen.getByText("New task child")).toBeInTheDocument()
    expect(screen.queryByText("Old task child")).not.toBeInTheDocument()
    await act(async () => { await vi.advanceTimersByTimeAsync(10000) })
    expect(request).toHaveBeenCalledTimes(2)
  })

  it("cancels polling when closed", async () => {
    request.mockResolvedValue(response("running", "In progress"))
    let rendered!: ReturnType<typeof render>
    await act(async () => { rendered = render(view()) })
    const signal = request.mock.calls[0][1].signal as AbortSignal
    rendered.unmount()
    expect(signal.aborted).toBe(true)
    await act(async () => { await vi.advanceTimersByTimeAsync(10000) })
    expect(request).toHaveBeenCalledTimes(1)
  })

  it("stops a worker's steps once the inspector reports it interrupted", async () => {
    realPanel.enabled = true
    request.mockResolvedValue({
      ok: true,
      json: async () => ({
        status: "interrupted",
        output: "",
        trace_events: [
          { event_id: "start", event_type: "react_task_start", step_id: "s1", timestamp: 1000, data: {} },
          {
            event_id: "tool",
            event_type: "tool_execution_start",
            step_id: "s1",
            timestamp: 2000,
            data: { tool_name: "web_search", tool_args: { query: "probe" } },
          },
        ],
      }),
    })
    await act(async () => { render(view()) })

    expect(document.body.querySelector(".animate-spin")).toBeNull()
    expect(document.body.querySelector("svg[data-icon='interrupted']")).not.toBeNull()
    expect(document.body.querySelector(".bg-amber-100")).toHaveTextContent("Interrupted")
  })
})

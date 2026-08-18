import React from "react"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { I18nProvider } from "@/contexts/i18n-context"
import type { AppState } from "@/contexts/app-context-chat"

const navigation = vi.hoisted(() => ({
  params: { id: "1" },
  push: vi.fn(),
}))

vi.mock("next/navigation", () => ({
  useParams: () => navigation.params,
  useRouter: () => ({ push: navigation.push }),
}))

vi.mock("@/components/task/task-conversation-panel", () => ({
  TaskConversationPanel: () => <div data-testid="conversation-panel" />,
}))

const app = vi.hoisted(() => ({
  state: {} as Partial<AppState>,
  setTaskId: vi.fn(),
  closeFilePreview: vi.fn(),
}))

vi.mock("@/contexts/app-context-chat", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/contexts/app-context-chat")>()
  return {
    ...actual,
    useApp: () => ({
      state: app.state,
      setTaskId: app.setTaskId,
      closeFilePreview: app.closeFilePreview,
    }),
  }
})

import TaskDetailPage from "./page-client"

function baseState(overrides: Partial<AppState> = {}): Partial<AppState> {
  return {
    taskId: 1,
    currentTask: {
      id: "1",
      title: "Test task",
      status: "running",
      description: "Test task",
      createdAt: "2026-05-27T05:00:00Z",
      updatedAt: "2026-05-27T05:00:00Z",
    },
    dagExecution: null,
    steps: [],
    ...overrides,
  } as Partial<AppState>
}

function renderPage() {
  return render(
    <I18nProvider initialLocale="en">
      <TaskDetailPage />
    </I18nProvider>
  )
}

afterEach(() => {
  cleanup()
  navigation.params = { id: "1" }
  navigation.push.mockReset()
  app.setTaskId.mockReset()
  app.closeFilePreview.mockReset()
})

describe("TaskDetailPage progress panel lifecycle", () => {
  beforeEach(() => {
    app.state = baseState()
  })

  it("does not render the panel toggle or panel while there is no dagExecution", () => {
    renderPage()
    expect(screen.queryByTitle("Show execution progress")).not.toBeInTheDocument()
    expect(screen.queryByText("Progress")).not.toBeInTheDocument()
  })

  it("auto-opens the panel the moment dagExecution first appears", () => {
    const { rerender } = renderPage()
    app.state = baseState({
      dagExecution: { phase: "planning", current_plan: {}, created_at: "t", updated_at: "t", turn_id: "turn-A" },
    })
    rerender(
      <I18nProvider initialLocale="en">
        <TaskDetailPage />
      </I18nProvider>
    )
    expect(screen.getByText("Progress")).toBeInTheDocument()
  })

  it("suppresses re-opening for the SAME run once manually dismissed, but reopens for a new run", () => {
    app.state = baseState({
      dagExecution: { phase: "planning", current_plan: {}, created_at: "t", updated_at: "t", turn_id: "turn-A" },
    })
    const { rerender } = renderPage()
    expect(screen.getByText("Progress")).toBeInTheDocument()

    // Manually collapse via the header toggle.
    fireEvent.click(screen.getByTitle("Collapse"))
    expect(screen.queryByText("Progress")).not.toBeInTheDocument()

    // A further event for the SAME run (turn_id unchanged) must not reopen it.
    app.state = baseState({
      dagExecution: { phase: "executing", current_plan: {}, created_at: "t", updated_at: "t2", turn_id: "turn-A" },
    })
    rerender(
      <I18nProvider initialLocale="en">
        <TaskDetailPage />
      </I18nProvider>
    )
    expect(screen.queryByText("Progress")).not.toBeInTheDocument()

    // A NEW run (different turn_id) must reopen it.
    app.state = baseState({
      dagExecution: { phase: "planning", current_plan: {}, created_at: "t3", updated_at: "t3", turn_id: "turn-B" },
    })
    rerender(
      <I18nProvider initialLocale="en">
        <TaskDetailPage />
      </I18nProvider>
    )
    expect(screen.getByText("Progress")).toBeInTheDocument()
  })

  it("closes the panel when the backdrop is clicked", () => {
    app.state = baseState({
      dagExecution: { phase: "planning", current_plan: {}, created_at: "t", updated_at: "t", turn_id: "turn-A" },
    })
    const { container } = renderPage()
    expect(screen.getByText("Progress")).toBeInTheDocument()

    const backdrop = container.querySelector('[aria-hidden="true"]')
    expect(backdrop).not.toBeNull()
    fireEvent.click(backdrop as Element)
    expect(screen.queryByText("Progress")).not.toBeInTheDocument()
  })

  it("closes the panel on Escape", () => {
    app.state = baseState({
      dagExecution: { phase: "planning", current_plan: {}, created_at: "t", updated_at: "t", turn_id: "turn-A" },
    })
    renderPage()
    expect(screen.getByText("Progress")).toBeInTheDocument()

    act(() => {
      fireEvent.keyDown(window, { key: "Escape" })
    })
    expect(screen.queryByText("Progress")).not.toBeInTheDocument()
  })

  it("reopens via the header toggle after a manual collapse", () => {
    app.state = baseState({
      dagExecution: { phase: "planning", current_plan: {}, created_at: "t", updated_at: "t", turn_id: "turn-A" },
    })
    renderPage()
    fireEvent.click(screen.getByTitle("Collapse"))
    expect(screen.queryByText("Progress")).not.toBeInTheDocument()

    fireEvent.click(screen.getByTitle("Show execution progress"))
    expect(screen.getByText("Progress")).toBeInTheDocument()
  })

  it("freezes the elapsed time at dagTerminatedAt instead of the live-ticking now", () => {
    // dagTerminatedAt (not the mutable currentTask.updatedAt) is what this
    // page passes through as ProgressPanel's endedAt - it's stamped once when
    // the run actually finishes and survives later, unrelated task metadata
    // refreshes untouched. Start the task "running" so the panel auto-opens
    // (a terminal status from the very first render is deliberately excluded
    // from auto-open - see the effect above - so this exercises the realistic
    // sequence: open while live, then the run ends under it).
    app.state = baseState({
      dagExecution: { phase: "executing", current_plan: {}, created_at: "2026-05-27T05:00:00Z", updated_at: "2026-05-27T05:00:00Z", turn_id: "turn-A" },
    })
    const { rerender } = renderPage()
    expect(screen.getByText("Progress")).toBeInTheDocument()

    app.state = baseState({
      currentTask: {
        id: "1",
        title: "Test task",
        status: "completed",
        description: "Test task",
        createdAt: "2026-05-27T05:00:00Z",
        updatedAt: "2026-05-27T05:05:00Z",
        dagTerminatedAt: "2026-05-27T05:01:05Z",
      },
      dagExecution: { phase: "completed", current_plan: {}, created_at: "2026-05-27T05:00:00Z", updated_at: "2026-05-27T05:01:05Z", turn_id: "turn-A" },
    })
    rerender(
      <I18nProvider initialLocale="en">
        <TaskDetailPage />
      </I18nProvider>
    )
    // 05:01:05 - 05:00:00 = 65s -> "1m 05s". If the page fell back to
    // currentTask.updatedAt (05:05:00) instead, this would read "5m 00s".
    expect(screen.getByText("1m 05s")).toBeInTheDocument()
  })

  it("does not auto-open for state populated by history/replay or an already-terminal task", () => {
    // A finished task's page load populates dagExecution/progressRunKey from
    // history through the exact same state the auto-open effect watches -
    // without the history/terminal gates, just VIEWING a finished task would
    // pop the panel open unprompted.
    app.state = baseState({
      currentTask: {
        id: "1",
        title: "Test task",
        status: "completed",
        description: "Test task",
        createdAt: "2026-05-27T05:00:00Z",
        updatedAt: "2026-05-27T05:01:00Z",
        dagTerminatedAt: "2026-05-27T05:01:00Z",
      },
      dagExecution: { phase: "completed", current_plan: {}, created_at: "t", updated_at: "t", turn_id: "turn-A" },
    })
    renderPage()
    expect(screen.queryByText("Progress")).not.toBeInTheDocument()
    // The header toggle still works - the gate only stops AUTO-opening.
    fireEvent.click(screen.getByTitle("Show execution progress"))
    expect(screen.getByText("Progress")).toBeInTheDocument()
  })

  it("hides superseded step rows while a replan is generating the replacement plan", () => {
    // A replanning event carries no replacement steps and shares the run's
    // turn_id, so the old plan's rows survive in state.steps - the panel
    // must not keep presenting them as current (the graph view already hides
    // the old plan for this phase).
    app.state = baseState({
      dagExecution: { phase: "executing", current_plan: {}, created_at: "t", updated_at: "t", turn_id: "turn-A" },
      steps: [{
        id: "step-one",
        name: "Old step",
        description: "",
        status: "completed",
        dependencies: [],
      }],
    })
    const { rerender } = renderPage()
    expect(screen.getByText("Old step")).toBeInTheDocument()

    app.state = baseState({
      dagExecution: { phase: "replanning", current_plan: {}, created_at: "t", updated_at: "t2", turn_id: "turn-A" },
      steps: [{
        id: "step-one",
        name: "Old step",
        description: "",
        status: "completed",
        dependencies: [],
      }],
    })
    rerender(
      <I18nProvider initialLocale="en">
        <TaskDetailPage />
      </I18nProvider>
    )
    expect(screen.queryByText("Old step")).not.toBeInTheDocument()
  })

  it("still shows the last known steps when the task has already ended while stuck in replanning", () => {
    // The backend's phase can get stuck at "replanning" if the task ends via
    // a path that never syncs dagExecution.phase, or a malformed/dropped
    // executing-transition event is dropped outright by the normalizer.
    // ProgressPanel itself renders NOTHING for zero steps once endedAt is
    // set (no spinner, no rows) - hiding steps here too would blank the
    // entire panel body for a finished run instead of showing its last
    // known state, which is strictly worse than a rare stale-plan flash.
    app.state = baseState({
      currentTask: {
        id: "1",
        title: "Test task",
        status: "completed",
        description: "Test task",
        createdAt: "2026-05-27T05:00:00Z",
        updatedAt: "2026-05-27T05:02:00Z",
        dagTerminatedAt: "2026-05-27T05:02:00Z",
      },
      dagExecution: { phase: "replanning", current_plan: {}, created_at: "t", updated_at: "t2", turn_id: "turn-A" },
      steps: [{
        id: "step-one",
        name: "Old step",
        description: "",
        status: "completed",
        dependencies: [],
      }],
    })
    renderPage()
    // A terminal task from the very first render doesn't auto-open (see the
    // history/terminal auto-open test above) - open it via the header toggle.
    fireEvent.click(screen.getByTitle("Show execution progress"))
    expect(screen.getByText("Old step")).toBeInTheDocument()
  })
})

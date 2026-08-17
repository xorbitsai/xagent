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
})

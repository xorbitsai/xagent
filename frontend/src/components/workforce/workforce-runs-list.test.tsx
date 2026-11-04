/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, render, screen, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { WorkforceRunHistoryItem } from "@/types/workforce"

const listWorkforceRunsMock = vi.hoisted(() => vi.fn())
// Stable across renders: the list's loader depends on `t`, so a fresh function
// per render would re-run its load effect forever.
const translateMock = vi.hoisted(
  () => (key: string, vars?: Record<string, string | number>) =>
    vars?.date !== undefined ? `${key}:${vars.date}` : key,
)

vi.mock("@/lib/workforces-api", () => ({
  listWorkforceRuns: listWorkforceRunsMock,
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: translateMock }),
}))

import { WorkforceRunsList } from "./workforce-runs-list"

function makeRun(overrides: Partial<WorkforceRunHistoryItem> & { id: number }): WorkforceRunHistoryItem {
  return {
    task_id: null,
    status: "completed",
    is_preview: false,
    task_title: null,
    message: null,
    created_at: null,
    completed_at: null,
    ...overrides,
  }
}

describe("WorkforceRunsList", () => {
  beforeEach(() => {
    listWorkforceRunsMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it("tells a run whose conversation expired apart from one whose task was deleted", async () => {
    listWorkforceRunsMock.mockResolvedValue({
      items: [
        // Retention expired the conversation: the run still completed.
        makeRun({
          id: 1,
          task_title: "Expired run",
          status: "completed",
          task_expired_at: "2026-09-01T08:30:00+00:00",
        }),
        // Deleted some other way: no expiry timestamp.
        makeRun({ id: 2, task_title: "Deleted run", status: "failed" }),
        makeRun({ id: 3, task_title: "Live run", task_id: 30 }),
      ],
      total: 3,
      page: 1,
      size: 20,
      pages: 1,
    })

    render(<WorkforceRunsList workforceId={5} onSelectRun={vi.fn()} />)

    const expired = (await screen.findByText("Expired run")).closest("button") as HTMLElement
    expect(within(expired).getByText("workforces.runs.status.completed")).toBeInTheDocument()
    expect(
      within(expired).getByText(/^workforces\.runs\.taskExpired:.+/),
    ).toBeInTheDocument()
    expect(within(expired).queryByText("workforces.runs.taskDeleted")).not.toBeInTheDocument()
    // No conversation to open either way.
    expect(expired).toBeDisabled()

    const deleted = screen.getByText("Deleted run").closest("button") as HTMLElement
    expect(within(deleted).getByText("workforces.runs.status.failed")).toBeInTheDocument()
    expect(within(deleted).getByText("workforces.runs.taskDeleted")).toBeInTheDocument()
    expect(within(deleted).queryByText(/workforces\.runs\.taskExpired/)).not.toBeInTheDocument()

    const live = screen.getByText("Live run").closest("button") as HTMLElement
    expect(within(live).queryByText("workforces.runs.taskDeleted")).not.toBeInTheDocument()
    expect(within(live).queryByText(/workforces\.runs\.taskExpired/)).not.toBeInTheDocument()
    expect(live).toBeEnabled()
  })
})

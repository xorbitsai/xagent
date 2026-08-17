import React from "react"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { I18nProvider } from "@/contexts/i18n-context"
import { ProgressPanel, type ProgressStepView } from "./progress-panel"

afterEach(() => {
  cleanup()
})

const NOW = "2026-05-27T05:10:00Z"

function renderPanel(props: Partial<React.ComponentProps<typeof ProgressPanel>> = {}) {
  const onCollapse = vi.fn()
  const onStepClick = vi.fn()
  const defaults: React.ComponentProps<typeof ProgressPanel> = {
    steps: [],
    startedAt: undefined,
    endedAt: undefined,
    onCollapse,
    onStepClick,
  }
  render(
    <I18nProvider initialLocale="en">
      <ProgressPanel {...defaults} {...props} />
    </I18nProvider>
  )
  return { onCollapse, onStepClick }
}

const step = (overrides: Partial<ProgressStepView> = {}): ProgressStepView => ({
  id: "step-one",
  title: "Step one",
  status: "pending",
  ...overrides,
})

describe("ProgressPanel", () => {
  it("shows the planning placeholder when there are no steps and the run hasn't ended", () => {
    renderPanel({ steps: [], endedAt: undefined })
    expect(screen.getByText("Generating plan…")).toBeInTheDocument()
  })

  it("hides the planning placeholder once the run has ended, even with no steps", () => {
    // A plan-time failure can end a run before any step ever started - the
    // placeholder must not spin forever over a run that's already over.
    renderPanel({ steps: [], endedAt: "2026-05-27T05:00:05Z" })
    expect(screen.queryByText("Generating plan…")).not.toBeInTheDocument()
  })

  it("counts completed/skipped/failed as resolved but not interrupted/pending/running", () => {
    renderPanel({
      steps: [
        step({ id: "a", status: "completed" }),
        step({ id: "b", status: "skipped" }),
        step({ id: "c", status: "failed" }),
        step({ id: "d", status: "interrupted" }),
        step({ id: "e", status: "pending" }),
        step({ id: "f", status: "running" }),
      ],
    })
    expect(screen.getByText("3/6")).toBeInTheDocument()
  })

  it("does not render a step-count fraction when there are no steps yet", () => {
    renderPanel({ steps: [] })
    expect(screen.queryByText(/^\d+\/\d+$/)).not.toBeInTheDocument()
  })

  it("renders distinct sr-only status text for interrupted and clarification_invalidated steps", () => {
    renderPanel({
      steps: [
        step({ id: "a", status: "interrupted", title: "Interrupted step" }),
        step({ id: "b", status: "clarification_invalidated", title: "Invalidated step" }),
      ],
    })
    expect(screen.getByText("Interrupted:")).toBeInTheDocument()
    expect(screen.getByText("Waiting on a new answer:")).toBeInTheDocument()
  })

  it("disables a step row with no startedAt and never calls onStepClick for it", () => {
    // A disabled button is excluded from the accessible role/name query by
    // default, so find it via its text content instead of getByRole.
    const { onStepClick } = renderPanel({
      steps: [step({ id: "never-started", status: "pending", startedAt: undefined })],
    })
    const button = screen.getByText("Step one").closest("button")
    expect(button).toBeDisabled()
    fireEvent.click(button as HTMLButtonElement)
    expect(onStepClick).not.toHaveBeenCalled()
  })

  it("calls onStepClick with the step id for a step that has started", () => {
    const { onStepClick } = renderPanel({
      steps: [step({ id: "started", status: "completed", startedAt: "2026-05-27T05:00:00Z" })],
    })
    fireEvent.click(screen.getByRole("button", { name: "Completed: Step one" }))
    expect(onStepClick).toHaveBeenCalledWith("started")
  })

  it("calls onCollapse when the collapse button is clicked", () => {
    const { onCollapse } = renderPanel()
    fireEvent.click(screen.getByLabelText("Collapse"))
    expect(onCollapse).toHaveBeenCalledOnce()
  })

  describe("elapsed time", () => {
    beforeEach(() => {
      vi.useFakeTimers({ now: new Date(NOW) })
    })
    afterEach(() => {
      vi.useRealTimers()
    })

    it("shows a live, ticking total elapsed time while the run hasn't ended", () => {
      // Started 90s before the fixed "now".
      renderPanel({ startedAt: "2026-05-27T05:08:30Z", endedAt: undefined })
      expect(screen.getByText("1m 30s")).toBeInTheDocument()

      act(() => {
        vi.advanceTimersByTime(5000)
      })
      expect(screen.getByText("1m 35s")).toBeInTheDocument()
    })

    it("freezes the total elapsed time at endedAt - startedAt once the run has ended", () => {
      renderPanel({ startedAt: "2026-05-27T05:00:00Z", endedAt: "2026-05-27T05:05:00Z" })
      expect(screen.getByText("5m 00s")).toBeInTheDocument()

      // Time passing after the freeze must not change the displayed value.
      act(() => {
        vi.advanceTimersByTime(60_000)
      })
      expect(screen.getByText("5m 00s")).toBeInTheDocument()
    })

    it("keeps ticking a running step's own duration even when the panel's own startedAt is missing", () => {
      // Regression coverage for the shared-clock coupling bug: the clock must
      // stay active off a running row's own startedAt, not just the panel's.
      renderPanel({
        startedAt: undefined,
        endedAt: undefined,
        steps: [step({ id: "running-step", status: "running", startedAt: "2026-05-27T05:09:00Z" })],
      })
      expect(screen.getByText("1m 00s")).toBeInTheDocument()

      act(() => {
        vi.advanceTimersByTime(1000)
      })
      expect(screen.getByText("1m 01s")).toBeInTheDocument()
    })

    it("freezes a still-running step's duration at the run's own endedAt once the run ends", () => {
      // The backend cancels sibling steps without a terminal event when a run
      // ends, so a step stuck "running" must freeze against endedAt, not keep
      // counting against "now".
      renderPanel({
        startedAt: "2026-05-27T05:00:00Z",
        endedAt: "2026-05-27T05:03:00Z",
        steps: [step({ id: "orphaned", status: "running", startedAt: "2026-05-27T05:01:00Z" })],
      })
      expect(screen.getByText("2m 00s")).toBeInTheDocument()

      act(() => {
        vi.advanceTimersByTime(60_000)
      })
      expect(screen.getByText("2m 00s")).toBeInTheDocument()
    })

    it("shows a static (non-ticking) duration for a step with both startedAt and completedAt", () => {
      renderPanel({
        steps: [step({
          id: "done",
          status: "completed",
          startedAt: "2026-05-27T05:00:00Z",
          completedAt: "2026-05-27T05:00:42Z",
        })],
      })
      expect(screen.getByText("42s")).toBeInTheDocument()
    })

    it("treats epoch zero as a real timestamp rather than as absent", () => {
      renderPanel({ startedAt: 0, endedAt: undefined })
      // now (fixed at NOW) - epoch 0 is a huge duration, not "0s"/absent.
      expect(screen.queryByText("Elapsed")).toBeInTheDocument()
      expect(screen.queryByText("0s")).not.toBeInTheDocument()
    })
  })
})

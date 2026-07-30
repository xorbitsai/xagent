/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

vi.mock("@/lib/utils", () => ({
  cn: (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(" "),
}))

const translateMock = (key: string, vars?: Record<string, string | number>) => {
  if (vars) {
    const firstValue = Object.values(vars)[0]
    return `${key}:${firstValue}`
  }
  return key
}

import {
  ScheduleFields,
  scheduleFieldsDefaults,
  summarizeSchedule,
  type ScheduleFieldsValue,
} from "./agent-triggers-schedule-fields"

// PR #1051 review: this ~300-line component had no dedicated test file, so
// summarizeSchedule (the user-visible rendering of every recurrence
// semantic this PR introduces) went completely uncovered.
describe("summarizeSchedule", () => {
  const base = (overrides: Partial<ScheduleFieldsValue> = {}): ScheduleFieldsValue => ({
    ...scheduleFieldsDefaults(),
    startDate: "",
    ...overrides,
  })

  it("summarizes hourly with no timezone suffix (flat interval, tz-agnostic)", () => {
    const text = summarizeSchedule(base({ recurrence: "hourly" }), translateMock, "en")
    expect(text).toBe("triggers.schedule.summaryHourly")
  })

  it("summarizes custom with no timezone suffix (flat interval, tz-agnostic)", () => {
    const text = summarizeSchedule(
      base({ recurrence: "custom", customAmount: "30", customUnit: "minutes" }),
      translateMock,
      "en",
    )
    expect(text).toBe("triggers.schedule.summaryCustom:30")
  })

  it("appends the timezone for daily (a real tz-aware recurrence)", () => {
    const text = summarizeSchedule(
      base({ recurrence: "daily", timeOfDay: "09:00", timezone: "Asia/Shanghai" }),
      translateMock,
      "en",
    )
    // summaryWithTimezone wraps the base summary; the mock only echoes the
    // first interpolated value, so this proves the wrapping call happened
    // (base summary correctness is covered by the other cases; the tz-vs-no
    // -tz DISTINCTION is what a translateMock echoing every var would hide).
    // timeOfDay is real Intl-formatted ("9:00 AM"), not the raw "09:00".
    expect(text.startsWith("triggers.schedule.summaryWithTimezone:")).toBe(true)
    expect(text).toContain("triggers.schedule.summaryDaily")
  })

  it("appends the timezone for weekly and monthly too", () => {
    const weekly = summarizeSchedule(
      base({ recurrence: "weekly", weekdays: [0], timezone: "UTC" }),
      translateMock,
      "en",
    )
    expect(weekly.startsWith("triggers.schedule.summaryWithTimezone:")).toBe(true)

    const monthly = summarizeSchedule(
      base({ recurrence: "monthly", dayOfMonth: 1, timezone: "UTC" }),
      translateMock,
      "en",
    )
    expect(monthly.startsWith("triggers.schedule.summaryWithTimezone:")).toBe(true)
  })

  it("includes the start-date suffix before the timezone suffix", () => {
    const text = summarizeSchedule(
      base({ recurrence: "daily", startDate: "2026-01-01" }),
      translateMock,
      "en",
    )
    // summaryStartsOnly wraps the base ("...at 09:00"), and the whole thing
    // is then wrapped again by summaryWithTimezone — both wrappers must
    // have fired, in that order, not just one of them.
    expect(text.startsWith("triggers.schedule.summaryWithTimezone:")).toBe(true)
    expect(text).toContain("triggers.schedule.summaryStartsOnly")
  })
})

describe("scheduleFieldsDefaults", () => {
  it("defaults timezone to the environment's own IANA zone", () => {
    const defaults = scheduleFieldsDefaults()
    expect(defaults.timezone).toBe(Intl.DateTimeFormat().resolvedOptions().timeZone)
  })
})

describe("ScheduleFields", () => {
  afterEach(() => {
    cleanup()
  })

  function renderField(overrides: Partial<ScheduleFieldsValue> = {}) {
    const onChange = vi.fn()
    const value: ScheduleFieldsValue = { ...scheduleFieldsDefaults(), recurrence: "daily", ...overrides }
    render(<ScheduleFields value={value} onChange={onChange} t={translateMock} locale="en" />)
    return onChange
  }

  it("shows a read-only timezone label for daily/weekly/monthly", () => {
    renderField({ recurrence: "daily", timezone: "Asia/Shanghai" })
    expect(
      screen.getByTestId("schedule-timezone-label"),
    ).toHaveTextContent("triggers.schedule.timezoneLabel:Asia/Shanghai")
  })

  it("hides the timezone label for hourly and custom (not actually tz-aware)", () => {
    renderField({ recurrence: "hourly" })
    expect(screen.queryByTestId("schedule-timezone-label")).not.toBeInTheDocument()
  })

  it("calls onChange with the new recurrence when a chip is clicked", () => {
    const onChange = renderField({ recurrence: "hourly" })
    fireEvent.click(screen.getByText("triggers.schedule.daily"))
    expect(onChange).toHaveBeenCalledWith("recurrence", "daily")
  })
})

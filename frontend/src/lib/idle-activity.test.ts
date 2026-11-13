import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { isUserActivityEvent, startIdleActivityReporter, type IdleActivityReporterOptions } from "@/lib/idle-activity"
import type { IdleSessionRecord } from "@/lib/idle-session"

function record(overrides: Partial<IdleSessionRecord> = {}): IdleSessionRecord {
  return {
    schemaVersion: 1, sessionId: "browser-login", enforced: true, sid: "sid-a",
    idleTimeoutSeconds: 1800, idleExpiresIn: 1700, absoluteExpiresIn: null, activityMinIntervalSeconds: 60,
    observedAt: Date.now(), lastReportAt: null, serverTime: null, ...overrides,
  }
}

function setup(initial: IdleSessionRecord | null = record(), overrides: Partial<IdleActivityReporterOptions> = {}) {
  let current = initial
  const report = vi.fn(async () => true)
  const reporter = startIdleActivityReporter({
    target: window,
    isUserEvent: () => true,
    getRecord: () => current,
    markReported: () => { if (current) current = { ...current, lastReportAt: Date.now() } },
    report,
    ...overrides,
  })
  return {
    reporter, report,
    setRecord: (next: IdleSessionRecord | null) => { current = next },
    fire: (type: string) => window.dispatchEvent(new Event(type)),
  }
}

describe("idle activity reporting", () => {
  beforeEach(() => { vi.useFakeTimers(); vi.setSystemTime(new Date("2026-11-13T10:00:00Z")) })
  afterEach(() => { vi.useRealTimers() })

  it("reports keyboard, pointer and wheel input but not mouse movement, scrolling or focus", () => {
    const { reporter, report, fire } = setup()
    for (const type of ["mousemove", "scroll", "focus", "visibilitychange", "pointermove"]) fire(type)
    expect(report).not.toHaveBeenCalled()
    fire("keydown")
    expect(report).toHaveBeenCalledTimes(1)
    reporter.stop()
  })

  it.each(["pointerdown", "wheel"])("treats %s as user activity", type => {
    const { reporter, report, fire } = setup()
    fire(type)
    expect(report).toHaveBeenCalledTimes(1)
    reporter.stop()
  })

  it("ignores events that were not produced by the user", () => {
    expect(isUserActivityEvent(new KeyboardEvent("keydown"))).toBe(false)
    const { reporter, report, fire } = setup(record(), { isUserEvent: isUserActivityEvent })
    fire("keydown")
    expect(report).not.toHaveBeenCalled()
    reporter.stop()
  })

  it("sends at most one report per interval and one trailing report for input inside it", async () => {
    const { reporter, report, fire } = setup()
    fire("keydown")
    await vi.advanceTimersByTimeAsync(0)
    for (let i = 0; i < 50; i += 1) fire("pointerdown")
    // The trailing report waits slightly past the server interval, so the
    // server sees a full interval since its last write and does not drop it.
    await vi.advanceTimersByTimeAsync(61_000)
    expect(report).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(1_000)
    expect(report).toHaveBeenCalledTimes(2)
    // No input since the trailing report: nothing more is sent.
    await vi.advanceTimersByTimeAsync(10 * 60_000)
    expect(report).toHaveBeenCalledTimes(2)
    reporter.stop()
  })

  it("respects a report another tab of the same login sent recently", async () => {
    const { reporter, report, fire } = setup(record({ lastReportAt: Date.now() - 20_000 }))
    fire("keydown")
    await vi.advanceTimersByTimeAsync(41_000)
    expect(report).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(1_000)
    expect(report).toHaveBeenCalledTimes(1)
    reporter.stop()
  })

  it("does not report for a server without idle support, and reports while the switch is off", async () => {
    const unsupported = setup(null)
    unsupported.fire("keydown")
    await vi.advanceTimersByTimeAsync(120_000)
    expect(unsupported.report).not.toHaveBeenCalled()
    unsupported.reporter.stop()

    const off = setup(record({ enforced: false }))
    off.fire("keydown")
    expect(off.report).toHaveBeenCalledTimes(1)
    off.reporter.stop()
  })

  it("retries after the interval when a report fails, without sending in a loop", async () => {
    const report = vi.fn(async () => false)
    const { reporter, fire } = setup(record(), { report })
    fire("keydown")
    await vi.advanceTimersByTimeAsync(0)
    expect(report).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(61_000)
    expect(report).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(1_000)
    expect(report).toHaveBeenCalledTimes(2)
    reporter.stop()
  })

  it("still sends a trailing report whose timer runs a few seconds late", async () => {
    const { reporter, report, fire } = setup()
    fire("keydown")
    await vi.advanceTimersByTimeAsync(0)
    fire("keydown")
    // The trailing timer is due 62 s after the report; it runs 4 s late.
    vi.setSystemTime(Date.now() + 4_000)
    await vi.advanceTimersByTimeAsync(62_000)
    expect(report).toHaveBeenCalledTimes(2)
    reporter.stop()
  })

  it("drops a trailing report when the device slept past the interval", async () => {
    const { reporter, report, fire } = setup()
    fire("keydown")
    await vi.advanceTimersByTimeAsync(10_000)
    fire("keydown")
    // The device sleeps for 20 minutes before the trailing timer can run.
    vi.setSystemTime(Date.now() + 20 * 60_000)
    await vi.advanceTimersByTimeAsync(60_000)
    expect(report).toHaveBeenCalledTimes(1)
    reporter.stop()
  })

  it("stops retrying a failed report once the input behind it is stale", async () => {
    const report = vi.fn(async () => false)
    const { reporter, fire } = setup(record(), { report })
    fire("keydown")
    await vi.advanceTimersByTimeAsync(40 * 60_000)
    // The first attempt and one retry within the interval; then the input is stale.
    expect(report).toHaveBeenCalledTimes(2)
    reporter.stop()
  })

  it("is not held back by a report time left in the future by a clock moved backwards", async () => {
    const { reporter, report, fire } = setup(record({ lastReportAt: Date.now() + 60 * 60_000 }))
    fire("keydown")
    await vi.advanceTimersByTimeAsync(0)
    expect(report).toHaveBeenCalledTimes(1)
    reporter.stop()
  })

  it("treats a report that throws like one that did not reach the server", async () => {
    const report = vi.fn(async () => { throw new Error("offline") })
    const { reporter, fire } = setup(record(), { report })
    fire("keydown")
    await vi.advanceTimersByTimeAsync(62_000)
    expect(report).toHaveBeenCalledTimes(2)
    reporter.stop()
  })

  it("reports immediately on an explicit continue, even inside the interval", async () => {
    const { reporter, report, fire } = setup()
    fire("keydown")
    await vi.advanceTimersByTimeAsync(5_000)
    await reporter.reportNow()
    expect(report).toHaveBeenCalledTimes(2)
    reporter.stop()
  })

  it("joins a report already in flight instead of sending a second one", async () => {
    let finish!: (value: boolean) => void
    const report = vi.fn(() => new Promise<boolean>(resolve => { finish = resolve }))
    const { reporter, fire } = setup(record(), { report })
    fire("pointerdown")
    const continued = reporter.reportNow()
    finish(true)
    await continued
    expect(report).toHaveBeenCalledTimes(1)
    reporter.stop()
  })

  it("stops listening and drops a pending trailing report when stopped", async () => {
    const { reporter, report, fire } = setup()
    fire("keydown")
    await vi.advanceTimersByTimeAsync(0)
    fire("keydown")
    reporter.stop()
    await vi.advanceTimersByTimeAsync(120_000)
    fire("keydown")
    expect(report).toHaveBeenCalledTimes(1)
  })
})

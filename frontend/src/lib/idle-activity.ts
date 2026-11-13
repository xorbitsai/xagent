import type { IdleSessionRecord } from "@/lib/idle-session"

/**
 * Deliberate input that keeps an idle session alive. Mouse movement and
 * scroll events are excluded on purpose: the browser can produce them while
 * nobody is at the keyboard (content moving under a resting cursor, streamed
 * output auto-scrolling), and only explicit actions count as activity.
 */
export const IDLE_ACTIVITY_EVENTS = ["keydown", "pointerdown", "wheel"] as const

/**
 * A trailing or retried report must stand for input that happened within
 * about one interval. Anything older (the device slept, the network was down,
 * the server failed) is dropped: an overdue timer must never renew the login.
 */
const STALE_INPUT_TOLERANCE_MS = 5_000
/**
 * The server measures its write interval between request arrivals. A report
 * sent exactly one interval after the previous send can arrive a few
 * milliseconds short of it and be dropped, so every other write would be lost.
 * Waiting this much longer keeps one write per interval.
 */
const SERVER_INTERVAL_MARGIN_MS = 2_000

export function isUserActivityEvent(event: Event): boolean {
  return event.isTrusted
}

export interface IdleActivityReporterOptions {
  target: Pick<Window, "addEventListener" | "removeEventListener">
  /** The current login's stored policy; null means the server has no idle support. */
  getRecord: () => IdleSessionRecord | null
  /** Records the report time for every tab of this login before the request is sent. */
  markReported: () => void
  /** Sends the activity report; resolves false when it did not reach the server. */
  report: () => Promise<boolean>
  isUserEvent?: (event: Event) => boolean
}

export interface IdleActivityReporter {
  /** An explicit "continue session": report now, regardless of the interval. */
  reportNow: () => Promise<void>
  stop: () => void
}

/**
 * Coalesces user input into at most one report per server interval, plus
 * one trailing report for input that arrived inside the interval.
 */
export function startIdleActivityReporter(options: IdleActivityReporterOptions): IdleActivityReporter {
  const isUserEvent = options.isUserEvent ?? isUserActivityEvent
  let pending = false
  let lastInputAt = Number.NEGATIVE_INFINITY
  let inFlight: Promise<void> | null = null
  let timer: ReturnType<typeof setTimeout> | null = null
  let stopped = false

  const clearTimer = () => {
    if (timer !== null) clearTimeout(timer)
    timer = null
  }
  const send = (): Promise<void> => {
    pending = false
    options.markReported()
    inFlight = options.report()
      .catch(() => false)
      .then(delivered => {
        inFlight = null
        if (!delivered) pending = true
        schedule()
      })
    return inFlight
  }
  const schedule = () => {
    if (stopped || !pending || inFlight || timer !== null) return
    const record = options.getRecord()
    if (!record) {
      pending = false
      return
    }
    const intervalMs = record.activityMinIntervalSeconds * 1000
    const now = Date.now()
    // A normal trailing report runs interval + margin after the last report.
    if (now - lastInputAt > intervalMs + SERVER_INTERVAL_MARGIN_MS + STALE_INPUT_TOLERANCE_MS) {
      pending = false
      return
    }
    // A report time in the future means the clock moved backwards; ignore it.
    const lastReportAt = record.lastReportAt !== null && record.lastReportAt <= now ? record.lastReportAt : 0
    const wait = lastReportAt === 0 ? 0 : lastReportAt + intervalMs + SERVER_INTERVAL_MARGIN_MS - now
    if (wait > 0) {
      timer = setTimeout(() => { timer = null; schedule() }, wait)
      return
    }
    void send()
  }
  const onEvent = (event: Event) => {
    if (!isUserEvent(event)) return
    lastInputAt = Date.now()
    pending = true
    schedule()
  }

  for (const type of IDLE_ACTIVITY_EVENTS) options.target.addEventListener(type, onEvent, { capture: true, passive: true })
  return {
    reportNow: async () => {
      if (stopped || !options.getRecord()) return
      clearTimer()
      // The press on the continue button may already have started a report.
      if (inFlight) return inFlight
      await send()
    },
    stop: () => {
      stopped = true
      clearTimer()
      for (const type of IDLE_ACTIVITY_EVENTS) options.target.removeEventListener(type, onEvent, { capture: true })
    },
  }
}

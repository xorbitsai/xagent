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
  markReported: () => void | Promise<void>
  /**
   * Sends the activity report; resolves false when it did not reach the server.
   * `isStillFresh` tells whether the input is still recent, so a report that
   * waited (for example for a token refresh) can be dropped before a replay.
   */
  report: (isStillFresh: () => boolean) => Promise<boolean>
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
  // This tab's own last attempt, so retries stay paced even when the shared
  // report time cannot be written.
  let lastAttemptAt = Number.NEGATIVE_INFINITY
  let inFlight: Promise<void> | null = null
  let timer: ReturnType<typeof setTimeout> | null = null
  let stopped = false

  const clearTimer = () => {
    if (timer !== null) clearTimeout(timer)
    timer = null
  }
  const freshnessWindowMs = (record: IdleSessionRecord) =>
    record.activityMinIntervalSeconds * 1000 + SERVER_INTERVAL_MARGIN_MS + STALE_INPUT_TOLERANCE_MS
  // A negative age means the clock moved backwards since the input, so its
  // real age is unknown and it is treated as stale.
  const inputIsFresh = (record: IdleSessionRecord, now: number) => {
    const inputAge = now - lastInputAt
    return inputAge >= 0 && inputAge <= freshnessWindowMs(record)
  }
  const send = (): Promise<void> => {
    pending = false
    lastAttemptAt = Date.now()
    inFlight = Promise.resolve()
      .then(() => options.markReported())
      .catch(() => undefined)
      .then(() => {
        // The page may have been suspended while waiting for the auth lock:
        // check again that the input is still recent before reporting it.
        const isStillFresh = () => {
          const record = options.getRecord()
          return record !== null && inputIsFresh(record, Date.now())
        }
        if (!isStillFresh()) return true
        return options.report(isStillFresh)
      })
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
    if (!inputIsFresh(record, now)) {
      pending = false
      return
    }
    // Times in the future mean the clock moved backwards; ignore them.
    const sharedLast = record.lastReportAt !== null && record.lastReportAt <= now ? record.lastReportAt : 0
    const localLast = lastAttemptAt <= now ? lastAttemptAt : 0
    const lastReportAt = Math.max(sharedLast, localLast)
    const wait = lastReportAt <= 0 ? 0 : lastReportAt + intervalMs + SERVER_INTERVAL_MARGIN_MS - now
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
      // The press on "Continue session" is itself recent input.
      lastInputAt = Date.now()
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

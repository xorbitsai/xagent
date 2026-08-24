/**
 * Time formatting utility functions
 * Unified handling of timestamp and ISO format time display
 */
import type { Locale } from "@/i18n/translations"
import type { Translate } from "@/contexts/i18n-context"

// Partial with a lookup fallback: distributions may replace @/i18n/translations
// with a widened Locale union, and any unmapped locale is itself a BCP 47 tag.
const displayDateLocales: Partial<Record<Locale, string>> = {
  en: "en-US",
  zh: "zh-CN",
}

/** Formats a non-blank, parseable display-date string, returning empty text for invalid input. */
export function formatDisplayDate(
  value: unknown,
  locale: Locale,
  options: Intl.DateTimeFormatOptions,
): string {
  if (typeof value !== "string" || value.trim() === "") return ""

  const date = new Date(value)
  if (!Number.isFinite(date.getTime())) return ""

  return new Intl.DateTimeFormat(displayDateLocales[locale] ?? locale, options).format(date)
}

/**
 * Normalizes various timestamp formats (seconds, milliseconds, numeric strings, ISO strings)
 * to a standard millisecond timestamp.
 * @param ts The timestamp to normalize
 * @returns Timestamp in milliseconds
 */
export function normalizeTimestampMs(ts?: string | number | Date | null): number {
  // Checked for exact absence, not truthiness: epoch zero (`0`) is a valid,
  // if unlikely, timestamp - a `!ts` check would collapse it into "now" the
  // same way it does undefined/null/"", silently discarding a real value.
  //
  // Deliberately NOT special-casing NaN/Infinity here (a prior version of
  // this function did, then a PR review caught that it was a regression):
  // this function's callers (formatTime, getTimeDuration) already guard
  // against an invalid result via `isNaN(date.getTime())` - new Date(NaN)
  // and new Date(Infinity) both produce an actually-invalid Date, which
  // those checks correctly treat as "malformed input" and fall back to a
  // safe default. Converting NaN/Infinity into Date.now() here instead would
  // make a malformed timestamp look like a perfectly valid "just now" to
  // every caller, defeating that existing downstream safety net. Callers
  // that need non-finite values to look like "absent" for their own UI
  // purposes (e.g. progress-panel.tsx's hasTimestamp) should filter them out
  // themselves before ever calling this function, as that one already does.
  if (ts === undefined || ts === null || ts === '') return Date.now()
  if (ts instanceof Date) return ts.getTime()

  if (typeof ts === 'string') {
    const num = Number(ts)
    // If it's a valid number string (not empty), treat as numeric timestamp
    if (!isNaN(num) && ts.trim() !== '') {
      return num < 1e10 ? num * 1000 : num
    }
    // Otherwise try parsing as date string
    const parsedMs = new Date(ts).getTime()
    return isNaN(parsedMs) ? Date.now() : parsedMs
  }

  if (typeof ts === 'number') {
    return ts < 1e10 ? ts * 1000 : ts
  }

  return Date.now()
}

/** Formats a timestamp as "N min/hr/days ago". Malformed/absent input
 * normalizes to "now" (see `normalizeTimestampMs`'s own convention), so
 * this renders "Just now" rather than a raw "NaN ago" in that case. */
export function formatRelativeTime(
  value: string | number | Date | null | undefined,
  t: Translate,
): string {
  const diff = Date.now() - normalizeTimestampMs(value)
  const minute = 60 * 1000
  const hour = 60 * minute
  const day = 24 * hour
  const month = 30 * day
  const year = 365 * day

  if (diff < minute) return t("common.time.justNow")
  if (diff < hour) return t("common.time.minsAgo", { count: Math.floor(diff / minute) })
  if (diff < day) return t("common.time.hoursAgo", { count: Math.floor(diff / hour) })
  if (diff < month) return t("common.time.daysAgo", { count: Math.floor(diff / day) })
  if (diff < year) return t("common.time.monthsAgo", { count: Math.floor(diff / month) })
  return t("common.time.yearsAgo", { count: Math.floor(diff / year) })
}

/**
 * Format time to local time string
 * @param timestamp Timestamp (seconds or milliseconds) or ISO string
 * @param format Output format: 'time' | 'date' | 'datetime'
 * @returns Formatted time string
 */
export function formatTime(
  timestamp: string | number | null | undefined,
  format: 'time' | 'date' | 'datetime' = 'time'
): string {
  if (!timestamp) {
    return ''
  }

  try {
    const date = new Date(normalizeTimestampMs(timestamp))

    if (isNaN(date.getTime())) {
      return String(timestamp)
    }

    switch (format) {
      case 'time':
        return date.toLocaleTimeString()
      case 'date':
        return date.toLocaleDateString()
      case 'datetime':
        return date.toLocaleString()
      default:
        return date.toLocaleTimeString()
    }
  } catch {
    return String(timestamp)
  }
}

/**
 * Calculate time duration
 * @param start Start time (timestamp or ISO string)
 * @param end End time (timestamp or ISO string)
 * @returns Time duration (milliseconds)
 */
export function getTimeDuration(
  start: string | number | null | undefined,
  end: string | number | null | undefined
): number {
  if (!start || !end) {
    return 0
  }

  try {
    const startDate = new Date(normalizeTimestampMs(start))
    const endDate = new Date(normalizeTimestampMs(end))

    if (isNaN(startDate.getTime()) || isNaN(endDate.getTime())) {
      return 0
    }

    return endDate.getTime() - startDate.getTime()
  } catch {
    return 0
  }
}

/**
 * Format time duration
 * @param duration Time duration (milliseconds)
 * @returns Formatted time string (e.g., 1h 30m 25s)
 */
export function formatDuration(duration: number): string {
  if (duration <= 0) {
    return '0s'
  }

  const seconds = Math.floor(duration / 1000)
  const minutes = Math.floor(seconds / 60)
  const hours = Math.floor(minutes / 60)
  const days = Math.floor(hours / 24)

  const parts: string[] = []

  if (days > 0) {
    parts.push(`${days}d`)
  }
  if (hours % 24 > 0) {
    parts.push(`${hours % 24}h`)
  }
  if (minutes % 60 > 0) {
    parts.push(`${minutes % 60}m`)
  }
  if (seconds % 60 > 0) {
    parts.push(`${seconds % 60}s`)
  }

  return parts.join(' ') || '0s'
}

/**
 * Get current timestamp (seconds)
 * @returns Current timestamp
 */
export function getCurrentTimestamp(): number {
  return Math.floor(Date.now() / 1000)
}

/**
 * Check if time is expired
 * @param timestamp Timestamp (seconds)
 * @param seconds Expiration time (seconds)
 * @returns Whether it is expired
 */
export function isExpired(timestamp: number, seconds: number): boolean {
  return getCurrentTimestamp() - timestamp > seconds
}

import { readFileSync } from "node:fs"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { Locale } from "@/i18n/translations"
import { formatDisplayDate, formatTime, getTimeDuration, normalizeTimestampMs } from "./time-utils"

const bigintValue = (globalThis as { BigInt: (value: number) => bigint }).BigInt(1)

const dateOptions = Object.freeze({
  year: "numeric",
  month: "numeric",
  day: "numeric",
} as const)

const homeDateOptions = Object.freeze({
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
} as const)

describe("formatDisplayDate", () => {
  let priorTimeZone: string | undefined

  beforeEach(() => {
    priorTimeZone = process.env.TZ
    process.env.TZ = "UTC"
  })

  afterEach(() => {
    vi.restoreAllMocks()
    if (priorTimeZone === undefined) {
      delete process.env.TZ
    } else {
      process.env.TZ = priorTimeZone
    }
  })

  it.each([
    undefined,
    null,
    1,
    new Date(),
    {},
    [],
    true,
    new String("2024-05-06T07:08:09Z"),
    bigintValue,
    Symbol("date"),
    () => "2024-05-06T07:08:09Z",
    "",
    "   ",
    "not a date",
  ])("returns empty for unsupported input %#", (value) => {
    expect(() => formatDisplayDate(value, "en", dateOptions)).not.toThrow()
    expect(formatDisplayDate(value, "en", dateOptions)).toBe("")
  })

  it.each([
    ["en", "en-US"],
    ["zh", "zh-CN"],
  ] as const)("uses %s locale with each caller option contract", (locale, expectedLocale) => {
    const formatSpy = vi.spyOn(Intl, "DateTimeFormat")
    const buildDate = formatDisplayDate("2024-05-06T07:08:09Z", locale, dateOptions)
    const homeDate = formatDisplayDate("2024-05-06T07:08:09Z", locale, homeDateOptions)

    expect(buildDate).not.toBe("")
    expect(homeDate).not.toBe("")
    expect(formatSpy).toHaveBeenNthCalledWith(1, expectedLocale, dateOptions)
    expect(formatSpy).toHaveBeenNthCalledWith(2, expectedLocale, homeDateOptions)
  })

  it("falls back to the locale tag itself for locales without an explicit mapping", () => {
    const formatSpy = vi.spyOn(Intl, "DateTimeFormat")
    const widenedLocale = "vi" as unknown as Locale

    const formatted = formatDisplayDate("2024-05-06T07:08:09Z", widenedLocale, dateOptions)

    expect(formatted).not.toBe("")
    expect(formatSpy).toHaveBeenCalledWith("vi", dateOptions)

    const source = readFileSync("src/lib/time-utils.ts", "utf8")
    expect(source).toContain("const displayDateLocales: Partial<Record<Locale, string>> = {")
  })

  it("formats a cross-day timestamp in the fixed UTC test environment", () => {
    const timestamp = "2024-05-06T23:30:00Z"
    const expected = new Intl.DateTimeFormat("en-US", {
      ...dateOptions,
      timeZone: "UTC",
    }).format(new Date(timestamp))

    expect(formatDisplayDate(timestamp, "en", dateOptions)).toBe(expected)

    const homeExpected = new Intl.DateTimeFormat("en-US", {
      ...homeDateOptions,
      timeZone: "UTC",
    }).format(new Date(timestamp))
    expect(formatDisplayDate(timestamp, "en", homeDateOptions)).toBe(homeExpected)

    const offsetTimestamp = "2024-05-07T01:30:00+02:00"
    const offsetExpected = new Intl.DateTimeFormat("en-US", {
      ...homeDateOptions,
      timeZone: "UTC",
    }).format(new Date(offsetTimestamp))
    expect(formatDisplayDate(offsetTimestamp, "en", homeDateOptions)).toBe(offsetExpected)
  })

  it("does not reuse permissive legacy formatting owners", () => {
    const source = readFileSync("src/lib/time-utils.ts", "utf8")
    const start = source.indexOf("export function formatDisplayDate")
    const end = source.indexOf("/**\n * Normalizes", start)
    expect(start).toBeGreaterThanOrEqual(0)
    expect(end).toBeGreaterThan(start)
    const helper = source.slice(start, end)
    expect(helper).toContain("const date = new Date(value)")
    expect(helper).not.toMatch(/normalizeTimestampMs|formatTime|toLocaleDateString|utils\.formatDate/)
  })
})

describe("normalizeTimestampMs", () => {
  it("treats epoch zero as a present timestamp, not an absent one", () => {
    expect(normalizeTimestampMs(0)).toBe(0)
    // The string form takes a completely different branch (numeric-string
    // parsing, not the absence check) - assert it independently rather than
    // assuming it agrees with the numeric case.
    expect(normalizeTimestampMs("0")).toBe(0)
  })

  it("falls back to now for genuinely absent values", () => {
    const before = Date.now()
    expect(normalizeTimestampMs(undefined)).toBeGreaterThanOrEqual(before)
    expect(normalizeTimestampMs(null)).toBeGreaterThanOrEqual(before)
    expect(normalizeTimestampMs("")).toBeGreaterThanOrEqual(before)
  })

  it("propagates NaN/Infinity unchanged rather than disguising them as a valid now", () => {
    // A prior version of this function collapsed non-finite numbers into
    // Date.now(), which made a malformed timestamp look like a perfectly
    // valid "just now" to every caller - regressing the safety net formatTime/
    // getTimeDuration already have via `isNaN(date.getTime())`. Propagating
    // the non-finite value unchanged lets `new Date(...)` produce an
    // actually-invalid Date, which those callers correctly detect.
    expect(normalizeTimestampMs(NaN)).toBeNaN()
    expect(normalizeTimestampMs(Infinity)).toBe(Infinity)
    expect(normalizeTimestampMs(-Infinity)).toBe(-Infinity)
    expect(normalizeTimestampMs("Infinity")).toBe(Infinity)
    expect(normalizeTimestampMs("-Infinity")).toBe(-Infinity)
  })

  it("formatTime and getTimeDuration safely fall back for a non-finite timestamp instead of showing a fake current time", () => {
    expect(formatTime(Infinity)).toBe(String(Infinity))
    expect(formatTime(NaN)).toBe("")
    expect(getTimeDuration(Infinity, Infinity)).toBe(0)
    expect(getTimeDuration(0, NaN)).toBe(0)
  })
})

"use client"

import React, { useMemo } from "react"
import { CalendarDays } from "lucide-react"

import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select } from "@/components/ui/select"
import type { Translate } from "@/contexts/i18n-context"
import { cn } from "@/lib/utils"

export type ScheduleRecurrence = "hourly" | "daily" | "weekly" | "monthly" | "custom"
export type ScheduleCustomUnit = "minutes" | "hours" | "days"

// Fields consumed by the schedule recurrence editor. Kept as a narrow slice
// (rather than importing the dialog's full TriggerFormState) so this file has
// no dependency on the dialog module.
export interface ScheduleFieldsValue {
  recurrence: ScheduleRecurrence
  timeOfDay: string
  weekdays: number[]
  dayOfMonth: number
  customAmount: string
  customUnit: ScheduleCustomUnit
  startDate: string
  // IANA name (e.g. "Asia/Shanghai"). Only daily/weekly/monthly are actually
  // computed in this zone server-side (hourly/custom are a flat interval
  // with no civil time-of-day for a zone to apply to) — shown regardless of
  // recurrence so switching into a tz-aware one doesn't silently start from
  // an unset zone.
  timezone: string
}

interface ScheduleFieldsProps {
  value: ScheduleFieldsValue
  onChange: <K extends keyof ScheduleFieldsValue>(key: K, value: ScheduleFieldsValue[K]) => void
  t: Translate
  locale: string
}

const FIELD_LABEL_CLASS = "text-xs font-semibold text-muted-foreground"
const WEEKDAY_KEYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] as const
export const RECURRENCE_TYPES: ScheduleRecurrence[] = ["hourly", "daily", "weekly", "monthly", "custom"]

/** Shared selected/unselected styling for the recurrence and weekday chips. */
function chipClass(selected: boolean): string {
  return cn(
    "rounded-full border px-3 py-1 text-xs font-medium transition-colors",
    selected
      ? "border-primary bg-primary/10 text-primary"
      : "bg-background text-muted-foreground hover:border-primary/50 hover:text-foreground",
  )
}

const ORDINAL_SUFFIXES: Record<string, string> = { one: "st", two: "nd", few: "rd", other: "th" }

function ordinalDay(day: number, locale?: string): string {
  const loc = locale || "en"
  if (loc.startsWith("zh")) return `${day}日`
  const category = new Intl.PluralRules(loc, { type: "ordinal" }).select(day)
  return `${day}${ORDINAL_SUFFIXES[category] ?? "th"}`
}

function formatTimeLabel(hhmm: string, locale: string): string {
  const [hour, minute] = (hhmm || "00:00").split(":").map((part) => Number(part))
  if (Number.isNaN(hour) || Number.isNaN(minute)) return hhmm
  return new Intl.DateTimeFormat(locale, { hour: "numeric", minute: "2-digit" }).format(
    new Date(2000, 0, 1, hour, minute),
  )
}

function formatDateLabel(dateStr: string, locale: string): string {
  const date = new Date(`${dateStr}T00:00:00`)
  if (Number.isNaN(date.getTime())) return dateStr
  return new Intl.DateTimeFormat(locale, { month: "short", day: "numeric", year: "numeric" }).format(
    date,
  )
}

const CUSTOM_UNIT_KEYS: Record<ScheduleCustomUnit, string> = {
  minutes: "triggers.schedule.customUnitMinutes",
  hours: "triggers.schedule.customUnitHours",
  days: "triggers.schedule.customUnitDays",
}

/** Pure "Runs every X - starts …" preview line, shared by the form and (once
 * saved) could be reused for a read-only summary. */
export function summarizeSchedule(
  form: ScheduleFieldsValue,
  t: Translate,
  locale: string,
): string {
  const time = formatTimeLabel(form.timeOfDay, locale)
  let base: string
  if (form.recurrence === "daily") {
    base = t("triggers.schedule.summaryDaily", { time })
  } else if (form.recurrence === "weekly") {
    const days = [...form.weekdays]
      .sort((a, b) => a - b)
      .map((day) => t(`triggers.schedule.weekday${WEEKDAY_KEYS[day]}` as never))
      .join(", ")
    base = t("triggers.schedule.summaryWeekly", { days: days || "-", time })
  } else if (form.recurrence === "monthly") {
    base = t("triggers.schedule.summaryMonthly", {
      day: ordinalDay(form.dayOfMonth, locale),
      time,
    })
  } else if (form.recurrence === "custom") {
    base = t("triggers.schedule.summaryCustom", {
      amount: form.customAmount || "0",
      unit: t(CUSTOM_UNIT_KEYS[form.customUnit] as never),
    })
  } else {
    base = t("triggers.schedule.summaryHourly")
  }

  if (form.startDate) {
    const date = formatDateLabel(form.startDate, locale)
    base =
      form.recurrence === "hourly" || form.recurrence === "custom"
        ? t("triggers.schedule.summaryStartsWithTime", { base, date, time })
        : t("triggers.schedule.summaryStartsOnly", { base, date })
  }

  // Only daily/weekly/monthly are actually computed in this zone
  // server-side; appending it to hourly/custom would imply a dependency
  // that doesn't exist.
  if (form.recurrence === "hourly" || form.recurrence === "custom") return base
  return t("triggers.schedule.summaryWithTimezone", { base, timezone: form.timezone })
}

/** A Date's LOCAL calendar date as YYYY-MM-DD (what <input type="date"> expects). */
export function localIsoDate(date: Date): string {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(
    date.getDate(),
  ).padStart(2, "0")}`
}

/** A Date's LOCAL wall-clock time as HH:MM (what <input type="time"> expects). */
export function localTimeOfDay(date: Date): string {
  return `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`
}

function datePartsInZone(date: Date, timeZone: string, options: Intl.DateTimeFormatOptions) {
  const parts = new Intl.DateTimeFormat("en-US", { ...options, timeZone }).formatToParts(date)
  return (type: string) => parts.find((part) => part.type === type)?.value ?? "00"
}

/** A Date's calendar date in an ARBITRARY IANA zone, as YYYY-MM-DD. Unlike
 * `localIsoDate`, this doesn't depend on the browser's own zone — needed to
 * show a daily/weekly/monthly trigger's stored legacy anchor (a full ISO
 * instant predating the plain-date `start_at` format) on the calendar date
 * it actually starts on, not whatever date that instant happens to fall on
 * in the zone of whoever is currently editing it. */
export function zonedIsoDate(date: Date, timeZone: string): string {
  const part = datePartsInZone(date, timeZone, { year: "numeric", month: "2-digit", day: "2-digit" })
  return `${part("year")}-${part("month")}-${part("day")}`
}

/** A Date's wall-clock time in an ARBITRARY IANA zone, as HH:MM. See
 * `zonedIsoDate`. */
export function zonedTimeOfDay(date: Date, timeZone: string): string {
  const part = datePartsInZone(date, timeZone, { hour: "2-digit", minute: "2-digit", hour12: false })
  // hour12: false can still render midnight as "24" in some locales/engines.
  const hour = part("hour") === "24" ? "00" : part("hour")
  return `${hour}:${part("minute")}`
}

export function scheduleFieldsDefaults(): ScheduleFieldsValue {
  const iso = localIsoDate(new Date())
  return {
    recurrence: "hourly",
    timeOfDay: "09:00",
    weekdays: [0],
    dayOfMonth: 1,
    customAmount: "30",
    customUnit: "minutes",
    // The start date is always shown (reference design), defaulting to today.
    startDate: iso,
    // A brand-new schedule has nothing stored yet — default to the editing
    // browser's own zone. An EXISTING trigger's stored timezone must be
    // preserved instead of re-derived here on every edit (see
    // formFromTrigger in agent-triggers-dialog.tsx): re-deriving it would
    // silently relocate the schedule when edited from a different zone than
    // it was created in.
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
  }
}

export function ScheduleFields({ value, onChange, t, locale }: ScheduleFieldsProps) {
  const dayOfMonthOptions = useMemo(
    () =>
      Array.from({ length: 31 }, (_, index) => index + 1).map((day) => ({
        value: String(day),
        label: ordinalDay(day, locale),
      })),
    [locale],
  )
  const customUnitOptions = useMemo(
    () => [
      { value: "minutes", label: t("triggers.schedule.customUnitMinutes") },
      { value: "hours", label: t("triggers.schedule.customUnitHours") },
      { value: "days", label: t("triggers.schedule.customUnitDays") },
    ],
    [t],
  )

  const toggleWeekday = (day: number) => {
    const next = value.weekdays.includes(day)
      ? value.weekdays.filter((item) => item !== day)
      : [...value.weekdays, day].sort((a, b) => a - b)
    onChange("weekdays", next)
  }

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label className={FIELD_LABEL_CLASS}>{t("triggers.schedule.recurrenceLabel")}</Label>
        <div className="flex flex-wrap gap-1.5">
          {RECURRENCE_TYPES.map((type) => (
            <button
              key={type}
              type="button"
              onClick={() => onChange("recurrence", type)}
              className={chipClass(value.recurrence === type)}
            >
              {t(`triggers.schedule.${type}` as never)}
            </button>
          ))}
        </div>
      </div>

      {value.recurrence === "weekly" && (
        <div className="space-y-2">
          <Label className={FIELD_LABEL_CLASS}>{t("triggers.schedule.onWhichDays")}</Label>
          <div className="flex flex-wrap gap-1.5">
            {WEEKDAY_KEYS.map((key, index) => (
              <button
                key={key}
                type="button"
                onClick={() => toggleWeekday(index)}
                className={chipClass(value.weekdays.includes(index))}
              >
                {t(`triggers.schedule.weekday${key}` as never)}
              </button>
            ))}
          </div>
        </div>
      )}

      {value.recurrence === "monthly" && (
        <div className="space-y-2">
          <Label className={FIELD_LABEL_CLASS} id="schedule-day-of-month-label">
            {t("triggers.schedule.onWhichDayOfMonth")}
          </Label>
          <div aria-labelledby="schedule-day-of-month-label">
            <Select
              value={String(value.dayOfMonth)}
              onValueChange={(next) => onChange("dayOfMonth", Number(next))}
              options={dayOfMonthOptions}
            />
          </div>
        </div>
      )}

      {/* Hourly and custom schedules take their time from the start row's
          "from" input instead (reference design). */}
      {value.recurrence === "custom" ? (
        <div className="space-y-2">
          <Label className={FIELD_LABEL_CLASS} htmlFor="schedule-custom-amount">
            {t("triggers.schedule.runEvery")}
          </Label>
          <div className="flex gap-2">
            <Input
              id="schedule-custom-amount"
              type="number"
              min={1}
              className="h-10 w-24"
              value={value.customAmount}
              onChange={(event) => onChange("customAmount", event.target.value)}
            />
            <div className="flex-1">
              <Select
                value={value.customUnit}
                onValueChange={(next) => onChange("customUnit", next as ScheduleCustomUnit)}
                options={customUnitOptions}
              />
            </div>
          </div>
          <p className="text-xs text-muted-foreground">{t("triggers.schedule.customHelp")}</p>
        </div>
      ) : value.recurrence !== "hourly" ? (
        <div className="space-y-2">
          <Label className={FIELD_LABEL_CLASS} htmlFor="schedule-time-of-day">
            {t("triggers.schedule.atWhatTime")}
          </Label>
          <Input
            id="schedule-time-of-day"
            type="time"
            value={value.timeOfDay}
            onChange={(event) => onChange("timeOfDay", event.target.value)}
            className="w-40"
          />
          {/* Read-only: this is the zone the schedule was created in (or the
              browser's own, for a new one) — not a picker, since changing it
              would silently relocate an already-armed schedule. */}
          <p className="text-xs text-muted-foreground" data-testid="schedule-timezone-label">
            {t("triggers.schedule.timezoneLabel", { timezone: value.timezone })}
          </p>
        </div>
      ) : null}

      <div className="space-y-2">
        <Label
          htmlFor="schedule-start-date"
          className={cn(FIELD_LABEL_CLASS, "flex items-center gap-1.5")}
        >
          <CalendarDays className="h-3.5 w-3.5" />
          {t("triggers.schedule.startCheckbox")}
        </Label>
        <div className="flex items-center gap-2">
          <Input
            id="schedule-start-date"
            type="date"
            value={value.startDate}
            onChange={(event) => onChange("startDate", event.target.value)}
            className="w-40"
          />
          {(value.recurrence === "hourly" || value.recurrence === "custom") && (
            <>
              <span className="text-xs text-muted-foreground">
                {t("triggers.schedule.startFrom")}
              </span>
              <Input
                type="time"
                value={value.timeOfDay}
                onChange={(event) => onChange("timeOfDay", event.target.value)}
                className="w-32"
              />
            </>
          )}
        </div>
      </div>
    </div>
  )
}

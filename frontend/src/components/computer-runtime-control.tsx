"use client"

import { Check, ChevronDown, ExternalLink, Globe2, Monitor } from "lucide-react"
import Link from "next/link"
import React, { useEffect, useState } from "react"

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { useI18n } from "@/contexts/i18n-context"
import { apiRequest } from "@/lib/api-wrapper"
import { cn, getApiUrl } from "@/lib/utils"

export type ComputerRuntimeKind = "extension_relay" | "desktop_relay"

export const DEFAULT_COMPUTER_RUNTIME_KIND: ComputerRuntimeKind =
  "extension_relay"

const STORAGE_KEY = "xagent.computerRuntimeKind"
const READINESS_POLL_INTERVAL_MS = 3_000

type ComputerReadinessIssueCode =
  | "disconnected"
  | "not_attached"
  | "screen_recording_permission_missing"
  | "accessibility_permission_missing"
  | "paused"
  | "emergency_stopped"

export interface ComputerTargetReadiness {
  runtime_kind: ComputerRuntimeKind
  ready: boolean
  connected: boolean
  attached: boolean
  issues: Array<{
    code: ComputerReadinessIssueCode
    message: string
  }>
}

interface ComputerReadinessResponse {
  targets: Record<ComputerRuntimeKind, ComputerTargetReadiness>
}

export function getStoredComputerRuntimeKind(): ComputerRuntimeKind {
  if (typeof window === "undefined") {
    return DEFAULT_COMPUTER_RUNTIME_KIND
  }
  try {
    return window.localStorage.getItem(STORAGE_KEY) === "desktop_relay"
      ? "desktop_relay"
      : DEFAULT_COMPUTER_RUNTIME_KIND
  } catch {
    return DEFAULT_COMPUTER_RUNTIME_KIND
  }
}

export function storeComputerRuntimeKind(value: ComputerRuntimeKind): void {
  if (typeof window !== "undefined") {
    try {
      window.localStorage.setItem(STORAGE_KEY, value)
    } catch {
      // Storage can be unavailable in hardened/private browser contexts.
    }
  }
}

interface ComputerRuntimeControlProps {
  value: ComputerRuntimeKind
  onValueChange?: (value: ComputerRuntimeKind) => void
  disabled?: boolean
  dark?: boolean
  className?: string
}

export function ComputerRuntimeControl({
  value,
  onValueChange,
  disabled = false,
  dark = false,
  className,
}: ComputerRuntimeControlProps) {
  const { t } = useI18n()
  const [open, setOpen] = useState(false)
  const [readiness, setReadiness] =
    useState<ComputerReadinessResponse["targets"] | null>(null)
  const selectionLocked = disabled || !onValueChange
  const isDesktop = value === "desktop_relay"
  const Icon = isDesktop ? Monitor : Globe2
  const label = isDesktop
    ? t("computerRuntime.desktop.label")
    : t("computerRuntime.browser.label")
  const selectedReadiness = readiness?.[value]

  useEffect(() => {
    let active = true

    const refresh = async () => {
      try {
        const response = await apiRequest(
          `${getApiUrl()}/api/computer/readiness`
        )
        if (!response.ok) return
        const payload = (await response.json()) as Partial<ComputerReadinessResponse>
        if (
          active &&
          payload.targets?.extension_relay &&
          payload.targets?.desktop_relay
        ) {
          setReadiness(payload.targets)
        }
      } catch {
        // Keep the last known snapshot through short API/network interruptions.
      }
    }

    void refresh()
    const timer = window.setInterval(() => void refresh(), READINESS_POLL_INTERVAL_MS)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [])

  const trigger = (
    <button
      type="button"
      aria-label={label}
      className={cn(
        "inline-flex h-9 items-center justify-center gap-2 whitespace-nowrap rounded-xl px-3 text-xs font-normal transition-colors",
        dark
          ? "text-white/75 hover:bg-[hsl(234_30%_35%)] hover:text-white"
          : "text-muted-foreground hover:bg-secondary/80 hover:text-foreground",
        selectionLocked && "opacity-80",
        className
      )}
      title={
        disabled
          ? t("computerRuntime.boundHint", { target: label })
          : t("computerRuntime.title")
      }
    >
      <Icon className="h-4 w-4" />
      <span className="hidden sm:inline-block">{label}</span>
      <ReadinessDot
        readiness={selectedReadiness}
        readyLabel={t("computerRuntime.status.ready")}
        attentionLabel={t("computerRuntime.status.needsAttention")}
        checkingLabel={t("computerRuntime.status.checking")}
      />
      <ChevronDown className="h-3.5 w-3.5 opacity-60" />
    </button>
  )

  const options: Array<{
    value: ComputerRuntimeKind
    label: string
    description: string
    icon: typeof Globe2
    settingsHash: string
  }> = [
    {
      value: "extension_relay",
      label: t("computerRuntime.browser.label"),
      description: t("computerRuntime.browser.description"),
      icon: Globe2,
      settingsHash: "browser-relay",
    },
    {
      value: "desktop_relay",
      label: t("computerRuntime.desktop.label"),
      description: t("computerRuntime.desktop.description"),
      icon: Monitor,
      settingsHash: "desktop-relay",
    },
  ]

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>{trigger}</PopoverTrigger>
      <PopoverContent align="start" side="top" className="w-80 p-2">
        <div className="px-2 pb-2 pt-1">
          <div className="text-sm font-medium">{t("computerRuntime.title")}</div>
          <p className="mt-1 text-xs text-muted-foreground">
            {t("computerRuntime.description")}
          </p>
        </div>
        <div className="space-y-1">
          {options.map((option) => {
            const OptionIcon = option.icon
            const selected = option.value === value
            const optionReadiness = readiness?.[option.value]
            const firstIssue = optionReadiness?.issues[0]
            return (
              <button
                key={option.value}
                type="button"
                disabled={selectionLocked}
                className={cn(
                  "flex w-full items-start gap-3 rounded-lg px-3 py-2.5 text-left hover:bg-accent",
                  selected && "bg-accent",
                  selectionLocked && !selected && "opacity-45"
                )}
                onClick={() => {
                  if (!onValueChange || selectionLocked) return
                  onValueChange(option.value)
                  storeComputerRuntimeKind(option.value)
                  setOpen(false)
                }}
              >
                <OptionIcon className="mt-0.5 h-4 w-4 shrink-0" />
                <span className="min-w-0 flex-1">
                  <span className="block text-sm font-medium">{option.label}</span>
                  <span className="mt-0.5 block text-xs text-muted-foreground">
                    {option.description}
                  </span>
                  <span
                    className={cn(
                      "mt-1.5 block text-xs",
                      optionReadiness?.ready
                        ? "text-emerald-600"
                        : "text-amber-600"
                    )}
                  >
                    {optionReadiness
                      ? optionReadiness.ready
                        ? t("computerRuntime.status.ready")
                        : firstIssue
                          ? t(`computerRuntime.issues.${firstIssue.code}`)
                          : t("computerRuntime.status.needsAttention")
                      : t("computerRuntime.status.checking")}
                  </span>
                </span>
                {selected && <Check className="mt-0.5 h-4 w-4 text-primary" />}
              </button>
            )
          })}
        </div>
        <div className="mt-2 border-t px-2 pt-2">
          <Link
            href={`/settings#${
              options.find((option) => option.value === value)?.settingsHash
            }`}
            className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
            onClick={() => setOpen(false)}
          >
            {t("computerRuntime.manageConnections")}
            <ExternalLink className="h-3 w-3" />
          </Link>
        </div>
      </PopoverContent>
    </Popover>
  )
}

function ReadinessDot({
  readiness,
  readyLabel,
  attentionLabel,
  checkingLabel,
}: {
  readiness: ComputerTargetReadiness | undefined
  readyLabel: string
  attentionLabel: string
  checkingLabel: string
}) {
  const label = readiness
    ? readiness.ready
      ? readyLabel
      : attentionLabel
    : checkingLabel
  return (
    <span
      role="status"
      aria-label={label}
      title={label}
      className={cn(
        "h-2 w-2 shrink-0 rounded-full",
        readiness
          ? readiness.ready
            ? "bg-emerald-500"
            : "bg-amber-500"
          : "bg-muted-foreground/35"
      )}
    />
  )
}

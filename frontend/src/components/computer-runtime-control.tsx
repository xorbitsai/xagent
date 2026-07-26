"use client"

import {
  Check,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  FilePlus2,
  Globe2,
  Monitor,
  Plus,
  X,
} from "lucide-react"
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

interface ComposerAddMenuProps {
  value?: ComputerRuntimeKind
  onValueChange?: (value: ComputerRuntimeKind | undefined) => void
  onAddFiles?: () => void
  showFileUpload?: boolean
  disabled?: boolean
  selectionLocked?: boolean
  dark?: boolean
  className?: string
}

/**
 * Optional composer capabilities belong behind the add button. A local
 * computer target is an explicit per-task grant; leaving it unset lets the
 * deployment use its managed browser without adding a choice to every task.
 */
export function ComposerAddMenu({
  value,
  onValueChange,
  onAddFiles,
  showFileUpload = false,
  disabled = false,
  selectionLocked = false,
  dark = false,
  className,
}: ComposerAddMenuProps) {
  const { t } = useI18n()
  const [open, setOpen] = useState(false)
  const [view, setView] = useState<"root" | "computer">("root")
  const [readiness, setReadiness] =
    useState<ComputerReadinessResponse["targets"] | null>(null)

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

  const selectedOption = options.find((option) => option.value === value)
  const SelectedIcon = selectedOption?.icon
  const selectedReadiness = value ? readiness?.[value] : undefined
  const closeMenu = () => {
    setOpen(false)
    setView("root")
  }

  return (
    <div className={cn("flex min-w-0 items-center gap-1.5", className)}>
      <Popover
        open={open}
        onOpenChange={(nextOpen) => {
          setOpen(nextOpen)
          if (!nextOpen) setView("root")
        }}
      >
        <PopoverTrigger asChild>
          <button
            type="button"
            aria-label={t("computerRuntime.addMenu.title")}
            title={t("computerRuntime.addMenu.title")}
            disabled={disabled}
            className={cn(
              "inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-full transition-colors",
              dark
                ? "text-white/75 hover:bg-[hsl(234_30%_35%)] hover:text-white"
                : "text-muted-foreground hover:bg-secondary/80 hover:text-foreground",
              disabled && "cursor-not-allowed opacity-50"
            )}
          >
            <Plus className="h-4 w-4" />
          </button>
        </PopoverTrigger>
        <PopoverContent
          align="start"
          side="top"
          aria-label={t("computerRuntime.addMenu.title")}
          className="w-80 p-2"
        >
          {view === "root" ? (
            <div className="space-y-1">
              {showFileUpload && onAddFiles && (
                <button
                  type="button"
                  className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left hover:bg-accent"
                  onClick={() => {
                    closeMenu()
                    onAddFiles()
                  }}
                >
                  <FilePlus2 className="h-4 w-4 shrink-0" />
                  <span className="text-sm font-medium">
                    {t("computerRuntime.addMenu.files")}
                  </span>
                </button>
              )}
              <button
                type="button"
                className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left hover:bg-accent"
                onClick={() => setView("computer")}
              >
                <Monitor className="h-4 w-4 shrink-0" />
                <span className="min-w-0 flex-1 text-sm font-medium">
                  {t("computerRuntime.addMenu.computerAccess")}
                </span>
                <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />
              </button>
            </div>
          ) : (
            <>
              <button
                type="button"
                aria-label={t("computerRuntime.addMenu.back")}
                className="mb-1 flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left text-sm font-medium hover:bg-accent"
                onClick={() => setView("root")}
              >
                <ChevronLeft className="h-4 w-4 shrink-0" />
                <span>{t("computerRuntime.addMenu.computerAccess")}</span>
              </button>
              <div className="space-y-1 border-t pt-1">
                {options.map((option) => {
                  const OptionIcon = option.icon
                  const selected = option.value === value
                  const optionReadiness = readiness?.[option.value]
                  const firstIssue = optionReadiness?.issues[0]
                  const canGrant =
                    !selectionLocked &&
                    Boolean(onValueChange) &&
                    optionReadiness?.ready
                  const optionContent = (
                    <>
                      <OptionIcon className="mt-0.5 h-4 w-4 shrink-0" />
                      <span className="min-w-0 flex-1">
                        <span className="block text-sm font-medium">
                          {option.label}
                        </span>
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
                          {selectionLocked && !selected
                            ? t("computerRuntime.status.taskLocked")
                            : optionReadiness
                              ? optionReadiness.ready
                                ? selected
                                  ? t("computerRuntime.status.added")
                                  : t("computerRuntime.status.ready")
                                : firstIssue
                                  ? t(`computerRuntime.issues.${firstIssue.code}`)
                                  : t("computerRuntime.status.needsAttention")
                              : t("computerRuntime.status.checking")}
                        </span>
                      </span>
                      {selected && (
                        <Check className="mt-0.5 h-4 w-4 text-primary" />
                      )}
                      {!selected &&
                        optionReadiness &&
                        !optionReadiness.ready && (
                          <span className="mt-0.5 text-xs font-medium text-primary">
                            {t("computerRuntime.status.connect")}
                          </span>
                        )}
                    </>
                  )

                  if (
                    !selectionLocked &&
                    optionReadiness &&
                    !optionReadiness.ready
                  ) {
                    return (
                      <Link
                        key={option.value}
                        href={`/settings?tab=computer#${option.settingsHash}`}
                        className="flex w-full items-start gap-3 rounded-lg px-3 py-2.5 text-left hover:bg-accent"
                        onClick={closeMenu}
                      >
                        {optionContent}
                      </Link>
                    )
                  }

                  return (
                    <button
                      key={option.value}
                      type="button"
                      disabled={!canGrant}
                      className={cn(
                        "flex w-full items-start gap-3 rounded-lg px-3 py-2.5 text-left",
                        canGrant && "hover:bg-accent",
                        selected && "bg-accent",
                        !canGrant && !selected && "opacity-60"
                      )}
                      onClick={() => {
                        if (!canGrant || !onValueChange) return
                        onValueChange(option.value)
                        closeMenu()
                      }}
                    >
                      {optionContent}
                    </button>
                  )
                })}
              </div>

              <div className="mt-2 border-t px-2 pt-2">
                <Link
                  href={`/settings?tab=computer#${selectedOption?.settingsHash ?? "browser-relay"}`}
                  className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
                  onClick={closeMenu}
                >
                  {t("computerRuntime.manageConnections")}
                  <ExternalLink className="h-3 w-3" />
                </Link>
              </div>
            </>
          )}
        </PopoverContent>
      </Popover>

      {selectedOption && SelectedIcon && (
        <div
          className={cn(
            "inline-flex h-8 min-w-0 max-w-[180px] items-center gap-1.5 rounded-lg border px-2.5 text-xs",
            dark
              ? "border-white/15 bg-white/10 text-white/85"
              : "border-border bg-secondary/70 text-foreground"
          )}
          title={t("computerRuntime.boundHint", {
            target: selectedOption.label,
          })}
        >
          <SelectedIcon className="h-3.5 w-3.5 shrink-0" />
          <span className="truncate">{selectedOption.label}</span>
          <ReadinessDot
            readiness={selectedReadiness}
            readyLabel={t("computerRuntime.status.ready")}
            attentionLabel={t("computerRuntime.status.needsAttention")}
            checkingLabel={t("computerRuntime.status.checking")}
          />
          {!selectionLocked && onValueChange && (
            <button
              type="button"
              aria-label={t("computerRuntime.removeAccess", {
                target: selectedOption.label,
              })}
              className={cn(
                "ml-0.5 rounded-sm p-0.5",
                dark ? "hover:bg-white/15" : "hover:bg-foreground/10"
              )}
              onClick={() => onValueChange(undefined)}
            >
              <X className="h-3 w-3" />
            </button>
          )}
        </div>
      )}
    </div>
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

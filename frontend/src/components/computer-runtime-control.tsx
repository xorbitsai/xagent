"use client"

import { Check, ChevronDown, Globe2, Monitor } from "lucide-react"
import React, { useState } from "react"

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { useI18n } from "@/contexts/i18n-context"
import { cn } from "@/lib/utils"

export type ComputerRuntimeKind = "extension_relay" | "desktop_relay"

export const DEFAULT_COMPUTER_RUNTIME_KIND: ComputerRuntimeKind =
  "extension_relay"

const STORAGE_KEY = "xagent.computerRuntimeKind"

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
  const isDesktop = value === "desktop_relay"
  const Icon = isDesktop ? Monitor : Globe2
  const label = isDesktop
    ? t("computerRuntime.desktop.label")
    : t("computerRuntime.browser.label")

  const trigger = (
    <button
      type="button"
      disabled={disabled}
      aria-label={label}
      className={cn(
        "inline-flex h-9 items-center justify-center gap-2 whitespace-nowrap rounded-xl px-3 text-xs font-normal transition-colors",
        dark
          ? "text-white/75 hover:bg-[hsl(234_30%_35%)] hover:text-white"
          : "text-muted-foreground hover:bg-secondary/80 hover:text-foreground",
        disabled && "cursor-default opacity-70",
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
      {!disabled && <ChevronDown className="h-3.5 w-3.5 opacity-60" />}
    </button>
  )

  if (disabled || !onValueChange) {
    return trigger
  }

  const options: Array<{
    value: ComputerRuntimeKind
    label: string
    description: string
    icon: typeof Globe2
  }> = [
    {
      value: "extension_relay",
      label: t("computerRuntime.browser.label"),
      description: t("computerRuntime.browser.description"),
      icon: Globe2,
    },
    {
      value: "desktop_relay",
      label: t("computerRuntime.desktop.label"),
      description: t("computerRuntime.desktop.description"),
      icon: Monitor,
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
            return (
              <button
                key={option.value}
                type="button"
                className={cn(
                  "flex w-full items-start gap-3 rounded-lg px-3 py-2.5 text-left hover:bg-accent",
                  selected && "bg-accent"
                )}
                onClick={() => {
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
                </span>
                {selected && <Check className="mt-0.5 h-4 w-4 text-primary" />}
              </button>
            )
          })}
        </div>
      </PopoverContent>
    </Popover>
  )
}

"use client"

import React, { useId } from "react"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { useI18n } from "@/contexts/i18n-context"
import type { VoiceModelReason } from "@/hooks/use-voice-models"
import { cn } from "@/lib/utils"
import { VoiceWaveIcon } from "./voice-call-ui"

export function VoiceModeButton({ reason, disabled, disabledHint, iconOnly = false, onClick }: {
  reason?: VoiceModelReason; disabled?: boolean
  /** Why the button is disabled when it is not a model problem. */
  disabledHint?: string
  iconOnly?: boolean; onClick: () => void
}) {
  const { t } = useI18n()
  const id = useId()
  const unavailable = Boolean(reason || disabled)
  const hint = reason ? t(`voiceConversation.${reason}`) : disabled && disabledHint ? disabledHint : t("voiceConversation.open")
  const described = reason || (disabled && disabledHint)
  return <TooltipProvider><Tooltip delayDuration={200}>
    <TooltipTrigger asChild>
      <span className="inline-flex" tabIndex={unavailable ? 0 : undefined} title={hint}>
        <Button type="button" variant={iconOnly ? "default" : "ghost"} size="sm" aria-label={t("voiceConversation.open")}
          aria-describedby={described ? id : undefined} disabled={unavailable}
          // Icon-only is the composer's main action while nothing is typed: the
          // same size and shape as the send button it swaps with.
          className={cn(iconOnly && "h-8 w-8 rounded-lg p-0 motion-safe:animate-voice-pop", iconOnly && unavailable && "bg-muted text-muted-foreground/60")}
          onMouseDown={event => event.preventDefault()} onClick={onClick}>
          <VoiceWaveIcon className="h-4 w-4" />
          {!iconOnly && <span className="ml-2">{t("voiceConversation.open")}</span>}
        </Button>
      </span>
    </TooltipTrigger>
    {described && <span id={id} className="sr-only">{hint}</span>}
    <TooltipContent><p className="max-w-xs">{hint}</p></TooltipContent>
  </Tooltip></TooltipProvider>
}

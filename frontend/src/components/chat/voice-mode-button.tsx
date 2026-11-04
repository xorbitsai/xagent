"use client"

import React, { useId } from "react"
import { Headphones } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { useI18n } from "@/contexts/i18n-context"
import type { VoiceModelReason } from "@/hooks/use-voice-models"

export function VoiceModeButton({ reason, disabled, iconOnly = false, onClick }: {
  reason?: VoiceModelReason; disabled?: boolean; iconOnly?: boolean; onClick: () => void
}) {
  const { t } = useI18n()
  const id = useId()
  const unavailable = Boolean(reason || disabled)
  const hint = reason ? t(`voiceConversation.${reason}`) : t("voiceConversation.open")
  return <TooltipProvider><Tooltip delayDuration={200}>
    <TooltipTrigger asChild>
      <span className="inline-flex" tabIndex={unavailable ? 0 : undefined} title={hint}>
        <Button type="button" variant="ghost" size="sm" aria-label={t("voiceConversation.open")}
          aria-describedby={reason ? id : undefined} disabled={unavailable}
          className={iconOnly ? "h-8 w-8 p-0 rounded-full text-muted-foreground hover:text-foreground" : undefined}
          onMouseDown={event => event.preventDefault()} onClick={onClick}>
          <Headphones className="h-4 w-4" />
          {!iconOnly && <span className="ml-2">{t("voiceConversation.open")}</span>}
        </Button>
      </span>
    </TooltipTrigger>
    {reason && <span id={id} className="sr-only">{hint}</span>}
    <TooltipContent><p className="max-w-xs">{hint}</p></TooltipContent>
  </Tooltip></TooltipProvider>
}

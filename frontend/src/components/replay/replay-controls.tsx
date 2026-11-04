"use client"

import { Button } from "@/components/ui/button"
import { Play, Pause, SkipBack } from "lucide-react"
import { cn } from "@/lib/utils"
import { useI18n } from "@/contexts/i18n-context"

interface ReplayControlsProps {
  isPlaying: boolean
  playbackSpeed: number
  onPlay: () => void
  onPause: () => void
  onStop: () => void
  onSpeedChange: (speed: number) => void
  className?: string
}

export function ReplayControls({
  isPlaying,
  playbackSpeed,
  onPlay,
  onPause,
  onStop,
  onSpeedChange,
  className
}: ReplayControlsProps) {
  const { t } = useI18n()
  const speedOptions = [0.5, 1, 2, 4]

  return (
    <div className={cn(
      "flex items-center gap-2 bg-background/80 border border-border rounded-lg px-3 py-1.5",
      className
    )}>
      {/* Playback Controls */}
      <div className="flex items-center gap-1">
        <Button
          size="sm"
          variant="ghost"
          onClick={onStop}
          className="h-7 w-7 p-0"
          title={t('agent.replay.controls.stop')}
        >
          <SkipBack className="h-3 w-3" />
        </Button>

        <Button
          size="sm"
          variant="ghost"
          onClick={isPlaying ? onPause : onPlay}
          className="h-7 w-7 p-0"
          title={isPlaying ? t('agent.replay.controls.pause') : t('agent.replay.controls.play')}
        >
          {isPlaying ? (
            <Pause className="h-3 w-3" />
          ) : (
            <Play className="h-3 w-3" />
          )}
        </Button>
      </div>

      {/* Speed Control */}
      <select
        value={playbackSpeed}
        onChange={(e) => onSpeedChange(parseFloat(e.target.value))}
        className="text-xs bg-transparent border border-border rounded px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-primary"
        title={t('agent.replay.controls.speed')}
      >
        {speedOptions.map((speed) => (
          <option key={speed} value={speed}>
            {speed}{t('agent.replay.controls.speedSuffix')}
          </option>
        ))}
      </select>
    </div>
  )
}

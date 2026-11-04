import React from 'react'
import { useI18n } from '@/contexts/i18n-context'

interface ContextUsageRingProps {
  tokens: number
  threshold: number
  className?: string
}

/**
 * Compact circular gauge showing how full the model context window is relative
 * to the compaction threshold. The ring fills as the conversation grows; once
 * it reaches 100% the backend compacts the context (the gauge then resets on
 * the next turn). Mirrors the "context left" indicator in Codex-style UIs.
 */
export function ContextUsageRing({ tokens, threshold, className }: ContextUsageRingProps) {
  const { t } = useI18n()

  if (!threshold || threshold <= 0) return null

  const ratio = Math.max(0, tokens / threshold)
  const clamped = Math.min(1, ratio)
  const pct = Math.round(clamped * 100)
  // Backend compacts strictly above the threshold (tokens > threshold), so only
  // flag "full/compacting" past 100%, not at exactly 100%.
  const isFull = ratio > 1

  // 70% amber, full red, otherwise the theme accent.
  const color = isFull
    ? 'rgb(239 68 68)' // red-500
    : ratio >= 0.7
      ? 'rgb(245 158 11)' // amber-500
      : 'rgb(99 102 241)' // indigo-500

  const size = 16
  const stroke = 2.5
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  const offset = c * (1 - clamped)

  const tooltip = isFull
    ? t('chatPage.contextUsage.full')
    : t('chatPage.contextUsage.tooltip', {
        pct,
        used: tokens.toLocaleString(),
        total: threshold.toLocaleString(),
      })

  return (
    <span
      className={`inline-flex items-center gap-1.5 whitespace-nowrap ${className || ''}`}
      title={tooltip}
    >
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="currentColor"
          strokeWidth={stroke}
          className="text-muted-foreground/25"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={offset}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: 'stroke-dashoffset 0.4s ease, stroke 0.4s ease' }}
        />
      </svg>
      <span className="text-muted-foreground tabular-nums">
        {isFull ? t('chatPage.contextUsage.full') : `${pct}%`}
      </span>
    </span>
  )
}

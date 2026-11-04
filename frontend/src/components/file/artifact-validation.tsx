import React, { useEffect, useState } from 'react'
import { useFileAccess } from '@/contexts/file-access-context'
import { useI18n } from '@/contexts/i18n-context'

type Status = 'valid' | 'invalid' | 'unchecked'
type DisplayReport = { status: Status | 'error'; supported: boolean }
const STATUSES: ReadonlySet<string> = new Set(['valid', 'invalid', 'unchecked'])

async function readReport(response: Response): Promise<DisplayReport> {
  if (!response.ok) throw new Error('Validation unavailable')
  // An older backend may ignore validation_only and return the attachment
  // itself. Never interpret arbitrary JSON file contents as a report.
  if (response.headers.get('content-type')?.split(';')[0] !== 'application/vnd.xagent.validation+json') {
    throw new Error('Not a validation response')
  }
  const data = await response.json()
  if (!data || !STATUSES.has(data.status) ||
      !Array.isArray(data.checks) || !data.checks.length ||
      !data.checks.every((c: { status?: unknown } | null) => c && STATUSES.has(String(c.status)))) {
    throw new Error('Invalid report')
  }
  if (data.status !== 'unchecked' &&
      (typeof data.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(data.sha256))) {
    throw new Error('Missing snapshot')
  }
  if (data.status === 'valid' && !data.checks.every((c: { status: string }) => c.status === 'valid')) {
    throw new Error('Incomplete checks')
  }
  // The backend owns format support. Do not duplicate a suffix allowlist in
  // the UI or show an endless Recheck action for formats without a validator.
  // Machine statuses drive localized presentation; English parser diagnostics
  // remain in the API/model report rather than leaking into the user's locale.
  return { status: data.status, supported: data.supported !== false }
}

/** Server-authoritative, current-byte checks, independent of preview renderers. */
export function ArtifactValidation({ fileId, children }: {
  fileId: string
  children: React.ReactNode
}) {
  const policy = useFileAccess()
  const { t } = useI18n()
  const [attempt, setAttempt] = useState(0)
  const [result, setResult] = useState<DisplayReport & { key: string }>()
  const url = policy.validationUrl?.(fileId)
  const key = `${url}:${attempt}`

  useEffect(() => {
    if (!url) return
    let active = true
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 20_000)
    const check = async () => {
      try {
        const response = await policy.request(url, { signal: controller.signal, cache: 'no-store' })
        const report = await readReport(response)
        if (active) setResult({ key, ...report })
      } catch {
        if (active) setResult({ key, status: 'error', supported: true })
      } finally {
        clearTimeout(timeout)
      }
    }
    void check()
    return () => {
      active = false
      clearTimeout(timeout)
      controller.abort()
    }
  }, [key, url, policy])

  if (!url) return <>{children}</>
  const current = result?.key === key ? result : undefined
  if (current?.supported === false) return <>{children}</>
  const label = current?.status ?? 'checking'
  return (
    <div data-artifact-validation={label}>
      <div className="flex items-center gap-2 py-1 text-xs text-muted-foreground" role="status">
        <span className={label === 'invalid' ? 'text-destructive' : undefined}>
          {t(`files.validation.${label}`)}
        </span>
        {current ? (
          <button type="button" className="underline" onClick={() => setAttempt(n => n + 1)}>
            {t('files.validation.recheck')}
          </button>
        ) : null}
      </div>
      <React.Fragment key={key}>{children}</React.Fragment>
    </div>
  )
}

"use client"

import React, { useEffect, useState } from "react"
import { useParams, useRouter } from "next/navigation"
import { useI18n } from "@/contexts/i18n-context"
import { getWorkforce } from "@/lib/workforces-api"
import type { WorkforceDetail, WorkforceRunResponse } from "@/types/workforce"
import { WorkforceSummary, WorkforceTestPanel } from "@/components/workforce"
import { getRunDisabledReason } from "../../workforce-ui-state"
import { toast } from "sonner"

export default function WorkforceRunPage() {
  const { t } = useI18n()
  const router = useRouter()
  const params = useParams()
  const [workforce, setWorkforce] = useState<WorkforceDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true)
        setError(null)
        const id = Array.isArray(params.id) ? params.id[0] : params.id
        if (!id) {
          setWorkforce(null)
          return
        }
        const data = await getWorkforce(id)
        setWorkforce(data)
      } catch (err) {
        const nextError = err instanceof Error ? err.message : t("workforces.errors.load")
        setError(nextError)
        toast.error(nextError)
      } finally {
        setLoading(false)
      }
    }
    void load()
  }, [params.id, t])

  if (loading) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">{t("workforces.loading.runView")}</div>
  if (error) return <div className="h-full overflow-y-auto p-4 text-red-500 sm:p-8">{error}</div>
  if (!workforce) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">{t("workforces.errors.notFound")}</div>

  const runDisabledReason = getRunDisabledReason(workforce.status, t)

  const handleRunCreated = (result: WorkforceRunResponse) => {
    router.push(result.redirect_url || `/task/${result.task_id}`)
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto grid w-full gap-6 p-4 sm:p-8 lg:grid-cols-[1.1fr_0.9fr]">
        <WorkforceSummary workforce={workforce} />
        <WorkforceTestPanel
          workforceId={workforce.id}
          disabled={Boolean(runDisabledReason)}
          disabledReason={runDisabledReason || undefined}
          onRunCreated={handleRunCreated}
        />
      </div>
    </div>
  )
}

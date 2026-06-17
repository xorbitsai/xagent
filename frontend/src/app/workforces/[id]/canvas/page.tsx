"use client"

import Link from "next/link"
import React, { useEffect, useState } from "react"
import { useParams } from "next/navigation"
import { ArrowLeft } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useI18n } from "@/contexts/i18n-context"
import { getWorkforce } from "@/lib/workforces-api"
import type { WorkforceDetail } from "@/types/workforce"
import { WorkforceCanvas } from "@/components/workforce"
import { toast } from "sonner"

export default function WorkforceCanvasPage() {
  const { t } = useI18n()
  const params = useParams()
  const id = Array.isArray(params.id) ? params.id[0] : params.id
  const [workforce, setWorkforce] = useState<WorkforceDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true)
        setError(null)
        if (!id) {
          setWorkforce(null)
          return
        }
        const workforceData = await getWorkforce(id)
        setWorkforce(workforceData)
      } catch (err) {
        const nextError = err instanceof Error ? err.message : t("workforces.errors.loadCanvas")
        setError(nextError)
        toast.error(nextError)
      } finally {
        setLoading(false)
      }
    }
    void load()
  }, [id, t])

  const backHref = id ? `/workforces/${id}` : "/workforces"

  if (loading) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">{t("workforces.loading.canvas")}</div>
  if (error) return <div className="h-full overflow-y-auto p-4 text-red-500 sm:p-8">{error}</div>
  if (!workforce) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">{t("workforces.errors.canvasUnavailable")}</div>

  return (
    <div className="flex h-full flex-col">
      <div className="flex-none p-8 pb-0">
        <Link href={backHref}>
          <Button variant="outline" size="sm">
            <ArrowLeft className="mr-2 h-4 w-4" />
            {t("workforces.canvas.backToDetails")}
          </Button>
        </Link>
      </div>
      <div className="flex-1 p-8">
        <WorkforceCanvas workforce={workforce} />
      </div>
    </div>
  )
}

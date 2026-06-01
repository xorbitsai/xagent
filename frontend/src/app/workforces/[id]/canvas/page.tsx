"use client"

import Link from "next/link"
import React, { useEffect, useState } from "react"
import { useParams } from "next/navigation"
import { ArrowLeft } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useI18n } from "@/contexts/i18n-context"
import { getWorkforceCanvas } from "@/lib/workforces-api"
import type { WorkforceCanvasResponse } from "@/types/workforce"
import { WorkforceCanvas } from "@/components/workforce"
import { toast } from "sonner"

export default function WorkforceCanvasPage() {
  const { t } = useI18n()
  const params = useParams()
  const id = Array.isArray(params.id) ? params.id[0] : params.id
  const [canvas, setCanvas] = useState<WorkforceCanvasResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true)
        setError(null)
        if (!id) {
          setCanvas(null)
          return
        }
        const data = await getWorkforceCanvas(id)
        setCanvas(data)
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
  if (!canvas) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">{t("workforces.errors.canvasUnavailable")}</div>

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto flex w-full flex-col gap-4 p-4 sm:p-8">
        <div>
          <Link href={backHref}>
            <Button variant="outline" size="sm">
              <ArrowLeft className="mr-2 h-4 w-4" />
              {t("workforces.canvas.backToDetails")}
            </Button>
          </Link>
        </div>
        <WorkforceCanvas canvas={canvas} />
      </div>
    </div>
  )
}

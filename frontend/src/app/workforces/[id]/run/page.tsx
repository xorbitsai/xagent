"use client"

import React, { useEffect, useState } from "react"
import { useParams, useRouter } from "next/navigation"
import { ArrowLeft, Bot, Play } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useI18n } from "@/contexts/i18n-context"
import { getWorkforce, runWorkforce } from "@/lib/workforces-api"
import { WorkforceStatusBadge } from "@/components/workforce"
import type { WorkforceDetail, WorkforceRunResponse } from "@/types/workforce"
import { getRunDisabledReason } from "../../workforce-ui-state"
import { toast } from "sonner"
import { Textarea } from "@/components/ui/textarea"

export default function WorkforceRunPage() {
  const { t } = useI18n()
  const router = useRouter()
  const params = useParams()
  const id = Array.isArray(params.id) ? params.id[0] : params.id
  const [workforce, setWorkforce] = useState<WorkforceDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState("")
  const [running, setRunning] = useState(false)

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true)
        setError(null)
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
  }, [id, t])

  const handleRun = async () => {
    const value = message.trim()
    if (!value || running || !id) return
    setRunning(true)
    try {
      const result: WorkforceRunResponse = await runWorkforce(id, { message: value })
      const target = result?.redirect_url || (result?.task_id ? `/task/${result.task_id}` : null)
      if (target) {
        router.push(target)
      } else {
        throw new Error("Invalid run response: missing redirect_url or task_id")
      }
    } catch (err) {
      const nextError = err instanceof Error ? err.message : t("workforces.errors.run")
      toast.error(nextError)
    } finally {
      setRunning(false)
    }
  }

  const runDisabledReason = !workforce ? null : getRunDisabledReason(workforce.status, t)
  const canRun = Boolean(workforce && !runDisabledReason)

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center">
          <Bot className="h-12 w-12 mx-auto mb-4 animate-pulse text-muted-foreground" />
          <p className="text-muted-foreground">{t("workforces.loading.runView")}</p>
        </div>
      </div>
    )
  }

  if (error && !workforce) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="max-w-md w-full text-center space-y-6">
          <div className="flex justify-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-muted">
              <Bot className="h-6 w-6 text-muted-foreground" />
            </div>
          </div>
          <div className="space-y-2">
            <h2 className="text-lg font-semibold">{t("workforces.errors.notFound")}</h2>
            <p className="text-sm text-muted-foreground">{error}</p>
          </div>
          <Button className="w-full" onClick={() => router.push("/workforces")}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            {t("workforces.create.backToWorkforces")}
          </Button>
        </div>
      </div>
    )
  }

  if (!workforce) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="max-w-md w-full text-center space-y-6">
          <div className="flex justify-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-muted">
              <Bot className="h-6 w-6 text-muted-foreground" />
            </div>
          </div>
          <div className="space-y-2">
            <h2 className="text-lg font-semibold">{t("workforces.errors.notFound")}</h2>
            <p className="text-sm text-muted-foreground">{t("workforces.errors.canvasUnavailable")}</p>
          </div>
          <Button className="w-full" onClick={() => router.push("/workforces")}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            {t("workforces.create.backToWorkforces")}
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full bg-background flex flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto">
        <main className="container max-w-4xl mx-auto px-4 py-8">
          {/* Header */}
          <div className="flex items-center gap-3 mb-8">
            <Button
              variant="ghost"
              size="icon"
              className="shrink-0"
              onClick={() => router.push("/workforces")}
            >
              <ArrowLeft className="h-5 w-5" />
            </Button>
            <div className="flex items-center gap-3 min-w-0">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-blue-600 text-white">
                <Bot className="h-5 w-5" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <h1 className="text-xl font-bold truncate">{workforce.name}</h1>
                  <WorkforceStatusBadge status={workforce.status} />
                </div>
                {workforce.description && (
                  <p className="text-sm text-muted-foreground truncate">{workforce.description}</p>
                )}
              </div>
            </div>
          </div>

          {/* Back link */}
          <div className="mb-6">
            <Button variant="link" className="h-auto p-0 text-sm" onClick={() => router.push(`/workforces/${id}`)}>
              <ArrowLeft className="mr-1 h-4 w-4" />
              {t("workforces.canvas.backToDetails")}
            </Button>
          </div>

          {/* Chat-like interface */}
          <div className="space-y-6">
            {/* Info cards */}
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="rounded-xl border bg-card p-4">
                <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">
                  {t("workforces.fields.manager")}
                </div>
                <div className="font-medium">{workforce.manager?.name}</div>
                {workforce.manager?.description && (
                  <div className="text-sm text-muted-foreground mt-0.5">{workforce.manager.description}</div>
                )}
              </div>
              <div className="rounded-xl border bg-card p-4">
                <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">
                  {t("workforces.fields.workers")}
                </div>
                <div className="font-medium">{workforce.workers.length}</div>
                <div className="text-sm text-muted-foreground mt-0.5">
                  {t("workforces.summary.enabledCount", { count: workforce.workers.filter(w => w.enabled).length })}
                </div>
              </div>
            </div>

            {/* Workers list */}
            {workforce.workers.length > 0 && (
              <div className="rounded-xl border bg-card p-4">
                <div className="text-xs uppercase tracking-wide text-muted-foreground mb-3">
                  {t("workforces.fields.workers")}
                </div>
                <div className="space-y-2">
                  {workforce.workers.map((worker) => (
                    <div key={worker.id} className="flex items-center justify-between py-1.5 border-b last:border-b-0">
                      <div className="min-w-0">
                        <div className="text-sm font-medium truncate">{worker.alias || worker.agent.name}</div>
                        {worker.agent.description && (
                          <div className="text-xs text-muted-foreground truncate">{worker.agent.description}</div>
                        )}
                      </div>
                      <span className={`shrink-0 text-xs px-2 py-0.5 rounded-full ${worker.enabled ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'}`}>
                        {worker.enabled ? t("workforces.status.enabled") : t("workforces.status.disabled")}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Input area - styled like ChatStartScreen */}
            {canRun ? (
              <div className="rounded-2xl border-2 border-border bg-card shadow-sm overflow-hidden">
                <Textarea
                  placeholder={t("workforces.run.placeholder")}
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  className="border-0 bg-transparent text-[15px] outline-none placeholder:text-muted-foreground/60 resize-none focus-visible:ring-0 min-h-[130px] px-4 py-3 pb-16 max-h-[400px]"
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault()
                      void handleRun()
                    }
                  }}
                />
                <div className="flex items-center justify-between bg-card px-4 py-3">
                  <div className="flex items-center gap-2" />
                  <div className="flex items-center gap-3">
                    <span className="text-[13px] font-medium text-muted-foreground/50 select-none mr-1">
                      ⏎ {t("common.send")}
                    </span>
                    <Button
                      size="icon"
                      className="h-8 w-8 rounded-lg"
                      onClick={handleRun}
                      disabled={running || !message.trim()}
                    >
                      <Play className="h-4 w-4 fill-current" />
                    </Button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="rounded-xl border bg-muted/20 p-6 text-center">
                <p className="text-sm text-muted-foreground">{runDisabledReason}</p>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}

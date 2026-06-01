"use client"

import Link from "next/link"
import React, { useCallback, useEffect, useState } from "react"
import { ChevronLeft, ChevronRight, Play, Plus, Users, Zap, GitBranch, ShieldCheck } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { SearchInput } from "@/components/ui/search-input"
import { useI18n } from "@/contexts/i18n-context"
import { useRouter } from "next/navigation"
import { listWorkforces } from "@/lib/workforces-api"
import type { WorkforceListItem } from "@/types/workforce"
import { getRunDisabledReason } from "./workforce-ui-state"
import { FeatureEmptyState } from "@/components/ui/feature-empty-state"
import { toast } from "sonner"

export default function WorkforcesPage() {
  const { locale, t } = useI18n()
  const router = useRouter()
  const [items, setItems] = useState<WorkforceListItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState("")
  const [page, setPage] = useState(1)
  const [pages, setPages] = useState(1)
  const [total, setTotal] = useState(0)
  const pageSize = 10
  const hasActiveSearch = search.trim().length > 0

  const load = useCallback(async (nextPage: number, nextSearch: string) => {
    try {
      setLoading(true)
      setError(null)
      const data = await listWorkforces({ page: nextPage, size: pageSize, search: nextSearch })
      setItems(data.items)
      setPages(data.pages)
      setTotal(data.total)
    } catch (err) {
      const nextError = err instanceof Error ? err.message : t("workforces.errors.loadList")
      setError(nextError)
      toast.error(nextError)
      setItems([])
    } finally {
      setLoading(false)
    }
  }, [t])

  useEffect(() => {
    void load(page, search)
  }, [load, page, search])

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto flex w-full flex-col gap-6 p-4 sm:p-8">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <h1 className="text-3xl font-bold">{t("workforces.list.title")}</h1>
            <p className="mt-2 max-w-2xl text-muted-foreground">
              {t("workforces.list.description")}
            </p>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            <SearchInput
              placeholder={t("workforces.list.searchPlaceholder")}
              value={search}
              onChange={(value) => {
                setSearch(value)
                setPage(1)
              }}
              containerClassName="w-full sm:w-80"
            />
            <Link href="/workforces/new">
              <Button>
                <Plus className="mr-2 h-4 w-4" />
                {t("workforces.actions.new")}
              </Button>
            </Link>
          </div>
        </div>

        {loading ? <div className="p-8 text-muted-foreground">{t("workforces.loading.list")}</div> : null}
        {error ? <div className="p-8 text-red-500">{error}</div> : null}

        {!loading && !error ? (
          items.length === 0 ? (
            hasActiveSearch ? (
              <Card className="border-dashed">
                <CardContent className="flex flex-col items-center gap-4 p-12 text-center">
                  <div className="text-lg font-medium">{t("workforces.list.noResultsTitle")}</div>
                  <p className="max-w-xl text-sm text-muted-foreground">
                    {t("workforces.list.noResults")}
                  </p>
                </CardContent>
              </Card>
            ) : (
              <FeatureEmptyState
                icon={Users}
                title={t("workforces.emptyState.title")}
                description={t("workforces.emptyState.description")}
                features={[
                  {
                    icon: GitBranch,
                    title: t("workforces.emptyState.features.managerAgent.title"),
                    description: t("workforces.emptyState.features.managerAgent.description")
                  },
                  {
                    icon: Zap,
                    title: t("workforces.emptyState.features.subAgents.title"),
                    description: t("workforces.emptyState.features.subAgents.description")
                  },
                  {
                    icon: Play,
                    title: t("workforces.emptyState.features.parallelExecution.title"),
                    description: t("workforces.emptyState.features.parallelExecution.description")
                  },
                  {
                    icon: ShieldCheck,
                    title: t("workforces.emptyState.features.approvalGates.title"),
                    description: t("workforces.emptyState.features.approvalGates.description")
                  }
                ]}
                actionLabel={t("workforces.emptyState.action")}
                onAction={() => router.push("/workforces/new")}
                className="h-full mt-4"
              />
            )
          ) : (
            <>
              <div className="grid gap-4">
                {items.map((item) => {
                  const runDisabledReason = getRunDisabledReason(item.status, t)
                  return (
                    <Card key={item.id} className="overflow-hidden">
                      <CardContent className="p-0">
                        <div className="grid gap-6 p-6 lg:grid-cols-[1.4fr_0.6fr] lg:items-center">
                          <div className="space-y-4">
                            <div className="flex flex-wrap items-center gap-3">
                              <Link
                                href={`/workforces/${item.id}`}
                                className="text-xl font-semibold hover:underline"
                              >
                                {item.name}
                              </Link>
                              <span className="rounded-full border px-2.5 py-1 text-xs capitalize text-muted-foreground">
                                {t(`workforces.status.${item.status}`)}
                              </span>
                            </div>
                            <p className="text-sm text-muted-foreground">
                              {item.description || t("workforces.common.noDescription")}
                            </p>
                            <div className="flex flex-wrap gap-6 text-sm text-muted-foreground">
                              <span>{t("workforces.list.manager", { name: item.manager.name })}</span>
                              <span>{t("workforces.list.workers", { count: item.worker_count })}</span>
                              <span suppressHydrationWarning>
                                {t("workforces.list.lastUpdate", {
                                  value: item.updated_at
                                    ? new Date(item.updated_at).toLocaleString(locale)
                                    : t("workforces.common.notAvailable"),
                                })}
                              </span>
                            </div>
                          </div>
                          <div className="flex flex-col gap-3 lg:items-end">
                            <div className="flex w-full flex-col gap-1 lg:w-auto lg:items-end">
                              {runDisabledReason ? (
                                <Button className="w-full lg:w-auto" disabled>
                                  <Play className="mr-2 h-4 w-4" />
                                  {t("workforces.actions.run")}
                                </Button>
                              ) : (
                                <Link href={`/workforces/${item.id}/run`}>
                                  <Button className="w-full lg:w-auto">
                                    <Play className="mr-2 h-4 w-4" />
                                    {t("workforces.actions.run")}
                                  </Button>
                                </Link>
                              )}
                              {runDisabledReason ? (
                                <div className="text-xs text-muted-foreground">{runDisabledReason}</div>
                              ) : null}
                            </div>
                            <div className="flex gap-3">
                              <Link href={`/workforces/${item.id}`}>
                                <Button variant="outline">{t("workforces.actions.details")}</Button>
                              </Link>
                              <Link href={`/workforces/${item.id}/builder`}>
                                <Button variant="outline">{t("workforces.actions.builder")}</Button>
                              </Link>
                              <Link href={`/workforces/${item.id}/canvas`}>
                                <Button variant="outline">{t("workforces.actions.canvas")}</Button>
                              </Link>
                            </div>
                            {item.last_run ? (
                              <div className="text-xs text-muted-foreground">
                                {item.last_run.task_id != null
                                  ? t("workforces.list.lastRunWithTask", {
                                    runId: item.last_run.id,
                                    taskId: item.last_run.task_id,
                                    status: item.last_run.status,
                                  })
                                  : t("workforces.list.lastRun", {
                                    runId: item.last_run.id,
                                    status: item.last_run.status,
                                  })}
                              </div>
                            ) : (
                              <div className="text-xs text-muted-foreground">{t("workforces.list.noRuns")}</div>
                            )}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )
                })}
              </div>

              {pages > 1 ? (
                <div className="flex items-center justify-between">
                  <div className="text-sm text-muted-foreground">
                    {t("workforces.pagination.showing", {
                      start: (page - 1) * pageSize + 1,
                      end: Math.min(page * pageSize, total),
                      total,
                    })}
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setPage((current) => current - 1)}
                      disabled={page <= 1}
                    >
                      <ChevronLeft className="mr-1 h-4 w-4" />
                      {t("workforces.pagination.prev")}
                    </Button>
                    <span className="text-sm text-muted-foreground">
                      {t("workforces.pagination.page", { page, pages })}
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setPage((current) => current + 1)}
                      disabled={page >= pages}
                    >
                      {t("workforces.pagination.next")}
                      <ChevronRight className="ml-1 h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ) : null}
            </>
          )
        ) : null}
      </div>
    </div>
  )
}

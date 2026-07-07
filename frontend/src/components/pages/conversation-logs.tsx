"use client"

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Bot, ChevronLeft, ChevronRight, Inbox } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { CompactionNotice } from "@/components/chat/CompactionNotice"
import { TraceEventRenderer } from "@/components/chat/TraceEventRenderer"
import { MarkdownRenderer } from "@/components/ui/markdown-renderer"
import { SearchInput } from "@/components/ui/search-input"
import { useI18n } from "@/contexts/i18n-context"
import {
  type ConversationLogDetailResponse,
  type ConversationLogListResponse,
  type ConversationLogSource,
  type ConversationLogSummary,
  fetchConversationLogDetail,
  fetchConversationLogs,
} from "@/lib/conversation-logs-api"
import { cn } from "@/lib/utils"

const SOURCE_TABS: Array<{
  id: ConversationLogSource
  labelKey: string
}> = [
  { id: "all", labelKey: "conversationLogs.sources.all" },
  { id: "widget", labelKey: "conversationLogs.sources.widget" },
  { id: "rest_api", labelKey: "conversationLogs.sources.restApi" },
  { id: "shared_link", labelKey: "conversationLogs.sources.sharedLink" },
  { id: "webhook", labelKey: "conversationLogs.sources.webhook" },
]

const PER_PAGE = 20

const SOURCE_BADGE_CLASS = "border-blue-100 bg-blue-50 text-blue-700"

function formatRelativeTime(value: string | null | undefined, locale: string): string {
  if (!value) return "-"
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value

  const units = [
    ["year", 60 * 60 * 24 * 365],
    ["month", 60 * 60 * 24 * 30],
    ["week", 60 * 60 * 24 * 7],
    ["day", 60 * 60 * 24],
    ["hour", 60 * 60],
    ["minute", 60],
  ] as const
  const deltaSeconds = Math.round((date.getTime() - Date.now()) / 1000)
  const absoluteSeconds = Math.abs(deltaSeconds)
  const formatter = new Intl.RelativeTimeFormat(locale, { numeric: "auto" })

  for (const [unit, seconds] of units) {
    if (absoluteSeconds >= seconds) {
      return formatter.format(Math.round(deltaSeconds / seconds), unit)
    }
  }
  return formatter.format(deltaSeconds, "second")
}

function sourceLabel(log: ConversationLogSummary): string {
  return log.source_label || log.source
}

function conversationTitle(log: ConversationLogSummary): string {
  return log.agent_name || log.title || ""
}

export function ConversationLogsPage() {
  const { locale, t } = useI18n()
  const [source, setSource] = useState<ConversationLogSource>("all")
  const [agentId, setAgentId] = useState<number | null>(null)
  const [search, setSearch] = useState("")
  const [page, setPage] = useState(1)
  const [listData, setListData] = useState<ConversationLogListResponse | null>(null)
  const [selectedTaskId, setSelectedTaskId] = useState<number | null>(null)
  const selectedTaskIdRef = useRef<number | null>(null)
  const [detail, setDetail] = useState<ConversationLogDetailResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isDetailLoading, setIsDetailLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [detailError, setDetailError] = useState<string | null>(null)
  const unknownErrorRef = useRef("Unknown error")

  useEffect(() => {
    unknownErrorRef.current = t("common.errors.unknown")
  }, [t])

  const loadDetail = useCallback(async (taskId: number) => {
    setIsDetailLoading(true)
    setDetailError(null)
    try {
      const data = await fetchConversationLogDetail(taskId)
      if (taskId === selectedTaskIdRef.current) {
        setDetail(data)
      }
    } catch (err) {
      if (taskId === selectedTaskIdRef.current) {
        setDetail(null)
        setDetailError(err instanceof Error ? err.message : unknownErrorRef.current)
      }
    } finally {
      if (taskId === selectedTaskIdRef.current) {
        setIsDetailLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    selectedTaskIdRef.current = selectedTaskId
  }, [selectedTaskId])

  useEffect(() => {
    let cancelled = false

    async function loadLogs() {
      setIsLoading(true)
      setError(null)
      try {
        const data = await fetchConversationLogs({
          source,
          agentId,
          search,
          page,
          perPage: PER_PAGE,
        })
        if (cancelled) return
        setListData(data)

        const currentSelectedId = selectedTaskIdRef.current
        const currentStillVisible = data.logs.some(
          (log) => log.task_id === currentSelectedId
        )
        const nextSelectedId = currentStillVisible
          ? currentSelectedId
          : data.logs[0]?.task_id ?? null
        selectedTaskIdRef.current = nextSelectedId
        setSelectedTaskId(nextSelectedId)
        if (nextSelectedId) {
          void loadDetail(nextSelectedId)
        } else {
          setDetail(null)
          setDetailError(null)
          setIsDetailLoading(false)
        }
      } catch (err) {
        if (cancelled) return
        setListData(null)
        setDetail(null)
        selectedTaskIdRef.current = null
        setSelectedTaskId(null)
        setError(err instanceof Error ? err.message : unknownErrorRef.current)
      } finally {
        if (!cancelled) setIsLoading(false)
      }
    }

    void loadLogs()
    return () => {
      cancelled = true
    }
  }, [agentId, loadDetail, page, search, source])

  const tabs = useMemo(() => {
    const counts = listData?.source_counts
    return SOURCE_TABS.map((item) => ({
      id: item.id,
      label: `${t(item.labelKey)} ${counts?.[item.id] ?? 0}`,
    }))
  }, [listData?.source_counts, t])

  const logs = listData?.logs ?? []
  const pagination = listData?.pagination
  const hasFilter = source !== "all" || Boolean(search.trim()) || agentId !== null

  function selectLog(taskId: number) {
    selectedTaskIdRef.current = taskId
    setSelectedTaskId(taskId)
    void loadDetail(taskId)
  }

  function updateSource(nextSource: string) {
    setSource(nextSource as ConversationLogSource)
    setPage(1)
  }

  function updateSearch(nextSearch: string) {
    setSearch(nextSearch)
    setPage(1)
  }

  function updateAgent(nextAgentId: string) {
    setAgentId(nextAgentId === "all" ? null : Number(nextAgentId))
    setPage(1)
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-white text-slate-950">
      <header className="border-b border-slate-200 bg-white px-6 py-6 lg:px-10">
        <div className="max-w-2xl">
          <h1 className="text-2xl font-semibold tracking-normal">
            {t("conversationLogs.title")}
          </h1>
          <p className="mt-2 max-w-xl text-sm leading-5 text-slate-500">
            {t("conversationLogs.subtitle")}
          </p>
        </div>

        <div className="mt-6 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex flex-wrap gap-2">
            {tabs.map((item) => {
              const active = item.id === source
              return (
                <button
                  key={item.id}
                  type="button"
                  aria-pressed={active}
                  onClick={() => updateSource(item.id)}
                  className={cn(
                    "inline-flex h-8 items-center gap-2 rounded-full border px-3 text-sm font-medium transition-colors",
                    active
                      ? "border-blue-600 bg-blue-600 text-white shadow-sm"
                      : "border-slate-200 bg-white text-slate-600 hover:border-blue-200 hover:bg-blue-50"
                  )}
                >
                  {!active && item.id !== "all" ? (
                    <span className="h-1.5 w-1.5 rounded-full bg-blue-500" />
                  ) : null}
                  {item.label}
                </button>
              )
            })}
          </div>

          <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
            <select
              value={agentId ?? "all"}
              onChange={(event) => updateAgent(event.target.value)}
              className="h-9 min-w-44 rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-700 outline-none transition-colors focus:border-blue-400"
              aria-label={t("conversationLogs.agentFilterLabel")}
            >
              <option value="all">{t("conversationLogs.agentFilterAll")}</option>
              {(listData?.agents ?? []).map((agent) => (
                <option key={agent.agent_id} value={agent.agent_id}>
                  {agent.agent_name}
                </option>
              ))}
            </select>
            <SearchInput
              value={search}
              onChange={updateSearch}
              placeholder={t("conversationLogs.searchPlaceholder")}
              containerClassName="w-full sm:w-64"
              className="h-9 rounded-lg bg-white text-sm"
            />
          </div>
        </div>
      </header>

      <main className="grid min-h-0 flex-1 grid-cols-1 overflow-y-auto md:grid-cols-[minmax(320px,372px)_minmax(0,1fr)] md:overflow-hidden">
        <section className="flex min-h-0 max-h-[48vh] flex-col border-r border-slate-200 bg-white md:max-h-none">
          <div className="min-h-0 flex-1 overflow-y-auto">
            {isLoading && !listData ? (
              <div className="px-4 py-6 text-sm text-slate-500">
                {t("common.loading")}
              </div>
            ) : error ? (
              <div className="px-4 py-6 text-sm text-rose-600">{error}</div>
            ) : logs.length === 0 ? (
              <div className="px-4 py-10">
                <div className="rounded-md border border-dashed border-slate-300 bg-slate-50 px-4 py-8 text-center">
                  <Inbox className="mx-auto mb-3 h-7 w-7 text-slate-400" />
                  <div className="font-medium text-slate-800">
                    {hasFilter
                      ? t("conversationLogs.empty.filteredTitle")
                      : t("conversationLogs.empty.title")}
                  </div>
                </div>
              </div>
            ) : (
              <div className="divide-y divide-slate-100">
                {logs.map((log) => {
                  const selected = selectedTaskId === log.task_id
                  return (
                    <button
                      key={log.task_id}
                      type="button"
                      onClick={() => selectLog(log.task_id)}
                      className={cn(
                        "relative block w-full px-5 py-3 text-left transition-colors",
                        selected
                          ? "bg-blue-50/60 before:absolute before:inset-y-0 before:left-0 before:w-0.5 before:bg-blue-600"
                          : "hover:bg-slate-50"
                      )}
                    >
                      <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3">
                        <div className="flex min-w-0 items-center gap-2">
                          <span className="truncate text-sm font-semibold text-slate-950">
                            {conversationTitle(log) || t("conversationLogs.untitled")}
                          </span>
                          <Badge
                            variant="outline"
                            className={cn("h-5 rounded px-1.5 text-[11px]", SOURCE_BADGE_CLASS)}
                          >
                            {sourceLabel(log)}
                          </Badge>
                        </div>
                        <span
                          suppressHydrationWarning
                          className="whitespace-nowrap text-xs text-slate-500"
                        >
                          {formatRelativeTime(log.last_activity_at || log.updated_at, locale)}
                        </span>
                      </div>
                    </button>
                  )
                })}
              </div>
            )}
          </div>

          <div
            className={cn(
              "flex items-center justify-between border-t border-slate-100 px-4 py-3",
              (pagination?.total_pages ?? 1) <= 1 && "hidden"
            )}
          >
            <Button
              variant="outline"
              size="sm"
              disabled={page <= 1 || isLoading}
              onClick={() => setPage((value) => Math.max(1, value - 1))}
            >
              <ChevronLeft className="h-4 w-4" />
              {t("common.back")}
            </Button>
            <span className="text-xs text-slate-500">
              {page} / {pagination?.total_pages ?? 1}
            </span>
            <Button
              variant="outline"
              size="sm"
              disabled={page >= (pagination?.total_pages ?? 1) || isLoading}
              onClick={() => setPage((value) => value + 1)}
            >
              {t("common.next")}
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </section>

        <section className="min-h-[52vh] overflow-y-auto bg-white md:min-h-0">
          {!selectedTaskId ? (
            <div className="flex h-full items-center justify-center px-6 text-center text-sm text-slate-500">
              {t("conversationLogs.empty.detail")}
            </div>
          ) : isDetailLoading && !detail ? (
            <div className="px-6 py-6 text-sm text-slate-500">{t("common.loading")}</div>
          ) : detailError ? (
            <div className="px-6 py-6 text-sm text-rose-600">{detailError}</div>
          ) : detail ? (
            <ConversationLogDetail detail={detail} />
          ) : null}
        </section>
      </main>
    </div>
  )
}

function ConversationLogDetail({ detail }: { detail: ConversationLogDetailResponse }) {
  const { locale, t } = useI18n()
  const log = detail.log

  return (
    <div className="flex min-h-full flex-col">
      <span className="sr-only">{t("conversationLogs.readOnly")}</span>
      <div className="border-b border-slate-200 px-6 py-5 md:px-7">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="break-words text-xl font-semibold tracking-normal text-slate-950">
            {conversationTitle(log) || t("conversationLogs.untitled")}
          </h2>
          <Badge
            variant="outline"
            className={cn("h-5 rounded px-1.5 text-[11px]", SOURCE_BADGE_CLASS)}
          >
            {sourceLabel(log)}
          </Badge>
        </div>
        <p className="mt-2 text-sm text-slate-600">
          {t("conversationLogs.lastActivity")}{" "}
          <span suppressHydrationWarning className="text-slate-500">
            {formatRelativeTime(log.last_activity_at || log.updated_at, locale)}
          </span>
        </p>
      </div>

      <section className="flex-1 px-6 py-6 md:px-7">
        {detail.transcript.length === 0 ? (
          <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50 px-4 py-8 text-center text-sm text-slate-500">
            {t("conversationLogs.empty.transcript")}
          </div>
        ) : (
          <div className="space-y-4">
            {detail.transcript.map((message) => (
              message.message_type === "compaction" ? (
                <CompactionNotice
                  key={message.id}
                  text={t("agent.logs.event.messages.compactNotice", {
                    original: message.compaction?.original_tokens ?? "?",
                    compacted: message.compaction?.compacted_tokens ?? "?",
                  })}
                />
              ) : (
              <div
                key={message.id}
                className={cn(
                  "flex items-start gap-3",
                  message.role === "user" ? "justify-end" : "justify-start"
                )}
              >
                {message.role !== "user" ? (
                  <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-blue-50 text-blue-600">
                    <Bot className="h-4 w-4" />
                  </div>
                ) : null}
                <div
                  className={cn(
                    "max-w-[78%] rounded-xl px-4 py-3 text-sm leading-6 shadow-sm",
                    message.role === "user"
                      ? "rounded-br-sm bg-blue-600 text-white"
                      : "rounded-bl-sm border border-slate-200 bg-white text-slate-900"
                  )}
                >
                  {message.role === "user" ? (
                    <p className="whitespace-pre-wrap break-words">{message.content}</p>
                  ) : (
                    <MarkdownRenderer
                      content={message.content}
                      className="prose-sm break-words [overflow-wrap:anywhere]"
                    />
                  )}
                </div>
                {message.role === "user" ? (
                  <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-slate-100 text-xs font-semibold text-slate-500">
                    U
                  </div>
                ) : null}
              </div>
              )
            ))}
          </div>
        )}
        {detail.trace_events && detail.trace_events.length > 0 ? (
          <div className="mt-6 border-t border-slate-200 pt-4">
            <TraceEventRenderer events={detail.trace_events as any} taskStatus={log.status} />
          </div>
        ) : null}
      </section>
    </div>
  )
}

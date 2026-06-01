"use client"

import React, { useEffect, useMemo, useState } from "react"
import { useParams, useRouter } from "next/navigation"
import { toast } from "sonner"
import { useI18n } from "@/contexts/i18n-context"
import {
  applyWorkforceChanges,
  getWorkforce,
  getWorkforceBuilderMessages,
  proposeWorkforceChanges,
} from "@/lib/workforces-api"
import type {
  WorkforceBuilderMessage,
  WorkforceBuilderPatch,
  WorkforceDetail,
  WorkforceRunResponse,
} from "@/types/workforce"
import {
  ProposedPatchCard,
  WorkforceBuilderChat,
  WorkforceSummary,
  WorkforceTestPanel,
} from "@/components/workforce"
import { getBuilderReadOnlyReason, getRunDisabledReason } from "../../workforce-ui-state"

function latestProposedAssistantMessage(
  messages: WorkforceBuilderMessage[],
): WorkforceBuilderMessage | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const item = messages[index]
    if (item.role === "assistant" && item.proposed_patch) {
      return item
    }
  }
  return null
}

export default function WorkforceBuilderPage() {
  const { t } = useI18n()
  const router = useRouter()
  const params = useParams()
  const id = Array.isArray(params.id) ? params.id[0] : params.id
  const [workforce, setWorkforce] = useState<WorkforceDetail | null>(null)
  const [messages, setMessages] = useState<WorkforceBuilderMessage[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [applying, setApplying] = useState(false)

  const activeProposal = useMemo(() => latestProposedAssistantMessage(messages), [messages])

  useEffect(() => {
    if (!id) return

    const load = async () => {
      try {
        setLoading(true)
        setError(null)
        const [workforceData, historyData] = await Promise.all([
          getWorkforce(id),
          getWorkforceBuilderMessages(id),
        ])
        setWorkforce(workforceData)
        setMessages(historyData.items)
      } catch (err) {
        const nextError = err instanceof Error ? err.message : t("workforces.errors.loadBuilder")
        setError(nextError)
        toast.error(nextError)
      } finally {
        setLoading(false)
      }
    }

    void load()
  }, [id, t])

  const handleSubmit = async (message: string) => {
    if (!id) return
    try {
      setSubmitting(true)
      const result = await proposeWorkforceChanges(id, { message })
      const history = await getWorkforceBuilderMessages(id)
      setMessages(history.items)
      toast.success(result.assistant_message || t("workforces.messages.proposalCreated"))
    } catch (err) {
      const nextError = err instanceof Error ? err.message : t("workforces.errors.proposeChanges")
      toast.error(nextError)
    } finally {
      setSubmitting(false)
    }
  }

  const handleApply = async (messageId: number, patch: WorkforceBuilderPatch) => {
    if (!id) return
    try {
      setApplying(true)
      const result = await applyWorkforceChanges(id, {
        message_id: messageId,
        proposed_patch: patch,
      })
      setWorkforce(result.workforce)
      const history = await getWorkforceBuilderMessages(id)
      setMessages(history.items)
      toast.success(t("workforces.messages.changesApplied"))
    } catch (err) {
      const nextError = err instanceof Error ? err.message : t("workforces.errors.applyChanges")
      toast.error(nextError)
    } finally {
      setApplying(false)
    }
  }

  const handleRunCreated = (result: WorkforceRunResponse) => {
    router.push(result.redirect_url || `/task/${result.task_id}`)
  }

  if (loading) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">{t("workforces.loading.builder")}</div>
  if (error) return <div className="h-full overflow-y-auto p-4 text-red-500 sm:p-8">{error}</div>
  if (!workforce) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">{t("workforces.errors.notFound")}</div>

  const builderReadOnlyReason = getBuilderReadOnlyReason(workforce.status, t)
  const runDisabledReason = getRunDisabledReason(workforce.status, t)

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto grid w-full gap-6 p-4 sm:p-8 xl:grid-cols-[0.95fr_1.05fr_0.8fr]">
        <div className="min-h-[640px]">
          <WorkforceBuilderChat
            messages={messages}
            loading={loading}
            submitting={submitting}
            readOnly={Boolean(builderReadOnlyReason)}
            readOnlyReason={builderReadOnlyReason || undefined}
            onSubmit={handleSubmit}
          />
        </div>
        <div className="space-y-6">
          <WorkforceSummary workforce={workforce} />
          <ProposedPatchCard
            patch={activeProposal?.proposed_patch ?? null}
            messageId={activeProposal?.id ?? null}
            status={activeProposal?.status ?? null}
            applying={applying}
            readOnly={Boolean(builderReadOnlyReason)}
            onApply={handleApply}
          />
        </div>
        <div>
          <WorkforceTestPanel
            workforceId={workforce.id}
            disabled={Boolean(runDisabledReason)}
            disabledReason={runDisabledReason || undefined}
            onRunCreated={handleRunCreated}
          />
        </div>
      </div>
    </div>
  )
}

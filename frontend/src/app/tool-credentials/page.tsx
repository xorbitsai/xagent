"use client"

import React, { useState } from "react"
import { useSearchParams } from "next/navigation"
import { KeyRound } from "lucide-react"

import { ToolCredentialsPanel } from "@/components/tools/tool-credentials-panel"
import { useAuth } from "@/contexts/auth-context"
import { useI18n } from "@/contexts/i18n-context"

export default function ToolCredentialsPage() {
  const { t } = useI18n()
  const { user } = useAuth()
  const searchParams = useSearchParams()
  const initialToolName = searchParams.get("tool")
  const [instanceCredentialsAvailable, setInstanceCredentialsAvailable] = useState<
    boolean | null
  >(null)
  const isAdmin = Boolean(user?.is_admin)

  return (
    <div className="h-full w-full overflow-y-auto p-8">
      <div className="mb-8 flex items-center gap-3">
        <div className="rounded-md bg-muted p-2">
          <KeyRound className="h-5 w-5 text-slate-600" />
        </div>
        <div>
          <h1 className="text-3xl font-bold">{t("tools.configHeader.title")}</h1>
          <p className="text-muted-foreground">{t("tools.configHeader.description")}</p>
        </div>
      </div>
      <div className="space-y-8">
        <section className="space-y-4">
          <div>
            <h2 className="text-xl font-semibold">{t("tools.credentials.personalTitle")}</h2>
            <p className="text-sm text-muted-foreground">
              {t("tools.credentials.personalDescription")}
            </p>
          </div>
          <ToolCredentialsPanel
            scope="user"
            initialToolName={initialToolName}
            showTitle={false}
          />
        </section>

        {isAdmin && (
          <section className="space-y-4">
            {instanceCredentialsAvailable !== false && (
              <div>
                <h2 className="text-xl font-semibold">
                  {t("tools.credentials.instanceTitle")}
                </h2>
                <p className="text-sm text-muted-foreground">
                  {t("tools.credentials.instanceDescription")}
                </p>
              </div>
            )}
            <ToolCredentialsPanel
              scope="instance"
              initialToolName={initialToolName}
              onAvailabilityChange={setInstanceCredentialsAvailable}
              showTitle={false}
            />
          </section>
        )}
      </div>
    </div>
  )
}

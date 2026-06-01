"use client"

import React, { useState } from "react"
import { useRouter } from "next/navigation"
import { ArrowLeft, Sparkles, Settings2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { useI18n } from "@/contexts/i18n-context"
import { WorkforcePromptCreator } from "@/components/workforce/workforce-prompt-creator"
import { WorkforceWizard } from "@/components/workforce/workforce-wizard"

type Mode = "prompt" | "manual"

export default function NewWorkforcePage() {
  const router = useRouter()
  const { t } = useI18n()
  const [mode, setMode] = useState<Mode>("prompt")

  const onCreated = (workforce: { id: number }) => {
    router.push(`/workforces/${workforce.id}`)
  }

  const onBack = () => {
    router.push("/workforces")
  }

  const modes: { key: Mode; icon: React.ReactNode; title: string; subtitle: string; description: string }[] = [
    {
      key: "prompt",
      icon: <Sparkles className="h-5 w-5" />,
      title: t("workforces.create.modeSelect.aiTitle"),
      subtitle: t("workforces.create.modeSelect.aiSubtitle"),
      description: t("workforces.create.prompt.description"),
    },
    {
      key: "manual",
      icon: <Settings2 className="h-5 w-5" />,
      title: t("workforces.create.modeSelect.manualTitle"),
      subtitle: t("workforces.create.modeSelect.manualSubtitle"),
      description: t("workforces.create.manual.description"),
    },
  ]

  if (mode === "manual") {
    return (
      <WorkforceWizard
        onCreated={onCreated}
        onBack={() => setMode("prompt")}
      />
    )
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto flex w-full flex-col gap-6 p-4 sm:p-8">
        <div className="space-y-1">
          <Button variant="ghost" className="w-fit gap-2 px-0" onClick={onBack}>
            <ArrowLeft className="h-4 w-4" />
            {t("workforces.create.backToWorkforces")}
          </Button>
          <h1 className="text-3xl font-bold">{t("workforces.create.title")}</h1>
          <p className="text-muted-foreground">{t("workforces.create.description")}</p>
        </div>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 sm:max-w-3xl">
          {modes.map((m) => (
            <button
              key={m.key}
              type="button"
              onClick={() => setMode(m.key)}
              className={cn(
                "relative flex flex-col gap-3 rounded-xl border p-5 text-left transition-all hover:border-primary/60 hover:shadow-sm",
                mode === m.key
                  ? "border-primary bg-primary/5 shadow-sm ring-1 ring-primary"
                  : "border-border bg-card"
              )}
            >
              {m.key === "prompt" ? (
                <span className="absolute right-3 top-3 rounded-full bg-primary px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-primary-foreground">
                  {m.subtitle}
                </span>
              ) : (
                <span className="absolute right-3 top-3 text-xs text-muted-foreground">
                  {m.subtitle}
                </span>
              )}
              <div className={cn(
                "flex h-9 w-9 items-center justify-center rounded-lg",
                mode === m.key ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground"
              )}>
                {m.icon}
              </div>
              <div>
                <div className="font-semibold">{m.title}</div>
                <div className="mt-1 text-sm text-muted-foreground">{m.description}</div>
              </div>
            </button>
          ))}
        </div>

        <WorkforcePromptCreator
          onCreated={onCreated}
          onCancel={onBack}
        />
      </div>
    </div>
  )
}

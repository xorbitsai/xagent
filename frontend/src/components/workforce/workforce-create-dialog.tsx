"use client"

import React, { useState } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { WorkforcePromptCreator } from "./workforce-prompt-creator"
import { WorkforceWizard } from "./workforce-wizard"
import { useI18n } from "@/contexts/i18n-context"
import type { WorkforceDetail } from "@/types/workforce"
import { Sparkles, Settings2 } from "lucide-react"

interface WorkforceCreateDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onCreated: (workforce: WorkforceDetail) => void
}

export function WorkforceCreateDialog({ open, onOpenChange, onCreated }: WorkforceCreateDialogProps) {
  const { t } = useI18n()
  const [mode, setMode] = useState<"prompt" | "manual">("prompt")

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px] max-h-[90vh] overflow-y-auto flex flex-col">
        <DialogHeader>
          <DialogTitle className="text-2xl">{t("workforces.create.title")}</DialogTitle>
          <DialogDescription className="text-base">{t("workforces.create.description")}</DialogDescription>
        </DialogHeader>

        <Tabs value={mode} onValueChange={(v) => setMode(v as "prompt" | "manual")} className="flex-1 flex flex-col">
          <TabsList className="grid w-full grid-cols-2 mb-4">
            <TabsTrigger value="prompt" className="flex items-center gap-2">
              <Sparkles className="h-4 w-4" />
              {t("workforces.create.modeSelect.aiTitle")}
            </TabsTrigger>
            <TabsTrigger value="manual" className="flex items-center gap-2">
              <Settings2 className="h-4 w-4" />
              {t("workforces.create.modeSelect.manualTitle")}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="prompt" className="flex-1 mt-0 data-[state=inactive]:hidden" forceMount>
            <WorkforcePromptCreator onCreated={onCreated} onCancel={() => onOpenChange(false)} />
          </TabsContent>

          <TabsContent value="manual" className="flex-1 mt-0 data-[state=inactive]:hidden" forceMount>
            <WorkforceWizard onCreated={onCreated} onBack={() => onOpenChange(false)} />
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}

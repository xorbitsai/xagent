import React, { useState } from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { useI18n } from "@/contexts/i18n-context"
import { AlertCircle, Link2, Upload, CheckCircle2, Database } from "lucide-react"
import { MultiSelect } from "@/components/ui/multi-select"

interface TemplatePrerequisiteModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  unconnectedMcps: string[]
  needsKb: boolean
  availableKbs: { value: string; label: string }[]
  selectedKbs: string[]
  onKbsChange: (kbs: string[]) => void
  onConnectMcp: () => void
  onUploadKb: () => void
}

export function TemplatePrerequisiteModal({
  open,
  onOpenChange,
  unconnectedMcps,
  needsKb,
  availableKbs,
  selectedKbs,
  onKbsChange,
  onConnectMcp,
  onUploadKb
}: TemplatePrerequisiteModalProps) {
  const { t } = useI18n()

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-xl">
            <AlertCircle className="h-5 w-5 text-blue-500" />
            {t("builds.prerequisites.title") || "Setup Prerequisites"}
          </DialogTitle>
          <DialogDescription className="text-base">
            {t("builds.prerequisites.description") || "To fully utilize this template's capabilities, please complete the following configurations first:"}
          </DialogDescription>
        </DialogHeader>

        <div className="py-4 space-y-4">
          {unconnectedMcps.length > 0 && (
            <div className="flex flex-col gap-2 p-4 border rounded-lg bg-slate-50/50 dark:bg-slate-900/50">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Link2 className="h-5 w-5 text-muted-foreground" />
                  <span className="font-medium">{t("builds.prerequisites.mcpTitle") || "Connect Required Tools"}</span>
                </div>
                <Button variant="outline" size="sm" onClick={() => {
                  onConnectMcp();
                }}>
                  {t("builds.prerequisites.connectBtn") || "Connect"}
                </Button>
              </div>
              <div className="text-sm text-muted-foreground pl-7">
                {t("builds.prerequisites.mcpDesc") || "This template requires:"} <span className="font-medium text-foreground">{unconnectedMcps.join(', ')}</span>
              </div>
            </div>
          )}

          {needsKb && (
            <div className="flex flex-col gap-4 p-4 border rounded-lg bg-slate-50/50 dark:bg-slate-900/50">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Database className="h-5 w-5 text-muted-foreground" />
                  <span className="font-medium">{t("builds.prerequisites.kbTitle") || "Upload Knowledge Base"}</span>
                </div>
                <Button variant="outline" size="sm" onClick={() => {
                  onUploadKb();
                }}>
                  <Upload className="h-4 w-4 mr-2" />
                  {t("builds.prerequisites.uploadBtn") || "Upload New"}
                </Button>
              </div>
              <div className="text-sm text-muted-foreground pl-7">
                {t("builds.prerequisites.kbDesc") || "Upload necessary references or documents that this template can utilize."}
              </div>
              <div className="pl-7 pr-2">
                <MultiSelect
                  values={selectedKbs || []}
                  onValuesChange={onKbsChange}
                  options={availableKbs}
                  placeholder={t("builds.prerequisites.selectKbPlaceholder") || "Or select existing knowledge bases..."}
                />
              </div>
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            {t("builds.prerequisites.skipBtn") || "Skip for now"}
          </Button>
          <Button onClick={() => onOpenChange(false)} className="gap-2">
            <CheckCircle2 className="h-4 w-4" />
            {t("builds.prerequisites.doneBtn") || "Got it"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

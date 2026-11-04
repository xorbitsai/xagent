"use client"

import React, { type ReactNode } from "react"
import { Button } from "@/components/ui/button"
import { XIcon } from "lucide-react"

interface PreviewSheetProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  title?: ReactNode
  actions?: ReactNode
  children?: ReactNode
}

export function PreviewSheet({ open, onOpenChange, title, actions, children }: PreviewSheetProps) {
  if (!open) return null
  return (
    <div className="w-full border rounded-xl overflow-hidden bg-background flex flex-col h-[100%] shadow-lg">
      <div data-slot="sheet-header" className="flex flex-col gap-1.5 p-4 flex-shrink-0 bg-background/80 backdrop-blur-sm border-b">
        <div className="flex items-center justify-between">
          <h3 data-slot="sheet-title" className="text-foreground font-semibold flex items-center gap-2">
            {title}
          </h3>
          <div className="flex items-center gap-2">
            {actions}
            <Button
              variant="outline"
              size="sm"
              onClick={() => onOpenChange(false)}
              className="h-8 w-8 p-0"
              aria-label="Close"
            >
              <XIcon className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
      <div className="flex-1 overflow-hidden flex flex-col min-h-0">
        <div className="flex-1 overflow-hidden bg-muted/30">
          {children}
        </div>
      </div>
    </div>
  )
}

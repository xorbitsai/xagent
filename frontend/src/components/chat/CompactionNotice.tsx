import React from "react"
import { Shrink } from "lucide-react"

/** Inline "context compacted" notice, centered and muted. Shared by the live
 *  conversation timeline and the conversation-logs transcript. */
export function CompactionNotice({ text }: { text: string }) {
  return (
    <div className="flex justify-center py-1">
      <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
        <Shrink className="w-3.5 h-3.5" />
        {text}
      </span>
    </div>
  )
}

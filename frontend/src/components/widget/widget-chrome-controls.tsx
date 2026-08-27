"use client"

import React, { useEffect, useRef, useState } from "react"
import { MessageSquarePlus, MoreHorizontal, X } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"

// The host page's widget.js owns panel visibility; it has no direct handle
// into this iframe's React tree, so close intent is signalled back over
// postMessage instead. The host is an arbitrary third-party origin from in
// here, so targetOrigin can't be pinned tighter than "*" -- this carries no
// sensitive payload, unlike the parent -> iframe session protocol.
const postCloseToParent = () => {
  window.parent.postMessage({ xagent: true, v: 1, type: "widget_close" }, "*")
}

const iconButtonClassName =
  "p-2 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors"

interface WidgetChromeControlsProps {
  // Undefined hides the "..." menu entirely -- it would otherwise open onto
  // nothing. Each host page supplies its own already-resolved label/handler
  // since "new conversation" means something different (and is disabled/
  // pending differently) in guest vs. Session mode.
  newConversation?: {
    label: string
    onClick: () => void
    disabled?: boolean
  }
}

export function WidgetChromeControls({ newConversation }: WidgetChromeControlsProps) {
  const { t } = useI18n()
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!isMenuOpen) return

    const handlePointerDown = (event: PointerEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsMenuOpen(false)
      }
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setIsMenuOpen(false)
    }
    // widget.js can hide this whole iframe from the host page's own FAB, a
    // click entirely outside this document that the two listeners above
    // never see. The iframe is never unmounted, so without this the menu
    // would still be open the next time the panel is shown.
    const handleBlur = () => setIsMenuOpen(false)

    document.addEventListener("pointerdown", handlePointerDown)
    document.addEventListener("keydown", handleKeyDown)
    window.addEventListener("blur", handleBlur)
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown)
      document.removeEventListener("keydown", handleKeyDown)
      window.removeEventListener("blur", handleBlur)
    }
  }, [isMenuOpen])

  const handleNewConversation = () => {
    setIsMenuOpen(false)
    newConversation?.onClick()
  }

  const handleClose = () => {
    setIsMenuOpen(false)
    postCloseToParent()
  }

  return (
    <div className="ml-auto flex items-center gap-1">
      {newConversation ? (
        <div ref={menuRef} className="relative">
          <button
            type="button"
            onClick={() => setIsMenuOpen((open) => !open)}
            title={t("widgetChat.moreOptions")}
            aria-label={t("widgetChat.moreOptions")}
            aria-haspopup="menu"
            aria-expanded={isMenuOpen}
            className={iconButtonClassName}
          >
            <MoreHorizontal className="w-4 h-4" />
          </button>
          {isMenuOpen ? (
            <div
              role="menu"
              className="absolute right-0 top-full z-20 mt-1 w-max max-w-xs rounded-md border bg-popover py-1 text-popover-foreground shadow-md"
            >
              <button
                type="button"
                role="menuitem"
                onClick={handleNewConversation}
                disabled={newConversation.disabled}
                className="flex w-full items-center gap-2 whitespace-nowrap px-3 py-2 text-left text-sm hover:bg-muted/50 transition-colors disabled:pointer-events-none disabled:opacity-50"
              >
                <MessageSquarePlus className="w-4 h-4 shrink-0" />
                {newConversation.label}
              </button>
            </div>
          ) : null}
        </div>
      ) : null}
      <button
        type="button"
        onClick={handleClose}
        title={t("widgetChat.close")}
        aria-label={t("widgetChat.close")}
        className={iconButtonClassName}
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  )
}

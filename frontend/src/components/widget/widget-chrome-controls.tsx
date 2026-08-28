"use client"

import React, { useEffect, useRef, useState } from "react"
import { Loader2, MessageSquarePlus, MoreHorizontal, X } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { postToParentWidget } from "@/lib/widget-parent-message"

// Exported so the standalone share-mode reset button (public-agent-chat-page.tsx)
// can match this styling instead of drifting with its own duplicate string.
export const iconButtonClassName =
  "p-2 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors "
  + "disabled:pointer-events-none disabled:opacity-50 "
  + "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"

interface WidgetChromeControlsProps {
  // Undefined hides the "..." menu entirely -- it would otherwise open onto
  // nothing. Each host page supplies its own already-resolved label/handler
  // since "new conversation" means something different (and is disabled/
  // pending differently) in guest vs. Session mode.
  newConversation?: {
    label: string
    onClick: () => void
    disabled?: boolean
    // The menu closes as soon as an item is clicked (standard menu UX), so a
    // pending round-trip (e.g. Session mode's reset) would otherwise show no
    // feedback at all unless the visitor happens to reopen the menu during
    // it. Surfaced as a spinner on the trigger itself instead, which stays
    // visible with the menu closed.
    pending?: boolean
  }
}

export function WidgetChromeControls({ newConversation }: WidgetChromeControlsProps) {
  const { t } = useI18n()
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

  // This component rendering at all *is* "the iframe has a close control" --
  // widget.js's mobile full-screen FAB-hiding guard keys off exactly this
  // signal so it never hides its own fallback close button while this one
  // isn't actually mounted (a loading/auth-failure/degraded/terminal state
  // renders no header, and thus no instance of this component, at all).
  useEffect(() => {
    postToParentWidget("widget_chrome_ready")
    return () => postToParentWidget("widget_chrome_not_ready")
  }, [])

  useEffect(() => {
    if (!isMenuOpen) return

    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null
      // A target already detached from the tree (removed by some other
      // handler earlier in the same dispatch) fails Node.contains() the
      // same way a genuine outside click would; don't misread that as one.
      if (target && !target.isConnected) return
      if (menuRef.current && !menuRef.current.contains(target)) {
        setIsMenuOpen(false)
      }
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setIsMenuOpen(false)
    }
    // widget.js can hide this whole iframe from the host page's own FAB, a
    // click entirely outside this document that the two listeners above
    // never see. The iframe is never unmounted, so without this the menu
    // would still be open the next time the panel is shown. Guarded the same
    // way as widget.js's own onWindowBlur: focus moving into a child of this
    // same document (e.g. the chat composer autofocusing) also fires a
    // window blur without the panel actually being hidden.
    const handleBlur = () => {
      if (document.hasFocus()) return
      setIsMenuOpen(false)
    }

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
    postToParentWidget("widget_close")
  }

  return (
    <div className="ml-auto flex items-center gap-1">
      {newConversation ? (
        <div ref={menuRef} className="relative">
          <button
            type="button"
            onClick={() => setIsMenuOpen((open) => !open)}
            disabled={newConversation.pending}
            title={t("widgetChat.moreOptions")}
            aria-label={t("widgetChat.moreOptions")}
            aria-haspopup="menu"
            aria-expanded={isMenuOpen}
            className={iconButtonClassName}
          >
            {newConversation.pending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <MoreHorizontal className="w-4 h-4" />
            )}
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
                className="flex w-full items-center gap-2 whitespace-nowrap px-3 py-2 text-left text-sm hover:bg-muted/50 transition-colors disabled:pointer-events-none disabled:opacity-50 focus-visible:bg-muted/50 focus-visible:outline-none"
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

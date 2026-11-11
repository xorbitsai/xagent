"use client"

import React, { useCallback, useEffect, useRef, useState } from "react"
import { History, Loader2, Maximize2, MessageSquarePlus, Minimize2, MoreHorizontal, X } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import {
  listenForWidgetExpandState,
  listenForWidgetHostMessage,
  postToParentWidget,
} from "@/lib/widget-parent-message"

// Exported so the standalone share-mode reset button (public-agent-chat-page.tsx)
// can match this styling instead of drifting with its own duplicate string.
export const iconButtonClassName =
  "p-2 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors "
  + "disabled:pointer-events-none disabled:opacity-50 "
  + "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"

interface WidgetChromeControlsProps {
  // Undefined omits the new-conversation menu item -- expand/collapse is
  // always offered regardless, so the "..." trigger itself always renders.
  // Each host page supplies its own already-resolved label/handler since
  // "new conversation" means something different (and is disabled/pending
  // differently) in guest vs. Session mode.
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
  // Undefined omits the history menu item. Only the Session page supplies it,
  // and only while the session server offers conversation history.
  history?: {
    label: string
    onClick: () => void
  }
  // Expand/collapse state is owned here (uncontrolled) by default. Session
  // mode's header can remount independently of the panel (reconnect swaps its
  // content to an error subtree); widget.js answers each remount's
  // widget_chrome_ready with the current size, but until that reply lands, and
  // with a widget.js older than that reply, a fresh mount would show the
  // collapsed label on an expanded panel. Passing both props lets that caller
  // keep the state somewhere that doesn't unmount. onExpandedChange is called
  // for the visitor's own clicks and for every size widget.js reports
  // (widget_expand_state, widget_expand_rejected).
  expanded?: boolean
  onExpandedChange?: (expanded: boolean) => void
}

export function WidgetChromeControls({
  newConversation,
  history,
  expanded: controlledExpanded,
  onExpandedChange,
}: WidgetChromeControlsProps) {
  const { t } = useI18n()
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [internalExpanded, setInternalExpanded] = useState(false)
  const isControlled = controlledExpanded !== undefined
  const isExpanded = isControlled ? controlledExpanded : internalExpanded
  const setIsExpanded = useCallback((next: boolean) => {
    if (isControlled) {
      onExpandedChange?.(next)
    } else {
      setInternalExpanded(next)
    }
  }, [isControlled, onExpandedChange])
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

  useEffect(() => (
    // Correct the optimistic setIsExpanded(true) below when widget.js's own
    // mobile guard rejects the widget_expand that caused it -- otherwise the
    // menu keeps reading "Collapse window" with nothing actually expanded,
    // and the next click (a no-op collapse from here) needs a second click
    // after an eventual desktop-width resize to actually expand. Kept beside
    // the widget_expand_state listener below: an older widget.js sends only
    // this message, and widget.js's resize-to-mobile collapse still does.
    listenForWidgetHostMessage("widget_expand_rejected", () => setIsExpanded(false))
  ), [setIsExpanded])

  useEffect(() => (
    // widget.js reports the panel's actual size after every size request,
    // including this menu's own widget_expand/widget_collapse and the host
    // page's XagentWidget.open({ expanded }) / data-expanded, and in reply to
    // widget_chrome_ready when this component mounts, so the menu follows the
    // panel's actual size rather than its own guess.
    listenForWidgetExpandState((expanded) => setIsExpanded(expanded))
  ), [setIsExpanded])

  const handleNewConversation = () => {
    setIsMenuOpen(false)
    newConversation?.onClick()
  }

  const handleHistory = () => {
    setIsMenuOpen(false)
    history?.onClick()
  }

  const handleToggleExpand = () => {
    setIsMenuOpen(false)
    // Optimistic so the menu responds immediately: widget.js owns the actual
    // panel size and reports it back after every click (widget_expand_state,
    // plus widget_expand_rejected when its mobile guard refuses an expand),
    // which the two listeners above apply. The postMessage call is a side
    // effect, so it belongs here in the click handler, not inside
    // setIsExpanded's updater (React may invoke that updater more than once,
    // e.g. under StrictMode).
    const next = !isExpanded
    setIsExpanded(next)
    postToParentWidget(next ? "widget_expand" : "widget_collapse")
  }

  const handleClose = () => {
    setIsMenuOpen(false)
    postToParentWidget("widget_close")
  }

  return (
    <div className="ml-auto flex items-center gap-1">
      <div ref={menuRef} className="relative">
        <button
          type="button"
          onClick={() => setIsMenuOpen((open) => !open)}
          // Sizing (expand/collapse) has no dependency on the conversation
          // reset, so a pending reset must not block reaching it -- the
          // reset item already carries its own `disabled` for that action
          // specifically. The spinner below still surfaces reset-in-flight
          // status on the trigger itself.
          title={t("widgetChat.moreOptions")}
          aria-label={t("widgetChat.moreOptions")}
          aria-haspopup="menu"
          aria-expanded={isMenuOpen}
          className={iconButtonClassName}
        >
          {newConversation?.pending ? (
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
            {newConversation ? (
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
            ) : null}
            {history ? (
              <button
                type="button"
                role="menuitem"
                onClick={handleHistory}
                className="flex w-full items-center gap-2 whitespace-nowrap px-3 py-2 text-left text-sm hover:bg-muted/50 transition-colors focus-visible:bg-muted/50 focus-visible:outline-none"
              >
                <History className="w-4 h-4 shrink-0" />
                {history.label}
              </button>
            ) : null}
            <button
              type="button"
              role="menuitem"
              onClick={handleToggleExpand}
              className="flex w-full items-center gap-2 whitespace-nowrap px-3 py-2 text-left text-sm hover:bg-muted/50 transition-colors focus-visible:bg-muted/50 focus-visible:outline-none"
            >
              {isExpanded ? (
                <Minimize2 className="w-4 h-4 shrink-0" data-icon="collapse" />
              ) : (
                <Maximize2 className="w-4 h-4 shrink-0" data-icon="expand" />
              )}
              {isExpanded ? t("widgetChat.collapseWindow") : t("widgetChat.expandWindow")}
            </button>
          </div>
        ) : null}
      </div>
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

"use client"

import React, { useState, useCallback, useEffect } from "react"
import { cn } from "@/lib/utils"
import { GripVertical } from "lucide-react"

interface ResizableSplitLayoutProps {
  leftPanel: React.ReactNode
  rightPanel?: React.ReactNode
  initialLeftWidth?: number // Percentage (0-100)
  minLeftWidth?: number // Percentage (0-100)
  maxLeftWidth?: number // Percentage (0-100)
  className?: string
}

export function ResizableSplitLayout({
  leftPanel,
  rightPanel,
  initialLeftWidth = 50,
  minLeftWidth = 20,
  maxLeftWidth = 80,
  className
}: ResizableSplitLayoutProps) {
  const [leftWidth, setLeftWidth] = useState(initialLeftWidth)
  const [isDragging, setIsDragging] = useState(false)
  const containerRef = React.useRef<HTMLDivElement>(null)
  const rightPanelOpen = rightPanel !== undefined && rightPanel !== null
  const wasRightPanelOpenRef = React.useRef(rightPanelOpen)

  useEffect(() => {
    if (rightPanelOpen && !wasRightPanelOpenRef.current) {
      setLeftWidth(initialLeftWidth)
    }
    wasRightPanelOpenRef.current = rightPanelOpen
  }, [initialLeftWidth, rightPanelOpen])

  const handleMouseDown = useCallback(() => {
    setIsDragging(true)
    document.body.style.cursor = "col-resize"
    document.body.style.userSelect = "none"
  }, [])

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
    document.body.style.cursor = ""
    document.body.style.userSelect = ""
  }, [])

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging || !containerRef.current) return

      const containerRect = containerRef.current.getBoundingClientRect()
      const newLeftWidth =
        ((e.clientX - containerRect.left) / containerRect.width) * 100

      if (newLeftWidth >= minLeftWidth && newLeftWidth <= maxLeftWidth) {
        setLeftWidth(newLeftWidth)
      }
    },
    [isDragging, minLeftWidth, maxLeftWidth]
  )

  useEffect(() => {
    if (isDragging) {
      document.addEventListener("mousemove", handleMouseMove)
      document.addEventListener("mouseup", handleMouseUp)
    } else {
      document.removeEventListener("mousemove", handleMouseMove)
      document.removeEventListener("mouseup", handleMouseUp)
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove)
      document.removeEventListener("mouseup", handleMouseUp)
    }
  }, [isDragging, handleMouseMove, handleMouseUp])

  return (
    <div
      ref={containerRef}
      className={cn("flex w-full h-full overflow-hidden", className)}
    >
      {/* Left Panel */}
      <div
        style={{ width: rightPanelOpen ? `${leftWidth}%` : "100%" }}
        className="h-full overflow-auto"
      >
        {leftPanel}
      </div>

      {/* Resizer Handle */}
      {rightPanelOpen ? (
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize panels"
          className="w-1 bg-border hover:bg-primary/50 cursor-col-resize flex items-center justify-center relative transition-colors group z-10"
          onMouseDown={handleMouseDown}
        >
          <div className="absolute inset-y-0 -left-2 -right-2 z-10 cursor-col-resize" />
          <div className="h-8 w-4 bg-background border rounded flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity shadow-sm">
            <GripVertical className="h-3 w-3 text-muted-foreground" />
          </div>
        </div>
      ) : null}

      {/* Right Panel */}
      {rightPanelOpen ? (
        <div
          style={{ width: `${100 - leftWidth}%` }}
          className="h-full flex-1 overflow-auto"
        >
          {rightPanel}
        </div>
      ) : null}
    </div>
  )
}

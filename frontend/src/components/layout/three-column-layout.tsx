"use client"

import { ReactNode } from "react"
import { cn } from "@/lib/utils"

interface ThreeColumnLayoutProps {
  leftPanel: ReactNode
  centerPanel: ReactNode
  rightPanel: ReactNode
  className?: string
}

export function ThreeColumnLayout({
  leftPanel,
  centerPanel,
  rightPanel,
  className
}: ThreeColumnLayoutProps) {
  return (
    <div className={cn("flex w-full h-full bg-background overflow-hidden", className)}>
      {/* Left Panel - User Interaction & Results */}
      <div className="flex-[1] border-r border-border bg-card/50 backdrop-blur-sm flex flex-col min-w-0 overflow-hidden">
        {leftPanel}
      </div>

      {/* Center Panel - DAG Visualization */}
      <div className="flex-[1.2] border-r border-border bg-background/80 backdrop-blur-sm flex flex-col min-w-0 overflow-hidden">
        {centerPanel}
      </div>

      {/* Right Panel - Step Details */}
      <div className="flex-1 bg-card/50 backdrop-blur-sm flex flex-col min-w-[400px] overflow-hidden">
        {rightPanel}
      </div>
    </div>
  )
}

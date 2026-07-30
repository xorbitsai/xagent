"use client"

import React from "react"
import { usePathname } from "next/navigation"
import { AuthGuard } from "@/components/auth/auth-guard"
import { LayoutContent } from "@/components/layout/layout-content"
import { TaskErrorController } from "@/components/task-error-controller"
import { VoiceInputController } from "@/components/voice-input-controller"
import { AnonymousAuthProvider, AuthProvider } from "@/contexts/auth-context"
import { McpAppsProvider } from "@/contexts/mcp-apps-context"
import { isExternalRoutePath } from "@/lib/auth-pages"

export function ApplicationShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()

  // Do not render a potentially protected route before the client router has
  // resolved its pathname.
  if (!pathname) {
    return null
  }

  // Public widget/share pages must never initialize browser-owned personal
  // authentication.
  if (isExternalRoutePath(pathname)) {
    return <AnonymousAuthProvider>{children}</AnonymousAuthProvider>
  }

  return (
    <AuthProvider>
      <McpAppsProvider>
        <AuthGuard>
          <LayoutContent>{children}</LayoutContent>
          <VoiceInputController />
          <TaskErrorController />
        </AuthGuard>
      </McpAppsProvider>
    </AuthProvider>
  )
}

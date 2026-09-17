"use client"

import React, { createContext, useContext, useState, useEffect } from "react"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import type { AppIntegration } from "@/components/mcp/types"
import { useAuth } from "./auth-context"
import { useI18n } from "./i18n-context"

// auth_type/launch_config/configured_env_keys are picked from AppIntegration
// (components/mcp/types.ts), not redeclared, so this catalog-entry shape
// can't drift from the one the connector dialog and settings dialog share -
// the exact drift that file's own header comment exists to prevent. All
// three are already in the API response.
export interface McpApp
  extends Pick<AppIntegration, "auth_type" | "launch_config" | "configured_env_keys"> {
  id: string
  name: string
  description: string
  icon: string
  users: string
  transport: string
  provider: string
  category: string
  is_connected?: boolean
  server_id?: number
  connected_account?: string
}

interface McpAppsContextType {
  apps: McpApp[]
  isLoading: boolean
  error: string | null
  refresh: () => Promise<void>
  getAppIcon: (name: string) => string | undefined
}

const McpAppsContext = createContext<McpAppsContextType | undefined>(undefined)

export function McpAppsProvider({ children }: { children: React.ReactNode }) {
  const [apps, setApps] = useState<McpApp[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { user } = useAuth()
  const { t } = useI18n()

  const fetchApps = async () => {
    if (!user) return

    setIsLoading(true)
    setError(null)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/mcp/apps`)
      if (response.ok) {
        const data = await response.json()
        setApps(data || [])
      } else {
        setError(t('tools.mcp.dialog.fetchFailed'))
      }
    } catch (err) {
      console.error("Error fetching MCP apps:", err)
      setError(t('tools.mcp.dialog.fetchError'))
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchApps()
  }, [user, t])

  const getAppIcon = (name: string): string | undefined => {
    if (!name) return undefined;

    const lowerName = name.toLowerCase();

    // First try exact match
    const exactMatch = apps.find(app => app.name.toLowerCase() === lowerName || app.id.toLowerCase() === lowerName)
    if (exactMatch?.icon) return exactMatch.icon

    // Then try a more specific partial match based on provider or id prefix
    const partialMatch = apps.find(app => {
      const appLower = app.name.toLowerCase();
      const idLower = app.id.toLowerCase();
      return lowerName.startsWith(appLower + "_") || lowerName.startsWith(idLower + "_");
    })
    if (partialMatch?.icon) return partialMatch.icon

    return undefined
  }

  return (
    <McpAppsContext.Provider value={{ apps, isLoading, error, refresh: fetchApps, getAppIcon }}>
      {children}
    </McpAppsContext.Provider>
  )
}

export function useMcpApps() {
  const context = useContext(McpAppsContext)
  if (context === undefined) {
    throw new Error("useMcpApps must be used within a McpAppsProvider")
  }
  return context
}

"use client"

import { useEffect, useRef, useState } from "react"
import { Link2, Loader2, Unlink } from "lucide-react"
import { Button } from "@/components/ui/button"
import { toast } from "@/components/ui/sonner"
import { useI18n } from "@/contexts/i18n-context"
import {
  connectMcpServer,
  disconnectMcpServer,
  getMcpConnection,
  McpConnectionStatus,
} from "@/lib/mcp-utils"

const POLL_INTERVAL_MS = 1500

interface McpOAuthConnectControlProps {
  serverId: number
  /** Stop click events (e.g. card onClick) from firing when interacting with this control. */
  stopPropagation?: boolean
}

/**
 * Per-user OAuth connect/disconnect control for remote MCP connectors
 * configured with auth type `oauth_mcp`. Handles: initial status fetch,
 * popup-based authorization flow, status polling until connected or the
 * popup is closed, and disconnect.
 */
export function McpOAuthConnectControl({ serverId, stopPropagation = true }: McpOAuthConnectControlProps) {
  const { t } = useI18n()
  const [status, setStatus] = useState<McpConnectionStatus | null>(null)
  const [isLoadingStatus, setIsLoadingStatus] = useState(true)
  const [isConnecting, setIsConnecting] = useState(false)
  const [isDisconnecting, setIsDisconnecting] = useState(false)
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const isMountedRef = useRef(true)

  const clearPoll = () => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current)
      pollTimerRef.current = null
    }
  }

  const refreshStatus = async () => {
    try {
      const { status: nextStatus } = await getMcpConnection(serverId)
      if (isMountedRef.current) setStatus(nextStatus)
      return nextStatus
    } catch (error) {
      console.error("Failed to fetch MCP connection status:", error)
      return null
    }
  }

  useEffect(() => {
    isMountedRef.current = true
    setIsLoadingStatus(true)
    refreshStatus().finally(() => {
      if (isMountedRef.current) setIsLoadingStatus(false)
    })

    return () => {
      isMountedRef.current = false
      clearPoll()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [serverId])

  const handleStopPropagation = (e: React.MouseEvent) => {
    if (stopPropagation) e.stopPropagation()
  }

  const handleConnect = async (e: React.MouseEvent) => {
    handleStopPropagation(e)
    if (isConnecting) return

    setIsConnecting(true)
    try {
      const { authorization_url } = await connectMcpServer(serverId)
      const width = 520
      const height = 680
      const left = window.screenX + (window.outerWidth - width) / 2
      const top = window.screenY + (window.outerHeight - height) / 2
      const popup = window.open(
        authorization_url,
        "mcp-oauth",
        `width=${width},height=${height},left=${left},top=${top},scrollbars=yes`
      )

      if (!popup) {
        toast.error(t('tools.mcp.dialog.oauthPopupBlocked'))
        setIsConnecting(false)
        return
      }

      clearPoll()
      pollTimerRef.current = setInterval(async () => {
        const nextStatus = await refreshStatus()
        if (nextStatus === "connected") {
          clearPoll()
          popup.close()
          if (isMountedRef.current) {
            setIsConnecting(false)
            toast.success(t('tools.mcp.dialog.oauthConnectSuccess'))
          }
        } else if (popup.closed) {
          clearPoll()
          if (isMountedRef.current) setIsConnecting(false)
        }
      }, POLL_INTERVAL_MS)
    } catch (error) {
      console.error("Failed to start MCP OAuth connection:", error)
      toast.error(t('tools.mcp.dialog.oauthConnectFailed'))
      setIsConnecting(false)
    }
  }

  const handleDisconnect = async (e: React.MouseEvent) => {
    handleStopPropagation(e)
    if (isDisconnecting) return

    setIsDisconnecting(true)
    try {
      await disconnectMcpServer(serverId)
      if (isMountedRef.current) setStatus("not_connected")
      toast.success(t('tools.mcp.dialog.oauthDisconnectSuccess'))
    } catch (error) {
      console.error("Failed to disconnect MCP server:", error)
      toast.error(t('tools.mcp.dialog.oauthDisconnectFailed'))
    } finally {
      if (isMountedRef.current) setIsDisconnecting(false)
    }
  }

  if (isLoadingStatus) {
    return (
      <Button variant="outline" size="sm" disabled className="h-7 text-xs">
        <Loader2 className="h-3 w-3 mr-1.5 animate-spin" />
        {t('tools.mcp.dialog.oauthCheckingStatus')}
      </Button>
    )
  }

  const isConnected = status === "connected"
  const needsReconnect = status === "pending" || status === "expired" || status === "error"
  const connectLabel = isConnecting
    ? t('tools.mcp.dialog.oauthConnecting')
    : needsReconnect
      ? t('tools.mcp.dialog.oauthReconnect')
      : t('tools.mcp.dialog.oauthConnect')

  return (
    <div className="flex items-center gap-2" onClick={handleStopPropagation}>
      {isConnected ? (
        <>
          <Button variant="secondary" size="sm" disabled className="h-7 text-xs bg-green-50 text-green-700 hover:bg-green-50">
            <Link2 className="h-3 w-3 mr-1.5" />
            {t('tools.mcp.dialog.oauthConnected')}
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="h-7 text-xs text-red-600 hover:text-red-700 hover:bg-red-50 border-red-200"
            onClick={handleDisconnect}
            disabled={isDisconnecting}
          >
            {isDisconnecting ? (
              <Loader2 className="h-3 w-3 mr-1.5 animate-spin" />
            ) : (
              <Unlink className="h-3 w-3 mr-1.5" />
            )}
            {t('tools.mcp.dialog.oauthDisconnect')}
          </Button>
        </>
      ) : (
        <Button
          variant="outline"
          size="sm"
          className="h-7 text-xs"
          onClick={handleConnect}
          disabled={isConnecting}
        >
          {isConnecting ? (
            <Loader2 className="h-3 w-3 mr-1.5 animate-spin" />
          ) : (
            <Link2 className="h-3 w-3 mr-1.5" />
          )}
          {connectLabel}
        </Button>
      )}
    </div>
  )
}

import React from "react"
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { getApiUrl } from "@/lib/utils"
import { Settings, Unlink, Plus, Users, UserMinus, Loader2 } from "lucide-react"
import { apiRequest } from "@/lib/api-wrapper"
import { toast } from "@/components/ui/sonner"
import { useAuth } from "@/contexts/auth-context"
import { useI18n } from "@/contexts/i18n-context"

import type { AppIntegration } from "./types"

export type { AppIntegration }

interface OfficialMcpSettingsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  app: AppIntegration | null
  onSuccess?: () => void
  onDisconnect?: (app: AppIntegration) => void
  isGloballyConnected?: boolean
  onConnectStart?: (app: AppIntegration) => void
  onConfigure?: (app: AppIntegration) => void
  onManageKey?: (app: AppIntegration) => void
  // True while the parent-owned connect request for this app is in flight
  // (e.g. a keyless connect fired via onConnectStart). Disables the Connect
  // trigger so rapid double-clicks can't fire overlapping POSTs, and gives
  // the flow its only visible progress state — the catalog card spinner is
  // occluded by this dialog.
  isConnecting?: boolean
}

export function OfficialMcpSettingsDialog({
  open,
  onOpenChange,
  app,
  onSuccess,
  onDisconnect,
  isGloballyConnected = false,
  onConnectStart,
  onConfigure,
  onManageKey,
  isConnecting = false
}: OfficialMcpSettingsDialogProps) {
  const { token, inTeam } = useAuth()
  const { t } = useI18n()

  const handleConnectApp = (appToConnect: AppIntegration) => {
    if (appToConnect.is_custom && onConnectStart) {
      onConnectStart(appToConnect);
      return;
    } else if (!appToConnect.is_custom && onConnectStart) {
      // If we provided onConnectStart for official apps, we might still want to call it
      // But in our case we want to fall through to OAuth if it's official
      // Let's just say if it's not custom, we only call onConnectStart if we explicitly want to bypass OAuth
      // Actually, existing code:
      if (onConnectStart && !appToConnect.is_custom) {
        onConnectStart(appToConnect);
        return;
      }
    }

    const provider = appToConnect.provider || "linkedin"

    // Open OAuth in a popup window to handle the callback smoothly
    const width = 600;
    const height = 700;
    const left = window.screenX + (window.outerWidth - width) / 2;
    const top = window.screenY + (window.outerHeight - height) / 2;

    const authUrl = `${getApiUrl()}/api/auth/${provider}/login?token=${token || ''}&app_id=${appToConnect.id}&redirect=${encodeURIComponent(window.location.href)}`;
    const popup = window.open(
      authUrl,
      `${provider} OAuth`,
      `width=${width},height=${height},left=${left},top=${top},scrollbars=yes`
    );

    // Listen for the postMessage from the popup
    const handleMessage = (event: MessageEvent) => {
      if (event.data?.type === 'oauth-success') {
        window.removeEventListener('message', handleMessage)

        if (onSuccess) onSuccess();
        onOpenChange(false);
      }
    };

    window.addEventListener('message', handleMessage);

    // Fallback: check if popup was closed without success message
    const checkPopup = setInterval(() => {
      if (popup?.closed) {
        clearInterval(checkPopup);
        window.removeEventListener('message', handleMessage);
      }
    }, 500);
  }

  // Toggle team-sharing for the connector. On unshare 403 (still referenced by
  // a team agent) show the reason. server_id is the numeric connector id.
  const handleToggleShare = async (appToShare: AppIntegration, share: boolean) => {
    if (!Number.isInteger(appToShare.server_id)) return
    const type = appToShare.transport === "custom_api" ? "custom_api" : "mcp"
    const action = share ? "share" : "unshare"
    try {
      const response = await apiRequest(
        `${getApiUrl()}/api/connectors/${type}/${appToShare.server_id}/${action}`,
        { method: "POST" },
      )
      if (response.ok) {
        toast.success(share ? t('tools.mcp.sharing.shared') : t('tools.mcp.sharing.private'))
        if (onSuccess) onSuccess()
        onOpenChange(false)
      } else if (!share && response.status === 403) {
        toast.error(t('tools.mcp.sharing.stillUsedByAgent'))
      } else if (share && response.status === 403) {
        toast.error(t('tools.mcp.sharing.onlyOwnerOrAdmin'))
      } else {
        toast.error(t('tools.mcp.alerts.saveFailed'))
      }
    } catch (error) {
      console.error("Failed to toggle connector sharing:", error)
      toast.error(t('tools.mcp.alerts.saveFailed'))
    }
  }

  const handleDisconnectApp = async (appToDisconnect: any) => {
    try {
      const serverId = appToDisconnect.server_id || appToDisconnect.id; // Use server_id if available
      const isCustomApi = appToDisconnect.transport === 'custom_api';

      const url = isCustomApi
        ? `${getApiUrl()}/api/custom-apis/${serverId}`
        : `${getApiUrl()}/api/mcp/servers/${serverId}`;

      const response = await apiRequest(url, {
        method: 'DELETE',
      });

      if (response.ok) {
        if (appToDisconnect.is_custom) {
          toast.success(t('tools.mcp.dialog.deleteSuccess', { name: appToDisconnect.name }))
        } else {
          toast.success(t('tools.mcp.dialog.disconnectSuccess', { name: appToDisconnect.name }))
        }
        if (onDisconnect) onDisconnect(appToDisconnect)
        if (onSuccess) onSuccess()
        onOpenChange(false)
      } else {
        if (appToDisconnect.is_custom) {
          toast.error(t('tools.mcp.dialog.deleteFailed', { name: appToDisconnect.name }))
        } else {
          toast.error(t('tools.mcp.dialog.disconnectFailed', { name: appToDisconnect.name }))
        }
      }
    } catch (error) {
      console.error("Failed to disconnect app:", error)
      toast.error(t('tools.mcp.dialog.errorDisconnecting', { name: appToDisconnect.name }))
    }
  }

  if (!app) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md text-center p-0 overflow-hidden bg-white shadow-xl">
        <div className="relative pt-12 pb-8 px-8 flex flex-col items-center">
          <div className="absolute top-0 left-0 w-full h-24 bg-gradient-to-b from-slate-50 to-white -z-10" />

          <div className="h-20 w-20 bg-white rounded-2xl shadow-sm border border-slate-100 flex items-center justify-center mb-6 overflow-hidden relative group">
            {app.icon ? (
              <img
                src={app.icon}
                alt={`${app.name} logo`}
                className="w-12 h-12 object-contain group-hover:scale-110 transition-transform duration-300"
                onError={(e) => {
                  (e.target as HTMLImageElement).src = `https://ui-avatars.com/api/?name=${encodeURIComponent(app.name)}&background=random&color=fff&size=128`
                }}
              />
            ) : (
              <div className="w-12 h-12 flex items-center justify-center bg-blue-50 text-blue-600 rounded-xl font-bold text-2xl shrink-0">
                {app.name.charAt(0).toUpperCase()}
              </div>
            )}
          </div>

          <DialogTitle className="text-2xl font-bold text-slate-900 mb-3 tracking-tight">
            {app.name}
          </DialogTitle>

          {isGloballyConnected && app.connected_account && (
            <div className="mb-4 inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-100 text-blue-700 text-sm font-medium">
              <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
              {app.connected_account}
            </div>
          )}

          {inTeam && isGloballyConnected && Number.isInteger(app.server_id) && (
            <div className={`mb-4 inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border ${app.shared ? 'bg-blue-50 border-blue-100 text-blue-700' : 'bg-slate-50 border-slate-200 text-slate-600'}`}>
              {!app.shared
                ? t('tools.mcp.sharing.private')
                : app.is_owner
                  ? t('tools.mcp.sharing.shared')
                  : t('tools.mcp.sharing.teamTool')}
              {app.needs_config && ` · ${t('tools.mcp.sharing.needsConfig')}`}
            </div>
          )}

          <p className="text-slate-500 mb-8 max-w-sm leading-relaxed">
            {app.description}
          </p>

          <div className="flex flex-col items-center justify-center gap-3 w-full">
            {!isGloballyConnected && (
              <Button
                className="w-full max-w-[200px] rounded-full h-11 font-medium bg-blue-600 text-white hover:bg-blue-700"
                onClick={() => handleConnectApp(app)}
                disabled={isConnecting}
              >
                {isConnecting ? (
                  <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> {t('tools.mcp.dialog.connecting')}</>
                ) : (
                  <><Plus className="h-4 w-4 mr-2" /> {t('tools.mcp.dialog.connect')}</>
                )}
              </Button>
            )}

            {isGloballyConnected && (
              <>
                {(() => {
                  const isKeyBased = app.auth_type === "api_key"
                  // Keyless apps have no key and no editable config — once
                  // connected, disconnect is the only sensible action.
                  const isKeyless = app.auth_type === "keyless"
                  // Team tool the viewer doesn't own: they can use it, but the backend
                  // rejects edit/delete/unshare (403), so don't render those buttons.
                  const isNonOwnedTeamTool = Boolean(app.shared) && app.is_owner === false
                  return (
                    <>
                      {!isNonOwnedTeamTool && !isKeyless && (
                        <Button
                          className="w-full max-w-[200px] rounded-full h-11 font-medium bg-slate-900 text-white hover:bg-slate-800"
                          onClick={() => {
                            // Key-based apps: users only manage their own key, never the
                            // shared server config. Route to the key dialog, not the form.
                            if (isKeyBased && onManageKey) {
                              onManageKey(app);
                            } else if (app.is_custom && onConfigure) {
                              onConfigure(app);
                            } else {
                              handleConnectApp(app);
                            }
                          }}
                        >
                          <Settings className="h-4 w-4 mr-2" />
                          {isKeyBased ? t('tools.mcp.dialog.manageKey') : t('tools.mcp.dialog.configure')}
                        </Button>
                      )}
                      {/* "Make team" / "Make personal": only the owner (or a team admin,
                          enforced server-side) can change ownership. Hidden on non-owned
                          team tools. */}
                      {inTeam && Number.isInteger(app.server_id) && !isNonOwnedTeamTool && (
                        <Button
                          variant="outline"
                          className="w-full max-w-[200px] rounded-full h-11 font-medium"
                          onClick={() => handleToggleShare(app, !app.shared)}
                        >
                          {app.shared ? (
                            <><UserMinus className="h-4 w-4 mr-2" /> {t('tools.mcp.sharing.unshare')}</>
                          ) : (
                            <><Users className="h-4 w-4 mr-2" /> {t('tools.mcp.sharing.share')}</>
                          )}
                        </Button>
                      )}
                      {!isNonOwnedTeamTool && (
                        <Button
                          variant="outline"
                          className="w-full max-w-[200px] rounded-full h-11 font-medium text-red-600 hover:text-red-700 hover:bg-red-50 border-red-200"
                          onClick={() => handleDisconnectApp(app)}
                        >
                          <Unlink className="h-4 w-4 mr-2" /> {(app.is_custom && !isKeyBased) ? t('tools.mcp.dialog.deleteService') : t('tools.mcp.dialog.disconnect')}
                        </Button>
                      )}
                    </>
                  )
                })()}
              </>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

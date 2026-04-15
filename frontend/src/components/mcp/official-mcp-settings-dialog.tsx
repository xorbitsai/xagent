import React from "react"
import { Dialog, DialogContent } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { getApiUrl } from "@/lib/utils"
import { Settings, Unlink, Plus } from "lucide-react"
import { apiRequest } from "@/lib/api-wrapper"
import { toast } from "sonner"
import { useAuth } from "@/contexts/auth-context"
import { useI18n } from "@/contexts/i18n-context"

export interface AppIntegration {
  id: string
  name: string
  description: string
  icon: string
  is_connected?: boolean
  users?: string
  provider?: string
  category?: string
  is_local?: boolean
  server_id?: number
  connected_account?: string
  is_custom?: boolean
  server?: any
}

interface OfficialMcpSettingsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  app: AppIntegration | null
  onSuccess?: () => void
  onDisconnect?: (app: AppIntegration) => void
  isGloballyConnected?: boolean
  onConnectStart?: (app: AppIntegration) => void
  onConfigure?: (app: AppIntegration) => void
}

export function OfficialMcpSettingsDialog({
  open,
  onOpenChange,
  app,
  onSuccess,
  onDisconnect,
  isGloballyConnected = false,
  onConnectStart,
  onConfigure
}: OfficialMcpSettingsDialogProps) {
  const { token } = useAuth()
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

  const handleDisconnectApp = async (appToDisconnect: any) => {
    try {
      const serverId = appToDisconnect.server_id || appToDisconnect.id; // Use server_id if available
      const url = `${getApiUrl()}/api/mcp/servers/${serverId}`;

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

          <h2 className="text-2xl font-bold text-slate-900 mb-3 tracking-tight">
            {app.name}
          </h2>

          {isGloballyConnected && app.connected_account && (
            <div className="mb-4 inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-100 text-blue-700 text-sm font-medium">
              <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
              {app.connected_account}
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
              >
                <Plus className="h-4 w-4 mr-2" /> {t('tools.mcp.dialog.connect')}
              </Button>
            )}

            {isGloballyConnected && (
              <>
                <Button
                  className="w-full max-w-[200px] rounded-full h-11 font-medium bg-slate-900 text-white hover:bg-slate-800"
                  onClick={() => {
                    if (app.is_custom && onConfigure) {
                      onConfigure(app);
                    } else {
                      handleConnectApp(app);
                    }
                  }}
                >
                  <Settings className="h-4 w-4 mr-2" /> {t('tools.mcp.dialog.configure')}
                </Button>
                <Button
                  variant="outline"
                  className="w-full max-w-[200px] rounded-full h-11 font-medium text-red-600 hover:text-red-700 hover:bg-red-50 border-red-200"
                  onClick={() => handleDisconnectApp(app)}
                >
                  <Unlink className="h-4 w-4 mr-2" /> {app.is_custom ? t('tools.mcp.dialog.deleteService') : t('tools.mcp.dialog.disconnect')}
                </Button>
              </>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

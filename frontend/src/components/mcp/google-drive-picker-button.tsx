import React, { useState } from "react"
import { FolderOpen, Loader2 } from "lucide-react"

import { Button } from "@/components/ui/button"
import { toast } from "@/components/ui/sonner"
import { useI18n } from "@/contexts/i18n-context"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import {
  loadGooglePicker,
  sanitizeGooglePickerDocuments,
} from "@/lib/google-picker"

interface ConnectedAccount {
  id: number
  provider: string
  email?: string
}

interface GoogleDrivePickerButtonProps {
  connectedAccount?: string
  onBeforeOpen?: () => void
}

/**
 * Authorize existing Drive files/folders for the restricted drive.file scope.
 * Selecting an item is the authorization event; the connector does not need
 * to retain the Picker result locally.
 */
export function GoogleDrivePickerButton({
  connectedAccount,
  onBeforeOpen,
}: GoogleDrivePickerButtonProps) {
  const { t } = useI18n()
  const [loading, setLoading] = useState(false)

  const openPicker = async () => {
    onBeforeOpen?.()
    setLoading(true)
    try {
      const accountsResponse = await apiRequest(
        `${getApiUrl()}/api/cloud/accounts?provider=google-drive`,
      )
      if (!accountsResponse.ok) {
        throw new Error(
          accountsResponse.status === 401 || accountsResponse.status === 409
            ? t("kb.dialog.cloudConnect.auth.expired")
            : t("kb.dialog.cloudConnect.picker.error"),
        )
      }
      const accounts = await accountsResponse.json() as ConnectedAccount[]
      const account = accounts.find(item => item.email === connectedAccount) ?? accounts[0]
      if (!account) throw new Error(t("kb.dialog.cloudConnect.auth.expired"))

      const configResponse = await apiRequest(
        `${getApiUrl()}/api/cloud/google-drive/picker-config?account_id=${account.id}`,
      )
      if (!configResponse.ok) {
        throw new Error(
          configResponse.status === 401 || configResponse.status === 409
            ? t("kb.dialog.cloudConnect.auth.expired")
            : configResponse.status === 503
              ? t("kb.dialog.cloudConnect.picker.notConfigured")
              : t("kb.dialog.cloudConnect.picker.error"),
        )
      }
      const config = await configResponse.json() as {
        access_token?: string
        developer_key?: string
        app_id?: string
      }
      if (!config.access_token || !config.developer_key || !config.app_id) {
        throw new Error(t("kb.dialog.cloudConnect.picker.notConfigured"))
      }

      await loadGooglePicker()
      const pickerApi = window.google?.picker
      if (!pickerApi) throw new Error(t("kb.dialog.cloudConnect.picker.error"))

      const docsView = new pickerApi.DocsView(pickerApi.ViewId.DOCS)
        .setIncludeFolders(true)
        .setSelectFolderEnabled(true)
      const picker = new pickerApi.PickerBuilder()
        .setDeveloperKey(config.developer_key)
        .setAppId(config.app_id)
        .setOAuthToken(config.access_token)
        .addView(docsView)
        .enableFeature(pickerApi.Feature.MULTISELECT_ENABLED)
        .setCallback(data => {
          if (data.action !== pickerApi.Action.PICKED) return
          const selected = sanitizeGooglePickerDocuments(data.docs)
          if (selected.length > 0) {
            toast.success(t("kb.dialog.cloudConnect.picker.authorized"))
          }
        })
        .build()
      picker.setVisible(true)
    } catch (error) {
      console.error("Failed to open Google Drive Picker", error)
      toast.error(error instanceof Error
        ? error.message
        : t("kb.dialog.cloudConnect.picker.error"))
    } finally {
      setLoading(false)
    }
  }

  return (
    <Button
      type="button"
      variant="outline"
      className="w-full max-w-[260px] rounded-full h-11 font-medium"
      onClick={openPicker}
      disabled={loading}
    >
      {loading
        ? <Loader2 className="h-4 w-4 mr-2 animate-spin" />
        : <FolderOpen className="h-4 w-4 mr-2" />}
      {t("kb.dialog.cloudConnect.picker.open")}
    </Button>
  )
}

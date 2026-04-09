import { useState, useEffect, useRef, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Select } from "@/components/ui/select"
import { ConfirmDialog } from "@/components/ui/confirm-dialog"
import { useI18n } from "@/contexts/i18n-context"
import { useAuth } from "@/contexts/auth-context"
import {
  Check,
  ChevronRight,
  Folder,
  File,
  Loader2,
  Search,
  RefreshCw,
  Trash2,
} from "lucide-react"
import { toast } from "sonner"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"

export interface CloudFile {
  id: string
  name: string
  type: 'file' | 'folder'
  size?: string
  updatedAt?: string
}

interface ConnectedAccount {
  id: number
  provider: string
  email: string
  created_at: string
}

interface CloudConnectDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  provider: {
    id: string
    name: string
    hasDrives: boolean
    authPath: string
    logo: string
  } | null
  initialSelectedFiles?: CloudFile[]
  onConfirm: (selectedFiles: CloudFile[]) => void
}

export function CloudConnectDialog({
  open,
  onOpenChange,
  provider,
  initialSelectedFiles = [],
  onConfirm
}: CloudConnectDialogProps) {
  const { t } = useI18n()
  const { token } = useAuth()

  // Internal state
  const [cloudUser, setCloudUser] = useState<string | undefined>()
  const [selectedDrive, setSelectedDrive] = useState<string>("")
  const [currentPath, setCurrentPath] = useState<{ id: string, name: string }[]>([])
  const [selectedFiles, setSelectedFiles] = useState<CloudFile[]>([])
  const [files, setFiles] = useState<CloudFile[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [loading, setLoading] = useState(false)
  const [accountsLoading, setAccountsLoading] = useState(false)
  const [connectedAccounts, setConnectedAccounts] = useState<ConnectedAccount[]>([])
  const [driveOptions, setDriveOptions] = useState<{ value: string, label: string }[]>([])
  const [refreshTrigger, setRefreshTrigger] = useState(0)
  const [accountToDelete, setAccountToDelete] = useState<{ id: number | null, email: string | null }>({ id: null, email: null })
  const [isDeletingAccount, setIsDeletingAccount] = useState(false)

  // Helper to check if selected
  const isSelected = (id: string) => selectedFiles.some(f => f.id === id)

  // Fetch connected accounts
  const fetchAccounts = useCallback(async () => {
    if (!provider) return

    setAccountsLoading(true)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/cloud/accounts?provider=${provider.id}`)

      if (response.ok) {
        const data = await response.json()
        setConnectedAccounts(data)
      }
    } catch (error) {
      console.error("Failed to fetch connected accounts", error)
    } finally {
      setAccountsLoading(false)
    }
  }, [provider])

  // Handle OAuth login
  const handleAuth = () => {
    const width = 500
    const height = 600
    const left = window.screen.width / 2 - width / 2
    const top = window.screen.height / 2 - height / 2

    const authUrl = `${getApiUrl()}/api/auth/${provider?.authPath}/login?token=${token || ''}`

    window.open(
      authUrl,
      `${provider?.name} Auth`,
      `width=${width},height=${height},left=${left},top=${top}`
    )
  }

  useEffect(() => {
    const messageHandler = async (event: MessageEvent) => {
      if (event.data?.type === "oauth-success") {
        await fetchAccounts()
        setCloudUser(event.data.email)
        toast.success(t("kb.dialog.cloudConnect.auth.success"))
      }
    }

    window.addEventListener("message", messageHandler)
    return () => window.removeEventListener("message", messageHandler)
  }, [fetchAccounts, t])

  // Toggle selection
  const toggleSelection = (file: CloudFile) => {
    if (isSelected(file.id)) {
      setSelectedFiles(prev => prev.filter(f => f.id !== file.id))
    } else {
      setSelectedFiles(prev => [...prev, file])
    }
  }

  // Reset state when provider changes
  useEffect(() => {
    setCloudUser(undefined)
    setSelectedDrive("")
    setCurrentPath([])
    setFiles([])
    setSelectedFiles([])
    setConnectedAccounts([])
  }, [provider?.id])

  // Fetch connected accounts on mount
  useEffect(() => {
    fetchAccounts()
  }, [provider?.id])

  // Fetch drives when account changes
  useEffect(() => {
    if (!cloudUser || !provider?.hasDrives) return

    const fetchDrives = async () => {
      try {
        const selectedAccount = connectedAccounts.find(acc => (acc.email || `Account ${acc.id}`) === cloudUser)
        const accountIdParam = selectedAccount ? `?account_id=${selectedAccount.id}` : ''

        const response = await apiRequest(`${getApiUrl()}/api/cloud/${provider.id}/drives${accountIdParam}`)

        if (response.ok) {
          const data = await response.json()
          const options = data.map((d: any) => ({
            value: d.id,
            label: d.name
          }))
          setDriveOptions(options)

          // Ensure selectedDrive is valid
          if (!options.find((o: any) => o.value === selectedDrive)) {
            if (options.length > 0) setSelectedDrive(options[0].value)
          }
        }
      } catch (error) {
        console.error("Failed to fetch drives", error)
      }
    }

    fetchDrives()
  }, [cloudUser, connectedAccounts, provider?.id])

  // Fetch files from backend
  useEffect(() => {
    if (!cloudUser || !provider) return

    if (provider.hasDrives && !selectedDrive) return

    const fetchFiles = async () => {
      // Find selected account ID
      const selectedAccount = connectedAccounts.find(acc => (acc.email || `Account ${acc.id}`) === cloudUser)
      if (!selectedAccount) return

      setLoading(true)
      try {
        const folderId = currentPath.length > 0
          ? currentPath[currentPath.length - 1].id
          : (provider.hasDrives ? selectedDrive : 'root')
        // Use relative URL or configured API base

        const accountIdParam = `&account_id=${selectedAccount.id}`

        const response = await apiRequest(`${getApiUrl()}/api/cloud/${provider.id}/files?folder_id=${folderId}${accountIdParam}`)

        if (response.ok) {
          const data = await response.json()
          setFiles(data)
        } else {
          if (response.status === 401) {
            setCloudUser(undefined) // Reset user if token invalid
            toast.error(t("kb.dialog.cloudConnect.auth.expired"))
          } else {
            toast.error(t("kb.dialog.cloudConnect.error.fetchFailed"))
          }
        }
      } catch (error) {
        console.error(error)
        toast.error(t("kb.dialog.cloudConnect.error.fetchFailed"))
      } finally {
        setLoading(false)
      }
    }

    fetchFiles()
  }, [cloudUser, currentPath, connectedAccounts, selectedDrive, provider?.id, refreshTrigger])

  // Sync initial selection when opening
  const prevOpen = useRef(open)
  useEffect(() => {
    if (open && !prevOpen.current) {
      setSelectedFiles(initialSelectedFiles)
    }
    prevOpen.current = open
  }, [open, initialSelectedFiles])

  // Reset current path when drive changes
  useEffect(() => {
    setCurrentPath([])
  }, [selectedDrive])

  const handleConfirm = () => {
    onConfirm(selectedFiles)
    onOpenChange(false)
  }

  const handleCancel = () => {
    onOpenChange(false)
  }

  // Handle account deletion
  const handleDeleteAccount = async () => {
    if (!accountToDelete) return

    setIsDeletingAccount(true)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/cloud/accounts/${accountToDelete.id}`, {
        method: "DELETE",
      })

      if (response.ok) {
        toast.success(t("kb.dialog.cloudConnect.auth.deleteSuccess"))
        const accountWasSelected = connectedAccounts.find(acc => acc.id === accountToDelete.id)?.email === cloudUser ||
          connectedAccounts.find(acc => acc.id === accountToDelete.id)?.id.toString() === cloudUser?.replace('Account ', '')
        if (accountWasSelected) {
          setCloudUser(undefined)
          setSelectedDrive("")
          setCurrentPath([])
          setFiles([])
        }
        await fetchAccounts()
      } else {
        toast.error(t("kb.dialog.cloudConnect.auth.deleteFailed"))
      }
    } catch (error) {
      console.error("Failed to delete account", error)
      toast.error(t("kb.dialog.cloudConnect.auth.deleteFailed"))
    } finally {
      setIsDeletingAccount(false)
      setAccountToDelete({ id: null, email: null })
    }
  }

  // Filter files based on search
  const filteredFiles = files.filter(f =>
    f.name.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const folders = filteredFiles.filter(f => f.type === 'folder')
  const fileItems = filteredFiles.filter(f => f.type === 'file')

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[900px] h-[80vh] max-h-[800px] flex flex-col">
        <DialogHeader>
          <DialogTitle>
            {t("kb.dialog.cloudConnect.auth.title", {
              provider: provider?.name || t("kb.dialog.cloudConnect.auth.defaultProvider")
            })}
          </DialogTitle>
        </DialogHeader>

        <div className="flex-1 overflow-hidden flex flex-col space-y-4 py-4">
          {/* Account Selection Section */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Label>{t("kb.dialog.cloudConnect.auth.selectAccount")}</Label>
              {accountsLoading && <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />}
            </div>
            <Select
              disabled={accountsLoading}
              value={cloudUser}
              onValueChange={(value) => {
                if (value === "add_new") {
                  handleAuth()
                } else {
                  setCloudUser(value)
                }
              }}
              options={[
                ...connectedAccounts
                  .filter(acc => acc.provider === provider?.id)
                  .map(acc => ({
                    value: acc.email || `Account ${acc.id}`,
                    label: acc.email
                      ? t("kb.dialog.cloudConnect.auth.accountProviderLabel", { email: acc.email, provider: provider?.name || t("kb.dialog.cloudConnect.auth.defaultProvider") })
                      : t("kb.dialog.cloudConnect.auth.accountLabel", { id: acc.id }),
                    actionIcon: <Trash2 className="h-4 w-4 text-red-500 hover:text-red-600" />,
                    onAction: (e: React.MouseEvent) => {
                      e.stopPropagation()
                      setAccountToDelete({ id: acc.id, email: acc.email })
                    }
                  })),
                { value: "add_new", label: t("kb.dialog.cloudConnect.auth.addAccount") }
              ]}
              placeholder={t("kb.dialog.cloudConnect.select.accountPlaceholder")}
            />
          </div>

          {/* Drive Selection */}
          {provider?.hasDrives && (
            <div className="space-y-2">
              <Label>{t("kb.dialog.cloudConnect.select.driveLabel")}</Label>
              <Select
                value={selectedDrive}
                onValueChange={setSelectedDrive}
                options={driveOptions}
                disabled={!cloudUser}
                placeholder={t("kb.dialog.cloudConnect.select.drivePlaceholder")}
              />
            </div>
          )}

          {/* Main Content Area - Split View */}
          <div className="flex-1 grid grid-cols-2 gap-6 min-h-0">
            {/* Left Column: File Browser */}
            <div className="flex flex-col border rounded-md overflow-hidden">
              {/* Search Bar */}
              <div className="p-2 border-b">
                <div className="relative">
                  <Input
                    placeholder={t("kb.dialog.cloudConnect.search.placeholder")}
                    className="pl-8 h-9"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    disabled={!cloudUser}
                  />
                  <div className="absolute left-2.5 top-2.5 text-muted-foreground">
                    <Search className="h-4 w-4" />
                  </div>
                </div>
              </div>

              {/* Breadcrumbs */}
              <div className="flex items-center justify-between p-2 border-b bg-muted/20">
                <div className="flex items-center gap-1 text-sm text-muted-foreground">
                  <span
                    className="hover:underline cursor-pointer hover:text-foreground transition-colors"
                    onClick={() => setCurrentPath([])}
                  >
                    {provider?.hasDrives ? (driveOptions.find(o => o.value === selectedDrive)?.label || selectedDrive) : provider?.name}
                  </span>
                  {currentPath.map((folder, index) => (
                    <div key={index} className="flex items-center gap-1">
                      <ChevronRight size={14} />
                      <span
                        className={`hover:underline cursor-pointer hover:text-foreground transition-colors ${index === currentPath.length - 1 ? "font-medium text-foreground" : ""}`}
                        onClick={() => setCurrentPath(prev => prev.slice(0, index + 1))}
                      >
                        {folder.name}
                      </span>
                    </div>
                  ))}
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6 text-muted-foreground hover:text-foreground"
                  onClick={() => setRefreshTrigger(prev => prev + 1)}
                  disabled={loading || !cloudUser || (provider?.hasDrives && !selectedDrive)}
                  title={t("common.refresh")}
                >
                  <RefreshCw className={`h-3 w-3 ${loading ? 'animate-spin' : ''}`} />
                </Button>
              </div>

              {/* File List */}
              <ScrollArea className="flex-1 overflow-auto">
                <div className="p-2">
                  {loading ? (
                    <div className="flex flex-col items-center justify-center py-10 text-muted-foreground">
                      <Loader2 className="h-8 w-8 animate-spin mb-2 opacity-50" />
                      <span className="text-xs">{t("kb.dialog.cloudConnect.loading")}</span>
                    </div>
                  ) : (
                    <>
                      {folders.length === 0 && fileItems.length === 0 ? (
                        <div className="text-center text-muted-foreground py-8">
                          {searchQuery
                            ? t("kb.dialog.cloudConnect.fileList.noMatchingFiles")
                            : t("kb.dialog.cloudConnect.fileList.noFiles")}
                        </div>
                      ) : (
                        <div className="space-y-4">
                          {/* Folders Section */}
                          {folders.length > 0 && (
                            <div>
                              <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2 px-2">
                                {t("kb.dialog.cloudConnect.fileList.headers.folders")}
                              </h3>
                              <div className="space-y-1">
                                {folders.map(folder => (
                                  <div
                                    key={folder.id}
                                    className="flex items-center gap-2 p-2 hover:bg-muted/50 rounded-md cursor-pointer group"
                                    onClick={() => setCurrentPath(prev => [...prev, { id: folder.id, name: folder.name }])}
                                  >
                                    <Folder className="h-5 w-5 text-blue-500 fill-blue-500/20" />
                                    <span className="truncate flex-1 text-sm font-medium">{folder.name}</span>
                                    <ChevronRight className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Files Section */}
                          {fileItems.length > 0 && (
                            <div>
                              <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2 px-2">
                                {t("kb.dialog.cloudConnect.fileList.headers.files")}
                              </h3>
                              <div className="space-y-1">
                                {fileItems.map(file => {
                                  const selected = isSelected(file.id)
                                  return (
                                    <div
                                      key={file.id}
                                      className={`flex items-center gap-2 p-2 rounded-md cursor-pointer transition-colors ${selected
                                        ? "bg-primary/10 text-primary hover:bg-primary/15"
                                        : "hover:bg-muted/50"
                                        }`}
                                      onClick={() => toggleSelection(file)}
                                    >
                                      <div className={`flex items-center justify-center w-4 h-4 rounded border ${selected
                                        ? "bg-primary border-primary text-primary-foreground"
                                        : "border-muted-foreground/30 bg-background"
                                        }`}>
                                        {selected && <Check className="h-3 w-3" />}
                                      </div>
                                      <File className="h-4 w-4 text-muted-foreground" />
                                      <div className="flex flex-col flex-1 min-w-0">
                                        <span className="truncate text-sm font-medium">{file.name}</span>
                                      </div>
                                    </div>
                                  )
                                })}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </>
                  )}
                </div>
              </ScrollArea>
            </div>

            {/* Right Column: Selected Files */}
            <div className="flex flex-col border rounded-md overflow-hidden">
              <div className="p-3 border-b bg-muted/20 font-medium text-sm flex items-center justify-between">
                {t("kb.dialog.cloudConnect.selectedFiles.title")}
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-muted-foreground hover:text-foreground h-8 text-xs"
                  onClick={() => setSelectedFiles([])}
                  disabled={selectedFiles.length === 0}
                >
                  {t("kb.dialog.cloudConnect.selectedFiles.clearAll")}
                </Button>
              </div>
              <ScrollArea className="flex-1 overflow-auto">
                <div className="p-2 space-y-1">
                  {selectedFiles.map(file => (
                    <div key={file.id} className="flex items-center gap-2 p-2 rounded-md hover:bg-muted/50 group text-sm cursor-pointer" onClick={() => toggleSelection(file)}>
                      <div className="w-4 h-4 bg-primary text-primary-foreground rounded flex items-center justify-center">
                        <Check className="h-3 w-3" />
                      </div>
                      <File className="h-4 w-4 text-muted-foreground" />
                      <span className="truncate flex-1">{file.name}</span>
                    </div>
                  ))}
                  {selectedFiles.length === 0 && (
                    <div className="p-8 text-center text-muted-foreground text-sm">
                      {t("kb.dialog.cloudConnect.selectedFiles.empty")}
                    </div>
                  )}
                </div>
              </ScrollArea>
            </div>
          </div>
        </div>

        <div className="flex justify-end gap-2 pt-2 border-t mt-auto">
          <Button variant="outline" onClick={handleCancel}>
            {t("kb.dialog.cloudConnect.select.cancel")}
          </Button>
          <Button onClick={handleConfirm} disabled={selectedFiles.length === 0}>
            {t("kb.dialog.cloudConnect.select.confirm")}
          </Button>
        </div>
      </DialogContent>

      <ConfirmDialog
        isOpen={accountToDelete.id !== null}
        onOpenChange={(open) => {
          if (!open) setAccountToDelete({ id: null, email: null })
        }}
        onConfirm={handleDeleteAccount}
        isLoading={isDeletingAccount}
        title={t("kb.dialog.cloudConnect.auth.deleteAccount")}
        description={t("kb.dialog.cloudConnect.auth.deleteConfirm", { email: accountToDelete.email || "" })}
      />
    </Dialog>
  )
}

"use client"

import React, { useState, useEffect } from "react"
import { SearchInput } from "@/components/ui/search-input"
import { Button } from "@/components/ui/button"
import { Plus, Bot, Trash2, MessageSquare, Edit, MoreVertical, Globe, Calendar, Clock, Rocket, Sparkles, Settings2, ArrowRight } from "lucide-react"
import { Popover, PopoverTrigger, PopoverContent } from "@/components/ui/popover"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Textarea } from "@/components/ui/textarea"
import { DeployAgentDialog, Agent } from "@/components/build/deploy-agent-dialog"
import { useI18n } from "@/contexts/i18n-context"
import { useRouter, useSearchParams } from "next/navigation"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import { ConfirmDialog } from "@/components/ui/confirm-dialog"
import { toast } from "sonner"

export default function BuildsPage() {
  const { t } = useI18n()
  const router = useRouter()
  const searchParams = useSearchParams()
  const [searchTerm, setSearchTerm] = useState("")
  const [agents, setAgents] = useState<Agent[]>([])
  const [loading, setLoading] = useState(true)

  // Deploy Dialog State
  const [deployAgent, setDeployAgent] = useState<Agent | null>(null)

  // Check for template parameter and redirect to create page
  useEffect(() => {
    const templateId = searchParams.get("template")
    if (templateId) {
      // Redirect to create page with template parameter
      router.replace(`/build/new?template=${templateId}`)
    }
  }, [searchParams, router])

  // Fetch agents on mount
  const fetchAgents = async () => {
    try {
      setLoading(true)
      const response = await apiRequest(`${getApiUrl()}/api/agents`)
      if (response.ok) {
        const data = await response.json()
        setAgents(data)
      }
    } catch (error) {
      console.error("Failed to fetch agents:", error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchAgents()
  }, [])

  const handlePublish = async (agentId: number) => {
    try {
      const response = await apiRequest(`${getApiUrl()}/api/agents/${agentId}/publish`, {
        method: "POST",
      })
      if (response.ok) {
        fetchAgents() // Refresh list
      }
    } catch (error) {
      console.error("Failed to publish agent:", error)
    }
  }

  const handleUnpublish = async (agentId: number) => {
    try {
      const response = await apiRequest(`${getApiUrl()}/api/agents/${agentId}/unpublish`, {
        method: "POST",
      })
      if (response.ok) {
        fetchAgents() // Refresh list
      }
    } catch (error) {
      console.error("Failed to unpublish agent:", error)
    }
  }

  const [agentToDelete, setAgentToDelete] = useState<number | null>(null)
  const [isDeletingAgent, setIsDeletingAgent] = useState(false)

  const confirmDeleteAgent = async () => {
    if (agentToDelete === null) return
    const agentId = agentToDelete
    setIsDeletingAgent(true)

    try {
      const response = await apiRequest(`${getApiUrl()}/api/agents/${agentId}`, {
        method: "DELETE",
      })
      if (response.ok) {
        fetchAgents() // Refresh list
        setAgentToDelete(null)
      } else {
        toast.error(t('common.deleteFailed'))
      }
    } catch (error) {
      console.error("Failed to delete agent:", error)
      toast.error(t('common.deleteFailed'))
    } finally {
      setIsDeletingAgent(false)
    }
  }

  const handleDelete = (agentId: number) => {
    setAgentToDelete(agentId)
  }

  // Filter agents based on search term
  const filteredAgents = agents.filter(agent =>
    agent.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (agent.description && agent.description.toLowerCase().includes(searchTerm.toLowerCase()))
  )

  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false)
  const [createPrompt, setCreatePrompt] = useState("")

  const handleCreate = () => {
    setIsCreateModalOpen(true)
  }

  const handleBuildWithPrompt = () => {
    if (createPrompt.trim()) {
      router.push(`/build/new?prompt=${encodeURIComponent(createPrompt.trim())}`)
    } else {
      router.push("/build/new")
    }
    setIsCreateModalOpen(false)
  }

  const handleManualCreate = () => {
    router.push("/build/new")
    setIsCreateModalOpen(false)
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleDateString()
  }

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="flex justify-between items-center p-8">
        <div>
          <h1 className="text-3xl font-bold mb-1">{t("builds.list.header.title")}</h1>
          <p className="text-muted-foreground">{t("builds.list.header.description")}</p>
        </div>
        <div className="flex items-center gap-4">
          <SearchInput
            placeholder={t("builds.list.search.placeholder")}
            value={searchTerm}
            onChange={setSearchTerm}
            containerClassName="w-64"
          />
          <Button onClick={handleCreate}>
            <Plus className="mr-2 h-4 w-4" />
            {t("builds.list.header.create")}
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 px-6 pb-6 space-y-6 overflow-auto">
        {/* Loading State */}
        {loading ? (
          <div className="flex items-center justify-center h-[400px]">
            <div className="text-muted-foreground">{t("common.loading")}</div>
          </div>
        ) : (
          <>
            {/* List */}
            {filteredAgents.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {filteredAgents.map((agent) => (
                  <div
                    key={agent.id}
                    className="group relative flex flex-col justify-between space-y-4 rounded-xl border bg-card p-6 shadow-sm transition-all hover:shadow-md hover:border-primary/50 cursor-pointer"
                    onClick={() => router.push(`/build/${agent.id}`)}
                  >
                    <div className="flex-1">
                      <div className="space-y-4">
                        <div className="flex items-start gap-4">
                          <div className="h-10 w-10 shrink-0 rounded-lg bg-primary/10 flex items-center justify-center text-primary overflow-hidden">
                            {agent.logo_url ? (
                              <img src={`${getApiUrl()}${agent.logo_url}`} alt={agent.name} className="h-full w-full object-cover" />
                            ) : (
                              <Bot className="h-6 w-6" />
                            )}
                          </div>
                          <div className="flex-1 min-w-0 pr-6">
                            <h3 className="font-semibold text-base leading-tight truncate" title={agent.name}>
                              {agent.name}
                            </h3>
                            <div className="mt-2">
                              <span className={`inline-flex text-[11px] px-2 py-0.5 rounded-full capitalize font-medium ${agent.status === 'published'
                                ? 'bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-400'
                                : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'
                                }`}>
                                {agent.status === 'published' ? t('builds.list.status.published') : t('builds.list.status.draft')}
                              </span>
                            </div>
                          </div>
                        </div>
                        <div className="absolute right-4 top-1" onClick={(e) => e.stopPropagation()}>
                          <Popover>
                            <PopoverTrigger asChild>
                              <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-foreground">
                                <MoreVertical className="h-4 w-4" />
                              </Button>
                            </PopoverTrigger>
                            <PopoverContent align="end" className="w-32 p-1" onClick={(e) => e.stopPropagation()}>
                              <div className="flex flex-col">
                                <Button
                                  variant="ghost"
                                  className="justify-start px-2 py-1.5 h-auto font-normal text-sm"
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    if (agent.status === 'published') {
                                      handleUnpublish(agent.id)
                                    } else {
                                      handlePublish(agent.id)
                                    }
                                  }}
                                >
                                  <Globe className="mr-2 h-4 w-4" />
                                  {agent.status === 'published' ? t('builds.list.actions.unpublish') : t('builds.list.actions.publish')}
                                </Button>
                                <div className="h-px bg-border my-1 mx-1" />
                                <Button
                                  variant="ghost"
                                  className="justify-start px-2 py-1.5 h-auto font-normal text-sm text-destructive hover:text-destructive hover:bg-destructive/10"
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    handleDelete(agent.id)
                                  }}
                                >
                                  <Trash2 className="mr-2 h-4 w-4" />
                                  {t('builds.list.actions.delete')}
                                </Button>
                              </div>
                            </PopoverContent>
                          </Popover>
                        </div>

                        <p className="text-sm text-muted-foreground line-clamp-2 mt-4">
                          {agent.description || t('builds.card.noDescription')}
                        </p>
                      </div>
                    </div>

                    <div className="space-y-4 pt-2">
                      <div className="space-y-1.5">
                        <div className="flex items-center text-xs text-muted-foreground">
                          <Calendar className="h-3.5 w-3.5 mr-1.5" />
                          {t('builds.card.createdAt')}: {formatDate(agent.created_at)}
                        </div>
                        <div className="flex items-center text-xs text-muted-foreground">
                          <Clock className="h-3.5 w-3.5 mr-1.5" />
                          {t('builds.card.updatedAt')}: {formatDate(agent.updated_at || agent.created_at)}
                        </div>
                      </div>
                      <div className="flex gap-2" onClick={(e) => e.stopPropagation()}>
                        {agent.status === 'published' ? (
                          <>
                            <Button
                              variant="default"
                              className="flex-1 bg-blue-600 hover:bg-blue-700 text-white"
                              onClick={() => router.push(`/agent/${agent.id}`)}
                            >
                              <MessageSquare className="mr-1.5 h-4 w-4" />
                              {t('builds.list.actions.chat')}
                            </Button>
                            <Button
                              variant="outline"
                              size="icon"
                              onClick={() => {
                                setDeployAgent(agent);
                              }}
                              title="Deploy"
                            >
                              <Rocket className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="outline"
                              className="px-4"
                              onClick={() => router.push(`/build/${agent.id}`)}
                            >
                              <Edit className="mr-1.5 h-4 w-4" />
                              {t('builds.list.actions.edit')}
                            </Button>
                          </>
                        ) : (
                          <Button
                            variant="outline"
                            className="flex-1 w-full"
                            onClick={() => router.push(`/build/${agent.id}`)}
                          >
                            <Edit className="mr-1.5 h-4 w-4" />
                            {t('builds.list.actions.edit')}
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-[400px] text-center space-y-4 border rounded-lg bg-muted/10 border-dashed">
                <div className="h-12 w-12 rounded-full bg-muted flex items-center justify-center">
                  <Bot className="h-6 w-6 text-muted-foreground" />
                </div>
                <div className="space-y-2">
                  <h3 className="font-semibold text-lg">{t("builds.list.empty.title")}</h3>
                  <p className="text-muted-foreground max-w-sm mx-auto">
                    {t("builds.list.empty.description")}
                  </p>
                </div>
                <Button onClick={handleCreate} variant="outline">
                  <Plus className="mr-2 h-4 w-4" />
                  {t("builds.list.empty.create")}
                </Button>
              </div>
            )}
          </>
        )}
      </div>

      <ConfirmDialog
        isOpen={agentToDelete !== null}
        onOpenChange={(open) => !open && setAgentToDelete(null)}
        onConfirm={confirmDeleteAgent}
        isLoading={isDeletingAgent}
        description={t('builds.list.actions.deleteConfirm')}
      />

      {/* Deploy Agent Dialog */}
      <DeployAgentDialog
        deployAgent={deployAgent}
        onClose={() => setDeployAgent(null)}
        onUpdate={(updatedAgent) => {
          setDeployAgent(updatedAgent)
          setAgents(agents.map(a => a.id === updatedAgent.id ? updatedAgent : a))
        }}
      />

      <Dialog open={isCreateModalOpen} onOpenChange={setIsCreateModalOpen}>
        <DialogContent className="sm:max-w-[550px] gap-0 p-0 overflow-hidden bg-background shadow-lg rounded-xl">
          <DialogHeader className="px-6 py-5 border-b">
            <DialogTitle className="flex items-center gap-2 text-xl font-semibold">
              <Bot className="h-6 w-6" />
              {t("builds.list.createModal.title")}
            </DialogTitle>
          </DialogHeader>

          <div className="p-6 space-y-6">
            {/* Option 1: By Describing It */}
            <div className="flex flex-col space-y-4 rounded-xl border border-border p-5 bg-card">
              <div className="flex items-start gap-4">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-indigo-50 text-indigo-500">
                  <Sparkles className="h-5 w-5" />
                </div>
                <div className="space-y-1">
                  <h3 className="font-semibold text-base">
                    {t("builds.list.createModal.describeTitle")}
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    {t("builds.list.createModal.describeDesc", { appName: process.env.NEXT_PUBLIC_APP_NAME || "Xagent" })}
                  </p>
                </div>
              </div>

              <div className="relative rounded-lg border border-input bg-background focus-within:ring-1 focus-within:ring-ring">
                <Textarea
                  value={createPrompt}
                  onChange={(e) => setCreatePrompt(e.target.value)}
                  placeholder={t("builds.list.createModal.placeholder")}
                  className="min-h-[120px] resize-none border-0 shadow-none focus-visible:ring-0 pb-14"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault()
                      handleBuildWithPrompt()
                    }
                  }}
                />
                <div className="absolute bottom-2 right-2">
                  <Button
                    onClick={handleBuildWithPrompt}
                    disabled={!createPrompt.trim()}
                    className="bg-indigo-400 hover:bg-indigo-500 text-white shadow-none"
                  >
                    <Sparkles className="mr-2 h-4 w-4" />
                    {t("builds.list.createModal.buildBtn")}
                  </Button>
                </div>
              </div>
            </div>

            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t" />
              </div>
              <div className="relative flex justify-center text-xs">
                <span className="bg-background px-2 text-muted-foreground">
                  {t("common.or")}
                </span>
              </div>
            </div>

            {/* Option 2: Manually */}
            <div className="flex flex-col items-start gap-4 rounded-xl border border-border p-5 bg-card">
              <div className="flex items-start gap-4">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground">
                  <Settings2 className="h-5 w-5" />
                </div>
                <div className="space-y-1 flex-1">
                  <h3 className="font-semibold text-base">
                    {t("builds.list.createModal.manualTitle")}
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    {t("builds.list.createModal.manualDesc")}
                  </p>
                </div>
              </div>
              <Button
                variant="outline"
                onClick={handleManualCreate}
                className="gap-2"
              >
                {t("builds.list.createModal.manualBtn")}
                <ArrowRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}

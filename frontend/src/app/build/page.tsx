"use client"

import React, { useState, useEffect } from "react"
import { SearchInput } from "@/components/ui/search-input"
import { Button } from "@/components/ui/button"
import { Plus, Bot, Trash2, MessageSquare, Edit, MoreVertical, Globe, Calendar, Clock } from "lucide-react"
import { Popover, PopoverTrigger, PopoverContent } from "@/components/ui/popover"
import { useI18n } from "@/contexts/i18n-context"
import { useRouter, useSearchParams } from "next/navigation"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import { ConfirmDialog } from "@/components/ui/confirm-dialog"
import { toast } from "sonner"

interface Agent {
  id: number
  name: string
  description: string
  logo_url: string | null
  status: string
  created_at: string
  updated_at: string
}

export default function BuildsPage() {
  const { t } = useI18n()
  const router = useRouter()
  const searchParams = useSearchParams()
  const [searchTerm, setSearchTerm] = useState("")
  const [agents, setAgents] = useState<Agent[]>([])
  const [loading, setLoading] = useState(true)

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

  const handleCreate = () => {
    router.push("/build/new")
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
                    className="group relative flex flex-col justify-between space-y-4 rounded-xl border bg-card p-6 shadow-sm transition-all hover:shadow-md hover:border-primary/50"
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
    </div>
  )
}

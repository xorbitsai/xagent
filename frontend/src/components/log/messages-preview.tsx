"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { ChevronDown, ChevronRight, MessageSquare, Bot, User, Settings } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"

interface Message {
  role: string
  content: string
}

interface MessagesPreviewProps {
  contextPreview: any
}

// Single message component
function MessageItem({ message, index }: { message: Message; index: number }) {
  const { t } = useI18n()
  const getRoleConfig = () => {
    const configs = {
      system: {
        icon: <Settings className="h-4 w-4" />,
        color: "text-purple-500",
        bgColor: "bg-purple-500/10",
        label: t('agent.logs.messagesPreview.labels.role.system')
      },
      user: {
        icon: <User className="h-4 w-4" />,
        color: "text-blue-500",
        bgColor: "bg-blue-500/10",
        label: t('agent.logs.messagesPreview.labels.role.user')
      },
      assistant: {
        icon: <Bot className="h-4 w-4" />,
        color: "text-green-500",
        bgColor: "bg-green-500/10",
        label: t('agent.logs.messagesPreview.labels.role.assistant')
      },
    }

    return configs[message.role as keyof typeof configs] || configs.user
  }

  const config = getRoleConfig()

  return (
    <Card className="border-border">
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <span className={config.color}>{config.icon}</span>
          <Badge variant="outline" className={`text-xs ${config.bgColor} ${config.color} border-${config.color}/20`}>
            {config.label}
          </Badge>
          <span className="text-xs text-muted-foreground">
            {t('agent.logs.messagesPreview.labels.messageIndex', { index: index + 1 })}
          </span>
        </div>
        <div className="text-sm leading-relaxed whitespace-pre-wrap font-mono text-xs bg-muted/30 p-2 rounded">
          {message.content}
        </div>
      </div>
    </Card>
  )
}

// Messages preview main component
export function MessagesPreview({ contextPreview }: MessagesPreviewProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const { t } = useI18n()

  // Attempt to parse context_preview
  let messages: Message[] = []
  let isStructured = false
  let parseError: Error | null = null

  // Add debug info
  console.log('🔍 MessagesPreview Debug:', {
    contextPreviewType: typeof contextPreview,
    contextPreview: contextPreview,
    isArray: Array.isArray(contextPreview),
    isString: typeof contextPreview === 'string',
    isObject: typeof contextPreview === 'object'
  })

  try {
    if (Array.isArray(contextPreview)) {
      messages = contextPreview
      isStructured = true
      console.log('✅ Directly parsed as array, count:', messages.length)
    } else if (typeof contextPreview === 'string') {
      // Attempt to parse string array
      try {
        // First try standard JSON parsing
        const parsed = JSON.parse(contextPreview)
        if (Array.isArray(parsed)) {
          messages = parsed
          isStructured = true
          console.log('✅ JSON.parse success, count:', messages.length)
        }
      } catch (jsonError) {
        // If standard JSON parsing fails, try to fix single quotes
        console.log('⚠️ JSON.parse failed, trying to fix single quotes:', jsonError instanceof Error ? jsonError.message : String(jsonError))
        try {
          // Replace single quotes with double quotes, but be careful not to replace single quotes within content
          const fixedJson = contextPreview
            .replace(/'/g, '"')  // Simple replacement of all single quotes with double quotes
            .replace(/""/g, '\\"')  // Fix double quote escaping

          const parsed = JSON.parse(fixedJson)
          if (Array.isArray(parsed)) {
            messages = parsed
            isStructured = true
            console.log('✅ Parsed successfully after fix, count:', messages.length)
          }
        } catch (fixError) {
          parseError = fixError instanceof Error ? fixError : new Error(String(fixError))
          console.log('❌ Parse failed even after fix:', parseError.message)
        }
      }
    } else if (contextPreview && typeof contextPreview === 'object') {
      // Handle possible object format
      if (contextPreview.messages && Array.isArray(contextPreview.messages)) {
        messages = contextPreview.messages
        isStructured = true
        console.log('✅ Object format parsed successfully, count:', messages.length)
      }
    }
  } catch (e) {
    parseError = e instanceof Error ? e : new Error(String(e))
    console.log('❌ Parse failed:', parseError.message)
  }

  // Directly display all messages
  const displayMessages = messages

  // If not structured messages, show raw content
  if (!isStructured || messages.length === 0) {
    return (
      <Card className="border-border">
        <div className="p-3">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <MessageSquare className="h-4 w-4 text-blue-500" />
              {t('agent.logs.messagesPreview.infoTitle')} {parseError ? `(${t('agent.logs.messagesPreview.parse.error')})` : ''}
            </h4>
            {parseError && (
              <Badge variant="destructive" className="text-xs">
                {t('agent.logs.messagesPreview.parse.failed')}
              </Badge>
            )}
          </div>
          {parseError && (
            <div className="text-xs text-red-500 mb-2 p-2 bg-red-500/10 rounded">
              {t('agent.logs.messagesPreview.parse.errorPrefix')} {parseError instanceof Error ? parseError.message : String(parseError)}
            </div>
          )}
          <div className="bg-muted/30 p-2 rounded text-xs font-mono max-h-32 overflow-y-auto">
            {typeof contextPreview === 'string'
              ? contextPreview
              : JSON.stringify(contextPreview, null, 2)
            }
          </div>
        </div>
      </Card>
    )
  }

  return (
    <Card className="border-border">
      <div className="p-3">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-medium flex items-center gap-2">
            <MessageSquare className="h-4 w-4 text-blue-500" />
            {t('agent.logs.messagesPreview.title')} ({messages.length}{t('agent.logs.event.common.itemsSuffix')})
          </h4>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-xs h-7"
          >
            {isExpanded ? t('agent.logs.messagesPreview.actions.collapse') : t('agent.logs.messagesPreview.actions.expand')}
            {isExpanded ? <ChevronDown className="h-3 w-3 ml-1" /> : <ChevronRight className="h-3 w-3 ml-1" />}
          </Button>
        </div>

        {isExpanded && (
          <div className="space-y-2">
            {/* Message statistics */}
            <div className="grid grid-cols-3 gap-2 mb-3">
              <div className="text-center p-2 bg-purple-500/10 rounded">
                <div className="text-xs text-purple-500">{t('agent.logs.messagesPreview.labels.role.system')}</div>
                <div className="text-sm font-medium">
                  {messages.filter(m => m.role === 'system').length}
                </div>
              </div>
              <div className="text-center p-2 bg-blue-500/10 rounded">
                <div className="text-xs text-blue-500">{t('agent.logs.messagesPreview.labels.role.user')}</div>
                <div className="text-sm font-medium">
                  {messages.filter(m => m.role === 'user').length}
                </div>
              </div>
              <div className="text-center p-2 bg-green-500/10 rounded">
                <div className="text-xs text-green-500">{t('agent.logs.messagesPreview.labels.role.assistant')}</div>
                <div className="text-sm font-medium">
                  {messages.filter(m => m.role === 'assistant').length}
                </div>
              </div>
            </div>

            {/* Message list */}
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {displayMessages.map((message, index) => (
                <MessageItem key={index} message={message} index={index} />
              ))}
            </div>
          </div>
        )}
      </div>
    </Card>
  )
}

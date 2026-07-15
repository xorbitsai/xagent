import React, { useEffect, useState } from 'react'
import ReactMarkdown, { defaultUrlTransform } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { Components, ExtraProps } from 'react-markdown'
import { apiRequest } from '@/lib/api-wrapper'
import { AgentCard } from '@/components/chat/AgentCard'
import { useI18n } from '@/contexts/i18n-context'
import { InlineFilePreview } from '@/components/file/inline-file-preview'
import {
  getInlineFilePreviewKind,
  getInlineFilePreviewMimeType,
  isPreviewableInlineFileKind,
  resolveInlineFileId,
  type PreviewableInlineFileKind,
} from '@/components/file/inline-file-preview-utils'
import { getApiUrl } from '@/lib/utils'


interface AgentInfo {
  id: number
  name: string
  description?: string
  status: 'draft' | 'published'
  instructions?: string
}

// Enhanced Markdown detection function: covers broader Markdown features not limited to starting with #
const isLikelyMarkdown = (s: string): boolean => {
  const t = s.trim()
  if (!t) return false
  return (
    t.startsWith('#') || // Heading
    s.includes('```') || // Code block
    s.includes('**') || // Bold
    /(\n|^)\s*(-|\*|\d+\.)\s/.test(s) || // List (unordered/ordered)
    (s.includes('|') && s.includes('---')) || // Table
    /\[[^\]]+\]\([^\)]+\)/.test(s) || // Link [text](url)
    /!\[[^\]]*\]\([^\)]+\)/.test(s) || // Image ![alt](url)
    /(\n|^)\s*>\s/.test(s) || // Blockquote
    /(\n|^)\s*---\s*(\n|$)/.test(s) // Horizontal rule
  )
}

interface MarkdownRendererProps {
  content: string
  className?: string
  onFileClick?: (filePath: string, fileName: string) => void
  onAgentClick?: (agentId: string, agentName: string) => void
}

const safeUrlTransform = (url: string): string => {
  if (!url) return ''
  if (url.startsWith('file:')) return url
  if (url.startsWith('agent:')) return url
  return defaultUrlTransform(url)
}

// Hook to fetch agent details
function useAgentInfo(agentId: string) {
  const [agentInfo, setAgentInfo] = useState<AgentInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    let cancelled = false

    async function fetchAgentInfo() {
      try {
        setLoading(true)
        setError(null)

        const apiUrl = getApiUrl()
        const response = await apiRequest(`${apiUrl}/api/agents/${agentId}`)

        if (!response.ok) {
          throw new Error(`Failed to fetch agent: ${response.statusText}`)
        }

        const data: AgentInfo = await response.json()

        if (!cancelled) {
          setAgentInfo(data)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err as Error)
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    fetchAgentInfo()

    return () => {
      cancelled = true
    }
  }, [agentId])

  return { agentInfo, loading, error }
}


// Agent Card Container component that fetches data
function AgentCardContainer({
  agentId,
  agentName: initialAgentName,
  onAgentClick,
}: {
  agentId: string
  agentName: string
  onAgentClick?: (agentId: string, agentName: string) => void
}) {
  const { t } = useI18n()
  const { agentInfo, loading, error } = useAgentInfo(agentId)

  // Show loading state
  if (loading) {
    return (
      <div className="inline-flex items-center gap-2 bg-muted/50 border border-border rounded-lg p-3 my-2 max-w-sm">
        <div className="w-8 h-8 rounded-md bg-muted animate-pulse" />
        <div className="flex-1">
          <div className="h-4 bg-muted rounded animate-pulse w-32 mb-1" />
          <div className="h-3 bg-muted rounded animate-pulse w-24" />
        </div>
      </div>
    )
  }

  // Show error state with fallback name
  if (error || !agentInfo) {
    return (
      <AgentCard
        agentId={agentId}
        agentName={initialAgentName}
        description={t("markdownRenderer.loadAgentDetailsFailed")}
        status="draft"
      />
    )
  }

  // Show agent info
  // Don't pass onClick - let AgentCard handle navigation internally based on status
  return (
    <AgentCard
      agentId={agentId}
      agentName={agentInfo.name}
      description={agentInfo.description || agentInfo.instructions}
      status={agentInfo.status}
    />
  )
}

function containsAgentCardElement(children: React.ReactNode): boolean {
  return React.Children.toArray(children).some((child) => {
    if (!React.isValidElement(child)) {
      return false
    }

    if (child.props?.['data-agent-card-wrapper']) {
      return true
    }

    return containsAgentCardElement(child.props?.children)
  })
}

function containsBlockPreviewElement(children: React.ReactNode): boolean {
  return React.Children.toArray(children).some((child) => {
    if (!React.isValidElement(child)) {
      return false
    }

    if (child.props?.['data-inline-file-preview-wrapper']) {
      return true
    }

    return containsBlockPreviewElement(child.props?.children)
  })
}

function hastText(node: any): string {
  if (!node) return ''
  if (typeof node.value === 'string') return node.value
  if (!Array.isArray(node.children)) return ''
  return node.children.map(hastText).join('')
}

const nodeText = (children: React.ReactNode): string => {
  return React.Children.toArray(children)
    .map((child) => {
      if (typeof child === 'string' || typeof child === 'number') {
        return String(child)
      }
      if (React.isValidElement(child)) {
        return nodeText(child.props?.children)
      }
      return ''
    })
    .join('')
}

function resolvePreviewableFileLink({
  fileNameFromPath,
  fileName,
}: {
  fileNameFromPath: string
  fileName: string
}): { previewKind: PreviewableInlineFileKind; displayFilename: string } | null {
  const pathKind = getInlineFilePreviewKind({ filename: fileNameFromPath })
  if (isPreviewableInlineFileKind(pathKind)) {
    return { previewKind: pathKind, displayFilename: fileName }
  }

  const labelKind = getInlineFilePreviewKind({ filename: fileName })
  if (isPreviewableInlineFileKind(labelKind)) {
    return { previewKind: labelKind, displayFilename: fileName }
  }

  return null
}

function containsPreviewFileLinkNode(node: any): boolean {
  if (!node) return false
  const href = node.properties?.href
  if (typeof href === 'string' && href.startsWith('file:')) {
    const filePath = href.replace(/^file:/, '')
    const fileNameFromPath = filePath.split('/').pop() || filePath
    const title = typeof node.properties?.title === 'string' ? node.properties.title : ''
    const label = title || hastText(node)
    if (resolvePreviewableFileLink({ fileNameFromPath, fileName: label })) return true
  }
  const src = node.properties?.src
  if (typeof src === 'string' && src.startsWith('file:')) {
    return true
  }
  if (!Array.isArray(node.children)) return false
  return node.children.some(containsPreviewFileLinkNode)
}

type MarkdownRendererContextValue = {
  onFileClick?: (filePath: string, fileName: string) => void
  onAgentClick?: (agentId: string, agentName: string) => void
  openLabel: string
  loadErrorText: string
}

type MarkdownComponentProps<Tag extends keyof React.JSX.IntrinsicElements> =
  React.ComponentPropsWithoutRef<Tag> & ExtraProps

const MarkdownRendererContext = React.createContext<MarkdownRendererContextValue | null>(null)

function useMarkdownRendererContext(): MarkdownRendererContextValue {
  const context = React.useContext(MarkdownRendererContext)
  if (!context) {
    throw new Error('Markdown components must be rendered within MarkdownRenderer')
  }
  return context
}

function MarkdownParagraph({
  node,
  children,
  ...props
}: MarkdownComponentProps<'p'>) {
  if (
    containsAgentCardElement(children) ||
    containsBlockPreviewElement(children) ||
    containsPreviewFileLinkNode(node)
  ) {
    return (
      <div className="my-4" {...props}>
        {children}
      </div>
    )
  }

  return <p {...props}>{children}</p>
}

function MarkdownLink({
  node,
  href,
  title,
  children,
  ...props
}: MarkdownComponentProps<'a'>) {
  const { onFileClick, onAgentClick, openLabel, loadErrorText } =
    useMarkdownRendererContext()

  if (href && href.startsWith('file:')) {
    const filePath = href.replace(/^file:/, '')
    const fileNameFromPath = filePath.split('/').pop() || filePath
    const linkText = (node ? hastText(node) : nodeText(children)).trim()
    const fileName = title || linkText || fileNameFromPath
    const preview = resolvePreviewableFileLink({ fileNameFromPath, fileName })
    const fileId = resolveInlineFileId(filePath)

    if (preview) {
      return (
        <InlineFilePreview
          source={{
            fileId,
            filename: preview.displayFilename,
            type: preview.previewKind,
            mimeType: getInlineFilePreviewMimeType(preview.previewKind),
          }}
          openLabel={openLabel}
          loadErrorText={loadErrorText}
          onFileClick={onFileClick}
        />
      )
    }

    const handleClick = (event: React.MouseEvent<HTMLAnchorElement>) => {
      if (onFileClick) {
        event.preventDefault()
        const fallbackTitle = title || linkText || fileNameFromPath
        onFileClick(fileId, fallbackTitle)
      }
    }

    return (
      <a
        href="#"
        data-file-path={filePath}
        className="file-link"
        title={title || undefined}
        onClick={handleClick}
        {...props}
      >
        {children}
      </a>
    )
  }

  if (href && href.startsWith('agent:')) {
    const agentId = href.replace(/^agent:\/\//, '')
    const agentNameFromLink =
      (node ? hastText(node) : nodeText(children)).trim() || `Agent ${agentId}`

    return React.createElement('div', {
      className: 'my-2',
      key: `agent-${agentId}-wrapper`,
      'data-agent-card-wrapper': true,
    }, React.createElement(AgentCardContainer, {
      key: `agent-${agentId}`,
      agentId,
      agentName: agentNameFromLink,
      onAgentClick,
    }))
  }

  return (
    <a href={href || undefined} title={title || undefined} {...props}>
      {children}
    </a>
  )
}

function MarkdownImage({
  node: _node,
  src,
  alt,
  title,
  ...props
}: MarkdownComponentProps<'img'>) {
  const { onFileClick, openLabel, loadErrorText } =
    useMarkdownRendererContext()
  const resolvedSrc = src || ''

  if (resolvedSrc.startsWith('file:')) {
    const filePath = resolvedSrc.replace(/^file:/, '')
    const fileNameFromPath = filePath.split('/').pop() || filePath
    const fileName = title || alt || fileNameFromPath
    const preview = resolvePreviewableFileLink({ fileNameFromPath, fileName })
    const previewKind = preview?.previewKind ?? 'image'
    return (
      <InlineFilePreview
        source={{
          fileId: resolveInlineFileId(filePath),
          filename: preview?.displayFilename ?? fileName,
          type: previewKind,
          mimeType: getInlineFilePreviewMimeType(previewKind),
        }}
        openLabel={openLabel}
        loadErrorText={loadErrorText}
        onFileClick={onFileClick}
        imageClassName="file-image cursor-pointer"
      />
    )
  }

  return <img src={resolvedSrc} alt={alt || ''} title={title || alt || ''} {...props} />
}

// Keep these component identities stable across chat/trace updates. Replacing
// them makes React remount every custom Markdown node, including a playing
// <audio> element whose playback state would then be lost.
const markdownComponents: Components = {
  p: MarkdownParagraph,
  a: MarkdownLink,
  img: MarkdownImage,
}

export function MarkdownRenderer({ content, className = '', onFileClick, onAgentClick }: MarkdownRendererProps) {
  const { t } = useI18n()
  const contextValue = React.useMemo<MarkdownRendererContextValue>(
    () => ({
      onFileClick,
      onAgentClick,
      openLabel: t('files.previewDialog.buttons.open'),
      loadErrorText: t('files.previewDialog.errors.loadFailed'),
    }),
    [onFileClick, onAgentClick, t]
  )

  return (
    <MarkdownRendererContext.Provider value={contextValue}>
      <div className={`prose prose-invert max-w-none break-words [overflow-wrap:anywhere] ${className}`}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeKatex]}
          components={markdownComponents}
          urlTransform={safeUrlTransform}
        >
          {content}
        </ReactMarkdown>
      </div>
    </MarkdownRendererContext.Provider>
  )
}

interface JsonRendererProps {
  data: any
  className?: string
  onFileClick?: (filePath: string, fileName: string) => void
  onAgentClick?: (agentId: string, agentName: string) => void
}

export function JsonRenderer({ data, className = '', onFileClick, onAgentClick }: JsonRendererProps) {
  const [expanded, setExpanded] = React.useState(true)

  if (typeof data === 'string') {
    // Try to parse as JSON first
    try {
      const parsed = JSON.parse(data)
      return <JsonRenderer data={parsed} className={className} onFileClick={onFileClick} onAgentClick={onAgentClick} />
    } catch {
      // If not JSON, try to identify Markdown more comprehensively
      if (isLikelyMarkdown(data)) {
        return <MarkdownRenderer content={data} className={className} onFileClick={onFileClick} onAgentClick={onAgentClick} />
      }
      // Otherwise display as plain text
      return (
        <pre className={`py-3 rounded text-sm font-mono overflow-x-auto whitespace-pre-wrap ${className}`}>
          {data}
        </pre>
      )
    }
  }

  if (typeof data === 'object' && data !== null) {
    // Check if it's a result object with output that might be markdown
    if (data.output && typeof data.output === 'string' && isLikelyMarkdown(data.output.trim())) {
      return (
        <div className={`space-y-3 ${className}`}>
          <div className="bg-muted p-3 rounded text-sm font-mono overflow-x-auto whitespace-pre-wrap">
            <div className="text-green-400 mb-2">✅ Task completed successfully</div>
            <div className="text-gray-400">Goal: {data.goal}</div>
          </div>
          <div className="border-t border-border pt-3">
            <div className="text-sm font-medium text-foreground mb-2">Result:</div>
            <MarkdownRenderer content={data.output} onFileClick={onFileClick} onAgentClick={onAgentClick} />
          </div>
        </div>
      )
    }

    // For other objects, display as formatted JSON
    return (
      <div className={`space-y-2 ${className}`}>
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
        >
          {expanded ? '▼' : '▶'} JSON Data
        </button>
        {expanded && (
          <pre className="bg-muted p-3 rounded text-xs font-mono overflow-x-auto whitespace-pre-wrap">
            {JSON.stringify(data, null, 2)}
          </pre>
        )}
      </div>
    )
  }

  // For other types, display as string
  return (
    <pre className={`bg-muted py-3 rounded text-sm font-mono overflow-x-auto whitespace-pre-wrap ${className}`}>
      {String(data)}
    </pre>
  )
}

import React from 'react'
import ReactMarkdown, { defaultUrlTransform } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { Components } from 'react-markdown'
import { getApiUrl } from '@/lib/utils'
import { apiRequest } from '@/lib/api-wrapper'

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
}

const safeUrlTransform = (url: string): string => {
  if (!url) return ''
  if (url.startsWith('file:')) return url
  return defaultUrlTransform(url)
}

const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i

function MarkdownFileImage({
  filePath,
  alt,
  title,
  onFileClick,
  className,
  ...props
}: {
  filePath: string
  alt?: string
  title?: string
  onFileClick?: (filePath: string, fileName: string) => void
  className?: string
  [key: string]: any
}) {
  const apiUrl = getApiUrl()
  const publicUrl = `${apiUrl}/api/files/public/preview/${encodeURIComponent(filePath)}`
  const [resolvedUrl, setResolvedUrl] = React.useState(publicUrl)

  React.useEffect(() => {
    let objectUrl: string | null = null
    let isCancelled = false

    setResolvedUrl(publicUrl)

    const runFallback = async () => {
      if (UUID_PATTERN.test(filePath)) return
      try {
        const response = await apiRequest(
          `${apiUrl}/api/files/preview/${encodeURIComponent(filePath)}`,
          {
            cache: 'no-cache',
            headers: {
              'Cache-Control': 'no-cache',
              Pragma: 'no-cache',
            },
          }
        )
        if (!response.ok) return
        const blob = await response.blob()
        objectUrl = URL.createObjectURL(blob)
        if (!isCancelled) {
          setResolvedUrl(objectUrl)
        }
      } catch {
        return
      }
    }

    void runFallback()

    return () => {
      isCancelled = true
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl)
      }
    }
  }, [apiUrl, filePath, publicUrl])

  const fileName = filePath.split('/').pop() || filePath
  const handleClick = (e: React.MouseEvent<HTMLImageElement>) => {
    if (!onFileClick) return
    e.preventDefault()
    onFileClick(filePath, fileName)
  }

  return (
    <img
      src={resolvedUrl}
      alt={alt || ''}
      title={title || alt || ''}
      data-file-path={filePath}
      className={className || 'file-image cursor-pointer'}
      onClick={handleClick}
      {...props}
    />
  )
}

export function MarkdownRenderer({ content, className = '', onFileClick }: MarkdownRendererProps) {
  const components = React.useMemo<Components>(
    () => ({
      a({ node: _node, href, title, children, ...props }) {
        if (href && href.startsWith('file:')) {
          const filePath = href.replace(/^file:/, '')
          const fileNameFromPath = filePath.split('/').pop() || filePath
          const handleClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
            if (onFileClick) {
              e.preventDefault()
              const linkText =
                (typeof children === 'string' ? children : undefined) ??
                (Array.isArray(children)
                  ? children.map((c: any) => (typeof c === 'string' ? c : '')).join('').trim() || undefined
                  : undefined)
              const fallbackTitle = title || linkText || fileNameFromPath
              onFileClick(filePath, fallbackTitle)
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

        return (
          <a href={href || undefined} title={title || undefined} {...props}>
            {children}
          </a>
        )
      },
      img({ node: _node, src, alt, title, ...props }) {
        if (src && src.startsWith('file:')) {
          const filePath = src.replace(/^file:/, '')
          return (
            <MarkdownFileImage
              filePath={filePath}
              alt={alt || ''}
              title={title || alt || ''}
              onFileClick={onFileClick}
              className="file-image cursor-pointer"
              {...props}
            />
          )
        }

        return <img src={src || ''} alt={alt || ''} title={title || alt || ''} {...props} />
      }
    }),
    [onFileClick]
  )

  return (
    <div className={`prose prose-invert max-w-none ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={components}
        urlTransform={safeUrlTransform}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}

interface JsonRendererProps {
  data: any
  className?: string
  onFileClick?: (filePath: string, fileName: string) => void
}

export function JsonRenderer({ data, className = '', onFileClick }: JsonRendererProps) {
  const [expanded, setExpanded] = React.useState(true)

  if (typeof data === 'string') {
    // Try to parse as JSON first
    try {
      const parsed = JSON.parse(data)
      return <JsonRenderer data={parsed} className={className} onFileClick={onFileClick} />
    } catch {
      // If not JSON, try to identify Markdown more comprehensively
      if (isLikelyMarkdown(data)) {
        return <MarkdownRenderer content={data} className={className} onFileClick={onFileClick} />
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
            <MarkdownRenderer content={data.output} onFileClick={onFileClick} />
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

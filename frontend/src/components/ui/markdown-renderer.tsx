import React from 'react'
import { marked } from 'marked'
import { getApiUrl } from '@/lib/utils'

// 增强的 Markdown 识别函数：覆盖更广的 Markdown 特征而不局限于以 # 开头
const isLikelyMarkdown = (s: string): boolean => {
  const t = s.trim()
  if (!t) return false
  return (
    t.startsWith('#') || // 标题
    s.includes('```') || // 代码块
    /(\n|^)\s*(-|\*|\d+\.)\s/.test(s) || // 列表（无序/有序）
    (s.includes('|') && s.includes('---')) || // 表格
    /\[[^\]]+\]\([^\)]+\)/.test(s) || // 链接 [text](url)
    /!\[[^\]]*\]\([^\)]+\)/.test(s) || // 图片 ![alt](url)
    /(\n|^)\s*>\s/.test(s) || // 引用块
    /(\n|^)\s*---\s*(\n|$)/.test(s) // 水平分割线
  )
}

interface MarkdownRendererProps {
  content: string
  className?: string
  onFileClick?: (filePath: string, fileName: string) => void
}

export function MarkdownRenderer({ content, className = '', onFileClick }: MarkdownRendererProps) {
  const [html, setHtml] = React.useState('')
  const containerRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    const parseMarkdown = async () => {
      try {
        // Custom renderer for file: protocol links and images
        const renderer = new marked.Renderer()
        const defaultLinkRenderer = renderer.link.bind(renderer)
        const defaultImageRenderer = renderer.image.bind(renderer)

        renderer.link = (href: string | null, title: string | null, text: string) => {
          // Check if this is a file: protocol link
          if (href && href.startsWith('file:')) {
            const filePath = href.replace(/^file:/, '')
            // Return a data-link attribute so we can handle it with event delegation
            return `<a href="#" data-file-path="${filePath}" class="file-link" title="${title || ''}">${text}</a>`
          }
          // Use default renderer for other links
          return defaultLinkRenderer(href, title, text)
        }

        renderer.image = (href: string | null, title: string | null, text: string) => {
          // Handle image links with file: protocol
          if (href && href.startsWith('file:')) {
            const filePath = href.replace(/^file:/, '')
            const apiUrl = getApiUrl()

            // Extract task_id from filePath (e.g., "web_task_158/output/file.png" -> "158")
            const taskIdMatch = filePath.match(/web_task_(\d+)/)
            const taskId = taskIdMatch ? taskIdMatch[1] : null
            const taskIdFromUrl =
              typeof window !== 'undefined'
                ? (window.location.pathname.match(/\/task\/(\d+)/)?.[1] || null)
                : null

            let imageUrl: string
            if (taskId) {
              // Use public preview API (no auth required)
              // Remove web_task_XX/ prefix from filePath - backend expects relative path within task
              const relativePath = filePath.replace(/^web_task_\d+\//, '')
              imageUrl = `${apiUrl}/api/files/public/preview/${taskId}/${encodeURIComponent(relativePath)}`
            } else if (taskIdFromUrl) {
              // Task page: use public preview compat with task_id query (avoid 401 from /download)
              const normalizedPath = filePath.startsWith('output/')
                ? filePath
                : `output/${filePath}`
              imageUrl = `${apiUrl}/api/files/public/preview/${encodeURIComponent(normalizedPath)}?task_id=${encodeURIComponent(taskIdFromUrl)}`
            } else {
              // Fallback to download API (requires auth, won't work in img tag)
              imageUrl = `${apiUrl}/api/files/download/${encodeURIComponent(filePath)}`
            }

            // Also add data-file-path for click preview
            return `<img src="${imageUrl}" alt="${text || ''}" title="${title || text || ''}" data-file-path="${filePath}" class="file-image cursor-pointer" />`
          }
          // Proxy remote images through backend to avoid CORS/anti-hotlink issues
          if (href && (href.startsWith('http://') || href.startsWith('https://'))) {
            const apiUrl = getApiUrl()
            const imageUrl = `${apiUrl}/api/files/proxy?url=${encodeURIComponent(href)}`
            return `<img src="${imageUrl}" alt="${text || ''}" title="${title || text || ''}" class="remote-image" />`
          }
          // Use default renderer for other images
          return defaultImageRenderer(href, title, text)
        }

        // Use marked.use() to configure renderer and disable deprecated options (marked 5.x)
        marked.use({
          renderer,
          headerIds: false,
          mangle: false,
        })
        const parsed = await marked.parse(content)

        // Safety net: if content includes raw HTML that hardcodes preview endpoints like
        // <img src="/api/files/preview/output/x.png">, the backend needs a task_id to resolve.
        // On task pages, infer taskId from URL and append ?task_id=...
        let rewritten = parsed
        if (typeof window !== 'undefined') {
          const m = window.location.pathname.match(/\/task\/(\d+)/)
          const taskIdFromUrl = m?.[1] || null
          if (taskIdFromUrl) {
            rewritten = rewritten.replace(
              /(src|href)=["'](\/api\/files\/(?:public\/)?preview\/[^"']+)["']/g,
              (match, attr, url) => {
                // If already has explicit task segment (/preview/<id>/...), don't touch.
                if (/\/api\/files\/(?:public\/)?preview\/\d+\//.test(url)) return match
                // If already has task_id, don't touch.
                if (/[?&]task_id=/.test(url)) return match
                const sep = url.includes('?') ? '&' : '?'
                return `${attr}="${url}${sep}task_id=${encodeURIComponent(taskIdFromUrl)}"`
              }
            )

            // If raw HTML uses the auth-required download endpoint for task outputs,
            // rewrite it to public preview with task_id to avoid 401 broken images.
            rewritten = rewritten.replace(
              /(src|href)=["'](\/api\/files\/download\/[^"']+)["']/g,
              (match, attr, url) => {
                if (/[?&]task_id=/.test(url)) return match
                // Only rewrite obvious task output paths to avoid changing generic downloads.
                if (!url.includes('output')) return match
                const previewUrl = url.replace('/api/files/download/', '/api/files/public/preview/')
                const sep = previewUrl.includes('?') ? '&' : '?'
                return `${attr}="${previewUrl}${sep}task_id=${encodeURIComponent(taskIdFromUrl)}"`
              }
            )

            // Second safety net: handle relative asset links inside raw HTML, e.g.
            // <img src="screenshot.png"> or <img src="output/screenshot.png">
            // Rewrite to public preview so it can be served from task output.
            rewritten = rewritten.replace(
              /(src|href)=["']([^"']+)["']/g,
              (match, attr, url) => {
                // Skip absolute/protocol/data/hash/rooted URLs
                if (/^(https?:\/|data:|\/\/|#|\/)/.test(url)) return match
                // Only rewrite common file assets to avoid breaking normal relative links
                if (!/\.(png|jpg|jpeg|gif|webp|svg|html|htm|pdf|txt|json)$/i.test(url)) return match
                if (/[?&]task_id=/.test(url)) return match

                // If the URL already embeds the web_task_<id>/... structure, use canonical public preview.
                const webTaskMatch = url.match(/^web_task_(\d+)\//)
                if (webTaskMatch?.[1]) {
                  const embeddedTaskId = webTaskMatch[1]
                  return `${attr}="/api/files/public/preview/${embeddedTaskId}/${encodeURIComponent(url)}"`
                }

                // Otherwise assume it's a task output asset; default to output/<name>
                const normalizedPath = url.startsWith('output/')
                  ? url
                  : `output/${url}`
                return `${attr}="/api/files/public/preview/${normalizedPath}?task_id=${encodeURIComponent(taskIdFromUrl)}"`
              }
            )
          }
        }

        setHtml(rewritten)
      } catch (error) {
        console.error('Error parsing markdown:', error)
        setHtml(content)
      }
    }

    parseMarkdown()
  }, [content])

  // Handle file link clicks and image clicks
  React.useEffect(() => {
    const container = containerRef.current
    if (!container || !onFileClick) return

    const handleFileClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement

      // Handle file link clicks
      const link = target.closest('.file-link') as HTMLAnchorElement
      if (link) {
        e.preventDefault()
        const filePath = link.getAttribute('data-file-path')
        if (filePath) {
          // Extract filename from path
          const fileName = filePath.split('/').pop() || filePath
          onFileClick(filePath, fileName)
        }
        return
      }

      // Handle image clicks with data-file-path attribute
      const img = target as HTMLImageElement
      if (img.tagName === 'IMG' && img.hasAttribute('data-file-path')) {
        e.preventDefault()
        const filePath = img.getAttribute('data-file-path')
        if (filePath) {
          // Extract filename from path
          const fileName = filePath.split('/').pop() || filePath
          onFileClick(filePath, fileName)
        }
      }
    }

    container.addEventListener('click', handleFileClick)
    return () => {
      container.removeEventListener('click', handleFileClick)
    }
  }, [onFileClick])

  return (
    <div
      ref={containerRef}
      className={`prose prose-invert max-w-none ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
      style={{
        // Style file links differently
        '--link-color': '#3b82f6'
      } as React.CSSProperties}
    />
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
      // 如果不是 JSON，尝试更全面地识别 Markdown
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

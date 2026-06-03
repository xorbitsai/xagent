import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function getApiUrl(): string {
  // Client-side: use environment variable or fallback to relative path
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || ''
  return apiUrl
}

const PUBLIC_ACCESS_TOKEN_STORAGE_KEY = 'xagent_public_access_token'

export function setPublicAccessToken(token: string | null): void {
  if (typeof window === 'undefined') return
  if (!token) {
    window.sessionStorage.removeItem(PUBLIC_ACCESS_TOKEN_STORAGE_KEY)
    return
  }
  window.sessionStorage.setItem(PUBLIC_ACCESS_TOKEN_STORAGE_KEY, token)
}

export function getPublicAccessToken(): string | null {
  if (typeof window === 'undefined') return null
  return window.sessionStorage.getItem(PUBLIC_ACCESS_TOKEN_STORAGE_KEY)
}

export function withPublicAccessToken(url: string): string {
  const publicToken = getPublicAccessToken()
  if (!publicToken) {
    return url
  }

  try {
    const resolved = new URL(url, typeof window !== 'undefined' ? window.location.origin : 'http://localhost')
    if (!resolved.searchParams.has('token')) {
      resolved.searchParams.set('token', publicToken)
    }
    const apiUrl = getApiUrl()
    if (apiUrl) {
      return resolved.toString()
    }
    return `${resolved.pathname}${resolved.search}${resolved.hash}`
  } catch {
    return url
  }
}

export function getFilePublicPreviewUrl(fileId: string, apiUrl = getApiUrl()): string {
  return withPublicAccessToken(`${apiUrl}/api/files/public/preview/${encodeURIComponent(fileId)}`)
}

export function getFilePublicDownloadUrl(fileId: string, apiUrl = getApiUrl()): string {
  return withPublicAccessToken(`${apiUrl}/api/files/public/download/${encodeURIComponent(fileId)}`)
}

export function getFileRelativePreviewUrl(
  fileId: string,
  relativePath: string,
  apiUrl = getApiUrl(),
): string {
  const baseUrl = new URL(getFilePublicPreviewUrl(fileId, apiUrl), typeof window !== 'undefined' ? window.location.origin : 'http://localhost')
  baseUrl.searchParams.set('relative_path', relativePath)
  if (apiUrl) {
    return baseUrl.toString()
  }
  return `${baseUrl.pathname}${baseUrl.search}${baseUrl.hash}`
}

export function getUploadApiUrl(): string {
  // Direct-to-backend URL for file uploads. The Next.js dev-server proxy
  // (rewrites in next.config.mjs) buffers entire multipart bodies into Node
  // memory before forwarding; bodies over ~5MB trigger an internal 30s
  // timeout and the client sees a plain-text "Internal Server Error" 500.
  // To avoid that, point uploads directly at the FastAPI backend.
  //
  // - In dev, set NEXT_PUBLIC_UPLOAD_API_URL=http://localhost:8000 in
  //   frontend/.env.local. FastAPI's CORS middleware already allows "*".
  // - In production (nginx + same-origin), leave it unset and uploads will
  //   reuse NEXT_PUBLIC_API_URL (which is empty by default → same-origin).
  return (
    process.env.NEXT_PUBLIC_UPLOAD_API_URL
    || process.env.NEXT_PUBLIC_API_URL
    || ''
  )
}

export function getAuthHeaders(token: string | null): Record<string, string> {
  if (!token) return {}
  return {
    'Authorization': `Bearer ${token}`
  }
}

export function getWsUrl(): string {
  // 1. Use explicit env var if set (production/staging)
  if (process.env.NEXT_PUBLIC_WS_URL) {
    return process.env.NEXT_PUBLIC_WS_URL
  }

  // 2. Auto-construct from current location (development/same-domain)
  if (typeof window !== 'undefined') {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${protocol}//${window.location.host}`
  }

  // 3. Fallback for SSR (shouldn't happen for WS, but safe)
  return ''
}

export async function fetchWithAuth(url: string, token: string | null, options: RequestInit = {}): Promise<Response> {
  const headers = {
    ...options.headers,
    ...getAuthHeaders(token)
  }

  return fetch(url, {
    ...options,
    headers
  })
}

export function formatDate(date: Date | string): string {
  const d = new Date(date)
  return d.toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  })
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes'

  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))

  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null

  return (...args: Parameters<T>) => {
    if (timeout) {
      clearTimeout(timeout)
    }

    timeout = setTimeout(() => {
      func(...args)
    }, wait)
  }
}

export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean = false

  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args)
      inThrottle = true
      setTimeout(() => {
        inThrottle = false
      }, limit)
    }
  }
}

export function generateId(): string {
  return Math.random().toString(36).substr(2, 9)
}

export function isValidUrl(url: string): boolean {
  try {
    new URL(url)
    return true
  } catch {
    return false
  }
}

export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text
  return text.slice(0, maxLength) + '...'
}

export function sanitizeFileName(fileName: string): string {
  return fileName.replace(/[^\w\s.-]/g, '').replace(/\s+/g, '_')
}

export function isHtmlFile(fileName: string): boolean {
  if (!fileName) return false
  const lowerName = fileName.toLowerCase()
  return lowerName.endsWith('.html') || lowerName.endsWith('.htm')
}

export function isMarkdownFile(fileName: string): boolean {
  if (!fileName) return false
  return fileName.toLowerCase().endsWith('.md')
}

export function isCsvFile(fileName: string): boolean {
  if (!fileName) return false
  return fileName.toLowerCase().endsWith('.csv')
}

export function isImageFile(fileName: string): boolean {
  if (!fileName) return false
  return /\.(jpg|jpeg|png|gif|webp|svg)$/i.test(fileName)
}

export function shouldAutoOpenTaskPreview(fileName: string): boolean {
  if (!fileName) return false

  const lowerName = fileName.toLowerCase()
  if (isImageFile(lowerName)) return false

  return (
    lowerName.endsWith('.pdf')
    || lowerName.endsWith('.ppt')
    || lowerName.endsWith('.pptx')
    || lowerName.endsWith('.docx')
    || lowerName.endsWith('.xlsx')
    || isCsvFile(lowerName)
    || isHtmlFile(lowerName)
  )
}

export function isToggleableFile(fileName: string): boolean {
  return isHtmlFile(fileName) || isMarkdownFile(fileName) || isCsvFile(fileName)
}

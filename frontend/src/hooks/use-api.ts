"use client"

import { useState, useEffect } from "react"
import { api } from "@/lib/api-wrapper"

interface UseApiOptions {
  immediate?: boolean
  onSuccess?: (data: any) => void
  onError?: (error: Error) => void
}

export function useApi<T = any>(
  url: string | null,
  options: UseApiOptions = {}
) {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  const execute = async (requestUrl?: string) => {
    const targetUrl = requestUrl || url
    if (!targetUrl) return

    setLoading(true)
    setError(null)

    try {
      const response = await api.get(targetUrl)

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`)
      }

      const result = await response.json()
      setData(result)
      options.onSuccess?.(result)
    } catch (err) {
      const error = err as Error
      setError(error)
      options?.onError?.(error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (options.immediate && url) {
      execute()
    }
  }, [])

  return { data, loading, error, execute, refetch: execute }
}

// Convenient API call hooks
export const apiHooks = {
  // Get data
  useGet: <T>(url: string | null, options?: UseApiOptions) =>
    useApi<T>(url, { ...options, immediate: true }),

  // Create data
  usePost: <T = any>(url: string, options?: UseApiOptions) => {
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<Error | null>(null)

    const execute = async (data?: any) => {
      setLoading(true)
      setError(null)

      try {
        const response = await api.post(url, data)

        if (!response.ok) {
          throw new Error(`API request failed: ${response.status}`)
        }

        const result = await response.json()
        options?.onSuccess?.(result)
        return result
      } catch (err) {
        const error = err as Error
        setError(error)
        options?.onError?.(error)
        throw error
      } finally {
        setLoading(false)
      }
    }

    return { loading, error, execute }
    },

  // Update data
  usePut: <T = any>(url: string, options?: UseApiOptions) => {
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<Error | null>(null)

    const execute = async (data?: any) => {
      setLoading(true)
      setError(null)

      try {
        const response = await api.put(url, data)

        if (!response.ok) {
          throw new Error(`API request failed: ${response.status}`)
        }

        const result = await response.json()
        options?.onSuccess?.(result)
        return result
      } catch (err) {
        const error = err as Error
        setError(error)
        options?.onError?.(error)
        throw error
      } finally {
        setLoading(false)
      }
    }

    return { loading, error, execute }
    },

  // Delete data
  useDelete: (url: string, options?: UseApiOptions) => {
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<Error | null>(null)

    const execute = async () => {
      setLoading(true)
      setError(null)

      try {
        const response = await api.delete(url)

        if (!response.ok) {
          throw new Error(`API request failed: ${response.status}`)
        }

        const result = await response.json()
        options?.onSuccess?.(result)
        return result
      } catch (err) {
        const error = err as Error
        setError(error)
        options?.onError?.(error)
        throw error
      } finally {
        setLoading(false)
      }
    }

    return { loading, error, execute }
    },
}

import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const routerPushMock = vi.hoisted(() => vi.fn())
const routerReplaceMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper",
  )
  return { ...actual, apiRequest: apiRequestMock }
})

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return { ...actual, getApiUrl: () => "http://api.local" }
})

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: routerPushMock, replace: routerReplaceMock }),
  useSearchParams: () => ({ get: () => null }),
}))

vi.mock("next/link", () => ({
  default: ({ children, href, ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }) => (
    <a href={href} {...props}>{children}</a>
  ),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string | number>) =>
      vars?.name ? `${key}:${vars.name}` : key,
  }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    dispatch: vi.fn(),
    setTaskId: vi.fn(),
    setPendingMessage: vi.fn(),
  }),
}))

vi.mock("@/lib/branding", () => ({
  getBrandingFromEnv: () => ({ appName: "Xagent" }),
}))

vi.mock("@/components/voice-input-controller", () => ({
  useVoiceInputControls: () => ({
    status: "idle",
    hasAsrModel: false,
    startRecording: vi.fn(),
    stopRecording: vi.fn(),
  }),
}))

vi.mock("@/components/build/deploy-agent-dialog", () => ({
  DeployAgentDialog: () => null,
}))

vi.mock("@/components/build/agent-triggers-dialog", () => ({
  AgentTriggersDialog: () => null,
}))

vi.mock("@/components/ui/popover", () => ({
  Popover: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  PopoverTrigger: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  PopoverContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: { error: toastErrorMock, success: vi.fn() },
}))

import BuildsPage from "./page"

const agent = {
  id: 42,
  name: "Research Agent",
  description: "Researches launch topics",
  logo_url: null,
  status: "draft",
  created_at: "2026-07-01T00:00:00Z",
  updated_at: "2026-07-02T00:00:00Z",
  widget_enabled: false,
  allowed_domains: [],
  can_edit: true,
  can_publish: true,
  can_delete: true,
}

const conflictPayload = {
  detail: {
    code: "agent_in_use_by_workforce",
    message: "Agent is referenced by a workforce",
    references: [{
      workforce_id: 7,
      name: "Draft Workforce",
      status: "draft",
      roles: ["worker"],
      can_edit: true,
      can_discard: true,
    }],
    has_hidden_references: false,
  },
}

function jsonResponse(data: unknown, init?: ResponseInit) {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  })
}

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
}

describe("BuildsPage Agent deletion", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    routerPushMock.mockReset()
    routerReplaceMock.mockReset()
    toastErrorMock.mockReset()
  })

  afterEach(() => cleanup())

  it("discards an eligible draft but waits for explicit Retry Delete", async () => {
    let listRequests = 0
    let deleteRequests = 0

    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url === "http://api.local/api/agents" && !options?.method) {
        listRequests += 1
        return Promise.resolve(jsonResponse(listRequests === 1 ? [agent] : []))
      }
      if (url === "http://api.local/api/agents/42" && options?.method === "DELETE") {
        deleteRequests += 1
        return Promise.resolve(
          deleteRequests === 1
            ? jsonResponse(conflictPayload, { status: 409 })
            : new Response(null, { status: 204 }),
        )
      }
      if (url === "http://api.local/api/workforces/7/discard" && options?.method === "POST") {
        return Promise.resolve(new Response(null, { status: 204 }))
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    render(<BuildsPage />)
    await screen.findByText("Research Agent")

    fireEvent.click(screen.getByRole("button", { name: "builds.list.actions.delete" }))
    fireEvent.click(screen.getByRole("button", { name: "builds.list.deleteDialog.confirm" }))

    const discard = await screen.findByRole("button", {
      name: "builds.list.deleteDialog.discardDraft:Draft Workforce",
    })
    fireEvent.click(discard)
    fireEvent.click(screen.getByRole("button", {
      name: "builds.list.deleteDialog.confirmDiscardDraft:Draft Workforce",
    }))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/workforces/7/discard",
        { method: "POST" },
      )
    })
    expect(deleteRequests).toBe(1)
    expect(screen.getByText("builds.list.deleteDialog.readyToRetry")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", {
      name: "builds.list.deleteDialog.retryDelete",
    }))

    await waitFor(() => {
      expect(deleteRequests).toBe(2)
      expect(listRequests).toBe(2)
      expect(screen.queryByText("Research Agent")).not.toBeInTheDocument()
    })
    expect(toastErrorMock).toHaveBeenCalledTimes(1)
    expect(toastErrorMock).toHaveBeenCalledWith(
      "builds.list.deleteDialog.blockedToast:Research Agent",
    )
  })

  it("keeps a committed deletion removed when the background refresh fails", async () => {
    const refresh = deferred<Response>()
    let listRequests = 0

    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url === "http://api.local/api/agents" && !options?.method) {
        listRequests += 1
        return listRequests === 1
          ? Promise.resolve(jsonResponse([agent]))
          : refresh.promise
      }
      if (url === "http://api.local/api/agents/42" && options?.method === "DELETE") {
        return Promise.resolve(new Response(null, { status: 204 }))
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    render(<BuildsPage />)
    await screen.findByText("Research Agent")

    fireEvent.click(screen.getByRole("button", { name: "builds.list.actions.delete" }))
    fireEvent.click(screen.getByRole("button", { name: "builds.list.deleteDialog.confirm" }))

    await waitFor(() => {
      expect(listRequests).toBe(2)
      expect(screen.queryByText("Research Agent")).not.toBeInTheDocument()
    })

    await act(async () => {
      refresh.resolve(new Response(null, { status: 503 }))
      await refresh.promise
    })

    await waitFor(() => {
      expect(screen.queryByText("Research Agent")).not.toBeInTheDocument()
    })
    expect(toastErrorMock).not.toHaveBeenCalled()
  })

  it("ignores an older Agent list response after a newer post-delete refresh", async () => {
    const staleRefresh = deferred<Response>()
    let listRequests = 0

    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url === "http://api.local/api/agents" && !options?.method) {
        listRequests += 1
        if (listRequests === 1) return Promise.resolve(jsonResponse([agent]))
        if (listRequests === 2) return staleRefresh.promise
        return Promise.resolve(jsonResponse([]))
      }
      if (url === "http://api.local/api/agents/42/publish" && options?.method === "POST") {
        return Promise.resolve(new Response(null, { status: 204 }))
      }
      if (url === "http://api.local/api/agents/42" && options?.method === "DELETE") {
        return Promise.resolve(new Response(null, { status: 204 }))
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    render(<BuildsPage />)
    await screen.findByText("Research Agent")

    const publishButton = screen.getByRole("button", {
      name: "builds.list.actions.publish",
    })
    const deleteButton = screen.getByRole("button", {
      name: "builds.list.actions.delete",
    })
    fireEvent.click(publishButton)
    fireEvent.click(deleteButton)
    fireEvent.click(screen.getByRole("button", {
      name: "builds.list.deleteDialog.confirm",
    }))

    await waitFor(() => {
      expect(listRequests).toBe(3)
      expect(screen.queryByText("Research Agent")).not.toBeInTheDocument()
    })

    await act(async () => {
      staleRefresh.resolve(jsonResponse([agent]))
      await staleRefresh.promise
    })

    expect(screen.queryByText("Research Agent")).not.toBeInTheDocument()
  })

  it("does not continue a deferred Agent delete after unmount", async () => {
    const deleteRequest = deferred<Response>()

    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url === "http://api.local/api/agents" && !options?.method) {
        return Promise.resolve(jsonResponse([agent]))
      }
      if (url === "http://api.local/api/agents/42" && options?.method === "DELETE") {
        return deleteRequest.promise
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    const { unmount } = render(<BuildsPage />)
    await screen.findByText("Research Agent")
    fireEvent.click(screen.getByRole("button", { name: "builds.list.actions.delete" }))
    fireEvent.click(screen.getByRole("button", { name: "builds.list.deleteDialog.confirm" }))
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/agents/42",
      { method: "DELETE" },
    ))

    unmount()
    await act(async () => {
      deleteRequest.reject(new Error("late delete failure"))
      await deleteRequest.promise.catch(() => undefined)
      await Promise.resolve()
    })

    expect(toastErrorMock).not.toHaveBeenCalled()
  })

  it("does not continue a deferred Workforce discard after unmount", async () => {
    const discardRequest = deferred<Response>()

    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url === "http://api.local/api/agents" && !options?.method) {
        return Promise.resolve(jsonResponse([agent]))
      }
      if (url === "http://api.local/api/agents/42" && options?.method === "DELETE") {
        return Promise.resolve(jsonResponse(conflictPayload, { status: 409 }))
      }
      if (url === "http://api.local/api/workforces/7/discard" && options?.method === "POST") {
        return discardRequest.promise
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    const { unmount } = render(<BuildsPage />)
    await screen.findByText("Research Agent")
    fireEvent.click(screen.getByRole("button", { name: "builds.list.actions.delete" }))
    fireEvent.click(screen.getByRole("button", { name: "builds.list.deleteDialog.confirm" }))

    fireEvent.click(await screen.findByRole("button", {
      name: "builds.list.deleteDialog.discardDraft:Draft Workforce",
    }))
    fireEvent.click(screen.getByRole("button", {
      name: "builds.list.deleteDialog.confirmDiscardDraft:Draft Workforce",
    }))
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/workforces/7/discard",
      { method: "POST" },
    ))

    unmount()
    await act(async () => {
      discardRequest.reject(new Error("late discard failure"))
      await discardRequest.promise.catch(() => undefined)
      await Promise.resolve()
    })

    expect(toastErrorMock).toHaveBeenCalledTimes(1)
    expect(toastErrorMock).toHaveBeenCalledWith(
      "builds.list.deleteDialog.blockedToast:Research Agent",
    )
  })

  it.each([
    ["workforce_not_discardable", "builds.list.deleteDialog.discardNotAllowed"],
    ["workforce_has_runs", "builds.list.deleteDialog.discardHasRuns"],
  ])("localizes stable Workforce discard error %s", async (code, translationKey) => {
    let discardRequests = 0
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url === "http://api.local/api/agents" && !options?.method) {
        return Promise.resolve(jsonResponse([agent]))
      }
      if (url === "http://api.local/api/agents/42" && options?.method === "DELETE") {
        return Promise.resolve(jsonResponse(conflictPayload, { status: 409 }))
      }
      if (url === "http://api.local/api/workforces/7/discard" && options?.method === "POST") {
        discardRequests += 1
        return Promise.resolve(jsonResponse({
          detail: { code, message: "Backend English" },
        }, { status: 409 }))
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    render(<BuildsPage />)
    await screen.findByText("Research Agent")
    fireEvent.click(screen.getByRole("button", { name: "builds.list.actions.delete" }))
    fireEvent.click(screen.getByRole("button", { name: "builds.list.deleteDialog.confirm" }))

    fireEvent.click(await screen.findByRole("button", {
      name: "builds.list.deleteDialog.discardDraft:Draft Workforce",
    }))
    fireEvent.click(screen.getByRole("button", {
      name: "builds.list.deleteDialog.confirmDiscardDraft:Draft Workforce",
    }))

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        `${translationKey}:Draft Workforce`,
      )
    })

    fireEvent.click(screen.getByRole("button", {
      name: "builds.list.deleteDialog.discardDraft:Draft Workforce",
    }))
    expect(discardRequests).toBe(1)

    fireEvent.click(screen.getByRole("button", {
      name: "builds.list.deleteDialog.confirmDiscardDraft:Draft Workforce",
    }))
    await waitFor(() => expect(discardRequests).toBe(2))
  })
})

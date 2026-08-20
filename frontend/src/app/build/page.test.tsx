import React from "react"
import { readFileSync } from "node:fs"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const routerPushMock = vi.hoisted(() => vi.fn())
const routerReplaceMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const voiceInputState = vi.hoisted(() => ({ hasAsrModel: false }))
const resolveTaskLlmSelectionMock = vi.hoisted(() => vi.fn())
const dispatchMock = vi.hoisted(() => vi.fn())
const setTaskIdMock = vi.hoisted(() => vi.fn())
const setPendingMessageMock = vi.hoisted(() => vi.fn())
const resolveAgentLogoUrlMock = vi.hoisted(() => vi.fn())
const localeMock = vi.hoisted(() => ({ value: "en" as "en" | "zh" }))

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper",
  )
  return { ...actual, apiRequest: apiRequestMock }
})

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return {
    ...actual,
    getApiUrl: () => "http://api.local",
    resolveAgentLogoUrl: (...args: Parameters<typeof actual.resolveAgentLogoUrl>) => {
      resolveAgentLogoUrlMock(...args)
      return actual.resolveAgentLogoUrl(...args)
    },
  }
})

vi.mock("@/lib/models", () => ({
  resolveTaskLlmSelection: resolveTaskLlmSelectionMock,
}))

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
    locale: localeMock.value,
  }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    dispatch: dispatchMock,
    setTaskId: setTaskIdMock,
    setPendingMessage: setPendingMessageMock,
  }),
}))

vi.mock("@/lib/branding", () => ({
  getBrandingFromEnv: () => ({ appName: "Xagent" }),
}))

vi.mock("@/components/voice-input-controller", () => ({
  useVoiceInputControls: () => ({
    status: "idle",
    hasAsrModel: voiceInputState.hasAsrModel,
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

function resetMocks() {
  apiRequestMock.mockReset()
  routerPushMock.mockReset()
  routerReplaceMock.mockReset()
  toastErrorMock.mockReset()
  resolveTaskLlmSelectionMock.mockReset()
  dispatchMock.mockReset()
  setTaskIdMock.mockReset()
  setPendingMessageMock.mockReset()
  resolveAgentLogoUrlMock.mockReset()
  localeMock.value = "en"
  voiceInputState.hasAsrModel = false
}

describe("BuildsPage rendering", () => {
  beforeEach(resetMocks)

  afterEach(() => cleanup())

  it("renders voice input in the create dialog", async () => {
    apiRequestMock.mockResolvedValue(jsonResponse([]))
    voiceInputState.hasAsrModel = true

    render(<BuildsPage />)
    await waitFor(() => {
      expect(screen.queryByText("common.loading")).not.toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole("button", {
      name: "builds.list.header.create",
    }))

    expect(screen.getByRole("button", {
      name: "voiceInput.start",
    })).toBeInTheDocument()
  })

  it("renders privileged actions for an editable Agent", async () => {
    apiRequestMock.mockResolvedValue(jsonResponse([agent]))

    render(<BuildsPage />)
    await screen.findByText("Research Agent")

    for (const name of [
      "builds.list.actions.apiKey",
      "builds.list.actions.triggers",
      "builds.list.actions.publish",
      "builds.list.actions.delete",
      "builds.list.actions.edit",
    ]) {
      expect(screen.getByRole("button", { name })).toBeInTheDocument()
    }
    expect(screen.queryByRole("button", {
      name: "builds.list.actions.viewConfig",
    })).not.toBeInTheDocument()
  })

  it("limits a published read-only Agent to run and view actions", async () => {
    apiRequestMock.mockResolvedValue(jsonResponse([{
      ...agent,
      status: "published",
      can_edit: false,
      can_publish: false,
      can_delete: false,
    }]))

    render(<BuildsPage />)
    await screen.findByText("Research Agent")

    expect(screen.getByRole("button", {
      name: "builds.list.actions.chat",
    })).toBeInTheDocument()
    expect(screen.getByRole("button", {
      name: "builds.list.actions.viewConfig",
    })).toBeInTheDocument()
    for (const name of [
      "builds.list.actions.apiKey",
      "builds.list.actions.triggers",
      "builds.list.actions.publish",
      "builds.list.actions.delete",
      "builds.list.actions.edit",
    ]) {
      expect(screen.queryByRole("button", { name })).not.toBeInTheDocument()
    }
  })

  it("uses the shared logo resolver once per card, falling back to a persona monogram", async () => {
    apiRequestMock.mockResolvedValue(jsonResponse([
      {
        ...agent,
        id: 91,
        name: "Absolute Agent",
        logo_url: "HTTPS://assets.example/agent.png",
      },
      {
        ...agent,
        id: 92,
        name: "Relative Agent",
        logo_url: "/logos/relative.png",
      },
      {
        ...agent,
        id: 93,
        name: "Invalid Agent",
        logo_url: "javascript:alert(1)",
      },
    ]))

    const view = render(<BuildsPage />)

    const absoluteName = await screen.findByText("Absolute Agent")
    const absoluteCard = absoluteName.closest("[class*='cursor-pointer']")
    const relativeCard = screen.getByText("Relative Agent").closest("[class*='cursor-pointer']")
    const invalidCard = screen.getByText("Invalid Agent").closest("[class*='cursor-pointer']")
    expect(absoluteCard).not.toBeNull()
    expect(relativeCard).not.toBeNull()
    expect(invalidCard).not.toBeNull()
    expect(absoluteCard?.querySelector("img")).toHaveAttribute(
      "src",
      "HTTPS://assets.example/agent.png",
    )
    expect(relativeCard?.querySelector("img")).toHaveAttribute(
      "src",
      "http://api.local/logos/relative.png",
    )
    expect(invalidCard?.querySelector("img")).toBeNull()
    const monogram = Array.from(invalidCard?.querySelectorAll("div") ?? []).find(
      (element) => element.textContent === "I",
    )
    expect(monogram).toBeTruthy()

    resolveAgentLogoUrlMock.mockClear()
    view.rerender(<BuildsPage />)
    expect(resolveAgentLogoUrlMock).toHaveBeenCalledTimes(3)
    expect(resolveAgentLogoUrlMock).toHaveBeenNthCalledWith(1, "HTTPS://assets.example/agent.png", "http://api.local")
    expect(resolveAgentLogoUrlMock).toHaveBeenNthCalledWith(2, "/logos/relative.png", "http://api.local")
    expect(resolveAgentLogoUrlMock).toHaveBeenNthCalledWith(3, "javascript:alert(1)", "http://api.local")
  })
})

describe("BuildsPage Agent deletion", () => {
  beforeEach(resetMocks)

  afterEach(() => cleanup())

  it("discards an eligible draft but waits for explicit Retry Delete", async () => {
    expect(voiceInputState.hasAsrModel).toBe(false)
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

function publicationAgent(status: "draft" | "published", id = 42) {
  return { ...agent, id, status }
}

function publicationActionName(kind: "publish" | "unpublish") {
  return `builds.list.actions.${kind}`
}

describe("BuildsPage publication lifecycle", () => {
  let consoleErrorMock: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    resetMocks()
    consoleErrorMock = vi.spyOn(console, "error").mockImplementation(() => undefined)
  })

  afterEach(() => {
    consoleErrorMock.mockRestore()
    cleanup()
  })

  it.each([
    ["publish", "http://api.local/api/agents/42/publish", "builds.publication.publishFailed", "Failed to publish agent:"],
    ["unpublish", "http://api.local/api/agents/42/unpublish", "builds.publication.unpublishFailed", "Failed to unpublish agent:"],
  ] as const)("reports mounted non-OK %s failures with the operation-owned key and diagnostic", async (kind, endpoint, key, diagnostic) => {
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url === "http://api.local/api/agents" && !options?.method) {
        return Promise.resolve(jsonResponse([publicationAgent(kind === "publish" ? "draft" : "published")]))
      }
      if (url === endpoint && options?.method === "POST") {
        return Promise.resolve(new Response(null, { status: 503 }))
      }
      throw new Error(`Unexpected apiRequest: ${url}`)
    })

    render(<BuildsPage />)
    await screen.findByText("Research Agent")
    fireEvent.click(screen.getByRole("button", { name: publicationActionName(kind) }))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(endpoint, { method: "POST" })
      expect(toastErrorMock).toHaveBeenCalledWith(key)
      expect(consoleErrorMock).toHaveBeenCalledWith(diagnostic, expect.any(Response))
    })
  })

  it.each([
    ["publish", "http://api.local/api/agents/42/publish", "builds.publication.publishFailed", "Failed to publish agent:"],
    ["unpublish", "http://api.local/api/agents/42/unpublish", "builds.publication.unpublishFailed", "Failed to unpublish agent:"],
  ] as const)("reports mounted rejected %s requests with the operation-owned key and diagnostic", async (kind, endpoint, key, diagnostic) => {
    const rejection = new Error(`${kind} transport rejected`)
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url === "http://api.local/api/agents" && !options?.method) {
        return Promise.resolve(jsonResponse([publicationAgent(kind === "publish" ? "draft" : "published")]))
      }
      if (url === endpoint && options?.method === "POST") return Promise.reject(rejection)
      throw new Error(`Unexpected apiRequest: ${url}`)
    })

    render(<BuildsPage />)
    await screen.findByText("Research Agent")
    fireEvent.click(screen.getByRole("button", { name: publicationActionName(kind) }))

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(key)
      expect(consoleErrorMock).toHaveBeenCalledWith(diagnostic, rejection)
    })
  })

  it("starts one list refresh after a successful publication", async () => {
    const refresh = deferred<Response>()
    let listRequests = 0
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url === "http://api.local/api/agents" && !options?.method) {
        listRequests += 1
        return listRequests === 1 ? Promise.resolve(jsonResponse([publicationAgent("draft")])) : refresh.promise
      }
      if (url === "http://api.local/api/agents/42/publish" && options?.method === "POST") {
        return Promise.resolve(new Response(null, { status: 204 }))
      }
      throw new Error(`Unexpected apiRequest: ${url}`)
    })

    render(<BuildsPage />)
    await screen.findByText("Research Agent")
    fireEvent.click(screen.getByRole("button", { name: publicationActionName("publish") }))

    await waitFor(() => expect(listRequests).toBe(2))
    expect(toastErrorMock).not.toHaveBeenCalled()

    await act(async () => {
      refresh.resolve(jsonResponse([]))
      await refresh.promise
    })
  })

  it("keeps refresh rejection in the list owner without a publication toast", async () => {
    const refreshFailure = new Error("refresh rejected")
    let listRequests = 0
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url === "http://api.local/api/agents" && !options?.method) {
        listRequests += 1
        return listRequests === 1
          ? Promise.resolve(jsonResponse([publicationAgent("draft")]))
          : Promise.reject(refreshFailure)
      }
      if (url === "http://api.local/api/agents/42/publish" && options?.method === "POST") {
        return Promise.resolve(new Response(null, { status: 204 }))
      }
      throw new Error(`Unexpected apiRequest: ${url}`)
    })

    render(<BuildsPage />)
    await screen.findByText("Research Agent")
    fireEvent.click(screen.getByRole("button", { name: publicationActionName("publish") }))

    await waitFor(() => {
      expect(listRequests).toBe(2)
      expect(consoleErrorMock).toHaveBeenCalledWith("Failed to fetch agents:", refreshFailure)
    })
    expect(toastErrorMock).not.toHaveBeenCalled()
  })

  it("keeps a refresh 503 out of publication feedback", async () => {
    let listRequests = 0
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url === "http://api.local/api/agents" && !options?.method) {
        listRequests += 1
        return Promise.resolve(
          listRequests === 1
            ? jsonResponse([publicationAgent("draft")])
            : new Response(null, { status: 503 }),
        )
      }
      if (url === "http://api.local/api/agents/42/publish" && options?.method === "POST") {
        return Promise.resolve(new Response(null, { status: 204 }))
      }
      throw new Error(`Unexpected apiRequest: ${url}`)
    })

    render(<BuildsPage />)
    await screen.findByText("Research Agent")
    fireEvent.click(screen.getByRole("button", { name: publicationActionName("publish") }))

    await waitFor(() => expect(listRequests).toBe(2))
    expect(toastErrorMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).not.toHaveBeenCalled()
  })

  it.each([
    ["204", () => new Response(null, { status: 204 })],
    ["503", () => new Response(null, { status: 503 })],
    ["rejection", () => Promise.reject(new Error("late publication rejection"))],
  ] as const)("is silent and does not refresh when an unmounted publication settles as %s", async (_outcome, settle) => {
    const mutation = deferred<Response>()
    let listRequests = 0
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url === "http://api.local/api/agents" && !options?.method) {
        listRequests += 1
        return Promise.resolve(jsonResponse([publicationAgent("draft")]))
      }
      if (url === "http://api.local/api/agents/42/publish" && options?.method === "POST") return mutation.promise
      throw new Error(`Unexpected apiRequest: ${url}`)
    })

    const { unmount } = render(<BuildsPage />)
    await screen.findByText("Research Agent")
    fireEvent.click(screen.getByRole("button", { name: publicationActionName("publish") }))
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/agents/42/publish",
      { method: "POST" },
    ))

    unmount()
    await act(async () => {
      const result = settle()
      if (result instanceof Promise) {
        mutation.reject(await result.catch((error) => error))
      } else {
        mutation.resolve(result)
      }
      await mutation.promise.catch(() => undefined)
      await Promise.resolve()
    })

    expect(listRequests).toBe(1)
    expect(toastErrorMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).not.toHaveBeenCalled()
  })

  it("does not silence an older publication failure after a newer success", async () => {
    const older = deferred<Response>()
    const newer = deferred<Response>()
    const refresh = deferred<Response>()
    let publicationRequests = 0
    let listRequests = 0
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url === "http://api.local/api/agents" && !options?.method) {
        listRequests += 1
        return listRequests === 1 ? Promise.resolve(jsonResponse([publicationAgent("draft")])) : refresh.promise
      }
      if (url === "http://api.local/api/agents/42/publish" && options?.method === "POST") {
        publicationRequests += 1
        return publicationRequests === 1 ? older.promise : newer.promise
      }
      throw new Error(`Unexpected apiRequest: ${url}`)
    })

    render(<BuildsPage />)
    await screen.findByText("Research Agent")
    const publish = screen.getByRole("button", { name: publicationActionName("publish") })
    fireEvent.click(publish)
    fireEvent.click(publish)

    await act(async () => {
      newer.resolve(new Response(null, { status: 204 }))
      await newer.promise
    })
    await waitFor(() => expect(listRequests).toBe(2))

    await act(async () => {
      older.resolve(new Response(null, { status: 503 }))
      await older.promise
    })
    await waitFor(() => expect(toastErrorMock).toHaveBeenCalledWith("builds.publication.publishFailed"))

    await act(async () => {
      refresh.resolve(jsonResponse([]))
      await refresh.promise
    })
  })

  it("starts a refresh for each reverse-order successful publish and unpublish", async () => {
    const publish = deferred<Response>()
    const unpublish = deferred<Response>()
    const firstRefresh = deferred<Response>()
    const secondRefresh = deferred<Response>()
    let listRequests = 0
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url === "http://api.local/api/agents" && !options?.method) {
        listRequests += 1
        if (listRequests === 1) return Promise.resolve(jsonResponse([
          publicationAgent("draft", 42),
          publicationAgent("published", 43),
        ]))
        return listRequests === 2 ? firstRefresh.promise : secondRefresh.promise
      }
      if (url === "http://api.local/api/agents/42/publish" && options?.method === "POST") return publish.promise
      if (url === "http://api.local/api/agents/43/unpublish" && options?.method === "POST") return unpublish.promise
      throw new Error(`Unexpected apiRequest: ${url}`)
    })

    render(<BuildsPage />)
    await screen.findAllByText("Research Agent")
    const [publishButton] = screen.getAllByRole("button", { name: publicationActionName("publish") })
    const [unpublishButton] = screen.getAllByRole("button", { name: publicationActionName("unpublish") })
    fireEvent.click(publishButton)
    fireEvent.click(unpublishButton)

    await act(async () => {
      unpublish.resolve(new Response(null, { status: 204 }))
      await unpublish.promise
    })
    await act(async () => {
      publish.resolve(new Response(null, { status: 204 }))
      await publish.promise
    })
    await waitFor(() => expect(listRequests).toBe(3))

    await act(async () => {
      firstRefresh.resolve(jsonResponse([]))
      secondRefresh.resolve(jsonResponse([]))
      await Promise.all([firstRefresh.promise, secondRefresh.promise])
    })
  })

  it("keeps the mutation and refresh phases structurally separate", () => {
    const source = readFileSync(`${process.cwd()}/src/app/build/page.tsx`, "utf8")
    const mutationStart = source.indexOf("const performPublicationMutation")
    const wrapperStart = source.indexOf("const handlePublication")
    const nextOwnerStart = source.indexOf("const [agentDeleteSession", wrapperStart)

    expect(mutationStart).toBeGreaterThan(-1)
    expect(wrapperStart).toBeGreaterThan(mutationStart)
    expect(nextOwnerStart).toBeGreaterThan(wrapperStart)

    const mutation = source.slice(mutationStart, wrapperStart)
    const wrapper = source.slice(wrapperStart, nextOwnerStart)
    expect(mutation).not.toContain("fetchAgents")
    // A bare `not.toContain("catch")` also passes if the wrapper is rewritten as
    // `.then(onFulfilled, onRejected)`, which reintroduces error handling in the
    // wrapper without the literal word "catch". Pin the exact refresh statement
    // and additionally rule out any `.then(` usage so that rewrite is caught too.
    expect(wrapper).toContain(
      "const outcome = await performPublicationMutation(agentId, kind)\n"
      + "    if (outcome === \"success\" && isMountedRef.current) {\n"
      + "      void fetchAgents()\n"
      + "    }",
    )
    expect(wrapper.match(/\.then\(/g) ?? []).toHaveLength(0)
  })
})

const successfulSelection = {
  kind: "success" as const,
  llmIds: ["general", null, null, null] as [string, null, null, null],
}

function taskCore(taskId = 7) {
  return {
    task_id: taskId,
    title: "created task",
    status: "running",
    created_at: "2026-01-01T00:00:00Z",
  }
}

function createPromptInput(): HTMLTextAreaElement {
  return screen.getByPlaceholderText("builds.list.createModal.placeholder")
}

function typeCreatePrompt(value: string) {
  fireEvent.change(createPromptInput(), { target: { value } })
}

function startBuildWithEnter() {
  fireEvent.keyDown(createPromptInput(), { key: "Enter" })
}

// Reproduces only voice-input-controller.tsx's native-setter write + bubbled
// input/change event dispatch (setNativeValue + dispatchInputEvents): the native
// HTMLTextAreaElement.prototype value setter (bypassing React's tracked instance
// setter) followed by the same bubbled input/change events production dispatches
// after a transcription lands, rather than testing-library's fireEvent convenience
// helpers. This does NOT cover insertTranscribedText's focus/caret-splice/
// setSelectionRange/fragment-data behavior — caret handling is untested here.
function simulateVoiceTranscription(target: HTMLTextAreaElement, text: string) {
  const descriptor = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")
  act(() => {
    descriptor?.set?.call(target, text)
    try {
      target.dispatchEvent(new InputEvent("input", { bubbles: true, data: text, inputType: "insertText" }))
    } catch {
      target.dispatchEvent(new Event("input", { bubbles: true }))
    }
    target.dispatchEvent(new Event("change", { bubbles: true }))
  })
}

function openCreateModal() {
  fireEvent.click(screen.getByRole("button", { name: "builds.list.header.create" }))
  return createPromptInput()
}

function configureCreateApi(create: () => Promise<Response> | Response) {
  apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
    if (url === "http://api.local/api/agents" && !options?.method) {
      return Promise.resolve(jsonResponse([]))
    }
    if (url === "http://api.local/api/chat/task/create") return create()
    throw new Error(`Unexpected apiRequest: ${url}`)
  })
}

describe("BuildsPage task creation lifecycle", () => {
  let consoleErrorMock: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    resetMocks()
    resolveTaskLlmSelectionMock.mockResolvedValue(successfulSelection)
    consoleErrorMock = vi.spyOn(console, "error").mockImplementation(() => undefined)
  })

  afterEach(() => {
    consoleErrorMock.mockRestore()
    cleanup()
  })

  it("uses the shared resolver, real task body parser, and the exact success commit order", async () => {
    const events: string[] = []
    dispatchMock.mockImplementation((action) => events.push(action.type))
    setPendingMessageMock.mockImplementation(() => events.push("pending"))
    setTaskIdMock.mockImplementation(() => {
      events.push("taskId")
      expect(createPromptInput()).toHaveValue("  hello\n  world  ")
    })
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url === "http://api.local/api/agents" && !options?.method) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/task/create") {
        expect(options?.method).toBe("POST")
        expect(JSON.parse(String(options?.body))).toEqual({
          title: "hello world",
          description: "hello\n  world",
          llm_ids: ["general", null, null, null],
        })
        return Promise.resolve(jsonResponse(taskCore(9)))
      }
      throw new Error(`Unexpected apiRequest: ${url}`)
    })

    render(<BuildsPage />)
    await waitFor(() => expect(screen.queryByText("common.loading")).not.toBeInTheDocument())
    openCreateModal()
    typeCreatePrompt("  hello\n  world  ")
    startBuildWithEnter()

    await waitFor(() => expect(setTaskIdMock).toHaveBeenCalledWith(9))
    expect(resolveTaskLlmSelectionMock).toHaveBeenCalledTimes(1)
    expect(events).toEqual(["RESET_STATE", "pending", "TRIGGER_TASK_UPDATE", "taskId"])
    expect(setPendingMessageMock).toHaveBeenCalledWith({
      message: "hello\n  world",
      files: [],
      targetTaskId: 9,
    })
    expect(screen.queryByPlaceholderText("builds.list.createModal.placeholder")).not.toBeInTheDocument()

    openCreateModal()
    expect(createPromptInput()).toHaveValue("")
  })

  it.each([
    ["no_model", { kind: "no_model" as const }, "chatPage.input.noModelAlert"],
    [
      "operational_error",
      { kind: "operational_error" as const, error: new Error("resolver failure") },
      "builds.list.createModal.startTaskFailed",
    ],
  ])("keeps the prompt and modal open for resolver %s", async (_kind, selection, expectedToast) => {
    configureCreateApi(() => Promise.resolve(jsonResponse(taskCore())))
    resolveTaskLlmSelectionMock.mockResolvedValueOnce(selection)
    render(<BuildsPage />)
    await waitFor(() => expect(screen.queryByText("common.loading")).not.toBeInTheDocument())
    openCreateModal()
    typeCreatePrompt("  preserve this exact\n  draft  ")
    startBuildWithEnter()

    await waitFor(() => expect(toastErrorMock).toHaveBeenCalledWith(expectedToast))
    expect(createPromptInput()).toHaveValue("  preserve this exact\n  draft  ")
    expect(screen.getByPlaceholderText("builds.list.createModal.placeholder")).toBeInTheDocument()
    expect(apiRequestMock).not.toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything())
    expect(dispatchMock).not.toHaveBeenCalled()
    expect(setTaskIdMock).not.toHaveBeenCalled()
    if (_kind === "operational_error") expect(consoleErrorMock).toHaveBeenCalled()
    else expect(consoleErrorMock).not.toHaveBeenCalled()
  })

  it("uses a synchronous ref latch for same-act Enter and click", async () => {
    const selection = deferred<typeof successfulSelection>()
    resolveTaskLlmSelectionMock.mockReturnValueOnce(selection.promise)
    configureCreateApi(() => Promise.resolve(jsonResponse(taskCore())))
    render(<BuildsPage />)
    await waitFor(() => expect(screen.queryByText("common.loading")).not.toBeInTheDocument())
    openCreateModal()
    typeCreatePrompt("prompt")

    await act(async () => {
      startBuildWithEnter()
      fireEvent.click(screen.getByRole("button", { name: "builds.list.createModal.buildBtn" }))
    })

    expect(resolveTaskLlmSelectionMock).toHaveBeenCalledTimes(1)
    expect(apiRequestMock).not.toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything())
  })

  it("does not POST after unmount or close while model resolution is pending", async () => {
    const selection = deferred<typeof successfulSelection>()
    resolveTaskLlmSelectionMock.mockReturnValueOnce(selection.promise)
    configureCreateApi(() => Promise.resolve(jsonResponse(taskCore())))
    const view = render(<BuildsPage />)
    await waitFor(() => expect(screen.queryByText("common.loading")).not.toBeInTheDocument())
    openCreateModal()
    typeCreatePrompt("prompt")
    startBuildWithEnter()
    view.unmount()
    await act(async () => selection.resolve(successfulSelection))
    expect(apiRequestMock).not.toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything())

    const closedSelection = deferred<typeof successfulSelection>()
    resolveTaskLlmSelectionMock.mockReturnValueOnce(closedSelection.promise)
    render(<BuildsPage />)
    await waitFor(() => expect(screen.queryByText("common.loading")).not.toBeInTheDocument())
    openCreateModal()
    typeCreatePrompt("prompt")
    startBuildWithEnter()
    fireEvent.click(screen.getByRole("button", { name: "Close" }))
    await act(async () => closedSelection.resolve(successfulSelection))
    expect(apiRequestMock).not.toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything())
    expect(dispatchMock).not.toHaveBeenCalled()
    expect(toastErrorMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).not.toHaveBeenCalled()
  })

  it("does not parse after close before transport completion, or publish after parsing starts", async () => {
    const response = deferred<Response>()
    const taskResponse = new Response(JSON.stringify(taskCore()))
    const text = vi.fn(taskResponse.text.bind(taskResponse))
    Object.defineProperty(taskResponse, "text", { value: text })
    configureCreateApi(() => response.promise)
    render(<BuildsPage />)
    await waitFor(() => expect(screen.queryByText("common.loading")).not.toBeInTheDocument())
    openCreateModal()
    typeCreatePrompt("prompt")
    startBuildWithEnter()
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything()))
    fireEvent.click(screen.getByRole("button", { name: "Close" }))
    await act(async () => response.resolve(taskResponse))
    expect(text).not.toHaveBeenCalled()
    expect(dispatchMock).not.toHaveBeenCalled()

    cleanup()
    const body = deferred<string>()
    const parsingResponse = new Response("")
    const parsingText = vi.fn(() => body.promise)
    Object.defineProperty(parsingResponse, "text", { value: parsingText })
    configureCreateApi(() => Promise.resolve(parsingResponse))
    render(<BuildsPage />)
    await waitFor(() => expect(screen.queryByText("common.loading")).not.toBeInTheDocument())
    openCreateModal()
    typeCreatePrompt("prompt")
    startBuildWithEnter()
    await waitFor(() => expect(parsingText).toHaveBeenCalledTimes(1))
    fireEvent.click(screen.getByRole("button", { name: "Close" }))
    await act(async () => body.resolve(JSON.stringify(taskCore())))
    expect(dispatchMock).not.toHaveBeenCalled()
    expect(setPendingMessageMock).not.toHaveBeenCalled()
    expect(setTaskIdMock).not.toHaveBeenCalled()
    expect(toastErrorMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).not.toHaveBeenCalled()
  })

  it("does not parse or publish after unmount at either task-body boundary", async () => {
    const response = deferred<Response>()
    const taskResponse = new Response(JSON.stringify(taskCore()))
    const text = vi.fn(taskResponse.text.bind(taskResponse))
    Object.defineProperty(taskResponse, "text", { value: text })
    configureCreateApi(() => response.promise)
    const firstView = render(<BuildsPage />)
    await waitFor(() => expect(screen.queryByText("common.loading")).not.toBeInTheDocument())
    openCreateModal()
    typeCreatePrompt("prompt")
    startBuildWithEnter()
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything()))
    firstView.unmount()
    await act(async () => response.resolve(taskResponse))
    expect(text).not.toHaveBeenCalled()
    expect(dispatchMock).not.toHaveBeenCalled()
    expect(setPendingMessageMock).not.toHaveBeenCalled()
    expect(setTaskIdMock).not.toHaveBeenCalled()
    expect(toastErrorMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).not.toHaveBeenCalled()

    const body = deferred<string>()
    const parsingResponse = new Response("")
    const parsingText = vi.fn(() => body.promise)
    Object.defineProperty(parsingResponse, "text", { value: parsingText })
    configureCreateApi(() => Promise.resolve(parsingResponse))
    const secondView = render(<BuildsPage />)
    await waitFor(() => expect(screen.queryByText("common.loading")).not.toBeInTheDocument())
    openCreateModal()
    typeCreatePrompt("prompt")
    startBuildWithEnter()
    await waitFor(() => expect(parsingText).toHaveBeenCalledTimes(1))
    secondView.unmount()
    await act(async () => body.resolve(JSON.stringify(taskCore())))
    expect(dispatchMock).not.toHaveBeenCalled()
    expect(setPendingMessageMock).not.toHaveBeenCalled()
    expect(setTaskIdMock).not.toHaveBeenCalled()
    expect(toastErrorMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).not.toHaveBeenCalled()
  })

  it("preserves the controlled prompt when a user closes and reopens the modal", async () => {
    configureCreateApi(() => Promise.resolve(jsonResponse(taskCore())))
    render(<BuildsPage />)
    await waitFor(() => expect(screen.queryByText("common.loading")).not.toBeInTheDocument())
    openCreateModal()
    typeCreatePrompt("  preserve this draft  ")
    fireEvent.click(screen.getByRole("button", { name: "Close" }))
    openCreateModal()
    expect(createPromptInput()).toHaveValue("  preserve this draft  ")
  })

  it("cancels before Manual navigation and keeps late completion silent", async () => {
    const response = deferred<Response>()
    configureCreateApi(() => response.promise)
    render(<BuildsPage />)
    await waitFor(() => expect(screen.queryByText("common.loading")).not.toBeInTheDocument())
    openCreateModal()
    typeCreatePrompt("prompt")
    startBuildWithEnter()
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything()))

    fireEvent.click(screen.getByRole("button", { name: "builds.list.createModal.manualBtn" }))
    expect(routerPushMock).toHaveBeenCalledWith("/build/new")
    await act(async () => response.resolve(jsonResponse(taskCore())))

    expect(dispatchMock).not.toHaveBeenCalled()
    expect(setPendingMessageMock).not.toHaveBeenCalled()
    expect(setTaskIdMock).not.toHaveBeenCalled()
    expect(toastErrorMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).not.toHaveBeenCalled()
  })

  it.each([
    ["non-OK", jsonResponse(taskCore(), { status: 500 })],
    ["empty", new Response("")],
    ["malformed", new Response("{")],
    ["invalid core", jsonResponse({ id: 7 })],
  ])("keeps task state at zero for current %s task-body failures", async (_name, response) => {
    configureCreateApi(() => Promise.resolve(response))
    render(<BuildsPage />)
    await waitFor(() => expect(screen.queryByText("common.loading")).not.toBeInTheDocument())
    openCreateModal()
    typeCreatePrompt("prompt")
    startBuildWithEnter()
    await waitFor(() => expect(toastErrorMock).toHaveBeenCalledWith("builds.list.createModal.startTaskFailed"))
    expect(dispatchMock).not.toHaveBeenCalled()
    expect(setPendingMessageMock).not.toHaveBeenCalled()
    expect(setTaskIdMock).not.toHaveBeenCalled()
  })

  it("treats an unreadable task body as a current operational failure", async () => {
    const response = new Response("unreadable")
    Object.defineProperty(response, "text", {
      value: vi.fn().mockRejectedValue(new Error("body unavailable")),
    })
    configureCreateApi(() => Promise.resolve(response))
    render(<BuildsPage />)
    await waitFor(() => expect(screen.queryByText("common.loading")).not.toBeInTheDocument())
    openCreateModal()
    typeCreatePrompt("prompt")
    startBuildWithEnter()
    await waitFor(() => expect(toastErrorMock).toHaveBeenCalledWith("builds.list.createModal.startTaskFailed"))
    expect(dispatchMock).not.toHaveBeenCalled()
  })

  it("preserves B after A succeeds or fails, and preserves A after ABA", async () => {
    const success = deferred<Response>()
    configureCreateApi(() => success.promise)
    render(<BuildsPage />)
    await waitFor(() => expect(screen.queryByText("common.loading")).not.toBeInTheDocument())
    openCreateModal()
    typeCreatePrompt("A")
    startBuildWithEnter()
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything()))
    typeCreatePrompt("B")
    await act(async () => success.resolve(jsonResponse(taskCore())))
    expect(screen.queryByPlaceholderText("builds.list.createModal.placeholder")).not.toBeInTheDocument()
    openCreateModal()
    expect(createPromptInput()).toHaveValue("B")

    cleanup()
    const failed = deferred<Response>()
    configureCreateApi(() => failed.promise)
    render(<BuildsPage />)
    await waitFor(() => expect(screen.queryByText("common.loading")).not.toBeInTheDocument())
    openCreateModal()
    typeCreatePrompt("A")
    startBuildWithEnter()
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything()))
    typeCreatePrompt("B")
    await act(async () => failed.resolve(jsonResponse(taskCore(), { status: 500 })))
    expect(createPromptInput()).toHaveValue("B")
    typeCreatePrompt("A")
    const aba = deferred<Response>()
    configureCreateApi(() => aba.promise)
    fireEvent.click(screen.getByRole("button", { name: "builds.list.createModal.buildBtn" }))
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything()))
    typeCreatePrompt("B")
    typeCreatePrompt("A")
    await act(async () => aba.resolve(jsonResponse(taskCore())))
    openCreateModal()
    expect(createPromptInput()).toHaveValue("A")
  })

  it("treats a native voice transcription write as an edit that blocks a stale successful clear (A3)", async () => {
    const result = deferred<Response>()
    configureCreateApi(() => result.promise)
    render(<BuildsPage />)
    await waitFor(() => expect(screen.queryByText("common.loading")).not.toBeInTheDocument())
    openCreateModal()
    typeCreatePrompt("A")
    startBuildWithEnter()
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything()))

    simulateVoiceTranscription(createPromptInput(), "A transcribed by voice")
    expect(createPromptInput()).toHaveValue("A transcribed by voice")

    await act(async () => result.resolve(jsonResponse(taskCore())))

    expect(setPendingMessageMock).toHaveBeenCalledTimes(1)
    expect(setTaskIdMock).toHaveBeenCalledWith(7)
    openCreateModal()
    expect(createPromptInput()).toHaveValue("A transcribed by voice")
  })

  it("keeps B pending when stale A settles first, and does not operationalize commit exceptions", async () => {
    const firstSelection = deferred<typeof successfulSelection>()
    const secondSelection = deferred<typeof successfulSelection>()
    resolveTaskLlmSelectionMock
      .mockReturnValueOnce(firstSelection.promise)
      .mockReturnValueOnce(secondSelection.promise)
    configureCreateApi(() => Promise.resolve(jsonResponse(taskCore(8))))
    render(<BuildsPage />)
    await waitFor(() => expect(screen.queryByText("common.loading")).not.toBeInTheDocument())
    openCreateModal()
    typeCreatePrompt("A")
    startBuildWithEnter()
    fireEvent.click(screen.getByRole("button", { name: "Close" }))
    openCreateModal()
    typeCreatePrompt("B")
    startBuildWithEnter()
    await act(async () => firstSelection.resolve(successfulSelection))
    expect(screen.getByRole("button", { name: "common.loading" })).toBeDisabled()
    await act(async () => secondSelection.resolve(successfulSelection))
    await waitFor(() => expect(setTaskIdMock).toHaveBeenCalledWith(8))

    cleanup()
    dispatchMock.mockReset()
    setPendingMessageMock.mockReset()
    setTaskIdMock.mockReset()
    toastErrorMock.mockReset()
    dispatchMock.mockImplementation((action) => {
      if (action.type === "TRIGGER_TASK_UPDATE") throw new Error("commit failed")
    })
    configureCreateApi(() => Promise.resolve(jsonResponse(taskCore())))
    render(<BuildsPage />)
    await waitFor(() => expect(screen.queryByText("common.loading")).not.toBeInTheDocument())
    openCreateModal()
    typeCreatePrompt("prompt")
    startBuildWithEnter()
    await waitFor(() => expect(dispatchMock).toHaveBeenCalledWith({ type: "TRIGGER_TASK_UPDATE" }))
    expect(dispatchMock).toHaveBeenCalledWith({ type: "RESET_STATE" })
    expect(setPendingMessageMock).toHaveBeenCalledTimes(1)
    expect(setTaskIdMock).not.toHaveBeenCalled()
    expect(toastErrorMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).toHaveBeenCalled()

    const resolverCallsBeforeRetry = resolveTaskLlmSelectionMock.mock.calls.length
    dispatchMock.mockReset()
    setTaskIdMock.mockReset()
    await waitFor(() => expect(screen.getByRole("button", { name: "builds.list.createModal.buildBtn" })).toBeEnabled())
    fireEvent.click(screen.getByRole("button", { name: "builds.list.createModal.buildBtn" }))
    await waitFor(() => expect(resolveTaskLlmSelectionMock).toHaveBeenCalledTimes(resolverCallsBeforeRetry + 1))
    await waitFor(() => expect(setTaskIdMock).toHaveBeenCalledWith(7))
    expect(toastErrorMock).not.toHaveBeenCalled()
  })

  it.each([
    "resolver rejects",
    "transport rejects",
    "Response.text() rejects and parseApiResponse falls back to empty parsed data",
  ] as const)(
    "keeps B pending and silent when stale A's %s",
    async (phase) => {
      const firstSelection = deferred<typeof successfulSelection>()
      const secondSelection = deferred<typeof successfulSelection>()
      const firstTransport = deferred<Response>()
      const firstBody = deferred<string>()
      const firstResponse = new Response("")
      Object.defineProperty(firstResponse, "text", {
        value: vi.fn(() => firstBody.promise),
      })

      if (phase === "resolver rejects") {
        resolveTaskLlmSelectionMock
          .mockReturnValueOnce(firstSelection.promise)
          .mockReturnValueOnce(secondSelection.promise)
      } else {
        resolveTaskLlmSelectionMock
          .mockResolvedValueOnce(successfulSelection)
          .mockReturnValueOnce(secondSelection.promise)
      }
      let taskRequestCount = 0
      configureCreateApi(() => {
        taskRequestCount += 1
        return phase !== "resolver rejects" && taskRequestCount === 1
          ? firstTransport.promise
          : Promise.resolve(jsonResponse(taskCore()))
      })

      render(<BuildsPage />)
      await waitFor(() => expect(screen.queryByText("common.loading")).not.toBeInTheDocument())
      openCreateModal()
      typeCreatePrompt("A")
      startBuildWithEnter()

      if (phase === "transport rejects") {
        await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith(
          "http://api.local/api/chat/task/create",
          expect.anything(),
        ))
      }
      if (phase === "Response.text() rejects and parseApiResponse falls back to empty parsed data") {
        await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith(
          "http://api.local/api/chat/task/create",
          expect.anything(),
        ))
        await act(async () => firstTransport.resolve(firstResponse))
        await waitFor(() => expect(firstResponse.text).toHaveBeenCalledTimes(1))
      }

      fireEvent.click(screen.getByRole("button", { name: "Close" }))
      openCreateModal()
      typeCreatePrompt("B")
      startBuildWithEnter()
      expect(screen.getByRole("button", { name: "common.loading" })).toBeDisabled()

      dispatchMock.mockClear()
      setPendingMessageMock.mockClear()
      setTaskIdMock.mockClear()
      toastErrorMock.mockClear()
      consoleErrorMock.mockClear()
      const staleFailure = new Error(`stale ${phase}`)
      if (phase === "resolver rejects") await act(async () => firstSelection.reject(staleFailure))
      if (phase === "transport rejects") await act(async () => firstTransport.reject(staleFailure))
      if (phase === "Response.text() rejects and parseApiResponse falls back to empty parsed data") {
        await act(async () => firstBody.reject(staleFailure))
      }

      expect(screen.getByPlaceholderText("builds.list.createModal.placeholder")).toHaveValue("B")
      expect(screen.getByRole("button", { name: "common.loading" })).toBeDisabled()
      expect(dispatchMock).not.toHaveBeenCalled()
      expect(setPendingMessageMock).not.toHaveBeenCalled()
      expect(setTaskIdMock).not.toHaveBeenCalled()
      expect(toastErrorMock).not.toHaveBeenCalled()
      expect(consoleErrorMock).not.toHaveBeenCalled()

      await act(async () => secondSelection.resolve(successfulSelection))
      await waitFor(() => expect(setTaskIdMock).toHaveBeenCalledWith(7))
    },
  )
})

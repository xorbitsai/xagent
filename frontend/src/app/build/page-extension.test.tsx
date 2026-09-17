import React from "react"
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const routerPushMock = vi.hoisted(() => vi.fn())
const routerReplaceMock = vi.hoisted(() => vi.fn())
const cardRenderMock = vi.hoisted(() => vi.fn())
const cardPropsMock = vi.hoisted(() => vi.fn())
const cardActionMock = vi.hoisted(() => vi.fn())
const providerLoaderMock = vi.hoisted(() => vi.fn())
const providerLifetime = vi.hoisted(() => ({ mounts: 0, unmounts: 0 }))

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
  default: ({
    children,
    href,
    ...props
  }: React.AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }) => (
    <a href={href} {...props}>{children}</a>
  ),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
    locale: "en",
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
    hasAsrModel: true,
    startRecording: vi.fn(),
    stopRecording: vi.fn(),
  }),
}))

vi.mock("@/lib/build-page-extension", async () => {
  const ReactModule = await vi.importActual<typeof import("react")>("react")
  const missingProvider = Symbol("missing Build page extension Provider")
  const BuildPageExtensionContext = ReactModule.createContext<
    Record<string, string> | null | typeof missingProvider
  >(missingProvider)
  const BuildPageExtensionProvider = ReactModule.memo(
    ({ children }: { children: React.ReactNode }) => {
      const [sharedData, setSharedData] = ReactModule.useState<
        Record<string, string> | null
      >(null)
      ReactModule.useEffect(() => {
        let active = true
        providerLifetime.mounts += 1
        void providerLoaderMock().then((data: Record<string, string>) => {
          if (active) {
            setSharedData(data)
          }
        })
        return () => {
          active = false
          providerLifetime.unmounts += 1
        }
      }, [])
      return ReactModule.createElement(
        "div",
        { "data-testid": "build-extension-provider" },
        ReactModule.createElement(
          BuildPageExtensionContext.Provider,
          { value: sharedData },
          children,
        ),
      )
    },
  )
  const BuildAgentCardExtension = ReactModule.memo(
    (props: { agentId: number }) => {
      ReactModule.useState(null)
      const sharedData = ReactModule.useContext(BuildPageExtensionContext)
      if (sharedData === missingProvider) {
        throw new Error("Build Agent card extension requires its Provider")
      }
      const { agentId } = props
      cardRenderMock(agentId)
      cardPropsMock(props)
      const supplement = sharedData?.[String(agentId)]
      return ReactModule.createElement(
        ReactModule.Fragment,
        null,
        supplement
          ? ReactModule.createElement(
            "div",
            { "data-testid": `agent-card-supplement-${agentId}` },
            supplement,
          )
          : null,
        ReactModule.createElement(
          "button",
          {
            type: "button",
            onClick: () => cardActionMock(agentId),
          },
          `extension-action-${agentId}`,
        ),
      )
    },
  )
  return {
    BuildPageExtensionProvider,
    BuildAgentCardExtension,
  }
})

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
  toast: { error: vi.fn(), success: vi.fn() },
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

function jsonResponse(data: unknown, init?: ResponseInit) {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  })
}

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise
  })
  return { promise, resolve }
}

describe("BuildsPage extension boundaries", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    routerPushMock.mockReset()
    routerReplaceMock.mockReset()
    cardRenderMock.mockClear()
    cardPropsMock.mockClear()
    cardActionMock.mockClear()
    providerLoaderMock.mockReset()
    providerLoaderMock.mockResolvedValue({ "42": "supplement-42" })
    providerLifetime.mounts = 0
    providerLifetime.unmounts = 0
  })

  afterEach(() => cleanup())

  it("renders a hook-bearing supplement for a real Agent card", async () => {
    apiRequestMock.mockResolvedValue(jsonResponse([agent]))

    render(<BuildsPage />)

    await screen.findByText("Research Agent")
    expect(await screen.findByTestId("agent-card-supplement-42"))
      .toBeInTheDocument()
    expect(cardRenderMock).toHaveBeenCalledWith(42)
    expect(cardPropsMock).toHaveBeenCalledWith({ agentId: 42 })
  })

  it("keeps interactive extension controls inside the card event boundary", async () => {
    apiRequestMock.mockResolvedValue(jsonResponse([agent]))

    render(<BuildsPage />)

    const cardTitle = await screen.findByText("Research Agent")
    fireEvent.click(await screen.findByRole("button", {
      name: "extension-action-42",
    }))

    expect(cardActionMock).toHaveBeenCalledOnce()
    expect(cardActionMock).toHaveBeenCalledWith(42)
    expect(routerPushMock).not.toHaveBeenCalled()

    fireEvent.click(cardTitle)
    expect(routerPushMock).toHaveBeenCalledOnce()
    expect(routerPushMock).toHaveBeenCalledWith("/build/42")
  })

  it("shares one provider load across multiple Agent card extensions", async () => {
    const agents = [1, 2, 3].map((id) => ({
      ...agent,
      id,
      name: `Agent ${id}`,
    }))
    const providerLoad = deferred<Record<string, string>>()
    providerLoaderMock.mockReturnValue(providerLoad.promise)
    apiRequestMock.mockResolvedValue(jsonResponse(agents))

    render(<BuildsPage />)
    await screen.findByText("Agent 3")

    for (const id of [1, 2, 3]) {
      expect(screen.queryByTestId(
        `agent-card-supplement-${id}`,
      )).not.toBeInTheDocument()
    }
    expect(providerLoaderMock).toHaveBeenCalledTimes(1)

    await act(async () => {
      providerLoad.resolve({
        "1": "shared-value-1",
        "2": "shared-value-2",
        "3": "shared-value-3",
      })
      await providerLoad.promise
    })

    for (const id of [1, 2, 3]) {
      const supplement = await screen.findByTestId(
        `agent-card-supplement-${id}`,
      )
      expect(within(supplement).getByText(`shared-value-${id}`))
        .toBeInTheDocument()
    }
    expect(providerLoaderMock).toHaveBeenCalledTimes(1)
    expect(new Set(cardPropsMock.mock.calls.map(([props]) => props.agentId)))
      .toEqual(new Set([1, 2, 3]))
    for (const [props] of cardPropsMock.mock.calls) {
      expect(props).toEqual({ agentId: props.agentId })
    }
  })

  it("keeps the page provider mounted across loading and a real publish refresh", async () => {
    const initialAgents = deferred<Response>()
    const refreshedAgents = deferred<Response>()
    let agentListRequests = 0

    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url === "http://api.local/api/agents" && !options?.method) {
        agentListRequests += 1
        return agentListRequests === 1
          ? initialAgents.promise
          : refreshedAgents.promise
      }
      if (
        url === "http://api.local/api/agents/42/publish" &&
        options?.method === "POST"
      ) {
        return Promise.resolve(new Response(null, { status: 204 }))
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    const view = render(<BuildsPage />)
    expect(screen.getByTestId("build-extension-provider")).toBeInTheDocument()
    expect(providerLifetime).toEqual({ mounts: 1, unmounts: 0 })

    await act(async () => {
      initialAgents.resolve(jsonResponse([agent]))
      await initialAgents.promise
    })
    await screen.findByText("Research Agent")

    fireEvent.click(screen.getByRole("button", {
      name: "builds.list.actions.publish",
    }))
    await waitFor(() => expect(agentListRequests).toBe(2))
    expect(screen.getByText("common.loading")).toBeInTheDocument()
    expect(providerLifetime).toEqual({ mounts: 1, unmounts: 0 })

    await act(async () => {
      refreshedAgents.resolve(jsonResponse([]))
      await refreshedAgents.promise
    })
    await screen.findByText("builds.emptyState.title")
    expect(providerLifetime).toEqual({ mounts: 1, unmounts: 0 })

    view.unmount()
    expect(providerLifetime).toEqual({ mounts: 1, unmounts: 1 })
  })
})

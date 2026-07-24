import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

// Issue #976: the remove-logo button should let a logo be cleared without a
// replacement, and issue #975's cache-busting fix must not mark an agent
// dirty for a logo it never had (upload-then-remove on a logo-less agent).

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper"
  )
  return { ...actual, apiRequest: apiRequestMock }
})

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return {
    ...actual,
    getApiUrl: () => "http://api.local",
    getUploadApiUrl: () => "http://api.local",
    getWsUrl: () => "ws://api.local",
  }
})

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    state: {
      messages: [],
      traceEvents: [],
      currentTask: null,
      isProcessing: false,
      isHistoryLoading: false,
      taskId: null,
      filePreview: { isOpen: false },
      dagExecution: null,
      steps: [],
    },
    setTaskId: vi.fn(),
    sendMessage: vi.fn(),
    dispatch: vi.fn(),
    closeFilePreview: vi.fn(),
    pauseTask: vi.fn(),
    resumeTask: vi.fn(),
    openFilePreview: vi.fn(),
    requestStatus: vi.fn(),
  }),
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ token: "token", user: { id: "1", is_admin: false } }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    locale: "en",
    t: (key: string, vars?: Record<string, string>) =>
      vars?.appName ? `${key}:${vars.appName}` : key,
  }),
}))

vi.mock("@/contexts/mcp-apps-context", () => ({
  useMcpApps: () => ({ apps: [], getAppIcon: () => null }),
}))

vi.mock("@/lib/branding", () => ({
  getBrandingFromEnv: () => ({ appName: "Xagent" }),
}))

vi.mock("sonner", () => ({ toast: { error: vi.fn(), success: vi.fn() } }))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useSearchParams: () => ({ get: () => null }),
}))

vi.mock("@/components/layout/resizable-three-column-layout", () => ({
  ResizableThreeColumnLayout: ({ middlePanel }: { middlePanel: React.ReactNode }) => (
    <div>{middlePanel}</div>
  ),
}))

vi.mock("@/components/task/task-conversation-panel", () => ({
  TaskConversationPanel: () => null,
}))

vi.mock("@/components/build/agent-builder-chat", () => ({ AgentBuilderChat: () => null }))
vi.mock("@/components/kb/knowledge-base-creation-dialog", () => ({
  KnowledgeBaseCreationDialog: () => null,
}))
vi.mock("@/components/mcp/connect-mcp-dialog", () => ({
  ConnectMcpDialog: () => null,
}))
vi.mock("@/components/chat/FileMentionDropdown", () => ({ FileMentionDropdown: () => null }))
vi.mock("@/hooks/use-file-mention", () => ({
  useFileMention: () => ({
    checkTrigger: vi.fn(),
    isOpen: false,
    items: [],
    selectedIndex: 0,
    selectItem: vi.fn(),
    close: vi.fn(),
  }),
}))
vi.mock("@/components/ui/multi-select", () => ({
  MultiSelect: (props: any) => (
    <div data-testid="multi-select" data-placeholder={props.placeholder}>
      {(props.options || []).map((o: any) => o.value).join("|")}
    </div>
  ),
}))
vi.mock("@/components/ui/select", () => ({ Select: () => null }))
vi.mock("@/components/build/build-file-preview-sheet", () => ({
  BuildFilePreviewSheet: () => null,
}))

import { AgentBuilder } from "./agent-builder"

const AGENT_ID = "5"

function agentResponse(logoUrl: string | null) {
  return {
    id: Number(AGENT_ID),
    user_id: 1,
    team_id: null,
    name: "Logo Test Agent",
    description: "",
    instructions: "You are a test agent.",
    execution_mode: "balanced",
    models: { general: "10" },
    knowledge_bases: [],
    skills: [],
    tool_categories: [],
    suggested_prompts: [],
    logo_url: logoUrl,
    status: "draft",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    widget_enabled: false,
    allowed_domains: [],
    share_enabled: false,
    share_updated_at: null,
    can_edit: true,
  }
}

function installApi(logoUrl: string | null) {
  apiRequestMock.mockImplementation((url: string, opts?: { method?: string; body?: string }) => {
    if (opts?.method === "PUT") {
      const body = JSON.parse(opts.body || "{}")
      const nextLogoUrl = body.logo_base64 === "" ? null : logoUrl
      return Promise.resolve(
        new Response(JSON.stringify(agentResponse(nextLogoUrl)), { status: 200 })
      )
    }
    if (url.endsWith("/api/kb/collections"))
      return Promise.resolve(new Response(JSON.stringify({ collections: [] }), { status: 200 }))
    if (url.endsWith("/api/skills/"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.endsWith("/api/tools/available"))
      return Promise.resolve(new Response(JSON.stringify({ tools: [] }), { status: 200 }))
    if (url.endsWith("/api/models/?category=llm"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.endsWith("/api/models/user-default"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.includes(`/api/agents/${AGENT_ID}/triggers`))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    if (url.endsWith(`/api/agents/${AGENT_ID}`))
      return Promise.resolve(new Response(JSON.stringify(agentResponse(logoUrl)), { status: 200 }))
    if (url.includes("/api/mcp/servers"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))
    return Promise.resolve(new Response(JSON.stringify({}), { status: 200 }))
  })
}

const updateButton = () => screen.getByText("builds.editor.header.update")
const logoFileInput = (container: HTMLElement) =>
  container.querySelector('input[type="file"][accept="image/*"]') as HTMLInputElement
const removeLogoButton = () =>
  screen.queryByRole("button", { name: "builds.configForm.logo.remove" })
const pngFile = (name: string) => new File(["fake-bytes"], name, { type: "image/png" })

beforeEach(() => {
  apiRequestMock.mockReset()
  ;(globalThis as any).WebSocket = vi.fn()
})

afterEach(() => cleanup())

describe("AgentBuilder logo removal (issue #976)", () => {
  it("does not show a remove button when the agent has no logo", async () => {
    installApi(null)
    render(<AgentBuilder agentId={AGENT_ID} />)

    await waitFor(() =>
      expect(screen.getByDisplayValue("Logo Test Agent")).toBeInTheDocument()
    )
    expect(removeLogoButton()).toBeNull()
  })

  it("clears an existing logo and sends logo_base64: '' on save", async () => {
    installApi("/uploads/agent_logos/agent_5_abcd1234.png")
    const { container } = render(<AgentBuilder agentId={AGENT_ID} />)

    await waitFor(() => expect(removeLogoButton()).toBeTruthy())
    expect(updateButton()).toBeDisabled()

    fireEvent.click(removeLogoButton()!)

    // Removing a logo the agent actually had is a real change.
    expect(updateButton()).not.toBeDisabled()
    expect(removeLogoButton()).toBeNull()

    fireEvent.click(updateButton())

    await waitFor(() => {
      const putCall = apiRequestMock.mock.calls.find(
        ([, opts]) => (opts as any)?.method === "PUT"
      )
      expect(putCall).toBeTruthy()
      const body = JSON.parse((putCall![1] as any).body)
      expect(body.logo_base64).toBe("")
    })

    void container
  })

  it("does not enable Update for upload-then-remove on a logo-less agent", async () => {
    installApi(null)
    const { container } = render(<AgentBuilder agentId={AGENT_ID} />)

    await waitFor(() =>
      expect(screen.getByDisplayValue("Logo Test Agent")).toBeInTheDocument()
    )
    expect(updateButton()).toBeDisabled()

    fireEvent.change(logoFileInput(container), { target: { files: [pngFile("logo.png")] } })
    await waitFor(() => expect(removeLogoButton()).toBeTruthy())
    expect(updateButton()).not.toBeDisabled()

    fireEvent.click(removeLogoButton()!)

    // Net effect is a no-op: this agent never had a logo to begin with.
    expect(updateButton()).toBeDisabled()
  })
})

import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

// Regression coverage for handleShareConnectorsAndContinue's KB-promotion 202
// handling: a 202 body that fails isBackgroundJobResponse used to fall through
// silently into the agent-promotion retry, which could re-promote the agent
// while the KB was still personal. The fix throws instead. This test proves
// that (a) the user sees an error toast and (b) the agent promote-team
// endpoint is never called a second time (only the original 422 attempt that
// opened the share dialog).

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())

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

// inTeam: true is required for the ownership control to render at all.
vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({
    token: "token",
    user: { id: "1", is_admin: false },
    inTeam: true,
    teamRole: "member",
  }),
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

// Mocking @/components/ui/sonner directly (rather than the underlying "sonner"
// package) sidesteps the { duration } second arg toast.error appends, so
// assertions below can check call[0] without worrying about it.
vi.mock("@/components/ui/sonner", () => ({
  toast: { error: toastErrorMock, success: vi.fn() },
}))

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
  ConnectMcpDialog: ({ open }: { open: boolean }) => (
    <output data-testid="connect-mcp-dialog">{String(open)}</output>
  ),
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
vi.mock("@/components/ui/multi-select", () => ({ MultiSelect: () => null }))

// Radix selects aren't drivable with fireEvent in jsdom, so render the
// ownership control as a native <select> instead. SelectTrigger/SelectValue/
// SelectContent just need to pass their children through so SelectItem's
// <option> nodes land inside the native <select>.
vi.mock("@/components/ui/select", () => ({
  // The custom dropdown (model pickers etc.) is unrelated to this flow; a
  // no-op stub matches the admin-mcp sibling test's approach.
  Select: () => null,
  SelectRadix: ({
    value,
    onValueChange,
    children,
  }: {
    value?: string
    onValueChange: (value: string) => void
    children: React.ReactNode
  }) => (
    <select value={value} onChange={(e) => onValueChange(e.target.value)}>
      {children}
    </select>
  ),
  SelectTrigger: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  SelectValue: () => null,
  SelectContent: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  SelectItem: ({ value, children }: { value: string; children: React.ReactNode }) => (
    <option value={value}>{children}</option>
  ),
}))

// The real Radix Dialog is awkward to drive in jsdom; render children
// unconditionally-gated on `open` like sibling tests do (e.g.
// connect-mcp-dialog.test.tsx).
vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ open, children }: { open: boolean; children: React.ReactNode }) =>
    open ? <div>{children}</div> : null,
  DialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogTitle: ({ children }: { children: React.ReactNode }) => <h1>{children}</h1>,
  DialogDescription: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogFooter: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

vi.mock("@/components/build/build-file-preview-sheet", () => ({
  BuildFilePreviewSheet: () => null,
}))

// Unrelated to the KB-share flow. It doesn't import React and this project's
// vitest config has no jsx-automatic-runtime plugin, so rendering it for real
// with inTeam: true (required below) throws "React is not defined" -- a
// pre-existing gap unrelated to the fix under test. Stub it out like the
// other sibling panels above.
vi.mock("@/components/build/agent-ssh-bindings", () => ({
  AgentSshBindings: () => null,
}))

import { AgentBuilder } from "./agent-builder"

const AGENT_ID = "5"

function agentResponse() {
  return {
    id: Number(AGENT_ID),
    user_id: 1,
    name: "Some Agent",
    description: "",
    instructions: "Do the thing",
    execution_mode: "balanced",
    // handleCreate's save validation requires a general model to be set.
    models: { general: 1, small_fast: null, visual: null, compact: null },
    knowledge_bases: [],
    skills: [],
    tool_categories: [],
    suggested_prompts: [],
    logo_url: null,
    status: "draft",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    widget_enabled: false,
    allowed_domains: [],
    share_enabled: false,
    share_updated_at: null,
    team_id: null,
    readonly: false,
    can_edit: true,
  }
}

// Routes every apiRequest call by URL. /api/agents/5 is used for both the
// mount-time GET and the save PUT, and both return the same agent shape, so
// the request method doesn't need to be inspected here.
function installApi() {
  apiRequestMock.mockImplementation((url: string) => {
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
    if (url.includes("/api/mcp/servers"))
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }))

    // Original 422 that opens the share-connectors dialog. The knowledge-base
    // entry must satisfy sanitizeUnsharedKnowledgeBases (a non-empty `name`
    // string) or the dialog never opens.
    if (url.endsWith(`/api/agents/${AGENT_ID}/promote-team`))
      return Promise.resolve(
        new Response(
          JSON.stringify({ detail: { unshared_knowledge_bases: [{ name: "demo" }] } }),
          { status: 422 },
        ),
      )

    // The bug under test: 202 with a body that fails isBackgroundJobResponse.
    if (url.endsWith("/api/knowledge-bases/demo/promote-team"))
      return Promise.resolve(
        new Response(JSON.stringify({ not_a_valid_job: true }), { status: 202 }),
      )

    if (url.endsWith(`/api/agents/${AGENT_ID}`))
      return Promise.resolve(new Response(JSON.stringify(agentResponse()), { status: 200 }))

    return Promise.resolve(new Response(JSON.stringify({}), { status: 200 }))
  })
}

const promoteTeamCalls = () =>
  apiRequestMock.mock.calls
    .map(([u]) => String(u))
    .filter((u) => u === "http://api.local/api/agents/5/promote-team")

describe("AgentBuilder KB-promotion 202 handling", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
    installApi()
    ;(globalThis as unknown as { WebSocket: unknown }).WebSocket = vi.fn()
  })

  afterEach(() => cleanup())

  it("surfaces an error and does not re-attempt agent promotion when the KB 202 body is not a valid job", async () => {
    render(<AgentBuilder agentId={AGENT_ID} />)

    // Wait for the agent load to finish before touching the ownership select,
    // otherwise loadAgent's setOwnership(agent.team_id == null ? ...) would
    // stomp our selection once the fetch resolves.
    await waitFor(() =>
      expect(screen.getByPlaceholderText("builds.configForm.name.placeholder")).toHaveValue(
        "Some Agent",
      ),
    )

    // Only the ownership select exists until ownership === "team".
    const ownershipSelect = document.querySelector("select") as HTMLSelectElement
    expect(ownershipSelect).not.toBeNull()
    fireEvent.change(ownershipSelect, { target: { value: "team" } })

    const updateButton = await screen.findByRole("button", {
      name: "builds.editor.header.update",
    })
    fireEvent.click(updateButton)

    // The 422 from the first promote-team attempt opens the share dialog.
    await screen.findByText("builds.configForm.connectorNotShared.title")
    expect(promoteTeamCalls()).toHaveLength(1)

    const shareButton = await screen.findByRole("button", {
      name: "builds.configForm.connectorNotShared.shareAndContinue",
    })
    fireEvent.click(shareButton)

    await waitFor(() => expect(toastErrorMock).toHaveBeenCalled())
    // toast.error's message arg (call[0]) should reflect the unknown-error
    // fallback the fix throws; the second arg is a { duration } object added
    // by the real @/components/ui/sonner wrapper, irrelevant here since it's
    // mocked away.
    expect(toastErrorMock.mock.calls[0][0]).toBe("builds.editor.error.unknown")

    // The unreadable 202 body must abort before reconcileOwnership runs again,
    // so the agent promote-team endpoint is still only called once (the
    // original 422 attempt) -- never a second, silently-continued retry.
    expect(promoteTeamCalls()).toHaveLength(1)
  })
})

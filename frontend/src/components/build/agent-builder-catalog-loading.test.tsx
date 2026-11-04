import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

// #3007: the builder used to wait for every catalog request in one Promise.all,
// so a slow or failing /api/tools/available left Model, Skills and Tools all
// rendering "no data". Each list now loads on its own, and only a list that
// actually loaded empty may say so.

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

// Stable across renders, as the real context is: fresh vi.fn()s per render
// would re-run every effect that depends on them.
const appContext = vi.hoisted(() => ({
  value: null as unknown,
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => appContext.value,
}))

function makeAppContext() {
  return {
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
  }
}

const authUser = vi.hoisted(() => ({
  current: { id: "1", is_admin: false } as { id: string; is_admin: boolean },
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ token: "token", user: authUser.current }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ locale: "en", t: (key: string) => key }),
}))

vi.mock("@/contexts/mcp-apps-context", () => ({
  useMcpApps: () => ({ apps: [], getAppIcon: () => null, refresh: () => Promise.resolve() }),
}))

vi.mock("@/lib/branding", () => ({
  getBrandingFromEnv: () => ({ appName: "Xagent" }),
}))

vi.mock("sonner", () => ({ toast: { error: vi.fn(), success: vi.fn() } }))

const search = vi.hoisted(() => ({ template: null as string | null }))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useSearchParams: () => ({
    get: (key: string) => (key === "template" ? search.template : null),
  }),
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
  ConnectMcpDialog: ({ onSuccess }: { onSuccess?: () => void }) => (
    <button type="button" onClick={() => onSuccess?.()}>
      mock-connect-mcp-success
    </button>
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
vi.mock("@/components/ui/multi-select", () => ({
  MultiSelect: (props: any) => (
    <div
      data-testid="multi-select"
      data-placeholder={props.placeholder}
      data-values={(props.values || []).join("|")}
    >
      {(props.options || []).map((o: any) => o.value).join("|")}
    </div>
  ),
}))
vi.mock("@/components/ui/select", () => ({
  Select: ({ value, onValueChange, options }: any) => (
    <select
      data-testid="model-select"
      value={value ?? ""}
      onChange={(e) => onValueChange(e.target.value)}
    >
      <option value="" />
      {(options || []).map((o: any) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  ),
}))
vi.mock("@/components/build/build-file-preview-sheet", () => ({
  BuildFilePreviewSheet: () => null,
}))

import { AgentBuilder } from "./agent-builder"

const AGENT_ID = "5"
const MODEL_ID = 42
const TEMPLATE_ID = "tpl-1"

type Route =
  | "kbs"
  | "skills"
  | "tools"
  | "models"
  | "defaults"
  | "mcp"
  | "ownerMcp"
  | "agent"
  | "template"
  | "previewTask"

type Handler = () => Promise<Response>

const json = (body: unknown, status = 200): Handler => () =>
  Promise.resolve(new Response(JSON.stringify(body), { status }))

// A request that resolves only when the test says so.
function gate(body: unknown, status = 200) {
  let release = () => {}
  const handler: Handler = () =>
    new Promise<Response>((resolve) => {
      release = () => resolve(new Response(JSON.stringify(body), { status }))
    })
  return { handler, release: () => release() }
}

const hang: Handler = () => new Promise<Response>(() => {})

const webSearchTool = {
  name: "web_search_tool",
  description: "",
  type: "builtin",
  category: "web_search",
  enabled: true,
  always_available: false,
}

function agentResponse(overrides: Record<string, unknown> = {}) {
  return {
    id: Number(AGENT_ID),
    user_id: 1,
    team_id: null,
    name: "Catalog Agent",
    description: "",
    instructions: "You are a test agent.",
    execution_mode: "balanced",
    models: null,
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
    can_edit: true,
    ...overrides,
  }
}

function templateResponse(overrides: Record<string, unknown> = {}) {
  return {
    id: TEMPLATE_ID,
    type: "agent",
    name: "Template Agent",
    description: "From a template",
    agent_config: { instructions: "Do it.", skills: [], tool_categories: [] },
    connections: [],
    ...overrides,
  }
}

function installApi(overrides: Partial<Record<Route, Handler>> = {}) {
  const routes: Record<Route, Handler> = {
    kbs: json({ collections: [] }),
    skills: json([{ name: "skill-a" }]),
    tools: json({ tools: [webSearchTool] }),
    models: json([{ id: MODEL_ID, model_id: "m", model_name: "llm-a", model_provider: "p", category: "llm" }]),
    defaults: json([{ config_type: "general", model: { id: MODEL_ID } }]),
    mcp: json([]),
    ownerMcp: json([]),
    agent: json(agentResponse()),
    template: json(templateResponse()),
    previewTask: json({}),
    ...overrides,
  }
  apiRequestMock.mockImplementation((url: string) => {
    if (url.endsWith("/api/kb/collections")) return routes.kbs()
    if (url.endsWith("/api/skills/")) return routes.skills()
    if (url.endsWith("/api/tools/available?include_usage=false")) return routes.tools()
    if (url.endsWith("/api/models/?category=llm")) return routes.models()
    if (url.endsWith("/api/models/user-default")) return routes.defaults()
    if (url.includes("/api/mcp/servers?user_id=")) return routes.ownerMcp()
    if (url.endsWith("/api/mcp/servers")) return routes.mcp()
    if (url.endsWith(`/api/agents/${AGENT_ID}/preview-task`)) return routes.previewTask()
    if (url.includes(`/api/agents/${AGENT_ID}/triggers`)) return json([])()
    if (url.endsWith(`/api/agents/${AGENT_ID}`)) return routes.agent()
    if (url.endsWith(`/api/templates/${TEMPLATE_ID}`)) return routes.template()
    return json({})()
  })
}

// The config section that wraps a picker's label.
const section = (labelKey: string) =>
  screen
    .getAllByText(labelKey)
    .map((el) => el.closest<HTMLElement>(".space-y-2"))
    .find((el): el is HTMLElement => el !== null)!
const toolsSection = () => section("builds.configForm.tools.label")
const skillsSection = () => section("builds.configForm.skills.label")
const kbSection = () => section("builds.configForm.knowledgeBase.label")
const mcpSection = () => section("tools.mcp.dialog.connector")
const modelSelect = () => screen.getAllByTestId("model-select")[0] as HTMLSelectElement
const configStepComplete = () =>
  screen.getByRole("button", { name: /builds.editor.stepGuide.configure/ }).querySelector("svg") !== null
const isHighlighted = (el: HTMLElement) => el.className.includes("border-primary/30")
const calls = (fragment: string) =>
  apiRequestMock.mock.calls.filter(([u]) => String(u).endsWith(fragment)).length

beforeEach(() => {
  apiRequestMock.mockReset()
  authUser.current = { id: "1", is_admin: false }
  search.template = null
  appContext.value = makeAppContext()
  ;(globalThis as any).WebSocket = vi.fn()
})

afterEach(() => cleanup())

describe("AgentBuilder catalog lists load independently", () => {
  it("renders models and skills while the tools request is still pending", async () => {
    installApi({ tools: hang })
    render(<AgentBuilder />)

    await waitFor(() => expect(modelSelect().value).toBe(String(MODEL_ID)))
    expect(within(skillsSection()).getByTestId("multi-select")).toHaveTextContent("skill-a")

    const tools = toolsSection()
    expect(within(tools).getByRole("status")).toHaveTextContent("common.loading")
    expect(within(tools).queryByText("builds.configForm.tools.noData")).toBeNull()
  })

  it("shows an error with retry for a failing tools request, then recovers", async () => {
    let attempts = 0
    installApi({
      tools: () => {
        attempts += 1
        return attempts === 1
          ? json({ detail: "boom" }, 500)()
          : json({ tools: [webSearchTool] })()
      },
    })
    render(<AgentBuilder />)

    await waitFor(() =>
      expect(within(toolsSection()).getByText("builds.configForm.listLoadFailed")).toBeInTheDocument()
    )
    expect(within(toolsSection()).queryByText("builds.configForm.tools.noData")).toBeNull()
    expect(modelSelect().value).toBe(String(MODEL_ID))
    expect(within(skillsSection()).getByTestId("multi-select")).toHaveTextContent("skill-a")

    fireEvent.click(within(toolsSection()).getByText("common.retry"))

    await waitFor(() =>
      expect(within(toolsSection()).getByTestId("multi-select")).toHaveTextContent("web_search")
    )
    expect(calls("/api/tools/available?include_usage=false")).toBe(2)
    expect(within(toolsSection()).queryByText("builds.configForm.listLoadFailed")).toBeNull()
  })

  it("treats a network failure like any other failed request", async () => {
    installApi({ skills: () => Promise.reject(new TypeError("Failed to fetch")) })
    render(<AgentBuilder />)

    await waitFor(() =>
      expect(within(skillsSection()).getByText("builds.configForm.listLoadFailed")).toBeInTheDocument()
    )
    expect(within(skillsSection()).queryByText("builds.configForm.skills.noData")).toBeNull()
  })

  it("treats a malformed 200 body as a failure, not an empty list", async () => {
    installApi({ tools: json({}) })
    render(<AgentBuilder />)

    await waitFor(() =>
      expect(within(toolsSection()).getByText("builds.configForm.listLoadFailed")).toBeInTheDocument()
    )
    expect(within(toolsSection()).queryByText("builds.configForm.tools.noData")).toBeNull()
  })

  it("treats a KB listing that reports status error as a failure", async () => {
    // The KB endpoint answers a failed listing with 200 and status "error".
    installApi({ kbs: json({ status: "error", collections: [], total_count: 0 }) })
    render(<AgentBuilder />)

    await waitFor(() =>
      expect(within(kbSection()).getByText("builds.configForm.listLoadFailed")).toBeInTheDocument()
    )
  })

  it("only says noData once a list has loaded empty", async () => {
    installApi({ tools: json({ tools: [] }), skills: json([]) })
    render(<AgentBuilder />)

    await waitFor(() =>
      expect(within(toolsSection()).getByText("builds.configForm.tools.noData")).toBeInTheDocument()
    )
    expect(within(skillsSection()).getByText("builds.configForm.skills.noData")).toBeInTheDocument()
  })

  it("shows the knowledge-base picker's loading and error states", async () => {
    const kbs = gate({ detail: "boom" }, 500)
    installApi({ kbs: kbs.handler })
    render(<AgentBuilder />)

    await waitFor(() => expect(modelSelect().value).toBe(String(MODEL_ID)))
    expect(within(kbSection()).getByRole("status")).toHaveTextContent("common.loading")

    kbs.release()
    await waitFor(() =>
      expect(within(kbSection()).getByText("builds.configForm.listLoadFailed")).toBeInTheDocument()
    )
  })
})

describe("AgentBuilder tools summary while tools load", () => {
  it("hides the selected-tools count until the tools list has loaded", async () => {
    const tools = gate({ tools: [webSearchTool] })
    installApi({ tools: tools.handler, agent: json(agentResponse({ tool_categories: ["web_search"] })) })
    render(<AgentBuilder agentId={AGENT_ID} />)

    await waitFor(() => expect(screen.getByDisplayValue("Catalog Agent")).toBeInTheDocument())
    expect(within(toolsSection()).getByRole("status")).toHaveTextContent("common.loading")
    expect(within(toolsSection()).queryByText("builds.configForm.tools.selectedCount")).toBeNull()

    tools.release()
    await waitFor(() =>
      expect(within(toolsSection()).getByText("builds.configForm.tools.selectedCount")).toBeInTheDocument()
    )
  })
})

describe("AgentBuilder new-agent default models", () => {
  it("does not overwrite a model picked before the user default lands", async () => {
    const defaults = gate([
      { config_type: "general", model: { id: MODEL_ID } },
      { config_type: "small_fast", model: { id: MODEL_ID } },
    ])
    installApi({
      defaults: defaults.handler,
      models: json([
        { id: MODEL_ID, model_id: "m", model_name: "llm-a", model_provider: "p", category: "llm" },
        { id: 7, model_id: "n", model_name: "llm-b", model_provider: "p", category: "llm" },
      ]),
    })
    render(<AgentBuilder />)

    await waitFor(() => expect(screen.getByTestId("model-select")).toBeInTheDocument())
    fireEvent.change(modelSelect(), { target: { value: "7" } })
    expect(modelSelect().value).toBe("7")

    defaults.release()
    // The seed did run: it filled the slot the user left unset.
    fireEvent.click(screen.getByText("builds.configForm.model.configure"))
    await waitFor(() =>
      expect(screen.getAllByTestId("model-select")[1]).toHaveValue(String(MODEL_ID))
    )
    expect(modelSelect().value).toBe("7")
  })
})

describe("AgentBuilder edit mode with the KB list loading", () => {
  it("keeps the agent's saved knowledge bases visible", async () => {
    installApi({ kbs: hang, agent: json(agentResponse({ knowledge_bases: ["kb-1"] })) })
    render(<AgentBuilder agentId={AGENT_ID} />)

    await waitFor(() =>
      expect(within(kbSection()).getByTestId("multi-select")).toHaveAttribute("data-values", "kb-1")
    )
    expect(within(kbSection()).getByRole("status")).toHaveTextContent("common.loading")
  })
})

describe("AgentBuilder template entry with lists still loading", () => {
  beforeEach(() => {
    search.template = TEMPLATE_ID
  })

  it("does not hold the configure step for a pending tools list", async () => {
    installApi({
      tools: hang,
      template: json(templateResponse({
        agent_config: { instructions: "Do it.", skills: [], tool_categories: ["web_search"] },
      })),
    })
    render(<AgentBuilder />)

    await waitFor(() => expect(configStepComplete()).toBe(true))
    expect(isHighlighted(toolsSection())).toBe(false)
  })

  it("does not report a required connector missing until the MCP list lands", async () => {
    const mcp = gate([{ name: "foo", description: "Foo server" }])
    installApi({ mcp: mcp.handler, template: json(templateResponse({ connections: ["foo"] })) })
    render(<AgentBuilder />)

    await waitFor(() => expect(screen.getByDisplayValue("Template Agent")).toBeInTheDocument())
    await waitFor(() => expect(modelSelect().value).toBe(String(MODEL_ID)))
    expect(isHighlighted(mcpSection())).toBe(false)
    expect(configStepComplete()).toBe(false)

    mcp.release()

    await waitFor(() => expect(configStepComplete()).toBe(true))
    expect(isHighlighted(mcpSection())).toBe(false)
  })

  it("holds the configure step for a required connector that needs a reconnect (#2931)", async () => {
    installApi({
      mcp: json([{ name: "foo", description: "Foo", connection_status: "needs_reconnect" }]),
      template: json(templateResponse({ connections: ["foo"] })),
    })
    render(<AgentBuilder />)

    await waitFor(() => expect(screen.getByText("tools.mcp.needsReconnect")).toBeInTheDocument())
    await waitFor(() => expect(modelSelect().value).toBe(String(MODEL_ID)))
    expect(isHighlighted(mcpSection())).toBe(true)
    expect(configStepComplete()).toBe(false)
  })

  it("completes the configure step for a required connector that is connected", async () => {
    installApi({
      mcp: json([{ name: "foo", description: "Foo", connection_status: "connected" }]),
      template: json(templateResponse({ connections: ["foo"] })),
    })
    render(<AgentBuilder />)

    await waitFor(() => expect(configStepComplete()).toBe(true))
    expect(screen.queryByText("tools.mcp.needsReconnect")).toBeNull()
    expect(isHighlighted(mcpSection())).toBe(false)
  })
})

describe("AgentBuilder edit-mode seed with independent lists", () => {
  it("seeds once the model list lands", async () => {
    const models = gate([{ id: MODEL_ID, model_id: "m", model_name: "llm-a", model_provider: "p", category: "llm" }])
    installApi({ models: models.handler, tools: hang })
    render(<AgentBuilder agentId={AGENT_ID} />)

    await waitFor(() => expect(screen.getByDisplayValue("Catalog Agent")).toBeInTheDocument())
    expect(screen.queryByTestId("model-select")).toBeNull()

    models.release()
    await waitFor(() => expect(modelSelect().value).toBe(String(MODEL_ID)))
  })

  it("seeds once the user default lands", async () => {
    const defaults = gate([{ config_type: "general", model: { id: MODEL_ID } }])
    installApi({ defaults: defaults.handler, tools: hang })
    render(<AgentBuilder agentId={AGENT_ID} />)

    await waitFor(() => expect(screen.getByDisplayValue("Catalog Agent")).toBeInTheDocument())
    await waitFor(() => expect(screen.getByTestId("model-select")).toBeInTheDocument())
    expect(modelSelect().value).toBe("")

    defaults.release()
    await waitFor(() => expect(modelSelect().value).toBe(String(MODEL_ID)))
  })
})

describe("AgentBuilder preview history with independent lists", () => {
  it("loads preview history without waiting for tools", async () => {
    installApi({ tools: hang })
    render(<AgentBuilder agentId={AGENT_ID} />)

    await waitFor(() => expect(calls(`/api/agents/${AGENT_ID}/preview-task`)).toBe(1))
  })

  it("waits for the MCP list, which feeds the preview config key", async () => {
    const mcp = gate([])
    installApi({ mcp: mcp.handler })
    render(<AgentBuilder agentId={AGENT_ID} />)

    await waitFor(() => expect(screen.getByDisplayValue("Catalog Agent")).toBeInTheDocument())
    await waitFor(() => expect(modelSelect().value).toBe(String(MODEL_ID)))
    expect(calls(`/api/agents/${AGENT_ID}/preview-task`)).toBe(0)

    mcp.release()
    await waitFor(() => expect(calls(`/api/agents/${AGENT_ID}/preview-task`)).toBe(1))
  })
})

describe("AgentBuilder owner-scoped MCP list", () => {
  it("is not clobbered by a mount-time MCP response that lands later", async () => {
    authUser.current = { id: "1", is_admin: true }
    const mount = gate([{ name: "foo", description: "admin's own foo" }])
    installApi({
      mcp: mount.handler,
      ownerMcp: json([{ name: "foo", description: "owner's foo" }]),
      agent: json(agentResponse({ user_id: 99, tool_categories: ["mcp:foo"] })),
    })
    render(<AgentBuilder agentId={AGENT_ID} />)

    await waitFor(() => expect(screen.getByText("owner's foo")).toBeInTheDocument())

    mount.release()
    // Preview history waits for the mount MCP fetch to settle, which happens
    // after that fetch would have written its list: a positive signal.
    await waitFor(() => expect(calls(`/api/agents/${AGENT_ID}/preview-task`)).toBe(1))
    expect(screen.getByText("owner's foo")).toBeInTheDocument()
    expect(screen.queryByText("admin's own foo")).toBeNull()
  })

  it("is not clobbered when the owner list is claimed while the mount response parses", async () => {
    authUser.current = { id: "1", is_admin: true }
    const agent = gate(agentResponse({ user_id: 99, tool_categories: ["mcp:foo"] }))
    let releaseBody = () => {}
    let parsing = false
    installApi({
      // Headers arrive at once; the body parse is held open.
      mcp: () =>
        Promise.resolve({
          ok: true,
          status: 200,
          json: () => {
            parsing = true
            return new Promise((resolve) => {
              releaseBody = () => resolve([{ name: "foo", description: "admin's own foo" }])
            })
          },
        } as unknown as Response),
      ownerMcp: json([{ name: "foo", description: "owner's foo" }]),
      agent: agent.handler,
    })
    render(<AgentBuilder agentId={AGENT_ID} />)

    await waitFor(() => expect(parsing).toBe(true))
    agent.release()
    await waitFor(() => expect(screen.getByText("owner's foo")).toBeInTheDocument())

    expect(calls(`/api/agents/${AGENT_ID}/preview-task`)).toBe(0)
    releaseBody()
    // Same positive signal as above: the mount fetch has fully settled.
    await waitFor(() => expect(calls(`/api/agents/${AGENT_ID}/preview-task`)).toBe(1))
    expect(screen.getByText("owner's foo")).toBeInTheDocument()
    expect(screen.queryByText("admin's own foo")).toBeNull()
  })
})

describe("AgentBuilder MCP refresh after connecting", () => {
  it("keeps the loaded list when the refresh request fails", async () => {
    let mcpCalls = 0
    installApi({
      mcp: () => {
        mcpCalls += 1
        // The failing refresh's body resolves at once, so act() below flushes
        // everything a regression could write.
        return mcpCalls === 1
          ? json([{ name: "foo", description: "Foo server" }])()
          : Promise.resolve({
              ok: false,
              status: 500,
              json: () => Promise.resolve({ detail: "boom" }),
            } as unknown as Response)
      },
      agent: json(agentResponse({ tool_categories: ["mcp:foo"] })),
    })
    render(<AgentBuilder agentId={AGENT_ID} />)

    await waitFor(() => expect(screen.getByText("Foo server")).toBeInTheDocument())

    await act(async () => {
      fireEvent.click(screen.getByText("mock-connect-mcp-success"))
    })
    expect(mcpCalls).toBe(2)
    await act(async () => {})

    expect(screen.getByText("Foo server")).toBeInTheDocument()
  })
})

describe("AgentBuilder MCP lists fetched by the agent load", () => {
  it("ignores a non-list owner-scoped MCP body instead of crashing", async () => {
    authUser.current = { id: "1", is_admin: true }
    let ownerCalls = 0
    installApi({
      ownerMcp: () => {
        ownerCalls += 1
        return json({})()
      },
      agent: json(agentResponse({ user_id: 99, tool_categories: ["mcp:foo"] })),
    })
    render(<AgentBuilder agentId={AGENT_ID} />)

    await waitFor(() => expect(ownerCalls).toBe(1))
    await act(async () => {})

    expect(screen.getByDisplayValue("Catalog Agent")).toBeInTheDocument()
    expect(within(mcpSection()).getByText("tools.mcp.mcpUnavailable")).toBeInTheDocument()
  })
})

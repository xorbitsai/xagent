import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const sendMessageMock = vi.hoisted(() => vi.fn())
const dispatchMock = vi.hoisted(() => vi.fn())
const closeFilePreviewMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const searchParamsMock = vi.hoisted(() => ({ value: new URLSearchParams() }))

interface MockAgent {
  id: number | string
  name: string
  suggested_prompts?: string[]
}

interface MockChatStartScreenProps {
  inputValue?: string
  agents?: MockAgent[]
  selectedAgents?: MockAgent[]
  onAgentClick?: (agent: MockAgent) => void
  onInputChange?: (value: string) => void
  onSend?: (message: string, files: unknown[], config?: unknown) => void
  taskConfig?: unknown
}

const chatStartScreenProps = vi.hoisted(() => ({
  current: null as null | MockChatStartScreenProps,
}))

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", () => ({
  getApiUrl: () => "http://api.local",
}))

vi.mock("@/lib/branding", () => ({
  getBrandingFromEnv: () => ({ appName: "Xagent" }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string | number>) =>
      vars ? `${key}:${JSON.stringify(vars)}` : key,
  }),
}))

vi.mock("next/navigation", () => ({
  useSearchParams: () => searchParamsMock.value,
}))

vi.mock("sonner", () => ({
  toast: { error: toastErrorMock },
}))

vi.mock("@/components/file/file-preview-dialog", () => ({
  FilePreviewDialog: () => null,
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    sendMessage: sendMessageMock,
    state: { isProcessing: false, filePreview: { isOpen: false } },
    dispatch: dispatchMock,
    closeFilePreview: closeFilePreviewMock,
  }),
}))

vi.mock("@/components/chat/ChatStartScreen", () => ({
  ChatStartScreen: (props: MockChatStartScreenProps) => {
    chatStartScreenProps.current = props
    return (
      <div>
        <input data-testid="composer" readOnly value={props.inputValue ?? ""} />
        {(props.agents ?? []).map((agent) => (
          <button key={agent.id} onClick={() => props.onAgentClick?.(agent)}>
            pick-{agent.name}
          </button>
        ))}
        <button onClick={() => props.onSend?.("hello", [], { mode: "balanced" })}>send</button>
      </div>
    )
  },
}))

import TaskHomePage from "./page"

function jsonResponse(data: unknown): Response {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })
}

const VERA = { id: 1, name: "Vera", status: "published", suggested_prompts: ["Research a topic and report back"] }
const KEVIN = { id: 2, name: "Kevin", status: "published", suggested_prompts: [] }
const DRAFT_AGENT = { id: 3, name: "Draft Only", status: "draft", suggested_prompts: ["Should not appear"] }

beforeEach(() => {
  apiRequestMock.mockReset()
  // Picking an agent triggers a second, best-effort GET /api/agents/{id}
  // fetch for its execution config - default it to a benign 404 so tests
  // that don't care about that fetch don't spam console.error.
  apiRequestMock.mockResolvedValue(new Response(null, { status: 404 }))
  sendMessageMock.mockReset()
  dispatchMock.mockReset()
  closeFilePreviewMock.mockReset()
  toastErrorMock.mockReset()
  chatStartScreenProps.current = null
  searchParamsMock.value = new URLSearchParams()
})

afterEach(cleanup)

describe("TaskHomePage agents", () => {
  it("fetches agents once and forwards only the published ones", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse([VERA, KEVIN, DRAFT_AGENT]))

    render(<TaskHomePage />)

    await waitFor(() => {
      expect(chatStartScreenProps.current?.agents).toEqual([VERA, KEVIN])
    })
    expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/agents")
  })

  it("selects the agent named in a ?agent= deep link once it's loaded, and also auto-fills its prompt", async () => {
    searchParamsMock.value = new URLSearchParams({ agent: String(VERA.id) })
    apiRequestMock.mockResolvedValueOnce(jsonResponse([VERA, KEVIN]))

    render(<TaskHomePage />)

    await waitFor(() => {
      expect(chatStartScreenProps.current?.selectedAgents).toEqual([VERA])
    })
    expect(screen.getByTestId("composer")).toHaveValue("Research a topic and report back")
  })

  it("ignores a ?agent= deep link that doesn't match any published agent", async () => {
    searchParamsMock.value = new URLSearchParams({ agent: "does-not-exist" })
    apiRequestMock.mockResolvedValueOnce(jsonResponse([VERA]))

    render(<TaskHomePage />)
    await screen.findByText("pick-Vera")

    expect(chatStartScreenProps.current?.selectedAgents).toEqual([])
  })

  it("picking a teammate with a suggested prompt fills the composer and selects them", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse([VERA]))
    render(<TaskHomePage />)
    await screen.findByText("pick-Vera")

    fireEvent.click(screen.getByText("pick-Vera"))

    expect(screen.getByTestId("composer")).toHaveValue("Research a topic and report back")
    expect(chatStartScreenProps.current?.selectedAgents).toEqual([VERA])
  })

  it("picking the already-selected teammate again deselects them and clears the auto-filled prompt", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse([VERA]))
    render(<TaskHomePage />)
    await screen.findByText("pick-Vera")

    fireEvent.click(screen.getByText("pick-Vera"))
    expect(screen.getByTestId("composer")).toHaveValue("Research a topic and report back")

    fireEvent.click(screen.getByText("pick-Vera"))

    expect(screen.getByTestId("composer")).toHaveValue("")
    expect(chatStartScreenProps.current?.selectedAgents).toEqual([])
  })

  it("does not clobber a prompt the user has since edited when deselecting", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse([VERA]))
    render(<TaskHomePage />)
    await screen.findByText("pick-Vera")

    fireEvent.click(screen.getByText("pick-Vera"))
    // Simulate the user editing the composer after the auto-fill.
    act(() => {
      chatStartScreenProps.current?.onInputChange?.("Research a topic and report back on competitors")
    })

    fireEvent.click(screen.getByText("pick-Vera"))

    expect(screen.getByTestId("composer")).toHaveValue(
      "Research a topic and report back on competitors"
    )
  })

  it("does not overwrite a task the user already typed by picking a first teammate with a suggested prompt", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse([VERA]))
    render(<TaskHomePage />)
    await screen.findByText("pick-Vera")

    act(() => {
      chatStartScreenProps.current?.onInputChange?.("Build me a report on our top customers")
    })
    fireEvent.click(screen.getByText("pick-Vera"))

    expect(screen.getByTestId("composer")).toHaveValue("Build me a report on our top customers")
    expect(chatStartScreenProps.current?.selectedAgents).toEqual([VERA])
  })

  it("does not resurrect a teammate's prompt after the user manually clears the composer back to empty", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse([VERA]))
    render(<TaskHomePage />)
    await screen.findByText("pick-Vera")

    // Typed custom text (never auto-filled), then picked a teammate - the
    // composer is correctly left untouched, same as the test above.
    act(() => {
      chatStartScreenProps.current?.onInputChange?.("my own task")
    })
    fireEvent.click(screen.getByText("pick-Vera"))
    expect(screen.getByTestId("composer")).toHaveValue("my own task")

    // The user then deliberately clears it themselves.
    act(() => {
      chatStartScreenProps.current?.onInputChange?.("")
    })

    // A composer that's merely empty must not read as "never touched" and
    // trigger the retroactive-fill effect to silently reinsert Vera's
    // prompt - the user asked for it to be empty.
    expect(screen.getByTestId("composer")).toHaveValue("")
  })

  it("switching from one teammate to another replaces the selection and the filled prompt", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse([VERA, { ...KEVIN, suggested_prompts: ["Turn my meetings into next steps"] }]))
    render(<TaskHomePage />)
    await screen.findByText("pick-Vera")

    fireEvent.click(screen.getByText("pick-Vera"))
    fireEvent.click(screen.getByText("pick-Kevin"))

    expect(screen.getByTestId("composer")).toHaveValue("Turn my meetings into next steps")
    expect(chatStartScreenProps.current?.selectedAgents).toEqual([
      { ...KEVIN, suggested_prompts: ["Turn my meetings into next steps"] },
    ])
  })

  it("switching to a teammate with no suggested prompts clears the previous teammate's unedited auto-fill", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse([VERA, KEVIN]))
    render(<TaskHomePage />)
    await screen.findByText("pick-Vera")

    fireEvent.click(screen.getByText("pick-Vera"))
    expect(screen.getByTestId("composer")).toHaveValue("Research a topic and report back")

    fireEvent.click(screen.getByText("pick-Kevin"))

    expect(screen.getByTestId("composer")).toHaveValue("")
    expect(chatStartScreenProps.current?.selectedAgents).toEqual([KEVIN])
  })

  it("selects a teammate with no suggested prompts without touching the composer", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse([KEVIN]))
    render(<TaskHomePage />)
    await screen.findByText("pick-Kevin")

    act(() => {
      chatStartScreenProps.current?.onInputChange?.("Already typed something")
    })
    fireEvent.click(screen.getByText("pick-Kevin"))

    expect(screen.getByTestId("composer")).toHaveValue("Already typed something")
    expect(chatStartScreenProps.current?.selectedAgents).toEqual([KEVIN])
  })

  it("enriches a hired agent with its template's persona photo and category, leaving a custom agent untouched", async () => {
    const HIRED = { ...VERA, template_id: "sales-research-enricher" }
    const CUSTOM = { ...KEVIN, template_id: null }
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/agents") return Promise.resolve(jsonResponse([HIRED, CUSTOM]))
      if (url.startsWith("http://api.local/api/templates/")) {
        return Promise.resolve(
          jsonResponse([
            {
              id: "sales-research-enricher",
              category: "Sales",
              persona: { avatar: "/marketplace/avatars/vera.png" },
            },
          ])
        )
      }
      return Promise.resolve(new Response(null, { status: 404 }))
    })

    render(<TaskHomePage />)

    await waitFor(() => {
      expect(chatStartScreenProps.current?.agents).toEqual([
        {
          ...HIRED,
          persona_avatar: "/marketplace/avatars/vera.png",
          specialty: "templates.categoryTitles.sales",
        },
        CUSTOM,
      ])
    })
  })

  it("skips malformed template entries instead of losing the whole batch's enrichment", async () => {
    const HIRED = { ...VERA, template_id: "sales-research-enricher" }
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/agents") return Promise.resolve(jsonResponse([HIRED]))
      if (url.startsWith("http://api.local/api/templates/")) {
        return Promise.resolve(
          jsonResponse([
            null,
            { category: "Sales" }, // missing id - must not break the valid entry below
            {
              id: "sales-research-enricher",
              category: "Sales",
              persona: { avatar: "/marketplace/avatars/vera.png" },
            },
          ])
        )
      }
      return Promise.resolve(new Response(null, { status: 404 }))
    })

    render(<TaskHomePage />)

    await waitFor(() => {
      expect(chatStartScreenProps.current?.agents).toEqual([
        {
          ...HIRED,
          persona_avatar: "/marketplace/avatars/vera.png",
          specialty: "templates.categoryTitles.sales",
        },
      ])
    })
  })

  it("re-syncs the selection and retroactively fills the prompt once a click-time-unenriched agent's template loads", async () => {
    const HIRED_RAW = { id: 1, name: "Vera", status: "published", suggested_prompts: [], template_id: "sales-research-enricher" }
    let resolveTemplates: (value: Response) => void = () => {}
    const templatesPromise = new Promise<Response>((resolve) => {
      resolveTemplates = resolve
    })
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/agents") return Promise.resolve(jsonResponse([HIRED_RAW]))
      if (url.startsWith("http://api.local/api/templates/")) return templatesPromise
      return Promise.resolve(new Response(null, { status: 404 }))
    })

    render(<TaskHomePage />)
    await screen.findByText("pick-Vera")

    // Clicked before the template lookup resolves - nothing to enrich with
    // yet, so the raw (unenriched) agent is selected and the composer stays
    // empty, same as any agent with no prompt of its own.
    fireEvent.click(screen.getByText("pick-Vera"))
    expect(chatStartScreenProps.current?.selectedAgents).toEqual([HIRED_RAW])
    expect(screen.getByTestId("composer")).toHaveValue("")

    await act(async () => {
      resolveTemplates(
        jsonResponse([
          {
            id: "sales-research-enricher",
            category: "Sales",
            persona: { avatar: "/marketplace/avatars/vera.png" },
            sample_prompts: [{ title: "Research", prompt: "Research a topic and give me an evidence-backed recommendation" }],
          },
        ])
      )
    })

    await waitFor(() => {
      expect(chatStartScreenProps.current?.selectedAgents).toEqual([
        {
          ...HIRED_RAW,
          persona_avatar: "/marketplace/avatars/vera.png",
          specialty: "templates.categoryTitles.sales",
          suggested_prompts: ["Research a topic and give me an evidence-backed recommendation"],
        },
      ])
    })
    expect(screen.getByTestId("composer")).toHaveValue(
      "Research a topic and give me an evidence-backed recommendation"
    )
  })

  it("does not retroactively fill the prompt once the user has started typing their own", async () => {
    const HIRED_RAW = { id: 1, name: "Vera", status: "published", suggested_prompts: [], template_id: "sales-research-enricher" }
    let resolveTemplates: (value: Response) => void = () => {}
    const templatesPromise = new Promise<Response>((resolve) => {
      resolveTemplates = resolve
    })
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/agents") return Promise.resolve(jsonResponse([HIRED_RAW]))
      if (url.startsWith("http://api.local/api/templates/")) return templatesPromise
      return Promise.resolve(new Response(null, { status: 404 }))
    })

    render(<TaskHomePage />)
    await screen.findByText("pick-Vera")

    fireEvent.click(screen.getByText("pick-Vera"))
    act(() => {
      chatStartScreenProps.current?.onInputChange?.("My own task description")
    })

    await act(async () => {
      resolveTemplates(
        jsonResponse([
          {
            id: "sales-research-enricher",
            category: "Sales",
            sample_prompts: [{ title: "Research", prompt: "Research a topic and give me an evidence-backed recommendation" }],
          },
        ])
      )
    })

    expect(screen.getByTestId("composer")).toHaveValue("My own task description")
  })

  it("clears the previous agent's execution config synchronously on switch, so it can't leak into the new agent's send", async () => {
    const AGENT_A = { id: 1, name: "Vera", status: "published", suggested_prompts: [] }
    const AGENT_B = { id: 2, name: "Kevin", status: "published", suggested_prompts: [] }
    let resolveAConfig: (value: Response) => void = () => {}
    const aConfigPromise = new Promise<Response>((resolve) => {
      resolveAConfig = resolve
    })
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/agents") return Promise.resolve(jsonResponse([AGENT_A, AGENT_B]))
      if (url === "http://api.local/api/agents/1") return aConfigPromise
      return Promise.resolve(new Response(null, { status: 404 }))
    })

    render(<TaskHomePage />)
    await screen.findByText("pick-Vera")

    fireEvent.click(screen.getByText("pick-Vera"))
    await act(async () => {
      resolveAConfig(jsonResponse({ models: { general: "gpt-x" }, execution_mode: "think" }))
    })
    await waitFor(() => {
      expect(chatStartScreenProps.current?.taskConfig).toEqual({ model: "gpt-x", executionMode: "think" })
    })

    fireEvent.click(screen.getByText("pick-Kevin"))

    // Kevin's own config fetch (a 404 in this test) is still in flight -
    // Vera's stale config must already be gone, not still sitting there
    // ready to be sent under Kevin's id.
    expect(chatStartScreenProps.current?.taskConfig).toBeUndefined()
  })

  it("prefers a hired agent's template sample prompt over its own (usually empty) suggested_prompts", async () => {
    const HIRED = { id: 1, name: "Vera", status: "published", suggested_prompts: [], template_id: "sales-research-enricher" }
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/agents") return Promise.resolve(jsonResponse([HIRED]))
      if (url.startsWith("http://api.local/api/templates/")) {
        return Promise.resolve(
          jsonResponse([
            {
              id: "sales-research-enricher",
              category: "Sales",
              persona: { avatar: "/marketplace/avatars/vera.png" },
              sample_prompts: [{ title: "Research", prompt: "Research a topic and give me an evidence-backed recommendation" }],
            },
          ])
        )
      }
      return Promise.resolve(new Response(null, { status: 404 }))
    })

    render(<TaskHomePage />)
    await screen.findByText("pick-Vera")

    fireEvent.click(screen.getByText("pick-Vera"))

    expect(screen.getByTestId("composer")).toHaveValue(
      "Research a topic and give me an evidence-backed recommendation"
    )
  })

  it("keeps a hired agent's own customized suggested_prompts over its template's generic sample prompt", async () => {
    // /build's Suggested Prompts editor lets a user customize a hired
    // agent's prompts after the fact - that edit must win here, not get
    // silently replaced by the template's original marketplace sample.
    const HIRED = {
      id: 1,
      name: "Vera",
      status: "published",
      suggested_prompts: ["My customized starting prompt"],
      template_id: "sales-research-enricher",
    }
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/agents") return Promise.resolve(jsonResponse([HIRED]))
      if (url.startsWith("http://api.local/api/templates/")) {
        return Promise.resolve(
          jsonResponse([
            {
              id: "sales-research-enricher",
              category: "Sales",
              sample_prompts: [{ title: "Research", prompt: "The template's generic sample prompt" }],
            },
          ])
        )
      }
      return Promise.resolve(new Response(null, { status: 404 }))
    })

    render(<TaskHomePage />)
    await screen.findByText("pick-Vera")

    fireEvent.click(screen.getByText("pick-Vera"))

    expect(screen.getByTestId("composer")).toHaveValue("My customized starting prompt")
  })
})

describe("TaskHomePage send", () => {
  it("includes the selected agent's id and resets state on a successful send", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse([VERA]))
    sendMessageMock.mockResolvedValueOnce(undefined)
    render(<TaskHomePage />)
    await screen.findByText("pick-Vera")

    fireEvent.click(screen.getByText("pick-Vera"))
    fireEvent.click(screen.getByText("send"))

    await waitFor(() => {
      expect(sendMessageMock).toHaveBeenCalledWith(
        "hello",
        { mode: "balanced", agentId: 1 },
        []
      )
    })
    await waitFor(() => {
      expect(screen.getByTestId("composer")).toHaveValue("")
    })
    expect(chatStartScreenProps.current?.selectedAgents).toEqual([])
  })

  it("toasts and keeps state when sendMessage rejects", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse([]))
    sendMessageMock.mockRejectedValueOnce(new Error("network down"))
    render(<TaskHomePage />)
    await waitFor(() => expect(chatStartScreenProps.current).not.toBeNull())

    fireEvent.click(screen.getByText("send"))

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith("network down")
    })
  })
})

import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return {
    ...actual,
    getApiUrl: () => "http://api.local",
  }
})

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string>) => {
      if (key === "tools.credentials.configure") return "Configure"
      if (key === "tools.credentials.dialog.description" && vars?.tool) {
        return `Configure ${vars.tool}`
      }
      if (key === "tools.credentials.status.team") return "Team credential"
      if (key === "tools.credentials.toolNames.web_search") return "Google Web Search"
      if (key === "tools.credentials.toolNames.zhipu_web_search") return "Zhipu Web Search"
      if (key === "tools.credentials.toolNames.sql_query") return "SQL Query"
      return key
    },
  }),
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: {
    error: vi.fn(),
    success: vi.fn(),
  },
}))

vi.mock("@/components/ui/select", () => ({
  Select: ({
    value,
    options = [],
    onValueChange,
  }: {
    value?: string
    options?: Array<{ value: string; label: string }>
    onValueChange: (value: string) => void
  }) => (
    <select value={value} onChange={(event) => onValueChange(event.target.value)}>
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  ),
}))

import { ToolCredentialsPanel } from "./tool-credentials-panel"

const sqlTool = {
  tool_name: "sql_query",
  display_name: "SQL Query",
  configured: false,
  fields: {},
}

const googleTool = {
  tool_name: "web_search",
  display_name: "Google Search",
  configured: true,
  fields: {
    GOOGLE_API_KEY: {
      label: "API Key",
      required: true,
      secret: true,
      source: "user",
      is_configured: true,
      masked: "********1234",
    },
    GOOGLE_CSE_ID: {
      label: "CSE ID",
      required: true,
      secret: false,
      source: "user",
      is_configured: true,
      masked: "cx-1234",
    },
    FALLBACK_KEY: {
      label: "Fallback Key",
      required: false,
      secret: true,
      source: "env",
      is_configured: true,
      masked: "********env",
    },
  },
}

const sqlToolWithConnections = {
  ...sqlTool,
  configured: true,
  fields: {
    analytics: {
      label: "analytics",
      required: false,
      secret: true,
      source: "user",
      is_configured: true,
      masked: "postgresql://db/analytics",
    },
    warehouse: {
      label: "warehouse",
      required: false,
      secret: true,
      source: "user",
      is_configured: true,
      masked: "mysql://db/warehouse",
    },
    inherited: {
      label: "inherited",
      required: false,
      secret: true,
      source: "shared",
      is_configured: true,
      masked: "postgresql://db/inherited",
    },
  },
}

const sqlToolWithTeamConnection = {
  ...sqlTool,
  configured: true,
  fields: {
    inherited: {
      label: "inherited",
      required: false,
      secret: true,
      source: "team",
      is_configured: true,
      masked: "postgresql://db/inherited",
    },
  },
}

const toolsResponse = (tools: unknown[]) =>
  new Response(JSON.stringify({ tools }), { status: 200 })

describe("ToolCredentialsPanel", () => {
  beforeEach(() => {
    vi.spyOn(window, "confirm").mockReturnValue(true)
    apiRequestMock.mockImplementation(() => Promise.resolve(toolsResponse([sqlTool])))
  })

  it("prefers localized credential tool names over backend display names", async () => {
    apiRequestMock.mockImplementation(() =>
      Promise.resolve(
        toolsResponse([
          {
            tool_name: "zhipu_web_search",
            display_name: "智谱网络搜索",
            configured: false,
            fields: {},
          },
        ]),
      ),
    )

    render(<ToolCredentialsPanel scope="user" />)

    expect(await screen.findByText("Zhipu Web Search")).toBeInTheDocument()
    expect(screen.queryByText("智谱网络搜索")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Configure" }))
    expect(await screen.findByText("Configure Zhipu Web Search")).toBeInTheDocument()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    vi.clearAllMocks()
  })

  it("renders SQL Query as a normal configurable tool card", async () => {
    render(<ToolCredentialsPanel scope="user" />)

    expect(await screen.findByText("SQL Query")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Configure" })).toBeInTheDocument()
    expect(screen.queryByText("tools.database.existingConnections")).not.toBeInTheDocument()
    expect(screen.getByText("tools.credentials.status.none")).toBeInTheDocument()
    expect(screen.queryByText("tools.credentials.configured")).not.toBeInTheDocument()
  })

  it("shows credential source badges on tool cards instead of a generic configured label", async () => {
    apiRequestMock.mockImplementation(() => Promise.resolve(toolsResponse([googleTool])))

    render(<ToolCredentialsPanel scope="user" />)

    expect(await screen.findByText("Google Web Search")).toBeInTheDocument()
    expect(screen.getByText("tools.credentials.status.user")).toBeInTheDocument()
    expect(screen.getByText("tools.credentials.status.env")).toBeInTheDocument()
    expect(screen.queryByText("tools.credentials.configured")).not.toBeInTheDocument()
  })

  it("uses custom endpoints and labels for embedded shared credential scopes", async () => {
    apiRequestMock.mockImplementation(() => Promise.resolve(toolsResponse([sqlToolWithConnections])))

    render(
      <ToolCredentialsPanel
        scope="user"
        credentialScopeKey="shared"
        endpointBase="/api/shared-scopes/7/tool-credentials"
        sourceLabels={{ shared: "Shared credential" }}
        initialToolName="sql_query"
      />,
    )

    await screen.findByText("Configure SQL Query")
    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/shared-scopes/7/tool-credentials",
    )
    expect(screen.getAllByText("Shared credential").length).toBeGreaterThan(0)

    fireEvent.click(screen.getByRole("button", { name: "tools.credentials.deleteAll" }))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/shared-scopes/7/tool-credentials/sql_query/inherited",
        { method: "DELETE" },
      )
    })
  })

  it("uses custom source labels for inherited credentials on personal pages", async () => {
    apiRequestMock.mockImplementation(() =>
      Promise.resolve(toolsResponse([sqlToolWithTeamConnection])),
    )

    render(
      <ToolCredentialsPanel
        scope="user"
        initialToolName="sql_query"
      />,
    )

    await screen.findByText("Configure SQL Query")
    expect(screen.getAllByText("Team credential").length).toBeGreaterThan(0)
    expect(screen.queryByText("tools.credentials.status.db")).not.toBeInTheDocument()
  })

  it("auto-opens SQL Query through the shared credentials dialog", async () => {
    render(<ToolCredentialsPanel scope="user" initialToolName="sql_query" />)

    expect(await screen.findByText("Configure SQL Query")).toBeInTheDocument()
    expect(screen.getByText("tools.database.connectionName")).toBeInTheDocument()
    expect(screen.queryByText("tools.database.existingConnections")).not.toBeInTheDocument()

    const formGrid = screen.getByText("tools.database.connectionName").closest(".grid")
    expect(formGrid).toHaveClass("grid-cols-1")
    expect(formGrid?.className).not.toContain("md:grid-cols-2")
    expect(formGrid?.className).not.toContain("xl:grid-cols-4")
  })

  it("deletes all current-scope credential fields for a provider tool from the dialog", async () => {
    apiRequestMock.mockImplementation(() => Promise.resolve(toolsResponse([googleTool])))

    render(<ToolCredentialsPanel scope="user" />)
    fireEvent.click(await screen.findByRole("button", { name: "Configure" }))
    fireEvent.click(screen.getByRole("button", { name: "tools.credentials.deleteAll" }))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/tool-credentials/web_search/GOOGLE_API_KEY?scope=user",
        { method: "DELETE" },
      )
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/tool-credentials/web_search/GOOGLE_CSE_ID?scope=user",
        { method: "DELETE" },
      )
    })
    expect(apiRequestMock).not.toHaveBeenCalledWith(
      "http://api.local/api/tool-credentials/web_search/FALLBACK_KEY?scope=user",
      { method: "DELETE" },
    )
  })

  it("deletes all current-scope SQL connections from the shared credentials dialog", async () => {
    apiRequestMock.mockImplementation(() =>
      Promise.resolve(toolsResponse([sqlToolWithConnections])),
    )

    render(<ToolCredentialsPanel scope="user" initialToolName="sql_query" />)
    await screen.findByText("Configure SQL Query")
    fireEvent.click(screen.getByRole("button", { name: "tools.credentials.deleteAll" }))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/tool-credentials/sql_query/analytics?scope=user",
        { method: "DELETE" },
      )
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/tool-credentials/sql_query/warehouse?scope=user",
        { method: "DELETE" },
      )
    })
    expect(apiRequestMock).not.toHaveBeenCalledWith(
      "http://api.local/api/tool-credentials/sql_query/inherited?scope=user",
      { method: "DELETE" },
    )
  })
})

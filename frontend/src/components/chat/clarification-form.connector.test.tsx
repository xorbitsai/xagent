/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { resolveTranslation } from "@/i18n/translations"
import type { Interaction } from "@/contexts/app-context-chat"
import { cosmosConnector } from "./connector/fixtures"

const appContextMock = vi.hoisted(() => ({
  dispatch: vi.fn(), filesDisabled: true, sendMessage: vi.fn(),
  state: { currentTask: null as unknown },
}))
const api = vi.hoisted(() => ({ saveConnector: vi.fn(), skipConnector: vi.fn(), getConnectorState: vi.fn(), resumeConnector: vi.fn() }))
vi.mock("@/contexts/app-context-chat", () => ({ useApp: () => appContextMock }))
vi.mock("@/lib/connector-connect", () => api)
vi.mock("@/components/ui/sonner", () => ({ toast: { error: vi.fn() } }))
vi.mock("@/contexts/mcp-apps-context", () => ({ useMcpApps: () => ({ apps: [], refresh: vi.fn() }) }))
vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string | number>) =>
      resolveTranslation("en", key as Parameters<typeof resolveTranslation>[1], vars),
  }),
}))

import { ClarificationForm } from "./clarification-form"

const SECRET = "super-secret-key-value-fdsf"
// The id travels on the interaction, beside `connector` (which has none).
const { interaction_id: _omit, ...connectorWithoutId } = cosmosConnector
const interactions: Interaction[] = [{
  type: "connect_connector", field: "connect_connector", label: "Connect Azure Cosmos DB",
  interaction_id: cosmosConnector.interaction_id, connector: connectorWithoutId,
}]

const result = (state: string, extra = {}) => ({
  state, errorCode: null, errorField: null, secretMasked: "•••• •••• fdsf", scope: "local", agentName: "Cesar",
  tools: [{ name: "q", label: "Query containers" }], taskResumed: true, ...extra,
})
const waiting = (list: Interaction[] = interactions) => ({ status: "waiting_for_user", waitingInteractions: list })

describe("ClarificationForm · connect_connector", () => {
  beforeEach(() => {
    Object.values(api).forEach((m) => m.mockReset())
    appContextMock.sendMessage.mockReset().mockResolvedValue(undefined)
    appContextMock.dispatch.mockReset()
    appContextMock.state.currentTask = waiting()
    api.getConnectorState.mockResolvedValue(result("pending"))
  })
  afterEach(cleanup)

  it("renders the probe and connect cards with no Submit button and no free-text box", () => {
    render(<ClarificationForm interactions={interactions} requestId="req_1" active />)
    expect(screen.getByTestId("connector-probe-card")).toBeInTheDocument()
    expect(screen.getByTestId("connector-connect-card")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /submit/i })).toBeNull()
    expect(screen.getAllByRole("textbox").map((e) => e.getAttribute("name"))).toEqual(["COSMOS_ENDPOINT", "COSMOS_DATABASE"])
  })

  it("takes the interaction id only from the interaction item, not from connector.interaction_id", () => {
    api.skipConnector.mockResolvedValue(result("skipped"))
    render(<ClarificationForm interactions={interactions} requestId="req_1" active />)
    fireEvent.click(screen.getByRole("button", { name: "Not now" }))
    expect(api.skipConnector).toHaveBeenCalledWith(cosmosConnector.interaction_id)
    cleanup()
    render(<ClarificationForm interactions={[{ ...interactions[0], interaction_id: undefined, connector: cosmosConnector }]} active />)
    expect(screen.queryByTestId("connector-connect-card")).toBeNull()
  })

  it("never sends a chat message: not on save, not on skip, not on Continue", async () => {
    api.saveConnector.mockResolvedValue(result("connected", { taskResumed: false }))
    api.resumeConnector.mockResolvedValue(result("connected"))
    const onSend = vi.fn()
    render(<ClarificationForm interactions={interactions} requestId="req_1" active onSend={onSend} />)
    fireEvent.change(screen.getByLabelText("Account endpoint"), { target: { value: "https://acct.example/" } })
    fireEvent.change(screen.getByLabelText("Read-only key"), { target: { value: SECRET } })
    fireEvent.change(screen.getByLabelText("Database"), { target: { value: "customers" } })
    fireEvent.click(screen.getByRole("button", { name: "Save securely" }))
    fireEvent.click(await screen.findByRole("button", { name: "Continue" }))
    await waitFor(() => expect(api.resumeConnector).toHaveBeenCalledWith(cosmosConnector.interaction_id))
    expect(appContextMock.sendMessage).not.toHaveBeenCalled()
    expect(onSend).not.toHaveBeenCalled()
    expect(appContextMock.dispatch).not.toHaveBeenCalled()
    expect(window.location.href + document.body.innerHTML).not.toContain(SECRET)
  })

  it("skips through the host endpoint without a chat message", async () => {
    api.skipConnector.mockResolvedValue(result("skipped"))
    render(<ClarificationForm interactions={interactions} active />)
    fireEvent.click(screen.getByRole("button", { name: "Not now" }))
    await waitFor(() => expect(api.skipConnector).toHaveBeenCalledWith(cosmosConnector.interaction_id))
    expect(appContextMock.sendMessage).not.toHaveBeenCalled()
  })

  it("leaves a historical card read-only: no request, no resume turn", async () => {
    appContextMock.state.currentTask = { status: "running", waitingInteractions: [] }
    render(<ClarificationForm interactions={interactions} requestId="req_old" active={false} />)
    expect(screen.getByRole("button", { name: "Save securely" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Not now" })).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Not now" }))
    await act(async () => {})
    expect(api.skipConnector).not.toHaveBeenCalled()
    expect(appContextMock.sendMessage).not.toHaveBeenCalled()
  })

  it("leaves a card read-only when the task waits on a different interaction", () => {
    appContextMock.state.currentTask = waiting([{ ...interactions[0], interaction_id: "other" }])
    render(<ClarificationForm interactions={interactions} requestId="req_1" active />)
    expect(screen.getByRole("button", { name: "Not now" })).toBeDisabled()
  })

  it("offers Continue for a settled card only while the task waits on this interaction", async () => {
    api.getConnectorState.mockResolvedValue(result("connected", { taskResumed: null }))
    appContextMock.state.currentTask = waiting([{ ...interactions[0], interaction_id: "other" }])
    render(<ClarificationForm interactions={interactions} requestId="req_1" active />)
    await screen.findByText("Connected")
    expect(screen.queryByRole("button", { name: "Continue" })).toBeNull()
    cleanup()
    appContextMock.state.currentTask = waiting()
    render(<ClarificationForm interactions={interactions} requestId="req_1" active />)
    expect(await screen.findByRole("button", { name: "Continue" })).toBeInTheDocument()
    expect(appContextMock.sendMessage).not.toHaveBeenCalled()
  })

  it("shows the probe's final state at once for a card that is not live (reload)", () => {
    appContextMock.state.currentTask = { status: "running" }
    render(<ClarificationForm interactions={interactions} requestId="req_1" active={false} />)
    expect(screen.getByText("Found a way · MCP server")).toBeInTheDocument()
  })

  describe("a batch that mixes connect_connector with other entries", () => {
    const second: Interaction = {
      ...interactions[0], field: "connect_connector_2", interaction_id: "i-2",
      connector: { ...connectorWithoutId, name: "Second App" },
    }
    const question: Interaction = {
      type: "select_one", field: "region", label: "Which region?",
      options: [{ label: "EU", value: "eu" }, { label: "US", value: "us" }],
    }
    const reply: Interaction = { type: "text_input", field: "response", label: "Anything to add?" }

    it("shows the card and the question; Submit answers only the question", async () => {
      const batch = [interactions[0], question]
      appContextMock.state.currentTask = waiting(batch)
      render(<ClarificationForm interactions={batch} requestId="req_1" active />)
      expect(screen.getByTestId("connector-connect-card")).toBeInTheDocument()
      expect(screen.getByText(/Which region\?/)).toBeInTheDocument()
      fireEvent.click(screen.getByRole("radio", { name: "EU" }))
      fireEvent.click(screen.getByRole("button", { name: /submit/i }))
      await waitFor(() => expect(appContextMock.sendMessage).toHaveBeenCalledTimes(1))
      expect(appContextMock.sendMessage.mock.calls[0][0]).toBe("Which region?: EU")
      expect(api.skipConnector).not.toHaveBeenCalled()
    })

    it("shows the card beside the default text box and sends the reply as an ordinary answer", async () => {
      const batch = [interactions[0], reply]
      appContextMock.state.currentTask = waiting(batch)
      render(<ClarificationForm interactions={batch} requestId="req_1" active />)
      expect(screen.getByTestId("connector-connect-card")).toBeInTheDocument()
      const replyBox = screen.getAllByRole("textbox").find((el) => !el.getAttribute("name"))!
      fireEvent.change(replyBox, { target: { value: "hello" } })
      fireEvent.click(screen.getByRole("button", { name: /submit/i }))
      await waitFor(() => expect(appContextMock.sendMessage).toHaveBeenCalledTimes(1))
      expect(appContextMock.sendMessage.mock.calls[0][0]).toBe("Anything to add?: hello")
    })

    it("renders one independent card per connector entry", async () => {
      const batch = [interactions[0], second]
      appContextMock.state.currentTask = waiting(batch)
      api.skipConnector.mockResolvedValue(result("skipped"))
      render(<ClarificationForm interactions={batch} requestId="req_1" active />)
      expect(screen.getAllByTestId("connector-connect-card")).toHaveLength(2)
      expect(screen.queryByRole("button", { name: /submit/i })).toBeNull()
      fireEvent.click(screen.getAllByRole("button", { name: "Not now" })[1])
      await waitFor(() => expect(api.skipConnector).toHaveBeenCalledWith("i-2"))
      expect(api.skipConnector).toHaveBeenCalledTimes(1)
      expect(screen.getAllByRole("button", { name: "Save securely" })).toHaveLength(1)
    })

    it("falls back to a plain reply box for an entry the card cannot show, when nothing else is there", () => {
      const noId: Interaction = { ...interactions[0], interaction_id: undefined }
      render(<ClarificationForm interactions={[noId]} active />)
      expect(screen.queryByTestId("connector-connect-card")).toBeNull()
      expect(screen.getByRole("textbox")).toBeInTheDocument()
      expect(screen.getByRole("button", { name: /submit/i })).toBeInTheDocument()
    })

    it.each([
      ["no fields", { ...interactions[0], connector: { ...connectorWithoutId, fields: [] } }],
      ["a payload that is not an object", { ...interactions[0], connector: undefined }],
    ] as Array<[string, Interaction]>)("falls back for %s, keeping the other entries", (_label, bad) => {
      render(<ClarificationForm interactions={[bad, question]} active />)
      expect(screen.queryByTestId("connector-connect-card")).toBeNull()
      expect(screen.getByText(/Which region\?/)).toBeInTheDocument()
    })
  })
})

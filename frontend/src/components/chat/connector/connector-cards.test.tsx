/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { resolveTranslation } from "@/i18n/translations"
import type { ConnectorConnectResult } from "@/lib/page-extension-contracts"
import { cosmosConnector, cosmosResultCard } from "./fixtures"

const api = vi.hoisted(() => ({ saveConnector: vi.fn(), skipConnector: vi.fn(), getConnectorState: vi.fn(), resumeConnector: vi.fn() }))
vi.mock("@/lib/connector-connect", () => api)
vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string | number>) =>
      resolveTranslation("en", key as Parameters<typeof resolveTranslation>[1], vars),
  }),
}))

import { ConnectorConnectCard } from "./ConnectorConnectCard"
import { ConnectorProbeCard, PROBE_STEP_MS } from "./ConnectorProbeCard"
import { ConnectorResultCard } from "./ConnectorResultCard"

const result = (over: Partial<ConnectorConnectResult> = {}): ConnectorConnectResult => ({
  state: "connected", errorCode: null, errorField: null, secretMasked: "•••• •••• fdsf", scope: "local",
  agentName: "Cesar", tools: [{ name: "query_items", label: "Query containers" }], taskResumed: true, ...over,
})
const SECRET = "super-secret-key-value-fdsf"
const ENDPOINT = "https://acct.documents.azure.com:443/"

function fill() {
  fireEvent.change(screen.getByLabelText("Account endpoint"), { target: { value: ENDPOINT } })
  fireEvent.change(screen.getByLabelText("Read-only key"), { target: { value: SECRET } })
  fireEvent.change(screen.getByLabelText("Database"), { target: { value: "customers" } })
}

describe("ConnectorProbeCard", () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => { cleanup(); vi.useRealTimers() })

  it("steps through the checks every 750 ms and stops on the found route", () => {
    render(<ConnectorProbeCard connector={cosmosConnector} animate />)
    expect(screen.getByText("Checking for a native connector…")).toBeInTheDocument()
    act(() => { vi.advanceTimersByTime(PROBE_STEP_MS) })
    expect(screen.getByText("Checking for an MCP server…")).toBeInTheDocument()
    act(() => { vi.advanceTimersByTime(PROBE_STEP_MS) })
    expect(screen.getByText("Found a way · MCP server")).toBeInTheDocument()
    expect(screen.getByTestId("connector-probe-card")).toHaveAttribute("data-s", "ok")
    act(() => { vi.advanceTimersByTime(5000) })
    expect(screen.getByText("Found a way · MCP server")).toBeInTheDocument()
  })

  it("shows the final state at once on reload, with no replay", () => {
    render(<ConnectorProbeCard connector={cosmosConnector} animate={false} />)
    expect(screen.getByText("Found a way · MCP server")).toBeInTheDocument()
    expect(screen.queryByText(/Checking/)).toBeNull()
  })

  it("ignores inherited property names as checks", () => {
    const connector = { ...cosmosConnector, probe: { checks: ["constructor", "__proto__", "mcp"], found: "mcp" as const } }
    render(<ConnectorProbeCard connector={connector} animate />)
    expect(screen.getByText("Checking for an MCP server…")).toBeInTheDocument()
  })

  it("jumps to the final state when it learns it is a replay", () => {
    const { rerender } = render(<ConnectorProbeCard connector={cosmosConnector} animate />)
    rerender(<ConnectorProbeCard connector={cosmosConnector} animate={false} />)
    expect(screen.getByText("Found a way · MCP server")).toBeInTheDocument()
  })
})

describe("ConnectorConnectCard", () => {
  const props = (over: { pending?: boolean } = {}) => ({ connector: cosmosConnector, pending: true, ...over })
  beforeEach(() => {
    Object.values(api).forEach((m) => m.mockReset())
    api.getConnectorState.mockResolvedValue(result({ state: "pending", taskResumed: null }))
  })
  afterEach(() => { cleanup(); vi.useRealTimers() })

  it("renders the idle card with password-typed secret fields only", () => {
    render(<ConnectorConnectCard {...props()} />)
    expect(screen.getByText("Not connected")).toBeInTheDocument()
    expect(screen.getByText("MCP")).toBeInTheDocument()
    expect(screen.getByLabelText("Read-only key")).toHaveAttribute("type", "password")
    expect(screen.getByLabelText("Read-only key")).toHaveAttribute("autocomplete", "new-password")
    expect(screen.getByLabelText("Account endpoint")).toHaveAttribute("type", "text")
    expect(screen.getByLabelText("Account endpoint")).toHaveAttribute("autocomplete", "off")
    expect(screen.getByRole("button", { name: "Save securely" })).toBeEnabled()
  })

  it("shows a short no-longer-needed state, and no form, while the task is not waiting on it", async () => {
    render(<ConnectorConnectCard {...props({ pending: false })} />)
    expect(screen.getByText("No longer needed")).toBeInTheDocument()
    expect(screen.getByText("This connection is no longer needed.")).toBeInTheDocument()
    expect(screen.queryByLabelText("Read-only key")).toBeNull()
    expect(screen.queryByRole("button", { name: "Not now" })).toBeNull()
    await act(async () => {})
    expect(api.skipConnector).not.toHaveBeenCalled()
  })

  it("shows a field error, without a request, for a whitespace-only value", async () => {
    render(<ConnectorConnectCard {...props()} />)
    fill()
    fireEvent.change(screen.getByLabelText("Database"), { target: { value: "   " } })
    fireEvent.click(screen.getByRole("button", { name: "Save securely" }))
    expect(await screen.findByText("This field is required.")).toBeInTheDocument()
    expect(screen.getByLabelText("Database")).toHaveAttribute("aria-invalid", "true")
    expect(api.saveConnector).not.toHaveBeenCalled()
    fireEvent.change(screen.getByLabelText("Database"), { target: { value: "customers" } })
    expect(screen.queryByText("This field is required.")).toBeNull()
  })

  it("leaves out the how-to heading, list and \"Lets your team use\" line when their text is empty", () => {
    render(<ConnectorConnectCard {...props()} connector={{ ...cosmosConnector, key_label: "", how: [], what: "" }} />)
    expect(screen.queryByText(/How to get/)).toBeNull()
    expect(screen.queryByRole("list")).toBeNull()
    expect(screen.queryByText(/Lets your team use/)).toBeNull()
    expect(screen.getByLabelText("Read-only key")).toBeInTheDocument()
  })

  it("ignores a late restore once a save has started", async () => {
    let restore!: (r: ConnectorConnectResult) => void
    api.getConnectorState.mockReturnValue(new Promise<ConnectorConnectResult>((r) => { restore = r }))
    let finish!: (r: ConnectorConnectResult) => void
    api.saveConnector.mockReturnValue(new Promise<ConnectorConnectResult>((r) => { finish = r }))
    render(<ConnectorConnectCard {...props()} />)
    fill()
    fireEvent.click(screen.getByRole("button", { name: "Save securely" }))
    await act(async () => { restore(result({ state: "unavailable", taskResumed: null })) })
    expect(screen.getAllByText("Saving…").length).toBeGreaterThan(0)
    expect(screen.queryByText("Unavailable")).toBeNull()
    await act(async () => { finish(result({ taskResumed: true })) })
    await act(async () => { restore(result({ taskResumed: null })) })
    expect(screen.getByText("Connected")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Continue" })).toBeNull()
  })

  it("goes saving → connected and clears the inputs; no Continue when the task was resumed", async () => {
    let finish!: (r: ConnectorConnectResult) => void
    api.saveConnector.mockReturnValue(new Promise<ConnectorConnectResult>((r) => { finish = r }))
    render(<ConnectorConnectCard {...props()} />)
    fill()
    fireEvent.click(screen.getByRole("button", { name: "Save securely" }))
    expect(screen.getAllByText("Saving…").length).toBeGreaterThan(0)
    expect(screen.queryByText("Testing…")).toBeNull()
    await act(async () => { finish(result({ taskResumed: true })) })
    expect(screen.getByText("Connected")).toBeInTheDocument()
    expect(screen.getByText("•••• •••• fdsf · stored in your vault")).toBeInTheDocument()
    expect(screen.getByText("Added to Cesar's connected apps · Local")).toBeInTheDocument()
    expect(screen.getByText("Query containers")).toBeInTheDocument()
    expect(screen.queryByLabelText("Read-only key")).toBeNull()
    expect(document.body.innerHTML).not.toContain(SECRET)
    expect(screen.queryByRole("button", { name: "Continue" })).toBeNull()
    expect(api.resumeConnector).not.toHaveBeenCalled()
  })

  it("sends field values only in the save call", async () => {
    api.saveConnector.mockResolvedValue(result())
    render(<ConnectorConnectCard {...props()} />)
    fill()
    fireEvent.click(screen.getByRole("button", { name: "Save securely" }))
    await waitFor(() => expect(screen.getByText("Connected")).toBeInTheDocument())
    expect(api.saveConnector).toHaveBeenCalledWith(cosmosConnector.interaction_id, "cosmos", {
      COSMOS_ENDPOINT: ENDPOINT, COSMOS_KEY: SECRET, COSMOS_DATABASE: "customers",
    })
    expect(JSON.stringify(api.skipConnector.mock.calls)).not.toContain(SECRET)
    expect(JSON.stringify(api.resumeConnector.mock.calls)).not.toContain(SECRET)
  })

  it("fails with red state text, the field error, and a still-editable form", async () => {
    api.saveConnector.mockResolvedValue(result({ state: "failed", errorCode: "invalid_field", errorField: "COSMOS_ENDPOINT", tools: [], taskResumed: null }))
    render(<ConnectorConnectCard {...props()} />)
    fill()
    fireEvent.click(screen.getByRole("button", { name: "Save securely" }))
    const state = await screen.findByText("Couldn't connect")
    expect(state.className).toContain("text-[#C0322E]")
    expect(screen.getByText("This value doesn't look right.")).toBeInTheDocument()
    const endpoint = screen.getByLabelText("Account endpoint")
    expect(endpoint).toHaveAttribute("aria-invalid", "true")
    expect(endpoint).toHaveValue(ENDPOINT)
    expect(endpoint).not.toBeDisabled()
    expect(endpoint.getAttribute("aria-describedby")).toBe(screen.getByText("This value doesn't look right.").id)
    api.saveConnector.mockResolvedValue(result())
    fireEvent.click(screen.getByRole("button", { name: "Save securely" }))
    await waitFor(() => expect(screen.getByText("Connected")).toBeInTheDocument())
  })

  it("falls back to the general alert when error_field matches no field", async () => {
    api.saveConnector.mockResolvedValue(result({ state: "failed", errorCode: "auth_failed", errorField: "NOT_A_FIELD", tools: [], taskResumed: null }))
    render(<ConnectorConnectCard {...props()} />)
    fill()
    fireEvent.click(screen.getByRole("button", { name: "Save securely" }))
    expect(await screen.findByText("The service rejected these credentials.")).toBeInTheDocument()
    expect(screen.getByLabelText("Account endpoint")).not.toHaveAttribute("aria-invalid")
  })

  it("shows a general error for a transport failure", async () => {
    api.saveConnector.mockRejectedValue(new Error("network"))
    render(<ConnectorConnectCard {...props()} />)
    fill()
    fireEvent.click(screen.getByRole("button", { name: "Save securely" }))
    expect(await screen.findByText("Couldn't connect")).toBeInTheDocument()
    expect(screen.getByText("Something went wrong. Check the details and try again.")).toBeInTheDocument()
  })

  it("Not now collapses the body and shows Skipped", async () => {
    api.skipConnector.mockResolvedValue(result({ state: "skipped", taskResumed: true }))
    render(<ConnectorConnectCard {...props()} />)
    fireEvent.click(screen.getByRole("button", { name: "Not now" }))
    expect(await screen.findByText("Skipped")).toBeInTheDocument()
    expect(screen.queryByLabelText("Account endpoint")).toBeNull()
    expect(api.skipConnector).toHaveBeenCalledWith(cosmosConnector.interaction_id)
    expect(screen.queryByRole("button", { name: "Continue" })).toBeNull()
  })

  it("disables both buttons while a skip is in flight, so a double click sends one request", async () => {
    let finish!: (r: ConnectorConnectResult) => void
    api.skipConnector.mockReturnValue(new Promise<ConnectorConnectResult>((r) => { finish = r }))
    render(<ConnectorConnectCard {...props()} />)
    const notNow = screen.getByRole("button", { name: "Not now" })
    fireEvent.click(notNow)
    fireEvent.click(notNow)
    expect(notNow).toBeDisabled()
    expect(screen.getByRole("button", { name: "Save securely" })).toBeDisabled()
    expect(api.skipConnector).toHaveBeenCalledTimes(1)
    await act(async () => { finish(result({ state: "skipped" })) })
  })

  it.each([
    ["a failed result", result({ state: "failed", taskResumed: null })],
    ["an unexpected state", result({ state: "connected" })],
  ])("treats %s from skip as a failed skip: no Skipped, no Continue, inputs kept", async (_label, res) => {
    api.skipConnector.mockResolvedValue(res)
    render(<ConnectorConnectCard {...props()} />)
    fireEvent.change(screen.getByLabelText("Account endpoint"), { target: { value: ENDPOINT } })
    fireEvent.click(screen.getByRole("button", { name: "Not now" }))
    expect(await screen.findByText("Couldn't record that. Please try again.")).toBeInTheDocument()
    expect(screen.queryByText("Skipped")).toBeNull()
    expect(screen.getByText("Not connected")).toBeInTheDocument()
    expect(screen.getByLabelText("Account endpoint")).toHaveValue(ENDPOINT)
    expect(screen.queryByRole("button", { name: "Continue" })).toBeNull()
  })

  it.each([
    ["restore", async () => { api.getConnectorState.mockResolvedValue(result({ state: "unavailable", taskResumed: null })) }],
    ["save", async () => { api.saveConnector.mockResolvedValue(result({ state: "unavailable", taskResumed: null })) }],
    ["skip", async () => { api.skipConnector.mockResolvedValue(result({ state: "unavailable", taskResumed: null })) }],
  ])("shows a clear unavailable message, not a failure, when the host is unavailable on %s", async (label, arrange) => {
    await arrange()
    render(<ConnectorConnectCard {...props()} />)
    if (label === "save") { fill(); fireEvent.click(screen.getByRole("button", { name: "Save securely" })) }
    if (label === "skip") fireEvent.click(screen.getByRole("button", { name: "Not now" }))
    expect(await screen.findByText("Connecting apps from chat isn't available here.")).toBeInTheDocument()
    expect(screen.getByText("Unavailable")).toBeInTheDocument()
    expect(screen.queryByText("Couldn't connect")).toBeNull()
    expect(screen.queryByLabelText("Read-only key")).toBeNull()
  })

  it("shows an error and stays idle when the skip request fails", async () => {
    api.skipConnector.mockRejectedValue(new Error("network"))
    render(<ConnectorConnectCard {...props()} />)
    fireEvent.click(screen.getByRole("button", { name: "Not now" }))
    expect(await screen.findByText("Couldn't record that. Please try again.")).toBeInTheDocument()
    expect(screen.getByText("Not connected")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Not now" })).toBeEnabled()
  })

  describe("Continue", () => {
    it.each([
      ["save", async () => { api.saveConnector.mockResolvedValue(result({ taskResumed: false })); fill(); fireEvent.click(screen.getByRole("button", { name: "Save securely" })) }],
      ["skip", async () => { api.skipConnector.mockResolvedValue(result({ state: "skipped", taskResumed: false })); fireEvent.click(screen.getByRole("button", { name: "Not now" })) }],
    ])("is offered after a %s the host did not resume, and resumes on click", async (_label, act1) => {
      render(<ConnectorConnectCard {...props()} />)
      await act1()
      const button = await screen.findByRole("button", { name: "Continue" })
      expect(screen.getByText("I haven't picked up your request yet.")).toBeInTheDocument()
      api.resumeConnector.mockResolvedValue(result({ taskResumed: true }))
      fireEvent.click(button)
      await waitFor(() => expect(screen.queryByRole("button", { name: "Continue" })).toBeNull())
      expect(api.resumeConnector).toHaveBeenCalledWith(cosmosConnector.interaction_id)
    })

    it("is also offered when the host does not say whether the task was resumed", async () => {
      api.saveConnector.mockResolvedValue(result({ taskResumed: null }))
      render(<ConnectorConnectCard {...props()} />)
      fill()
      fireEvent.click(screen.getByRole("button", { name: "Save securely" }))
      expect(await screen.findByRole("button", { name: "Continue" })).toBeInTheDocument()
    })

    it("is offered for a settled card restored while the task still waits on it", async () => {
      api.getConnectorState.mockResolvedValue(result({ taskResumed: null }))
      render(<ConnectorConnectCard {...props()} />)
      expect(await screen.findByRole("button", { name: "Continue" })).toBeInTheDocument()
    })

    it("is not offered for a settled card the task is no longer waiting on", async () => {
      api.getConnectorState.mockResolvedValue(result({ taskResumed: null }))
      render(<ConnectorConnectCard {...props({ pending: false })} />)
      await screen.findByText("Connected")
      expect(screen.queryByRole("button", { name: "Continue" })).toBeNull()
    })

    it("keeps the connected body and shows the continue error when the host is unavailable on resume", async () => {
      api.getConnectorState.mockResolvedValue(result({ taskResumed: null }))
      api.resumeConnector.mockResolvedValue(result({ state: "unavailable", taskResumed: null }))
      render(<ConnectorConnectCard {...props()} />)
      fireEvent.click(await screen.findByRole("button", { name: "Continue" }))
      expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't continue the task. Try again.")
      expect(screen.getByText("Connected")).toBeInTheDocument()
      expect(screen.queryByText("Connecting apps from chat isn't available here.")).toBeNull()
    })

    it("is disabled while a call is in flight, and a failure keeps it available", async () => {
      api.getConnectorState.mockResolvedValue(result({ taskResumed: null }))
      let finish!: (r: ConnectorConnectResult) => void
      api.resumeConnector.mockReturnValue(new Promise<ConnectorConnectResult>((r) => { finish = r }))
      render(<ConnectorConnectCard {...props()} />)
      const button = await screen.findByRole("button", { name: "Continue" })
      fireEvent.click(button)
      fireEvent.click(button)
      expect(button).toBeDisabled()
      expect(api.resumeConnector).toHaveBeenCalledTimes(1)
      await act(async () => { finish(result({ taskResumed: false })) })
      expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't continue the task. Try again.")
      expect(screen.getByRole("button", { name: "Continue" })).toBeEnabled()
      api.resumeConnector.mockRejectedValue(new Error("offline"))
      fireEvent.click(screen.getByRole("button", { name: "Continue" }))
      await waitFor(() => expect(api.resumeConnector).toHaveBeenCalledTimes(2))
      expect(await screen.findByText("Couldn't continue the task. Try again.")).toBeInTheDocument()
    })
  })
})

describe("ConnectorResultCard", () => {
  afterEach(cleanup)
  it("renders title, source line, items and the footnote as the last item", () => {
    render(<ConnectorResultCard card={cosmosResultCard} />)
    expect(screen.getByText("Query · top customers by revenue")).toBeInTheDocument()
    expect(screen.getByText("From Azure Cosmos DB · just now")).toBeInTheDocument()
    const items = screen.getAllByRole("listitem").map((li) => li.textContent)
    expect(items).toEqual([...cosmosResultCard.items, cosmosResultCard.footnote])
  })
  it("formats the relative time", () => {
    const hourAgo = new Date(Date.now() - 2 * 3600_000).toISOString()
    render(<ConnectorResultCard card={{ ...cosmosResultCard, created_at: hourAgo, footnote: null }} />)
    expect(screen.getByText("From Azure Cosmos DB · 2 h ago")).toBeInTheDocument()
  })
})

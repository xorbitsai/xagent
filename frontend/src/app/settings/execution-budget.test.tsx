import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ExecutionBudgetSettings } from "@/components/settings/execution-budget"

const apiRequest = vi.hoisted(() => vi.fn())
const updateUserPreferences = vi.hoisted(() => vi.fn())
const t = (key: string) => key
vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t }) }))
vi.mock("@/lib/api-wrapper", () => ({ apiRequest }))
vi.mock("@/lib/user-preferences", () => ({ updateUserPreferences }))

const effective = { max_tokens: 10000, soft_limit_percent: 80, source: "system", soft_limit_source: "system" }
function response(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
}
afterEach(cleanup)
beforeEach(() => {
  apiRequest.mockReset()
  updateUserPreferences.mockReset().mockResolvedValue({ ok: true })
  apiRequest.mockImplementation(async (url: string) => url.endsWith("/defaults")
    ? response({ default_tokens: 10000, max_tokens: 20000, soft_limit_percent: 80 })
    : response({ effective, preferences: { execution_budget_tokens: null, execution_budget_soft_percent: null } }))
})

describe("execution budget settings", () => {
  it("shows effective inherited values and saves both personal settings", async () => {
    render(<ExecutionBudgetSettings />)
    await screen.findByTestId("effective-budget")
    expect(screen.getByTestId("effective-budget")).toHaveTextContent("80%")
    expect(screen.queryByText("settings.budget.admin")).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText("settings.budget.tokens"), { target: { value: "8000" } })
    fireEvent.change(screen.getByLabelText("settings.budget.percent"), { target: { value: "65" } })
    fireEvent.click(screen.getByText("settings.budget.savePersonal"))
    await waitFor(() => expect(updateUserPreferences).toHaveBeenCalledWith({ execution_budget_tokens: 8000, execution_budget_soft_percent: 65 }))
    await screen.findByText("settings.budget.saved")
  })

  it("uses null to restore inheritance, not zero or unlimited", async () => {
    render(<ExecutionBudgetSettings />)
    await screen.findByTestId("effective-budget")
    fireEvent.click(screen.getByText("settings.budget.savePersonal"))
    await waitFor(() => expect(updateUserPreferences).toHaveBeenCalledWith({ execution_budget_tokens: null, execution_budget_soft_percent: null }))
  })

  it("allows administrators to save defaults and the separate maximum", async () => {
    render(<ExecutionBudgetSettings isAdmin />)
    const button = screen.getByText("settings.budget.saveDefaults")
    await waitFor(() => expect(button).toBeEnabled())
    fireEvent.change(screen.getByLabelText("settings.budget.maximumTokens"), { target: { value: "12000" } })
    fireEvent.click(button)
    await waitFor(() => expect(apiRequest).toHaveBeenCalledWith(expect.stringContaining("/execution-budget/defaults"), expect.objectContaining({ method: "PUT", body: JSON.stringify({ default_tokens: 10000, max_tokens: 12000, soft_limit_percent: 80 }) })))
    await screen.findByText("settings.budget.saved")
  })

  it("does not present an unavailable policy as unlimited or enable saving", async () => {
    apiRequest.mockResolvedValue(response({}, 503))
    render(<ExecutionBudgetSettings />)
    await screen.findByText("settings.budget.failed")
    expect(screen.queryByTestId("effective-budget")).not.toBeInTheDocument()
    expect(screen.getByText("settings.budget.savePersonal")).toBeDisabled()
  })

  it("does not claim a failed save succeeded", async () => {
    updateUserPreferences.mockResolvedValue({ ok: false })
    render(<ExecutionBudgetSettings />)
    await screen.findByTestId("effective-budget")
    fireEvent.click(screen.getByText("settings.budget.savePersonal"))
    await screen.findByText("settings.budget.failed")
    expect(screen.queryByText("settings.budget.saved")).not.toBeInTheDocument()
  })
})

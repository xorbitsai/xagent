import React, { useState } from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { CustomApiForm } from "./custom-api-form"
import type { MCPServerFormData } from "./custom-api-form"
import { RuntimeInputsForm, getRuntimeConfigError } from "./runtime-inputs-form"
import {
  buildCustomApiPayload,
  customApiDetailToEditState,
  type CustomApiDetail,
} from "../../lib/mcp-utils"

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
  }),
}))

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

function RuntimeFormHarness({
  connectorType = "mcp",
  onValidationErrorChange,
}: {
  connectorType?: "mcp" | "custom_api"
  onValidationErrorChange?: (error: string | null) => void
}) {
  const [formData, setFormData] = useState<MCPServerFormData>({
    name: "records",
    transport: connectorType === "mcp" ? "streamable_http" : "custom_api",
    description: "",
    config: {},
  })

  return (
    <>
      <RuntimeInputsForm
        connectorType={connectorType}
        formData={formData}
        setFormData={setFormData}
        onValidationErrorChange={onValidationErrorChange}
      />
      <pre data-testid="form-state">{JSON.stringify(formData)}</pre>
    </>
  )
}

function formState(): MCPServerFormData {
  return JSON.parse(screen.getByTestId("form-state").textContent || "{}")
}

function CustomApiHarness() {
  const [formData, setFormData] = useState<MCPServerFormData>({
    name: "records",
    transport: "custom_api",
    description: "",
    method: "POST",
    headers: {
      account_id: "static",
      "X-Static": "ok",
    },
    body: "{}",
    config: {},
    runtime_input_schema: {
      context: {
        account_id: { type: "string", required: false },
      },
    },
    runtime_bindings: [
      {
        source: { input_type: "context", key: "account_id" },
        target: { target_type: "headers", key: "account_id" },
      },
      {
        source: { input_type: "context", key: "account_id" },
        target: { target_type: "body_field", path: "account.id" },
      },
    ],
  })
  const [env, setEnv] = useState<{ key: string; value: string }[]>([])

  return (
    <CustomApiForm
      mcpFormData={formData}
      setMcpFormData={setFormData}
      customApiEnv={env}
      setCustomApiEnv={setEnv}
    />
  )
}

const storedEnvDetail: CustomApiDetail = {
  id: 51,
  name: "records",
  description: "Stored description",
  url: "https://example.com/records",
  method: "POST",
  headers: { Authorization: "Bearer $BEARER_TOKEN" },
  body: "{}",
  env: {
    BEARER_TOKEN: "********",
    BASIC_AUTH: "********",
    TENANT: "********",
  },
  runtime_input_schema: null,
  runtime_bindings: null,
  allow_delegated_authorization: false,
}

function CustomApiStoredEnvHarness() {
  const initial = customApiDetailToEditState(storedEnvDetail)
  const [formData, setFormData] = useState<MCPServerFormData>(initial.formData)
  const [env, setEnv] = useState(initial.env)
  const payload = buildCustomApiPayload(formData, env, storedEnvDetail).payload

  return (
    <>
      <CustomApiForm
        mcpFormData={formData}
        setMcpFormData={setFormData}
        customApiEnv={env}
        setCustomApiEnv={setEnv}
        originalEnvObj={storedEnvDetail.env ?? {}}
      />
      <pre data-testid="stored-form-state">{JSON.stringify(formData)}</pre>
      <pre data-testid="stored-env-state">{JSON.stringify(env)}</pre>
      <pre data-testid="stored-payload">{JSON.stringify(payload)}</pre>
    </>
  )
}

const nonCanonicalAuthDetail: CustomApiDetail = {
  id: 52,
  name: "records",
  description: "Stored description",
  url: "https://example.com/records?key=$API_KEY",
  method: "POST",
  headers: {
    Authorization: "Bearer ${BEARER_TOKEN}",
    "X-Basic-Backup": "$BASIC_AUTH",
  },
  body: '{"key":"${API_KEY}"}',
  env: {
    BEARER_TOKEN: "********",
    API_KEY: "********",
    BASIC_AUTH: "********",
    TENANT: "********",
  },
  runtime_input_schema: null,
  runtime_bindings: null,
  allow_delegated_authorization: false,
}

function CustomApiNonCanonicalAuthHarness() {
  const initial = customApiDetailToEditState(nonCanonicalAuthDetail)
  const [formData, setFormData] = useState<MCPServerFormData>(initial.formData)
  const [env, setEnv] = useState(initial.env)
  const payload = buildCustomApiPayload(formData, env, nonCanonicalAuthDetail).payload

  return (
    <>
      <CustomApiForm
        mcpFormData={formData}
        setMcpFormData={setFormData}
        customApiEnv={env}
        setCustomApiEnv={setEnv}
        originalEnvObj={nonCanonicalAuthDetail.env ?? {}}
      />
      <pre data-testid="non-canonical-env-state">{JSON.stringify(env)}</pre>
      <pre data-testid="non-canonical-payload">{JSON.stringify(payload)}</pre>
    </>
  )
}

function CustomApiSessionAuthHarness() {
  const [formData, setFormData] = useState<MCPServerFormData>({
    name: "records",
    transport: "custom_api",
    description: "",
    url: "https://example.com/records",
    method: "GET",
    headers: {},
    config: {},
    runtime_input_schema: null,
    runtime_bindings: null,
    allow_delegated_authorization: false,
  })
  const [env, setEnv] = useState<{ key: string; value: string }[]>([])

  return (
    <>
      <CustomApiForm
        mcpFormData={formData}
        setMcpFormData={setFormData}
        customApiEnv={env}
        setCustomApiEnv={setEnv}
      />
      <pre data-testid="session-env-state">{JSON.stringify(env)}</pre>
    </>
  )
}

describe("RuntimeInputsForm", () => {
  it("writes MCP runtime input declarations and bindings to top-level form data", () => {
    render(<RuntimeFormHarness connectorType="mcp" />)

    fireEvent.click(screen.getByText("tools.mcp.runtime.addInput"))
    expect(formState().runtime_input_schema).toEqual({
      context: {
        account_id: { type: "string", required: false },
      },
    })

    fireEvent.click(screen.getByText("tools.mcp.runtime.addBinding"))
    expect(formState().runtime_bindings).toEqual([
      {
        source: { input_type: "context", key: "account_id" },
        target: { target_type: "mcp_meta", key: "account_id" },
      },
    ])
  })

  it("keeps input focus while editing runtime keys", () => {
    render(<RuntimeFormHarness connectorType="mcp" />)

    fireEvent.click(screen.getByText("tools.mcp.runtime.addInput"))
    const keyInput = screen.getByPlaceholderText("tools.mcp.runtime.key")
    keyInput.focus()

    fireEvent.change(keyInput, { target: { value: "tenant_id" } })

    expect(document.activeElement).toBe(keyInput)
    expect(formState().runtime_input_schema).toEqual({
      context: {
        tenant_id: { type: "string", required: false },
      },
    })
  })

  it("keeps runtime input rows visible while a key is temporarily empty", () => {
    render(<RuntimeFormHarness connectorType="mcp" />)

    fireEvent.click(screen.getByText("tools.mcp.runtime.addInput"))
    const keyInput = screen.getByPlaceholderText("tools.mcp.runtime.key")
    keyInput.focus()

    fireEvent.change(keyInput, { target: { value: "" } })

    expect(document.activeElement).toBe(keyInput)
    expect(screen.getByPlaceholderText("tools.mcp.runtime.key")).toBeInTheDocument()
    expect(screen.queryByText("tools.mcp.runtime.noInputs")).not.toBeInTheDocument()

    fireEvent.change(keyInput, { target: { value: "tenant_id" } })
    expect(formState().runtime_input_schema).toEqual({
      context: {
        tenant_id: { type: "string", required: false },
      },
    })
  })

  it("warns when runtime input keys are duplicated within an input type", () => {
    render(<RuntimeFormHarness connectorType="mcp" />)

    fireEvent.click(screen.getByText("tools.mcp.runtime.addInput"))
    fireEvent.click(screen.getByText("tools.mcp.runtime.addInput"))
    const keyInputs = screen.getAllByPlaceholderText("tools.mcp.runtime.key")
    fireEvent.change(keyInputs[1], { target: { value: "account_id" } })

    expect(
      screen.getByText("tools.mcp.runtime.errors.duplicateInput"),
    ).toBeInTheDocument()
  })

  it("reports duplicate runtime input errors before schema folding hides them", async () => {
    const onValidationErrorChange = vi.fn()
    render(
      <RuntimeFormHarness
        connectorType="mcp"
        onValidationErrorChange={onValidationErrorChange}
      />,
    )

    fireEvent.click(screen.getByText("tools.mcp.runtime.addInput"))
    fireEvent.click(screen.getByText("tools.mcp.runtime.addInput"))
    const keyInputs = screen.getAllByPlaceholderText("tools.mcp.runtime.key")
    fireEvent.change(keyInputs[1], { target: { value: "account_id" } })

    expect(formState().runtime_input_schema).toEqual({
      context: {
        account_id: { type: "string", required: false },
      },
    })
    await waitFor(() => {
      expect(onValidationErrorChange).toHaveBeenLastCalledWith(
        "tools.mcp.runtime.errors.duplicateInput",
      )
    })
  })

  it("writes Custom API bindings and delegated authorization to top-level form data", () => {
    render(<RuntimeFormHarness connectorType="custom_api" />)

    fireEvent.click(screen.getByText("tools.mcp.runtime.addInput"))
    fireEvent.click(screen.getByText("tools.mcp.runtime.addBinding"))
    fireEvent.click(
      screen.getByRole("switch", {
        name: "tools.mcp.runtime.delegatedAuthorization",
      }),
    )

    expect(formState().runtime_bindings).toEqual([
      {
        source: { input_type: "context", key: "account_id" },
        target: { target_type: "headers", key: "account_id" },
      },
    ])
    expect(formState().allow_delegated_authorization).toBe(true)
  })

  it("shows validation errors for authorization bindings without delegated authorization", () => {
    render(
      <RuntimeInputsForm
        connectorType="custom_api"
        formData={{
          name: "records",
          transport: "custom_api",
          description: "",
          config: {},
          runtime_input_schema: {
            secrets: {
              authorization: { type: "string", required: false },
            },
          },
          runtime_bindings: [
            {
              source: { input_type: "secrets", key: "authorization" },
              target: { target_type: "headers", key: "Authorization" },
            },
          ],
          allow_delegated_authorization: false,
        }}
        setFormData={vi.fn()}
      />,
    )

    expect(
      screen.getByText("tools.mcp.runtime.errors.authorizationRequiresDelegated"),
    ).toBeInTheDocument()
  })

  it("returns a save-blocking validation error for invalid runtime bindings", () => {
    expect(
      getRuntimeConfigError(
        {
          name: "records",
          transport: "custom_api",
          description: "",
          config: {},
          runtime_input_schema: {
            secrets: {
              authorization: { type: "string", required: false },
            },
          },
          runtime_bindings: [
            {
              source: { input_type: "secrets", key: "authorization" },
              target: { target_type: "headers", key: "Authorization" },
            },
          ],
          allow_delegated_authorization: false,
        },
        "custom_api",
      ),
    ).toBe("tools.mcp.runtime.errors.authorizationRequiresDelegated")
  })

  it("renders Custom API runtime-bound headers and body fields as read-only references", () => {
    render(<CustomApiHarness />)

    fireEvent.click(screen.getByText("tools.mcp.dialog.advancedOptions"))

    expect(screen.getByText("tools.mcp.runtime.boundHeaders")).toBeInTheDocument()
    expect(screen.getByText("tools.mcp.runtime.boundBodyFields")).toBeInTheDocument()
    expect(
      screen.getAllByDisplayValue("account_id").some((item) => item.hasAttribute("disabled")),
    ).toBe(true)
    expect(screen.getAllByDisplayValue("$account_id").length).toBeGreaterThanOrEqual(2)
    expect(
      screen.getAllByDisplayValue("account.id").some((item) => item.hasAttribute("disabled")),
    ).toBe(true)
  })

  it("preserves non-auth environment entries while initializing and editing an authenticated API", async () => {
    render(<CustomApiStoredEnvHarness />)

    await waitFor(() => {
      expect(JSON.parse(screen.getByTestId("stored-env-state").textContent || "[]")).toEqual([
        { key: "BEARER_TOKEN", value: "********" },
        { key: "BASIC_AUTH", value: "********" },
        { key: "TENANT", value: "********" },
      ])
    })

    fireEvent.change(screen.getByLabelText("tools.mcp.form.descriptionLabel"), {
      target: { value: "Updated description" },
    })

    await waitFor(() => {
      expect(
        JSON.parse(screen.getByTestId("stored-form-state").textContent || "{}")
          .description,
      ).toBe("Updated description")
      expect(JSON.parse(screen.getByTestId("stored-env-state").textContent || "[]")).toEqual([
        { key: "BEARER_TOKEN", value: "********" },
        { key: "BASIC_AUTH", value: "********" },
        { key: "TENANT", value: "********" },
      ])
    })
  })

  it("keeps a persisted secret key immutable while allowing its value to change", async () => {
    render(<CustomApiStoredEnvHarness />)

    fireEvent.click(screen.getByText("tools.mcp.dialog.advancedOptions"))
    const keyInput = await screen.findByLabelText(
      "tools.mcp.dialog.customApiSecretName TENANT",
    )
    const valueInput = screen.getByLabelText(
      "tools.mcp.dialog.customApiSecretValue TENANT",
    )

    expect(keyInput).toBeDisabled()
    fireEvent.change(valueInput, { target: { value: "replacement-secret" } })
    await waitFor(() => expect(valueInput).toHaveValue("replacement-secret"))
    expect(keyInput).toBeDisabled()

    fireEvent.change(valueInput, { target: { value: "" } })
    await waitFor(() => {
      expect(JSON.parse(screen.getByTestId("stored-payload").textContent || "{}")).toEqual({})
    })
    fireEvent.blur(valueInput)
    await waitFor(() => expect(valueInput).toHaveValue("********"))
  })

  it("does not turn non-canonical reserved-name secrets into an edit delta", async () => {
    render(<CustomApiNonCanonicalAuthHarness />)

    await waitFor(() => {
      expect(
        JSON.parse(screen.getByTestId("non-canonical-env-state").textContent || "[]"),
      ).toEqual([
        { key: "BEARER_TOKEN", value: "********" },
        { key: "API_KEY", value: "********" },
        { key: "BASIC_AUTH", value: "********" },
        { key: "TENANT", value: "********" },
      ])
      expect(
        JSON.parse(screen.getByTestId("non-canonical-payload").textContent || "{}"),
      ).toEqual({})
    })

    fireEvent.change(screen.getByLabelText("tools.mcp.form.descriptionLabel"), {
      target: { value: "Updated description" },
    })

    await waitFor(() => {
      expect(
        JSON.parse(screen.getByTestId("non-canonical-payload").textContent || "{}"),
      ).toEqual({ description: "Updated description" })
    })
  })

  it("preserves baseline secrets while switching authentication types", async () => {
    window.HTMLElement.prototype.scrollIntoView = vi.fn()
    render(<CustomApiStoredEnvHarness />)

    await waitFor(() => {
      expect(screen.getByRole("combobox")).toHaveTextContent(
        "tools.mcp.dialog.authTypes.bearer",
      )
    })

    fireEvent.click(screen.getByRole("combobox"))
    fireEvent.click(await screen.findByText("tools.mcp.dialog.authTypes.apiKey"))

    await waitFor(() => {
      expect(JSON.parse(screen.getByTestId("stored-env-state").textContent || "[]")).toEqual([
        { key: "BEARER_TOKEN", value: "********" },
        { key: "BASIC_AUTH", value: "********" },
        { key: "TENANT", value: "********" },
        { key: "API_KEY", value: "" },
      ])
    })
  })

  it("removes only authentication secrets created in the current form session", async () => {
    window.HTMLElement.prototype.scrollIntoView = vi.fn()
    render(<CustomApiSessionAuthHarness />)

    fireEvent.click(screen.getByRole("combobox"))
    fireEvent.click(await screen.findByText("tools.mcp.dialog.authTypes.bearer"))
    fireEvent.change(screen.getByLabelText("tools.mcp.dialog.token"), {
      target: { value: "new-token" },
    })

    await waitFor(() => {
      expect(JSON.parse(screen.getByTestId("session-env-state").textContent || "[]")).toEqual([
        { key: "BEARER_TOKEN", value: "new-token" },
      ])
    })

    fireEvent.click(screen.getByRole("combobox"))
    fireEvent.click(await screen.findByText("tools.mcp.dialog.authTypes.none"))

    await waitFor(() => {
      expect(JSON.parse(screen.getByTestId("session-env-state").textContent || "[]")).toEqual([])
    })
  })

  it("removes a persisted canonical authentication secret only through an explicit action", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true)
    render(<CustomApiStoredEnvHarness />)

    fireEvent.click(screen.getByText("tools.mcp.dialog.advancedOptions"))

    fireEvent.click(screen.getByRole("button", {
      name: "tools.mcp.dialog.removeSecret BEARER_TOKEN",
    }))

    await waitFor(() => {
      expect(JSON.parse(screen.getByTestId("stored-env-state").textContent || "[]")).toEqual([
        { key: "BASIC_AUTH", value: "********" },
        { key: "TENANT", value: "********" },
      ])
      expect(
        JSON.parse(screen.getByTestId("stored-form-state").textContent || "{}").headers,
      ).toEqual({})
      expect(JSON.parse(screen.getByTestId("stored-payload").textContent || "{}")).toEqual({
        headers: {},
        env: {
          BASIC_AUTH: "********",
          TENANT: "********",
        },
      })
    })
  })

  it("removes a non-auth secret without changing canonical authentication", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true)
    render(<CustomApiStoredEnvHarness />)
    fireEvent.click(screen.getByText("tools.mcp.dialog.advancedOptions"))

    fireEvent.click(screen.getByRole("button", {
      name: "tools.mcp.dialog.removeSecret TENANT",
    }))

    await waitFor(() => {
      expect(JSON.parse(screen.getByTestId("stored-env-state").textContent || "[]")).toEqual([
        { key: "BEARER_TOKEN", value: "********" },
        { key: "BASIC_AUTH", value: "********" },
      ])
      expect(
        JSON.parse(screen.getByTestId("stored-form-state").textContent || "{}").headers,
      ).toEqual({ Authorization: "Bearer $BEARER_TOKEN" })
    })
  })

  it("keeps a persisted secret when explicit removal is cancelled", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(false)
    render(<CustomApiStoredEnvHarness />)
    fireEvent.click(screen.getByText("tools.mcp.dialog.advancedOptions"))

    fireEvent.click(screen.getByRole("button", {
      name: "tools.mcp.dialog.removeSecret BEARER_TOKEN",
    }))

    expect(JSON.parse(screen.getByTestId("stored-payload").textContent || "{}")).toEqual({})
    expect(JSON.parse(screen.getByTestId("stored-env-state").textContent || "[]")).toEqual([
      { key: "BEARER_TOKEN", value: "********" },
      { key: "BASIC_AUTH", value: "********" },
      { key: "TENANT", value: "********" },
    ])
  })
})

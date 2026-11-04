import React, { useState } from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { I18nProvider, type Locale } from "@/contexts/i18n-context"

import { CustomApiForm, type MCPServerFormData } from "./custom-api-form"
import { RuntimeInputsForm } from "./runtime-inputs-form"

const runtimeSchema = {
  context: {
    account_id: { type: "string", required: true },
  },
}

function RuntimeFormHarness({ locale }: { locale: Locale }) {
  const [formData, setFormData] = useState<MCPServerFormData>({
    name: "ShiftCare",
    transport: "streamable_http",
    description: "",
    runtime_input_schema: runtimeSchema,
    runtime_bindings: null,
  })

  return (
    <I18nProvider initialLocale={locale}>
      <RuntimeInputsForm
        connectorType="mcp"
        formData={formData}
        setFormData={setFormData}
      />
    </I18nProvider>
  )
}

function CustomApiHarness() {
  const [formData, setFormData] = useState<MCPServerFormData>({
    name: "ShiftCare API",
    transport: "custom_api",
    description: "",
    url: "https://example.test/api",
    method: "POST",
    headers: {},
    runtime_input_schema: runtimeSchema,
    runtime_bindings: [
      {
        source: { input_type: "context", key: "account_id" },
        target: { target_type: "headers", key: "X-Account-Id" },
      },
      {
        source: { input_type: "context", key: "account_id" },
        target: { target_type: "body_field", path: "tenant.account_id" },
      },
    ],
  })
  const [env, setEnv] = useState<{ key: string; value: string }[]>([])

  return (
    <I18nProvider initialLocale="en">
      <CustomApiForm
        mcpFormData={formData}
        setMcpFormData={setFormData}
        customApiEnv={env}
        setCustomApiEnv={setEnv}
      />
    </I18nProvider>
  )
}

describe("connector runtime translations", () => {
  it.each([
    ["en", "Runtime Inputs", "Bindings"],
    ["zh", "运行时入参", "绑定"],
  ] as const)("renders %s MCP labels through the real provider", (locale, title, bindings) => {
    const { container } = render(<RuntimeFormHarness locale={locale} />)

    expect(screen.getByText(title)).toBeInTheDocument()
    expect(screen.getByText(bindings)).toBeInTheDocument()
    expect(container.textContent).not.toContain("tools.mcp.runtime.")
  })

  it("renders Custom API runtime-bound header and body labels", () => {
    const { container } = render(<CustomApiHarness />)

    fireEvent.click(screen.getByRole("button", { name: "Advanced Options" }))

    expect(screen.getByText("Runtime-bound headers")).toBeInTheDocument()
    expect(screen.getByText("Runtime-bound body fields")).toBeInTheDocument()
    expect(container.textContent).not.toContain("tools.mcp.runtime.")
  })
})

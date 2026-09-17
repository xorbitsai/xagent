import { describe, expect, it } from "vitest"

import {
  buildCustomApiPayload,
  buildMcpServerPayload,
  customApiDetailToEditState,
  mcpServerDetailToEditState,
  MCP_OAUTH_POPUP_WINDOW_NAME,
  MCP_OAUTH_SUCCESS_PARAM,
  parseCustomApiDetail,
  parseMcpServerDetail,
  shouldSelfCloseMcpOauthPopup,
} from "./mcp-utils"

describe("shouldSelfCloseMcpOauthPopup", () => {
  it("closes when the success param is present and the window is the mcp_oauth popup", () => {
    const params = new URLSearchParams({ [MCP_OAUTH_SUCCESS_PARAM]: "1" })
    expect(shouldSelfCloseMcpOauthPopup(MCP_OAUTH_POPUP_WINDOW_NAME, params)).toBe(true)
  })

  it("does not close without the success param, even under the popup window name", () => {
    const params = new URLSearchParams()
    expect(shouldSelfCloseMcpOauthPopup(MCP_OAUTH_POPUP_WINDOW_NAME, params)).toBe(false)
  })

  it("does not close on an error redirect, even under the popup window name", () => {
    const params = new URLSearchParams({
      mcp_oauth_error: "invalid_resource",
      mcp_oauth_error_message: "MCP OAuth token_type must be at most 50 characters",
    })
    expect(shouldSelfCloseMcpOauthPopup(MCP_OAUTH_POPUP_WINDOW_NAME, params)).toBe(false)
  })

  it("does not close a differently-named window even with the success param", () => {
    const params = new URLSearchParams({ [MCP_OAUTH_SUCCESS_PARAM]: "1" })
    expect(shouldSelfCloseMcpOauthPopup("", params)).toBe(false)
    expect(shouldSelfCloseMcpOauthPopup("some-other-window", params)).toBe(false)
  })
})

const detailPayload = {
  id: 41,
  user_id: 7,
  name: "records_api",
  description: "Stored description",
  url: "https://example.com/records",
  method: "POST",
  headers: { Authorization: "Bearer $BEARER_TOKEN" },
  body: '{"limit": 10}',
  env: { BEARER_TOKEN: "********", TENANT: "********" },
  runtime_input_schema: {
    context: {
      account_id: { type: "string", required: true },
    },
  },
  runtime_bindings: [
    {
      source: { input_type: "context", key: "account_id" },
      target: { target_type: "headers", key: "account_id" },
    },
  ],
  allow_delegated_authorization: true,
  is_active: true,
  is_default: false,
  created_at: "2026-07-01T00:00:00",
  updated_at: "2026-07-02T00:00:00",
}

const mcpDetailPayload = {
  id: 73,
  user_id: 7,
  name: "records_mcp",
  transport: "streamable_http",
  description: "Stored MCP description",
  config: {
    url: "https://example.com/mcp",
    headers: { "X-Static": "stored" },
    env: { GLOBAL_TOKEN: "********" },
  },
  is_active: true,
  is_default: false,
  user_env: { USER_TOKEN: "********" },
  env_source: "own",
  runtime_input_schema: {
    context: { account_id: { type: "string", required: true } },
  },
  runtime_bindings: [
    {
      source: { input_type: "context", key: "account_id" },
      target: { target_type: "mcp_meta", key: "account_id" },
    },
  ],
  allow_delegated_authorization: true,
  can_edit_global: false,
  transport_display: "Streamable HTTP",
  created_at: "2026-07-01T00:00:00",
  updated_at: "2026-07-02T00:00:00",
}

describe("parseCustomApiDetail", () => {
  it("parses the authoritative detail and converts it losslessly to edit state", () => {
    const detail = parseCustomApiDetail(detailPayload)
    const editState = customApiDetailToEditState(detail)

    expect(editState.formData).toEqual({
      name: "records_api",
      transport: "custom_api",
      description: "Stored description",
      url: "https://example.com/records",
      method: "POST",
      headers: { Authorization: "Bearer $BEARER_TOKEN" },
      body: '{"limit": 10}',
      config: {
        url: "https://example.com/records",
        method: "POST",
        headers: { Authorization: "Bearer $BEARER_TOKEN" },
        body: '{"limit": 10}',
        env: { BEARER_TOKEN: "********", TENANT: "********" },
      },
      runtime_input_schema: detailPayload.runtime_input_schema,
      runtime_bindings: detailPayload.runtime_bindings,
      allow_delegated_authorization: true,
    })
    expect(editState.env).toEqual([
      { key: "BEARER_TOKEN", value: "********" },
      { key: "TENANT", value: "********" },
    ])
  })

  it.each([
    ["a missing runtime_input_schema", { runtime_input_schema: undefined }],
    ["an invalid runtime_input_schema", { runtime_input_schema: [] }],
    ["a missing runtime_bindings", { runtime_bindings: undefined }],
    ["invalid runtime_bindings", { runtime_bindings: {} }],
    ["a missing delegated-authorization flag", { allow_delegated_authorization: undefined }],
    ["a non-boolean delegated-authorization flag", { allow_delegated_authorization: "false" }],
  ])("rejects %s", (_label, replacement) => {
    expect(() => parseCustomApiDetail({ ...detailPayload, ...replacement })).toThrow(
      "Invalid Custom API detail response",
    )
  })
})

describe("buildCustomApiPayload", () => {
  it("whitelists and fully emits the create contract", () => {
    const result = buildCustomApiPayload(
      {
        name: "new_api",
        transport: "custom_api",
        description: "",
        url: "https://example.com/new",
        method: "GET",
        headers: {},
        body: "",
        config: { must_not_leak: true },
        user_env: { must_not_leak: "secret" },
        runtime_input_schema: null,
        runtime_bindings: null,
        allow_delegated_authorization: false,
      },
      [{ key: "TOKEN", value: "new-secret" }],
    )

    expect(result.payload).toEqual({
      name: "new_api",
      description: "",
      url: "https://example.com/new",
      method: "GET",
      headers: {},
      body: "",
      env: { TOKEN: "new-secret" },
      runtime_input_schema: null,
      runtime_bindings: null,
      allow_delegated_authorization: false,
    })
  })

  it("emits no fields for an unchanged edit, including unchanged masked env", () => {
    const detail = parseCustomApiDetail(detailPayload)
    const { formData, env } = customApiDetailToEditState(detail)

    expect(buildCustomApiPayload(formData, env, detail).payload).toEqual({})
  })

  it("emits only an unrelated changed field", () => {
    const detail = parseCustomApiDetail(detailPayload)
    const { formData, env } = customApiDetailToEditState(detail)

    expect(
      buildCustomApiPayload(
        { ...formData, description: "Updated description" },
        env,
        detail,
      ).payload,
    ).toEqual({ description: "Updated description" })
  })

  it("keeps the authoritative baseline isolated from nested form edits", () => {
    const detail = parseCustomApiDetail(detailPayload)
    const { formData, env } = customApiDetailToEditState(detail)

    formData.headers.Authorization = "Bearer replacement"
    const schema = formData.runtime_input_schema as {
      context: { account_id: { required: boolean } }
    }
    schema.context.account_id.required = false

    expect(detail.headers?.Authorization).toBe("Bearer $BEARER_TOKEN")
    expect(
      (detail.runtime_input_schema as { context: { account_id: { required: boolean } } })
        .context.account_id.required,
    ).toBe(true)
    expect(buildCustomApiPayload(formData, env, detail).payload).toMatchObject({
      headers: { Authorization: "Bearer replacement" },
      runtime_input_schema: {
        context: { account_id: { type: "string", required: false } },
      },
    })
  })

  it("sends the complete env replacement when a secret changes", () => {
    const detail = parseCustomApiDetail(detailPayload)
    const { formData, env } = customApiDetailToEditState(detail)
    const changedEnv = env.map((entry) =>
      entry.key === "BEARER_TOKEN" ? { ...entry, value: "replacement" } : entry,
    )

    expect(buildCustomApiPayload(formData, changedEnv, detail).payload).toEqual({
      env: { BEARER_TOKEN: "replacement", TENANT: "********" },
    })
  })

  it("sends the remaining masked siblings when a secret is explicitly removed", () => {
    const detail = parseCustomApiDetail(detailPayload)
    const { formData, env } = customApiDetailToEditState(detail)

    expect(
      buildCustomApiPayload(
        formData,
        env.filter((entry) => entry.key !== "BEARER_TOKEN"),
        detail,
      ).payload,
    ).toEqual({ env: { TENANT: "********" } })
  })

  it("does not treat a transient blank persisted secret value as deletion", () => {
    const detail = parseCustomApiDetail(detailPayload)
    const { formData, env } = customApiDetailToEditState(detail)

    expect(
      buildCustomApiPayload(
        formData,
        env.map((entry) => entry.key === "TENANT" ? { ...entry, value: "" } : entry),
        detail,
      ).payload,
    ).toEqual({})
  })

  it("preserves explicit falsy clears in an edit delta", () => {
    const detail = parseCustomApiDetail(detailPayload)
    const { formData } = customApiDetailToEditState(detail)

    expect(
      buildCustomApiPayload(
        {
          ...formData,
          description: "",
          body: "",
          runtime_input_schema: null,
          runtime_bindings: null,
          allow_delegated_authorization: false,
        },
        [{ key: "", value: "" }],
        detail,
      ).payload,
    ).toEqual({
      description: "",
      body: "",
      env: {},
      runtime_input_schema: null,
      runtime_bindings: null,
      allow_delegated_authorization: false,
    })
  })
})

describe("MCP edit contract", () => {
  it("parses the authoritative detail and converts every editable field", () => {
    const detail = parseMcpServerDetail(mcpDetailPayload)

    expect(mcpServerDetailToEditState(detail).formData).toEqual({
      name: "records_mcp",
      transport: "streamable_http",
      description: "Stored MCP description",
      config: mcpDetailPayload.config,
      user_env: { USER_TOKEN: "********" },
      can_edit_global: false,
      runtime_input_schema: mcpDetailPayload.runtime_input_schema,
      runtime_bindings: mcpDetailPayload.runtime_bindings,
      allow_delegated_authorization: true,
    })
  })

  it.each([
    ["user_env", { user_env: undefined }],
    ["can_edit_global", { can_edit_global: undefined }],
    ["runtime_input_schema", { runtime_input_schema: undefined }],
    ["runtime_bindings", { runtime_bindings: undefined }],
    ["allow_delegated_authorization", { allow_delegated_authorization: undefined }],
  ])("rejects a detail response missing %s", (_field, replacement) => {
    expect(() => parseMcpServerDetail({ ...mcpDetailPayload, ...replacement })).toThrow(
      "Invalid MCP server detail response",
    )
  })

  it("emits no fields for an unchanged edit", () => {
    const detail = parseMcpServerDetail(mcpDetailPayload)
    const { formData } = mcpServerDetailToEditState(detail)

    expect(buildMcpServerPayload(formData, detail)).toEqual({})
  })

  it("emits only an unrelated changed field", () => {
    const detail = parseMcpServerDetail(mcpDetailPayload)
    const { formData } = mcpServerDetailToEditState(detail)

    expect(
      buildMcpServerPayload({ ...formData, description: "Updated description" }, detail),
    ).toEqual({ description: "Updated description" })
  })

  it("sends the complete per-user env replacement when a key is added", () => {
    const detail = parseMcpServerDetail(mcpDetailPayload)
    const { formData } = mcpServerDetailToEditState(detail)

    expect(
      buildMcpServerPayload(
        {
          ...formData,
          user_env: { ...formData.user_env, NEW_TOKEN: "new-secret" },
        },
        detail,
      ),
    ).toEqual({
      user_env: { USER_TOKEN: "********", NEW_TOKEN: "new-secret" },
    })
  })

  it("preserves transient blank masked values in global and per-user env", () => {
    const detail = parseMcpServerDetail(mcpDetailPayload)
    const { formData } = mcpServerDetailToEditState(detail)

    expect(
      buildMcpServerPayload(
        {
          ...formData,
          config: {
            ...formData.config,
            env: { GLOBAL_TOKEN: "" },
          },
          user_env: { USER_TOKEN: "" },
        },
        detail,
      ),
    ).toEqual({})
  })

  it("whitelists the complete create contract", () => {
    expect(
      buildMcpServerPayload({
        name: "new_mcp",
        transport: "stdio",
        description: "",
        config: { command: "python" },
        user_env: { API_KEY: "secret" },
        can_edit_global: true,
        runtime_input_schema: null,
        runtime_bindings: null,
        allow_delegated_authorization: false,
        must_not_leak: "value",
      }),
    ).toEqual({
      name: "new_mcp",
      transport: "stdio",
      description: "",
      config: { command: "python" },
      user_env: { API_KEY: "secret" },
      runtime_input_schema: null,
      runtime_bindings: null,
      allow_delegated_authorization: false,
    })
  })
})

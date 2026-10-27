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
  signed_uploads: null,
  signed_uploads_error: null,
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
      signed_uploads: null,
      signed_uploads_error: null,
    })
  })

  it.each([
    ["user_env", { user_env: undefined }],
    ["can_edit_global", { can_edit_global: undefined }],
    ["runtime_input_schema", { runtime_input_schema: undefined }],
    ["runtime_bindings", { runtime_bindings: undefined }],
    ["allow_delegated_authorization", { allow_delegated_authorization: undefined }],
    ["a non-string signed_uploads_error", { signed_uploads_error: 1 }],
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

const SIGNED_UPLOADS = [
  {
    tool_name: "attach_document",
    file_argument: "document_file_id",
    issuing_tool_name: "request_upload",
    allowed_upload_hosts: ["uploads.example.test"],
    upload_path_prefix: "/staged/",
    content_types: { ".pdf": "application/pdf" },
  },
]
const SETTINGS_ERROR = "signed_uploads[0] must have exactly the documented keys"
// Stored entries that cannot be used: the detail echoes them as they are.
const BAD_SIGNED_UPLOADS = [{ tool_name: "attach_document", file_argument: "document_file_id" }]
// A stored value that is not a list: the detail reads it as null.
const NOT_A_LIST = { tool_name: "attach_document" }
const NOT_A_LIST_ERROR = "signed_uploads must be a list"

function withoutKey(object: Record<string, unknown>, key: string): Record<string, unknown> {
  return Object.fromEntries(Object.entries(object).filter(([name]) => name !== key))
}

function ownerDetail(overrides: Record<string, unknown> = {}) {
  return parseMcpServerDetail({ ...mcpDetailPayload, can_edit_global: true, ...overrides })
}

describe("MCP signed upload settings", () => {
  it("parses signed upload settings and the settings error into edit state", () => {
    const detail = ownerDetail({ signed_uploads: SIGNED_UPLOADS, signed_uploads_error: null })
    expect(detail.signed_uploads).toEqual(SIGNED_UPLOADS)
    expect(detail.signed_uploads_error).toBeNull()

    const { formData } = mcpServerDetailToEditState(detail)
    expect(formData.signed_uploads).toEqual(SIGNED_UPLOADS)
    expect(formData.signed_uploads_error).toBeNull()

    ;(formData.signed_uploads as Record<string, unknown>[])[0].tool_name = "changed"
    expect(detail.signed_uploads).toEqual(SIGNED_UPLOADS)

    const bad = ownerDetail({
      signed_uploads: BAD_SIGNED_UPLOADS,
      signed_uploads_error: SETTINGS_ERROR,
    })
    const badForm = mcpServerDetailToEditState(bad).formData
    expect(badForm.signed_uploads).toEqual(BAD_SIGNED_UPLOADS)
    expect(badForm.signed_uploads_error).toBe(SETTINGS_ERROR)
  })

  it("rejects a detail response without a signed upload key", () => {
    expect(() =>
      parseMcpServerDetail(withoutKey(mcpDetailPayload, "signed_uploads")),
    ).toThrow("Invalid MCP server detail response: signed_uploads is missing")
    expect(() =>
      parseMcpServerDetail(withoutKey(mcpDetailPayload, "signed_uploads_error")),
    ).toThrow("Invalid MCP server detail response: signed_uploads_error is missing")
  })

  it.each([
    ["an object", {}],
    ["a string", "x"],
    ["a number", 1],
    ["a boolean", true],
    ["a stored object", NOT_A_LIST],
  ])("treats a signed_uploads value that is not a list as no settings", (_name, stored) => {
    const detail = ownerDetail({ signed_uploads: stored, signed_uploads_error: NOT_A_LIST_ERROR })
    expect(detail.signed_uploads).toBeNull()
    expect(detail.signed_uploads_error).toBe(NOT_A_LIST_ERROR)
  })

  it("emits no signed_uploads for an unchanged edit", () => {
    const detail = ownerDetail({ signed_uploads: SIGNED_UPLOADS })
    const { formData } = mcpServerDetailToEditState(detail)

    expect(buildMcpServerPayload(formData, detail)).toEqual({})
  })

  it("sends edited signed upload settings", () => {
    const detail = ownerDetail({ signed_uploads: SIGNED_UPLOADS })
    const { formData } = mcpServerDetailToEditState(detail)
    const edited = [{ ...SIGNED_UPLOADS[0], upload_path_prefix: "/incoming/" }]

    expect(buildMcpServerPayload({ ...formData, signed_uploads: edited }, detail)).toEqual({
      signed_uploads: edited,
    })
  })

  it("sends an empty list when clearing invalid stored settings", () => {
    const detail = ownerDetail({
      signed_uploads: BAD_SIGNED_UPLOADS,
      signed_uploads_error: SETTINGS_ERROR,
    })
    const { formData } = mcpServerDetailToEditState(detail)

    expect(buildMcpServerPayload({ ...formData, signed_uploads: [] }, detail)).toEqual({
      signed_uploads: [],
    })
  })

  it("sends an empty list when clearing settings that are not a list", () => {
    const detail = ownerDetail({ signed_uploads: NOT_A_LIST, signed_uploads_error: NOT_A_LIST_ERROR })
    const { formData } = mcpServerDetailToEditState(detail)

    expect(buildMcpServerPayload({ ...formData, signed_uploads: [] }, detail)).toEqual({
      signed_uploads: [],
    })
  })

  it.each([
    ["entries that cannot be used", BAD_SIGNED_UPLOADS, SETTINGS_ERROR],
    ["a value that is not a list", NOT_A_LIST, NOT_A_LIST_ERROR],
  ])("leaves invalid stored settings alone when the form does not touch them", (_name, stored, error) => {
    const detail = ownerDetail({ signed_uploads: stored, signed_uploads_error: error })
    const { formData } = mcpServerDetailToEditState(detail)

    expect(buildMcpServerPayload({ ...formData, description: "Updated" }, detail)).toEqual({
      description: "Updated",
    })
  })

  it.each(
    ["stdio", "websocket"].flatMap((target) => [
      [target, { signed_uploads: SIGNED_UPLOADS, signed_uploads_error: null }],
      [target, { signed_uploads: BAD_SIGNED_UPLOADS, signed_uploads_error: SETTINGS_ERROR }],
      [target, { signed_uploads: NOT_A_LIST, signed_uploads_error: NOT_A_LIST_ERROR }],
      [target, { signed_uploads: null, signed_uploads_error: null }],
    ]),
  )("sends an empty list when an edit leaves sse or streamable_http", (target, stored) => {
    const detail = ownerDetail(stored as Record<string, unknown>)
    const { formData } = mcpServerDetailToEditState(detail)

    expect(buildMcpServerPayload({ ...formData, transport: target as string }, detail)).toEqual({
      transport: target,
      signed_uploads: [],
    })
  })

  it("keeps signed upload settings when switching between sse and streamable_http", () => {
    const detail = ownerDetail({ transport: "sse", signed_uploads: SIGNED_UPLOADS })
    const { formData } = mcpServerDetailToEditState(detail)

    expect(
      buildMcpServerPayload({ ...formData, transport: "streamable_http" }, detail),
    ).toEqual({ transport: "streamable_http" })
  })

  it("sends nothing for transports that never carry settings", () => {
    const detail = ownerDetail({ transport: "stdio" })
    const { formData } = mcpServerDetailToEditState(detail)

    expect(buildMcpServerPayload({ ...formData, transport: "websocket" }, detail)).toEqual({
      transport: "websocket",
    })
  })

  it("never sends signed_uploads for a viewer who cannot edit the shared config", () => {
    const detail = parseMcpServerDetail({ ...mcpDetailPayload, signed_uploads: SIGNED_UPLOADS })
    const { formData } = mcpServerDetailToEditState(detail)
    const edited = [{ ...SIGNED_UPLOADS[0], upload_path_prefix: "/other/" }]

    expect(
      "signed_uploads" in buildMcpServerPayload({ ...formData, signed_uploads: edited }, detail),
    ).toBe(false)
    expect(
      "signed_uploads" in buildMcpServerPayload(
        { ...formData, signed_uploads: edited, transport: "stdio" },
        detail,
      ),
    ).toBe(false)
  })

  it.each([
    ["sse", SIGNED_UPLOADS, true],
    ["streamable_http", SIGNED_UPLOADS, true],
    ["stdio", SIGNED_UPLOADS, false],
    ["websocket", SIGNED_UPLOADS, false],
    ["sse", [], false],
    ["sse", null, false],
  ])(
    "sends signed_uploads on create only for sse and streamable_http with entries",
    (transport, signedUploads, expected) => {
      const payload = buildMcpServerPayload({
        name: "new_mcp",
        transport,
        description: "",
        config: { url: "https://mcp.example.com/mcp" },
        user_env: {},
        runtime_input_schema: null,
        runtime_bindings: null,
        allow_delegated_authorization: false,
        signed_uploads: signedUploads,
      })

      expect("signed_uploads" in payload).toBe(expected)
      if (expected) expect(payload.signed_uploads).toEqual(SIGNED_UPLOADS)
    },
  )

  it("never sends signed_uploads_error", () => {
    const created = buildMcpServerPayload({
      name: "new_mcp",
      transport: "sse",
      config: { url: "https://mcp.example.com/mcp" },
      signed_uploads: SIGNED_UPLOADS,
      signed_uploads_error: SETTINGS_ERROR,
    })
    expect("signed_uploads_error" in created).toBe(false)

    const detail = ownerDetail({
      signed_uploads: BAD_SIGNED_UPLOADS,
      signed_uploads_error: SETTINGS_ERROR,
    })
    const { formData } = mcpServerDetailToEditState(detail)
    const edited = buildMcpServerPayload({ ...formData, signed_uploads: [] }, detail)
    expect("signed_uploads_error" in edited).toBe(false)
  })

  it.each([
    ["a list with a non-object", [1]],
    ["an object", {}],
  ])("rejects signed_uploads form data that is not a list of objects", (_name, value) => {
    const detail = ownerDetail({ signed_uploads: SIGNED_UPLOADS })
    const { formData } = mcpServerDetailToEditState(detail)

    expect(() => buildMcpServerPayload({ ...formData, signed_uploads: value }, detail)).toThrow(
      "Invalid MCP server form data: signed_uploads",
    )
  })
})

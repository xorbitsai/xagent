import { readFileSync } from "node:fs"
import { beforeEach, describe, expect, it, vi } from "vitest"

vi.mock("./api-wrapper", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./api-wrapper")>()),
  apiRequest: vi.fn(),
}))

import { apiRequest } from "./api-wrapper"
import {
  getUserDefaultModels,
  getUserModels,
  hostnameFromUrl,
  parseModelList,
  parseUserDefaultModels,
  resolveTaskLlmSelection,
} from "./models"

const mockedRequest = vi.mocked(apiRequest)

const currentBackendDefaultTypes = [
  "general",
  "small_fast",
  "visual",
  "compact",
  "embedding",
  "image",
  "image_edit",
  "video",
  "asr",
  "tts",
  "speech",
  "sound_effect",
  "music",
  "rerank",
] as const

const model = (overrides: Record<string, unknown> = {}) => ({
  id: 1,
  model_id: "gpt",
  category: "llm",
  model_provider: "openai",
  model_name: "GPT",
  base_url: null,
  temperature: null,
  context_window: null,
  dimension: null,
  abilities: null,
  description: null,
  created_at: null,
  updated_at: null,
  is_active: true,
  is_owner: true,
  can_edit: true,
  can_delete: true,
  is_shared: false,
  ...overrides,
})

const defaultEntry = (
  configType: string,
  modelOverrides: Record<string, unknown> = {},
  overrides: Record<string, unknown> = {},
) => ({
  id: 2,
  user_id: 3,
  model_id: 1,
  config_type: configType,
  created_at: null,
  updated_at: null,
  model: model(modelOverrides),
  ...overrides,
})

const jsonResponse = (value: unknown, init?: ResponseInit) => new Response(JSON.stringify(value), init)

const unreadableResponse = () => {
  const response = new Response("unreadable")
  Object.defineProperty(response, "text", {
    value: vi.fn().mockRejectedValue(new Error("body unavailable")),
  })
  return response
}

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
}

beforeEach(() => {
  mockedRequest.mockReset()
})

describe("hostnameFromUrl", () => {
  it("returns empty string for missing url", () => {
    expect(hostnameFromUrl(undefined)).toBe("")
    expect(hostnameFromUrl(null)).toBe("")
    expect(hostnameFromUrl("")).toBe("")
  })

  it("keeps non-standard ports so same-host models stay distinguishable", () => {
    expect(hostnameFromUrl("http://localhost:9997/v1")).toBe("localhost:9997")
    expect(hostnameFromUrl("http://localhost:11434")).toBe("localhost:11434")
  })

  it("drops default ports per URL semantics", () => {
    expect(hostnameFromUrl("https://api.openai.com/v1")).toBe("api.openai.com")
  })

  it("returns empty string for non-absolute urls instead of a garbled fallback", () => {
    expect(hostnameFromUrl("/v1")).toBe("")
    expect(hostnameFromUrl("not a url")).toBe("")
  })
})

describe("model producer decoders", () => {
  it("copies complete producer records and ignores unconsumed extras", () => {
    const raw = model({
      abilities: ["chat"],
      temperature: 0.7,
      context_window: 128000,
      dimension: 1536,
      is_default: true,
    })

    const parsed = parseModelList([raw])

    expect(parsed).toEqual([expect.objectContaining({ model_id: "gpt", abilities: ["chat"] })])
    expect(parsed?.[0]).not.toBe(raw)
    expect(parsed?.[0]).not.toHaveProperty("is_default")
    expect(Object.keys(parsed?.[0] ?? {}).sort()).toEqual([
      "abilities",
      "base_url",
      "can_delete",
      "can_edit",
      "category",
      "context_window",
      "created_at",
      "description",
      "dimension",
      "id",
      "is_active",
      "is_owner",
      "is_shared",
      "model_id",
      "model_name",
      "model_provider",
      "temperature",
      "updated_at",
    ])
    ;(raw.abilities as unknown as string[]).push("mutated")
    expect(parsed?.[0].abilities).toEqual(["chat"])
  })

  it.each([
    [null],
    [{}],
    ["model"],
    [42],
    [true],
    [[model({ id: 0 })]],
    [[model({ id: 1.5 })]],
    [[model({ id: Number.MAX_SAFE_INTEGER + 1 })]],
    [[model({ model_id: 1 })]],
    [[model({ temperature: Infinity })]],
    [[model({ context_window: 1.5 })]],
    [[model({ abilities: ["chat", 1] })]],
    [[model({ is_shared: "false" })]],
  ])("rejects malformed model-list body %#", (value) => {
    expect(parseModelList(value)).toBeNull()
  })

  it("rejects every missing or wrong consumed ModelWithAccessInfo field", () => {
    const requiredStrings = ["model_id", "category", "model_provider", "model_name"]
    const nullableStrings = ["base_url", "description", "created_at", "updated_at"]
    const nullableFiniteNumbers = ["temperature"]
    const nullableIntegers = ["context_window", "dimension"]
    const booleans = ["is_active", "is_owner", "can_edit", "can_delete", "is_shared"]

    for (const value of ["1", 0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1, NaN, Infinity]) {
      expect(parseModelList([model({ id: value })])).toBeNull()
    }
    const missingId = model() as Record<string, unknown>
    delete missingId.id
    expect(parseModelList([missingId])).toBeNull()

    for (const field of requiredStrings) {
      expect(parseModelList([model({ [field]: 1 })])).toBeNull()
      const missing = model() as Record<string, unknown>
      delete missing[field]
      expect(parseModelList([missing])).toBeNull()
    }

    for (const field of nullableStrings) {
      expect(parseModelList([model({ [field]: 1 })])).toBeNull()
      expect(parseModelList([model({ [field]: undefined })])).toBeNull()
    }

    for (const field of nullableFiniteNumbers) {
      expect(parseModelList([model({ [field]: "0.7" })])).toBeNull()
      for (const value of [NaN, Infinity, -Infinity]) {
        expect(parseModelList([model({ [field]: value })])).toBeNull()
      }
      expect(parseModelList([model({ [field]: undefined })])).toBeNull()
    }

    for (const field of nullableIntegers) {
      expect(parseModelList([model({ [field]: "1" })])).toBeNull()
      expect(parseModelList([model({ [field]: 1.5 })])).toBeNull()
      expect(parseModelList([model({ [field]: NaN })])).toBeNull()
      expect(parseModelList([model({ [field]: Infinity })])).toBeNull()
      expect(parseModelList([model({ [field]: undefined })])).toBeNull()
    }

    for (const value of ["chat", ["chat", 1], 1, undefined]) {
      expect(parseModelList([model({ abilities: value })])).toBeNull()
    }

    for (const field of booleans) {
      expect(parseModelList([model({ [field]: "true" })])).toBeNull()
      expect(parseModelList([model({ [field]: undefined })])).toBeNull()
    }
  })

  it("indexes the raw default array by config type, copies nested models, and recognizes rerank", () => {
    const raw = defaultEntry("rerank", { model_name: "Reranker", abilities: ["rank"] })
    const parsed = parseUserDefaultModels([raw])

    expect(parsed).toEqual({
      rerank: expect.objectContaining({
        config_type: "rerank",
        model: expect.objectContaining({ model_name: "Reranker", abilities: ["rank"] }),
      }),
    })
    expect(parsed?.rerank).not.toBe(raw)
    expect(parsed?.rerank?.model).not.toBe(raw.model)
    expect(Object.keys(parsed?.rerank ?? {}).sort()).toEqual([
      "config_type",
      "created_at",
      "id",
      "model",
      "model_id",
      "updated_at",
      "user_id",
    ])
    ;(raw.model.abilities as unknown as string[]).push("mutated")
    expect(parsed?.rerank?.model.abilities).toEqual(["rank"])
  })

  it.each(currentBackendDefaultTypes)("indexes current backend-known default type %s", (configType) => {
    const parsed = parseUserDefaultModels([
      defaultEntry(configType, { model_id: `${configType}-model` }),
    ])

    expect(parsed).toEqual({
      [configType]: expect.objectContaining({
        config_type: configType,
        model: expect.objectContaining({ model_id: `${configType}-model` }),
      }),
    })
  })

  it.each(["id", "user_id", "model_id"] as const)("rejects every invalid %s", (field) => {
    for (const value of ["1", 0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1]) {
      expect(parseUserDefaultModels([defaultEntry("general", {}, { [field]: value })])).toBeNull()
    }
  })

  it("rejects every missing or wrong top-level user-default field", () => {
    const positiveIds = ["id", "user_id", "model_id"]
    for (const field of positiveIds) {
      expect(parseUserDefaultModels([defaultEntry("general", {}, { [field]: "1" })])).toBeNull()
      const missing = defaultEntry("general") as Record<string, unknown>
      delete missing[field]
      expect(parseUserDefaultModels([missing])).toBeNull()
    }

    expect(parseUserDefaultModels([defaultEntry("general", {}, { config_type: 1 })])).toBeNull()
    const missingConfigType = defaultEntry("general") as Record<string, unknown>
    delete missingConfigType.config_type
    expect(parseUserDefaultModels([missingConfigType])).toBeNull()

    for (const field of ["created_at", "updated_at"]) {
      expect(parseUserDefaultModels([defaultEntry("general", {}, { [field]: false })])).toBeNull()
      const missing = defaultEntry("general") as Record<string, unknown>
      delete missing[field]
      expect(parseUserDefaultModels([missing])).toBeNull()
    }

    expect(parseUserDefaultModels([defaultEntry("general", {}, { model: [] })])).toBeNull()
    const missingModel = defaultEntry("general") as Record<string, unknown>
    delete missingModel.model
    expect(parseUserDefaultModels([missingModel])).toBeNull()
  })

  it.each([
    [null],
    [{}],
    ["defaults"],
    [1],
    [true],
    [[defaultEntry("general", { category: 1 })]],
    [[defaultEntry("general", {}, { config_type: 1 })]],
    [[defaultEntry("general", {}, { created_at: 1 })]],
  ])("rejects malformed user-default body %#", (value) => {
    expect(parseUserDefaultModels(value)).toBeNull()
  })

  it("ignores valid unknown config types only after validating their full record", () => {
    expect(parseUserDefaultModels([defaultEntry("future_default")])).toEqual({})
    expect(parseUserDefaultModels([defaultEntry("future_default", { model_id: 1 })])).toBeNull()
  })

  it("rejects mixed arrays and duplicate recognized defaults", () => {
    expect(parseModelList([model(), model({ category: 1 })])).toBeNull()
    expect(parseUserDefaultModels([defaultEntry("general"), defaultEntry("visual", { category: 1 })])).toBeNull()
    expect(parseUserDefaultModels([defaultEntry("general"), defaultEntry("general", { model_id: "next" })])).toBeNull()
  })
})

describe("model readers", () => {
  it("uses exact filtered and unfiltered list URLs and enforces a requested category", async () => {
    mockedRequest.mockResolvedValueOnce(jsonResponse([model()]))
    await expect(getUserModels({ category: "llm" })).resolves.toEqual([expect.objectContaining({ category: "llm" })])
    expect(mockedRequest).toHaveBeenLastCalledWith("/api/models/?category=llm")

    mockedRequest.mockResolvedValueOnce(jsonResponse([model()]))
    await expect(getUserModels()).resolves.toEqual([expect.objectContaining({ model_id: "gpt" })])
    expect(mockedRequest).toHaveBeenLastCalledWith("/api/models/")

    mockedRequest.mockResolvedValueOnce(jsonResponse([model({ category: "embedding" })]))
    await expect(getUserModels({ category: "llm" })).rejects.toThrow("Invalid models response")
  })

  it("accepts a valid HTTP-200 JSON empty model collection", async () => {
    mockedRequest.mockResolvedValueOnce(jsonResponse([]))
    await expect(getUserModels({ category: "llm" })).resolves.toEqual([])
  })

  it("accepts a valid HTTP-200 JSON empty default collection", async () => {
    mockedRequest.mockResolvedValueOnce(jsonResponse([]))
    await expect(getUserDefaultModels()).resolves.toEqual({})
  })

  it.each([
    ["primitive", new Response("true")],
    ["object", jsonResponse({ model: model() })],
    ["empty", new Response("")],
    ["malformed JSON", new Response("{")],
    ["unreadable", unreadableResponse()],
    ["non-OK", jsonResponse([model()], { status: 500 })],
    ["schema-invalid", jsonResponse([model({ category: 1 })])],
  ])("rejects %s model responses through the real body parser", async (_name, response) => {
    mockedRequest.mockResolvedValueOnce(response)
    await expect(getUserModels()).rejects.toThrow()
  })

  it.each([
    ["primitive", new Response("true")],
    ["object", jsonResponse({ general: defaultEntry("general") })],
    ["empty", new Response("")],
    ["malformed JSON", new Response("{")],
    ["unreadable", unreadableResponse()],
    ["non-OK", jsonResponse([defaultEntry("general")], { status: 500 })],
    ["nested model malformed", jsonResponse([defaultEntry("general", { is_active: "yes" })])],
  ])("rejects %s default responses through the real body parser", async (_name, response) => {
    mockedRequest.mockResolvedValueOnce(response)
    await expect(getUserDefaultModels()).rejects.toThrow()
  })
})

describe("task LLM selection", () => {
  it("starts both reads before either settles and waits for higher-priority defaults", async () => {
    const models = deferred<Response>()
    const defaults = deferred<Response>()
    mockedRequest.mockReturnValueOnce(models.promise).mockReturnValueOnce(defaults.promise)

    const selection = resolveTaskLlmSelection()
    expect(mockedRequest).toHaveBeenCalledTimes(2)
    expect(mockedRequest).toHaveBeenNthCalledWith(1, "/api/models/?category=llm")
    expect(mockedRequest).toHaveBeenNthCalledWith(2, "/api/models/user-default")

    models.resolve(jsonResponse([model({ model_id: "models-first" })]))
    await Promise.resolve()
    let settled = false
    void selection.then(() => { settled = true })
    await Promise.resolve()
    expect(settled).toBe(false)

    defaults.resolve(jsonResponse([defaultEntry("general", { model_id: "defaults-win" })]))
    await expect(selection).resolves.toEqual({
      kind: "success",
      llmIds: ["defaults-win", null, null, null],
    })
    expect(mockedRequest).toHaveBeenCalledTimes(2)
  })

  it("uses defaults general even when the model reader fails", async () => {
    mockedRequest
      .mockRejectedValueOnce(new Error("models down"))
      .mockResolvedValueOnce(jsonResponse([defaultEntry("general", { model_id: "default" })]))

    await expect(resolveTaskLlmSelection()).resolves.toEqual({
      kind: "success",
      llmIds: ["default", null, null, null],
    })
  })

  it.each([
    ["non-OK", jsonResponse([model()], { status: 500 })],
    ["malformed", new Response("{")],
  ])("uses defaults general when Models returns %s", async (_name, modelsResponse) => {
    mockedRequest
      .mockResolvedValueOnce(modelsResponse)
      .mockResolvedValueOnce(jsonResponse([defaultEntry("general", { model_id: "default" })]))

    await expect(resolveTaskLlmSelection()).resolves.toEqual({
      kind: "success",
      llmIds: ["default", null, null, null],
    })
  })

  it("uses the first non-empty response-order model when defaults fail", async () => {
    mockedRequest
      .mockResolvedValueOnce(jsonResponse([
        model({ model_id: "" }),
        model({ model_id: "first" }),
        model({ model_id: "later", is_default: true }),
      ]))
      .mockRejectedValueOnce(new Error("defaults down"))

    await expect(resolveTaskLlmSelection()).resolves.toEqual({
      kind: "success",
      llmIds: ["first", null, null, null],
    })
  })

  it("uses a fulfilled Models fallback while retaining fulfilled specialized defaults", async () => {
    mockedRequest
      .mockResolvedValueOnce(jsonResponse([model({ model_id: "  fallback\t" })]))
      .mockResolvedValueOnce(jsonResponse([
        defaultEntry("small_fast", { model_id: "\tsmall fast  " }),
        defaultEntry("visual", { model_id: "  visual\t" }),
        defaultEntry("compact", { model_id: " compact " }),
      ]))

    await expect(resolveTaskLlmSelection()).resolves.toEqual({
      kind: "success",
      llmIds: ["  fallback\t", "\tsmall fast  ", "  visual\t", " compact "],
    })
  })

  it("preserves a whitespace-bearing Defaults general ID byte-for-byte", async () => {
    mockedRequest
      .mockResolvedValueOnce(jsonResponse([model({ model_id: "fallback" })]))
      .mockResolvedValueOnce(jsonResponse([
        defaultEntry("general", { model_id: "  default general\t" }),
      ]))

    await expect(resolveTaskLlmSelection()).resolves.toEqual({
      kind: "success",
      llmIds: ["  default general\t", null, null, null],
    })
  })

  it.each([
    ["non-OK", jsonResponse([defaultEntry("general")], { status: 500 })],
    ["malformed", new Response("{")],
  ])("uses the model fallback when Defaults returns %s", async (_name, defaultsResponse) => {
    mockedRequest
      .mockResolvedValueOnce(jsonResponse([model({ model_id: "fallback" })]))
      .mockResolvedValueOnce(defaultsResponse)

    await expect(resolveTaskLlmSelection()).resolves.toEqual({
      kind: "success",
      llmIds: ["fallback", null, null, null],
    })
  })

  it("projects specialized defaults only from a fulfilled source and maps empty IDs to null", async () => {
    mockedRequest
      .mockResolvedValueOnce(jsonResponse([model({ model_id: "fallback" })]))
      .mockResolvedValueOnce(jsonResponse([
        defaultEntry("general", { model_id: "general" }),
        defaultEntry("small_fast", { model_id: "" }),
        defaultEntry("visual", { model_id: "vision" }),
        defaultEntry("compact", { model_id: "" }),
      ]))

    await expect(resolveTaskLlmSelection()).resolves.toEqual({
      kind: "success",
      llmIds: ["general", null, "vision", null],
    })
  })

  it.each([
    ["defaults", async () => jsonResponse([]), async () => { throw new Error("defaults down") }],
    ["models", async () => { throw new Error("models down") }, async () => jsonResponse([])],
    ["both", async () => { throw new Error("models down") }, async () => { throw new Error("defaults down") }],
  ])("returns an operational error when %s fails and no candidate remains", async (_name, first, second) => {
    mockedRequest.mockImplementationOnce(first)
    mockedRequest.mockImplementationOnce(second)
    await expect(resolveTaskLlmSelection()).resolves.toMatchObject({ kind: "operational_error" })
  })

  it("returns no_model only when both valid readers prove there is no candidate", async () => {
    mockedRequest.mockResolvedValueOnce(jsonResponse([model({ model_id: "" })]))
    mockedRequest.mockResolvedValueOnce(jsonResponse([defaultEntry("general", { model_id: "" })]))
    await expect(resolveTaskLlmSelection()).resolves.toEqual({ kind: "no_model" })
  })

  it("returns no_model when both valid readers return empty JSON collections", async () => {
    mockedRequest
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValueOnce(jsonResponse([]))

    await expect(resolveTaskLlmSelection()).resolves.toEqual({ kind: "no_model" })
  })

  it.each([
    [
      "Models",
      jsonResponse([]),
      jsonResponse([defaultEntry("general", { model_id: "default candidate" })]),
      ["default candidate", null, null, null],
    ],
    [
      "Defaults",
      jsonResponse([model({ model_id: "model candidate" })]),
      jsonResponse([]),
      ["model candidate", null, null, null],
    ],
  ])("allows an empty valid %s source when its sibling has a candidate", async (_emptySource, modelsResponse, defaultsResponse, llmIds) => {
    mockedRequest
      .mockResolvedValueOnce(modelsResponse)
      .mockResolvedValueOnce(defaultsResponse)

    await expect(resolveTaskLlmSelection()).resolves.toEqual({
      kind: "success",
      llmIds,
    })
  })
})

describe("Home and Build model-reader architecture", () => {
  it("keeps endpoint decoding out of pages and imports only the shared resolver", () => {
    const home = readFileSync(`${process.cwd()}/src/app/page.tsx`, "utf8")
    const build = readFileSync(`${process.cwd()}/src/app/build/page.tsx`, "utf8")

    for (const source of [home, build]) {
      expect(source).not.toContain("/api/models")
      expect(source).not.toMatch(/\b(?:LlmModel|DefaultModelRecord|resolveTaskLlmIds|parseModelList|parseUserDefaultModels)\b/)
      expect(source).not.toMatch(/\b(?:getUserModels|getUserDefaultModels)\b/)
      expect(source).toMatch(/import\s*\{\s*resolveTaskLlmSelection\s*\}\s*from\s*["']@\/lib\/models["']/)
      expect(source).not.toMatch(/import\s*\{[^}]*\b(?:getUserModels|getUserDefaultModels|parseModelList|parseUserDefaultModels)\b[^}]*\}\s*from\s*["']@\/lib\/models["']/)
      expect(source).toContain("parseApiResponse")
    }
  })
})

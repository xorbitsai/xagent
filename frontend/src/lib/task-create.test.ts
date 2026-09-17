import { describe, expect, it } from "vitest"
import { normalizeTaskPromptTitle, parseTaskCreateCore } from "./task-create"

const validCore = (overrides: Record<string, unknown> = {}) => ({
  task_id: 7,
  title: "task",
  status: "running",
  created_at: "2026-01-01T00:00:00Z",
  ...overrides,
})

describe("task create core contract", () => {
  it("returns only a copied camelCase core projection", () => {
    const raw = validCore({ extra: true, nested: { retained: false } })
    const parsed = parseTaskCreateCore(raw)

    expect(parsed).toEqual({
      taskId: 7,
      title: "task",
      status: "running",
      createdAt: "2026-01-01T00:00:00Z",
    })
    expect(parsed).not.toBe(raw)
    expect(parsed).not.toHaveProperty("extra")
    expect(parsed).not.toHaveProperty("nested")
    raw.title = "mutated"
    expect(parsed?.title).toBe("task")
  })

  it.each(["task_id", "title", "status", "created_at"])("rejects missing core field %s", (field) => {
    const payload = validCore()
    delete payload[field as keyof typeof payload]
    expect(parseTaskCreateCore(payload)).toBeNull()
  })

  it.each([
    ["id alias", validCore({ task_id: undefined, id: 7 })],
    ["numeric string", validCore({ task_id: "7" })],
    ["zero", validCore({ task_id: 0 })],
    ["negative", validCore({ task_id: -1 })],
    ["fraction", validCore({ task_id: 1.5 })],
    ["unsafe integer", validCore({ task_id: Number.MAX_SAFE_INTEGER + 1 })],
    ["null id", validCore({ task_id: null })],
    ["array id", validCore({ task_id: [] })],
    ["object id", validCore({ task_id: {} })],
    ["boolean id", validCore({ task_id: true })],
    ["undefined id", validCore({ task_id: undefined })],
  ])("rejects %s", (_name, value) => {
    expect(parseTaskCreateCore(value)).toBeNull()
  })

  it.each(["title", "status", "created_at"] as const)("rejects every non-string %s", (field) => {
    for (const value of [null, [], {}, true, false, 1, undefined]) {
      expect(parseTaskCreateCore(validCore({ [field]: value }))).toBeNull()
    }
  })

  it("accepts empty strings because the backend contract requires types, not minimum lengths", () => {
    expect(parseTaskCreateCore(validCore({ title: "", status: "", created_at: "" }))).toEqual({
      taskId: 7,
      title: "",
      status: "",
      createdAt: "",
    })
  })

  it.each([null, [], {}, true, false, "task", 1, undefined])("rejects non-record JSON value %#", (value) => {
    expect(parseTaskCreateCore(value)).toBeNull()
  })

  it("normalizes Unicode and internal whitespace without truncation", () => {
    const long = "x".repeat(10_000)
    expect(normalizeTaskPromptTitle("\u00a0\t hello\n\u2003world \r\n ")).toBe("hello world")
    expect(normalizeTaskPromptTitle(`  ${long}  `)).toBe(long)
  })
})

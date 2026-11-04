import { describe, expect, it } from "vitest"

import {
  PROMPT_CONTEXT_DESCRIPTION_MAX_LENGTH,
  PROMPT_CONTEXT_MAX_FIELDS,
  draftsFromDeclarations,
  newPromptContextFieldDraft,
  promptContextFieldsKey,
  toPromptContextFieldsPayload,
  validatePromptContextFields,
  type PromptContextFieldDraft,
} from "./prompt-context-fields"

const draft = (name: string, maxLength = "100", description = ""): PromptContextFieldDraft => ({
  name,
  maxLength,
  description,
})
const rowErrors = (drafts: PromptContextFieldDraft[]) => validatePromptContextFields(drafts).rows

describe("validatePromptContextFields", () => {
  it.each([
    "a",
    "_",
    "customer.tier",
    "order-id",
    "A_b.c-9",
    `a${"b".repeat(63)}`,
  ])("accepts the server-valid name %s", (name) => {
    const result = validatePromptContextFields([draft(name)])
    expect(result.valid).toBe(true)
    expect(result.rows).toEqual([{}])
  })

  it.each([
    "1tier",
    ".tier",
    "-tier",
    "customer tier",
    "customer/tier",
    "客户",
    `a${"b".repeat(64)}`,
  ])("rejects the server-invalid name %s", (name) => {
    expect(rowErrors([draft(name)])).toEqual([{ name: "invalid" }])
  })

  it("trims surrounding whitespace before matching, as the payload does", () => {
    expect(rowErrors([draft("  tier  ")])).toEqual([{}])
    expect(toPromptContextFieldsPayload([draft("  tier  ", " 12 ")])).toEqual([
      { name: "tier", max_length: 12, description: null },
    ])
  })

  it("flags a blank name as required", () => {
    expect(rowErrors([draft("   ")])).toEqual([{ name: "required" }])
  })

  it("flags every row that shares a name, case-sensitively", () => {
    expect(rowErrors([draft("tier"), draft("Tier"), draft(" tier")])).toEqual([
      { name: "duplicate" },
      {},
      { name: "duplicate" },
    ])
  })

  it.each(["1", "1024", "007"])("accepts the max length %s", (maxLength) => {
    expect(rowErrors([draft("tier", maxLength)])).toEqual([{}])
  })

  it.each(["0", "1025", "-1", "1.5", "1e3", "", "abc"])(
    "rejects the max length %j",
    (maxLength) => {
      expect(rowErrors([draft("tier", maxLength)])).toEqual([{ maxLength: "invalid" }])
    },
  )

  it("rejects more than the server limit of declarations", () => {
    const atLimit = Array.from({ length: PROMPT_CONTEXT_MAX_FIELDS }, (_, i) => draft(`f${i}`))
    expect(validatePromptContextFields(atLimit)).toMatchObject({ valid: true, tooMany: false })

    const overLimit = [...atLimit, draft("extra")]
    expect(validatePromptContextFields(overLimit)).toMatchObject({ valid: false, tooMany: true })
  })

  it("sends a trimmed description, or null when blank", () => {
    expect(toPromptContextFieldsPayload([draft("tier", "8", "  VIP or standard  "), draft("plan", "8", "   ")])).toEqual([
      { name: "tier", max_length: 8, description: "VIP or standard" },
      { name: "plan", max_length: 8, description: null },
    ])
  })

  it("limits a description to the server's length in code points, after trimming", () => {
    const atLimit = "a".repeat(PROMPT_CONTEXT_DESCRIPTION_MAX_LENGTH)
    expect(rowErrors([draft("tier", "8", ` ${atLimit} `)])).toEqual([{}])
    expect(rowErrors([draft("tier", "8", `${atLimit}a`)])).toEqual([{ description: "tooLong" }])
    // 200 emoji are 400 UTF-16 units but 200 characters to the server.
    expect(rowErrors([draft("tier", "8", "😀".repeat(PROMPT_CONTEXT_DESCRIPTION_MAX_LENGTH))])).toEqual([{}])
  })

  it("strips the same surrounding whitespace as the server's str.strip()", () => {
    expect(rowErrors([draft("tier", "8", "\u001fVIP\u0085")])).toEqual([{}])
    expect(toPromptContextFieldsPayload([draft("tier", "8", "\u001fVIP\u0085 ")])[0].description).toBe("VIP")
    expect(toPromptContextFieldsPayload([draft("tier", "8", "\ufeffVIP")])[0].description).toBe("\ufeffVIP")
    expect(toPromptContextFieldsPayload([draft("tier", "8", "\u3000\u2028")])[0].description).toBeNull()
  })

  it.each(["line\nbreak", "tab\there", "back`tick", "nul\u0000", "del\u007f", "c1\u0085c1"])(
    "rejects the description %j",
    (description) => {
      expect(rowErrors([draft("tier", "8", description)])).toEqual([{ description: "invalidChars" }])
    },
  )

  it("treats an empty list as valid", () => {
    expect(validatePromptContextFields([])).toEqual({ rows: [], tooMany: false, valid: true })
  })
})

describe("draftsFromDeclarations", () => {
  it("round-trips server declarations", () => {
    const declarations = [
      { name: "customer.tier", max_length: 32, description: "Membership level, VIP or standard" },
      { name: "order_id", max_length: 1024, description: null },
    ]
    expect(toPromptContextFieldsPayload(draftsFromDeclarations(declarations))).toEqual(declarations)
  })

  it("returns no rows for a missing or malformed value", () => {
    expect(draftsFromDeclarations(undefined)).toEqual([])
    expect(draftsFromDeclarations(null)).toEqual([])
    expect(draftsFromDeclarations({ name: "x", max_length: 1 })).toEqual([])
    expect(
      draftsFromDeclarations([null, "x", { name: 1, max_length: 1 }, { name: "ok", max_length: 5 }]),
    ).toEqual([draft("ok", "5")])
    expect(draftsFromDeclarations([{ name: "ok", max_length: 5, description: null }])).toEqual([draft("ok", "5")])
  })
})

describe("promptContextFieldsKey", () => {
  it("compares valid limits by value and invalid ones by their text", () => {
    expect(promptContextFieldsKey([draft("a", "007")])).toBe(promptContextFieldsKey([draft("a", "7")]))
    expect(promptContextFieldsKey([draft("a", "07.0")])).not.toBe(promptContextFieldsKey([draft("a", "7")]))
    expect(promptContextFieldsKey([draft("a", "1e3")])).not.toBe(promptContextFieldsKey([draft("a", "1E3")]))
  })

  it("ignores surrounding whitespace but keeps order significant", () => {
    expect(promptContextFieldsKey([draft(" a ", " 5", " d ")])).toBe(promptContextFieldsKey([draft("a", "5", "d")]))
    expect(promptContextFieldsKey([draft("a", "5", "d")])).not.toBe(promptContextFieldsKey([draft("a", "5")]))
    expect(promptContextFieldsKey([draft("a", "5", "\u001fd\u0085")])).toBe(promptContextFieldsKey([draft("a", "5", "d")]))
    expect(promptContextFieldsKey([draft("a"), draft("b")])).not.toBe(
      promptContextFieldsKey([draft("b"), draft("a")]),
    )
  })
})

it("starts a new row blank with a valid default limit", () => {
  expect(rowErrors([newPromptContextFieldDraft()])).toEqual([{ name: "required" }])
})

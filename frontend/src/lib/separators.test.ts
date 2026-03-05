import { describe, it, expect } from "vitest"
import { parseSeparatorsInput } from "./separators"

describe("parseSeparatorsInput", () => {
  it("returns empty array for empty or whitespace-only input", () => {
    expect(parseSeparatorsInput("")).toEqual([])
    expect(parseSeparatorsInput("   ")).toEqual([])
    expect(parseSeparatorsInput("\t\n")).toEqual([])
  })

  it("splits by comma", () => {
    expect(parseSeparatorsInput("a,b,c")).toEqual(["a", "b", "c"])
  })

  it("unescapes \\n to newline", () => {
    expect(parseSeparatorsInput("\\n")).toEqual(["\n"])
  })

  it("unescapes \\t to tab", () => {
    expect(parseSeparatorsInput("\\t")).toEqual(["\t"])
  })

  it("unescapes \\r to carriage return", () => {
    expect(parseSeparatorsInput("\\r")).toEqual(["\r"])
  })

  it("unescapes \\, to literal comma", () => {
    expect(parseSeparatorsInput("\\,")).toEqual([","])
  })

  it("unescapes \\\\ to backslash", () => {
    expect(parseSeparatorsInput("\\\\")).toEqual(["\\"])
  })

  it("treats \\, as literal comma so item can contain comma", () => {
    expect(parseSeparatorsInput("a\\,b")).toEqual(["a,b"])
  })

  it("parses \\n\\n,\\n,. as three items", () => {
    expect(parseSeparatorsInput("\\n\\n,\\n,。")).toEqual(["\n\n", "\n", "。"])
  })

  it("trims each item and drops empty items", () => {
    expect(parseSeparatorsInput(" a , b , ")).toEqual(["a", "b"])
  })

  it("returns [] for null/undefined-like (empty string)", () => {
    expect(parseSeparatorsInput("")).toEqual([])
  })

  it("single character items", () => {
    expect(parseSeparatorsInput("a")).toEqual(["a"])
  })

  it("mixed escape and normal characters", () => {
    expect(parseSeparatorsInput("a\\nb")).toEqual(["a\nb"])
  })

  it("multiple items with escapes", () => {
    expect(parseSeparatorsInput("\\n\\n,\\n, ,\\t")).toEqual(["\n\n", "\n", "\t"])
  })
})

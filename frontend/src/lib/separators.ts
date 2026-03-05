/**
 * Parse a single-line separators input (comma-separated with escape support)
 * into a list of strings for chunk splitting.
 *
 * Rules:
 * - Split by comma, but \, is treated as literal comma (not list separator).
 * - Per-item escapes: \n -> newline, \t -> tab, \r -> carriage return,
 *   \, -> comma, \\ -> backslash.
 * - Trim each item; empty items are dropped.
 */
export function parseSeparatorsInput(input: string): string[] {
  if (!input || typeof input !== "string") return []

  const trimmed = input.trim()
  if (!trimmed) return []

  // Replace \, with placeholder so we split only by unescaped comma
  const PLACEHOLDER = "\u0000"
  const withPlaceholder = trimmed.replace(/\\,/g, PLACEHOLDER)
  const rawItems = withPlaceholder.split(",").map((item) => item.trim())

  const result: string[] = []
  for (const item of rawItems) {
    const withComma = item.replace(new RegExp(PLACEHOLDER, "g"), ",")
    const unescaped = unescapeSeparatorItem(withComma)
    if (unescaped !== "") result.push(unescaped)
  }
  return result
}

function unescapeSeparatorItem(item: string): string {
  const out: string[] = []
  for (let i = 0; i < item.length; i++) {
    if (item[i] === "\\" && i + 1 < item.length) {
      const next = item[i + 1]
      switch (next) {
        case "n":
          out.push("\n")
          break
        case "t":
          out.push("\t")
          break
        case "r":
          out.push("\r")
          break
        case ",":
          out.push(",")
          break
        case "\\":
          out.push("\\")
          break
        default:
          out.push("\\", next)
      }
      i++
      continue
    }
    out.push(item[i])
  }
  return out.join("")
}

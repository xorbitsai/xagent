// Pure helpers that derive an agent's "plan" (a numbered/bulleted list of
// steps) from its free-form instructions text, and write edits made on the
// Flow view's Agent node back into that same text. The instructions field
// stays the single source of truth — the plan is just a projection of it.

export interface PlanStep {
  text: string
  lineIdx: number | null
  prefix: string | null
}

export interface ParsedPlan {
  steps: PlanStep[]
  explicit: boolean
}

const LIST_RE = /^(\s*(?:step\s*\d+\s*[:.)\-]|\d+\s*[.)]|[-*•])\s+)(.+)$/i

export function parseInstructionSteps(text: string): ParsedPlan {
  const lines = text.split("\n")
  const steps: PlanStep[] = []
  lines.forEach((line, i) => {
    const m = line.match(LIST_RE)
    if (m && m[2].trim()) steps.push({ text: m[2].trim(), lineIdx: i, prefix: m[1] })
  })
  const hasNumbered = steps.some((s) => /\d/.test(s.prefix || ""))
  if (steps.length && (hasNumbered || steps.length >= 2)) {
    return { steps, explicit: true }
  }

  // Strip any leading list markers (e.g. "- ", "* ") before extracting sentences
  // so a lone bullet like "- Be concise and professional" doesn't leak its marker.
  const cleaned = text.replace(/^[-*•]\s+/gm, "")

  // Avoid a lookbehind assertion here — unsupported on Safari < 16.4/macOS < 13.3
  // and would throw a SyntaxError on load in older browsers. Extract each
  // punctuation-terminated (or end-of-string-terminated) run instead of
  // splitting after the punctuation.
  // Upper bound raised to 400 so long-but-reasonable sentences are still shown.
  const sentences = (cleaned.replace(/\n+/g, " ").match(/[^.!?]+(?:[.!?]+|$)/g) || [])
    .map((s) => s.trim())
    .filter((s) => s.length >= 15 && s.length <= 400)

  return {
    steps: sentences.slice(0, 3).map((s) => ({ text: s.replace(/[.!?]+$/, ""), lineIdx: null, prefix: null })),
    explicit: false,
  }
}

// Write an explicit numbered plan. The result replaces the instructions entirely
// rather than appending a Plan: block to the original prose — appending would
// duplicate content already present in the source and silently drop any
// sentences beyond the 3-step display limit.
function writeExplicitPlan(texts: string[]): string {
  return "Plan:\n" + texts.map((t, i) => `${i + 1}. ${t}`).join("\n")
}

// Extract ALL qualifying sentences from prose text with no slice cap, used
// when materializing an inferred plan to explicit. Must use the same filter
// bounds as the display path in parseInstructionSteps (>= 15 && <= 400) so
// that display-step indices map 1-to-1 onto the materialized list — a sentence
// hidden by the upper bound in the display would otherwise sit at an unexpected
// index and cause an edit to silently overwrite the wrong sentence.
function inferSentences(text: string): string[] {
  const cleaned = text.replace(/^[-*•]\s+/gm, "")
  return (cleaned.replace(/\n+/g, " ").match(/[^.!?]+(?:[.!?]+|$)/g) || [])
    .map((s) => s.trim().replace(/[.!?]+$/, ""))
    .filter((s) => s.length >= 15 && s.length <= 400)
}

// Only renumber lines known to belong to the plan (from the caller's own
// step-index bookkeeping) rather than re-scanning the whole text with a
// second regex pass — the instructions may contain unrelated numbered
// lines outside the plan that happen to match the same shape.
// Handles both "1. text" / "1) text" and "Step 1: text" / "Step 1. text" formats.
function renumber(lines: string[], stepLineIndices: number[]): string[] {
  const indices = new Set(stepLineIndices)
  let n = 1
  return lines.map((l, i) => {
    if (!indices.has(i)) return l
    let m = l.match(/^(\s*)(step\s*)\d+(\s*[:.)\-]\s+)(.+)$/i)
    if (m) return `${m[1]}${m[2]}${n++}${m[3]}${m[4]}`
    m = l.match(/^(\s*)\d+(\s*[.)]\s+)(.+)$/)
    if (m) return `${m[1]}${n++}${m[2]}${m[3]}`
    return l
  })
}

function nextPrefix(prefix: string): string {
  let m = prefix.match(/^(\s*)step\s*(\d+)(\s*[:.)\-]\s*)$/i)
  if (m) return `${m[1]}Step ${Number(m[2]) + 1}${m[3]}`
  m = prefix.match(/^(\s*)(\d+)(\s*[.)]\s*)$/)
  if (m) return `${m[1]}${Number(m[2]) + 1}${m[3]}`
  return prefix // bullets keep the same marker
}

export function updatePlanStep(instructions: string, idx: number, newText: string): string {
  const plan = parseInstructionSteps(instructions)
  if (plan.explicit) {
    const lines = instructions.split("\n")
    const s = plan.steps[idx]
    if (s && s.lineIdx !== null && s.prefix !== null) lines[s.lineIdx] = s.prefix + newText
    return lines.join("\n")
  }
  // NOTE: continuation lines (indented detail below a step marker) are not
  // modelled — each step tracks only its marker line. Multi-line steps will
  // desync on move/edit/delete until the data model is extended to line ranges.
  const texts = inferSentences(instructions)
  if (texts.length) texts[idx] = newText
  else return writeExplicitPlan([newText])
  return writeExplicitPlan(texts)
}

export function deletePlanStep(instructions: string, idx: number): string {
  const plan = parseInstructionSteps(instructions)
  if (plan.explicit) {
    const lines = instructions.split("\n")
    const s = plan.steps[idx]
    if (s && s.lineIdx !== null) {
      const removedLineIdx = s.lineIdx
      lines.splice(removedLineIdx, 1)
      const remainingLineIndices = plan.steps
        .filter((_, i) => i !== idx)
        .map((step) => (step.lineIdx === null ? null : step.lineIdx > removedLineIdx ? step.lineIdx - 1 : step.lineIdx))
        .filter((lineIdx): lineIdx is number => lineIdx !== null)
      return renumber(lines, remainingLineIndices).join("\n")
    }
    return lines.join("\n")
  }
  const texts = inferSentences(instructions)
  texts.splice(idx, 1)
  return writeExplicitPlan(texts)
}

export function movePlanStep(instructions: string, idx: number, dir: -1 | 1): string {
  const plan = parseInstructionSteps(instructions)
  const j = idx + dir
  if (j < 0 || j >= plan.steps.length) return instructions
  if (plan.explicit) {
    const lines = instructions.split("\n")
    const a = plan.steps[idx]
    const b = plan.steps[j]
    if (a.lineIdx !== null && b.lineIdx !== null && a.prefix !== null && b.prefix !== null) {
      lines[a.lineIdx] = a.prefix + b.text
      lines[b.lineIdx] = b.prefix + a.text
    }
    return lines.join("\n")
  }
  const texts = inferSentences(instructions)
  if (j < 0 || j >= texts.length) return instructions
  const tmp = texts[idx]
  texts[idx] = texts[j]
  texts[j] = tmp
  return writeExplicitPlan(texts)
}

export function addPlanStep(instructions: string, placeholder: string): string {
  const plan = parseInstructionSteps(instructions)
  if (plan.explicit && plan.steps.length) {
    const lines = instructions.split("\n")
    const last = plan.steps[plan.steps.length - 1]
    if (last.lineIdx !== null && last.prefix !== null) {
      lines.splice(last.lineIdx + 1, 0, nextPrefix(last.prefix) + placeholder)
    }
    return lines.join("\n")
  }
  const texts = inferSentences(instructions)
  return writeExplicitPlan([...texts, placeholder])
}

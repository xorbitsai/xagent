// Mirrors the server-side PromptContextFieldDeclaration rules; the server stays
// authoritative and rejects anything that slips past these checks with a 422.
export const PROMPT_CONTEXT_FIELD_NAME_PATTERN = /^[A-Za-z_][A-Za-z0-9_.-]{0,63}$/
export const PROMPT_CONTEXT_MAX_FIELDS = 32
export const PROMPT_CONTEXT_MIN_LENGTH = 1
export const PROMPT_CONTEXT_MAX_LENGTH = 1024
export const PROMPT_CONTEXT_DEFAULT_LENGTH = 256
export const PROMPT_CONTEXT_DESCRIPTION_MAX_LENGTH = 200
const DESCRIPTION_FORBIDDEN_CHARS = /[\u0000-\u001f\u007f-\u009f`]/
// Python str.strip()'s whitespace set; JS trim() differs (\x1c-\x1f, \x85 vs \ufeff).
const PY_STRIP = /^[\t-\r\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+|[\t-\r\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+$/g

function stripDescription(value: string): string {
  return value.replace(PY_STRIP, "")
}

export interface PromptContextFieldDeclaration {
  name: string
  max_length: number
  description: string | null
}

export interface PromptContextFieldDraft {
  name: string
  maxLength: string
  description: string
}

export type PromptContextFieldNameError = "required" | "invalid" | "duplicate"

export interface PromptContextFieldRowErrors {
  name?: PromptContextFieldNameError
  maxLength?: "invalid"
  description?: "tooLong" | "invalidChars"
}

export interface PromptContextFieldsValidation {
  rows: PromptContextFieldRowErrors[]
  tooMany: boolean
  valid: boolean
}

export function newPromptContextFieldDraft(): PromptContextFieldDraft {
  return { name: "", maxLength: String(PROMPT_CONTEXT_DEFAULT_LENGTH), description: "" }
}

export function draftsFromDeclarations(value: unknown): PromptContextFieldDraft[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item) => {
    if (!item || typeof item !== "object") return []
    const { name, max_length, description } = item as Record<string, unknown>
    if (typeof name !== "string" || typeof max_length !== "number") return []
    return [{ name, maxLength: String(max_length), description: typeof description === "string" ? description : "" }]
  })
}

function parseMaxLength(raw: string): number | null {
  const trimmed = raw.trim()
  if (!/^\d+$/.test(trimmed)) return null
  const value = Number(trimmed)
  if (value < PROMPT_CONTEXT_MIN_LENGTH || value > PROMPT_CONTEXT_MAX_LENGTH) return null
  return value
}

export function validatePromptContextFields(
  drafts: PromptContextFieldDraft[],
): PromptContextFieldsValidation {
  const counts = new Map<string, number>()
  for (const draft of drafts) {
    const name = draft.name.trim()
    if (name) counts.set(name, (counts.get(name) ?? 0) + 1)
  }

  const rows = drafts.map((draft) => {
    const errors: PromptContextFieldRowErrors = {}
    const name = draft.name.trim()
    if (!name) errors.name = "required"
    else if (!PROMPT_CONTEXT_FIELD_NAME_PATTERN.test(name)) errors.name = "invalid"
    else if ((counts.get(name) ?? 0) > 1) errors.name = "duplicate"
    if (parseMaxLength(draft.maxLength) === null) errors.maxLength = "invalid"
    const description = stripDescription(draft.description)
    if (DESCRIPTION_FORBIDDEN_CHARS.test(description)) errors.description = "invalidChars"
    // Code points, not UTF-16 units, to match the server's len().
    else if ([...description].length > PROMPT_CONTEXT_DESCRIPTION_MAX_LENGTH) errors.description = "tooLong"
    return errors
  })

  const tooMany = drafts.length > PROMPT_CONTEXT_MAX_FIELDS
  const valid = !tooMany && rows.every((row) => !row.name && !row.maxLength && !row.description)
  return { rows, tooMany, valid }
}

// Only meaningful for drafts that passed validatePromptContextFields.
export function toPromptContextFieldsPayload(
  drafts: PromptContextFieldDraft[],
): PromptContextFieldDeclaration[] {
  return drafts.map((draft) => ({
    name: draft.name.trim(),
    max_length: Number(draft.maxLength.trim()),
    description: stripDescription(draft.description) || null,
  }))
}

export function promptContextFieldsKey(drafts: PromptContextFieldDraft[]): string {
  return JSON.stringify(
    drafts.map((draft) => [
      draft.name.trim(),
      // Valid limits compare as the number the server stores; invalid drafts keep their raw text.
      parseMaxLength(draft.maxLength) ?? draft.maxLength,
      stripDescription(draft.description),
    ]),
  )
}

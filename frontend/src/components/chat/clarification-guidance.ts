import type { Interaction } from "@/contexts/app-context-chat"

export const CLARIFICATION_PAGE_SIZE = 3

// These are missing-information answers, not approval or connector controls.
export const canDeferClarification = (interaction: Interaction): boolean =>
  ["text_input", "number_input", "select_one", "select_multiple", "file_upload"].includes(interaction.type)

/** A suggestion is never an answer until the user explicitly chooses it. */
export function suggestedClarificationValue(interaction: Interaction): string | number | undefined {
  const value = interaction.default_value !== undefined ? interaction.default_value : interaction.default
  if (interaction.type === "text_input") {
    return typeof value === "string" && value.trim() ? value : undefined
  }
  if (interaction.type === "number_input") {
    if (typeof value !== "number" && !(typeof value === "string" && value.trim())) return undefined
    const number = Number(value)
    return Number.isFinite(number)
      && (interaction.min === undefined || number >= interaction.min)
      && (interaction.max === undefined || number <= interaction.max)
      ? number : undefined
  }
  if (["select_one", "select_multiple", "action_cards"].includes(interaction.type)) {
    return typeof value === "string" && interaction.options?.some(option => option.value === value)
      ? value : undefined
  }
  return undefined
}

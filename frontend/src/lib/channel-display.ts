const COMPACTABLE_CHANNEL_TERMS: Record<string, string[]> = {
  telegram: ["telegram"],
  feishu: ["feishu", "lark", "飞书"],
}

const CHANNEL_TYPE_LABELS: Record<string, string> = {
  telegram: "Telegram",
  feishu: "Feishu",
}

const CHANNEL_TERM_SEPARATOR_PATTERN = "[\\s\\-_:·|]"

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
}

function getChannelTermPattern(term: string): RegExp {
  const escapedTerm = escapeRegExp(term)
  const separator = CHANNEL_TERM_SEPARATOR_PATTERN
  const edgeSeparator = /^[a-z0-9_]+$/i.test(term)
    ? `${separator}+`
    : `${separator}*`
  return new RegExp(
    `(?:^${escapedTerm}(?:${edgeSeparator}|$)|${edgeSeparator}${escapedTerm}$|${separator}+${escapedTerm}${separator}+)`,
    "gi",
  )
}

export function getCompactChannelName(
  channelName: string | null | undefined,
  channelType?: string,
): string {
  if (!channelName) return ""

  const normalizedName = channelName.trim()
  const normalizedType = channelType?.trim().toLowerCase()
  const removableTerms = normalizedType
    ? COMPACTABLE_CHANNEL_TERMS[normalizedType]
    : undefined

  if (!normalizedName || !removableTerms) return normalizedName

  const compactName = removableTerms
    .reduce(
      (name, term) =>
        name.replace(getChannelTermPattern(term), " "),
      normalizedName,
    )
    .replace(/\s+/g, " ")
    .trim()

  return compactName || normalizedName
}

export function getChannelTooltip(
  channelName: string | null | undefined,
  channelType?: string,
): string {
  if (!channelName) return ""

  const normalizedType = channelType?.trim().toLowerCase()
  if (!normalizedType) return channelName

  const typeLabel = CHANNEL_TYPE_LABELS[normalizedType] || channelType?.trim()
  return `${typeLabel} · ${channelName}`
}

/**
 * Normalize the identifiers returned for one upload operation.
 *
 * The upload response is an untrusted transport boundary: callers may only
 * attach files when every requested item has one non-empty, unique identifier.
 */
export function normalizeUploadFileIds(
  values: readonly unknown[],
  expectedCount: number,
): string[] | null {
  if (values.length !== expectedCount) return null

  const normalized: string[] = []
  const seen = new Set<string>()

  for (const value of values) {
    if (typeof value !== "string") return null
    const fileId = value.trim()
    if (!fileId || seen.has(fileId)) return null
    seen.add(fileId)
    normalized.push(fileId)
  }

  return normalized
}

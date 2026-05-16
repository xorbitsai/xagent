export function unwrapFinalAnswerContent(content: string): string {
  const text = stripCodeFence(content.trim())
  if (!text.startsWith("{") || !text.endsWith("}")) {
    return content
  }

  try {
    const parsed = JSON.parse(text)
    if (!parsed || typeof parsed !== "object") {
      return content
    }

    if (parsed.action === "final_answer") {
      return stringifyAnswerValue(
        Object.prototype.hasOwnProperty.call(parsed, "action_input")
          ? parsed.action_input
          : parsed.answer
      )
    }

    if (Object.prototype.hasOwnProperty.call(parsed, "final_answer")) {
      return stringifyAnswerValue(parsed.final_answer)
    }
  } catch {
    return content
  }

  return content
}

function stripCodeFence(content: string): string {
  const lines = content.split(/\r?\n/)
  if (lines.length >= 2 && lines[0].trim().startsWith("```")) {
    const closing = lines[lines.length - 1].trim()
    if (closing === "```" || closing.startsWith("```")) {
      return lines.slice(1, -1).join("\n").trim()
    }
  }
  return content
}

function stringifyAnswerValue(value: unknown): string {
  if (typeof value === "string") {
    return value
  }
  return JSON.stringify(value, null, 2)
}

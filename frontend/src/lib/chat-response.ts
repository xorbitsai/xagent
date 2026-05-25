export interface ParsedChatPayload {
    message?: string
    interactions?: unknown[]
}

export interface ParsedChatResponse {
    message: string
    interactions?: unknown[]
}

const readChatPayload = (value: unknown): ParsedChatPayload | null => {
    if (!value || typeof value !== "object") return null

    const chat = value as {
        message?: unknown
        interactions?: unknown
    }

    const message = typeof chat.message === "string" ? chat.message : ""
    const interactions = Array.isArray(chat.interactions)
        ? chat.interactions
        : undefined

    if (!message && !interactions) {
        return null
    }

    return { message, interactions }
}

const parseStructuredChatContent = (value: string): ParsedChatPayload | null => {
    const parseCandidate = (candidate: string): ParsedChatPayload | null => {
        try {
            const parsed = JSON.parse(candidate) as {
                type?: unknown
                chat?: {
                    message?: unknown
                    interactions?: unknown
                }
            }

            if (parsed?.type !== "chat" || !parsed.chat || typeof parsed.chat !== "object") {
                return null
            }

            const message = typeof parsed.chat.message === "string" ? parsed.chat.message : ""
            const interactions = Array.isArray(parsed.chat.interactions)
                ? parsed.chat.interactions
                : undefined

            if (!message && !interactions) {
                return null
            }

            return { message, interactions }
        } catch {
            return null
        }
    }

    const direct = parseCandidate(value)
    if (direct) return direct

    const jsonMatch = value.match(/```json\s*([\s\S]*?)\s*```/i)
    if (!jsonMatch) return null

    return parseCandidate(jsonMatch[1])
}

export const extractSharedChatResponse = (payload: unknown): ParsedChatPayload | null => {
    const root = payload && typeof payload === "object"
        ? (payload as {
            chat_response?: unknown
            result?: unknown
            metadata?: unknown
            interactions?: unknown
        })
        : null

    const directPayload = readChatPayload(root)
    if (directPayload) {
        return directPayload
    }

    const directChatResponse = readChatPayload(root?.chat_response)
    if (directChatResponse) {
        return directChatResponse
    }

    if (root?.result && typeof root.result === "object") {
        const nestedResult = extractSharedChatResponse(root.result)
        if (nestedResult) {
            return nestedResult
        }
    }

    if (typeof root?.result === "string") {
        const parsedContent = parseStructuredChatContent(root.result)
        if (parsedContent) {
            return parsedContent
        }
    }

    if (root?.metadata && typeof root.metadata === "object") {
        const metadataPayload = extractSharedChatResponse(root.metadata)
        if (metadataPayload) {
            return metadataPayload
        }
    }

    return null
}

export const extractBuildPreviewResponse = (payload: unknown): ParsedChatResponse => {
    const sharedPayload = extractSharedChatResponse(payload)
    if (sharedPayload) {
        return {
            message: sharedPayload.message || "",
            interactions: sharedPayload.interactions,
        }
    }

    const root = payload && typeof payload === "object"
        ? (payload as {
            result?: unknown
            output?: unknown
        })
        : null

    if (root?.result && typeof root.result === "object") {
        const nestedResult = root.result as { content?: unknown }
        if (typeof nestedResult.content === "string") {
            return {
                message: nestedResult.content,
            }
        }
    }

    if (typeof root?.result === "string") {
        return {
            message: root.result,
        }
    }

    return {
        message: typeof root?.output === "string" ? root.output : "",
    }
}

import { unwrapFinalAnswerContent } from "@/lib/final-answer"

type ResultMessageLike = {
  id: string
  role: string
  isResult?: boolean
}

type TraceEventLike = {
  event_id?: string
}

type WebSocketEventLike = {
  type?: unknown
  event_type?: unknown
  data?: unknown
}

export type FinalAnswerStreamEventType =
  | "final_answer_start"
  | "final_answer_delta"
  | "final_answer_end"
  | "final_answer_error"

export type FinalAnswerStreamActionPayload = {
  messageId: string
  delta?: string
  content?: string
  status: "running" | "completed" | "failed"
  timestamp: string
}

export const isStreamingFinalAnswerMessage = (message: ResultMessageLike): boolean => {
  return (
    message.role === "assistant" &&
    message.isResult === true &&
    message.id.startsWith("final_answer_")
  )
}

export const isFinalAnswerStreamEventType = (
  value: unknown,
): value is FinalAnswerStreamEventType => {
  return (
    value === "final_answer_start" ||
    value === "final_answer_delta" ||
    value === "final_answer_end" ||
    value === "final_answer_error"
  )
}

export const getWebSocketEventType = (
  message: WebSocketEventLike,
): string | undefined => {
  if (typeof message.type !== "string") {
    return undefined
  }
  if (message.type !== "trace_event") {
    return message.type
  }

  if (typeof message.event_type === "string") {
    return message.event_type
  }

  const data =
    message.data && typeof message.data === "object"
      ? (message.data as Record<string, unknown>)
      : {}
  return typeof data.event_type === "string" ? data.event_type : message.type
}

export const shouldBufferMessageForHistoricalReplay = ({
  isReplaying,
  isHistoryLoading,
  message,
}: {
  isReplaying: boolean
  isHistoryLoading: boolean
  message: WebSocketEventLike
}): boolean => {
  if (!isReplaying) {
    return false
  }

  const eventType = getWebSocketEventType(message)
  if (isFinalAnswerStreamEventType(eventType)) {
    return false
  }

  return isHistoryLoading || eventType === "historical_data_complete"
}

export const getFinalAnswerStreamMessageId = (
  value: unknown,
): string | undefined => {
  const data =
    value && typeof value === "object"
      ? (value as Record<string, unknown>)
      : {}
  const result =
    data.result && typeof data.result === "object"
      ? (data.result as Record<string, unknown>)
      : {}
  const candidates = [
    data.stream_message_id,
    data.streamMessageId,
    result.stream_message_id,
    result.streamMessageId,
  ]

  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate) {
      return candidate
    }
  }
  return undefined
}

export const mergeTraceEventsById = <T extends TraceEventLike>(
  ...eventGroups: Array<readonly T[] | undefined>
): T[] => {
  const merged: T[] = []
  const seenIds = new Set<string>()

  for (const events of eventGroups) {
    for (const event of events || []) {
      const eventId = typeof event.event_id === "string" ? event.event_id : ""
      if (eventId) {
        if (seenIds.has(eventId)) {
          continue
        }
        seenIds.add(eventId)
      }
      merged.push(event)
    }
  }

  return merged
}

export const getFinalAnswerStreamActionPayload = ({
  eventType,
  eventData,
  eventId,
  timestamp,
  fallbackMessageId,
}: {
  eventType: FinalAnswerStreamEventType
  eventData: unknown
  eventId?: unknown
  timestamp?: unknown
  fallbackMessageId?: string
}): FinalAnswerStreamActionPayload | null => {
  const data =
    eventData && typeof eventData === "object"
      ? (eventData as Record<string, unknown>)
      : {}
  const nestedData =
    data.data && typeof data.data === "object"
      ? (data.data as Record<string, unknown>)
      : {}
  const streamData = { ...nestedData, ...data }
  const messageId = String(
    streamData.message_id || eventId || fallbackMessageId || "",
  )
  if (!messageId) {
    return null
  }

  const normalizedTimestamp = String(timestamp || new Date().toISOString())
  if (eventType === "final_answer_start") {
    return {
      messageId,
      timestamp: normalizedTimestamp,
      status: "running",
    }
  }
  if (eventType === "final_answer_delta") {
    const delta = typeof streamData.delta === "string" ? streamData.delta : ""
    if (!delta) {
      return null
    }
    return {
      messageId,
      delta,
      timestamp: normalizedTimestamp,
      status: "running",
    }
  }
  if (eventType === "final_answer_error") {
    const error = typeof streamData.error === "string" ? streamData.error : ""
    return {
      messageId,
      content: error,
      timestamp: normalizedTimestamp,
      status: "failed",
    }
  }

  const content =
    typeof streamData.content === "string"
      ? unwrapFinalAnswerContent(streamData.content)
      : ""
  return {
    messageId,
    content,
    timestamp: normalizedTimestamp,
    status: "completed",
  }
}

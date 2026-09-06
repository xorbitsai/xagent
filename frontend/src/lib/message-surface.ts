export type MessageSurface =
  "chat" | "timeline" | "status" | "stream" | "ignore";

type MessageSurfaceData = {
  display?: unknown;
  expect_response?: unknown;
  message_type?: unknown;
  metadata?: { display?: unknown };
  visible?: unknown;
};

const MESSAGE_SURFACES = new Set<MessageSurface>([
  "chat",
  "timeline",
  "status",
  "stream",
  "ignore",
]);

const FINAL_ANSWER_EVENT_TYPES = new Set([
  "final_answer_start",
  "final_answer_delta",
  "final_answer_end",
  "final_answer_error",
]);

export const isMessageDisplayEventType = (eventType: string): boolean =>
  [
    "agent_message",
    "agent_progress",
    "agent_status",
    "ai_message",
    "chat_message",
    "user_message",
  ].includes(eventType);

export const expectsUserResponse = (
  eventType: string,
  data: unknown,
): boolean => {
  const messageData =
    data && typeof data === "object" ? (data as MessageSurfaceData) : undefined;
  return (
    eventType === "agent_message" &&
    messageData?.expect_response === true
  );
};

export const getMessageSurface = (
  eventType: string,
  data: unknown,
): MessageSurface => {
  const messageData =
    data && typeof data === "object" ? (data as MessageSurfaceData) : undefined;
  if (expectsUserResponse(eventType, messageData)) {
    return "chat";
  }
  if (messageData?.visible === false) return "ignore";

  const configured = messageData?.display ?? messageData?.metadata?.display;
  if (MESSAGE_SURFACES.has(configured as MessageSurface)) {
    return configured as MessageSurface;
  }
  if (FINAL_ANSWER_EVENT_TYPES.has(eventType)) return "stream";
  if (eventType === "agent_status") return "status";
  if (
    eventType === "agent_progress" ||
    messageData?.message_type === "progress"
  ) {
    return "timeline";
  }
  if (
    eventType === "agent_message" &&
    messageData?.message_type === "question"
  ) {
    return "chat";
  }
  if (
    ["agent_message", "ai_message", "chat_message", "user_message"].includes(
      eventType,
    )
  ) {
    return "chat";
  }
  return "timeline";
};

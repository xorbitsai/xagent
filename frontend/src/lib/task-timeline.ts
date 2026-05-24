export type TimelineAnchor = {
  timestamp: number;
  role?: string;
};

export function getProcessGroupIndex(
  sortedMessages: TimelineAnchor[],
  eventTime: number
): number {
  let groupIndex = 0;
  while (groupIndex < sortedMessages.length) {
    const message = sortedMessages[groupIndex];
    if (message.timestamp < eventTime) {
      groupIndex += 1;
      continue;
    }
    if (message.timestamp === eventTime && message.role === "user") {
      groupIndex += 1;
      continue;
    }
    break;
  }
  return groupIndex;
}

export type TimelineAnchor = {
  timestamp: number;
  role?: string;
};

/** User messages delimit execution turns; assistant result messages should not split trace groups. */
export function getUserTimelineAnchors<T extends TimelineAnchor>(
  sortedMessages: T[],
): T[] {
  return sortedMessages.filter((message) => message.role === "user");
}

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

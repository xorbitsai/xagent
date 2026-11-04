import { describe, expect, it } from "vitest";

import { getProcessGroupIndex, getUserTimelineAnchors } from "./task-timeline";

describe("getProcessGroupIndex", () => {
  it("places same-time process events after the user message that triggered them", () => {
    expect(
      getProcessGroupIndex([{ role: "user", timestamp: 1000 }], 1000)
    ).toBe(1);
  });

  it("keeps same-time process events before an assistant message", () => {
    expect(
      getProcessGroupIndex([{ role: "assistant", timestamp: 1000 }], 1000)
    ).toBe(0);
  });

  it("places process events between a same-time user and assistant pair", () => {
    expect(
      getProcessGroupIndex(
        [
          { role: "user", timestamp: 1000 },
          { role: "assistant", timestamp: 1000 },
        ],
        1000
      )
    ).toBe(1);
  });

  it("keeps ordinary before and after timestamp ordering", () => {
    const messages = [
      { role: "user", timestamp: 1000 },
      { role: "assistant", timestamp: 2000 },
    ];

    expect(getProcessGroupIndex(messages, 500)).toBe(0);
    expect(getProcessGroupIndex(messages, 1500)).toBe(1);
    expect(getProcessGroupIndex(messages, 2500)).toBe(2);
  });

  it("keeps late trace events in the same turn when grouping by user anchors only", () => {
    const userAnchors = getUserTimelineAnchors([
      { role: "user", timestamp: 1000 },
      { role: "assistant", timestamp: 3000 },
    ]);

    expect(getProcessGroupIndex(userAnchors, 1500)).toBe(1);
    expect(getProcessGroupIndex(userAnchors, 4000)).toBe(1);
  });
});

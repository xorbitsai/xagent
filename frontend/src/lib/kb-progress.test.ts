import { describe, expect, it } from "vitest"

import { findMatchingIngestionTask, getKBTaskProgressDetail, getKBTaskProgressPercent } from "./kb-progress"

describe("kb progress helpers", () => {
  it("matches the latest ingestion task by collection and filename", () => {
    const task = findMatchingIngestionTask(
      [
        {
          task_id: "older",
          status: "running",
          start_time: 1,
          metadata: { collection: "demo", source_path: "/tmp/demo/file.xlsx" },
        },
        {
          task_id: "latest",
          status: "running",
          start_time: 2,
          metadata: { collection: "demo", source_path: "/tmp/demo/file.xlsx" },
        },
      ],
      "demo",
      "file.xlsx"
    )

    expect(task?.task_id).toBe("latest")
  })

  it("extracts detailed step message and percent from step counts", () => {
    const task = {
      task_id: "task-1",
      status: "running",
      current_step: "compute_embeddings",
      overall_progress: 0.2,
      metadata: {
        collection: "demo",
        source_path: "/tmp/demo/file.xlsx",
        steps: {
          compute_embeddings: {
            current_count: 37,
            total_count: 254,
            message: "Embedding 37/254",
          },
        },
      },
    }

    expect(getKBTaskProgressDetail(task)).toBe("Embedding 37/254")
    expect(getKBTaskProgressPercent(task)).toBeCloseTo((37 / 254) * 100)
  })
})

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

  it("extracts detailed step message and keeps percent monotonic", () => {
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
    expect(getKBTaskProgressPercent(task)).toBeCloseTo(20)
  })

  it("keeps progress monotonic across multiple tracked steps", () => {
    const task = {
      task_id: "task-2",
      status: "running",
      current_step: "write_vectors_to_db",
      overall_progress: 0.25,
      metadata: {
        collection: "demo",
        source_path: "/tmp/demo/file.xlsx",
        steps: {
          compute_embeddings: {
            completed: true,
            step_progress: 1,
            message: "Embeddings complete",
          },
          write_vectors_to_db: {
            current_count: 10,
            total_count: 20,
            message: "Writing 10/20",
          },
        },
      },
    }

    expect(getKBTaskProgressPercent(task)).toBeCloseTo(75)
  })
})

import { describe, expect, it } from "vitest"
import {
  addPlanStep,
  deletePlanStep,
  movePlanStep,
  parseInstructionSteps,
  updatePlanStep,
} from "./agent-flow-plan"

// ─── parseInstructionSteps ────────────────────────────────────────────────────

describe("parseInstructionSteps", () => {
  describe("explicit detection", () => {
    it("detects a numbered list as explicit", () => {
      const result = parseInstructionSteps("1. First step\n2. Second step\n3. Third step")
      expect(result.explicit).toBe(true)
      expect(result.steps).toHaveLength(3)
      expect(result.steps[0].text).toBe("First step")
      expect(result.steps[1].text).toBe("Second step")
      expect(result.steps[0].lineIdx).toBe(0)
      expect(result.steps[0].prefix).toBe("1. ")
    })

    it("detects a 'Step X:' format list as explicit", () => {
      const result = parseInstructionSteps("Step 1: Gather data\nStep 2: Analyze results")
      expect(result.explicit).toBe(true)
      expect(result.steps).toHaveLength(2)
      expect(result.steps[0].text).toBe("Gather data")
      expect(result.steps[1].text).toBe("Analyze results")
    })

    it("detects two or more bullets as explicit", () => {
      const result = parseInstructionSteps("- First bullet\n- Second bullet")
      expect(result.explicit).toBe(true)
      expect(result.steps).toHaveLength(2)
      expect(result.steps[0].text).toBe("First bullet")
    })

    it("treats a single numbered item as explicit", () => {
      const result = parseInstructionSteps("1. Only one step here")
      expect(result.explicit).toBe(true)
      expect(result.steps).toHaveLength(1)
    })

    it("treats a single bullet as inferred (falls through)", () => {
      // One lone bullet without a number is not considered explicit
      const result = parseInstructionSteps("- Be concise and professional")
      expect(result.explicit).toBe(false)
    })

    it("strips the bullet marker in inferred mode so it does not leak into step text", () => {
      const result = parseInstructionSteps("- Be concise and always answer professionally.")
      expect(result.explicit).toBe(false)
      // step text must not start with "- "
      if (result.steps.length > 0) {
        expect(result.steps[0].text).not.toMatch(/^- /)
      }
    })
  })

  describe("inferred mode", () => {
    it("extracts qualifying sentences from prose", () => {
      const result = parseInstructionSteps(
        "Always greet the user warmly. Provide accurate information. Ask clarifying questions when needed.",
      )
      expect(result.explicit).toBe(false)
      expect(result.steps.length).toBeGreaterThanOrEqual(2)
      expect(result.steps[0].text).not.toMatch(/\.$/) // trailing punctuation stripped
      expect(result.steps[0].lineIdx).toBeNull()
      expect(result.steps[0].prefix).toBeNull()
    })

    it("limits the displayed steps to 3", () => {
      const result = parseInstructionSteps(
        "First sentence here. Second sentence here. Third sentence here. Fourth sentence here.",
      )
      expect(result.explicit).toBe(false)
      expect(result.steps.length).toBeLessThanOrEqual(3)
    })

    it("filters out sentences shorter than 15 chars", () => {
      const result = parseInstructionSteps("Ok. Sure. Answer user questions carefully and thoroughly.")
      expect(result.explicit).toBe(false)
      // Short sentences "Ok." and "Sure." should be filtered
      result.steps.forEach((s) => expect(s.text.length).toBeGreaterThanOrEqual(14))
    })

    it("filters out sentences longer than 400 chars", () => {
      const longSentence = "A".repeat(401) + "."
      const result = parseInstructionSteps(longSentence + " Keep replies short and clear.")
      expect(result.explicit).toBe(false)
      // The long sentence should not appear in displayed steps
      result.steps.forEach((s) => expect(s.text.length).toBeLessThanOrEqual(400))
    })

    it("returns empty steps for empty input", () => {
      const result = parseInstructionSteps("")
      expect(result.explicit).toBe(false)
      expect(result.steps).toHaveLength(0)
    })
  })
})

// ─── updatePlanStep ───────────────────────────────────────────────────────────

describe("updatePlanStep", () => {
  describe("explicit mode", () => {
    it("updates the correct numbered step in place", () => {
      const instructions = "1. First step\n2. Second step\n3. Third step"
      const result = updatePlanStep(instructions, 1, "Updated second step")
      expect(result).toBe("1. First step\n2. Updated second step\n3. Third step")
    })

    it("updates a 'Step X:' formatted step", () => {
      const instructions = "Step 1: Gather data\nStep 2: Analyze results"
      const result = updatePlanStep(instructions, 0, "Collect all data sources")
      expect(result).toBe("Step 1: Collect all data sources\nStep 2: Analyze results")
    })

    it("preserves surrounding non-plan text", () => {
      const instructions = "Introduction here.\n\n1. Step one\n2. Step two\n\nConclusion."
      const result = updatePlanStep(instructions, 0, "New step one")
      expect(result).toContain("Introduction here.")
      expect(result).toContain("1. New step one")
      expect(result).toContain("Conclusion.")
    })
  })

  describe("inferred mode — critical regression: must not duplicate original text", () => {
    it("replaces instructions with an explicit plan instead of appending", () => {
      const instructions =
        "Always be helpful and professional. Provide accurate information. Ask questions when unclear."
      const result = updatePlanStep(instructions, 0, "Be extremely helpful and friendly")
      // Must NOT contain the original prose as a preamble before the Plan block
      expect(result).not.toContain("Always be helpful and professional")
      // Must produce an explicit plan
      expect(result).toMatch(/^Plan:\n1\./)
    })

    it("captures all qualifying sentences, not just the 3 displayed", () => {
      // 4 sentences all >= 15 chars; only 3 are displayed but all should be preserved
      const instructions =
        "First qualifying sentence here. Second qualifying sentence here. " +
        "Third qualifying sentence here. Fourth qualifying sentence here."
      const result = updatePlanStep(instructions, 0, "Updated first sentence here")
      // The explicit plan should have 4 steps (all sentences were captured)
      const lines = result.split("\n")
      const stepLines = lines.filter((l) => /^\d+\./.test(l))
      expect(stepLines.length).toBe(4)
    })

    it("edits the correct sentence when a >400-char sentence precedes shorter ones", () => {
      // The long sentence must be filtered out of both display and materialize paths
      const longSentence = "L".repeat(401) + "."
      const instructions = longSentence + " Short qualifying sentence one. Short qualifying sentence two."
      const result = updatePlanStep(instructions, 0, "Replacement for sentence one")
      expect(result).toContain("1. Replacement for sentence one")
      expect(result).toContain("2. Short qualifying sentence two")
      // The >400-char sentence should not appear as a plan step
      expect(result).not.toContain("L".repeat(10))
    })
  })
})

// ─── deletePlanStep ───────────────────────────────────────────────────────────

describe("deletePlanStep", () => {
  describe("explicit mode", () => {
    it("deletes the correct step and renumbers", () => {
      const instructions = "1. First step\n2. Second step\n3. Third step"
      const result = deletePlanStep(instructions, 1)
      expect(result).toBe("1. First step\n2. Third step")
    })

    it("deletes the first step and renumbers from 1", () => {
      const instructions = "1. First step\n2. Second step\n3. Third step"
      const result = deletePlanStep(instructions, 0)
      expect(result).toBe("1. Second step\n2. Third step")
    })

    it("deletes the last step correctly", () => {
      const instructions = "1. First step\n2. Second step\n3. Third step"
      const result = deletePlanStep(instructions, 2)
      expect(result).toBe("1. First step\n2. Second step")
    })

    it("renumbers 'Step X:' format correctly after deletion", () => {
      const instructions = "Step 1: Gather data\nStep 2: Analyze results\nStep 3: Write report"
      const result = deletePlanStep(instructions, 1)
      expect(result).toBe("Step 1: Gather data\nStep 2: Write report")
    })

    it("does NOT renumber a digit-prefixed line the parser skipped (not a plan step)", () => {
      // "2.   " (whitespace-only content after the marker) matches renumber's
      // digit-prefix shape but is skipped by parseInstructionSteps (empty step
      // text), making it the one line form that collides with the plan's
      // numbering pattern without being a plan step. An unrestricted per-line
      // scan (the pre-fix behavior) would rewrite it to "1." and shift every
      // following step's number off by one; the index-restricted renumber
      // must leave it untouched.
      const instructions = "1. First step here\n2.   \n3. Second real step\n4. Third real step"
      const result = deletePlanStep(instructions, 0)
      const lines = result.split("\n")
      expect(lines[0]).toBe("2.   ") // untouched — not part of the plan
      expect(lines[1]).toBe("1. Second real step")
      expect(lines[2]).toBe("2. Third real step")
    })

    it("correctly shifts indices of remaining steps after splice", () => {
      // Regression: earlier bug miscalculated remaining line indices after splice,
      // corrupting the renumber pass when a middle step was deleted.
      const instructions = "1. Alpha step here\n2. Beta step here\n3. Gamma step here\n4. Delta step here"
      const result = deletePlanStep(instructions, 1) // delete "Beta"
      expect(result).toBe("1. Alpha step here\n2. Gamma step here\n3. Delta step here")
    })
  })

  describe("inferred mode", () => {
    it("removes the correct sentence and produces an explicit plan", () => {
      const instructions =
        "Always greet the user warmly. Provide accurate information. Ask clarifying questions when needed."
      const result = deletePlanStep(instructions, 0)
      expect(result).toMatch(/^Plan:\n/)
      expect(result).not.toContain("Always greet the user warmly")
      expect(result).toContain("Provide accurate information")
    })
  })
})

// ─── movePlanStep ─────────────────────────────────────────────────────────────

describe("movePlanStep", () => {
  describe("explicit mode", () => {
    it("moves a step up", () => {
      const instructions = "1. First step\n2. Second step\n3. Third step"
      const result = movePlanStep(instructions, 1, -1)
      expect(result).toBe("1. Second step\n2. First step\n3. Third step")
    })

    it("moves a step down", () => {
      const instructions = "1. First step\n2. Second step\n3. Third step"
      const result = movePlanStep(instructions, 0, 1)
      expect(result).toBe("1. Second step\n2. First step\n3. Third step")
    })

    it("is a no-op when moving the first step up", () => {
      const instructions = "1. First step\n2. Second step"
      expect(movePlanStep(instructions, 0, -1)).toBe(instructions)
    })

    it("is a no-op when moving the last step down", () => {
      const instructions = "1. First step\n2. Second step"
      expect(movePlanStep(instructions, 1, 1)).toBe(instructions)
    })
  })

  describe("inferred mode", () => {
    it("swaps sentences and produces an explicit plan", () => {
      const instructions = "First sentence here is long enough. Second sentence here is long enough."
      const result = movePlanStep(instructions, 0, 1)
      expect(result).toMatch(/^Plan:\n/)
      expect(result).toMatch(/1\. Second sentence here/)
      expect(result).toMatch(/2\. First sentence here/)
    })
  })
})

// ─── addPlanStep ──────────────────────────────────────────────────────────────

describe("addPlanStep", () => {
  describe("explicit mode", () => {
    it("appends a new step after the last one", () => {
      const instructions = "1. First step\n2. Second step"
      const result = addPlanStep(instructions, "New step")
      expect(result).toBe("1. First step\n2. Second step\n3. New step")
    })

    it("increments numeric prefix correctly", () => {
      const instructions = "1. Only step"
      const result = addPlanStep(instructions, "Another step")
      expect(result).toBe("1. Only step\n2. Another step")
    })

    it("increments 'Step X:' prefix correctly", () => {
      const instructions = "Step 1: First\nStep 2: Second"
      const result = addPlanStep(instructions, "Third task")
      expect(result).toContain("Step 3:")
      expect(result).toContain("Third task")
    })
  })

  describe("inferred mode", () => {
    it("materializes existing sentences plus the new step", () => {
      const instructions = "Always be helpful and respond promptly. Keep answers concise and clear."
      const result = addPlanStep(instructions, "New placeholder step")
      expect(result).toMatch(/^Plan:\n/)
      expect(result).toContain("New placeholder step")
      // Original sentences should also appear as steps
      expect(result).toContain("Always be helpful and respond promptly")
    })

    it("creates a plan with just the placeholder for empty instructions", () => {
      const result = addPlanStep("", "First step placeholder text")
      expect(result).toBe("Plan:\n1. First step placeholder text")
    })
  })
})

// ─── Round-trip / integration ─────────────────────────────────────────────────

describe("round-trip integrity", () => {
  it("edit → re-parse returns explicit with updated text", () => {
    const original = "Always greet users kindly. Provide helpful and accurate answers always."
    const edited = updatePlanStep(original, 0, "Greet users with warmth and care")
    const reparsed = parseInstructionSteps(edited)
    expect(reparsed.explicit).toBe(true)
    expect(reparsed.steps[0].text).toBe("Greet users with warmth and care")
  })

  it("add → re-parse gives one more explicit step", () => {
    const instructions = "1. Step one\n2. Step two"
    const added = addPlanStep(instructions, "Step three content")
    const reparsed = parseInstructionSteps(added)
    expect(reparsed.explicit).toBe(true)
    expect(reparsed.steps).toHaveLength(3)
  })

  it("delete middle → re-parse leaves contiguous numbering", () => {
    const instructions = "1. Alpha\n2. Beta\n3. Gamma"
    const deleted = deletePlanStep(instructions, 1)
    const reparsed = parseInstructionSteps(deleted)
    expect(reparsed.explicit).toBe(true)
    expect(reparsed.steps).toHaveLength(2)
    expect(reparsed.steps[0].text).toBe("Alpha")
    expect(reparsed.steps[1].text).toBe("Gamma")
    // Numbers must be contiguous (1, 2) not (1, 3)
    expect(reparsed.steps[0].prefix).toBe("1. ")
    expect(reparsed.steps[1].prefix).toBe("2. ")
  })
})

export interface KBProgressStepData {
  current_count?: number
  total_count?: number
  step_progress?: number
  message?: string
  metadata?: Record<string, unknown>
}

export interface KBProgressTask {
  task_id: string
  status: string
  current_step?: string | null
  overall_progress?: number | null
  start_time?: number | null
  metadata?: {
    collection?: string
    source_path?: string
    doc_id?: string
    steps?: Record<string, KBProgressStepData>
    [key: string]: unknown
  }
}

export function findMatchingIngestionTask(
  tasks: KBProgressTask[],
  collection: string,
  fileName: string
): KBProgressTask | null {
  const matches = tasks.filter(task => {
    const sourcePath = String(task.metadata?.source_path || "")
    return task.metadata?.collection === collection && sourcePath.endsWith(`/${fileName}`)
  })

  if (matches.length === 0) return null

  return matches.sort((a, b) => (b.start_time || 0) - (a.start_time || 0))[0]
}

export function getKBTaskProgressDetail(task: KBProgressTask | null): string | null {
  if (!task) return null

  const currentStepName = task.current_step || ""
  const currentStep = task.metadata?.steps?.[currentStepName]
  if (!currentStep) return null

  if (
    typeof currentStep.current_count === "number" &&
    typeof currentStep.total_count === "number" &&
    currentStep.total_count > 0
  ) {
    return currentStep.message || `${currentStep.current_count}/${currentStep.total_count}`
  }

  return currentStep.message || null
}

export function getKBTaskProgressPercent(task: KBProgressTask | null): number | null {
  if (!task) return null

  const currentStepName = task.current_step || ""
  const currentStep = task.metadata?.steps?.[currentStepName]
  if (
    currentStep &&
    typeof currentStep.current_count === "number" &&
    typeof currentStep.total_count === "number" &&
    currentStep.total_count > 0
  ) {
    return Math.max(0, Math.min(100, (currentStep.current_count / currentStep.total_count) * 100))
  }

  if (typeof task.overall_progress === "number") {
    return Math.max(0, Math.min(100, task.overall_progress * 100))
  }

  return null
}

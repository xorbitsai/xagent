export interface KBProgressStepData {
  completed?: boolean
  current_count?: number
  total_count?: number
  step_progress?: number
  message?: string
  metadata?: Record<string, unknown>
}

function clampPercent(value: number): number {
  return Math.max(0, Math.min(100, value))
}

function getStepPercent(step: KBProgressStepData | undefined): number | null {
  if (!step) return null

  if (
    typeof step.current_count === "number" &&
    typeof step.total_count === "number" &&
    step.total_count > 0
  ) {
    return clampPercent((step.current_count / step.total_count) * 100)
  }

  if (typeof step.step_progress === "number") {
    return clampPercent(step.step_progress * 100)
  }

  if (step.completed) {
    return 100
  }

  return null
}

function getStepsOverallPercent(task: KBProgressTask): number | null {
  const steps = task.metadata?.steps
  if (!steps) return null

  const stepEntries = Object.values(steps)
  if (stepEntries.length === 0) return null

  const stepPercents = stepEntries
    .map(step => getStepPercent(step))
    .filter((value): value is number => value !== null)

  if (stepPercents.length === 0) return null

  return clampPercent(
    stepPercents.reduce((sum, value) => sum + value, 0) / stepPercents.length
  )
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
  const currentStepPercent = getStepPercent(currentStep)
  const stepsOverallPercent = getStepsOverallPercent(task)

  if (typeof task.overall_progress === "number") {
    return clampPercent(
      Math.max(task.overall_progress * 100, stepsOverallPercent ?? 0, currentStepPercent ?? 0)
    )
  }

  if (stepsOverallPercent !== null) {
    return stepsOverallPercent
  }

  if (currentStepPercent !== null) {
    return currentStepPercent
  }

  return null
}

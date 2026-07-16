import React from "react"
import { Check } from "lucide-react"
import { cn } from "@/lib/utils"

export interface Step {
  label: string
  content: React.ReactNode
}

export interface StepperProps extends Omit<React.HTMLAttributes<HTMLDivElement>, 'content'> {
  steps: Step[]
  currentStep: number
  contentClassName?: string
}

export function Stepper({
  steps,
  currentStep,
  className,
  contentClassName,
  ...props
}: StepperProps) {
  const currentStepContent = steps[currentStep - 1]?.content

  return (
    <div
      className={cn("flex flex-col w-full", className)}
      role="region"
      aria-label="Progress Stepper"
      {...props}
    >
      <div
        className="flex items-center gap-4 mb-6 overflow-x-auto overflow-y-hidden py-2"
        role="list"
        aria-label="Steps"
      >
        {steps.map((step, index) => {
          const stepNumber = index + 1
          const isCompleted = currentStep > stepNumber
          const isActive = currentStep === stepNumber

          return (
            <React.Fragment key={step.label}>
              <div
                role="listitem"
                aria-current={isActive ? "step" : undefined}
                className={cn(
                  "flex items-center gap-2 shrink-0",
                  isActive || isCompleted ? "text-primary" : "text-muted-foreground"
                )}
              >
                <div
                  className={cn(
                    "w-6 h-6 rounded-full flex items-center justify-center border",
                    isCompleted
                      ? "bg-green-500 text-white border-green-500"
                      : isActive
                        ? "border-primary bg-primary text-primary-foreground"
                        : ""
                  )}
                  aria-hidden="true"
                >
                  {isCompleted ? <Check className="w-4 h-4" /> : stepNumber}
                </div>
                <span className="font-medium">
                  {isCompleted && <span className="sr-only">Completed: </span>}
                  {isActive && <span className="sr-only">Current Step: </span>}
                  {step.label}
                </span>
              </div>
              {index < steps.length - 1 && <div className="flex-1 min-w-6 h-px bg-border" aria-hidden="true" />}
            </React.Fragment>
          )
        })}
      </div>
      <div
        className={cn("flex-1 min-h-0 overflow-y-auto", contentClassName)}
        role="tabpanel"
        aria-label={`Content for ${steps[currentStep - 1]?.label}`}
      >
        {currentStepContent}
      </div>
    </div>
  )
}

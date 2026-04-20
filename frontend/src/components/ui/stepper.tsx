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
}

export function Stepper({ steps, currentStep, className, ...props }: StepperProps) {
  const currentStepContent = steps[currentStep - 1]?.content

  return (
    <div className={cn("flex flex-col w-full", className)} {...props}>
      <div className="flex items-center gap-4 mb-6">
        {steps.map((step, index) => {
          const stepNumber = index + 1
          const isCompleted = currentStep > stepNumber
          const isActive = currentStep === stepNumber

          return (
            <React.Fragment key={step.label}>
              <div
                className={cn(
                  "flex items-center gap-2",
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
                >
                  {isCompleted ? <Check className="w-4 h-4" /> : stepNumber}
                </div>
                <span className="font-medium">{step.label}</span>
              </div>
              {index < steps.length - 1 && <div className="flex-1 h-px bg-border" />}
            </React.Fragment>
          )
        })}
      </div>
      <div className="flex-1">
        {currentStepContent}
      </div>
    </div>
  )
}

import React from "react"
import { Button } from "@/components/ui/button"
import { LucideIcon } from "lucide-react"

export interface FeatureItem {
  icon: LucideIcon
  title: string
  description: string
}

export interface FeatureEmptyStateProps {
  icon: LucideIcon
  title: string
  description: string
  features: FeatureItem[]
  actionLabel: string
  onAction: () => void
  className?: string
}

export function FeatureEmptyState({
  icon: Icon,
  title,
  description,
  features,
  actionLabel,
  onAction,
  className
}: FeatureEmptyStateProps) {
  return (
    <div className={`flex items-center justify-center w-full py-8 ${className || ""}`}>
      <div className="w-full max-w-[700px] p-10 sm:p-12 shadow-sm border border-border/60 rounded-2xl bg-card">
        <div className="flex flex-col items-center text-center mb-10">
          <div className="h-16 w-16 rounded-2xl bg-primary/10 flex items-center justify-center text-primary mb-6 shadow-sm border border-primary/5">
            <Icon className="h-8 w-8" />
          </div>
          <h2 className="text-[20px] font-bold mb-3">{title}</h2>
          <p className="text-muted-foreground text-[15px] max-w-lg leading-relaxed">
            {description}
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-10 mb-12">
          {features.map((feature, idx) => (
            <div key={idx} className="flex flex-col text-left">
              <div className="flex items-center gap-2.5 mb-2.5">
                <feature.icon className="h-[18px] w-[18px] text-primary" />
                <h3 className="font-bold text-[15px] text-foreground">{feature.title}</h3>
              </div>
              <p className="text-[14px] text-muted-foreground leading-relaxed">
                {feature.description}
              </p>
            </div>
          ))}
        </div>

        <div className="flex justify-center">
          <Button onClick={onAction} className="bg-blue-600 hover:bg-blue-700 text-white font-semibold px-6 py-2.5 h-auto rounded-lg text-[14px]">
            {actionLabel}
          </Button>
        </div>
      </div>
    </div>
  )
}

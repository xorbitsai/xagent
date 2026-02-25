"use client"

import type { ReactNode } from "react"
import type { LucideIcon } from "lucide-react"

interface AuthFeature {
  icon: LucideIcon
  title: string
  description: string
}

interface AuthPageShellProps {
  appName: string
  logoPath: string
  logoAlt: string
  leftDescription: string
  mobileSubtitle?: string
  features: AuthFeature[]
  children: ReactNode
}

export function AuthPageShell({
  appName,
  logoPath,
  logoAlt,
  leftDescription,
  mobileSubtitle,
  features,
  children,
}: AuthPageShellProps) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-primary/10 to-background relative overflow-hidden">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]" />
      <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent" />

      <div className="absolute top-1/4 left-1/4 w-72 h-72 bg-accent/30 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent/20 rounded-full blur-3xl animate-pulse delay-1000" />

      <div className="relative z-10 flex min-h-screen">
        <div className="hidden lg:flex lg:w-1/2 items-center justify-center p-12">
          <div className="max-w-lg">
            <div className="mb-8">
              <div className="flex items-center gap-3 mb-4">
                <img src={logoPath} alt={logoAlt} className="h-16 w-16" />
                <h1 className="text-4xl font-bold bg-gradient-to-r from-primary via-primary/80 to-primary/60 bg-clip-text text-transparent">
                  {appName}
                </h1>
              </div>
              <p className="text-xl text-muted-foreground leading-relaxed">
                {leftDescription}
              </p>
            </div>

            <div className="space-y-6">
              {features.map((feature, index) => (
                <div key={index} className="flex items-start gap-4 group">
                  <div className="h-12 w-12 rounded-lg bg-background/10 backdrop-blur-sm flex items-center justify-center group-hover:bg-accent transition-colors">
                    <feature.icon className="h-6 w-6 text-muted-foreground" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-foreground mb-1">
                      {feature.title}
                    </h3>
                    <p className="text-muted-foreground">{feature.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="flex-1 flex items-center justify-center p-8">
          <div className="w-full max-w-md">
            <div className="lg:hidden text-center mb-8">
              <div className="flex items-center justify-center gap-3 mb-4">
                <img src={logoPath} alt={logoAlt} className="h-12 w-12" />
                <h1 className="text-3xl font-bold bg-gradient-to-r from-primary via-primary/80 to-primary/60 bg-clip-text text-transparent">
                  {appName}
                </h1>
              </div>
              {mobileSubtitle ? (
                <p className="text-muted-foreground">{mobileSubtitle}</p>
              ) : null}
            </div>

            {children}
          </div>
        </div>
      </div>
    </div>
  )
}

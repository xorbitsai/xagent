"use client"

import type { ReactNode } from "react"
import Image from "next/image"
import type { LucideIcon } from "lucide-react"
import { Check } from "lucide-react"

interface AuthFeature {
  icon: LucideIcon
  title: string
  description: string
}

interface AuthPageShellProps {
  appName: string
  logoPath: string
  logoAlt: string
  heroTitle?: string
  leftDescription: string
  mobileSubtitle?: string
  features: AuthFeature[]
  children: ReactNode
}

export function AuthPageShell({
  appName,
  logoPath,
  logoAlt,
  heroTitle,
  leftDescription,
  mobileSubtitle,
  features,
  children,
}: AuthPageShellProps) {
  const heroLines = (heroTitle || appName)
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)

  const [headlinePrimary, ...headlineAccentLines] = heroLines
  const headlineAccent = headlineAccentLines.join(" ")

  return (
    <div className="relative min-h-screen overflow-hidden bg-[linear-gradient(180deg,#F8FAFF_0%,#F7FAFF_48%,#F3F7FF_100%)] text-[#111827]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(110,132,255,0.18),transparent_28%),radial-gradient(circle_at_bottom_right,rgba(89,146,255,0.12),transparent_26%)]" />
      <div className="absolute inset-0 bg-[radial-gradient(rgba(74,106,255,0.16)_1px,transparent_1px)] bg-[size:26px_26px] opacity-70" />
      <div className="absolute -left-24 top-16 h-72 w-72 rounded-full bg-[#DDE6FF] blur-3xl" />
      <div className="absolute bottom-8 right-4 h-96 w-96 rounded-full bg-[#D7E4FF] blur-3xl" />

      <div className="relative z-10 mx-auto flex min-h-screen max-w-[1440px] flex-col lg:flex-row">
        <div className="hidden lg:flex lg:w-[54%] items-center px-16 py-12 xl:px-24">
          <div className="max-w-[620px]">
            <div className="mb-12">
              <h1 className="max-w-[12ch] text-5xl font-semibold leading-[1.02] tracking-[-0.05em] text-[#18214D] xl:text-[4.25rem]">
                <span>{headlinePrimary}</span>
                {headlineAccent ? (
                  <>
                    <br />
                    <span className="bg-[linear-gradient(92deg,#2F54EB_8%,#3B82F6_54%,#00A6FF_100%)] bg-clip-text text-transparent">
                      {headlineAccent}
                    </span>
                  </>
                ) : null}
              </h1>
              <p className="mt-8 max-w-[34rem] text-lg leading-8 text-[#687184]">
                {leftDescription}
              </p>
            </div>

            <div className="space-y-7">
              {features.map((feature, index) => (
                <div key={index} className="flex items-start gap-4">
                  <div className="mt-1 flex h-7 w-7 items-center justify-center rounded-[10px] border border-[#D5DEFF] bg-[#EEF3FF]">
                    <Check className="h-4 w-4 text-[#4B67FF]" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-[#18214D]">
                      {feature.title}
                    </h3>
                    <p className="mt-1 max-w-[32rem] text-base leading-7 text-[#7B8496]">
                      {feature.description}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="flex flex-1 items-center justify-center px-5 py-10 sm:px-8 lg:w-[48%] lg:px-12">
          <div className="w-full max-w-[500px]">
            <div className="mb-8 text-center lg:hidden">
              <div className="mb-4 flex items-center justify-center gap-3">
                <Image src={logoPath} alt={logoAlt} width={40} height={40} className="h-10 w-10 object-contain" />
                <h1 className="bg-[linear-gradient(92deg,#2F54EB_8%,#3B82F6_54%,#00A6FF_100%)] bg-clip-text text-3xl font-semibold tracking-[-0.04em] text-transparent">
                  {appName}
                </h1>
              </div>
              {heroTitle ? (
                <p className="text-balance text-3xl font-semibold leading-tight tracking-[-0.04em] text-[#18214D]">
                  {heroTitle.replace(/\n/g, " ")}
                </p>
              ) : null}
              {mobileSubtitle ? (
                <p className="mt-3 text-sm leading-6 text-[#7B8496]">{mobileSubtitle}</p>
              ) : null}
            </div>

            {children}
          </div>
        </div>
      </div>
    </div>
  )
}

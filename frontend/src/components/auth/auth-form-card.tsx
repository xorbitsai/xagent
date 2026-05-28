"use client"

import type { ReactNode } from "react"
import Image from "next/image"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"

interface AuthFormCardProps {
  appName: string
  logoPath: string
  logoAlt: string
  modeLabel: string
  title: string
  description: string
  children: ReactNode
  footer: ReactNode
  showSocialLogin?: boolean
  socialHint?: string
}

function SocialIcon({ provider }: { provider: "google" | "microsoft" }) {
  if (provider === "google") {
    return (
      <span className="grid h-5 w-5 place-items-center rounded-full border border-[#E6EAF2] bg-white text-[11px] font-bold text-[#4285F4]">
        G
      </span>
    )
  }

  return (
    <span className="grid h-5 w-5 grid-cols-2 grid-rows-2 gap-[2px] rounded-[4px] bg-white p-[2px]">
      <span className="rounded-[1px] bg-[#F25022]" />
      <span className="rounded-[1px] bg-[#7FBA00]" />
      <span className="rounded-[1px] bg-[#00A4EF]" />
      <span className="rounded-[1px] bg-[#FFB900]" />
    </span>
  )
}

export function AuthFormCard({
  appName,
  logoPath,
  logoAlt,
  modeLabel,
  title,
  description,
  children,
  footer,
  showSocialLogin = true,
  socialHint = "OR CONTINUE WITH",
}: AuthFormCardProps) {
  return (
    <Card className="overflow-hidden rounded-[28px] border border-[#E6EAF2] bg-white/95 py-0 text-[#111827] shadow-[0_25px_80px_rgba(47,84,235,0.14),0_10px_30px_rgba(15,23,42,0.08)] backdrop-blur-xl">
      <div className="border-b border-[#EEF2F7] px-7 pb-6 pt-7">
        <div className="mb-6 flex items-center gap-3">
          <Image src={logoPath} alt={logoAlt} width={96} height={24} className="h-6 w-auto object-contain" />
          <span className="text-sm font-semibold text-[#2F54EB]">{appName}</span>
          <span className="h-4 w-px bg-[#D9E0EC]" />
          <span className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[#8B95A7]">
            {modeLabel} {appName}
          </span>
        </div>

        <div className="space-y-2">
          <h2 className="text-[2rem] font-semibold tracking-[-0.03em] text-[#171A2F]">
            {title}
          </h2>
          <p className="text-sm leading-6 text-[#7B8496]">{description}</p>
        </div>
      </div>

      <div className="px-7">{children}</div>

      {showSocialLogin ? (
        <div className="px-7 pb-7">
          <div className="mb-4 flex items-center gap-4">
            <div className="h-px flex-1 bg-[#E9EDF5]" />
            <span className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[#A2ABBB]">
              {socialHint}
            </span>
            <div className="h-px flex-1 bg-[#E9EDF5]" />
          </div>

          <div className="grid grid-cols-2 gap-3">
            {[
              { key: "google" as const, label: "Google" },
              { key: "microsoft" as const, label: "Microsoft" },
            ].map((provider) => (
              <button
                key={provider.key}
                type="button"
                disabled
                className={cn(
                  "inline-flex h-12 items-center justify-center gap-2 rounded-[14px] border border-[#E6EAF2] bg-white px-4 text-sm font-semibold text-[#4C5567] transition-colors",
                  "cursor-not-allowed opacity-80"
                )}
              >
                <SocialIcon provider={provider.key} />
                <span>{provider.label}</span>
              </button>
            ))}
          </div>
        </div>
      ) : null}

      <div className="px-7 pb-7 text-center text-sm text-[#8B95A7]">{footer}</div>
    </Card>
  )
}

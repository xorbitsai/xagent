"use client"

import { useTheme } from "@/contexts/theme-context"
import { Toaster as Sonner, toast as sonnerToast } from "sonner"

type ToasterProps = React.ComponentProps<typeof Sonner>
type PatchedToast = typeof sonnerToast & { __xagentErrorPatched?: boolean }

const ERROR_TOAST_DURATION = 8000
const patchedToast = sonnerToast as PatchedToast

if (!patchedToast.__xagentErrorPatched) {
  const originalError = patchedToast.error.bind(patchedToast)

  // Keep a longer default for errors while still allowing per-call overrides.
  patchedToast.error = (message, data) =>
    originalError(message, {
      duration: ERROR_TOAST_DURATION,
      ...data,
    })
  patchedToast.__xagentErrorPatched = true
}

const Toaster = ({ ...props }: ToasterProps) => {
  const { theme } = useTheme()

  return (
    <Sonner
      theme={theme.mode as ToasterProps["theme"]}
      className="toaster group"
      position="top-center"
      toastOptions={{
        classNames: {
          toast:
            "group toast group-[.toaster]:bg-background group-[.toaster]:text-foreground group-[.toaster]:border-border group-[.toaster]:shadow-lg data-[type=error]:!bg-red-500 data-[type=error]:!text-white data-[type=error]:!border-red-600 data-[type=success]:!text-green-600 data-[type=success]:!border-green-600",
          description: "group-[.toast]:text-muted-foreground group-data-[type=error]:!text-white/90 group-data-[type=success]:!text-green-600/90",
          actionButton:
            "group-[.toast]:bg-primary group-[.toast]:text-primary-foreground",
          cancelButton:
            "group-[.toast]:bg-muted group-[.toast]:text-muted-foreground",
          icon: "group-data-[type=error]:!text-white group-data-[type=success]:!text-green-600",
        },
      }}
      {...props}
    />
  )
}

export { Toaster }

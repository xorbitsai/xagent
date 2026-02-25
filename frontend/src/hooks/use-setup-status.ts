"use client"

import { useEffect, useState } from "react"

import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"

interface SetupStatusOptions {
  redirectToSetupIfNeeded?: boolean
  redirectToLoginIfRegistrationClosed?: boolean
  redirectToLoginIfInitialized?: boolean
}

interface SetupStatusState {
  isLoading: boolean
  initialized: boolean
  needsSetup: boolean
  registrationEnabled: boolean
}

export function useSetupStatus(
  options: SetupStatusOptions = {}
): SetupStatusState {
  const {
    redirectToSetupIfNeeded = false,
    redirectToLoginIfRegistrationClosed = false,
    redirectToLoginIfInitialized = false,
  } = options

  const [state, setState] = useState<SetupStatusState>({
    isLoading: true,
    initialized: false,
    needsSetup: false,
    registrationEnabled: true,
  })

  useEffect(() => {
    let isCancelled = false

    const checkSetupStatus = async () => {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/auth/setup-status`)
        if (!response.ok) {
          if (!isCancelled) {
            setState((prev) => ({ ...prev, isLoading: false }))
          }
          return
        }

        const data = await response.json()
        const needsSetup = Boolean(data.needs_setup)
        const initialized = Boolean(data.initialized)
        const registrationEnabled = Boolean(data.registration_enabled)

        if (redirectToSetupIfNeeded && needsSetup) {
          window.location.href = "/setup"
          return
        }

        if (redirectToLoginIfRegistrationClosed && !needsSetup && !registrationEnabled) {
          window.location.href = "/login"
          return
        }

        if (redirectToLoginIfInitialized && initialized && !needsSetup) {
          window.location.href = "/login"
          return
        }

        if (!isCancelled) {
          setState({
            isLoading: false,
            initialized,
            needsSetup,
            registrationEnabled,
          })
        }
      } catch {
        if (!isCancelled) {
          setState((prev) => ({ ...prev, isLoading: false }))
        }
      }
    }

    checkSetupStatus()

    return () => {
      isCancelled = true
    }
  }, [
    redirectToLoginIfInitialized,
    redirectToLoginIfRegistrationClosed,
    redirectToSetupIfNeeded,
  ])

  return state
}

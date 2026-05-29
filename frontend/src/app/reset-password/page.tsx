import { Suspense } from "react"

import { ResetPasswordPage } from "@/components/pages/reset-password"

export default function ResetPassword() {
  return (
    <Suspense fallback={null}>
      <ResetPasswordPage />
    </Suspense>
  )
}

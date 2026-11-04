import { Suspense } from "react"
import { ApiKeysPage } from "@/components/pages/api-keys"

export default function Page() {
  return (
    <Suspense fallback={null}>
      <ApiKeysPage />
    </Suspense>
  )
}

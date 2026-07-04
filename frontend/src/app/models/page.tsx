"use client"

import { Suspense } from "react"
import { ModelsPage } from "@/components/pages/models"

export default function Models() {
  return (
    <Suspense fallback={null}>
      <ModelsPage />
    </Suspense>
  )
}

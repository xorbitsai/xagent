"use client"

import React from "react"
import { useParams } from "next/navigation"
import { WorkforceBuilder } from "@/components/workforce/workforce-builder"

export default function WorkforceDetailPage() {
    const params = useParams()
    const id = Array.isArray(params.id) ? params.id[0] : params.id

    return <WorkforceBuilder key={id} workforceId={id} />
}

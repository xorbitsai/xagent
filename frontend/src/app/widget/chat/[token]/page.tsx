import React from "react"
import PageClient from "./page-client"

// Server wrapper for static export: provides a placeholder param so a shell
// HTML is emitted for this dynamic route. FastAPI serves the shell for any real
// token; the client component reads the actual value from the URL via useParams().
export function generateStaticParams() {
  return [{ token: "__shell__" }]
}

export default function Page() {
  return <PageClient />
}

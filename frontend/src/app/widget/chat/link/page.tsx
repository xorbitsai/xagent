import React from "react"
import PageClient from "./page-client"

// Host page of a one-time chat link. The static `link` segment takes precedence
// over the dynamic `[token]` segment next to it. The link code arrives in the
// URL fragment, which never reaches the server, so this page is a static shell.
export default function Page() {
  return <PageClient />
}

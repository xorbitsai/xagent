export const AUTH_PUBLIC_PATHS = ["/login", "/register", "/setup"] as const

export function isAuthPublicPath(pathname: string | null): boolean {
  if (!pathname) {
    return false
  }
  if (pathname.startsWith("/widget")) {
    return true
  }
  return AUTH_PUBLIC_PATHS.includes(pathname as (typeof AUTH_PUBLIC_PATHS)[number])
}

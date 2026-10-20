import type { ExtraNavResolver } from "@/lib/sidebar-navigation"

// Hosts can keep their existing array/resolver and opt into group placement:
// "workspace" for Team / billing; "builder" for AI creation tools.
// Rendering, active-route handling and collapse behavior stay in core.
const extraNav: ExtraNavResolver = []

export default extraNav

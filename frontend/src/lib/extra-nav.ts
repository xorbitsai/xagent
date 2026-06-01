import type { NavigationGroup } from "@/lib/sidebar-navigation"

type ExtraNavResolver = NavigationGroup[] | ((user: any) => NavigationGroup[])

const extraNav: ExtraNavResolver = []

export default extraNav

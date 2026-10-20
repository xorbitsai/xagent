import type { ComponentType, SVGProps } from "react"
import type { TranslationKey } from "@/i18n/translations"
import {
    Activity,
    FileText,
    Code2,
    Settings,
    Wrench,
    Users,
    Brain,
    Server,
    Library,
    MessageSquare,
    Bot,
    Box,
    ClipboardList,
    Globe,
    Key,
} from "lucide-react"

interface SidebarUser {
    is_admin?: boolean | null
}

export interface NavigationItem {
    name: string
    href: string
    icon: ComponentType<SVGProps<SVGSVGElement>>
    color?: string
    children?: NavigationItem[]
    showTasks?: boolean
    nameKey?: TranslationKey
    descriptionKey?: TranslationKey
}

export interface NavigationGroup {
    title: string
    titleKey?: TranslationKey
    descriptionKey?: TranslationKey
    icon?: ComponentType<SVGProps<SVGSVGElement>>
    items: NavigationItem[]
    /** Whether this group's items are hidden behind a collapse toggle by default. */
    defaultCollapsed?: boolean
    /**
     * Stable identity for this group, used instead of `title` for React keys and collapse
     * state so a downstream-injected group (e.g. via extra-nav) sharing a display title
     * doesn't collide with a built-in group's identity. Falls back to `title` when unset.
     */
    id?: string
    /**
     * Host-provided placement: workspace adds a first-level group before
     * Settings; builder appends its items to the shared Builder group.
     * Omit to preserve the legacy standalone-group behavior.
     */
    placement?: "workspace" | "builder"
}

export type ExtraNavResolver<User = unknown> = NavigationGroup[] | ((user: User) => NavigationGroup[])

/** Compose host configuration without copying the sidebar or mutating its inputs. */
export function composeSidebarNavigation(
    baseGroups: NavigationGroup[],
    baseUserItems: NavigationItem[],
    extraGroups: NavigationGroup[],
): { groups: NavigationGroup[]; userItems: NavigationItem[] } {
    const groups = baseGroups.map((group) => ({ ...group, items: [...group.items] }))
    const userItems = [...baseUserItems]
    for (const group of extraGroups) {
        if (group.items.length === 0) continue
        if (group.placement === "builder") {
            const builder = groups.find((candidate) => candidate.id === "builder")
            if (builder) {
                builder.items.push(...group.items)
                continue
            }
        }
        if (group.placement === "workspace") {
            const entry: NavigationItem = {
                name: group.title,
                nameKey: group.titleKey,
                descriptionKey: group.descriptionKey,
                href: `__workspace_${group.id ?? group.title}__`,
                icon: group.icon ?? Users,
                children: [...group.items],
            }
            const settingsIndex = userItems.findIndex((item) => item.href === "/settings")
            userItems.splice(settingsIndex < 0 ? userItems.length : settingsIndex, 0, entry)
        } else {
            groups.push({ ...group, items: [...group.items] })
        }
    }
    return { groups, userItems }
}

const baseMoreResourceItems: NavigationItem[] = [
    {
        name: "Channels",
        nameKey: "nav.channels",
        href: "/channels",
        icon: MessageSquare,
        color: "text-blue-400"
    },
    {
        name: "Conversation Logs",
        nameKey: "nav.conversationLogs",
        href: "/conversation-logs",
        icon: ClipboardList,
        color: "text-blue-400"
    },
    {
        name: "Monitoring",
        nameKey: "nav.monitoring",
        href: "/monitoring",
        icon: Activity,
        color: "text-blue-400"
    }
]

const getMoreResourceItemsForUser = (user?: SidebarUser | null): NavigationItem[] => {
    const items = [...baseMoreResourceItems]

    if (user?.is_admin) {
        items.push({
            name: "User Management",
            nameKey: "nav.userManagement",
            href: "/users",
            icon: Users,
            color: "text-blue-400"
        })
        items.push({
            name: "Public MCP Apps",
            nameKey: "nav.adminMcp",
            href: "/admin-mcp",
            icon: Server,
            color: "text-blue-400"
        })
    }

    return items
}

export const getNavigationGroupsForUser = (user?: SidebarUser | null): NavigationGroup[] => [
    {
        id: "builder",
        title: "Builder",
        titleKey: "nav.builder",
        descriptionKey: "nav.builderDescription",
        icon: Code2,
        defaultCollapsed: true,
        items: [
            {
                name: "Teammates",
                nameKey: "nav.teammates",
                href: "/workforces?view=teammates",
                icon: Bot,
                color: "text-yellow-400"
            },
            {
                name: "Knowledge",
                nameKey: "nav.knowledge",
                href: "/kb",
                icon: Globe,
                color: "text-gray-500"
            },
            {
                name: "Models",
                nameKey: "nav.models",
                href: "/models",
                icon: Box,
                color: "text-gray-500"
            },
            {
                name: "API Keys",
                nameKey: "nav.apiKeys",
                href: "/api-keys",
                icon: Key,
                color: "text-gray-500"
            },
            {
                name: "Memory",
                nameKey: "nav.memory",
                href: "/memory",
                icon: Brain,
                color: "text-gray-500"
            },
            ...getMoreResourceItemsForUser(user),
        ]
    }
]

const baseUserMenuItems: NavigationItem[] = [
    {
        name: "Tools and apps",
        nameKey: "nav.toolsAndApps",
        href: "/tools",
        icon: Wrench,
    },
    {
        name: "Skills",
        nameKey: "nav.skills",
        descriptionKey: "nav.skillsDescription",
        href: "/skill-hub",
        icon: Library,
    },
    {
        name: "Files",
        nameKey: "nav.files",
        href: "/files",
        icon: FileText,
    },
    {
        name: "Settings",
        nameKey: "nav.settings",
        href: "/settings",
        icon: Settings,
        color: "text-blue-400"
    }
]

export const getUserMenuItemsForUser = (user?: SidebarUser | null): NavigationItem[] => {
    void user
    return [...baseUserMenuItems]
}

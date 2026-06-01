import type { ComponentType, SVGProps } from "react"
import {
    Activity,
    FileText,
    Sparkles,
    Settings,
    Wrench,
    Users,
    Brain,
    Server,
    Layers,
    MessageSquare,
    Bot,
    Box,
    LayoutTemplate,
    Globe,
} from "lucide-react"

export interface NavigationItem {
    name: string
    href: string
    icon: ComponentType<SVGProps<SVGSVGElement>>
    color?: string
    children?: NavigationItem[]
    showTasks?: boolean
    nameKey?: string
}

export interface NavigationGroup {
    title: string
    titleKey?: string
    items: NavigationItem[]
}

const baseMoreResourceItems: NavigationItem[] = [
    {
        name: "Tools",
        nameKey: "nav.tools",
        href: "/tools",
        icon: Wrench,
        color: "text-blue-400"
    },
    {
        name: "Files",
        nameKey: "nav.files",
        href: "/files",
        icon: FileText,
        color: "text-blue-400"
    },
    {
        name: "Channels",
        nameKey: "nav.channels",
        href: "/channels",
        icon: MessageSquare,
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

const getMoreResourceItemsForUser = (user: any): NavigationItem[] => {
    const items = [...baseMoreResourceItems]

    if (user?.is_admin) {
        items.push({
            name: "User Management",
            nameKey: "nav.userManagement",
            href: "/users/",
            icon: Users,
            color: "text-blue-400"
        })
        items.push({
            name: "Public MCP Apps",
            nameKey: "nav.adminMcp",
            href: "/admin-mcp/",
            icon: Server,
            color: "text-blue-400"
        })
    }

    return items
}

export const getNavigationGroupsForUser = (user: any): NavigationGroup[] => [
    {
        title: "Agent Development",
        titleKey: "nav.sections.agentDevelopment",
        items: [
            {
                name: "Task",
                nameKey: "nav.task",
                href: "/task",
                icon: Sparkles,
                color: "text-blue-500"
            },
            {
                name: "Agents",
                nameKey: "nav.build",
                href: "/build",
                icon: Bot,
                color: "text-yellow-400"
            },
            {
                name: "Templates",
                nameKey: "nav.templates",
                href: "/templates",
                icon: LayoutTemplate,
                color: "text-purple-400"
            },
            {
                name: "Workforces",
                nameKey: "nav.workforces",
                href: "/workforces",
                icon: Layers,
                color: "text-cyan-500"
            },
        ]
    },
    {
        title: "Resources",
        titleKey: "nav.sections.resources",
        items: [
            {
                name: "Knowledge Base",
                nameKey: "nav.knowledgeBase",
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
                name: "Memory",
                nameKey: "nav.memory",
                href: "/memory",
                icon: Brain,
                color: "text-gray-500"
            },
            {
                name: "More",
                nameKey: "nav.more",
                href: "__resources_more__",
                icon: Layers,
                color: "text-gray-500",
                children: getMoreResourceItemsForUser(user)
            }
        ]
    }
]

const baseUserMenuItems: NavigationItem[] = [
    {
        name: "Settings",
        nameKey: "nav.settings",
        href: "/settings",
        icon: Settings,
        color: "text-blue-400"
    }
]

export const getUserMenuItemsForUser = (_user: any): NavigationItem[] => {
    return [...baseUserMenuItems]
}

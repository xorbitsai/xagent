import type { ComponentType, SVGProps } from "react"
import type { TranslationKey } from "@/i18n/translations"
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
    Library,
    MessageSquare,
    Bot,
    Box,
    ClipboardList,
    LayoutTemplate,
    Globe,
    Key,
    MoreHorizontal,
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
}

export interface NavigationGroup {
    title: string
    titleKey?: TranslationKey
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

export const getNavigationGroupsForUser = (user?: SidebarUser | null): NavigationGroup[] => [
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
                name: "Skill Hub",
                nameKey: "nav.skillHub",
                href: "/skill-hub",
                icon: Library,
                color: "text-emerald-400"
            },
            {
                name: "API Keys",
                nameKey: "nav.apiKeys",
                href: "/api-keys",
                icon: Key,
                color: "text-gray-500"
            },
            {
                name: "More",
                nameKey: "nav.more",
                href: "__resources_more__",
                icon: MoreHorizontal,
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

export const getUserMenuItemsForUser = (user?: SidebarUser | null): NavigationItem[] => {
    void user
    return [...baseUserMenuItems]
}

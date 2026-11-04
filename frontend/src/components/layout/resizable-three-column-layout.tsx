import React, { useState, useCallback, useEffect } from "react"
import { cn } from "@/lib/utils"
import { GripVertical } from "lucide-react"

interface ResizableThreeColumnLayoutProps {
    leftPanel: React.ReactNode
    middlePanel: React.ReactNode
    rightPanel: React.ReactNode
    initialLeftWidth?: number // Percentage (0-100)
    initialMiddleWidth?: number // Percentage (0-100)
    initialRightWidth?: number // Percentage (0-100)
    minLeftWidth?: number // Percentage (0-100)
    minMiddleWidth?: number // Percentage (0-100)
    minRightWidth?: number // Percentage (0-100)
    className?: string
    showLeftPanel?: boolean
}

export function ResizableThreeColumnLayout({
    leftPanel,
    middlePanel,
    rightPanel,
    initialLeftWidth = 25,
    initialMiddleWidth = 45,
    initialRightWidth,
    minLeftWidth = 15,
    minMiddleWidth = 30,
    minRightWidth = 20,
    className,
    showLeftPanel = true
}: ResizableThreeColumnLayoutProps) {
    const [leftWidth, setLeftWidth] = useState(initialLeftWidth)
    const [middleWidth, setMiddleWidth] = useState(() => {
        if (typeof initialRightWidth === "number") {
            return Math.max(0, 100 - initialLeftWidth - initialRightWidth)
        }
        return initialMiddleWidth
    })
    const [isMobile, setIsMobile] = useState(false)

    useEffect(() => {
        const checkMobile = () => {
            setIsMobile(window.innerWidth < 768)
        }
        checkMobile()
        window.addEventListener('resize', checkMobile)
        return () => window.removeEventListener('resize', checkMobile)
    }, [])

    const [activeHandle, setActiveHandle] = useState<'left' | 'right' | null>(null)
    const containerRef = React.useRef<HTMLDivElement>(null)

    const handleMouseDownLeft = useCallback(() => {
        setActiveHandle('left')
        document.body.style.cursor = "col-resize"
        document.body.style.userSelect = "none"
    }, [])

    const handleMouseDownRight = useCallback(() => {
        setActiveHandle('right')
        document.body.style.cursor = "col-resize"
        document.body.style.userSelect = "none"
    }, [])

    const handleMouseUp = useCallback(() => {
        setActiveHandle(null)
        document.body.style.cursor = ""
        document.body.style.userSelect = ""
    }, [])

    const handleMouseMove = useCallback(
        (e: MouseEvent) => {
            if (!activeHandle || !containerRef.current) return

            const containerRect = containerRef.current.getBoundingClientRect()
            const mousePosition = ((e.clientX - containerRect.left) / containerRect.width) * 100

            if (activeHandle === 'left') {
                let newLeftWidth = Math.max(minLeftWidth, Math.min(mousePosition, 100 - minMiddleWidth - minRightWidth))
                const currentRightWidth = 100 - leftWidth - middleWidth
                const newMiddleWidth = 100 - newLeftWidth - currentRightWidth

                setLeftWidth(newLeftWidth)
                setMiddleWidth(newMiddleWidth)
            } else if (activeHandle === 'right') {
                let newBoundary = Math.max(leftWidth + minMiddleWidth, Math.min(mousePosition, 100 - minRightWidth))

                setMiddleWidth(newBoundary - leftWidth)
            }
        },
        [activeHandle, minLeftWidth, minMiddleWidth, minRightWidth, leftWidth, middleWidth]
    )

    useEffect(() => {
        if (activeHandle) {
            document.addEventListener("mousemove", handleMouseMove)
            document.addEventListener("mouseup", handleMouseUp)
        } else {
            document.removeEventListener("mousemove", handleMouseMove)
            document.removeEventListener("mouseup", handleMouseUp)
        }

        return () => {
            document.removeEventListener("mousemove", handleMouseMove)
            document.removeEventListener("mouseup", handleMouseUp)
        }
    }, [activeHandle, handleMouseMove, handleMouseUp])

    const rightWidth = 100 - leftWidth - middleWidth

    return (
        <div
            ref={containerRef}
            className={cn("flex flex-col md:flex-row w-full h-full min-h-0 md:overflow-hidden overflow-y-auto pb-16 lg:pb-0", className)}
        >
            {/* Left Panel */}
            <div
                style={{
                    width: isMobile ? '100%' : (showLeftPanel ? `${leftWidth}%` : '0%'),
                }}
                className={cn(
                    "w-full md:w-auto h-[60vh] max-h-[600px] md:h-full md:max-h-none flex-col min-h-0 md:overflow-hidden border-b md:border-b-0 shrink-0",
                    showLeftPanel ? 'flex' : 'hidden md:hidden'
                )}
            >
                {leftPanel}
            </div>

            {/* Left Resizer Handle (Hidden on mobile) */}
            <div
                className={cn(
                    "hidden w-1 bg-border hover:bg-primary/50 cursor-col-resize items-center justify-center relative transition-colors group z-10",
                    showLeftPanel && "md:flex"
                )}
                onMouseDown={handleMouseDownLeft}
            >
                <div className="absolute inset-y-0 -left-2 -right-2 z-10 cursor-col-resize" />
                <div className="h-8 w-4 bg-background border rounded flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity shadow-sm">
                    <GripVertical className="h-3 w-3 text-muted-foreground" />
                </div>
            </div>

            {/* Middle Panel */}
            <div
                style={{ width: isMobile ? '100%' : `${showLeftPanel ? middleWidth : middleWidth + leftWidth}%` }}
                className="w-full md:w-auto h-auto md:h-full flex-1 shrink-0 md:overflow-y-auto md:overflow-x-hidden"
            >
                {middlePanel}
            </div>

            {/* Right Resizer Handle (Hidden on mobile) */}
            <div
                className="hidden md:flex w-1 bg-border hover:bg-primary/50 cursor-col-resize items-center justify-center relative transition-colors group z-10"
                onMouseDown={handleMouseDownRight}
            >
                <div className="absolute inset-y-0 -left-2 -right-2 z-10 cursor-col-resize" />
                <div className="h-8 w-4 bg-background border rounded flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity shadow-sm">
                    <GripVertical className="h-3 w-3 text-muted-foreground" />
                </div>
            </div>

            {/* Right Panel */}
            <div
                style={{ width: isMobile ? '100%' : `${rightWidth}%` }}
                className="w-full md:w-auto h-[60vh] max-h-[600px] md:h-full md:max-h-none flex flex-col min-h-0 md:overflow-hidden border-t md:border-t-0 shrink-0"
            >
                {rightPanel}
            </div>
        </div>
    )
}

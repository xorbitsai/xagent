"use client"

import { useState, useRef, useEffect } from "react"
import { cn } from "@/lib/utils"
import { ChevronDown, Check } from "lucide-react"

export interface SelectOption {
  value: string
  label: string
  description?: string
  isDefault?: boolean
  isSmallFast?: boolean
  isVisual?: boolean
  isCompact?: boolean
}

interface SelectProps {
  value?: string
  onValueChange: (value: string) => void
  options?: SelectOption[]
  placeholder?: string
  className?: string
  disabled?: boolean
}

export function Select({ value, onValueChange, options = [], placeholder, className, disabled }: SelectProps) {
  const [open, setOpen] = useState(false)
  const [dropdownDirection, setDropdownDirection] = useState<'down' | 'up'>('down')
  const buttonRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // 处理点击外部关闭下拉框
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setOpen(false)
      }
    }

    if (open) {
      document.addEventListener("mousedown", handleClickOutside)
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside)
    }
  }, [open])

  // 检查下拉菜单应该向上还是向下展开
  useEffect(() => {
    if (open && buttonRef.current) {
      const buttonRect = buttonRef.current.getBoundingClientRect()
      const spaceBelow = window.innerHeight - buttonRect.bottom - 50 // 50px 为预留空间
      const spaceAbove = buttonRect.top - 50

      // 如果下方空间不足且上方空间更充足，则向上展开
      if (spaceBelow < 200 && spaceAbove > spaceBelow) {
        setDropdownDirection('up')
      } else {
        setDropdownDirection('down')
      }
    }
  }, [open])

  const selectedOption = options.find(opt => opt.value === value)

  const handleOptionClick = (optionValue: string) => {
    onValueChange(optionValue)
    setOpen(false)
  }

  return (
    <div ref={containerRef} className={cn("relative", className)}>
      <div
        ref={buttonRef}
        onClick={() => !disabled && setOpen(!open)}
        className={cn(
          "w-full flex items-center justify-between px-3 py-2 text-sm bg-background border border-input rounded-md min-h-[40px] cursor-pointer",
          "hover:bg-accent hover:text-accent-foreground",
          "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
          disabled && "opacity-50 cursor-not-allowed pointer-events-none"
        )}
      >
        <div className="flex items-center gap-2 flex-1 min-w-0">
          {selectedOption ? (
            <div className="flex items-center gap-2 min-w-0 flex-1">
              <span className="font-medium truncate">{selectedOption.label}</span>
              {(selectedOption.isDefault || selectedOption.isSmallFast || selectedOption.isVisual || selectedOption.isCompact) && (
                <div className="flex gap-1 flex-shrink-0">
                  {selectedOption.isDefault && (
                    <span className="text-xs px-1.5 py-0.5 rounded bg-primary/10 text-primary">默认</span>
                  )}
                  {selectedOption.isSmallFast && (
                    <span className="text-xs px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-500">快速</span>
                  )}
                  {selectedOption.isVisual && (
                    <span className="text-xs px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-500">视觉</span>
                  )}
                  {selectedOption.isCompact && (
                    <span className="text-xs px-1.5 py-0.5 rounded bg-green-500/10 text-green-500">长上下文</span>
                  )}
                </div>
              )}
            </div>
          ) : (
            <span className="text-muted-foreground">{placeholder || "请选择..."}</span>
          )}
        </div>
        <ChevronDown className={cn("h-4 w-4 text-muted-foreground transition-transform flex-shrink-0", open && "rotate-180")} />
      </div>

      {open && (
        <div className={cn(
          "absolute left-0 right-0 z-[9999] bg-popover border border-border rounded-md shadow-lg",
          dropdownDirection === 'down' ? "top-full mt-1" : "bottom-full mb-1"
        )}>
          <div className="max-h-60 overflow-auto">
            {options.length === 0 ? (
              <div className="px-3 py-2 text-sm text-muted-foreground">无可用选项</div>
            ) : (
              options.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => handleOptionClick(option.value)}
                  className={cn(
                    "w-full px-3 py-2 text-sm text-left hover:bg-accent hover:text-accent-foreground",
                    "border-b border-border last:border-b-0 transition-colors",
                    value === option.value && "bg-accent text-accent-foreground"
                  )}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2 min-w-0 flex-1">
                      <span className="font-medium truncate">{option.label}</span>
                      {(option.isDefault || option.isSmallFast || option.isVisual || option.isCompact) && (
                        <div className="flex gap-1 flex-shrink-0">
                          {option.isDefault && (
                            <span className="text-xs px-1.5 py-0.5 rounded bg-primary/10 text-primary">默认</span>
                          )}
                          {option.isSmallFast && (
                            <span className="text-xs px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-500">快速</span>
                          )}
                          {option.isVisual && (
                            <span className="text-xs px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-500">视觉</span>
                          )}
                          {option.isCompact && (
                            <span className="text-xs px-1.5 py-0.5 rounded bg-green-500/10 text-green-500">长上下文</span>
                          )}
                        </div>
                      )}
                    </div>
                    {value === option.value && (
                      <Check className="h-4 w-4 text-primary flex-shrink-0" />
                    )}
                  </div>
                  {option.description && (
                    <div className="text-xs text-muted-foreground mt-1 truncate">{option.description}</div>
                  )}
                </button>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export {
  Select as SelectRadix,
  SelectGroup,
  SelectValue,
  SelectTrigger,
  SelectContent,
  SelectLabel,
  SelectItem,
  SelectSeparator,
  SelectScrollUpButton,
  SelectScrollDownButton,
} from "./select-radix"

"use client"

import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react"
import {
  Bot,
  Check,
  ChevronDown,
  ChevronUp,
  CornerDownLeft,
  Cpu,
  Database,
  Gauge,
  ListOrdered,
  LogIn,
  LogOut,
  MessageCircle,
  MessageSquare,
  Pencil,
  Plug,
  Plus,
  Sparkles,
  Wrench,
  X,
  Zap,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { useI18n } from "@/contexts/i18n-context"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Popover, PopoverAnchor, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import {
  addPlanStep,
  deletePlanStep,
  movePlanStep,
  parseInstructionSteps,
  updatePlanStep,
} from "@/lib/agent-flow-plan"

type CapKey = "kb" | "skills" | "tools" | "connectors"

interface FlowOption {
  value: string
  label: string
}

export interface AgentFlowTriggerRow {
  key: string
  label: string
  description: string
}

export interface AgentFlowViewProps {
  name: string
  modelLabel: string
  executionMode: "flash" | "balanced" | "think"
  instructions: string
  onInstructionsChange: (value: string) => void
  readOnly?: boolean
  maxInstructionsLength?: number

  kbSelected: string[]
  kbOptions: FlowOption[]
  onKbChange: (values: string[]) => void

  skillsSelected: string[]
  skillOptions: FlowOption[]
  onSkillsChange: (values: string[]) => void

  toolsSelected: string[]
  toolOptions: FlowOption[]
  onToolsChange: (values: string[]) => void

  connectorNames: string[]
  onOpenConnectors: () => void

  triggerRows: AgentFlowTriggerRow[]
  onOpenTriggers: () => void

  promptCount: number
}

const MODE_META: Record<AgentFlowViewProps["executionMode"], { icon: React.ElementType }> = {
  flash: { icon: Zap },
  balanced: { icon: Gauge },
  think: { icon: Bot },
}

export function AgentFlowView({
  name,
  modelLabel,
  executionMode,
  instructions,
  onInstructionsChange,
  readOnly,
  maxInstructionsLength,
  kbSelected,
  kbOptions,
  onKbChange,
  skillsSelected,
  skillOptions,
  onSkillsChange,
  toolsSelected,
  toolOptions,
  onToolsChange,
  connectorNames,
  onOpenConnectors,
  triggerRows,
  onOpenTriggers,
  promptCount,
}: AgentFlowViewProps) {
  const { t } = useI18n()
  const plan = parseInstructionSteps(instructions)

  const containerRef = useRef<HTMLDivElement>(null)
  const inputNodeRef = useRef<HTMLDivElement>(null)
  const brainNodeRef = useRef<HTMLDivElement>(null)
  const outputNodeRef = useRef<HTMLDivElement>(null)
  const capRefs = useRef<Record<CapKey, HTMLDivElement | null>>({
    kb: null,
    skills: null,
    tools: null,
    connectors: null,
  })

  // ── Node dragging — cosmetic per-session arrangement; layout & edges stay
  // derived from the config. "Reset layout" clears back to auto-layout. ──
  const [offsets, setOffsets] = useState<Record<string, { x: number; y: number }>>({})
  const [draggingId, setDraggingId] = useState<string | null>(null)
  const dragRef = useRef<{ id: string; startX: number; startY: number; baseX: number; baseY: number; moved: boolean } | null>(null)
  const suppressClickRef = useRef(false)
  const rafPendingRef = useRef(false)

  const [execLines, setExecLines] = useState<{ x1: number; y1: number; x2: number; y2: number }[]>([])
  const [capPaths, setCapPaths] = useState<{ key: CapKey; d: string }[]>([])

  const recomputeConnectors = useCallback(() => {
    const container = containerRef.current
    const input = inputNodeRef.current
    const brain = brainNodeRef.current
    const output = outputNodeRef.current
    if (!container || !input || !brain || !output) return

    const cRect = container.getBoundingClientRect()
    const iR = input.getBoundingClientRect()
    const bR = brain.getBoundingClientRect()
    const oR = output.getBoundingClientRect()
    const cx = (rect: DOMRect) => rect.left - cRect.left + rect.width / 2
    const cy = (rect: DOMRect) => rect.top - cRect.top + rect.height / 2

    setExecLines([
      { x1: cx(iR), y1: iR.bottom - cRect.top, x2: cx(bR), y2: bR.top - cRect.top },
      { x1: cx(bR), y1: bR.bottom - cRect.top, x2: cx(oR), y2: oR.top - cRect.top },
    ])

    const bx = bR.right - cRect.left + 5
    const by = cy(bR)
    const caps: { key: CapKey; d: string }[] = []
    ;(["kb", "skills", "tools", "connectors"] as CapKey[]).forEach((key) => {
      const el = capRefs.current[key]
      if (!el) return
      const capR = el.getBoundingClientRect()
      const ex = capR.left - cRect.left - 4
      const ey = cy(capR)
      const midX = (bx + ex) / 2
      caps.push({ key, d: `M ${bx} ${by} C ${midX} ${by}, ${midX} ${ey}, ${ex} ${ey}` })
    })
    setCapPaths(caps)
  }, [])

  const scheduleConnectors = useCallback(() => {
    if (rafPendingRef.current) return
    rafPendingRef.current = true
    requestAnimationFrame(() => {
      rafPendingRef.current = false
      recomputeConnectors()
    })
  }, [recomputeConnectors])

  useLayoutEffect(() => {
    recomputeConnectors()
    let innerRafId: number | undefined
    const outerRafId = requestAnimationFrame(() => {
      innerRafId = requestAnimationFrame(recomputeConnectors)
    })
    return () => {
      cancelAnimationFrame(outerRafId)
      if (innerRafId !== undefined) cancelAnimationFrame(innerRafId)
    }
    // Re-measure whenever anything that changes node size/position renders.
    // `instructions` covers step text edits that reflow the Agent node's
    // height (shifting Output below it) without changing the step count.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recomputeConnectors, instructions, plan.steps.length, kbSelected.length, skillsSelected.length, toolsSelected.length, connectorNames.length, triggerRows.length, promptCount, name, modelLabel, executionMode])

  useEffect(() => {
    const el = containerRef.current
    if (!el || typeof ResizeObserver === "undefined") return
    const observer = new ResizeObserver(() => scheduleConnectors())
    observer.observe(el)
    return () => observer.disconnect()
  }, [scheduleConnectors])

  useEffect(() => {
    window.addEventListener("resize", scheduleConnectors)
    return () => window.removeEventListener("resize", scheduleConnectors)
  }, [scheduleConnectors])

  useEffect(() => {
    const handleMove = (e: PointerEvent) => {
      const drag = dragRef.current
      if (!drag) return
      const dx = e.clientX - drag.startX
      const dy = e.clientY - drag.startY
      if (!drag.moved) {
        if (Math.hypot(dx, dy) < 4) return
        drag.moved = true
        setDraggingId(drag.id)
      }
      setOffsets((prev) => ({ ...prev, [drag.id]: { x: drag.baseX + dx, y: drag.baseY + dy } }))
      scheduleConnectors()
    }
    const endDrag = () => {
      const drag = dragRef.current
      if (drag?.moved) {
        suppressClickRef.current = true
        setTimeout(() => {
          suppressClickRef.current = false
        }, 0)
      }
      setDraggingId(null)
      dragRef.current = null
    }
    window.addEventListener("pointermove", handleMove)
    window.addEventListener("pointerup", endDrag)
    window.addEventListener("pointercancel", endDrag)
    return () => {
      window.removeEventListener("pointermove", handleMove)
      window.removeEventListener("pointerup", endDrag)
      window.removeEventListener("pointercancel", endDrag)
    }
  }, [scheduleConnectors])

  const handleNodePointerDown = (id: string) => (e: React.PointerEvent<HTMLDivElement>) => {
    if (readOnly || e.button !== 0) return
    const target = e.target as HTMLElement
    if (target.closest("button, input, textarea, [data-plan-step], [data-no-drag]")) return
    const o = offsets[id] || { x: 0, y: 0 }
    dragRef.current = { id, startX: e.clientX, startY: e.clientY, baseX: o.x, baseY: o.y, moved: false }
  }

  const suppressableClick = (fn: () => void) => () => {
    if (suppressClickRef.current) return
    fn()
  }

  const resetLayout = () => {
    setOffsets({})
    scheduleConnectors()
  }

  const nodeStyle = (id: string): React.CSSProperties => {
    const o = offsets[id]
    return o ? { transform: `translate(${o.x}px, ${o.y}px)` } : {}
  }

  // ── Plan step inline editing ──────────────────────────────────
  const [editingStepIdx, setEditingStepIdx] = useState<number | null>(null)
  const [stepDraft, setStepDraft] = useState("")
  // Move/reorder/remove buttons fire on the same step (or shift indices for
  // others) while a step is being edited. Their mousedown blurs the <input>
  // first, which would otherwise commit a stale edit right before the
  // button's own onClick applies a move/delete against the same stale
  // `instructions` snapshot — silently discarding whichever change lands
  // second. Suppress the blur-triggered commit when a structural action is
  // about to run instead of trying to reconcile the two.
  const ignoreBlurRef = useRef(false)

  const commitStep = (idx: number) => {
    if (editingStepIdx !== idx) return
    if (ignoreBlurRef.current) {
      ignoreBlurRef.current = false
      setEditingStepIdx(null)
      return
    }
    const orig = plan.steps[idx]?.text ?? ""
    const value = stepDraft.trim()
    if (value && value !== orig) {
      onInstructionsChange(updatePlanStep(instructions, idx, value))
    }
    setEditingStepIdx(null)
  }

  const startEditStep = (idx: number) => {
    if (readOnly) return
    setStepDraft(plan.steps[idx]?.text ?? "")
    setEditingStepIdx(idx)
  }

  const handleAddStep = () => {
    const placeholder = t("builds.editor.flow.agent.newStep")
    const next = addPlanStep(instructions, placeholder)
    onInstructionsChange(next)
    // Seed the editor directly from the newly parsed plan rather than
    // going through startEditStep, which closes over the pre-add `plan`
    // and would initialize stepDraft from the stale snapshot.
    requestAnimationFrame(() => {
      const newPlan = parseInstructionSteps(next)
      const newIdx = newPlan.steps.length - 1
      const newStep = newPlan.steps[newIdx]
      if (newStep) {
        setStepDraft(newStep.text)
        setEditingStepIdx(newIdx)
      }
    })
  }

  // ── Instructions edit popover ─────────────────────────────────
  const [instructionsDraft, setInstructionsDraft] = useState(instructions)
  const [instructionsPopoverOpen, setInstructionsPopoverOpen] = useState(false)

  const openInstructionsPopover = () => {
    setInstructionsDraft(instructions)
    setInstructionsPopoverOpen(true)
  }

  const saveInstructionsPopover = () => {
    const value = maxInstructionsLength ? instructionsDraft.slice(0, maxInstructionsLength) : instructionsDraft
    onInstructionsChange(value)
    setInstructionsPopoverOpen(false)
  }

  const modeMeta = MODE_META[executionMode] || MODE_META.balanced
  const ModeIcon = modeMeta.icon
  const modeLabel = t(`builds.configForm.executionMode.${executionMode}.title`)

  const hasOffsets = Object.keys(offsets).length > 0

  return (
    <div className="relative flex flex-1 min-h-0 flex-col overflow-hidden">
      <div
        className="flex-1 overflow-auto"
        style={{
          backgroundImage: "radial-gradient(circle, hsl(var(--border)) 1px, transparent 1px)",
          backgroundSize: "22px 22px",
        }}
      >
        <div
          ref={containerRef}
          className="relative mx-auto flex min-w-max items-start justify-center gap-16 px-10 py-10"
          style={{ paddingBottom: 90 }}
        >
          <svg className="pointer-events-none absolute inset-0 h-full w-full overflow-visible">
            <defs>
              <marker id="cafl-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto" markerUnits="userSpaceOnUse">
                <path d="M0,0 L8,3 L0,6 Z" className="fill-border" />
              </marker>
              <marker id="cafl-dot" markerWidth="6" markerHeight="6" refX="3" refY="3" markerUnits="userSpaceOnUse">
                <circle cx="3" cy="3" r="2.4" className="fill-border" />
              </marker>
            </defs>
            {execLines.map((l, i) => (
              <line
                key={i}
                x1={l.x1}
                y1={l.y1}
                x2={l.x2}
                y2={l.y2}
                className="stroke-border"
                strokeWidth={1.5}
                markerEnd="url(#cafl-arrow)"
              />
            ))}
            {capPaths.map((p) => (
              <path
                key={p.key}
                d={p.d}
                fill="none"
                className="stroke-border"
                strokeWidth={1.5}
                strokeDasharray="4 4"
                markerEnd="url(#cafl-dot)"
              />
            ))}
          </svg>

          {/* Main column: Input -> Agent -> Output */}
          <div className="relative z-[1] flex flex-col items-center gap-16">
            {/* Input node */}
            <div
              ref={inputNodeRef}
              id="cafl-node-input"
              onPointerDown={handleNodePointerDown("cafl-node-input")}
              style={nodeStyle("cafl-node-input")}
              className={cn(
                "flex w-[300px] cursor-grab flex-col gap-2.5 rounded-2xl border bg-card p-4 shadow-sm transition-colors hover:border-primary/40 hover:shadow-md",
                draggingId === "cafl-node-input" && "cursor-grabbing border-primary shadow-lg transition-none",
              )}
            >
              <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wide text-muted-foreground">
                <LogIn className="h-3 w-3" />
                {t("builds.editor.flow.input.eyebrow")}
                <button
                  type="button"
                  disabled={readOnly}
                  onClick={suppressableClick(onOpenTriggers)}
                  className="ml-auto flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-semibold normal-case tracking-normal text-muted-foreground hover:bg-muted hover:text-foreground disabled:pointer-events-none disabled:opacity-50"
                >
                  <Plus className="h-3 w-3" /> {t("builds.editor.flow.input.trigger")}
                </button>
              </div>
              <div className="flex items-center gap-2.5">
                <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <MessageSquare className="h-3.5 w-3.5" />
                </div>
                <div className="min-w-0">
                  <div className="text-[12.5px] font-semibold">{t("builds.editor.flow.input.userMessage")}</div>
                  <div className="text-[11px] text-muted-foreground">{t("builds.editor.flow.input.userMessageDesc")}</div>
                </div>
              </div>
              {triggerRows.map((row) => (
                <div key={row.key} className="flex items-center gap-2.5">
                  <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-amber-500/10 text-amber-600 dark:text-amber-400">
                    <Zap className="h-3.5 w-3.5" />
                  </div>
                  <div className="min-w-0">
                    <div className="text-[12.5px] font-semibold">{row.label}</div>
                    <div className="text-[11px] text-muted-foreground">{row.description}</div>
                  </div>
                </div>
              ))}
            </div>

            {/* Agent / brain node */}
            <div
              ref={brainNodeRef}
              id="cafl-node-brain"
              onPointerDown={handleNodePointerDown("cafl-node-brain")}
              style={nodeStyle("cafl-node-brain")}
              className={cn(
                "flex w-[330px] cursor-grab flex-col gap-2.5 rounded-2xl border-2 border-primary bg-card p-4 shadow-[0_0_0_5px_hsl(var(--primary)/0.08)] transition-colors hover:shadow-md",
                draggingId === "cafl-node-brain" && "cursor-grabbing shadow-lg transition-none",
              )}
            >
              <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wide text-muted-foreground">
                <Bot className="h-3 w-3" />
                {t("builds.editor.flow.agent.eyebrow")}
                <Popover open={instructionsPopoverOpen} onOpenChange={setInstructionsPopoverOpen}>
                  <PopoverTrigger asChild>
                    <button
                      type="button"
                      data-no-drag
                      disabled={readOnly}
                      onClick={suppressableClick(openInstructionsPopover)}
                      className="ml-auto flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-semibold normal-case tracking-normal text-muted-foreground hover:bg-muted hover:text-foreground disabled:pointer-events-none disabled:opacity-50"
                    >
                      <Pencil className="h-3 w-3" /> {t("builds.editor.flow.agent.edit")}
                    </button>
                  </PopoverTrigger>
                  <PopoverContent className="w-[340px] space-y-2.5" align="start">
                    <div className="flex items-center gap-1.5 text-xs font-bold">
                      <Pencil className="h-3 w-3" /> {t("builds.editor.flow.instructionsPopover.title")}
                    </div>
                    <Textarea
                      value={instructionsDraft}
                      onChange={(e) => setInstructionsDraft(e.target.value)}
                      className="min-h-[150px] font-mono text-xs"
                      autoFocus
                    />
                    <div className="text-[11px] text-muted-foreground">
                      {t("builds.editor.flow.instructionsPopover.note")}
                    </div>
                    <div className="flex justify-end gap-2">
                      <Button size="sm" variant="outline" onClick={() => setInstructionsPopoverOpen(false)}>
                        {t("builds.editor.flow.instructionsPopover.cancel")}
                      </Button>
                      <Button size="sm" onClick={saveInstructionsPopover}>
                        {t("builds.editor.flow.instructionsPopover.save")}
                      </Button>
                    </div>
                  </PopoverContent>
                </Popover>
              </div>

              <div className="flex items-center gap-2.5">
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary text-sm font-extrabold text-primary-foreground">
                  {(name || "A").trim().charAt(0).toUpperCase()}
                </div>
                <div className="min-w-0">
                  <div className="truncate text-[13.5px] font-semibold">
                    {name || t("builds.editor.flow.agent.untitled")}
                  </div>
                  <div className="mt-1 flex flex-wrap gap-1.5">
                    <span className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-[10.5px] font-semibold text-primary">
                      <Cpu className="h-2.5 w-2.5" /> {modelLabel}
                    </span>
                    <span className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-[10.5px] font-semibold text-primary">
                      <ModeIcon className="h-2.5 w-2.5" /> {modeLabel}
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex flex-col gap-1.5 border-t pt-2.5">
                <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wide text-muted-foreground">
                  <ListOrdered className="h-2.5 w-2.5" />
                  {plan.steps.length
                    ? plan.explicit
                      ? t("builds.editor.flow.agent.planFromInstructions")
                      : t("builds.editor.flow.agent.planInferred")
                    : t("builds.editor.flow.agent.plan")}
                </div>

                {plan.steps.length === 0 && (
                  <div className="text-[11px] italic leading-relaxed text-muted-foreground">
                    {t("builds.editor.flow.agent.emptyPlan")}
                  </div>
                )}

                <div data-no-drag className="flex max-h-[360px] flex-col gap-1.5 overflow-y-auto pr-0.5">
                {plan.steps.map((step, idx) => (
                  <div
                    key={idx}
                    data-plan-step
                    onClick={() => editingStepIdx !== idx && startEditStep(idx)}
                    className={cn(
                      "group relative flex items-start gap-2 rounded-lg border bg-background px-2 py-1.5 transition-colors hover:border-primary/40",
                      readOnly ? "cursor-default" : "cursor-text",
                    )}
                  >
                    <div
                      className={cn(
                        "mt-px flex h-[18px] w-[18px] shrink-0 items-center justify-center rounded-full bg-primary/10 text-[10px] font-bold text-primary",
                        !plan.explicit && "border border-dashed border-primary/50 bg-transparent",
                      )}
                    >
                      {idx + 1}
                    </div>
                    {editingStepIdx === idx ? (
                      <input
                        autoFocus
                        value={stepDraft}
                        onChange={(e) => setStepDraft(e.target.value)}
                        onBlur={() => commitStep(idx)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault()
                            e.currentTarget.blur()
                          }
                          if (e.key === "Escape") {
                            e.preventDefault()
                            // Blur (rather than clearing state directly) so the
                            // onBlur handler consumes and resets ignoreBlurRef —
                            // a stale true flag would swallow the next commit.
                            ignoreBlurRef.current = true
                            e.currentTarget.blur()
                          }
                        }}
                        className="min-w-0 flex-1 border-none bg-transparent p-0 text-[11.5px] leading-relaxed text-foreground outline-none"
                      />
                    ) : (
                      <div className="min-w-0 flex-1 break-words text-[11.5px] leading-relaxed">{step.text}</div>
                    )}
                    {!readOnly && editingStepIdx === null && <div className="absolute -top-[11px] right-1.5 z-[2] hidden gap-px rounded-md border bg-card p-px shadow-sm group-hover:inline-flex">
                      <button
                        type="button"
                        title={t("builds.editor.flow.agent.moveUp")}
                        onMouseDown={() => {
                          ignoreBlurRef.current = true
                        }}
                        onClick={(e) => {
                          e.stopPropagation()
                          ignoreBlurRef.current = false
                          onInstructionsChange(movePlanStep(instructions, idx, -1))
                        }}
                        className="flex h-[18px] w-5 items-center justify-center rounded text-muted-foreground hover:bg-muted hover:text-foreground"
                      >
                        <ChevronUp className="h-2.5 w-2.5" />
                      </button>
                      <button
                        type="button"
                        title={t("builds.editor.flow.agent.moveDown")}
                        onMouseDown={() => {
                          ignoreBlurRef.current = true
                        }}
                        onClick={(e) => {
                          e.stopPropagation()
                          ignoreBlurRef.current = false
                          onInstructionsChange(movePlanStep(instructions, idx, 1))
                        }}
                        className="flex h-[18px] w-5 items-center justify-center rounded text-muted-foreground hover:bg-muted hover:text-foreground"
                      >
                        <ChevronDown className="h-2.5 w-2.5" />
                      </button>
                      <button
                        type="button"
                        title={t("builds.editor.flow.agent.removeStep")}
                        onMouseDown={() => {
                          ignoreBlurRef.current = true
                        }}
                        onClick={(e) => {
                          e.stopPropagation()
                          ignoreBlurRef.current = false
                          onInstructionsChange(deletePlanStep(instructions, idx))
                        }}
                        className="flex h-[18px] w-5 items-center justify-center rounded text-muted-foreground hover:bg-muted hover:text-foreground"
                      >
                        <X className="h-2.5 w-2.5" />
                      </button>
                    </div>}
                  </div>
                ))}
                </div>

                {!readOnly && (
                  <button
                    type="button"
                    onClick={handleAddStep}
                    className="inline-flex items-center gap-1 self-start rounded-lg border border-dashed px-2.5 py-1 text-[11px] font-semibold text-muted-foreground hover:border-primary/50 hover:text-primary"
                  >
                    <Plus className="h-3 w-3" /> {t("builds.editor.flow.agent.addStep")}
                  </button>
                )}
              </div>
            </div>

            {/* Output node */}
            <div
              ref={outputNodeRef}
              id="cafl-node-output"
              onPointerDown={handleNodePointerDown("cafl-node-output")}
              style={nodeStyle("cafl-node-output")}
              className={cn(
                "flex w-[300px] cursor-grab flex-col gap-2.5 rounded-2xl border bg-card p-4 shadow-sm transition-colors hover:border-primary/40 hover:shadow-md",
                draggingId === "cafl-node-output" && "cursor-grabbing border-primary shadow-lg transition-none",
              )}
            >
              <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wide text-muted-foreground">
                <LogOut className="h-3 w-3" />
                {t("builds.editor.flow.output.eyebrow")}
              </div>
              <div className="flex items-center gap-2.5">
                <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                  <CornerDownLeft className="h-3.5 w-3.5" />
                </div>
                <div className="min-w-0">
                  <div className="text-[12.5px] font-semibold">{t("builds.editor.flow.output.reply")}</div>
                  <div className="text-[11px] text-muted-foreground">{t("builds.editor.flow.output.replyDesc")}</div>
                </div>
              </div>
              {promptCount > 0 && (
                <div className="flex items-center gap-2.5">
                  <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                    <MessageCircle className="h-3.5 w-3.5" />
                  </div>
                  <div className="min-w-0">
                    <div className="text-[12.5px] font-semibold">
                      {t("builds.editor.flow.output.prompts", { count: promptCount })}
                    </div>
                    <div className="text-[11px] text-muted-foreground">{t("builds.editor.flow.output.promptsDesc")}</div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Side column: capability nodes */}
          <div className="relative z-[1] mt-[140px] flex flex-col gap-4">
            <CapNode
              nodeKey="kb"
              nodeRef={(el) => (capRefs.current.kb = el)}
              icon={<Database className="h-3 w-3" />}
              title={t("builds.editor.flow.caps.knowledgeBase")}
              hint={t("builds.editor.flow.caps.attachKb")}
              items={kbSelected}
              readOnly={readOnly}
              style={nodeStyle("cafl-cap-kb")}
              dragging={draggingId === "cafl-cap-kb"}
              onPointerDown={handleNodePointerDown("cafl-cap-kb")}
              suppressableClick={suppressableClick}
              options={kbOptions}
              selected={kbSelected}
              onChange={onKbChange}
            />

            <CapNode
              nodeKey="skills"
              nodeRef={(el) => (capRefs.current.skills = el)}
              icon={<Sparkles className="h-3 w-3" />}
              title={t("builds.editor.flow.caps.skills")}
              hint={t("builds.editor.flow.caps.addSkills")}
              items={skillsSelected}
              readOnly={readOnly}
              style={nodeStyle("cafl-cap-skills")}
              dragging={draggingId === "cafl-cap-skills"}
              onPointerDown={handleNodePointerDown("cafl-cap-skills")}
              suppressableClick={suppressableClick}
              options={skillOptions}
              selected={skillsSelected}
              onChange={onSkillsChange}
            />

            <CapNode
              nodeKey="tools"
              nodeRef={(el) => (capRefs.current.tools = el)}
              icon={<Wrench className="h-3 w-3" />}
              title={t("builds.editor.flow.caps.tools")}
              hint={t("builds.editor.flow.caps.addToolCategories")}
              items={toolsSelected.map((v) => toolOptions.find((o) => o.value === v)?.label || v)}
              readOnly={readOnly}
              style={nodeStyle("cafl-cap-tools")}
              dragging={draggingId === "cafl-cap-tools"}
              onPointerDown={handleNodePointerDown("cafl-cap-tools")}
              suppressableClick={suppressableClick}
              options={toolOptions}
              selected={toolsSelected}
              onChange={onToolsChange}
            />

            <CapNode
              nodeKey="connectors"
              nodeRef={(el) => (capRefs.current.connectors = el)}
              icon={<Plug className="h-3 w-3" />}
              title={t("builds.editor.flow.caps.connectors")}
              hint={t("builds.editor.flow.caps.connectIntegrations")}
              items={connectorNames}
              readOnly={readOnly}
              style={nodeStyle("cafl-cap-connectors")}
              dragging={draggingId === "cafl-cap-connectors"}
              onPointerDown={handleNodePointerDown("cafl-cap-connectors")}
              suppressableClick={suppressableClick}
              onEdit={onOpenConnectors}
            />
          </div>
        </div>
      </div>

      <div className="pointer-events-none absolute bottom-3.5 left-1/2 z-10 flex max-w-[92%] -translate-x-1/2 items-center gap-2 whitespace-nowrap rounded-full border bg-card px-4 py-1.5 text-xs text-muted-foreground shadow-md">
        <span className="truncate">{t("builds.editor.flow.hintBar")}</span>
        {hasOffsets && (
          <button
            type="button"
            onClick={resetLayout}
            className="pointer-events-auto rounded-md px-1.5 py-0.5 font-semibold text-primary hover:bg-primary/10"
          >
            {t("builds.editor.flow.resetLayout")}
          </button>
        )}
      </div>
    </div>
  )
}

function CapNode({
  nodeKey,
  nodeRef,
  icon,
  title,
  hint,
  items,
  readOnly,
  style,
  dragging,
  onPointerDown,
  suppressableClick,
  onEdit,
  options,
  selected,
  onChange,
}: {
  nodeKey: CapKey
  nodeRef: (el: HTMLDivElement | null) => void
  icon: React.ReactNode
  title: string
  hint: string
  items: string[]
  readOnly?: boolean
  style: React.CSSProperties
  dragging: boolean
  onPointerDown: (e: React.PointerEvent<HTMLDivElement>) => void
  suppressableClick: (fn: () => void) => () => void
  onEdit?: () => void
  options?: FlowOption[]
  selected?: string[]
  onChange?: (values: string[]) => void
}) {
  const [open, setOpen] = useState(false)
  const hasItems = items.length > 0

  const cardClassName = cn(
    "flex w-[230px] flex-col gap-2 rounded-2xl border bg-card p-3.5 shadow-sm transition-colors hover:border-primary/40 hover:shadow-md",
    !readOnly && "cursor-pointer",
    !hasItems && "border-dashed bg-transparent",
    dragging && "cursor-grabbing border-primary shadow-lg transition-none",
  )

  const body = (
    <>
      <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wide text-muted-foreground">
        {icon}
        {title}
        {!readOnly && (
          <span className="ml-auto flex items-center text-muted-foreground">
            {hasItems ? <Pencil className="h-3 w-3" /> : <Plus className="h-3 w-3" />}
          </span>
        )}
      </div>

      {hasItems ? (
        <div className="flex flex-col gap-1">
          {items.map((item, i) => (
            <div key={i} className="flex items-center gap-1.5 text-[11.5px]">
              <span className="h-1 w-1 shrink-0 rounded-full bg-emerald-500" />
              <span className="truncate">{item}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
          <Plus className="h-3 w-3" /> {hint}
        </div>
      )}
    </>
  )

  if (readOnly) {
    return (
      <div id={`cafl-cap-${nodeKey}`} ref={nodeRef} onPointerDown={onPointerDown} style={style} className={cardClassName}>
        {body}
      </div>
    )
  }

  if (onEdit) {
    return (
      <div
        id={`cafl-cap-${nodeKey}`}
        ref={nodeRef}
        onPointerDown={onPointerDown}
        style={style}
        className={cardClassName}
        role="button"
        tabIndex={0}
        onClick={suppressableClick(onEdit)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault()
            onEdit()
          }
        }}
      >
        {body}
      </div>
    )
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverAnchor asChild>
        <div
          id={`cafl-cap-${nodeKey}`}
          ref={nodeRef}
          onPointerDown={onPointerDown}
          style={style}
          className={cardClassName}
          role="button"
          tabIndex={0}
          onClick={suppressableClick(() => setOpen(true))}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault()
              setOpen(true)
            }
          }}
        >
          {body}
        </div>
      </PopoverAnchor>
      <PopoverContent align="start" className="max-h-72 w-64 overflow-y-auto p-1">
        {!options || options.length === 0 ? (
          <div className="px-2 py-1.5 text-xs text-muted-foreground">{hint}</div>
        ) : (
          options.map((opt) => {
            const isSelected = selected?.includes(opt.value)
            return (
              <button
                key={opt.value}
                type="button"
                onClick={() => {
                  const current = selected || []
                  onChange?.(isSelected ? current.filter((v) => v !== opt.value) : [...current, opt.value])
                }}
                className={cn(
                  "flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-left text-[12.5px] transition-colors hover:bg-muted",
                  isSelected && "bg-primary/5",
                )}
              >
                <span className="min-w-0 flex-1 truncate">{opt.label}</span>
                {isSelected && <Check className="h-3.5 w-3.5 shrink-0 text-primary" />}
              </button>
            )
          })
        )}
      </PopoverContent>
    </Popover>
  )
}

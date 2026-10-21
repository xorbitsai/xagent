"use client"

import React from "react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { InfoTooltip } from "@/components/ui/tooltip"
import { useI18n } from "@/contexts/i18n-context"
import { cn } from "@/lib/utils"
import {
  PROMPT_CONTEXT_DESCRIPTION_MAX_LENGTH,
  PROMPT_CONTEXT_MAX_FIELDS,
  PROMPT_CONTEXT_MAX_LENGTH,
  PROMPT_CONTEXT_MIN_LENGTH,
  newPromptContextFieldDraft,
  validatePromptContextFields,
  type PromptContextFieldDraft,
} from "@/lib/prompt-context-fields"

interface PromptContextFieldsEditorProps {
  value: PromptContextFieldDraft[]
  onChange: (value: PromptContextFieldDraft[]) => void
  // A freshly added row is blank by design; flag it only once a save was attempted.
  revealEmpty: boolean
}

// Column titles live in the first row's cells so they wrap with their inputs in a narrow panel.
const NAME_CELL = "min-w-0 flex-[1_1_9rem]"
const LENGTH_CELL = "w-24 flex-none"
const DESCRIPTION_CELL = "min-w-0 flex-[2_1_12rem]"
const COLUMN_TITLE = "mb-1 block text-xs font-medium text-muted-foreground"
const FIELDS = "builds.configForm.promptContextFields"

export function PromptContextFieldsEditor({
  value,
  onChange,
  revealEmpty,
}: PromptContextFieldsEditorProps) {
  const { t } = useI18n()
  const validation = validatePromptContextFields(value)
  const atLimit = value.length >= PROMPT_CONTEXT_MAX_FIELDS

  const updateRow = (index: number, patch: Partial<PromptContextFieldDraft>) => {
    onChange(value.map((row, i) => (i === index ? { ...row, ...patch } : row)))
  }

  return (
    <div className="space-y-2" data-testid="prompt-context-fields">
      <div className="flex items-center gap-1.5">
        <Label>{t(`${FIELDS}.label`)}</Label>
        <InfoTooltip
          content={
            <span className="block space-y-1">
              <span className="block">{t(`${FIELDS}.tooltip.declaredOnly`)}</span>
              <span className="block">{t(`${FIELDS}.tooltip.overLimit`)}</span>
              <span className="block">{t(`${FIELDS}.tooltip.instructions`)}</span>
              <span className="block">{t(`${FIELDS}.tooltip.nameRule`)}</span>
            </span>
          }
        />
      </div>
      <p className="text-xs text-muted-foreground mb-2">{t(`${FIELDS}.description`)}</p>
      <div className="space-y-3">
        {value.map((row, index) => {
          const errors = validation.rows[index] ?? {}
          const nameError =
            errors.name === "required" && !revealEmpty ? undefined : errors.name
          const nameErrorId = `prompt-context-field-${index}-name-error`
          const lengthErrorId = `prompt-context-field-${index}-length-error`
          const descriptionErrorId = `prompt-context-field-${index}-description-error`
          return (
            <div key={index} className="space-y-1">
              <div className="flex flex-wrap items-end gap-2">
                <div className={NAME_CELL}>
                  {index === 0 && (
                    <span className={COLUMN_TITLE} aria-hidden="true">
                      {t(`${FIELDS}.nameColumn`)}
                    </span>
                  )}
                  <Input
                    value={row.name}
                    onChange={(e) => updateRow(index, { name: e.target.value })}
                    placeholder={t(`${FIELDS}.namePlaceholder`)}
                    aria-label={t(`${FIELDS}.nameLabel`, { index: index + 1 })}
                    aria-invalid={nameError ? true : undefined}
                    aria-describedby={nameError ? nameErrorId : undefined}
                    className={cn("font-mono", nameError && "border-destructive")}
                  />
                </div>
                <div className={LENGTH_CELL}>
                  {index === 0 && (
                    <span className={COLUMN_TITLE} aria-hidden="true">
                      {t(`${FIELDS}.maxLengthColumn`)}
                    </span>
                  )}
                  <Input
                    type="number"
                    min={PROMPT_CONTEXT_MIN_LENGTH}
                    max={PROMPT_CONTEXT_MAX_LENGTH}
                    step={1}
                    value={row.maxLength}
                    onChange={(e) => updateRow(index, { maxLength: e.target.value })}
                    aria-label={t(`${FIELDS}.maxLengthLabel`, { index: index + 1 })}
                    aria-invalid={errors.maxLength ? true : undefined}
                    aria-describedby={errors.maxLength ? lengthErrorId : undefined}
                    className={cn(errors.maxLength && "border-destructive")}
                  />
                </div>
                <div className={DESCRIPTION_CELL}>
                  {index === 0 && (
                    <span className={COLUMN_TITLE} aria-hidden="true">
                      {t(`${FIELDS}.descriptionColumn`)}
                    </span>
                  )}
                  <Input
                    value={row.description}
                    onChange={(e) => updateRow(index, { description: e.target.value })}
                    placeholder={t(`${FIELDS}.descriptionPlaceholder`)}
                    aria-label={t(`${FIELDS}.descriptionLabel`, { index: index + 1 })}
                    aria-invalid={errors.description ? true : undefined}
                    aria-describedby={errors.description ? descriptionErrorId : undefined}
                    className={cn(errors.description && "border-destructive")}
                  />
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="flex-none"
                  onClick={() => onChange(value.filter((_, i) => i !== index))}
                >
                  {t(`${FIELDS}.delete`)}
                </Button>
              </div>
              {nameError && (
                <p id={nameErrorId} className="text-xs text-destructive">
                  {t(`${FIELDS}.errors.name.${nameError}`)}
                </p>
              )}
              {errors.maxLength && (
                <p id={lengthErrorId} className="text-xs text-destructive">
                  {t(`${FIELDS}.errors.maxLength`, {
                    min: PROMPT_CONTEXT_MIN_LENGTH,
                    max: PROMPT_CONTEXT_MAX_LENGTH,
                  })}
                </p>
              )}
              {errors.description && (
                <p id={descriptionErrorId} className="text-xs text-destructive">
                  {t(`${FIELDS}.errors.description.${errors.description}`, {
                    max: PROMPT_CONTEXT_DESCRIPTION_MAX_LENGTH,
                  })}
                </p>
              )}
            </div>
          )
        })}
        {validation.tooMany && (
          <p className="text-xs text-destructive">
            {t(`${FIELDS}.errors.tooMany`, { max: PROMPT_CONTEXT_MAX_FIELDS })}
          </p>
        )}
        <div className="flex items-center gap-3">
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={atLimit}
            onClick={() => onChange([...value, newPromptContextFieldDraft()])}
          >
            {t(`${FIELDS}.add`)}
          </Button>
          <span className="text-xs text-muted-foreground">
            {t(`${FIELDS}.limit`, { count: value.length, max: PROMPT_CONTEXT_MAX_FIELDS })}
          </span>
        </div>
      </div>
    </div>
  )
}

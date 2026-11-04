"use client";

import type { ComponentType } from "react";

export type TaskRuntimeExtensionConfiguration = Record<
  string,
  Record<string, unknown>
>;

export interface TaskRuntimeComposerSelection {
  runtimeExtensions: TaskRuntimeExtensionConfiguration;
}

interface TaskRuntimeComposerBaseProps {
  disabled: boolean;
  selection: TaskRuntimeComposerSelection | null;
  onSelectionChange: (
    selection: TaskRuntimeComposerSelection | null,
  ) => void;
}

export interface TaskRuntimeComposerMenuExtensionProps
  extends TaskRuntimeComposerBaseProps {
  onRequestClose: () => void;
}

export type TaskRuntimeComposerSelectionExtensionProps =
  TaskRuntimeComposerBaseProps;

export interface TaskRuntimeMessageMetadataExtensionProps {
  bindings: readonly string[];
  publicMetadata: Readonly<Record<string, Record<string, unknown>>>;
}

/**
 * Build-time extension points for distributions that provide additional
 * task-scoped runtimes. The OSS implementation is deliberately inert. A
 * composed distribution may replace this module without overriding the
 * composer, Settings page, or conversation panel themselves.
 *
 * Replacements must preserve every export in this module, accept a null
 * composer selection, and render safely when metadata collections are empty.
 * `hasTaskRuntimeComposerExtension` controls whether the composer menu and
 * selected-state slots are mounted; the Settings and message slots are
 * mounted independently at their respective integration points.
 */
export const hasTaskRuntimeComposerExtension: boolean = false;

export const TaskRuntimeComposerMenuExtension: ComponentType<
  TaskRuntimeComposerMenuExtensionProps
> = () => null;

export const TaskRuntimeComposerSelectionExtension: ComponentType<
  TaskRuntimeComposerSelectionExtensionProps
> = () => null;

export const TaskRuntimeSettingsExtension: ComponentType = () => null;

export const TaskRuntimeMessageMetadataExtension: ComponentType<
  TaskRuntimeMessageMetadataExtensionProps
> = () => null;

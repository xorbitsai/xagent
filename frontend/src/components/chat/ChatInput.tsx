import React, { useState, useRef, useEffect } from "react";
import { createFileChipHTML } from "./FileChip";
import { useRouter } from "next/navigation";
import { Paperclip, X, File as FileIcon, Sparkles, Pause, Play, Loader2, ArrowUp, Globe } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn, getApiUrl, getUploadApiUrl } from "@/lib/utils";
import { useI18n } from "@/contexts/i18n-context";
import { useApp } from "@/contexts/app-context-chat";
import { ConfigDialog } from "@/components/config-dialog";
import { apiRequest, getUploadErrorMessage, isJsonRecord, parseApiResponse, UPLOAD_ERROR_MESSAGES } from "@/lib/api-wrapper";
import { useFileMention, FileItem } from "@/hooks/use-file-mention";
import { FileMentionDropdown } from "./FileMentionDropdown";
import { toast } from "@/components/ui/sonner";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

interface ChatInputProps {
  onSend: (message: string, config?: any) => void | Promise<void>;
  isLoading?: boolean;
  files?: File[];
  onFilesChange?: (files: File[]) => void;
  showModeToggle?: boolean;
  mode?: "task" | "process";
  onModeChange?: (mode: "task" | "process") => void;
  inputValue?: string;
  onInputChange?: (value: string) => void;
  taskStatus?: "pending" | "running" | "completed" | "failed" | "paused" | "waiting_for_user";
  onPause?: () => void;
  onResume?: () => void;
  taskConfig?: {
    model?: string;
    smallFastModel?: string;
    visualModel?: string;
    compactModel?: string;
    executionMode?: "flash" | "balanced" | "think";
  };
  hideConfig?: boolean;
  readOnlyConfig?: boolean;
  hideFileUpload?: boolean;
  compact?: boolean;
  autoFocus?: boolean;
  minHeightClass?: string;
  promptHighlightTerms?: string[];
  selectedAgents?: Array<{
    id: number | string;
    name: string;
  }>;
  onRemoveSelectedAgent?: (agentId: number | string) => void;
}

export function ChatInput({
  onSend,
  isLoading,
  files = [],
  onFilesChange,
  mode,
  inputValue,
  onInputChange,
  taskStatus,
  onPause,
  onResume,
  taskConfig,
  hideConfig = false,
  readOnlyConfig = false,
  hideFileUpload = false,
  compact = false,
  autoFocus = false,
  minHeightClass = "min-h-[130px]",
  promptHighlightTerms = [],
  selectedAgents = [],
  onRemoveSelectedAgent,
}: ChatInputProps) {
  const router = useRouter();
  const [internalMessage, setInternalMessage] = useState("");
  const [isFocused, setIsFocused] = useState(false);
  const [showNoModelAlert, setShowNoModelAlert] = useState(false);
  const [isDraggingFiles, setIsDraggingFiles] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const editorRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const isSubmittingRef = useRef(false);
  const dragDepthRef = useRef(0);
  const { t } = useI18n();
  const { openFilePreview } = useApp();

  useEffect(() => {
    if (!autoFocus || !editorRef.current) return;
    const el = editorRef.current;
    const frameId = window.requestAnimationFrame(() => {
      el.focus();
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(el);
      range.collapse(false);
      selection?.removeAllRanges();
      selection?.addRange(range);
    });
    return () => window.cancelAnimationFrame(frameId);
  }, [autoFocus]);

  const escapeHtml = (value: string) =>
    value
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  const replaceFirstOccurrence = (source: string, search: string, replacement: string) => {
    const index = source.indexOf(search);
    if (index === -1) return source;
    return source.slice(0, index) + replacement + source.slice(index + search.length);
  };

  const applyPromptHighlights = (html: string, highlightTerms: string[]) => {
    return highlightTerms.reduce((currentHtml, term) => {
      const escapedTerm = escapeHtml(term);
      const highlightedTerm = `<span data-prompt-highlight="true" class="font-medium" style="color:#3b5cff;text-decoration-line:underline;text-decoration-style:dashed;text-decoration-color:#3b5cff;text-underline-offset:4px;">${escapedTerm}</span>`;
      return replaceFirstOccurrence(currentHtml, escapedTerm, highlightedTerm);
    }, html);
  };

  const serializeEditorContent = (editor: HTMLElement) => {
    const clone = editor.cloneNode(true) as HTMLElement;
    const chips = clone.querySelectorAll("[data-file-path]");

    chips.forEach((chip) => {
      const path = chip.getAttribute("data-file-path");
      const fileId = chip.getAttribute("data-file-id");
      const filename =
        chip.getAttribute("data-filename") || path?.split("/").pop() || path;
      const id = fileId || path;
      chip.replaceWith(document.createTextNode(`[${filename}](file://${id})`));
    });

    clone.querySelectorAll("br").forEach((lineBreak) => {
      lineBreak.replaceWith(document.createTextNode("\n"));
    });

    clone.querySelectorAll("div, p").forEach((block) => {
      if (block.lastChild?.textContent?.endsWith("\n")) {
        return;
      }
      block.appendChild(document.createTextNode("\n"));
    });

    return (clone.textContent || "")
      .replace(/\u200B/g, "")
      .replace(/\n{3,}/g, "\n\n")
      .replace(/\n$/, "");
  };

  const handleInput = () => {
    const editor = editorRef.current;
    if (!editor) return;

    const text = serializeEditorContent(editor);

    if (isControlled) {
      onInputChange?.(text);
    } else {
      setInternalMessage(text);
    }

    fileMention.checkTrigger();
  };

  const fileMention = useFileMention(editorRef, containerRef, handleInput, t);

  // Track files for async operations
  const filesRef = useRef(files);
  const uploadAbortControllersRef = useRef<Map<string, AbortController>>(new Map());

  useEffect(() => {
    filesRef.current = files;
  }, [files]);

  // Determine if controlled or uncontrolled
  const isControlled = inputValue !== undefined;
  const message = isControlled ? inputValue : internalMessage;


  // Handle click on delete button and file chip preview
  useEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;

    const handleClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const deleteBtn = target.closest('.file-chip-delete');
      if (deleteBtn) {
        e.preventDefault();
        e.stopPropagation();
        const chip = deleteBtn.closest('[data-file-path]');
        if (chip) {
          chip.remove();
          // Trigger input event manually to update state
          const event = new Event('input', { bubbles: true });
          editor.dispatchEvent(event);
        }
        return;
      }

      // Handle file chip preview click
      const chip = target.closest('.file-chip-preview');
      if (chip) {
        e.preventDefault();
        e.stopPropagation();
        const filePath = chip.getAttribute('data-file-path');
        if (filePath) {
          // If we have fileId mapped in our list, use it. Otherwise use the path as fileId fallback.
          const fileInfo = fileMention.fileList.find((f: FileItem) => f.relative_path === filePath || f.filename === filePath);
          const fileName = fileInfo?.filename || filePath.split('/').pop() || filePath;

          openFilePreview(
            fileInfo?.file_id || filePath, // use file_id as fileId if available, fallback to path
            fileName,
            [{ fileName, fileId: fileInfo?.file_id || filePath }]
          );
        }
      }
    };

    editor.addEventListener('click', handleClick);
    return () => editor.removeEventListener('click', handleClick);
  }, [fileMention.fileList, openFilePreview]);
  const [agentConfig, setAgentConfig] = useState<{
    model: string;
    smallFastModel?: string;
    visualModel?: string;
    compactModel?: string;
    memorySimilarityThreshold?: number;
    executionMode?: "flash" | "balanced" | "think";
  }>({ model: "", memorySimilarityThreshold: 1.5 });
  const [defaultAgentConfig, setDefaultAgentConfig] = useState<{
    model: string;
    smallFastModel?: string;
    visualModel?: string;
    compactModel?: string;
  }>({ model: "" });
  const [models, setModels] = useState<any[]>([]);

  // State to track files currently being uploaded
  const [uploadingFiles, setUploadingFiles] = useState<Set<string>>(new Set());

  const extractDroppedFiles = (dataTransfer: DataTransfer) => {
    const itemFiles = Array.from(dataTransfer.items || [])
      .filter((item) => item.kind === "file")
      .map((item) => item.getAsFile())
      .filter((file): file is File => file instanceof File);

    return itemFiles.length > 0 ? itemFiles : Array.from(dataTransfer.files || []);
  };

  const isFileDragEvent = (e: React.DragEvent) =>
    Array.from(e.dataTransfer?.types || []).includes("Files");

  const resetDragState = () => {
    dragDepthRef.current = 0;
    setIsDraggingFiles(false);
  };

  // Helper to upload files immediately
  const uploadFiles = async (newFiles: File[]) => {
    if (newFiles.length === 0) return;

    // Mark as uploading (use name + lastModified as rough unique ID)
    const fileIds = newFiles.map(f => `${f.name}-${f.lastModified}`);
    setUploadingFiles(prev => {
      const next = new Set(prev);
      fileIds.forEach(id => next.add(id));
      return next;
    });

    const failedFiles = new Set<File>();
    let uploadErrorMessage: string | null = null;

    // Upload files individually to ensure better reliability and progress tracking
    await Promise.all(newFiles.map(async (file) => {
      const fileId = `${file.name}-${file.lastModified}`;
      const controller = new AbortController();
      uploadAbortControllersRef.current.set(fileId, controller);

      try {
        const formData = new FormData();
        formData.append('file', file);
        // Default to task mode if not specified
        formData.append('task_type', mode || 'task');

        const response = await apiRequest(`${getUploadApiUrl()}/api/files/upload`, {
          method: 'POST',
          body: formData,
          signal: controller.signal
        });

        const parsed = await parseApiResponse(response);

        if (response.ok && isJsonRecord(parsed.data)) {
          const data = parsed.data;
          if (data.success && typeof data.file_id === 'string') {
            // Attach file_id to the File object
            (file as File & { file_id?: string }).file_id = data.file_id;
          } else {
            failedFiles.add(file);
          }
        } else {
          failedFiles.add(file);
          uploadErrorMessage = uploadErrorMessage || getUploadErrorMessage(response, parsed, {
            generic: t("files.uploadFailed") || "Failed to upload some files",
            ...UPLOAD_ERROR_MESSAGES,
          });
        }
      } catch (error: any) {
        if (error.name === 'AbortError') {
          // Upload cancelled, do nothing
        } else {
          console.error("Error uploading file:", error);
          failedFiles.add(file);
          uploadErrorMessage = uploadErrorMessage || (error instanceof Error ? error.message : null);
        }
      } finally {
        uploadAbortControllersRef.current.delete(fileId);
        setUploadingFiles(prev => {
          const next = new Set(prev);
          next.delete(fileId);
          return next;
        });
      }
    }));

    // Handle failed files
    if (failedFiles.size > 0) {
      toast.error(uploadErrorMessage || t("files.uploadFailed") || "Failed to upload some files");
      if (onFilesChange) {
        onFilesChange(filesRef.current.filter(f => !failedFiles.has(f)));
      }
    }
  };

  const appendFiles = (newFiles: File[]) => {
    if (newFiles.length === 0 || !onFilesChange || isInputBusy) return;
    onFilesChange([...filesRef.current, ...newFiles]);
    uploadFiles(newFiles);
  };

  // Fetch default models on mount
  useEffect(() => {
    const fetchDefaultModels = async () => {
      try {
        const apiUrl = getApiUrl();

        // Fetch all models first to have the list for display names
        const modelsResponse = await apiRequest(`${apiUrl}/api/models/?category=llm`, {
          headers: {}
        });

        let allModels: any[] = [];
        if (modelsResponse.ok) {
          allModels = await modelsResponse.json();
          if (Array.isArray(allModels)) {
            setModels(allModels);
          }
        }

        // Fetch user default models
        const defaultResponse = await apiRequest(`${apiUrl}/api/models/user-default`, {
          headers: {}
        });

        let defaultModels: Record<string, any> = {};
        if (defaultResponse.ok) {
          const defaults = await defaultResponse.json();
          if (Array.isArray(defaults)) {
            defaults.forEach((defaultConfig: any) => {
              if (defaultConfig && defaultConfig.config_type && defaultConfig.model) {
                defaultModels[defaultConfig.config_type] = defaultConfig.model;
              }
            });
          }
        }

        // Find default if no user preference
        if (!defaultModels.general && allModels.length > 0) {
          const defaultModel = allModels.find((m: any) => m.is_default) || allModels[0];
          if (defaultModel) {
            defaultModels.general = { model_id: defaultModel.model_id };
          }
        }

        const newDefaultConfig = {
          model: defaultModels.general?.model_id || "",
          smallFastModel: defaultModels.small_fast?.model_id,
          visualModel: defaultModels.visual?.model_id,
          compactModel: defaultModels.compact?.model_id
        };

        setDefaultAgentConfig(newDefaultConfig);

        setAgentConfig(prev => ({
          ...prev,
          model: prev.model || newDefaultConfig.model,
          smallFastModel: prev.smallFastModel || newDefaultConfig.smallFastModel,
          visualModel: prev.visualModel || newDefaultConfig.visualModel,
          compactModel: prev.compactModel || newDefaultConfig.compactModel
        }));
      } catch (error) {
        console.error('Failed to fetch default models:', error);
      }
    };

    fetchDefaultModels();
  }, []);

  // Update config when taskConfig changes
  useEffect(() => {
    if (taskConfig) {
      setAgentConfig(prev => ({
        ...prev,
        model: taskConfig.model || prev.model,
        smallFastModel: taskConfig.smallFastModel || prev.smallFastModel,
        visualModel: taskConfig.visualModel || prev.visualModel,
        compactModel: taskConfig.compactModel || prev.compactModel,
        executionMode: taskConfig.executionMode
      }));
    } else if (!readOnlyConfig) {
      setAgentConfig(prev => ({
        ...prev,
        model: defaultAgentConfig.model,
        smallFastModel: defaultAgentConfig.smallFastModel,
        visualModel: defaultAgentConfig.visualModel,
        compactModel: defaultAgentConfig.compactModel,
        executionMode: undefined
      }));
    }
  }, [taskConfig, readOnlyConfig, defaultAgentConfig]);

  const handleConfigChange = (config: {
    model: string;
    smallFastModel?: string;
    visualModel?: string;
    compactModel?: string;
    memorySimilarityThreshold?: number;
    executionMode?: "flash" | "balanced" | "think";
  }) => {
    setAgentConfig(config);
  };

  const allowsInterruptedInput = taskStatus === 'paused' || taskStatus === 'waiting_for_user';
  const isInputBusy = !!isLoading && !allowsInterruptedInput;

  const canSubmit = () => {
    const hasText = message.trim().length > 0;
    const hasFiles = files.length > 0;
    const isUploadingFiles = uploadingFiles.size > 0;
    return (hasText || hasFiles) && !isInputBusy && !isUploadingFiles;
  };
  const canPauseTask =
    taskStatus === 'running' ||
    (isLoading &&
      !!onPause &&
      !['completed', 'failed', 'paused', 'waiting_for_user'].includes(taskStatus || ''));

  const handleDragEnter = (e: React.DragEvent<HTMLFormElement>) => {
    if (!isFileDragEvent(e) || isInputBusy || hideFileUpload) return;
    e.preventDefault();
    e.stopPropagation();
    dragDepthRef.current += 1;
    setIsDraggingFiles(true);
  };

  const handleDragOver = (e: React.DragEvent<HTMLFormElement>) => {
    if (!isFileDragEvent(e) || isInputBusy || hideFileUpload) return;
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = "copy";
    if (!isDraggingFiles) {
      setIsDraggingFiles(true);
    }
  };

  const handleDragLeave = (e: React.DragEvent<HTMLFormElement>) => {
    if (!isFileDragEvent(e)) return;
    e.preventDefault();
    e.stopPropagation();
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (dragDepthRef.current === 0) {
      setIsDraggingFiles(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLFormElement>) => {
    if (!isFileDragEvent(e) || isInputBusy || hideFileUpload) return;
    e.preventDefault();
    e.stopPropagation();
    const droppedFiles = extractDroppedFiles(e.dataTransfer);
    resetDragState();
    appendFiles(droppedFiles);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!canSubmit() || isSubmittingRef.current) return;

    const hasSelectedAgent = selectedAgents.length > 0;
    if (!hasSelectedAgent && !agentConfig.model) {
      setShowNoModelAlert(true);
      return;
    }

    try {
      isSubmittingRef.current = true;
      const trimmed = message.trim();
      const messageToSend = trimmed;

      const configToSend = {
        ...agentConfig,
        ...(taskConfig?.executionMode ? { executionMode: { mode: taskConfig.executionMode } } : {}),
      };

      await onSend(messageToSend, configToSend);

      if (isControlled) {
        onInputChange?.("");
      } else {
        setInternalMessage("");
      }
    } finally {
      // Small delay to prevent double submission
      setTimeout(() => {
        isSubmittingRef.current = false;
      }, 500);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (fileMention.handleKeyDown(e)) {
      return;
    }

    if (e.key === "Enter" && !e.shiftKey) {
      // Prevent triggering submit when using IME (e.g., Chinese input method)
      if (e.nativeEvent.isComposing) {
        return;
      }
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  const handlePaste = (e: React.ClipboardEvent<HTMLDivElement>) => {
    const items = Array.from(e.clipboardData.items || []);
    const fileItems = items.filter(item => item.kind === 'file');

    if (fileItems.length > 0 && !hideFileUpload) {
      e.preventDefault();
      const pastedFiles: File[] = [];

      fileItems.forEach((item, index) => {
        const file = item.getAsFile();
        if (file) {
          const hasName = typeof (file as any).name === 'string' && (file as any).name.length > 0;
          // Handle default "image.png" name which causes conflicts when pasting multiple images
          if (hasName && file.name !== 'image.png') {
            pastedFiles.push(file);
          } else {
            const timestamp = Date.now();
            const mime = item.type || file.type || 'application/octet-stream';
            const ext = mime.split('/')[1] || 'bin';
            // If it was image.png, preserve extension but make unique. Otherwise default to pasted-file
            const baseName = file.name === 'image.png'
              ? `image-${timestamp}-${index}`
              : `pasted-file-${timestamp}-${index}`;

            const namedFile = new File([file], `${baseName}.${ext}`, {
              type: mime,
              lastModified: timestamp,
            });
            pastedFiles.push(namedFile);
          }
        }
      });

      appendFiles(pastedFiles);
    } else {
      // Strip formatting from text paste
      e.preventDefault();
      const text = e.clipboardData.getData("text/plain");
      document.execCommand("insertText", false, text);
      // Trigger input handling manually as execCommand might not bubble up to React's onInput reliably in all browsers
      handleInput();
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    appendFiles(selectedFiles);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const removeFile = (index: number) => {
    const fileToRemove = files[index];
    if (fileToRemove) {
      const fileId = `${fileToRemove.name}-${fileToRemove.lastModified}`;
      const controller = uploadAbortControllersRef.current.get(fileId);
      if (controller) {
        controller.abort();
        uploadAbortControllersRef.current.delete(fileId);
      }
    }
    onFilesChange?.(files.filter((_, i) => i !== index));
  };

  useEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;

    if (!message) {
      if (editor.innerHTML !== "") {
        editor.innerHTML = "";
      }
    } else if (document.activeElement !== editor) {
      const currentText = serializeEditorContent(editor);

      if (message !== currentText) {
        let html = escapeHtml(message);

        // Restore file:// links
        html = html.replace(/\[([^\]]+)\]\(file:\/\/([^)]+)\)/g, (_match, filename, id) => {
          // We use the ID as the path since we don't have the real path anymore
          return createFileChipHTML(id, id, filename);
        });
        // Fallback for old backticked messages to not break existing chat history
        html = html.replace(/`([^`]+)`/g, (_match, path) => {
          return createFileChipHTML(path);
        });
        html = applyPromptHighlights(html, promptHighlightTerms);
        html = html.replace(/\n/g, "<br>");

        editor.innerHTML = html;
      }
    }
  }, [message, promptHighlightTerms]);

  return (
    <div className="space-y-3">
      {/* Input area */}
      <div
        className={cn("relative", selectedAgents.length > 0 && "pt-9")}
        ref={containerRef}
      >
        <FileMentionDropdown
          show={fileMention.showFilePicker}
          isLoading={fileMention.isLoadingFiles}
          filteredFiles={fileMention.filteredFiles}
          selectedFileIndex={fileMention.selectedFileIndex}
          onInsert={fileMention.insertFile}
          t={t}
          position={fileMention.dropdownPosition}
        />
        {selectedAgents.length > 0 && (
          <div className="absolute top-0 z-10 flex flex-wrap gap-2">
            {selectedAgents.map((agent) => (
              <div
                key={agent.id}
                className="inline-flex h-9 items-center gap-1 rounded-t-xl rounded-b-none border border-b-0 px-3 text-xs font-medium shadow-[0_-1px_0_rgba(53,88,255,0.08)]"
                style={{ borderColor: "#3040cf", color: "#3040cf", backgroundColor: "#eef1ff" }}
              >
                <span className="italic">Using</span>
                <span
                  className="rounded-md border px-2 py-0.5 not-italic"
                  style={{ borderColor: "#3040cf", color: "#3040cf", backgroundColor: "#eef1ff" }}
                >{`@${agent.name}`}</span>
                {onRemoveSelectedAgent && (
                  <button
                    type="button"
                    onClick={() => onRemoveSelectedAgent(agent.id)}
                    className="rounded-sm p-0.5 hover:bg-[#dfe6ff]"
                    title={t("common.remove")}
                  >
                    <X className="h-3 w-3" />
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
        <form
          onSubmit={handleSubmit}
          className={cn(
            "relative flex flex-col overflow-hidden border-2 bg-card shadow-sm",
            selectedAgents.length > 0
              ? "rounded-tr-2xl rounded-br-2xl rounded-bl-2xl rounded-tl-none"
              : "rounded-2xl",
            isFocused
              ? "shadow-[0_0_0_3px_rgba(48,64,207,0.16)]"
              : ""
          )}
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          style={{
            borderColor: selectedAgents.length > 0 ? "#3040cf" : isDraggingFiles || isFocused ? "#3040cf" : "#d7deec"
          }}
        >
          {isDraggingFiles && (
            <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center bg-primary/5 px-4">
              <div className="flex max-w-sm items-center gap-3 rounded-2xl border border-primary/20 bg-background/95 px-4 py-3 text-left shadow-lg backdrop-blur-sm">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10 text-primary">
                  <Paperclip className="h-4 w-4" />
                </div>
                <div className="space-y-1">
                  <div className="text-sm font-medium text-foreground">{t("chatPage.fileUpload.dropHere")}</div>
                </div>
              </div>
            </div>
          )}
          {files.length > 0 && (
            <div className="flex flex-wrap gap-2 px-4 pt-3">
              {files.map((file, index) => {
                const isUploading = uploadingFiles.has(`${file.name}-${file.lastModified}`);
                return (
                  <div
                    key={index}
                    className={cn(
                      "inline-flex h-8 items-center gap-2 rounded-md border border-slate-200 bg-slate-100 px-3 text-sm text-slate-700 animate-fade-in-scale transition-colors",
                      isUploading && "opacity-70"
                    )}
                  >
                    <div className="flex h-4 w-4 items-center justify-center text-slate-600">
                      {isUploading ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <FileIcon className="h-3.5 w-3.5" />
                      )}
                    </div>
                    <span className="max-w-[180px] truncate font-medium">{file.name}</span>
                    <button
                      type="button"
                      onClick={() => removeFile(index)}
                      className="ml-0.5 rounded-sm p-0.5 text-slate-400 transition-colors hover:bg-slate-200 hover:text-slate-700"
                      title={isUploading ? t("common.cancel") : t("common.remove")}
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                );
              })}
            </div>
          )}

          <div className="relative flex-1">
            <div
              ref={editorRef}
              contentEditable
              className={cn(
                "w-full rounded-md border-0 bg-transparent text-[15px] outline-none placeholder:text-muted-foreground/60 overflow-y-auto resize-none focus-visible:ring-0 focus-visible:ring-offset-0 whitespace-pre-wrap break-words text-left",
                compact ? "min-h-[44px] px-3 py-3 pr-12 max-h-[150px]" : cn(minHeightClass, "px-4 py-3 pb-16 max-h-[400px]"),
                isInputBusy ? "opacity-50 pointer-events-none" : ""
              )}
              onInput={handleInput}
              onKeyDown={handleKeyDown}
              onPaste={handlePaste as any}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              role="textbox"
              aria-multiline="true"
            />
            {!message && (
              <div className="pointer-events-none absolute left-4 top-3 text-[14px] text-muted-foreground/60">
                {t("chatPage.input.placeholder")}
              </div>
            )}
          </div>

          {/* Bottom toolbar or inline button */}
          {compact ? (
            <div className="absolute right-2 bottom-2">
              <Button
                type="submit"
                size="icon"
                disabled={!canSubmit()}
                className={cn(
                  "h-8 w-8 rounded-lg transition-all duration-300",
                  !canSubmit() && "bg-muted text-muted-foreground/50"
                )}
              >
                {isInputBusy ? (
                  <Sparkles className="h-4 w-4 animate-pulse" />
                ) : (
                  <ArrowUp className="h-4 w-4" />
                )}
              </Button>
            </div>
          ) : (
            <div className="absolute bottom-0 left-0 right-0 flex items-center justify-between bg-card px-4 py-3">
              <div className="flex items-center gap-2">
                {/* Settings button - left of upload */}
                {!hideConfig && (
                  <>
                    {readOnlyConfig ? (
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="h-9 px-3 text-muted-foreground rounded-xl gap-2 cursor-default hover:bg-transparent"
                        disabled={true}
                        title={models.find(m => String(m.id) === String(agentConfig.model) || String(m.model_id) === String(agentConfig.model))?.model_name || agentConfig.model || t("chatPage.input.noModel")}
                      >
                        <Globe className="h-4 w-4" />
                        <span className="text-xs font-normal max-w-[150px] truncate hidden sm:inline-block">
                          {models.find(m => String(m.id) === String(agentConfig.model) || String(m.model_id) === String(agentConfig.model))?.model_name || agentConfig.model || t("chatPage.input.noModel")}
                        </span>
                      </Button>
                    ) : (
                      <ConfigDialog
                        onConfigChange={handleConfigChange}
                        currentConfig={agentConfig}
                        trigger={
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="h-9 px-3 text-muted-foreground hover:text-foreground hover:bg-secondary/80 rounded-xl gap-2"
                            disabled={isInputBusy}
                            title={t('agent.input.actions.config')}
                          >
                            <Globe className="h-4 w-4" />
                            <span className="text-xs font-normal max-w-[150px] truncate hidden sm:inline-block">
                              {models.find(m => String(m.id) === String(agentConfig.model) || String(m.model_id) === String(agentConfig.model))?.model_name || agentConfig.model || t("chatPage.input.noModel")}
                            </span>
                          </Button>
                        }
                      />
                    )}
                  </>
                )}
                {/* Upload button - adjacent to bottom toolbar */}
                {!hideFileUpload && (
                  <>
                    <input
                      ref={fileInputRef}
                      type="file"
                      multiple
                      onChange={handleFileSelect}
                      className="hidden"
                      accept=".pdf,.doc,.docx,.txt,.md,.csv,.json,.xlsx,.xls,.ppt,.pptx,.png,.jpg,.jpeg,.gif,.webp"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="h-9 w-9 p-0 text-muted-foreground hover:text-foreground hover:bg-secondary/80 rounded-full"
                      onClick={() => fileInputRef.current?.click()}
                      disabled={isInputBusy}
                      title={t("chatPage.input.actions.upload")}
                    >
                      <Paperclip className="h-4 w-4" />
                    </Button>
                  </>
                )}
              </div>

              <div className="flex items-center gap-3">
                {canPauseTask ? (
                  <Button
                    type="button"
                    size="icon"
                    onClick={onPause}
                    className="h-8 w-8 rounded-full transition-all duration-300 bg-yellow-500 hover:bg-yellow-600 text-white"
                  >
                    <Pause className="h-4 w-4" />
                  </Button>
                ) : (
                  <div className="flex items-center gap-2">
                    <span className="text-[13px] font-medium text-muted-foreground/50 select-none mr-1">
                      ⏎ {t("common.send")}
                    </span>
                    <Button
                      type="submit"
                      size="icon"
                      disabled={!canSubmit()}
                      className={cn(
                        "h-8 w-8 rounded-lg transition-all duration-300",
                        !canSubmit() && "bg-muted text-muted-foreground/50"
                      )}
                    >
                      {isInputBusy ? (
                        <Sparkles className="h-4 w-4 animate-pulse" />
                      ) : (
                        <ArrowUp className="h-4 w-4" />
                      )}
                    </Button>
                    {taskStatus === 'paused' && onResume && (
                      <Button
                        type="button"
                        size="icon"
                        onClick={onResume}
                        className="h-8 w-8 rounded-full transition-all duration-300 bg-green-500 hover:bg-green-600 text-white"
                        title={t('agent.input.actions.resumeTask')}
                      >
                        <Play className="h-4 w-4" />
                      </Button>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </form>
      </div>

      <AlertDialog open={showNoModelAlert} onOpenChange={setShowNoModelAlert}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("common.notice")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("chatPage.input.noModelAlert")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction onClick={() => router.push("/models")}>
              {t("common.confirm")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

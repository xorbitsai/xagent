import { useState, useRef, useEffect } from "react";
import { createFileChipHTML } from "./FileChip";
import { useRouter } from "next/navigation";
import { Paperclip, X, File as FileIcon, Sparkles, Pause, Play, Loader2, ArrowUp, Globe } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn, getApiUrl } from "@/lib/utils";
import { useI18n } from "@/contexts/i18n-context";
import { useApp } from "@/contexts/app-context-chat";
import { ConfigDialog } from "@/components/config-dialog";
import { apiRequest } from "@/lib/api-wrapper";
import { useFileMention, FileItem } from "@/hooks/use-file-mention";
import { FileMentionDropdown } from "./FileMentionDropdown";
import { toast } from "sonner";
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
  taskStatus?: "pending" | "running" | "completed" | "failed" | "paused";
  onPause?: () => void;
  onResume?: () => void;
  taskConfig?: {
    model?: string;
    smallFastModel?: string;
    visualModel?: string;
    compactModel?: string;
  };
  hideConfig?: boolean;
  readOnlyConfig?: boolean;
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
  readOnlyConfig = false
}: ChatInputProps) {
  const router = useRouter();
  const [internalMessage, setInternalMessage] = useState("");
  const [isFocused, setIsFocused] = useState(false);
  const [showNoModelAlert, setShowNoModelAlert] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const editorRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const isSubmittingRef = useRef(false);
  const { t } = useI18n();
  const { openFilePreview } = useApp();

  const handleInput = () => {
    const editor = editorRef.current;
    if (!editor) return;

    // Serialize content: replace chips with markdown link containing file:// scheme
    const clone = editor.cloneNode(true) as HTMLElement;
    const chips = clone.querySelectorAll('[data-file-path]');
    chips.forEach((chip) => {
      const path = chip.getAttribute('data-file-path');
      const fileId = chip.getAttribute('data-file-id');
      const filename = chip.getAttribute('data-filename') || path?.split('/').pop() || path;

      // Use fileId if available, otherwise path (fallback)
      const id = fileId || path;
      chip.replaceWith(document.createTextNode(`[${filename}](file://${id})`));
    });

    // Use innerText to preserve newlines
    let text = clone.innerText;
    // Remove zero-width spaces if any (sometimes added by contentEditable)
    text = text.replace(/\u200B/g, '');

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
  }>({ model: "", memorySimilarityThreshold: 1.5 });
  const [models, setModels] = useState<any[]>([]);

  // State to track files currently being uploaded
  const [uploadingFiles, setUploadingFiles] = useState<Set<string>>(new Set());

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

        const response = await apiRequest(`${getApiUrl()}/api/files/upload`, {
          method: 'POST',
          body: formData,
          signal: controller.signal
        });

        if (response.ok) {
          const data = await response.json();
          if (data.success && data.file_id) {
            // Attach file_id to the File object
            (file as any).file_id = data.file_id;
          } else {
            failedFiles.add(file);
          }
        } else {
          failedFiles.add(file);
        }
      } catch (error: any) {
        if (error.name === 'AbortError') {
          // Upload cancelled, do nothing
        } else {
          console.error("Error uploading file:", error);
          failedFiles.add(file);
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
      toast.error(t("files.uploadFailed") || "Failed to upload some files");
      if (onFilesChange) {
        onFilesChange(filesRef.current.filter(f => !failedFiles.has(f)));
      }
    }
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

        setAgentConfig(prev => ({
          ...prev,
          model: prev.model || defaultModels.general?.model_id || "",
          smallFastModel: prev.smallFastModel || defaultModels.small_fast?.model_id,
          visualModel: prev.visualModel || defaultModels.visual?.model_id,
          compactModel: prev.compactModel || defaultModels.compact?.model_id
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
        compactModel: taskConfig.compactModel || prev.compactModel
      }));
    }
  }, [taskConfig]);

  const handleConfigChange = (config: {
    model: string;
    smallFastModel?: string;
    visualModel?: string;
    compactModel?: string;
    memorySimilarityThreshold?: number;
  }) => {
    setAgentConfig(config);
  };

  const canSubmit = () => {
    const hasText = message.trim().length > 0;
    const hasFiles = files.length > 0;
    const isUploadingFiles = uploadingFiles.size > 0;
    return (hasText || hasFiles) && !isLoading && !isUploadingFiles;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!canSubmit() || isSubmittingRef.current) return;

    if (!agentConfig.model) {
      setShowNoModelAlert(true);
      return;
    }

    try {
      isSubmittingRef.current = true;
      const trimmed = message.trim();
      const messageToSend = trimmed;

      // Always send task mode config
      const configToSend = { ...agentConfig, vibeMode: { mode: "task" } };

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

    if (fileItems.length > 0) {
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

      if (pastedFiles.length > 0) {
        onFilesChange?.([...files, ...pastedFiles]);
        uploadFiles(pastedFiles);
      }
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
    onFilesChange?.([...files, ...selectedFiles]);
    uploadFiles(selectedFiles);
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
    } else if (document.activeElement !== editor && editor.innerText.trim() === "") {
      // Restore file:// links
      let html = message.replace(/\[([^\]]+)\]\(file:\/\/([^)]+)\)/g, (match, filename, id) => {
        // We use the ID as the path since we don't have the real path anymore
        return createFileChipHTML(id, id, filename);
      });
      // Fallback for old backticked messages to not break existing chat history
      html = html.replace(/`([^`]+)`/g, (match, path) => {
        return createFileChipHTML(path);
      });
      editor.innerHTML = html;
    }
  }, [message]);

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  return (
    <div className="space-y-3">
      {/* File list */}
      {files.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {files.map((file, index) => {
            const isUploading = uploadingFiles.has(`${file.name}-${file.lastModified}`);
            return (
              <div
                key={index}
                className={cn(
                  "flex items-center gap-2 px-3 py-2 bg-secondary/80 rounded-xl text-sm animate-fade-in-scale border border-border/50 hover:border-primary/30 transition-colors",
                  isUploading && "opacity-70"
                )}
              >
                <div className="w-6 h-6 rounded-lg bg-primary/10 flex items-center justify-center">
                  {isUploading ? (
                    <Loader2 className="w-3.5 h-3.5 text-primary animate-spin" />
                  ) : (
                    <FileIcon className="w-3.5 h-3.5 text-primary" />
                  )}
                </div>
                <span className="truncate max-w-[120px] font-medium text-foreground/90">{file.name}</span>
                <span className="text-xs text-muted-foreground">
                  {formatFileSize(file.size)}
                </span>
                <button
                  type="button"
                  onClick={() => removeFile(index)}
                  className={cn(
                    "ml-1 p-0.5 rounded-md transition-colors",
                    isUploading
                      ? "text-muted-foreground/50 hover:text-foreground hover:bg-secondary"
                      : "text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                  )}
                  title={isUploading ? t("common.cancel") : t("common.remove")}
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )
          })}
        </div>
      )}

      {/* Input area */}
      <div className="relative" ref={containerRef}>
        <FileMentionDropdown
          show={fileMention.showFilePicker}
          isLoading={fileMention.isLoadingFiles}
          filteredFiles={fileMention.filteredFiles}
          selectedFileIndex={fileMention.selectedFileIndex}
          onInsert={fileMention.insertFile}
          t={t}
          position={fileMention.dropdownPosition}
        />
        <form
          onSubmit={handleSubmit}
          className={cn(
            "relative rounded-2xl bg-card overflow-hidden transition-all duration-300 border",
            isFocused
              ? "border-primary/50 shadow-md"
              : "border-border shadow-sm hover:border-border/80"
          )}
        >
          <div
            ref={editorRef}
            contentEditable
            className={cn(
              "min-h-[130px] max-h-[300px] w-full rounded-md border-0 bg-transparent px-3 py-2 text-[15px] outline-none placeholder:text-muted-foreground/60 overflow-y-auto resize-none focus-visible:ring-0 focus-visible:ring-offset-0 pb-14 whitespace-pre-wrap break-words text-left",
              isLoading ? "opacity-50 pointer-events-none" : ""
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
            <div className="absolute top-2 left-3 text-muted-foreground/60 pointer-events-none text-[14px]">
              {t("chatPage.input.placeholder")}
            </div>
          )}

          {/* Bottom toolbar */}
          <div className="absolute bottom-0 left-0 right-0 flex items-center justify-between px-4 py-3 bg-card">
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
                      title={models.find(m => m.model_id === agentConfig.model)?.model_name || agentConfig.model || t("chatPage.input.noModel")}
                    >
                      <Globe className="h-4 w-4" />
                      <span className="text-xs font-normal max-w-[150px] truncate hidden sm:inline-block">
                        {models.find(m => m.model_id === agentConfig.model)?.model_name || agentConfig.model || t("chatPage.input.noModel")}
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
                          disabled={isLoading}
                          title={t('agent.input.actions.config')}
                        >
                          <Globe className="h-4 w-4" />
                          <span className="text-xs font-normal max-w-[150px] truncate hidden sm:inline-block">
                            {models.find(m => m.model_id === agentConfig.model)?.model_name || agentConfig.model || t("chatPage.input.noModel")}
                          </span>
                        </Button>
                      }
                    />
                  )}
                </>
              )}
              {/* Upload button - adjacent to bottom toolbar */}
              <input
                ref={fileInputRef}
                type="file"
                multiple
                onChange={handleFileSelect}
                className="hidden"
                accept=".pdf,.doc,.docx,.txt,.md,.csv,.json,.xlsx,.xls,.png,.jpg,.jpeg,.gif,.webp"
              />
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-9 w-9 p-0 text-muted-foreground hover:text-foreground hover:bg-secondary/80 rounded-full"
                onClick={() => fileInputRef.current?.click()}
                disabled={isLoading}
                title={t("chatPage.input.actions.upload")}
              >
                <Paperclip className="h-4 w-4" />
              </Button>
            </div>

            <div className="flex items-center gap-3">
              {taskStatus === 'running' ? (
                <Button
                  type="button"
                  size="icon"
                  onClick={onPause}
                  className="h-8 w-8 rounded-full transition-all duration-300 bg-yellow-500 hover:bg-yellow-600 text-white"
                >
                  <Pause className="h-4 w-4" />
                </Button>
              ) : taskStatus === 'paused' ? (
                <Button
                  type="button"
                  size="icon"
                  onClick={onResume}
                  className="h-8 w-8 rounded-full transition-all duration-300 bg-green-500 hover:bg-green-600 text-white"
                >
                  <Play className="h-4 w-4" />
                </Button>
              ) : (
                <Button
                  type="submit"
                  size="icon"
                  disabled={!canSubmit()}
                  className={cn(
                    "h-8 w-8 rounded-lg transition-all duration-300",
                    !canSubmit() && "bg-muted text-muted-foreground/50"
                  )}
                >
                  {isLoading ? (
                    <Sparkles className="h-4 w-4 animate-pulse" />
                  ) : (
                    <ArrowUp className="h-4 w-4" />
                  )}
                </Button>
              )}
            </div>
          </div>
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

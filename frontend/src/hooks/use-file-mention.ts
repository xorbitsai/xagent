import { useState, useEffect, useRef } from "react";
import { apiRequest } from "@/lib/api-wrapper";
import { getApiUrl } from "@/lib/utils";
import { toast } from "@/components/ui/sonner";
import { createFileChipHTML } from "@/components/chat/FileChip";

export interface FileItem {
  file_id: string;
  filename: string;
  file_size: number;
  modified_time: number;
  file_type?: string;
  relative_path?: string;
  task_id?: number;
  user_id?: number;
}

export function useFileMention(
  editorRef: React.RefObject<HTMLElement | null>,
  containerRef: React.RefObject<HTMLElement | null>,
  onInput: () => void,
  t: (key: string) => string
) {
  const [showFilePicker, setShowFilePicker] = useState(false);
  const [fileList, setFileList] = useState<FileItem[]>([]);
  const [filteredFiles, setFilteredFiles] = useState<FileItem[]>([]);
  const [selectedFileIndex, setSelectedFileIndex] = useState(0);
  const [currentQuery, setCurrentQuery] = useState("");
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [dropdownPosition, setDropdownPosition] = useState<{ top?: number; bottom?: number; left: number } | null>(null);
  const latestFetchRequestRef = useRef(0);

  const closePicker = () => {
    setShowFilePicker(false);
    setFilteredFiles([]);
    setSelectedFileIndex(0);
    setDropdownPosition(null);
  };

  const resetMention = () => {
    closePicker();
    setCurrentQuery("");
  };

  const fetchFiles = async (query: string) => {
    const requestId = latestFetchRequestRef.current + 1;
    latestFetchRequestRef.current = requestId;
    const params = new URLSearchParams({
      page: "1",
      size: "20",
    });
    const normalizedQuery = query.trim();
    if (normalizedQuery) {
      params.set("search", normalizedQuery);
    }

    setIsLoadingFiles(true);
    try {
      const response = await apiRequest(`${getApiUrl()}/api/files/list?${params.toString()}`);
      if (response.ok) {
        const data = await response.json();
        if (requestId !== latestFetchRequestRef.current) {
          return;
        }
        if (data && data.files) {
          setFileList(data.files);
          setFilteredFiles(data.files);
          setSelectedFileIndex(0);
          if (normalizedQuery.length > 0 && data.files.length === 0) {
            closePicker();
          }
        }
      }
    } catch (error) {
      if (requestId === latestFetchRequestRef.current) {
        console.error("Failed to load files", error);
        toast.error(t("files.previewDialog.errors.loadFailed"));
      }
    } finally {
      if (requestId === latestFetchRequestRef.current) {
        setIsLoadingFiles(false);
      }
    }
  };

  const checkTrigger = () => {
    const selection = window.getSelection();
    if (!selection || !selection.rangeCount) {
      resetMention();
      return;
    }

    const range = selection.getRangeAt(0);
    const node = range.startContainer;

    if (node.nodeType === Node.TEXT_NODE && node.textContent) {
      const text = node.textContent;
      const cursor = range.startOffset;
      const textBefore = text.slice(0, cursor);
      const lastAt = textBefore.lastIndexOf('@');

      if (lastAt !== -1) {
        const query = textBefore.slice(lastAt + 1);
        if (!query.includes(' ') && !query.includes('\n')) {
          setCurrentQuery(query);
          setShowFilePicker(true);

          // Calculate position based on the '@' symbol, not the end of the query
          const atRange = document.createRange();
          atRange.setStart(node, lastAt);
          atRange.setEnd(node, lastAt + 1);
          const rect = atRange.getBoundingClientRect();

          const editor = editorRef.current;
          const container = containerRef.current || editor?.closest('.relative') as HTMLElement || editor;
          if (editor && container) {
            const containerRect = container.getBoundingClientRect();

            let pos: { top?: number; bottom?: number; left: number } = {
              left: Math.max(0, rect.left - containerRect.left + container.scrollLeft)
            };

            // Default to positioning above the cursor
            if (rect.top < 250) {
              // If there's not enough space above (assuming ~250px dropdown height), position it below
              pos.top = rect.bottom - containerRect.top + container.scrollTop + 4;
            } else {
              // Position it above the cursor
              pos.bottom = containerRect.bottom - rect.top + 4;
            }

            setDropdownPosition(pos);
          }

          return;
        }
      }
    }

    resetMention();
  };

  useEffect(() => {
    if (!showFilePicker) {
      return;
    }

    const timer = window.setTimeout(() => {
      void fetchFiles(currentQuery);
    }, 150);

    return () => window.clearTimeout(timer);
  }, [showFilePicker, currentQuery]);

  const moveCursorToEnd = () => {
    const editor = editorRef.current;
    if (!editor) return;

    const selection = window.getSelection();
    const range = document.createRange();

    range.selectNodeContents(editor);
    range.collapse(false);

    selection?.removeAllRanges();
    selection?.addRange(range);
  };

  const insertFile = (file: FileItem) => {
    const filePath = file.relative_path || file.filename;
    const fileId = file.file_id || '';
    const filename = file.filename;
    const chipHTML = createFileChipHTML(filePath, fileId, filename);

    editorRef.current?.focus();

    const selection = window.getSelection();
    if (!selection || !selection.rangeCount) return;

    const range = selection.getRangeAt(0);
    const node = range.startContainer;

    if (node.nodeType === Node.TEXT_NODE && node.textContent) {
      const text = node.textContent;
      const cursor = range.startOffset;
      const textBefore = text.slice(0, cursor);
      const atIndex = textBefore.lastIndexOf('@');

      if (atIndex !== -1) {
        range.setStart(node, atIndex);
        range.setEnd(node, cursor);
        selection.removeAllRanges();
        selection.addRange(range);

        document.execCommand('delete');
        document.execCommand('insertHTML', false, chipHTML);
        moveCursorToEnd();
      }
    } else {
      document.execCommand('insertHTML', false, chipHTML);
      moveCursorToEnd();
    }

    resetMention();
    onInput();
  };

  const handleKeyDown = (e: React.KeyboardEvent): boolean => {
    if (showFilePicker) {
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedFileIndex(prev => Math.max(0, prev - 1));
        return true;
      }
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedFileIndex(prev => Math.min(filteredFiles.length - 1, prev + 1));
        return true;
      }
      if (e.key === "Enter" || e.key === "Tab") {
        e.preventDefault();
        if (filteredFiles.length > 0) {
          insertFile(filteredFiles[selectedFileIndex]);
        }
        return true;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        resetMention();
        return true;
      }
    }
    return false;
  };

  return {
    showFilePicker,
    isLoadingFiles,
    filteredFiles,
    selectedFileIndex,
    fileList,
    dropdownPosition,
    insertFile,
    handleKeyDown,
    checkTrigger,
    resetMention,
    setShowFilePicker
  };
}

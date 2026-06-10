"use client";

import React, { useRef, useState } from "react";

interface FileDropzoneRenderState {
    isDragging: boolean;
}

interface FileDropzoneProps {
    disabled?: boolean;
    onFilesDrop: (files: File[]) => void;
    children: (state: FileDropzoneRenderState) => React.ReactNode;
    className?: string;
}

const extractDroppedFiles = (dataTransfer: DataTransfer) => {
    const itemFiles = Array.from(dataTransfer.items || [])
        .filter((item) => item.kind === "file")
        .map((item) => item.getAsFile())
        .filter((file): file is File => file instanceof File);

    return itemFiles.length > 0 ? itemFiles : Array.from(dataTransfer.files || []);
};

const isFileDragEvent = (event: React.DragEvent) =>
    Array.from(event.dataTransfer?.types || []).includes("Files");

export function FileDropzone({
    disabled = false,
    onFilesDrop,
    children,
    className,
}: FileDropzoneProps) {
    const [isDragging, setIsDragging] = useState(false);
    const dragDepthRef = useRef(0);

    const resetDragState = () => {
        dragDepthRef.current = 0;
        setIsDragging(false);
    };

    const handleDragEnter = (event: React.DragEvent<HTMLDivElement>) => {
        if (!isFileDragEvent(event) || disabled) return;
        event.preventDefault();
        event.stopPropagation();
        dragDepthRef.current += 1;
        setIsDragging(true);
    };

    const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
        if (!isFileDragEvent(event) || disabled) return;
        event.preventDefault();
        event.stopPropagation();
        event.dataTransfer.dropEffect = "copy";
        if (!isDragging) {
            setIsDragging(true);
        }
    };

    const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
        if (!isFileDragEvent(event)) return;
        event.preventDefault();
        event.stopPropagation();
        dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
        if (dragDepthRef.current === 0) {
            setIsDragging(false);
        }
    };

    const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
        if (!isFileDragEvent(event) || disabled) return;
        event.preventDefault();
        event.stopPropagation();
        const droppedFiles = extractDroppedFiles(event.dataTransfer);
        resetDragState();
        if (droppedFiles.length > 0) {
            onFilesDrop(droppedFiles);
        }
    };

    return (
        <div
            className={className}
            onDragEnter={handleDragEnter}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
        >
            {children({ isDragging })}
        </div>
    );
}

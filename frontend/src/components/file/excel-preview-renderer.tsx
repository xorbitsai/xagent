"use client"

import { useEffect, useRef, useState } from "react"
import { useI18n } from "@/contexts/i18n-context"
import * as XLSX from "xlsx"

interface ExcelPreviewRendererProps {
    base64Content: string
}

export function ExcelPreviewRenderer({ base64Content }: ExcelPreviewRendererProps) {
    const containerRef = useRef<HTMLDivElement | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [activeSheet, setActiveSheet] = useState<string | null>(null)
    const [sheets, setSheets] = useState<{ [key: string]: string }>({})
    const { t } = useI18n()

    useEffect(() => {
        const render = async () => {
            if (!base64Content) {
                return
            }

            try {
                let workbook;

                // Check if content is likely base64
                const isBase64 = /^[A-Za-z0-9+/=]+$/.test(base64Content.replace(/\s/g, ''));

                if (isBase64) {
                    try {
                        const binary = atob(base64Content)
                        const bytes = new Uint8Array(binary.length)
                        for (let i = 0; i < binary.length; i++) {
                            bytes[i] = binary.charCodeAt(i)
                        }
                        workbook = XLSX.read(bytes, { type: "array" })
                    } catch (e) {
                        // Fallback for non-base64 text
                        workbook = XLSX.read(base64Content, { type: "string" })
                    }
                } else {
                    workbook = XLSX.read(base64Content, { type: "string" })
                }

                const sheetData: { [key: string]: string } = {}

                workbook.SheetNames.forEach((sheetName) => {
                    const worksheet = workbook.Sheets[sheetName]
                    const html = XLSX.utils.sheet_to_html(worksheet)
                    sheetData[sheetName] = html
                })

                setSheets(sheetData)
                if (workbook.SheetNames.length > 0) {
                    setActiveSheet(workbook.SheetNames[0])
                }
                setError(null)
            } catch (e) {
                console.error(e)
                setError(t("files.previewDialog.errors.excelRenderFailed") || "Failed to render Excel file")
            }
        }

        render()
    }, [base64Content, t])

    if (error) {
        return <div className="p-4 text-sm text-destructive">{error}</div>
    }

    if (!activeSheet || !sheets[activeSheet]) {
        return null
    }

    return (
        <div className="flex flex-col h-full overflow-hidden bg-background">
            {Object.keys(sheets).length > 1 && (
                <div className="flex border-b overflow-x-auto bg-muted/30 p-2 gap-2 flex-shrink-0">
                    {Object.keys(sheets).map((sheetName) => (
                        <button
                            key={sheetName}
                            onClick={() => setActiveSheet(sheetName)}
                            className={`px-3 py-1.5 text-sm rounded-md transition-colors whitespace-nowrap ${activeSheet === sheetName
                                ? "bg-background shadow-sm border border-border font-medium"
                                : "text-muted-foreground hover:bg-muted"
                                }`}
                        >
                            {sheetName}
                        </button>
                    ))}
                </div>
            )}

            <div
                className="flex-1 overflow-auto p-4 excel-preview-container"
                ref={containerRef}
            >
                <div
                    className="bg-background rounded-md shadow-sm border min-w-max"
                    dangerouslySetInnerHTML={{ __html: sheets[activeSheet] }}
                />
            </div>

            <style dangerouslySetInnerHTML={{
                __html: `
        .excel-preview-container table {
          border-collapse: collapse;
          width: 100%;
        }
        .excel-preview-container th,
        .excel-preview-container td {
          border: 1px solid hsl(var(--border));
          padding: 8px;
          min-width: 80px;
          text-align: left;
        }
        .excel-preview-container td[data-t="n"] {
          text-align: right;
        }
        .excel-preview-container tr:nth-child(even) {
          background-color: hsl(var(--muted) / 0.3);
        }
      `}} />
        </div>
    )
}

"use client"

import React from "react"
import ReactMarkdown from "react-markdown"

/** Descriptions can come from external connector catalogs, not trusted HTML. */
export function MarkdownDescription({ content }: { content: string }) {
  if (!content) return null
  return (
    <div className="text-xs text-muted-foreground mt-1 ml-2 [&_p]:m-0 [&_p]:leading-normal [&_strong]:text-foreground [&_code]:bg-muted [&_code]:px-1 [&_code]:rounded [&_code]:font-mono [&_ul]:pl-4 [&_ul]:m-0 [&_ol]:pl-4 [&_ol]:m-0">
      <ReactMarkdown
        skipHtml
        allowedElements={["p", "strong", "em", "code", "ul", "ol", "li", "a", "br", "del"]}
        components={{
          a: ({ children, href }) => (
            <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}

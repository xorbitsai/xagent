import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import {
  sanitizeUserMessageFiles,
  stripAttachedFileRefs,
  UserMessageContent,
} from "./user-message-content"

const image = {
  file_id: "355f1fee-48e4-4cb6-afd3-71654e2f5c7e",
  name: "telegram-photo.jpg",
  type: "image/jpeg",
  size: 140575,
}

afterEach(cleanup)

describe("UserMessageContent", () => {
  it("renders attached files inline with the user text", () => {
    const { container } = render(
      <UserMessageContent message="这是什么时候的现场" files={[image]} />
    )

    expect(container.firstElementChild?.textContent).toBe(
      "这是什么时候的现场telegram-photo.jpg"
    )
    expect(screen.queryByText("137.28 KB")).not.toBeInTheDocument()
    expect(container.querySelector("img")).not.toBeInTheDocument()
    expect(container.querySelector('[data-file-kind="image"]')).toBeInTheDocument()
  })

  it("removes a matching legacy runtime file link before adding the chip", () => {
    const message =
      "这是什么时候的现场\n\n[telegram-photo.jpg](file://355f1fee-48e4-4cb6-afd3-71654e2f5c7e)"

    render(<UserMessageContent message={message} files={[image]} />)

    expect(screen.queryByText(/file:\/\//)).not.toBeInTheDocument()
    expect(screen.getAllByText("telegram-photo.jpg")).toHaveLength(1)
  })

  it("opens the selected file in the supplied preview handler", () => {
    const onPreview = vi.fn()
    render(
      <UserMessageContent
        message="Inspect"
        files={[image]}
        onPreview={onPreview}
      />
    )

    fireEvent.click(screen.getByText("telegram-photo.jpg"))

    expect(onPreview).toHaveBeenCalledWith(image, [image], 0)
  })

  it("ignores malformed attachment entries", () => {
    const files = [null, "invalid", {}, image] as unknown as typeof image[]
    const { container } = render(
      <UserMessageContent message="Inspect" files={files} />
    )

    expect(container.firstElementChild?.textContent).toBe(
      "Inspecttelegram-photo.jpg"
    )
    expect(screen.getAllByText("telegram-photo.jpg")).toHaveLength(1)
  })
})

describe("stripAttachedFileRefs", () => {
  it("supports canonical file refs and preserves unrelated refs", () => {
    expect(
      stripAttachedFileRefs(
        "Inspect [photo](file:355f1fee-48e4-4cb6-afd3-71654e2f5c7e) [other](file:other-id)",
        [image]
      )
    ).toBe("Inspect  [other](file:other-id)")
  })

  it("sanitizes non-array and malformed file data", () => {
    expect(sanitizeUserMessageFiles(null)).toEqual([])
    expect(
      sanitizeUserMessageFiles([
        null,
        42,
        {},
        { name: "invalid", file_id: 42 },
        image,
      ])
    ).toEqual([image])

    expect(
      stripAttachedFileRefs(
        "Inspect [photo](file:355f1fee-48e4-4cb6-afd3-71654e2f5c7e)",
        [null, "invalid", image]
      )
    ).toBe("Inspect")
  })

  it("returns an empty message for malformed runtime content", () => {
    expect(stripAttachedFileRefs(null, [image])).toBe("")
    expect(stripAttachedFileRefs({ content: "Inspect" }, [image])).toBe("")
  })
})

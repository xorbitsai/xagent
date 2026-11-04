import { describe, expect, it } from "vitest"

import { extractBuildPreviewResponse, extractSharedChatResponse } from "./chat-response"

describe("chat-response", () => {
  it("extracts shared chat response from nested metadata payloads", () => {
    const payload = {
      metadata: {
        chat_response: {
          message: "请补充目标受众",
          interactions: [
            {
              type: "text_input",
              field: "target_user",
              label: "目标受众",
            },
          ],
        },
      },
    }

    expect(extractSharedChatResponse(payload)).toEqual({
      message: "请补充目标受众",
      interactions: [
        {
          type: "text_input",
          field: "target_user",
          label: "目标受众",
        },
      ],
    })
  })

  it("prefers top-level chat_response from task_completed payloads", () => {
    const payload = {
      type: "task_completed",
      result: "plain result that should not win",
      success: false,
      chat_response: {
        message: "请问你想让我分析什么内容呢？",
        interactions: [
          {
            type: "text_input",
            field: "analysis_content",
            label: "请输入你想分析的内容",
            multiline: true,
          },
        ],
      },
    }

    expect(extractBuildPreviewResponse(payload)).toEqual({
      message: "请问你想让我分析什么内容呢？",
      interactions: payload.chat_response.interactions,
    })
  })

  it("falls back to structured chat json embedded in result", () => {
    const payload = {
      type: "task_completed",
      result: JSON.stringify({
        type: "chat",
        chat: {
          message: "需要更多信息",
          interactions: [
            {
              type: "confirm",
              field: "continue",
              label: "是否继续",
            },
          ],
        },
      }),
    }

    expect(extractBuildPreviewResponse(payload)).toEqual({
      message: "需要更多信息",
      interactions: [
        {
          type: "confirm",
          field: "continue",
          label: "是否继续",
        },
      ],
    })
  })

  it("preserves ordinary json code blocks in non-chat preview results", () => {
    const payload = {
      type: "task_completed",
      result: "Here is the schema:\n```json\n{\"name\":\"demo\"}\n```",
    }

    expect(extractBuildPreviewResponse(payload)).toEqual({
      message: "Here is the schema:\n```json\n{\"name\":\"demo\"}\n```",
    })
  })
})

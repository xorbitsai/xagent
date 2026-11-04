// Test data for MessagesPreview component
export const testLLMContextData = [
  {
    "action": "LLM Call Start",
    "model_name": "glm-4.5",
    "step_name": "Write HTML File",
    "context_messages_count": 6,
    "context_preview": [
      {
        "role": "system",
        "content": "You are executing a specific step in a larger plan: Write HTML File\n\nYour task: Write HTML content to file\n\nYou have access to the results and context from previous dependency steps.\nUse this information to complete your specific task effectively.\n\nWhen you have sufficient information or have completed your task, provide a clear final answer."
      },
      {
        "role": "user",
        "content": "=== Results from dependency step: create_html_content (step_agent_create_html_content) ==="
      },
      {
        "role": "system",
        "content": "You are executing a specific step in a larger plan: Create HTML Content\n\nYour task: Create HTML content containing gradient Hello World\n\nYou have access to the results and context from previous dependency steps.\nUse this information to complete your specific task effectively.\n\nWhen you have sufficient information or have completed your task, provide a clear final answer."
      },
      {
        "role": "user",
        "content": "Execute: Create HTML Content\nDescription: Create HTML content containing gradient Hello World\nYou should call the tool `analysis` with arguments:\n{}"
      },
      {
        "role": "assistant",
        "content": "I'll create the HTML content with a gradient Hello World text. Let me call the analysis tool to understand the requirements better."
      },
      {
        "role": "user",
        "content": "Analysis complete. Now create the HTML content with gradient text animation."
      }
    ],
    "step_id": "write_html_file",
    "temperature": 0.7,
    "max_tokens": 2000
  }
]

export const testToolCallData = [
  {
    "action": "Tool Call Start",
    "tool_name": "write_file_restricted",
    "params_count": 2,
    "tool_params": {
      "file_path": "uploads/hello_world.html",
      "content": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Hello World</title>\n    <style>\n        body {\n            display: flex;\n            justify-content: center;\n            align-items: center;\n            height: 100vh;\n            margin: 0;\n            background-color: #f0f0f0;\n            font-family: Arial, sans-serif;\n        }\n        .gradient-text {\n            font-size: 4rem;\n            font-weight: bold;\n            background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);\n            background-size: 300% 300%;\n            -webkit-background-clip: text;\n            background-clip: text;\n            -webkit-text-fill-color: transparent;\n            animation: gradientShift 3s ease infinite;\n        }\n        @keyframes gradientShift {\n            0% { background-position: 0% 50%; }\n            50% { background-position: 100% 50%; }\n            100% { background-position: 0% 50%; }\n        }\n    </style>\n</head>\n<body>\n    <h1 class=\"gradient-text\">Hello World</h1>\n</body>\n</html>"
    },
    "step_id": "write_html_file",
    "step_name": "Write HTML File"
  }
]

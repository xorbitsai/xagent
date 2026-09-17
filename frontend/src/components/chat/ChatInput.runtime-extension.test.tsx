import React from "react";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiRequestMock = vi.hoisted(() => vi.fn());
const authUserMock = vi.hoisted(() => ({
  current: { id: "2", is_admin: false },
}));

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper",
  );
  return { ...actual, apiRequest: apiRequestMock };
});

vi.mock("@/lib/utils", () => ({
  cn: (...classes: Array<string | false | null | undefined>) =>
    classes.filter(Boolean).join(" "),
  generateClientMessageId: () => "client-message-runtime",
  getApiUrl: () => "http://api.local",
  getUploadApiUrl: () => "http://upload.local",
}));

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}));

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({ openFilePreview: vi.fn() }),
}));

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ user: authUserMock.current }),
}));

vi.mock("@/components/config-dialog", () => ({
  ConfigDialog: ({ trigger }: { trigger: unknown }) => trigger,
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}));

vi.mock("@/hooks/use-file-mention", () => ({
  useFileMention: () => ({
    checkTrigger: vi.fn(),
    dropdownPosition: null,
    fileList: [],
    filteredFiles: [],
    handleKeyDown: vi.fn(() => false),
    insertFile: vi.fn(),
    isLoadingFiles: false,
    resetMention: vi.fn(),
    selectedFileIndex: 0,
    showFilePicker: false,
  }),
}));

vi.mock("sonner", () => ({
  toast: { error: vi.fn(), success: vi.fn() },
}));

vi.mock("@/lib/task-runtime-ui-extension", async () => {
  const actual = await vi.importActual<
    typeof import("@/lib/task-runtime-ui-extension")
  >("@/lib/task-runtime-ui-extension");
  const ReactModule = await vi.importActual<typeof import("react")>("react");
  return {
    ...actual,
    hasTaskRuntimeComposerExtension: true,
    TaskRuntimeComposerMenuExtension: ({
      disabled,
      onSelectionChange,
      onRequestClose,
    }: {
      disabled: boolean;
      onSelectionChange: (selection: unknown) => void;
      onRequestClose: () => void;
    }) => ReactModule.createElement("button", {
      type: "button",
      disabled,
      onClick: () => {
        onSelectionChange({
          runtimeExtensions: { browser_relay: { target: "approved_tab" } },
        });
        onRequestClose();
      },
    }, "My browser"),
    TaskRuntimeComposerSelectionExtension: ({ selection, onSelectionChange }: {
      selection: unknown;
      onSelectionChange: (selection: unknown) => void;
    }) => ReactModule.createElement("button", {
      type: "button",
      onClick: () => onSelectionChange({
        runtimeExtensions: { browser_relay: { target: "approved_tab" } },
      }),
    }, selection ? "My browser selected" : "Select My browser"),
  };
});

import { ChatInput } from "./ChatInput";

function enableLocalBrowserWindow() {
  authUserMock.current = { id: "2", is_admin: true };
  apiRequestMock.mockImplementation((url: string) => Promise.resolve(
    new Response(JSON.stringify(
      url === "http://api.local/api/computer/local-browser/readiness"
        ? {
            ready: true,
            application: "Google Chrome",
            windows: [{
              pid: 100,
              window_id: 20,
              application: "Google Chrome",
              title: "GitHub",
            }],
            issues: [],
            message: "",
          }
        : [],
    ), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }),
  ));
}

describe("ChatInput task runtime UI extension", () => {
  beforeEach(() => {
    authUserMock.current = { id: "2", is_admin: false };
    apiRequestMock.mockReset();
    apiRequestMock.mockImplementation(() => Promise.resolve(
      new Response(JSON.stringify([]), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    ));
  });

  afterEach(() => cleanup());

  it("submits a distribution-provided runtime without exposing local browser", async () => {
    const onSend = vi.fn();
    const { container } = render(
      <ChatInput
        hideFileUpload
        inputValue="inspect my signed-in tab"
        onInputChange={vi.fn()}
        onSend={onSend}
        taskConfig={{ model: "model-1" }}
      />,
    );

    fireEvent.click(screen.getByLabelText("chatPage.input.actions.add"));
    expect(screen.queryByText("chatPage.input.localBrowser.label")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "My browser" }));
    expect(screen.getByText("My browser selected")).toBeInTheDocument();

    fireEvent.submit(container.querySelector("form") as HTMLFormElement);

    await waitFor(() => expect(onSend).toHaveBeenCalledWith(
      "inspect my signed-in tab",
      expect.objectContaining({
        runtimeExtensions: {
          browser_relay: { target: "approved_tab" },
        },
      }),
    ));
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Select My browser" })).toBeInTheDocument();
    });
  });

  it("replaces a local-browser target when the selection slot chooses a runtime", async () => {
    enableLocalBrowserWindow();
    const onSend = vi.fn();
    const { container } = render(
      <ChatInput
        hideFileUpload
        inputValue="inspect my signed-in tab"
        onInputChange={vi.fn()}
        onSend={onSend}
        taskConfig={{ model: "model-1" }}
      />,
    );

    fireEvent.click(screen.getByLabelText("chatPage.input.actions.add"));
    fireEvent.click(screen.getByText("chatPage.input.localBrowser.label"));
    fireEvent.click(await screen.findByText("GitHub"));
    expect(screen.getByLabelText("Google Chrome · GitHub")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Select My browser" }));
    expect(screen.queryByLabelText("Google Chrome · GitHub")).not.toBeInTheDocument();
    fireEvent.submit(container.querySelector("form") as HTMLFormElement);

    await waitFor(() => expect(onSend).toHaveBeenCalledWith(
      "inspect my signed-in tab",
      expect.objectContaining({
        runtimeExtensions: {
          browser_relay: { target: "approved_tab" },
        },
      }),
    ));
  });

  it("clears an extension selection when a local-browser window is picked", async () => {
    enableLocalBrowserWindow();
    const onSend = vi.fn();
    const { container } = render(
      <ChatInput
        hideFileUpload
        inputValue="inspect the selected window"
        onInputChange={vi.fn()}
        onSend={onSend}
        taskConfig={{ model: "model-1" }}
      />,
    );

    fireEvent.click(screen.getByLabelText("chatPage.input.actions.add"));
    fireEvent.click(screen.getByRole("button", { name: "My browser" }));
    expect(screen.getByRole("button", { name: "My browser selected" })).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.queryByRole("button", { name: "My browser" })).not.toBeInTheDocument();
    });

    fireEvent.click(screen.getByLabelText("chatPage.input.actions.add"));
    fireEvent.click(screen.getByText("chatPage.input.localBrowser.label"));
    fireEvent.click(await screen.findByText("GitHub"));
    expect(screen.getByLabelText("Google Chrome · GitHub")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Select My browser" })).toBeInTheDocument();

    fireEvent.submit(container.querySelector("form") as HTMLFormElement);

    await waitFor(() => expect(onSend).toHaveBeenCalledWith(
      "inspect the selected window",
      expect.objectContaining({
        runtimeExtensions: {
          local_browser: {
            pid: 100,
            window_id: 20,
            application: "Google Chrome",
            title: "GitHub",
          },
        },
      }),
    ));
  });

  it("clears a local-browser target when the menu chooses a runtime", async () => {
    enableLocalBrowserWindow();
    const onSend = vi.fn();
    const { container } = render(
      <ChatInput
        hideFileUpload
        inputValue="inspect my signed-in tab"
        onInputChange={vi.fn()}
        onSend={onSend}
        taskConfig={{ model: "model-1" }}
      />,
    );

    fireEvent.click(screen.getByLabelText("chatPage.input.actions.add"));
    fireEvent.click(screen.getByText("chatPage.input.localBrowser.label"));
    fireEvent.click(await screen.findByText("GitHub"));
    expect(screen.getByLabelText("Google Chrome · GitHub")).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("chatPage.input.actions.add"));
    fireEvent.click(screen.getByRole("button", { name: "My browser" }));
    expect(screen.queryByLabelText("Google Chrome · GitHub")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "My browser selected" })).toBeInTheDocument();

    fireEvent.submit(container.querySelector("form") as HTMLFormElement);

    await waitFor(() => expect(onSend).toHaveBeenCalledWith(
      "inspect my signed-in tab",
      expect.objectContaining({
        runtimeExtensions: {
          browser_relay: { target: "approved_tab" },
        },
      }),
    ));
  });
});

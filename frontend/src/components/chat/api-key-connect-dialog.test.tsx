/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { McpApp } from "@/contexts/mcp-apps-context";

const apiRequestMock = vi.hoisted(() => vi.fn());
const toastSuccessMock = vi.hoisted(() => vi.fn());
const toastErrorMock = vi.hoisted(() => vi.fn());

vi.mock("@/lib/api-wrapper", () => ({ apiRequest: apiRequestMock }));

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils");
  return { ...actual, getApiUrl: () => "http://api.local" };
});

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string | number>) =>
      vars ? `${key}:${JSON.stringify(vars)}` : key,
  }),
}));

vi.mock("@/components/ui/sonner", () => ({
  toast: { success: toastSuccessMock, error: toastErrorMock },
}));

import { ApiKeyConnectDialog } from "./api-key-connect-dialog";

function makeApp(overrides: Partial<McpApp> = {}): McpApp {
  return {
    id: "aws",
    name: "AWS",
    description: "",
    icon: "",
    users: "",
    transport: "builtin",
    provider: "",
    category: "Infrastructure",
    auth_type: "api_key",
    launch_config: { required_env: ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"] },
    ...overrides,
  };
}

function jsonResponse(data: unknown, init?: ResponseInit) {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

beforeEach(() => {
  apiRequestMock.mockReset();
  toastSuccessMock.mockReset();
  toastErrorMock.mockReset();
});

afterEach(cleanup);

describe("ApiKeyConnectDialog", () => {
  it("stays closed when app is null", () => {
    render(<ApiKeyConnectDialog app={null} onOpenChange={vi.fn()} onConnected={vi.fn()} />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("falls back to a generated avatar when the app's own icon fails to load", () => {
    render(
      <ApiKeyConnectDialog
        app={makeApp({ icon: "https://example.com/aws-icon.png" })}
        onOpenChange={vi.fn()}
        onConnected={vi.fn()}
      />
    );

    // Rendered via Dialog's portal, outside RTL's `container` - query the
    // document instead.
    const img = document.querySelector("img") as HTMLImageElement;
    expect(img).toHaveAttribute("src", "https://example.com/aws-icon.png");

    fireEvent.error(img);
    expect(img.src).toBe(
      "https://ui-avatars.com/api/?name=AWS&background=random&color=fff&size=128"
    );
  });

  it("renders one password input per required env var, labeled by its name", () => {
    render(<ApiKeyConnectDialog app={makeApp()} onOpenChange={vi.fn()} onConnected={vi.fn()} />);

    const accessKeyInput = screen.getByLabelText("AWS_ACCESS_KEY_ID");
    const secretKeyInput = screen.getByLabelText("AWS_SECRET_ACCESS_KEY");
    expect(accessKeyInput).toHaveAttribute("type", "password");
    expect(secretKeyInput).toHaveAttribute("type", "password");
  });

  it("posts the entered values with env_source 'own' and reports success", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse({ ok: true }));
    const onConnected = vi.fn();
    const onOpenChange = vi.fn();

    render(
      <ApiKeyConnectDialog app={makeApp()} onOpenChange={onOpenChange} onConnected={onConnected} />
    );

    fireEvent.change(screen.getByLabelText("AWS_ACCESS_KEY_ID"), {
      target: { value: "key-123" },
    });
    fireEvent.change(screen.getByLabelText("AWS_SECRET_ACCESS_KEY"), {
      target: { value: "secret-456" },
    });
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }));

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/mcp/apps/aws/connect",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({
            env: { AWS_ACCESS_KEY_ID: "key-123", AWS_SECRET_ACCESS_KEY: "secret-456" },
            env_source: "own",
          }),
        })
      );
    });

    await waitFor(() => {
      expect(toastSuccessMock).toHaveBeenCalledWith(
        'tools.mcp.dialog.connectSuccess:{"name":"AWS"}'
      );
      expect(onConnected).toHaveBeenCalledTimes(1);
      expect(onOpenChange).toHaveBeenCalledWith(false);
    });
  });

  it("sends every required key explicitly, defaulting an untouched one to an empty string", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse({ ok: true }));

    render(<ApiKeyConnectDialog app={makeApp()} onOpenChange={vi.fn()} onConnected={vi.fn()} />);

    // Only fill the first field; leave AWS_SECRET_ACCESS_KEY untouched.
    fireEvent.change(screen.getByLabelText("AWS_ACCESS_KEY_ID"), {
      target: { value: "key-123" },
    });
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }));

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/mcp/apps/aws/connect",
        expect.objectContaining({
          body: JSON.stringify({
            env: { AWS_ACCESS_KEY_ID: "key-123", AWS_SECRET_ACCESS_KEY: "" },
            env_source: "own",
          }),
        })
      );
    });
  });

  it("shows the server error and keeps the dialog open when the connect request fails", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse({ detail: "invalid key" }, { status: 400 }));
    const onOpenChange = vi.fn();

    render(
      <ApiKeyConnectDialog app={makeApp()} onOpenChange={onOpenChange} onConnected={vi.fn()} />
    );

    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }));

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith("invalid key");
    });
    expect(onOpenChange).not.toHaveBeenCalledWith(false);
  });

  it("shows the first message of a FastAPI validation-error array instead of stringifying the whole array", async () => {
    apiRequestMock.mockResolvedValueOnce(
      jsonResponse(
        { detail: [{ loc: ["body", "env"], msg: "field required", type: "missing" }] },
        { status: 422 }
      )
    );

    render(<ApiKeyConnectDialog app={makeApp()} onOpenChange={vi.fn()} onConnected={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }));

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith("field required");
    });
  });

  it("prefills already-configured keys with the masked sentinel instead of blank, so submitting untouched preserves the stored secret", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse({ ok: true }));

    render(
      <ApiKeyConnectDialog
        app={makeApp({ configured_env_keys: ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"] })}
        onOpenChange={vi.fn()}
        onConnected={vi.fn()}
      />
    );

    const accessKeyInput = screen.getByLabelText("AWS_ACCESS_KEY_ID") as HTMLInputElement;
    const secretKeyInput = screen.getByLabelText("AWS_SECRET_ACCESS_KEY") as HTMLInputElement;
    expect(accessKeyInput.value).toBe("********");
    expect(secretKeyInput.value).toBe("********");

    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }));

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/mcp/apps/aws/connect",
        expect.objectContaining({
          body: JSON.stringify({
            env: { AWS_ACCESS_KEY_ID: "********", AWS_SECRET_ACCESS_KEY: "********" },
            env_source: "own",
          }),
        })
      );
    });
  });

  it("prefills only the keys that are actually configured, leaving a missing one blank, for a partially-configured multi-key app", async () => {
    // The exact gap a single app-level "is this app configured" boolean
    // can't cover: AWS_REGION was never set, so app.user_env_configured
    // would be false even though the other two keys are - configured_env_keys
    // is per key precisely so this case doesn't blank (and on submit, wipe)
    // the two that are already there.
    apiRequestMock.mockResolvedValueOnce(jsonResponse({ ok: true }));

    render(
      <ApiKeyConnectDialog
        app={makeApp({
          launch_config: {
            required_env: ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
          },
          configured_env_keys: ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
        })}
        onOpenChange={vi.fn()}
        onConnected={vi.fn()}
      />
    );

    expect((screen.getByLabelText("AWS_ACCESS_KEY_ID") as HTMLInputElement).value).toBe(
      "********"
    );
    expect((screen.getByLabelText("AWS_SECRET_ACCESS_KEY") as HTMLInputElement).value).toBe(
      "********"
    );
    expect((screen.getByLabelText("AWS_REGION") as HTMLInputElement).value).toBe("");

    fireEvent.change(screen.getByLabelText("AWS_REGION"), { target: { value: "us-east-1" } });
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }));

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/mcp/apps/aws/connect",
        expect.objectContaining({
          body: JSON.stringify({
            env: {
              AWS_ACCESS_KEY_ID: "********",
              AWS_SECRET_ACCESS_KEY: "********",
              AWS_REGION: "us-east-1",
            },
            env_source: "own",
          }),
        })
      );
    });
  });

  it("does not prefill the masked sentinel for an app with no stored key yet", () => {
    render(
      <ApiKeyConnectDialog
        app={makeApp({ configured_env_keys: [] })}
        onOpenChange={vi.fn()}
        onConnected={vi.fn()}
      />
    );

    const accessKeyInput = screen.getByLabelText("AWS_ACCESS_KEY_ID") as HTMLInputElement;
    expect(accessKeyInput.value).toBe("");
  });

  it("shows a timeout-specific error when the request aborts", async () => {
    apiRequestMock.mockImplementation(() =>
      Promise.reject(new DOMException("timed out", "TimeoutError"))
    );

    render(<ApiKeyConnectDialog app={makeApp()} onOpenChange={vi.fn()} onConnected={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }));

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith("tools.mcp.alerts.connectTimedOut");
    });
  });

  it("clears entered values and calls onOpenChange(false) when cancelled", () => {
    const onOpenChange = vi.fn();
    render(<ApiKeyConnectDialog app={makeApp()} onOpenChange={onOpenChange} onConnected={vi.fn()} />);

    fireEvent.change(screen.getByLabelText("AWS_ACCESS_KEY_ID"), { target: { value: "key-123" } });
    fireEvent.click(screen.getByRole("button", { name: "common.cancel" }));

    expect(onOpenChange).toHaveBeenCalledWith(false);
  });
});

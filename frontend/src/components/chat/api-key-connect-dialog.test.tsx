/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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

    // If the fallback avatar itself also fails (ui-avatars.com down, an
    // ad-blocker), onError fires again - the handler must bail out instead
    // of writing src again forever. jsdom never actually loads images, so
    // re-writing the identical URL is otherwise invisible from the outside;
    // spy on the src setter itself to prove the second error is a no-op.
    // (Nulling .onerror would NOT catch this - React attaches this handler
    // via addEventListener, not the element's .onerror DOM property, so an
    // implementation that only did that would still write src again here.)
    const srcDescriptor = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, "src")!;
    const setSrcSpy = vi.fn();
    Object.defineProperty(img, "src", {
      configurable: true,
      get: () => srcDescriptor.get!.call(img),
      set: (value: string) => {
        setSrcSpy(value);
        srcDescriptor.set!.call(img, value);
      },
    });

    fireEvent.error(img);
    expect(setSrcSpy).not.toHaveBeenCalled();
  });

  it("renders one password input per required env var, labeled by its name", () => {
    render(<ApiKeyConnectDialog app={makeApp()} onOpenChange={vi.fn()} onConnected={vi.fn()} />);

    const accessKeyInput = screen.getByLabelText("AWS_ACCESS_KEY_ID");
    const secretKeyInput = screen.getByLabelText("AWS_SECRET_ACCESS_KEY");
    expect(accessKeyInput).toHaveAttribute("type", "password");
    expect(secretKeyInput).toHaveAttribute("type", "password");
  });

  it("encodes the app id in the connect route, since it is interpolated into a URL path segment", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse({ ok: true }));

    render(
      <ApiKeyConnectDialog
        app={makeApp({ id: "weird/app?id" })}
        onOpenChange={vi.fn()}
        onConnected={vi.fn()}
      />
    );

    fireEvent.change(screen.getByLabelText("AWS_ACCESS_KEY_ID"), { target: { value: "key-123" } });
    fireEvent.change(screen.getByLabelText("AWS_SECRET_ACCESS_KEY"), {
      target: { value: "secret-456" },
    });
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }));

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/mcp/apps/weird%2Fapp%3Fid/connect",
        expect.anything()
      );
    });
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

  it("disables Connect until every required field has a value, so a partial setup can't be submitted at all", () => {
    // connect_mcp_app accepts (and activates) a partial env - a blank field
    // is dropped, not rejected, and is_connected only checks association
    // membership, not env completeness (src/xagent/web/api/mcp.py). Unlike
    // connect-mcp-dialog.tsx's key form, this dialog offers no shared/
    // platform fallback, so a blank field has no honest reading other than
    // "not done yet" - the button must not be clickable until every field
    // has something in it, or this flow can create a "Connected" row a real
    // tool call would still fail against.
    render(<ApiKeyConnectDialog app={makeApp()} onOpenChange={vi.fn()} onConnected={vi.fn()} />);

    const connectButton = screen.getByRole("button", { name: "tools.mcp.dialog.connect" });
    expect(connectButton).toBeDisabled();

    // Only fill the first field; leave AWS_SECRET_ACCESS_KEY untouched.
    fireEvent.change(screen.getByLabelText("AWS_ACCESS_KEY_ID"), {
      target: { value: "key-123" },
    });
    expect(connectButton).toBeDisabled();

    fireEvent.change(screen.getByLabelText("AWS_SECRET_ACCESS_KEY"), {
      target: { value: "   " },
    });
    expect(connectButton).toBeDisabled();

    fireEvent.change(screen.getByLabelText("AWS_SECRET_ACCESS_KEY"), {
      target: { value: "secret-456" },
    });
    expect(connectButton).not.toBeDisabled();
    expect(apiRequestMock).not.toHaveBeenCalled();
  });

  it("ignores a same-tick double click on Connect, so a fast double-click can't issue two POSTs", async () => {
    // disabled={isSubmitting} alone isn't enough: React batches the
    // setIsSubmitting(true) update, so a second click landing before that
    // commit still reads disabled=false - the same race connect-apps-
    // field.tsx's connectingKeysRef guards against for its own handlers.
    // Both dispatches below go through one outer act() rather than two
    // separate fireEvent.click calls, which would each flush React's state
    // synchronously before returning - the DOM would already show
    // disabled=true before a second, separate fireEvent.click, which would
    // pass even without the ref guard for the wrong reason.
    let resolveConnect: (value: Response) => void = () => {};
    apiRequestMock.mockImplementation(
      () => new Promise((resolve) => { resolveConnect = resolve; })
    );

    render(<ApiKeyConnectDialog app={makeApp()} onOpenChange={vi.fn()} onConnected={vi.fn()} />);

    fireEvent.change(screen.getByLabelText("AWS_ACCESS_KEY_ID"), { target: { value: "key-123" } });
    fireEvent.change(screen.getByLabelText("AWS_SECRET_ACCESS_KEY"), {
      target: { value: "secret-456" },
    });

    const connectButton = screen.getByRole("button", { name: "tools.mcp.dialog.connect" });
    act(() => {
      fireEvent.click(connectButton);
      fireEvent.click(connectButton);
    });

    expect(apiRequestMock).toHaveBeenCalledTimes(1);

    resolveConnect(jsonResponse({ ok: true }));
    await waitFor(() => {
      expect(toastSuccessMock).toHaveBeenCalled();
    });
  });

  it("shows the server error and keeps the dialog open when the connect request fails", async () => {
    apiRequestMock.mockResolvedValueOnce(jsonResponse({ detail: "invalid key" }, { status: 400 }));
    const onOpenChange = vi.fn();

    render(
      <ApiKeyConnectDialog app={makeApp()} onOpenChange={onOpenChange} onConnected={vi.fn()} />
    );

    fireEvent.change(screen.getByLabelText("AWS_ACCESS_KEY_ID"), { target: { value: "key-123" } });
    fireEvent.change(screen.getByLabelText("AWS_SECRET_ACCESS_KEY"), {
      target: { value: "secret-456" },
    });
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

    fireEvent.change(screen.getByLabelText("AWS_ACCESS_KEY_ID"), { target: { value: "key-123" } });
    fireEvent.change(screen.getByLabelText("AWS_SECRET_ACCESS_KEY"), {
      target: { value: "secret-456" },
    });
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

    fireEvent.change(screen.getByLabelText("AWS_ACCESS_KEY_ID"), { target: { value: "key-123" } });
    fireEvent.change(screen.getByLabelText("AWS_SECRET_ACCESS_KEY"), {
      target: { value: "secret-456" },
    });
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

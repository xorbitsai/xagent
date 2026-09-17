import React from "react";
import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import {
  TaskRuntimeComposerMenuExtension,
  TaskRuntimeComposerSelectionExtension,
  TaskRuntimeMessageMetadataExtension,
  TaskRuntimeSettingsExtension,
  hasTaskRuntimeComposerExtension,
} from "./task-runtime-ui-extension";

const composerProps = {
  disabled: false,
  selection: null,
  onSelectionChange: vi.fn(),
  onRequestClose: vi.fn(),
};

describe("default task runtime UI extension", () => {
  it("disables the composer extension in the OSS build", () => {
    expect(hasTaskRuntimeComposerExtension).toBe(false);
  });

  it("keeps the composer menu slot inert", () => {
    const { container } = render(
      <TaskRuntimeComposerMenuExtension {...composerProps} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("keeps the composer selection slot inert", () => {
    const { container } = render(
      <TaskRuntimeComposerSelectionExtension
        disabled={composerProps.disabled}
        selection={composerProps.selection}
        onSelectionChange={composerProps.onSelectionChange}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("keeps the settings slot inert", () => {
    const { container } = render(<TaskRuntimeSettingsExtension />);
    expect(container).toBeEmptyDOMElement();
  });

  it("keeps the message metadata slot inert", () => {
    const { container } = render(
      <TaskRuntimeMessageMetadataExtension
        bindings={["local_browser"]}
        publicMetadata={{ local_browser: { kind: "local_browser" } }}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});

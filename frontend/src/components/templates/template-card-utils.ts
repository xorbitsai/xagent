const INTERACTIVE_ELEMENT_SELECTOR = [
  "button",
  "a[href]",
  "input",
  "select",
  "textarea",
  "summary",
  '[role="button"]',
  '[role="link"]',
].join(", ");

export function isNestedInteractiveElement(target: EventTarget | null, currentTarget: EventTarget | null) {
  if (!(target instanceof Element) || !(currentTarget instanceof Element)) {
    return false;
  }

  const interactiveElement = target.closest(INTERACTIVE_ELEMENT_SELECTOR);
  return interactiveElement !== null && interactiveElement !== currentTarget;
}

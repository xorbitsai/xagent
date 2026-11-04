import DOMPurify from "dompurify"

type DocxPreviewModule = Pick<typeof import("docx-preview"), "parseAsync" | "renderDocument">

export const DOCX_PREVIEW_OPTIONS = {
  className: "docx-preview",
  inWrapper: true,
  // Images and fonts become data: URLs, so the sanitized output never needs
  // blob: URLs or object URL lifetimes.
  useBase64URL: true,
  // altChunk embeds arbitrary HTML in an iframe; it is never rendered.
  renderAltChunks: false,
  renderComments: false,
  experimental: false,
}

// DOMPurify applies ALLOWED_URI_REGEXP to every attribute that is not
// URI-safe, so it must still accept plain values (widths, colors, ...). This
// is DOMPurify's default expression restricted to the http(s)/mailto schemes.
// Image data: URLs pass through its DATA_URI_TAGS rule (img/image only).
const ALLOWED_URI = /^(?:(?:https?|mailto):|[^a-z]|[a-z+.\-]+(?:[^a-z+.\-:]|$))/i
// Link targets are then narrowed further: absolute http(s)/mailto URLs, or a
// bookmark inside the document. Relative URLs would resolve against the app.
const SAFE_LINK = /^(?:(?:https?|mailto):|#)/i

const FORBID_TAGS = [
  "script", "noscript", "template", "iframe", "frame", "frameset", "object",
  "embed", "applet", "portal", "form", "input", "button", "textarea", "select",
  "option", "base", "link", "meta", "foreignobject", "use", "video", "audio",
  "source", "track", "picture",
]

const FORBID_ATTR = ["srcset", "action", "formaction", "ping", "target"]

/**
 * docx-preview writes `w:sym/@w:char` into `span.innerHTML` as `&#x${char};`.
 * That sink runs in the app document even for a detached container, so only
 * plain hex code points may reach the renderer.
 */
export function neutralizeDocxSymbols(model: unknown): void {
  const seen = new WeakSet<object>()
  const visit = (value: unknown) => {
    if (!value || typeof value !== "object" || seen.has(value)) return
    if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) return
    if (typeof Blob !== "undefined" && value instanceof Blob) return
    seen.add(value)
    const node = value as Record<string, unknown>
    if (node.type === "symbol" && "char" in node
      && !(typeof node.char === "string" && /^[0-9a-f]{1,6}$/i.test(node.char))) {
      node.char = "fffd"
    }
    for (const [key, child] of Object.entries(node)) {
      // The zip package holds raw file bytes, not rendered document nodes.
      if (key === "_package" || key === "parent") continue
      visit(child)
    }
  }
  visit(model)
}

const CSS_ESCAPE = /\\(?:([0-9a-f]{1,6})[ \t\n\f]?|\r\n|([\s\S]))/gi
const CSS_URL = /url\(\s*(?:"([^"]*)"|'([^']*)'|([^)"'\s]*))\s*\)/gi
const SAFE_CSS_DATA_URL = /^data:[^\s"'()\\]*$/i

/**
 * Strip CSS that can fetch or execute anything: only data: URLs survive in
 * url(); @import, image-set(), src() and legacy expression()/binding hooks are
 * disabled. Escapes for ASCII characters are dropped first, so they cannot
 * spell those tokens (e.g. `u\72l(`). docx-preview's own `\9` / `\a0`
 * numbering suffixes are kept.
 */
export function sanitizeDocxCss(css: string): string {
  let text = css.replace(/\u0000/g, "").replace(CSS_ESCAPE, (match, hex?: string, char?: string) => {
    const code = hex ? parseInt(hex, 16) : (char ?? "\n").codePointAt(0) ?? 0
    return code < 0x80 && code !== 0x09 ? "" : match
  })
  text = text.replace(/\/\*[\s\S]*?(?:\*\/|$)/g, "")

  const keptUrls: string[] = []
  text = text.replace(CSS_URL, (_match, dq?: string, sq?: string, bare?: string) => {
    const value = (dq ?? sq ?? bare ?? "").trim()
    if (!SAFE_CSS_DATA_URL.test(value)) return "none"
    keptUrls.push(`url("${value}")`)
    return `\u0000${keptUrls.length - 1}\u0000`
  })
  text = text
    .replace(/url\s*\(/gi, "invalid(")
    .replace(/(?:-webkit-)?image-set\s*\(|\bsrc\s*\(|expression\s*\(/gi, "invalid(")
    .replace(/@import/gi, "@invalid")
    .replace(/-moz-binding|\bbehavior\s*:|javascript\s*:/gi, "invalid")
    .replace(/</g, "\\3c ")
  return text.replace(/\u0000(\d+)\u0000/g, (_match, index: string) => keptUrls[Number(index)] ?? "none")
}

function sanitizeStagedCss(root: Element): void {
  for (const style of Array.from(root.querySelectorAll("style"))) {
    style.textContent = sanitizeDocxCss(style.textContent ?? "")
  }
  for (const element of [root, ...Array.from(root.querySelectorAll("*"))]) {
    const isSvg = element.namespaceURI === "http://www.w3.org/2000/svg"
    for (const attr of Array.from(element.attributes)) {
      // SVG presentation attributes (fill, stroke, filter, ...) are CSS values
      // and accept url() references too.
      if (attr.name === "style" || (isSvg && !/^(?:xlink:)?href$/i.test(attr.name))) {
        const value = sanitizeDocxCss(attr.value)
        if (value !== attr.value) element.setAttribute(attr.name, value)
      }
    }
  }
}

/**
 * Sanitize docx-preview output that was rendered into a detached container.
 * Returns an inert fragment that is safe to insert into the preview shadow
 * root: no scripts, frames, forms or event handlers; links restricted to
 * http(s)/mailto/#bookmark; images restricted to data: URLs.
 */
export function sanitizeDocxPreview(staging: Element): DocumentFragment {
  sanitizeStagedCss(staging)
  const fragment = DOMPurify.sanitize(staging, {
    RETURN_DOM_FRAGMENT: true,
    ALLOWED_URI_REGEXP: ALLOWED_URI,
    FORBID_TAGS,
    FORBID_ATTR,
    ALLOW_DATA_ATTR: false,
  })

  for (const image of Array.from(fragment.querySelectorAll("img, image"))) {
    for (const name of ["src", "href", "xlink:href"]) {
      const value = image.getAttribute(name)
      if (value !== null && !/^data:/i.test(value.trim())) image.removeAttribute(name)
    }
  }
  for (const link of Array.from(fragment.querySelectorAll("a"))) {
    link.removeAttribute("xlink:href")
    const href = link.getAttribute("href")
    if (href === null) continue
    if (!SAFE_LINK.test(href.trim())) {
      link.removeAttribute("href")
      continue
    }
    if (href.trim().startsWith("#")) continue
    link.setAttribute("target", "_blank")
    link.setAttribute("rel", "noopener noreferrer")
  }
  return fragment
}

/**
 * Parse and render an untrusted .docx without letting library output touch
 * the live document until DOMPurify has processed it.
 */
export async function renderSanitizedDocx(
  docxPreview: DocxPreviewModule,
  data: ArrayBuffer,
): Promise<DocumentFragment> {
  const model = await docxPreview.parseAsync(data, DOCX_PREVIEW_OPTIONS)
  neutralizeDocxSymbols(model)
  const staging = document.createElement("div")
  await docxPreview.renderDocument(model, staging, undefined, DOCX_PREVIEW_OPTIONS)
  return sanitizeDocxPreview(staging)
}

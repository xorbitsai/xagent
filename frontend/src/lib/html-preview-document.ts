import { defaultTreeAdapter, parse, serialize } from "parse5"
import type { DefaultTreeAdapterMap } from "parse5"

type Element = DefaultTreeAdapterMap["element"]
type ParentNode = DefaultTreeAdapterMap["parentNode"]

/** Parse as data, not a browser DOM that can initiate resource requests. */
export function withPreviewReferrerPolicy(source: string): string {
  const document = parse(source)
  const removeReferrerPolicies = (parent: ParentNode) => {
    for (const node of [...parent.childNodes]) {
      if (!("tagName" in node)) continue
      if (node.tagName === "meta" && node.attrs.some(
        (attr) => attr.name === "name" && attr.value.toLowerCase() === "referrer",
      )) {
        defaultTreeAdapter.detachNode(node)
      } else {
        removeReferrerPolicies(node)
      }
    }
  }
  removeReferrerPolicies(document)
  const html = document.childNodes.find((node): node is Element => "tagName" in node && node.tagName === "html")!
  const head = html.childNodes.find((node): node is Element => "tagName" in node && node.tagName === "head")!
  const policy = defaultTreeAdapter.createElement("meta", html.namespaceURI, [
    { name: "name", value: "referrer" },
    { name: "content", value: "no-referrer" },
  ])
  if (head.childNodes.length) defaultTreeAdapter.insertBefore(head, policy, head.childNodes[0])
  else defaultTreeAdapter.appendChild(head, policy)
  return serialize(document)
}

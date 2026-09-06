import remarkMath from 'remark-math'
import type { Plugin } from 'unified'

const DOLLAR_CODE = '$'.charCodeAt(0)
const ZERO_CODE = '0'.charCodeAt(0)
const NINE_CODE = '9'.charCodeAt(0)

/**
 * Exclude common price-pair delimiters while retaining explicit $math$.
 * This is not a semantic currency classifier: ambiguous $...$ pairs (including
 * $10$ and math followed by punctuation) remain math. Escape literal dollars
 * in ambiguous prose, or use code spans for expressions such as $X+$Y.
 * Tight single-dollar delimiters exclude all whitespace, including tabs and
 * Unicode spaces; explicit multi-dollar math retains its padding behavior.
 */
export const remarkCurrencySafeMath: Plugin = function () {
  const extensionStart = this.data().micromarkExtensions?.length ?? 0
  remarkMath.call(this)

  // Extend remark-math's tokenizer rather than rewriting Markdown source: code,
  // escapes, link destinations, and display math must retain their own parsing.
  // Only inspect this invocation's registrations: another plugin may already
  // own a dollar construct, or even an earlier mathText tokenizer.
  const extensions = this.data().micromarkExtensions?.slice(extensionStart)
  const mathText = extensions?.flatMap((extension) => {
    const text = extension.text?.[DOLLAR_CODE]
    return Array.isArray(text) ? text : text ? [text] : []
  }).find((construct) => construct.name === 'mathText')

  if (!mathText) {
    throw new Error('remarkCurrencySafeMath: remark-math did not register a mathText tokenizer')
  }

  const tokenize = mathText.tokenize
  mathText.tokenize = function (effects, ok, nok) {
    const start = this.now()

    return tokenize.call(this, effects, (code) => {
      const source = this.sliceSerialize({ start, end: this.now() })

      // A single-dollar opener/closer must touch non-whitespace content, and
      // a closer cannot precede a digit (the next price in "$10–$25"). Reject
      // the entire attempt so Markdown can still parse intervening emphasis
      // and links. Explicit multi-dollar math keeps remark-math's behavior.
      if (!source.startsWith('$$') && (
        /^\$\s|\s\$$/.test(source) ||
        (code !== null && code >= ZERO_CODE && code <= NINE_CODE)
      )) {
        return nok(code)
      }

      return ok(code)
    }, nok)
  }
}

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { unified, type Data, type Processor } from 'unified'
import remarkParse from 'remark-parse'
import remarkMath from 'remark-math'
import { remarkCurrencySafeMath } from './remark-currency-safe-math'

vi.mock('remark-math', () => ({ default: vi.fn() }))

let realRemarkMath: typeof remarkMath
const DOLLAR_CODE = '$'.charCodeAt(0)
const registrationError = 'remarkCurrencySafeMath: remark-math did not register a mathText tokenizer'

function createMathExtension() {
  return unified().use(realRemarkMath).freeze().data().micromarkExtensions![0]
}

function getMathText(extension: NonNullable<Data['micromarkExtensions']>[number]) {
  const text = extension.text![DOLLAR_CODE]!
  return (Array.isArray(text) ? text : [text]).find((construct) => construct.name === 'mathText')!
}

describe('currency-safe math registration', () => {
  beforeEach(async () => {
    realRemarkMath = (await vi.importActual<typeof import('remark-math')>('remark-math')).default
    vi.mocked(remarkMath).mockImplementation(realRemarkMath)
  })
  afterEach(() => vi.mocked(remarkMath).mockReset())

  it.each([
    ['missing registry', undefined],
    ['empty registry', []],
    ['missing dollar syntax', [{}]],
    ['empty dollar constructs', [{ text: { [DOLLAR_CODE]: [] } }]],
  ] satisfies Array<[string, Data['micromarkExtensions']]>)(
    'reports incompatible remark-math registration: %s',
    (_name, extensions) => {
      vi.mocked(remarkMath).mockImplementation(function (this: Processor) {
        this.data().micromarkExtensions = extensions
      })

      expect(() => unified().use(remarkCurrencySafeMath).freeze()).toThrow(registrationError)
    },
  )

  it('rejects a newly registered dollar construct with an unexpected name', () => {
    vi.mocked(remarkMath).mockImplementation(function (this: Processor) {
      const extension = createMathExtension()
      getMathText(extension).name = 'renamedMathText'
      this.data().micromarkExtensions = [extension]
    })

    expect(() => unified().use(remarkCurrencySafeMath).freeze()).toThrow(registrationError)
  })

  it('does not mistake an existing mathText for a missing new registration', () => {
    const existing = createMathExtension()
    const originalTokenizer = getMathText(existing).tokenize
    vi.mocked(remarkMath).mockImplementation(() => undefined)
    const processor = unified().data('micromarkExtensions', [existing])

    expect(() => processor.use(remarkCurrencySafeMath).freeze()).toThrow(registrationError)
    expect(getMathText(existing).tokenize).toBe(originalTokenizer)
  })

  it('selects the new mathText even with earlier dollar plugins and later registrations', () => {
    const existing = createMathExtension()
    const existingTokenizer = getMathText(existing).tokenize
    const otherDollar = { ...getMathText(createMathExtension()), name: 'otherDollarSyntax' }
    const otherTokenizer = otherDollar.tokenize
    let newTokenizer: typeof otherTokenizer | undefined

    vi.mocked(remarkMath).mockImplementation(function (this: Processor) {
      realRemarkMath.call(this)
      const extensions = this.data().micromarkExtensions!
      const added = extensions[extensions.length - 1]
      const mathText = getMathText(added)
      newTokenizer = mathText.tokenize
      added.text![DOLLAR_CODE] = [otherDollar, mathText]
      extensions.push({})
    })

    const processor = unified()
      .data('micromarkExtensions', [{ text: { [DOLLAR_CODE]: otherDollar } }, existing])
      .use(remarkCurrencySafeMath)
      .freeze()
    const extensions = processor.data().micromarkExtensions!

    expect(getMathText(extensions[2]).tokenize).not.toBe(newTokenizer)
    expect(getMathText(existing).tokenize).toBe(existingTokenizer)
    expect(otherDollar.tokenize).toBe(otherTokenizer)

    // Exercise the selected tokenizer, not just its identity. Isolate it from
    // the deliberately unguarded older parser fixtures; this is a registration
    // contract test, not a claim that remark-math currently emits this shape.
    const parser = unified().use(remarkParse).data({
      ...processor.data(),
      micromarkExtensions: [{ text: { [DOLLAR_CODE]: getMathText(extensions[2]) } }],
    })
    expect(parser.parse('Budget $10–$25; $x^2$.')).toMatchObject({
      children: [{
        type: 'paragraph',
        children: [
          { type: 'text', value: 'Budget $10–$25; ' },
          { type: 'inlineMath', value: 'x^2' },
          { type: 'text', value: '.' },
        ],
      }],
    })
  })

  it('keeps currency literal and math enabled with the real dependency', () => {
    const tree = unified().use(remarkParse).use(remarkCurrencySafeMath).parse('Budget $10–$25; $x^2$.')

    expect(tree).toMatchObject({
      children: [{
        type: 'paragraph',
        children: [
          { type: 'text', value: 'Budget $10–$25; ' },
          { type: 'inlineMath', value: 'x^2' },
          { type: 'text', value: '.' },
        ],
      }],
    })
  })
})

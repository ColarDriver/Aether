import { describe, expect, it } from 'vitest'

import {
  looksLikeIntendedToolUse,
  shouldShowPhantomHint,
  stripAllCommandFences,
  stripToolBlocks
} from '../lib/phantomTool.js'

describe('stripToolBlocks', () => {
  it('removes complete <tool_call> blocks', () => {
    const text =
      'before <tool_call name="x">{"foo": 1}</tool_call> after'
    expect(stripToolBlocks(text)).toBe('before  after')
  })

  it('hides everything from a partial open tag onward', () => {
    const text = 'visible <tool_call name="x">{"a": '
    expect(stripToolBlocks(text)).toBe('visible ')
  })

  it('strips <function=name>...</function=...> partial syntax', () => {
    const text = 'hello <function=foo>{"arg":1}'
    expect(stripToolBlocks(text)).toBe('hello ')
  })

  it('drops bare [tool_call(...)] placeholder lines', () => {
    const text = 'paragraph one\n[tool_call(name=foo)]\nparagraph two'
    expect(stripToolBlocks(text)).toContain('paragraph one')
    expect(stripToolBlocks(text)).toContain('paragraph two')
    expect(stripToolBlocks(text)).not.toContain('tool_call')
  })

  it('collapses runs of three or more newlines', () => {
    expect(stripToolBlocks('a\n\n\n\nb')).toBe('a\n\nb')
  })

  it('returns empty string for empty input', () => {
    expect(stripToolBlocks('')).toBe('')
  })

  it('strips <tool_result>, <function_call>, <invoke>, <thinking> blocks', () => {
    const cases: [string, string][] = [
      ['hello <tool_result>r</tool_result> world', 'hello  world'],
      ['hello <function_call>{"x":1}</function_call> world', 'hello  world'],
      ['hello <invoke name="x">y</invoke> world', 'hello  world'],
      ['hello <thinking>secret</thinking> world', 'hello  world']
    ]
    for (const [input, expected] of cases) {
      expect(stripToolBlocks(input)).toBe(expected)
    }
  })

  it('tolerates namespace-prefixed tags', () => {
    const input = 'a <anthropic:function_calls>x</anthropic:function_calls> b'
    expect(stripToolBlocks(input)).toBe('a  b')
  })

  it('hides partial <thinking onward (mid-stream)', () => {
    expect(stripToolBlocks('keep <thinking aloud')).toBe('keep ')
    expect(stripToolBlocks('keep <think')).toBe('keep ')
  })

  it('hides partial [tool: opener', () => {
    expect(stripToolBlocks('keep [tool: read_fi')).toBe('keep ')
  })

  it('does not strip arbitrary <span tags', () => {
    expect(stripToolBlocks('keep <span class')).toBe('keep <span class')
  })
})

describe('looksLikeIntendedToolUse', () => {
  it('detects fenced shell blocks', () => {
    expect(looksLikeIntendedToolUse('Try ```bash\nls -la\n``` here')).toBe(true)
  })

  it('detects $ shell prompts on their own line', () => {
    expect(looksLikeIntendedToolUse('result:\n$ ls -la')).toBe(true)
  })

  it('detects imperative phrasing', () => {
    expect(looksLikeIntendedToolUse('Let me run the script')).toBe(true)
    expect(looksLikeIntendedToolUse("I'll grep the codebase")).toBe(true)
  })

  it('returns false for plain greetings', () => {
    expect(looksLikeIntendedToolUse('hello, what can I help with?')).toBe(false)
  })
})

describe('stripAllCommandFences', () => {
  it('removes shell-tagged fences', () => {
    const input = 'before\n```bash\nls\n```\nafter'
    const out = stripAllCommandFences(input)
    expect(out).not.toContain('ls')
    expect(out).toContain('before')
    expect(out).toContain('after')
  })

  it('returns the original text when there are no fences', () => {
    expect(stripAllCommandFences('plain text')).toBe('plain text')
  })

  it('returns the original text when the only fences are non-shell', () => {
    const input = '```python\nprint(1)\n```'
    expect(stripAllCommandFences(input)).toBe(input)
  })
})

describe('shouldShowPhantomHint', () => {
  it('returns false once a tool actually dispatched', () => {
    expect(shouldShowPhantomHint({ text: 'I will run something', toolDispatched: true })).toBe(
      false
    )
  })

  it('returns true when prose suggests intent and no tool dispatched', () => {
    expect(shouldShowPhantomHint({ text: 'I will run something', toolDispatched: false })).toBe(
      true
    )
  })
})

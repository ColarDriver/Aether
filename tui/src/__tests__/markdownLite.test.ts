import { describe, expect, it } from 'vitest'

import { parseInline, parseMarkdownLite } from '../lib/markdownLite.js'

describe('markdownLite', () => {
  it('parses bold and inline code', () => {
    expect(parseInline('use **bold** and `code`')).toEqual([
      { kind: 'text', text: 'use ' },
      { kind: 'bold', text: 'bold' },
      { kind: 'text', text: ' and ' },
      { kind: 'code', text: 'code' }
    ])
  })

  it('parses fenced code blocks', () => {
    expect(parseMarkdownLite('hello\n```\nconst x = 1\n```\nbye')).toEqual([
      { kind: 'paragraph', segments: [{ kind: 'text', text: 'hello' }] },
      { kind: 'code', text: 'const x = 1' },
      { kind: 'paragraph', segments: [{ kind: 'text', text: 'bye' }] }
    ])
  })
})

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

  it('parses headings, lists, and fenced code blocks', () => {
    expect(parseMarkdownLite('# Hello\n\n- alpha\n- beta\n\n```js\nconst x = 1\n```\nbye')).toEqual([
      { kind: 'heading', depth: 1, segments: [{ kind: 'text', text: 'Hello' }] },
      {
        kind: 'list',
        ordered: false,
        start: 1,
        items: [
          [{ kind: 'text', text: 'alpha' }],
          [{ kind: 'text', text: 'beta' }]
        ]
      },
      { kind: 'code', text: 'const x = 1', language: 'js' },
      { kind: 'paragraph', segments: [{ kind: 'text', text: 'bye' }] }
    ])
  })

  it('parses italic inline segments', () => {
    expect(parseInline('use *italics* too')).toEqual([
      { kind: 'text', text: 'use ' },
      { kind: 'italic', text: 'italics' },
      { kind: 'text', text: ' too' }
    ])
  })
})

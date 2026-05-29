// @vitest-environment jsdom

import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { DiffViewer, parseUnifiedDiff } from './DiffViewer'

describe('DiffViewer', () => {
  it('marks added and removed lines', () => {
    render(<DiffViewer diff={'@@ -1,1 +1,1 @@\n-old\n+new'} />)

    expect(document.querySelector('.diff-line-remove')?.textContent).toContain('old')
    expect(document.querySelector('.diff-line-add')?.textContent).toContain('new')
    expect(document.querySelector('.diff-line-remove .diff-line-number')?.textContent).toContain('1')
    expect(document.querySelector('.diff-line-add .diff-line-number')?.textContent).toContain('1')
    expect(document.querySelectorAll('.diff-line-remove .diff-marker')).toHaveLength(1)
    expect(document.querySelectorAll('.diff-line-add .diff-marker')).toHaveLength(1)
  })

  it('parses unified diff hunks with old and new line numbers', () => {
    expect(parseUnifiedDiff('@@ -2,2 +2,3 @@\n keep\n-old\n+new\n+more')).toEqual([
      { kind: 'hunk', marker: '@', content: '@@ -2,2 +2,3 @@', oldLine: null, newLine: null },
      { kind: 'context', marker: ' ', content: 'keep', oldLine: 2, newLine: 2 },
      { kind: 'remove', marker: '-', content: 'old', oldLine: 3, newLine: null },
      { kind: 'add', marker: '+', content: 'new', oldLine: null, newLine: 3 },
      { kind: 'add', marker: '+', content: 'more', oldLine: null, newLine: 4 },
    ])
  })
})

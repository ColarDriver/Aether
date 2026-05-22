// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import type { DiffBlock } from '../../../chat-rendering'
import { CurrentTurnChangeCard } from './CurrentTurnChangeCard'

afterEach(cleanup)

const base = {
  id: 'diff-1',
  sessionId: 's1',
  runId: 'r1',
  timestamp: 1,
  source: 'live',
  kind: 'diff',
  origin: 'tool_result',
} as const

describe('CurrentTurnChangeCard', () => {
  it('summarizes changed files and expands per-file diffs', () => {
    const diffs: DiffBlock[] = [
      { ...base, id: 'd1', path: 'src/app.ts', diff: '@@ -1,2 +1,3 @@\n const a = 1\n-old\n+new\n+added' },
      { ...base, id: 'd2', path: 'src/lib.ts', diff: '@@ -1,1 +1,1 @@\n-before\n+after' },
    ]

    render(<CurrentTurnChangeCard diffs={diffs} />)

    expect(screen.getByRole('region', { name: 'Changed files' })).toBeTruthy()
    expect(screen.getByRole('button', { name: /2 changed files/ })).toBeTruthy()
    expect(screen.getByText('src/app.ts')).toBeTruthy()
    expect(screen.getByText('src/lib.ts')).toBeTruthy()
    expect(screen.getByText('+3')).toBeTruthy()
    expect(screen.getByText('-2')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /2 changed files/ }))

    expect(document.querySelector('.diff-line-add')?.textContent).toContain('new')
    expect(document.querySelector('.diff-line-remove')?.textContent).toContain('old')
  })

  it('does not render without usable diff content', () => {
    const { container } = render(<CurrentTurnChangeCard diffs={[{ ...base, id: 'empty', path: 'src/empty.ts', diff: '' }]} />)

    expect(container.firstChild).toBeNull()
  })
})

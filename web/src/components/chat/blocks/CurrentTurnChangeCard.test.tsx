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
    expect(screen.getAllByText('modified').length).toBeGreaterThan(0)
    expect(screen.getByText('src/app.ts')).toBeTruthy()
    expect(screen.getByText('src/lib.ts')).toBeTruthy()
    expect(screen.getByText('+3')).toBeTruthy()
    expect(screen.getByText('-2')).toBeTruthy()
    expect(screen.getAllByText('1 hunk')).toHaveLength(2)
    expect(screen.getByRole('button', { name: 'Copy src/app.ts' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /2 changed files/ }))

    expect(document.querySelector('.diff-line-add')?.textContent).toContain('new')
    expect(document.querySelector('.diff-line-remove')?.textContent).toContain('old')
  })

  it('summarizes diagnostics for changed files and expands diagnostic details', () => {
    const diffs: DiffBlock[] = [
      { ...base, id: 'd-auth', path: 'src/auth.ts', diff: '@@ -1,1 +1,2 @@\n-old auth\n+new auth\n+guard' },
    ]

    render(
      <CurrentTurnChangeCard
        diffs={diffs}
        diagnostics={[
          {
            ...base,
            id: 'diag-auth',
            kind: 'diagnostics',
            content: '<diagnostics />',
            files: [
              {
                path: '/workspace/Aether/src/auth.ts',
                diagnostics: [
                  { severity: 'error', line: 4, column: 8, source: 'pyright', code: 'reportGeneralTypeIssues', message: 'bad auth type' },
                  { severity: 'warning', line: 8, column: 2, source: 'eslint', message: 'unused guard' },
                ],
              },
            ],
          },
        ]}
      />,
    )

    expect(screen.getAllByText('1 error').length).toBeGreaterThan(0)
    expect(screen.getAllByText('1 warning').length).toBeGreaterThan(0)

    fireEvent.click(screen.getByRole('button', { name: /1 changed file/ }))

    expect(screen.getByLabelText('Diagnostics for changed file')).toBeTruthy()
    expect(screen.getByText('bad auth type')).toBeTruthy()
    expect(screen.getByText('unused guard')).toBeTruthy()
    expect(screen.getByText('4:8')).toBeTruthy()
  })

  it('classifies created and deleted files from unified diff headers', () => {
    const diffs: DiffBlock[] = [
      { ...base, id: 'create', path: null, diff: '--- /dev/null\n+++ b/src/new.ts\n@@ -0,0 +1,1 @@\n+export const created = true' },
      { ...base, id: 'delete', path: null, diff: '--- a/src/old.ts\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-export const old = true' },
    ]

    render(<CurrentTurnChangeCard diffs={diffs} />)

    expect(screen.getByText('src/new.ts')).toBeTruthy()
    expect(screen.getByText('src/old.ts')).toBeTruthy()
    expect(screen.getAllByText('created').length).toBeGreaterThan(0)
    expect(screen.getAllByText('deleted').length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: 'Copy src/new.ts' })).toBeTruthy()
  })

  it('does not render without usable diff content', () => {
    const { container } = render(<CurrentTurnChangeCard diffs={[{ ...base, id: 'empty', path: 'src/empty.ts', diff: '' }]} />)

    expect(container.firstChild).toBeNull()
  })
})

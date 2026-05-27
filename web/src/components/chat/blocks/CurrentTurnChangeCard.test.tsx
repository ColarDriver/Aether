// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../../../api/client'
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

  it('renders checkpoint-backed changed files and lazy-loads per-file diffs', async () => {
    vi.spyOn(api, 'sessionTurnCheckpointDiff').mockResolvedValue({
      session_id: 's1',
      state: 'ok',
      path: 'src/lazy.ts',
      diff: '--- a/src/lazy.ts\n+++ b/src/lazy.ts\n@@ -1,1 +1,1 @@\n-old\n+new\n',
      target: {
        target_user_message_id: 'turn-1',
        user_message_index: 0,
        user_message_count: 1,
        message_index: 0,
      },
      checkpoint_id: 'cp-1',
    })

    render(
      <CurrentTurnChangeCard
        diffs={[]}
        sessionId="s1"
        serverCheckpoint={{
          target: {
            target_user_message_id: 'turn-1',
            user_message_index: 0,
            user_message_count: 1,
            message_index: 0,
          },
          code: {
            available: true,
            files_changed: ['src/lazy.ts'],
            insertions: 1,
            deletions: 1,
            checkpoint_id: 'cp-1',
          },
        }}
      />,
    )

    expect(screen.getByRole('region', { name: 'Changed files' })).toBeTruthy()
    expect(screen.getByText('src/lazy.ts')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /1 changed file/ }))

    expect(api.sessionTurnCheckpointDiff).toHaveBeenCalledWith('s1', {
      path: 'src/lazy.ts',
      target_user_message_id: 'turn-1',
      user_message_index: 0,
    })
    expect(await screen.findByText(/old/)).toBeTruthy()
    expect(screen.getByText(/new/)).toBeTruthy()
  })

  it('allows checkpoint-backed revert when old text is unavailable', () => {
    const onRevertFile = vi.fn()
    const diffs: DiffBlock[] = [
      { ...base, id: 'd-clean', path: 'src/clean.ts', diff: '@@ -1,1 +1,1 @@\n-before\n+after' },
    ]

    render(
      <CurrentTurnChangeCard
        diffs={diffs}
        checkpoint={{ checkpointId: '20260527010101-abcdef12', label: 'Before web run', files: ['src/preexisting.ts'] }}
        onRevertFile={onRevertFile}
      />,
    )

    expect(screen.getByText('checkpoint 20260527')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Revert' }))

    expect(onRevertFile).toHaveBeenCalledWith(expect.objectContaining({
      path: 'src/clean.ts',
      checkpointId: '20260527010101-abcdef12',
      checkpointFiles: ['src/preexisting.ts'],
    }))
  })

  it('calls backend-backed accept handlers before marking a change accepted', async () => {
    const onAcceptFile = vi.fn().mockResolvedValue(undefined)
    const diffs: DiffBlock[] = [
      { ...base, id: 'd-accept', path: 'src/accept.ts', diff: '@@ -1,1 +1,1 @@\n-before\n+after' },
    ]

    render(<CurrentTurnChangeCard diffs={diffs} onAcceptFile={onAcceptFile} />)

    fireEvent.click(screen.getByRole('button', { name: 'Accept' }))

    expect(screen.getByRole('button', { name: 'Accepting' })).toBeTruthy()
    await screen.findByText('accepted')
    expect(onAcceptFile).toHaveBeenCalledWith(expect.objectContaining({
      path: 'src/accept.ts',
      kind: 'modified',
    }))
  })

  it('renders post-edit verification checks with command metadata', () => {
    const diffs: DiffBlock[] = [
      { ...base, id: 'd-verified', path: 'src/verified.ts', diff: '@@ -1,1 +1,1 @@\n-old\n+new' },
    ]

    render(
      <CurrentTurnChangeCard
        diffs={diffs}
        verifications={[
          {
            id: 'verification-typecheck',
            toolName: 'bash',
            label: 'Typecheck',
            command: 'npm run typecheck',
            status: 'passed',
            exitCode: 0,
            durationMs: 1240,
            summary: 'tsc --noEmit passed',
          },
          {
            id: 'verification-tests',
            toolName: 'bash',
            label: 'Tests',
            command: 'npm test -- auth.test.ts',
            status: 'failed',
            exitCode: 1,
            durationMs: 61000,
            summary: '1 test failed',
          },
        ]}
      />,
    )

    expect(screen.getByRole('region', { name: 'Post-edit verification' })).toBeTruthy()
    expect(screen.getByText('Verification')).toBeTruthy()
    expect(screen.getByText('2 checks')).toBeTruthy()
    expect(screen.getByText('Typecheck')).toBeTruthy()
    expect(screen.getByText('npm run typecheck')).toBeTruthy()
    expect(screen.getByText('bash · exit 0 · 1.2s')).toBeTruthy()
    expect(screen.getByText('tsc --noEmit passed')).toBeTruthy()
    expect(screen.getByText('Tests')).toBeTruthy()
    expect(screen.getByText('npm test -- auth.test.ts')).toBeTruthy()
    expect(screen.getByText('bash · exit 1 · 1m 1s')).toBeTruthy()
    expect(screen.getByText('1 test failed')).toBeTruthy()
    expect(screen.getByText('1 failed')).toBeTruthy()
  })
})

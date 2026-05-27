// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { AskUserQuestionBlock } from './AskUserQuestionBlock'
import { DiagnosticsBlock } from './DiagnosticsBlock'
import { ErrorBlock } from './ErrorBlock'
import { SystemNoticeBlock } from './SystemNoticeBlock'

const base = {
  id: 'block-1',
  sessionId: 'session-1',
  runId: 'run-1',
  timestamp: 1,
  source: 'live' as const,
}

afterEach(cleanup)

describe('edge timeline blocks', () => {
  it('renders ask_user_question options, descriptions, and answers', () => {
    render(
      <AskUserQuestionBlock
        block={{
          ...base,
          kind: 'ask_user_question',
          state: 'answered',
          questions: [
            {
              id: 'mode',
              header: 'Mode',
              question: 'Which mode should the agent use?',
              options: [
                { label: 'Fast', description: 'Less detail' },
                { label: 'Careful', description: 'More verification' },
              ],
            },
          ],
          answers: { mode: 'Careful' },
        }}
      />,
    )

    expect(screen.getByText('Input requested')).toBeTruthy()
    expect(screen.getByText('Which mode should the agent use?')).toBeTruthy()
    expect(screen.getByText('More verification')).toBeTruthy()
    expect(screen.getAllByText('Careful')).toHaveLength(2)
    expect(screen.getByText('Selected answer')).toBeTruthy()
    expect(screen.getByText('selected')).toBeTruthy()
  })

  it('renders multi-select, free-text, and unmatched ask_user_question answers', () => {
    render(
      <AskUserQuestionBlock
        block={{
          ...base,
          kind: 'ask_user_question',
          state: 'answered',
          questions: [
            {
              id: 'checks',
              question: 'Which checks should run?',
              multiSelect: true,
              options: [
                { id: 'lint', label: 'Lint' },
                { id: 'tests', label: 'Tests' },
                { id: 'build', label: 'Build' },
              ],
            },
            {
              id: 'notes',
              question: 'Any extra notes?',
              freeText: true,
            },
          ],
          answers: { checks: 'Lint, Tests', notes: 'Run the slow suite too.', legacy: 'kept' },
        }}
      />,
    )

    expect(screen.getByText('multi-select')).toBeTruthy()
    expect(screen.getByText('free text')).toBeTruthy()
    expect(screen.getAllByText('selected')).toHaveLength(2)
    expect(screen.getByText('Run the slow suite too.')).toBeTruthy()
    expect(screen.getByText('Additional answers')).toBeTruthy()
    expect(screen.getByText('legacy')).toBeTruthy()
    expect(screen.getByText('kept')).toBeTruthy()
  })

  it('renders diagnostics bundles without exposing raw XML', () => {
    render(
      <DiagnosticsBlock
        block={{
          ...base,
          kind: 'diagnostics',
          content: '<diagnostics>raw</diagnostics>',
          files: [
            {
              path: 'src/app.py',
              diagnostics: [
                { severity: 'error', line: 4, column: 8, source: 'pyright', code: 'reportGeneralTypeIssues', message: 'bad type' },
                { severity: 'warning', line: 9, column: 1, source: 'ruff', message: 'unused import' },
              ],
            },
          ],
        }}
      />,
    )

    expect(screen.getByLabelText('Diagnostics')).toBeTruthy()
    expect(screen.getByText('2 issues after recent edits')).toBeTruthy()
    expect(screen.getByText('src/app.py')).toBeTruthy()
    expect(screen.getByText('4:8')).toBeTruthy()
    expect(screen.getByText('pyright [reportGeneralTypeIssues]')).toBeTruthy()
    expect(screen.getByText('bad type')).toBeTruthy()
    expect(screen.queryByText('<diagnostics>raw</diagnostics>')).toBeNull()
  })

  it('renders rich system notices with a stable header', () => {
    render(
      <SystemNoticeBlock
        block={{
          ...base,
          kind: 'system_notice',
          content: '# Session updated\n\nPlan mode enabled.',
        }}
      />,
    )

    expect(screen.getByText('System notice')).toBeTruthy()
    expect(screen.getByRole('heading', { name: 'Session updated' })).toBeTruthy()
  })

  it('renders error metadata and message', () => {
    render(
      <ErrorBlock
        block={{
          ...base,
          kind: 'error',
          code: 'web_slash_command',
          message: 'Command failed',
          details: [
            { label: 'HTTP status', value: '404' },
            { label: 'Endpoint', value: 'https://provider.test/v1/chat/completions' },
          ],
          suggestions: ['Check the provider base URL.'],
        }}
      />,
    )

    expect(screen.getByRole('alert')).toBeTruthy()
    expect(screen.getByText('web_slash_command')).toBeTruthy()
    expect(screen.getByText('Command failed')).toBeTruthy()
    expect(screen.getByLabelText('Error diagnostics')).toBeTruthy()
    expect(screen.getByText('HTTP status')).toBeTruthy()
    expect(screen.getByText('https://provider.test/v1/chat/completions')).toBeTruthy()
    expect(screen.getByLabelText('Suggested fixes')).toBeTruthy()
    expect(screen.getByText('Check the provider base URL.')).toBeTruthy()
  })
})

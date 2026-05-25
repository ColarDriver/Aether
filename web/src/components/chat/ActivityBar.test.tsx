// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { ActivityBar } from './ActivityBar'

afterEach(cleanup)

describe('ActivityBar', () => {
  it('stays hidden without an active run', () => {
    render(<ActivityBar activeRunId={null} />)

    expect(screen.queryByRole('status', { name: 'Current activity' })).toBeNull()
  })

  it('renders a shimmered active run summary from run state and token usage', () => {
    render(
      <ActivityBar
        activeRunId="run-1"
        sessionId="session-abcdef"
        model="gpt-5.4"
        status={{ runId: 'run-1', sessionId: 'session-abcdef', state: 'responding', detail: 'writing final answer' }}
        tokens={{ output_tokens: 1400, cache_read_tokens: 200 }}
      />,
    )

    const status = screen.getByRole('status', { name: 'Current activity' })
    expect(status).toBeTruthy()
    expect(screen.getByText('Responding').className).toContain('aether-shimmer-text')
    expect(screen.getByText('writing final answer')).toBeTruthy()
    expect(screen.getByText('1.6k tokens (1.4k out / 200 cache) · session- · gpt-5.4')).toBeTruthy()
  })

  it('suppresses raw engine enum details', () => {
    render(
      <ActivityBar
        activeRunId="run-2"
        status={{ runId: 'run-2', sessionId: 'session-1', state: 'tool_use', detail: 'LLM_CALL' }}
      />,
    )

    expect(screen.getByText('Running tool')).toBeTruthy()
    expect(screen.queryByText('LLM_CALL')).toBeNull()
  })
})

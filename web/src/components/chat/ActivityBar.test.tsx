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
    // Down arrow (output streaming) is rendered as its own accented span, so
    // the token text and trailing meta sit in sibling nodes.
    expect(screen.getByText('↓')).toBeTruthy()
    expect(screen.getByText('1.6k tokens (1.4k out / 200 cache)')).toBeTruthy()
    expect(screen.getByText('session- · gpt-5.4')).toBeTruthy()
  })

  it('labels the requesting (model-wait) phase', () => {
    render(
      <ActivityBar
        activeRunId="run-r"
        status={{ runId: 'run-r', sessionId: 'session-1', state: 'requesting' }}
      />,
    )

    expect(screen.getByText('Requesting')).toBeTruthy()
  })

  it('labels the tool-input (streaming tool args) phase', () => {
    render(
      <ActivityBar
        activeRunId="run-t"
        status={{ runId: 'run-t', sessionId: 'session-1', state: 'tool_input' }}
      />,
    )

    expect(screen.getByText('Tool input')).toBeTruthy()
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

  it('includes run details in verbose mode', () => {
    render(
      <ActivityBar
        activeRunId="run-abcdef123"
        verbose
        status={{ runId: 'run-abcdef123', sessionId: 'session-1', state: 'responding', elapsedMs: 1200 }}
      />,
    )

    expect(screen.getByText('run run-abcd · 1.20s · responding')).toBeTruthy()
  })
})

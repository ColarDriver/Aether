// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../../api/client'
import { useToastStore } from '../../stores/toastStore'
import { AnalyticsView } from './AnalyticsView'

const report = {
  days: 30,
  summary: {
    session_count: 2,
    message_count: 8,
    assistant_message_count: 3,
    tool_call_count: 1,
    usage: {
      input_tokens: 1000,
      output_tokens: 500,
      cache_read_tokens: 0,
      cache_write_tokens: 0,
      reasoning_tokens: 0,
      total_tokens: 1500,
    },
  },
  daily: [
    {
      day: '2026-05-20',
      sessions: 2,
      messages: 8,
      tool_calls: 1,
      usage: {
        input_tokens: 1000,
        output_tokens: 500,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        reasoning_tokens: 0,
        total_tokens: 1500,
      },
    },
  ],
  models: [
    {
      provider: 'codex',
      model: 'gpt-5.4',
      sessions: 2,
      messages: 8,
      tool_calls: 1,
      usage: {
        input_tokens: 1000,
        output_tokens: 500,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        reasoning_tokens: 0,
        total_tokens: 1500,
      },
    },
  ],
  top_sessions: [
    {
      session_id: 'session-1',
      summary: 'Build web console',
      provider: 'codex',
      model: 'gpt-5.4',
      updated_at: 1,
      messages: 8,
      tool_calls: 1,
      usage: {
        input_tokens: 1000,
        output_tokens: 500,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        reasoning_tokens: 0,
        total_tokens: 1500,
      },
    },
  ],
}

afterEach(() => {
  cleanup()
  useToastStore.getState().clear()
  vi.restoreAllMocks()
})

describe('AnalyticsView', () => {
  it('renders usage summary, model table, and top sessions', async () => {
    const analytics = vi.spyOn(api, 'analytics').mockResolvedValue(report)

    render(<AnalyticsView />)

    expect(await screen.findByText('Analytics')).toBeTruthy()
    expect(screen.getAllByText('1.5K').length).toBeGreaterThan(1)
    expect(screen.getAllByText('codex / gpt-5.4').length).toBeGreaterThan(1)
    expect(screen.getByText('Build web console')).toBeTruthy()

    fireEvent.change(screen.getByLabelText('Analytics period'), { target: { value: '7' } })

    await waitFor(() => expect(analytics).toHaveBeenCalledWith({ days: 7, limit: 20 }))
  })
})

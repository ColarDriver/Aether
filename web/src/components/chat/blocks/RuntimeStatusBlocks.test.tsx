// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { StreamingStatusBlock } from './StreamingStatusBlock'
import { TaskNotificationBlock } from './TaskNotificationBlock'
import { ThinkingBlock } from './ThinkingBlock'

const base = {
  id: 'block-1',
  sessionId: 'session-1',
  runId: 'run-1',
  timestamp: 1,
  source: 'live' as const,
}

afterEach(cleanup)

describe('runtime status blocks', () => {
  it('renders active thinking as an expandable activity row', () => {
    render(
      <ThinkingBlock
        block={{
          ...base,
          kind: 'thinking',
          content: 'Need to inspect the renderer before editing.',
          isActive: true,
        }}
      />,
    )

    expect(screen.getByText('thinking')).toBeTruthy()
    expect(screen.getByText('active')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: /thinking/ }))
    expect(screen.getByText('Need to inspect the renderer before editing.')).toBeTruthy()
  })

  it('renders streaming state with status detail and token usage', () => {
    render(
      <StreamingStatusBlock
        block={{
          ...base,
          kind: 'streaming_status',
          state: 'responding',
          detail: 'writing final answer',
          tokens: { output_tokens: 1200 },
        }}
      />,
    )

    expect(screen.getByRole('status')).toBeTruthy()
    expect(screen.getByText('Responding')).toBeTruthy()
    expect(screen.getByText('writing final answer')).toBeTruthy()
    expect(screen.getByText('1,200 out')).toBeTruthy()
  })

  it('renders task notification metadata and opens task details', () => {
    const onOpenTask = vi.fn()
    render(
      <TaskNotificationBlock
        block={{
          ...base,
          kind: 'task_notification',
          taskId: 'task-1',
          status: 'completed',
          subagentType: 'worker',
          durationSeconds: 2.4,
          summary: 'Updated the renderer.',
          outputFile: 'tasks/task-1.md',
        }}
        onOpenTask={onOpenTask}
      />,
    )

    expect(screen.getByText('Subagent completed')).toBeTruthy()
    expect(screen.getByText('worker')).toBeTruthy()
    expect(screen.getByText('2.4s')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Open task details' }))
    expect(onOpenTask).toHaveBeenCalledWith('task-1')
  })
})

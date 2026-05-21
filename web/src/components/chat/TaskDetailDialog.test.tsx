// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../../api/client'
import type { TaskSummary } from '../../api/types'
import { TASK_DETAIL_REFRESH_MS, TaskDetailDialog } from './TaskDetailDialog'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

const task: TaskSummary = {
  task_id: 'task-1',
  parent_session_id: 'session-1',
  subagent_type: 'explorer',
  prompt: 'Inspect renderer',
  status: 'completed',
  started_at: 1700000000,
  finished_at: 1700000005,
  last_heartbeat: 1700000005,
  model: 'gpt-5.4',
  isolation: null,
  worktree_path: '/tmp/worktree',
  parent_task_id: null,
  child_depth: 1,
  background: true,
  tool_use_count: 3,
  input_tokens: 120,
  output_tokens: 30,
  iterations: 2,
  summary: 'Renderer inspected',
  error: null,
  result_path: '/tmp/result.json',
  output_tail: 'done\n',
  metadata: { agent_type: 'explorer' },
}

describe('TaskDetailDialog', () => {
  it('loads task detail and renders output tail metadata', async () => {
    vi.spyOn(api, 'taskDetail').mockResolvedValue(task)

    render(<TaskDetailDialog taskId="task-1" onClose={() => undefined} />)

    expect(screen.getByText('Loading task details...')).toBeTruthy()
    await waitFor(() => expect(screen.getByText('Inspect renderer')).toBeTruthy())
    expect(screen.getByText('Renderer inspected')).toBeTruthy()
    expect(screen.getByText('Output tail')).toBeTruthy()
    expect(screen.getByText('done')).toBeTruthy()
    expect(screen.getByText('Metadata')).toBeTruthy()
    expect(api.taskDetail).toHaveBeenCalledWith('task-1')
  })

  it('shows a readable load error', async () => {
    vi.spyOn(api, 'taskDetail').mockRejectedValue(new Error('Task not found'))

    render(<TaskDetailDialog taskId="missing" onClose={() => undefined} />)

    await waitFor(() => expect(screen.getByText('Task not found')).toBeTruthy())
  })

  it('refreshes task detail on demand', async () => {
    vi.spyOn(api, 'taskDetail')
      .mockResolvedValueOnce(task)
      .mockResolvedValueOnce({ ...task, output_tail: 'new output\n', output_tokens: 50 })

    render(<TaskDetailDialog taskId="task-1" onClose={() => undefined} />)

    await waitFor(() => expect(screen.getByText('done')).toBeTruthy())
    fireEvent.click(screen.getByRole('button', { name: 'Refresh task details' }))

    await waitFor(() => expect(screen.getByText('new output')).toBeTruthy())
    expect(api.taskDetail).toHaveBeenCalledTimes(2)
  })

  it('polls active tasks and stops after they reach a terminal state', async () => {
    vi.useFakeTimers()
    const running = { ...task, status: 'running', finished_at: null, output_tail: 'running output\n' }
    const completed = { ...task, output_tail: 'final output\n' }
    vi.spyOn(api, 'taskDetail')
      .mockResolvedValueOnce(running)
      .mockResolvedValueOnce(completed)

    render(<TaskDetailDialog taskId="task-1" onClose={() => undefined} />)

    await act(async () => {
      await Promise.resolve()
    })
    expect(screen.getByText('running output')).toBeTruthy()

    await act(async () => {
      vi.advanceTimersByTime(TASK_DETAIL_REFRESH_MS)
      await Promise.resolve()
    })

    expect(screen.getByText('final output')).toBeTruthy()
    expect(api.taskDetail).toHaveBeenCalledTimes(2)
  })
})

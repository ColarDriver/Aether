// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../../api/client'
import type { TaskSummary } from '../../api/types'
import { TaskDetailDialog } from './TaskDetailDialog'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
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
})

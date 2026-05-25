// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { TaskSummary } from '../../api/types'
import { isTaskTerminal, SessionTaskBar, sortTasksForSessionBar } from './SessionTaskBar'

afterEach(cleanup)

const baseTask: TaskSummary = {
  task_id: 'task-1',
  parent_session_id: 'session-1',
  subagent_type: 'explorer',
  prompt: 'Inspect renderer',
  status: 'running',
  started_at: 10,
  finished_at: null,
  last_heartbeat: 11,
  model: 'gpt-5.4',
  isolation: null,
  worktree_path: null,
  parent_task_id: null,
  child_depth: 1,
  background: true,
  tool_use_count: 2,
  input_tokens: 100,
  output_tokens: 25,
  iterations: 1,
  summary: null,
  error: null,
  result_path: null,
  output_tail: null,
  metadata: null,
}

describe('SessionTaskBar', () => {
  it('renders active and completed subagent tasks with progress', () => {
    render(
      <SessionTaskBar
        tasks={[
          baseTask,
          { ...baseTask, task_id: 'task-2', prompt: 'Patch files', status: 'completed', started_at: 8, summary: 'patched' },
        ]}
      />,
    )

    expect(screen.getByRole('region', { name: 'Session tasks' })).toBeTruthy()
    expect(screen.getByText('1/2 · 1 active')).toBeTruthy()
    expect(screen.getByText('Inspect renderer')).toBeTruthy()
    expect(screen.getByText('running').className).toContain('aether-shimmer-text')
    expect(screen.getByText('Patch files')).toBeTruthy()
    expect(screen.getByText('patched')).toBeTruthy()
  })

  it('collapses and expands terminal task history', () => {
    render(<SessionTaskBar tasks={[{ ...baseTask, status: 'completed', summary: 'done' }]} />)

    expect(screen.queryByText('Inspect renderer')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /Subagent tasks/ }))
    expect(screen.getByText('Inspect renderer')).toBeTruthy()
  })

  it('keeps child tasks grouped under their parent task', () => {
    const parent = { ...baseTask, task_id: 'parent', prompt: 'Parent task', started_at: 10, child_depth: 1, parent_task_id: null }
    const newerSibling = { ...baseTask, task_id: 'sibling', prompt: 'Sibling task', started_at: 12, child_depth: 1, parent_task_id: null }
    const child = { ...baseTask, task_id: 'child', prompt: 'Child task', started_at: 20, child_depth: 2, parent_task_id: 'parent' }

    expect(sortTasksForSessionBar([child, parent, newerSibling]).map((task) => task.task_id)).toEqual(['sibling', 'parent', 'child'])
  })

  it('classifies terminal statuses', () => {
    expect(isTaskTerminal({ status: 'completed' })).toBe(true)
    expect(isTaskTerminal({ status: 'failed' })).toBe(true)
    expect(isTaskTerminal({ status: 'running' })).toBe(false)
  })

  it('opens a task detail request from a task row', () => {
    const onOpenTask = vi.fn()
    render(<SessionTaskBar onOpenTask={onOpenTask} tasks={[baseTask]} />)

    fireEvent.click(screen.getByRole('button', { name: 'Open task task-1' }))

    expect(onOpenTask).toHaveBeenCalledWith(baseTask)
  })
})

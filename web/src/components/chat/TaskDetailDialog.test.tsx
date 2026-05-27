// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../../api/client'
import type { TaskSummary } from '../../api/types'
import { TASK_DETAIL_REFRESH_MS, TaskDetailDialog } from './TaskDetailDialog'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

beforeEach(() => {
  vi.spyOn(api, 'taskMessages').mockResolvedValue({ task_id: 'task-1', messages: [], pending_messages: [], delivered_messages: [], total_count: 0, truncated: false })
  vi.spyOn(api, 'taskChildMessages').mockResolvedValue({ task_id: 'task-1', streams: [], total_count: 0, truncated: false })
  vi.spyOn(api, 'taskResult').mockResolvedValue({ task_id: 'task-1', result_path: '/tmp/result.json', result: { status: 'completed', summary: 'Renderer inspected' } })
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
  metadata: { agent_type: 'explorer', provider: 'openai-compatible' },
}

describe('TaskDetailDialog', () => {
  it('loads task detail and renders output tail metadata', async () => {
    vi.spyOn(api, 'taskDetail').mockResolvedValue(task)

    render(<TaskDetailDialog taskId="task-1" onClose={() => undefined} />)

    expect(screen.getByText('Loading task details...')).toBeTruthy()
    await waitFor(() => expect(screen.getByText('Inspect renderer')).toBeTruthy())
    expect(screen.getByText('Renderer inspected')).toBeTruthy()
    expect(screen.getByText('openai-compatible')).toBeTruthy()
    expect(screen.getByText('5s')).toBeTruthy()
    expect(screen.getByText('Output tail')).toBeTruthy()
    expect(screen.getByText('done')).toBeTruthy()
    expect(screen.getByText('Metadata')).toBeTruthy()
    expect(api.taskDetail).toHaveBeenCalledWith('task-1')
    expect(api.taskMessages).toHaveBeenCalledWith('task-1', { limit: 100 })
    expect(api.taskChildMessages).toHaveBeenCalledWith('task-1', { limit: 50, perTaskLimit: 25 })
    expect(api.taskResult).toHaveBeenCalledWith('task-1')
    const artifact = screen.getByRole('region', { name: 'Task result artifact' })
    expect(within(artifact).getByText('Result artifact')).toBeTruthy()
    expect(within(artifact).getByText('/tmp/result.json')).toBeTruthy()
    expect(within(artifact).getByRole('button', { name: 'Copy task result path' })).toBeTruthy()
    expect(within(artifact).getByRole('button', { name: 'Copy task result JSON' })).toBeTruthy()
    expect(within(artifact).getAllByText('result.json').length).toBeGreaterThan(0)
  })

  it('renders explicit task result artifact links and local path copy actions', async () => {
    vi.spyOn(api, 'taskDetail').mockResolvedValue(task)
    vi.mocked(api.taskResult).mockResolvedValue({
      task_id: 'task-1',
      result_path: '/tmp/result.json',
      result: {
        artifacts: [
          { name: 'bundle.zip', path: '/tmp/aether/bundle.zip', download_url: 'https://example.com/bundle.zip', mime_type: 'application/zip', size_bytes: 1000 },
        ],
      },
    })

    render(<TaskDetailDialog taskId="task-1" onClose={() => undefined} />)

    await waitFor(() => expect(screen.getByLabelText('Task result files')).toBeTruthy())
    const files = screen.getByLabelText('Task result files')
    expect(within(files).getByText('bundle.zip')).toBeTruthy()
    expect(within(files).getByText((text) => text.includes('/tmp/aether/bundle.zip'))).toBeTruthy()
    expect(within(files).getByRole('link', { name: /Open/ }).getAttribute('href')).toBe('https://example.com/bundle.zip')
    expect(within(files).getByRole('button', { name: 'Copy bundle.zip path' })).toBeTruthy()
  })

  it('renders task observer messages from the subagent message stream', async () => {
    vi.spyOn(api, 'taskDetail').mockResolvedValue(task)
    vi.mocked(api.taskMessages).mockResolvedValue({
      task_id: 'task-1',
      total_count: 2,
      truncated: false,
      pending_messages: [],
      delivered_messages: [],
      messages: [
        { index: 0, role: 'assistant', content: 'I inspected the renderer.', iteration: 1 },
        { index: 1, role: 'tool', name: 'read_file', tool_call_id: 'call-1', content: 'file contents', elapsed_ms: 25 },
      ],
    })

    render(<TaskDetailDialog taskId="task-1" onClose={() => undefined} />)

    await waitFor(() => expect(screen.getByRole('region', { name: 'Task message stream' })).toBeTruthy())
    expect(screen.getByText('assistant')).toBeTruthy()
    expect(screen.getByText('I inspected the renderer.')).toBeTruthy()
    expect(screen.getByText('read_file')).toBeTruthy()
    expect(screen.getByText('file contents')).toBeTruthy()
    expect(screen.getByText(/25ms/)).toBeTruthy()
  })

  it('renders queued parent-to-subagent messages in task details', async () => {
    vi.spyOn(api, 'taskDetail').mockResolvedValue(task)
    vi.mocked(api.taskMessages).mockResolvedValue({
      task_id: 'task-1',
      total_count: 0,
      truncated: false,
      messages: [],
      delivered_messages: [],
      pending_messages: [{ index: 0, message: 'please also inspect auth.ts', ts: 1700000001 }],
    })

    render(<TaskDetailDialog taskId="task-1" onClose={() => undefined} />)

    await waitFor(() => expect(screen.getByRole('region', { name: 'Task message stream' })).toBeTruthy())
    expect(screen.getByLabelText('Queued parent messages')).toBeTruthy()
    expect(screen.getByText('parent message')).toBeTruthy()
    expect(screen.getByText('please also inspect auth.ts')).toBeTruthy()
  })

  it('does not request a result artifact when the task has no result path', async () => {
    vi.spyOn(api, 'taskDetail').mockResolvedValue({ ...task, result_path: null })

    render(<TaskDetailDialog taskId="task-1" onClose={() => undefined} />)

    await waitFor(() => expect(screen.getByText('Inspect renderer')).toBeTruthy())
    expect(api.taskResult).not.toHaveBeenCalled()
  })

  it('renders a readable result artifact load error', async () => {
    vi.spyOn(api, 'taskDetail').mockResolvedValue(task)
    vi.mocked(api.taskResult).mockRejectedValue(new Error('Task result not found'))

    render(<TaskDetailDialog taskId="task-1" onClose={() => undefined} />)

    await waitFor(() => expect(screen.getByText('Task result not found')).toBeTruthy())
  })

  it('renders delivered parent-to-subagent message history in task details', async () => {
    vi.spyOn(api, 'taskDetail').mockResolvedValue(task)
    vi.mocked(api.taskMessages).mockResolvedValue({
      task_id: 'task-1',
      total_count: 0,
      truncated: false,
      messages: [],
      pending_messages: [],
      delivered_messages: [{ index: 0, message: 'already delivered context', ts: 1700000001, delivered_at: 1700000002 }],
    })

    render(<TaskDetailDialog taskId="task-1" onClose={() => undefined} />)

    await waitFor(() => expect(screen.getByRole('region', { name: 'Task message stream' })).toBeTruthy())
    expect(screen.getByLabelText('Delivered parent messages')).toBeTruthy()
    expect(screen.getByText('delivered parent message')).toBeTruthy()
    expect(screen.getByText('already delivered context')).toBeTruthy()
    expect(screen.getAllByText(/delivered/).length).toBeGreaterThan(0)
  })

  it('renders descendant child task message streams', async () => {
    const onOpenTask = vi.fn()
    vi.spyOn(api, 'taskDetail').mockResolvedValue(task)
    vi.mocked(api.taskChildMessages).mockResolvedValue({
      task_id: 'task-1',
      total_count: 1,
      truncated: false,
      streams: [
        {
          task: { ...task, task_id: 'child-task', prompt: 'Child verification', status: 'completed', parent_task_id: 'task-1', child_depth: 2 },
          messages: [{ index: 0, role: 'assistant', content: 'child checked renderer', iteration: 1 }],
          pending_messages: [],
          delivered_messages: [{ index: 0, message: 'child context delivered', delivered_at: 1700000002 }],
          total_count: 1,
          truncated: false,
        },
      ],
    })

    render(<TaskDetailDialog taskId="task-1" onClose={() => undefined} onOpenTask={onOpenTask} />)

    await waitFor(() => expect(screen.getByRole('region', { name: 'Child task message streams' })).toBeTruthy())
    expect(screen.getByText('Child verification')).toBeTruthy()
    expect(screen.getByText('child checked renderer')).toBeTruthy()
    expect(screen.getByText('child context delivered')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: /Active/ }))
    expect(screen.getByText('No child streams match this filter.')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: /Finished/ }))
    expect(screen.getByText('child checked renderer')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Open child task' }))
    expect(onOpenTask).toHaveBeenCalledWith('child-task')
    const toggle = screen.getByRole('button', { name: /Child verification/ })
    expect(toggle.getAttribute('aria-expanded')).toBe('true')
    fireEvent.click(toggle)
    expect(toggle.getAttribute('aria-expanded')).toBe('false')
    expect(screen.queryByText('child checked renderer')).toBeNull()
  })

  it('renders related parent and child tasks with drill-down actions', async () => {
    const onOpenTask = vi.fn()
    const parent = { ...task, task_id: 'parent-task', prompt: 'Parent investigation', child_depth: 1, parent_task_id: null, started_at: 1 }
    const current = { ...task, task_id: 'task-1', prompt: 'Current task', child_depth: 2, parent_task_id: 'parent-task', started_at: 2 }
    const child = { ...task, task_id: 'child-task', prompt: 'Child verification', child_depth: 3, parent_task_id: 'task-1', started_at: 3, summary: 'verified' }
    vi.spyOn(api, 'taskDetail').mockResolvedValue(current)

    render(
      <TaskDetailDialog
        taskId="task-1"
        initialTask={current}
        sessionTasks={[parent, current, child]}
        onOpenTask={onOpenTask}
        onClose={() => undefined}
      />,
    )

    await waitFor(() => expect(screen.getByRole('region', { name: 'Related tasks' })).toBeTruthy())
    expect(screen.getByText('Parent investigation')).toBeTruthy()
    expect(screen.getAllByText('Current task').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('Child verification')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Open related task child-task' }))
    expect(onOpenTask).toHaveBeenCalledWith('child-task')
  })

  it('shows a readable load error', async () => {
    vi.spyOn(api, 'taskDetail').mockRejectedValue(new Error('Task not found'))

    render(<TaskDetailDialog taskId="missing" onClose={() => undefined} />)

    await waitFor(() => expect(screen.getByText('Task not found')).toBeTruthy())
  })

  it('sends a stop signal for active tasks and refreshes detail', async () => {
    const runningTask = { ...task, status: 'running', finished_at: null }
    vi.spyOn(api, 'taskDetail')
      .mockResolvedValueOnce(runningTask)
      .mockResolvedValueOnce({ ...runningTask, status: 'interrupted', finished_at: 1700000006 })
    vi.spyOn(api, 'stopTask').mockResolvedValue({
      task_id: 'task-1',
      delivered: true,
      status: 'running',
      message: 'Stop signal sent to running task.',
    })

    render(<TaskDetailDialog taskId="task-1" onClose={() => undefined} />)

    await waitFor(() => expect(screen.getByRole('button', { name: 'Stop task' })).toBeTruthy())
    fireEvent.click(screen.getByRole('button', { name: 'Stop task' }))

    await waitFor(() => expect(api.stopTask).toHaveBeenCalledWith('task-1'))
    expect(await screen.findByText('Stop signal sent to running task.')).toBeTruthy()
    await waitFor(() => expect(screen.getByText('interrupted')).toBeTruthy())
  })

  it('queues a follow-up message for running tasks and refreshes message history', async () => {
    const runningTask = { ...task, status: 'running', finished_at: null }
    vi.spyOn(api, 'taskDetail').mockResolvedValue(runningTask)
    vi.spyOn(api, 'sendTaskMessage').mockResolvedValue({
      task_id: 'task-1',
      queued: true,
      status: 'running',
      message: "Queued message for the subagent's next iteration boundary.",
      queued_chars: 18,
    })
    vi.mocked(api.taskMessages)
      .mockResolvedValueOnce({ task_id: 'task-1', messages: [], pending_messages: [], delivered_messages: [], total_count: 0, truncated: false })
      .mockResolvedValueOnce({
        task_id: 'task-1',
        messages: [],
        pending_messages: [{ index: 0, message: 'please inspect auth', ts: 1700000001 }],
        delivered_messages: [],
        total_count: 0,
        truncated: false,
      })

    render(<TaskDetailDialog taskId="task-1" onClose={() => undefined} />)

    const input = await screen.findByLabelText('Task follow-up message')
    fireEvent.change(input, { target: { value: 'please inspect auth' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send task message' }))

    await waitFor(() => expect(api.sendTaskMessage).toHaveBeenCalledWith('task-1', { message: 'please inspect auth' }))
    expect(await screen.findByText("Queued message for the subagent's next iteration boundary.")).toBeTruthy()
    expect(await screen.findByText('please inspect auth')).toBeTruthy()
    expect((input as HTMLTextAreaElement).value).toBe('')
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

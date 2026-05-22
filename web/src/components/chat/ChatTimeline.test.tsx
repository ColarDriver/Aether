// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ChatBlock } from '../../chat-rendering'
import { ChatTimeline } from './ChatTimeline'

const base = {
  sessionId: 's1',
  runId: 'r1',
  timestamp: 1,
  source: 'live' as const,
}

afterEach(cleanup)

describe('ChatTimeline', () => {
  it('renders user, assistant, thinking, tool, result, and diff blocks in one timeline', () => {
    const blocks: ChatBlock[] = [
      { ...base, id: 'u', kind: 'user_message', content: 'hello', attachments: [{ type: 'file', name: 'app.ts', path: 'src/app.ts' }] },
      { ...base, id: 'a', kind: 'assistant_message', content: 'I will read.' },
      { ...base, id: 't', kind: 'thinking', content: 'Need inspect', isActive: true },
      { ...base, id: 'tn', kind: 'task_notification', taskId: 'task-1', subagentType: 'explorer', status: 'completed', summary: 'Task done' },
      { ...base, id: 'tc', kind: 'tool_call', toolCallId: 'call-1', toolName: 'read_file', arguments: { path: 'README.md' }, status: 'finished' },
      { ...base, id: 'tr', kind: 'tool_result', toolCallId: 'call-1', toolName: 'read_file', content: 'contents', isError: false, metadata: {} },
    ]

    render(<ChatTimeline blocks={blocks} />)

    expect(screen.getByText('hello')).toBeTruthy()
    expect(screen.getByText('app.ts')).toBeTruthy()
    expect(screen.getByText('I will read.')).toBeTruthy()
    expect(screen.getByText(/thinking/)).toBeTruthy()
    expect(screen.getByText('Subagent completed')).toBeTruthy()
    expect(screen.getByText('Task done')).toBeTruthy()
    expect(screen.getByText('read_file')).toBeTruthy()
    expect(screen.getByText('contents')).toBeTruthy()
  })

  it('renders prompt blocks with actions', () => {
    const onPermission = vi.fn()
    const onApproval = vi.fn()
    const blocks: ChatBlock[] = [
      { ...base, id: 'p', kind: 'permission_request', promptId: 'p1', toolName: 'write_file', arguments: {}, allowSession: true, state: 'pending' },
      { ...base, id: 'ap', kind: 'approval_request', promptId: 'a1', approvalKind: 'plan', planText: '# Plan', questions: [], state: 'pending' },
    ]

    render(<ChatTimeline blocks={blocks} onRespondPermission={onPermission} onRespondApproval={onApproval} />)

    fireEvent.click(screen.getByRole('button', { name: 'Allow once' }))
    fireEvent.click(screen.getByRole('button', { name: 'Approve' }))

    expect(onPermission).toHaveBeenCalledWith({ type: 'allow_once' })
    expect(onApproval).toHaveBeenCalledWith({ confirmed: true })
  })

  it('renders a full web chat turn across every primary block family', () => {
    const blocks: ChatBlock[] = [
      { ...base, id: 'u-full', kind: 'user_message', content: 'Implement auth', attachments: [{ type: 'text', name: 'requirements.md', path: 'docs/requirements.md' }] },
      { ...base, id: 'sys-full', kind: 'system_notice', content: '# Mode changed\n\nPlan mode enabled.' },
      { ...base, id: 'status-full', kind: 'streaming_status', state: 'responding', detail: 'Drafting implementation', tokens: { output_tokens: 42 } },
      { ...base, id: 'think-full', kind: 'thinking', content: 'Map routes and storage', isActive: true },
      { ...base, id: 'assistant-full', kind: 'assistant_message', content: 'I will inspect the auth flow.' },
      { ...base, id: 'tool-full', kind: 'tool_call', toolCallId: 'call-auth', toolName: 'read_file', arguments: { path: 'src/auth.ts' }, status: 'finished' },
      { ...base, id: 'result-full', kind: 'tool_result', toolCallId: 'call-auth', toolName: 'read_file', content: 'auth module contents', isError: false, metadata: {} },
      { ...base, id: 'diff-full', kind: 'diff', origin: 'transcript', path: 'src/auth.ts', diff: '@@ -1,1 +1,1 @@\n-old auth\n+new auth' },
      {
        ...base,
        id: 'perm-full',
        kind: 'permission_request',
        promptId: 'perm-auth',
        toolName: 'write_file',
        arguments: { path: 'src/auth.ts' },
        preview: { title: 'Modify auth file', subtitle: 'src/auth.ts' },
        state: 'pending',
      },
      {
        ...base,
        id: 'approval-full',
        kind: 'approval_request',
        promptId: 'approval-auth',
        approvalKind: 'plan',
        planText: '# Plan\n\n1. Add auth middleware.',
        planPath: '/tmp/aether/plans/auth.md',
        questions: [],
        state: 'pending',
      },
      {
        ...base,
        id: 'ask-full',
        kind: 'ask_user_question',
        promptId: 'ask-auth',
        state: 'answered',
        questions: [
          {
            id: 'strategy',
            header: 'Auth strategy',
            question: 'Which strategy should Aether use?',
            options: [{ label: 'Session cookie', description: 'Server-side state' }],
          },
        ],
        answers: { strategy: 'Session cookie' },
      },
      { ...base, id: 'task-full', kind: 'task_notification', taskId: 'task-auth', subagentType: 'explorer', status: 'completed', summary: 'Auth files mapped' },
      { ...base, id: 'error-full', kind: 'error', code: 'preview_warning', message: 'Preview failed but run continued.' },
    ]

    render(<ChatTimeline blocks={blocks} />)

    expect(screen.getByText('Implement auth')).toBeTruthy()
    expect(screen.getByText('requirements.md')).toBeTruthy()
    expect(screen.getByRole('heading', { name: 'Mode changed' })).toBeTruthy()
    expect(screen.getByText('Responding')).toBeTruthy()
    expect(screen.getByText('Drafting implementation')).toBeTruthy()
    expect(screen.getByText('I will inspect the auth flow.')).toBeTruthy()
    expect(screen.getByText('read_file')).toBeTruthy()
    expect(screen.getByText('auth module contents')).toBeTruthy()
    expect(screen.getAllByText('src/auth.ts').length).toBeGreaterThanOrEqual(2)
    expect(document.querySelector('.diff-line-remove')?.textContent).toContain('old auth')
    expect(document.querySelector('.diff-line-add')?.textContent).toContain('new auth')
    expect(screen.getByText('Modify auth file')).toBeTruthy()
    expect(screen.getByText('Plan approval')).toBeTruthy()
    expect(screen.getByText('Which strategy should Aether use?')).toBeTruthy()
    expect(screen.getAllByText('Session cookie')).toHaveLength(2)
    expect(screen.getByText('Subagent completed')).toBeTruthy()
    expect(screen.getByText('Preview failed but run continued.')).toBeTruthy()
  })

  it('renders a changed-files summary for turn diffs', () => {
    const blocks: ChatBlock[] = [
      { ...base, id: 'u-change', kind: 'user_message', content: 'Patch auth' },
      { ...base, id: 'tc-change', kind: 'tool_call', toolCallId: 'call-change', toolName: 'file_edit', arguments: { path: 'src/auth.ts' }, status: 'finished' },
      { ...base, id: 'tr-change', kind: 'tool_result', toolCallId: 'call-change', toolName: 'file_edit', content: 'edited', isError: false, metadata: {} },
      { ...base, id: 'diff-call-change', kind: 'diff', origin: 'tool_result', path: 'src/auth.ts', diff: '@@ -1,1 +1,2 @@\n-old auth\n+new auth\n+guard' },
    ]

    render(<ChatTimeline blocks={blocks} />)

    expect(screen.getByRole('region', { name: 'Changed files' })).toBeTruthy()
    expect(screen.getByRole('button', { name: /1 changed file/ })).toBeTruthy()
    expect(screen.getAllByText('src/auth.ts').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('+2').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('-1').length).toBeGreaterThanOrEqual(1)
  })
  it('groups multiple consecutive tool calls behind an activity summary', () => {
    const blocks: ChatBlock[] = [
      { ...base, id: 'tc1', kind: 'tool_call', toolCallId: 'call-1', toolName: 'read_file', arguments: { path: 'README.md' }, status: 'finished' },
      { ...base, id: 'tc2', kind: 'tool_call', toolCallId: 'call-2', toolName: 'grep', arguments: { pattern: 'TODO' }, status: 'finished' },
      { ...base, id: 'tr1', kind: 'tool_result', toolCallId: 'call-1', toolName: 'read_file', content: 'contents', isError: false, metadata: {} },
      { ...base, id: 'tr2', kind: 'tool_result', toolCallId: 'call-2', toolName: 'grep', content: 'matches', isError: false, metadata: {} },
    ]

    render(<ChatTimeline blocks={blocks} />)

    expect(screen.getByRole('button', { name: /Activity/ }).getAttribute('aria-expanded')).toBe('false')
    expect(screen.queryByText('read_file')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: /Activity/ }))

    expect(screen.getByText('read_file')).toBeTruthy()
    expect(screen.getByText('grep')).toBeTruthy()
    expect(screen.getByText('contents')).toBeTruthy()
  })

  it('only renders allow-session when the permission block allows it', () => {
    render(
      <ChatTimeline
        blocks={[
          { ...base, id: 'p', kind: 'permission_request', promptId: 'p1', toolName: 'shell', arguments: {}, state: 'pending' },
        ]}
        onRespondPermission={() => undefined}
      />,
    )

    expect(screen.queryByRole('button', { name: 'Allow session' })).toBeNull()
  })

  it('opens task details from task notification blocks', () => {
    const onOpenTask = vi.fn()
    render(
      <ChatTimeline
        blocks={[
          { ...base, id: 'tn', kind: 'task_notification', taskId: 'task-1', status: 'completed', summary: 'done' },
        ]}
        onOpenTask={onOpenTask}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Open task details' }))

    expect(onOpenTask).toHaveBeenCalledWith('task-1')
  })
})

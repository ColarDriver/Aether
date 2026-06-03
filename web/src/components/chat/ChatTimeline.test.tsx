// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
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

  it('renders Studio-ready message metadata with author and time', () => {
    const timestamp = Date.UTC(2026, 0, 1, 14, 36)
    const blocks: ChatBlock[] = [
      { ...base, timestamp, id: 'u-meta', kind: 'user_message', content: 'hello' },
      { ...base, timestamp, id: 'a-meta', kind: 'assistant_message', content: 'reply' },
    ]

    render(<ChatTimeline blocks={blocks} />)

    expect(screen.getByText('You')).toBeTruthy()
    expect(screen.getByText('Aether')).toBeTruthy()
    expect(screen.getAllByText('2:36PM').length).toBe(2)
  })

  it('exposes local message lifecycle actions for user and assistant messages', () => {
    const onRetry = vi.fn()
    const onEdit = vi.fn()
    const onQuoteAssistant = vi.fn()
    const userBlock: ChatBlock = { ...base, id: 'u-actions', kind: 'user_message', content: 'change auth' }
    const assistantBlock: ChatBlock = { ...base, id: 'a-actions', kind: 'assistant_message', content: 'Auth summary' }

    render(
      <ChatTimeline
        blocks={[userBlock, assistantBlock]}
        onRetryUserMessage={onRetry}
        onEditUserMessage={onEdit}
        onQuoteAssistantMessage={onQuoteAssistant}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Retry prompt' }))
    fireEvent.click(screen.getByRole('button', { name: 'Edit prompt' }))
    fireEvent.click(screen.getByRole('button', { name: 'Quote reply' }))

    expect(screen.queryByRole('button', { name: 'Quote prompt' })).toBeNull()
    expect(onRetry).toHaveBeenCalledWith(userBlock)
    expect(onEdit).toHaveBeenCalledWith(userBlock)
    expect(onQuoteAssistant).toHaveBeenCalledWith(assistantBlock)
  })

  it('keeps persisted user actions compact while exposing backend-backed assistant actions', () => {
    const onFork = vi.fn()
    const onRewind = vi.fn()
    const onRetryAssistant = vi.fn()
    const userBlock: ChatBlock = {
      ...base,
      id: 'u-fork',
      source: 'transcript',
      messageIndex: 2,
      kind: 'user_message',
      content: 'fork this prompt',
    }
    const assistantBlock: ChatBlock = {
      ...base,
      id: 'a-fork',
      source: 'transcript',
      messageIndex: 3,
      kind: 'assistant_message',
      content: 'fork this reply',
    }

    render(
      <ChatTimeline
        blocks={[userBlock, assistantBlock]}
        onForkMessage={onFork}
        onRetryAssistantMessage={onRetryAssistant}
        onRewindMessage={onRewind}
      />,
    )

    expect(screen.queryByRole('button', { name: 'Rewind to prompt' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Fork from prompt' })).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Retry reply' }))
    fireEvent.click(screen.getByRole('button', { name: 'Rewind to reply' }))
    fireEvent.click(screen.getByRole('button', { name: 'Fork from reply' }))

    expect(onFork).toHaveBeenCalledOnce()
    expect(onFork).toHaveBeenCalledWith(assistantBlock)
    expect(onRetryAssistant).toHaveBeenCalledWith(assistantBlock)
    expect(onRewind).toHaveBeenCalledOnce()
    expect(onRewind).toHaveBeenCalledWith(assistantBlock)
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
    const inlineStatus = document.querySelector('.chat-status-inline')
    expect(inlineStatus?.textContent).toContain('Aether')
    expect(inlineStatus?.querySelector('.chat-status-verb')?.className).toContain('aether-shimmer-text')
    expect(screen.getByText('I will inspect the auth flow.')).toBeTruthy()
    expect(screen.getByText('read_file')).toBeTruthy()
    const filePreview = screen.getByRole('region', { name: 'File preview' })
    expect(within(filePreview).getByText('src/auth.ts')).toBeTruthy()
    expect(filePreview.textContent).toContain('auth module contents')
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
      {
        ...base,
        id: 'diag-change',
        kind: 'diagnostics',
        content: '<diagnostics />',
        files: [{ path: '/workspace/Aether/src/auth.ts', diagnostics: [{ severity: 'error', line: 4, column: 8, source: 'pyright', message: 'bad auth type' }] }],
      },
    ]

    render(<ChatTimeline blocks={blocks} />)

    expect(screen.getByRole('region', { name: 'Changed files' })).toBeTruthy()
    expect(screen.getByRole('button', { name: /1 changed file/ })).toBeTruthy()
    expect(screen.getAllByText('src/auth.ts').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('+2').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('-1').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('1 error').length).toBeGreaterThanOrEqual(1)
  })

  it('supports accepting and reverting oldText-backed file changes', async () => {
    const onRevert = vi.fn().mockResolvedValue(undefined)
    const blocks: ChatBlock[] = [
      { ...base, id: 'u-change-actions', kind: 'user_message', content: 'Patch auth' },
      {
        ...base,
        id: 'diff-change-actions',
        kind: 'diff',
        origin: 'tool_result',
        path: 'src/auth.ts',
        oldText: 'old auth\n',
        newText: 'new auth\n',
      },
    ]

    render(<ChatTimeline blocks={blocks} onRevertFileChange={onRevert} />)

    fireEvent.click(screen.getByRole('button', { name: 'Accept' }))
    expect(screen.getByText('accepted')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Revert' }))
    expect(screen.getByRole('dialog', { name: 'Revert file change' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Revert file' }))

    await waitFor(() => expect(onRevert).toHaveBeenCalledWith(expect.objectContaining({
      path: 'src/auth.ts',
      kind: 'modified',
      oldText: 'old auth\n',
      newText: 'new auth\n',
    })))
    await waitFor(() => expect(screen.getByText('reverted')).toBeTruthy())
  })

  it('passes run checkpoint metadata to changed-file revert actions', async () => {
    const onRevert = vi.fn().mockResolvedValue(undefined)
    const blocks: ChatBlock[] = [
      { ...base, id: 'u-checkpoint-change', kind: 'user_message', content: 'Patch auth' },
      {
        ...base,
        id: 'diff-checkpoint-change',
        kind: 'diff',
        origin: 'tool_result',
        path: 'src/auth.ts',
        diff: '@@ -1,1 +1,1 @@\n-old\n+new',
      },
      {
        ...base,
        id: 'assistant-checkpoint-change',
        kind: 'assistant_message',
        content: 'Changed auth.',
        metadata: {
          workspace_checkpoint: {
            checkpoint_id: '20260527010101-abcdef12',
            label: 'Before web run',
            files: [{ path: 'src/preexisting.ts' }],
          },
        },
      },
    ]

    render(<ChatTimeline blocks={blocks} onRevertFileChange={onRevert} />)

    expect(screen.getByText('checkpoint 20260527')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Revert' }))
    expect(screen.getByRole('dialog', { name: 'Revert file change' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Revert file' }))

    await waitFor(() => expect(onRevert).toHaveBeenCalledWith(expect.objectContaining({
      path: 'src/auth.ts',
      checkpointId: '20260527010101-abcdef12',
      checkpointFiles: ['src/preexisting.ts'],
    })))
  })

  it('exposes checkpoint-backed undo actions for completed turns', () => {
    const onUndoTurn = vi.fn()
    const blocks: ChatBlock[] = [
      {
        ...base,
        id: 'u-undo-turn',
        source: 'transcript',
        messageIndex: 0,
        kind: 'user_message',
        content: 'Patch auth',
        attachments: [{ type: 'file', name: 'auth.ts', path: 'src/auth.ts' }],
      },
      { ...base, id: 'a-undo-turn', source: 'transcript', messageIndex: 1, kind: 'assistant_message', content: 'Changed auth.' },
    ]

    render(
      <ChatTimeline
        blocks={blocks}
        sessionId="session-1"
        turnCheckpoints={[{
          target: {
            target_user_message_id: 'turn-1',
            user_message_index: 0,
            user_message_count: 1,
            message_index: 0,
            content: 'Patch auth',
          },
          code: {
            available: true,
            files_changed: ['src/auth.ts'],
            insertions: 2,
            deletions: 1,
            checkpoint_id: 'cp-undo',
          },
        }]}
        onUndoTurn={onUndoTurn}
      />,
    )

    expect(screen.getByRole('region', { name: 'Changed files' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Undo turn' }))

    expect(onUndoTurn).toHaveBeenCalledWith(expect.objectContaining({
      promptContent: 'Patch auth',
      attachments: [{ type: 'file', name: 'auth.ts', path: 'src/auth.ts' }],
      checkpointId: 'cp-undo',
      paths: ['src/auth.ts'],
      body: {
        target_user_message_id: 'turn-1',
        user_message_index: 0,
        expected_content: 'Patch auth',
        checkpoint_id: 'cp-undo',
        paths: ['src/auth.ts'],
      },
    }))
  })

  it('attaches same-turn verification tool results to the changed-file summary', () => {
    const blocks: ChatBlock[] = [
      { ...base, id: 'u-verify-change', kind: 'user_message', content: 'Patch auth and verify' },
      { ...base, id: 'tc-edit-verify', kind: 'tool_call', toolCallId: 'call-edit-verify', toolName: 'file_edit', arguments: { path: 'src/auth.ts' }, status: 'finished' },
      { ...base, id: 'tr-edit-verify', kind: 'tool_result', toolCallId: 'call-edit-verify', toolName: 'file_edit', content: 'edited', isError: false, metadata: {} },
      { ...base, id: 'diff-call-edit-verify', kind: 'diff', origin: 'tool_result', path: 'src/auth.ts', diff: '@@ -1,1 +1,1 @@\n-old auth\n+new auth' },
      { ...base, id: 'tc-typecheck', kind: 'tool_call', toolCallId: 'call-typecheck', toolName: 'bash', arguments: { command: 'npm run typecheck' }, status: 'finished' },
      {
        ...base,
        id: 'tr-typecheck',
        kind: 'tool_result',
        toolCallId: 'call-typecheck',
        toolName: 'bash',
        content: 'tsc --noEmit -p tsconfig.json',
        isError: false,
        metadata: { exit_code: 0, duration_ms: 1520 },
      },
      { ...base, id: 'tc-test', kind: 'tool_call', toolCallId: 'call-test', toolName: 'shell', arguments: { cmd: 'npm test -- auth.test.ts' }, status: 'failed' },
      {
        ...base,
        id: 'tr-test',
        kind: 'tool_result',
        toolCallId: 'call-test',
        toolName: 'shell',
        content: '1 test failed',
        isError: true,
        metadata: { exit_code: 1, duration_ms: 33000 },
      },
    ]

    render(<ChatTimeline blocks={blocks} />)

    expect(screen.getByRole('region', { name: 'Changed files' })).toBeTruthy()
    const verification = within(screen.getByRole('region', { name: 'Post-edit verification' }))
    expect(verification.getByText('Typecheck')).toBeTruthy()
    expect(verification.getByText('npm run typecheck')).toBeTruthy()
    expect(verification.getByText('bash · exit 0 · 1.5s')).toBeTruthy()
    expect(verification.getByText('Tests')).toBeTruthy()
    expect(verification.getByText('npm test -- auth.test.ts')).toBeTruthy()
    expect(verification.getByText('shell · exit 1 · 33s')).toBeTruthy()
    expect(screen.getByText('1 failed')).toBeTruthy()
  })

  it('does not attach ordinary shell commands as post-edit verification', () => {
    const blocks: ChatBlock[] = [
      { ...base, id: 'u-nonverify-change', kind: 'user_message', content: 'Patch auth' },
      { ...base, id: 'tc-edit-nonverify', kind: 'tool_call', toolCallId: 'call-edit-nonverify', toolName: 'file_edit', arguments: { path: 'src/auth.ts' }, status: 'finished' },
      { ...base, id: 'tr-edit-nonverify', kind: 'tool_result', toolCallId: 'call-edit-nonverify', toolName: 'file_edit', content: 'edited', isError: false, metadata: {} },
      { ...base, id: 'diff-call-edit-nonverify', kind: 'diff', origin: 'tool_result', path: 'src/auth.ts', diff: '@@ -1,1 +1,1 @@\n-old auth\n+new auth' },
      { ...base, id: 'tc-ls', kind: 'tool_call', toolCallId: 'call-ls', toolName: 'bash', arguments: { command: 'ls src' }, status: 'finished' },
      { ...base, id: 'tr-ls', kind: 'tool_result', toolCallId: 'call-ls', toolName: 'bash', content: 'auth.ts', isError: false, metadata: { exit_code: 0 } },
    ]

    render(<ChatTimeline blocks={blocks} />)

    expect(screen.getByRole('region', { name: 'Changed files' })).toBeTruthy()
    expect(screen.queryByRole('region', { name: 'Post-edit verification' })).toBeNull()
    expect(screen.queryByText('Command check')).toBeNull()
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

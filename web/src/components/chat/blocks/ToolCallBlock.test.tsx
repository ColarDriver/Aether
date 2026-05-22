// @vitest-environment jsdom

import { cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import type { ToolCallBlock as ToolCall, ToolResultBlock as ToolResult } from '../../../chat-rendering'
import { ToolCallBlock } from './ToolCallBlock'
import { todosFromToolArguments } from './TodoListPreview'

afterEach(cleanup)

const base = {
  id: 'tool-call',
  sessionId: 'session-1',
  runId: 'run-1',
  timestamp: 1,
  source: 'live',
} as const

describe('ToolCallBlock todo rendering', () => {
  it('renders todo_write arguments as a checklist instead of raw JSON', () => {
    const block: ToolCall = {
      ...base,
      kind: 'tool_call',
      toolCallId: 'tc1',
      toolName: 'todo_write',
      status: 'finished',
      arguments: {
        todos: [
          { id: '1', content: 'Inspect web renderer', status: 'completed' },
          { id: '2', content: 'Wire composer references', status: 'in_progress' },
          { id: '3', content: 'Run acceptance', status: 'pending' },
        ],
      },
    }
    const result: ToolResult = {
      ...base,
      id: 'tool-result',
      kind: 'tool_result',
      toolCallId: 'tc1',
      toolName: 'todo_write',
      content: 'todos updated: 3 total',
      isError: false,
      metadata: {},
    }

    render(<ToolCallBlock block={block} result={result} />)

    expect(screen.getByRole('region', { name: 'Todo checklist' })).toBeTruthy()
    expect(screen.getByText('1/3 complete · 1 active')).toBeTruthy()
    expect(screen.getByText('Inspect web renderer')).toBeTruthy()
    expect(screen.getByText('in progress')).toBeTruthy()
    expect(screen.queryByText('"todos"')).toBeNull()
  })

  it('renders shell results in terminal chrome', () => {
    const block: ToolCall = {
      ...base,
      kind: 'tool_call',
      toolCallId: 'tc-shell',
      toolName: 'exec_command',
      status: 'finished',
      arguments: { command: 'npm test -- Composer.test.tsx' },
    }
    const result: ToolResult = {
      ...base,
      id: 'tool-result-shell',
      kind: 'tool_result',
      toolCallId: 'tc-shell',
      toolName: 'exec_command',
      content: '15 tests passed',
      isError: false,
      metadata: { exit_code: 0, duration_ms: 1234 },
    }

    render(<ToolCallBlock block={block} result={result} />)

    const terminal = screen.getByRole('region', { name: 'Terminal output' })
    expect(terminal).toBeTruthy()
    expect(within(terminal).getByText('npm test -- Composer.test.tsx')).toBeTruthy()
    expect(within(terminal).getByText('15 tests passed')).toBeTruthy()
    expect(within(terminal).getByText('exit 0')).toBeTruthy()
    expect(screen.queryByText('Tool output')).toBeNull()
  })
  it('normalizes valid todo_write items only', () => {
    expect(todosFromToolArguments({
      todos: [
        { id: 'a', content: 'ok', status: 'pending' },
        { id: 'b', content: '', status: 'completed' },
        { id: 'c', content: 'bad status', status: 'blocked' },
      ],
    })).toEqual([{ id: 'a', content: 'ok', status: 'pending' }])
  })
})

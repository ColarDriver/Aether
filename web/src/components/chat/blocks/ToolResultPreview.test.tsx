// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import type { ToolResultBlock as ToolResult } from '../../../chat-rendering'
import { ToolResultBlock } from './ToolResultBlock'

afterEach(cleanup)

const base = {
  id: 'tool-result',
  sessionId: 'session-1',
  runId: 'run-1',
  timestamp: 1,
  source: 'live',
  kind: 'tool_result',
  toolCallId: 'tc1',
  isError: false,
  metadata: {},
} as const

function result(overrides: Partial<ToolResult>): ToolResult {
  return { ...base, content: '', toolName: 'tool', ...overrides } as ToolResult
}

describe('ToolResultPreview', () => {
  it('renders read_file output as a file preview', () => {
    render(
      <ToolResultBlock
        block={result({ toolName: 'read_file', content: 'export const answer = 42' })}
        toolArguments={{ path: 'src/app.ts' }}
      />,
    )

    const preview = screen.getByRole('region', { name: 'File preview' })
    expect(preview).toBeTruthy()
    expect(within(preview).getByText('src/app.ts')).toBeTruthy()
    expect(within(preview).getAllByText('typescript').length).toBeGreaterThan(0)
    expect(screen.queryByText('Tool output')).toBeNull()
  })

  it('renders grep output as structured search results', () => {
    render(
      <ToolResultBlock
        block={result({ toolName: 'grep', content: 'src/app.ts:12:const token = getToken()\nsrc/lib.ts:4:return token' })}
      />,
    )

    expect(screen.getByRole('region', { name: 'Search results' })).toBeTruthy()
    expect(screen.getByText('src/app.ts')).toBeTruthy()
    expect(screen.getByText(':12')).toBeTruthy()
    expect(screen.getByText('const token = getToken()')).toBeTruthy()
  })

  it('renders web search JSON as result cards', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'web_search',
          content: JSON.stringify({ results: [{ title: 'Aether docs', url: 'https://example.com/aether', snippet: 'Docs snippet' }] }),
        })}
      />,
    )

    expect(screen.getByRole('region', { name: 'Web results' })).toBeTruthy()
    expect(screen.getByText('Aether docs')).toBeTruthy()
    expect(screen.getByText('Docs snippet')).toBeTruthy()
  })


  it('renders nested provider web-search payloads and web-fetch metadata', () => {
    const { rerender } = render(
      <ToolResultBlock
        block={result({
          toolName: 'web_search',
          content: JSON.stringify({ web: { results: [{ title: 'Brave result', url: 'https://example.com/brave', description: 'Brave snippet' }] } }),
        })}
      />,
    )

    expect(screen.getByRole('region', { name: 'Web results' })).toBeTruthy()
    expect(screen.getByText('Brave result')).toBeTruthy()
    expect(screen.getByText('Brave snippet')).toBeTruthy()

    rerender(
      <ToolResultBlock
        block={result({
          toolName: 'web_search',
          content: JSON.stringify({ data: { webPages: { value: [{ name: 'Bocha result', url: 'https://example.com/bocha', snippet: 'Bocha snippet' }] } } }),
        })}
      />,
    )

    expect(screen.getByText('Bocha result')).toBeTruthy()
    expect(screen.getByText('Bocha snippet')).toBeTruthy()

    rerender(
      <ToolResultBlock
        block={result({
          toolName: 'web_fetch',
          content: 'Long document body',
          metadata: { title: 'Fetched document', url: 'https://example.com/doc' },
        })}
      />,
    )

    expect(screen.getByText('Fetched document')).toBeTruthy()
    expect(screen.getByText('Long document body')).toBeTruthy()
  })

  it('renders image artifacts from structured tool output', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'browser_screenshot',
          content: JSON.stringify({
            images: [{ title: 'Viewport capture', url: 'https://example.com/screenshot.png', caption: 'Browser screenshot' }],
          }),
        })}
      />,
    )

    const preview = screen.getByRole('region', { name: 'Tool image results' })
    expect(preview).toBeTruthy()
    expect(within(preview).getByRole('img', { name: 'Viewport capture' })).toBeTruthy()
    expect(within(preview).getByText('Browser screenshot')).toBeTruthy()
    expect(screen.queryByText('Tool output')).toBeNull()

    fireEvent.click(within(preview).getByRole('button', { name: 'Open image preview Viewport capture' }))
    expect(screen.getByRole('dialog', { name: 'Viewport capture' })).toBeTruthy()
  })

  it('renders subagent results as a structured inline task summary', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'spawn_agent',
          content: 'Mapped auth files',
          metadata: {
            task_id: 'task-123',
            model: 'gpt-5.4',
            subagent_type: 'explorer',
            duration_ms: 2400,
            input_tokens: 120,
            output_tokens: 80,
            result_path: 'tasks/task-123.md',
          },
        })}
        toolArguments={{ prompt: 'Explore auth flow' }}
      />,
    )

    const summary = screen.getByRole('region', { name: 'Subagent result' })
    expect(summary).toBeTruthy()
    expect(screen.getByText('Explore auth flow')).toBeTruthy()
    expect(screen.getByText('explorer / gpt-5.4')).toBeTruthy()
    expect(screen.getByText('2.4s')).toBeTruthy()
    expect(screen.getByText('200')).toBeTruthy()
    expect(screen.getByText('tasks/task-123.md')).toBeTruthy()
    expect(screen.getByText('Mapped auth files')).toBeTruthy()
  })

  it('renders file edit metadata as a structured change preview', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'file_edit',
          content: 'updated /workspace/Aether/src/auth.ts (2 substitutions)',
          metadata: {
            path: 'src/auth.ts',
            lines_added: 4,
            lines_removed: 2,
            hunks: 1,
            change_count: 2,
          },
        })}
      />,
    )

    const preview = screen.getByRole('region', { name: 'File change' })
    expect(preview).toBeTruthy()
    expect(within(preview).getByText('src/auth.ts')).toBeTruthy()
    expect(within(preview).getByText('+4')).toBeTruthy()
    expect(within(preview).getByText('-2')).toBeTruthy()
    expect(within(preview).getByText('updated /workspace/Aether/src/auth.ts (2 substitutions)')).toBeTruthy()
    expect(screen.queryByText('Tool output')).toBeNull()
  })

  it('renders notebook edits with cell metadata', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'notebook_edit',
          content: 'inserted new code cell at index 2 (id=abc123)',
          metadata: { path: 'analysis.ipynb', edit_mode: 'insert', cell_count: 8 },
        })}
        toolArguments={{ cell_idx: 1, cell_type: 'code' }}
      />,
    )

    const preview = screen.getByRole('region', { name: 'Notebook edit' })
    expect(preview).toBeTruthy()
    expect(within(preview).getByText('analysis.ipynb')).toBeTruthy()
    expect(within(preview).getAllByText('insert').length).toBeGreaterThan(0)
    expect(within(preview).getByText('cell 1')).toBeTruthy()
    expect(within(preview).getByText('8')).toBeTruthy()
  })

  it('renders notebook cell source and diff previews from tool arguments', () => {
    const { rerender } = render(
      <ToolResultBlock
        block={result({
          toolName: 'notebook_edit',
          content: 'inserted new markdown cell at index 2 (id=abc123)',
          metadata: { path: 'analysis.ipynb', edit_mode: 'insert', cell_count: 8 },
        })}
        toolArguments={{ cell_idx: 1, cell_type: 'markdown', new_source: '# Analysis\n\nNotes' }}
      />,
    )

    expect(screen.getByLabelText('Notebook cell source')).toBeTruthy()
    expect(screen.getByText('Cell source')).toBeTruthy()
    expect(screen.getByText(/# Analysis/)).toBeTruthy()

    rerender(
      <ToolResultBlock
        block={result({
          toolName: 'notebook_edit',
          content: 'replaced cell at index 1 (id=abc123, type=code)',
          metadata: { path: 'analysis.ipynb', edit_mode: 'replace', cell_count: 8 },
        })}
        toolArguments={{ cell_idx: 1, cell_type: 'code', old_source: 'print("old")', new_source: 'print("new")' }}
      />,
    )

    expect(screen.getByLabelText('Notebook cell diff')).toBeTruthy()
    expect(document.querySelector('.diff-line-remove')?.textContent).toContain('print("old")')
    expect(document.querySelector('.diff-line-add')?.textContent).toContain('print("new")')
  })

  it('renders LSP results as semantic navigation rows', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'lsp',
          content: '# LSP findReferences for src/auth.ts\n\n- /workspace/Aether/src/auth.ts:12:4\n- /workspace/Aether/src/app.ts:3:8\n',
          metadata: { operation: 'findReferences', file_path: 'src/auth.ts' },
        })}
      />,
    )

    const preview = screen.getByRole('region', { name: 'LSP result' })
    expect(preview).toBeTruthy()
    expect(within(preview).getByText('findReferences')).toBeTruthy()
    expect(within(preview).getByText('/workspace/Aether/src/auth.ts:12:4')).toBeTruthy()
  })

  it('renders browser screenshot metadata as an artifact card', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'web_browser',
          content: '# Browser screenshot saved\n- url: https://example.com\n- path: /tmp/call.png\n',
          metadata: { operation: 'screenshot', url: 'https://example.com', screenshot_path: '/tmp/call.png', bytes: 2048 },
        })}
      />,
    )

    const preview = screen.getByRole('region', { name: 'Browser result' })
    expect(preview).toBeTruthy()
    expect(within(preview).getByText('screenshot')).toBeTruthy()
    expect(within(preview).getByText('https://example.com')).toBeTruthy()
    expect(within(preview).getByText('Screenshot saved')).toBeTruthy()
    expect(within(preview).getByText('/tmp/call.png')).toBeTruthy()
  })

  it('renders structured non-image artifacts as an artifact bundle', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'artifact_tool',
          content: JSON.stringify({
            artifacts: [
              { name: 'report.json', path: '/tmp/report.json', mime_type: 'application/json', size_bytes: 128, summary: 'Structured result' },
            ],
          }),
        })}
      />,
    )

    const preview = screen.getByRole('region', { name: 'Tool artifacts' })
    expect(preview).toBeTruthy()
    expect(within(preview).getByText('report.json')).toBeTruthy()
    expect(within(preview).getByText((text) => text.includes('application/json'))).toBeTruthy()
    expect(within(preview).getByText('Structured result')).toBeTruthy()
    expect(within(preview).getByRole('button', { name: 'Copy report.json path' })).toBeTruthy()
    expect(screen.queryByText('Tool output')).toBeNull()
  })

  it('renders standard spill notices as readable artifacts', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'list_dir',
          content: 'preview\n\n... [output truncated: 100000 chars / 1200 lines saved to ~/.aether/tool_results/s/call.txt — use read_file to retrieve the full content] ...',
          metadata: { spilled: true, truncated: true },
        })}
      />,
    )

    const preview = screen.getByRole('region', { name: 'Tool artifacts' })
    expect(preview).toBeTruthy()
    expect(within(preview).getByText('call.txt')).toBeTruthy()
    expect(within(preview).getByText('Full output: 100000 chars / 1200 lines')).toBeTruthy()
    expect(within(preview).getByRole('button', { name: 'Copy call.txt path' })).toBeTruthy()
  })

  it('falls back to generic tool output for unknown tools', () => {
    render(<ToolResultBlock block={result({ toolName: 'unknown_tool', content: 'raw output' })} />)

    expect(screen.getByText('Tool output')).toBeTruthy()
    expect(screen.getByText('raw output')).toBeTruthy()
  })
})

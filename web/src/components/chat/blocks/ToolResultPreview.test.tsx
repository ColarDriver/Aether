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

  it('renders web search JSON as result cards with provider metadata', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'web_search',
          content: JSON.stringify({ results: [{ title: 'Aether docs', url: 'https://example.com/aether', snippet: 'Docs snippet' }] }),
          metadata: { provider: 'brave', source_count: 1 },
        })}
        toolArguments={{ query: 'Aether provider' }}
      />,
    )

    const preview = screen.getByRole('region', { name: 'Web results' })
    expect(preview).toBeTruthy()
    expect(within(preview).getByText('Aether docs')).toBeTruthy()
    expect(within(preview).getByText('Docs snippet')).toBeTruthy()
    expect(within(preview).getByText('brave')).toBeTruthy()
    expect(within(preview).getByText('Aether provider')).toBeTruthy()
    expect(within(preview).getByText('1')).toBeTruthy()
  })


  it('does not make unsafe web result URLs clickable', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'web_search',
          content: JSON.stringify({ results: [{ title: 'Unsafe result', url: 'javascript:alert(1)', snippet: 'Do not link this' }] }),
        })}
      />,
    )

    const preview = screen.getByRole('region', { name: 'Web results' })
    expect(within(preview).getByText('Unsafe result')).toBeTruthy()
    expect(within(preview).getByText('Do not link this')).toBeTruthy()
    expect(within(preview).queryByRole('link', { name: /Unsafe result/ })).toBeNull()
  })

  it('renders Aether markdown web_search output as result cards', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'web_search',
          content: '# Web search: aether\n\nFound 2 results:\n\n1. **Aether docs**\n   https://example.com/aether\n   Documentation snippet\n\n2. **Aether issue**\n   https://github.com/example/aether/issues/1\n   Issue snippet\n',
          metadata: { provider: 'brave', result_count: 2 },
        })}
      />,
    )

    const preview = screen.getByRole('region', { name: 'Web results' })
    expect(preview).toBeTruthy()
    expect(within(preview).getByText('Aether docs')).toBeTruthy()
    expect(within(preview).getByText('Documentation snippet')).toBeTruthy()
    expect(within(preview).getByText('https://example.com/aether')).toBeTruthy()
    expect(within(preview).getByText('Aether issue')).toBeTruthy()
    expect(screen.queryByText('Tool output')).toBeNull()
  })

  it('renders hosted provider web-search sources from metadata', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'web_search',
          content: 'Found docs.\n\nSources:\n- [Aether Docs](https://docs.example/aether)',
          metadata: {
            hosted_web_search: {
              provider: 'codex',
              source_count: 1,
              sources: [{ title: 'Aether Docs', url: 'https://docs.example/aether' }],
            },
          },
        })}
        toolArguments={{ query: 'Aether' }}
      />,
    )

    const preview = screen.getByRole('region', { name: 'Web results' })
    expect(within(preview).getByText('codex')).toBeTruthy()
    expect(within(preview).getByText('Aether')).toBeTruthy()
    expect(within(preview).getByText('Aether Docs')).toBeTruthy()
    expect(within(preview).getByText('https://docs.example/aether')).toBeTruthy()
    expect(within(preview).queryByText((text) => text.includes('Found docs.'))).toBeNull()
  })

  it('extracts hosted provider query and source count from call metadata', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'web_search',
          content: 'Found docs.',
          metadata: {
            hosted_web_search: {
              provider: 'codex',
              calls: [{ id: 'ws_1', status: 'completed', action: { type: 'search', query: 'codex hosted query' } }],
              sources: [
                { title: 'First source', url: 'https://docs.example/first' },
                { title: 'Second source', url: 'https://docs.example/second' },
              ],
            },
          },
        })}
      />,
    )

    const preview = screen.getByRole('region', { name: 'Web results' })
    expect(within(preview).getByText('codex')).toBeTruthy()
    expect(within(preview).getByText('codex hosted query')).toBeTruthy()
    expect(within(preview).getByText('2')).toBeTruthy()
    expect(within(preview).getByText('First source')).toBeTruthy()
    expect(within(preview).getByText('Second source')).toBeTruthy()
  })

  it('renders raw provider web-search payload variants', () => {
    const { rerender } = render(
      <ToolResultBlock
        block={result({
          toolName: 'web_search',
          content: JSON.stringify({
            answer: 'Aggregated answer',
            results: [{ title: 'Tavily result', url: 'https://example.com/tavily', content: 'Tavily content body', score: 0.98 }],
          }),
          metadata: { provider: 'tavily' },
        })}
      />,
    )

    let preview = screen.getByRole('region', { name: 'Web results' })
    expect(within(preview).getByText('tavily')).toBeTruthy()
    expect(within(preview).getByText('Tavily result')).toBeTruthy()
    expect(within(preview).getByText('Tavily content body')).toBeTruthy()

    rerender(
      <ToolResultBlock
        block={result({
          toolName: 'web_search',
          content: JSON.stringify({ web: { results: [{ title: 'Brave extra', url: 'https://example.com/brave-extra', extra_snippets: ['First extra', 'Second extra'] }] } }),
          metadata: { provider: 'brave' },
        })}
      />,
    )

    preview = screen.getByRole('region', { name: 'Web results' })
    expect(within(preview).getByText('Brave extra')).toBeTruthy()
    expect(within(preview).getByText('First extra Second extra')).toBeTruthy()

    rerender(
      <ToolResultBlock
        block={result({
          toolName: 'web_search',
          content: JSON.stringify({
            content: [
              {
                type: 'web_search_tool_result',
                content: [{ type: 'web_search_result', title: 'Anthropic result', url: 'https://example.com/anthropic' }],
              },
            ],
          }),
          metadata: { hosted_web_search: { provider: 'anthropic', calls: [{ input: { query: 'anthropic query' } }] } },
        })}
      />,
    )

    preview = screen.getByRole('region', { name: 'Web results' })
    expect(within(preview).getByText('anthropic')).toBeTruthy()
    expect(within(preview).getByText('anthropic query')).toBeTruthy()
    expect(within(preview).getByText('Anthropic result')).toBeTruthy()

    rerender(
      <ToolResultBlock
        block={result({
          toolName: 'web_search',
          content: JSON.stringify({ citations: ['https://example.com/plain-citation'] }),
        })}
      />,
    )

    preview = screen.getByRole('region', { name: 'Web results' })
    expect(within(preview).getAllByText('https://example.com/plain-citation').length).toBeGreaterThan(0)
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

  it('renders notebook stdout errors and display images as cell outputs', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'notebook_edit',
          content: JSON.stringify({
            summary: 'executed cell',
            outputs: [
              { output_type: 'stream', name: 'stdout', text: ['hello', '\nworld'] },
              { output_type: 'error', ename: 'ValueError', evalue: 'bad value', traceback: ['Traceback line', 'ValueError: bad value'] },
              { output_type: 'display_data', data: { 'image/png': 'iVBORw0KGgo=' } },
            ],
          }),
          metadata: { path: 'analysis.ipynb', edit_mode: 'execute', cell_count: 8 },
        })}
        toolArguments={{ cell_idx: 2, cell_type: 'code' }}
      />,
    )

    const outputs = screen.getByLabelText('Notebook outputs')
    expect(within(outputs).getByText('stdout')).toBeTruthy()
    expect(within(outputs).getByText((text) => text.includes('hello') && text.includes('world'))).toBeTruthy()
    expect(within(outputs).getByText('ValueError')).toBeTruthy()
    expect(within(outputs).getByText(/Traceback line/)).toBeTruthy()
    expect(within(outputs).getByRole('img', { name: 'image/png' }).getAttribute('src')).toBe('data:image/png;base64,iVBORw0KGgo=')
    expect(screen.getByText('executed cell')).toBeTruthy()
    expect(screen.queryByText(/display_data/)).toBeNull()
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

  it('renders notebook execution-state metadata', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'notebook_edit',
          content: JSON.stringify({ summary: 'executed cell', status: 'ok' }),
          metadata: {
            path: 'analysis.ipynb',
            edit_mode: 'execute',
            execution_count: 4,
            duration_ms: 1250,
            kernel_name: 'python3',
            outputs_truncated: true,
          },
        })}
        toolArguments={{ cell_idx: 2, cell_type: 'code' }}
      />,
    )

    const preview = screen.getByRole('region', { name: 'Notebook edit' })
    expect(within(preview).getAllByText('ok').length).toBeGreaterThan(0)
    expect(within(preview).getByText('#4')).toBeTruthy()
    expect(within(preview).getAllByText('1.3s').length).toBeGreaterThan(0)
    expect(within(preview).getByText('python3')).toBeTruthy()
    expect(within(preview).getByText('truncated')).toBeTruthy()
  })

  it('renders notebook lifecycle metadata as a timeline', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'notebook_edit',
          content: JSON.stringify({ summary: 'executed cell', status: 'ok' }),
          metadata: {
            path: 'analysis.ipynb',
            edit_mode: 'execute',
            queued_at: '10:00:00',
            started_at: '10:00:01',
            finished_at: '10:00:03',
            duration_ms: 2100,
          },
        })}
        toolArguments={{ cell_idx: 2, cell_type: 'code' }}
      />,
    )

    const lifecycle = screen.getByLabelText('Notebook lifecycle')
    expect(within(lifecycle).getByText('queued')).toBeTruthy()
    expect(within(lifecycle).getByText('10:00:00')).toBeTruthy()
    expect(within(lifecycle).getByText('started')).toBeTruthy()
    expect(within(lifecycle).getByText('finished')).toBeTruthy()
    expect(within(lifecycle).getByText('2.1s')).toBeTruthy()
  })

  it('renders notebook lifecycle arrays and nested timing records', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'notebook_edit',
          content: JSON.stringify({
            summary: 'executed cell',
            execution: { queued_at: '10:00:00', started_at: '10:00:01' },
            lifecycle_events: [
              { phase: 'kernel_start', at: '10:00:02' },
              { phase: 'execute_cell', duration_ms: 800, status: 'running' },
              { status: 'completed', at: '10:00:03' },
            ],
          }),
          metadata: { path: 'analysis.ipynb', edit_mode: 'execute' },
        })}
        toolArguments={{ cell_idx: 2, cell_type: 'code' }}
      />,
    )

    const lifecycle = screen.getByLabelText('Notebook lifecycle')
    expect(within(lifecycle).getByText('queued')).toBeTruthy()
    expect(within(lifecycle).getByText('10:00:00')).toBeTruthy()
    expect(within(lifecycle).getByText('started')).toBeTruthy()
    expect(within(lifecycle).getByText('10:00:01')).toBeTruthy()
    expect(within(lifecycle).getByText('kernel start')).toBeTruthy()
    expect(within(lifecycle).getByText('10:00:02')).toBeTruthy()
    expect(within(lifecycle).getByText('execute cell')).toBeTruthy()
    expect(within(lifecycle).getByText('0.8s')).toBeTruthy()
    expect(within(lifecycle).getByText('completed')).toBeTruthy()
    expect(within(lifecycle).getByText('10:00:03')).toBeTruthy()
  })

  it('renders browser screenshot URLs as inline visual previews', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'web_browser',
          content: '# Browser screenshot saved',
          metadata: {
            operation: 'screenshot',
            url: 'https://example.com',
            screenshot_url: 'https://example.com/call.png',
            screenshot_path: '/tmp/call.png',
            screenshot_name: 'Checkout page',
            bytes: 2048,
          },
        })}
      />,
    )

    const preview = screen.getByRole('region', { name: 'Browser result' })
    expect(within(preview).getByRole('img', { name: 'Checkout page' }).getAttribute('src')).toBe('https://example.com/call.png')
    expect(within(preview).getByText('/tmp/call.png')).toBeTruthy()
    expect(within(preview).getByRole('link', { name: 'Open' }).getAttribute('href')).toBe('https://example.com/call.png')
    expect(within(preview).queryByText('Screenshot saved')).toBeNull()
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

  it('renders structured browser image and artifact payloads inside the browser preview', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'web_browser',
          content: JSON.stringify({
            images: [
              { title: 'After click viewport', url: 'https://example.com/after-click.png', caption: 'Captured after clicking submit' },
            ],
            artifacts: [
              { name: 'trace.zip', path: '/tmp/aether/trace.zip', download_url: 'https://example.com/trace.zip', mime_type: 'application/zip', size_bytes: 4096 },
            ],
          }),
          metadata: { operation: 'click', url: 'https://example.com/form' },
        })}
      />,
    )

    const preview = screen.getByRole('region', { name: 'Browser result' })
    expect(within(preview).getByText('click')).toBeTruthy()
    expect(within(preview).getByText('https://example.com/form')).toBeTruthy()
    const images = within(preview).getByLabelText('Browser images')
    expect(within(images).getByRole('img', { name: 'After click viewport' }).getAttribute('src')).toBe('https://example.com/after-click.png')
    expect(within(images).getByText('Captured after clicking submit')).toBeTruthy()
    const artifacts = within(preview).getByLabelText('Browser artifacts')
    expect(within(artifacts).getByText('trace.zip')).toBeTruthy()
    expect(within(artifacts).getByRole('link', { name: 'Open' }).getAttribute('href')).toBe('https://example.com/trace.zip')
    expect(within(artifacts).getByRole('button', { name: 'Copy trace.zip path' })).toBeTruthy()
    expect(screen.queryByText('Tool output')).toBeNull()
  })

  it('renders structured non-image artifacts as an artifact bundle', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'artifact_tool',
          content: JSON.stringify({
            artifacts: [
              { name: 'report.json', path: '/tmp/report.json', mime_type: 'application/json', size_bytes: 128, summary: 'Structured result', content: { ok: true, count: 2 } },
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
    expect(within(preview).getByRole('button', { name: 'Copy report.json contents' })).toBeTruthy()
    expect(within(preview).getByLabelText('Preview report.json').textContent).toContain('"ok": true')
    expect(within(preview).getByLabelText('Preview report.json').textContent).toContain('"count": 2')
    expect(screen.queryByText('Tool output')).toBeNull()
  })

  it('keeps local binary artifacts copy-only with an unavailable preview state', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'artifact_tool',
          content: JSON.stringify({
            artifacts: [
              { name: 'weights.bin', path: '/tmp/aether/weights.bin', mime_type: 'application/octet-stream', size_bytes: 4096, binary: true },
            ],
          }),
        })}
      />,
    )

    const preview = screen.getByRole('region', { name: 'Tool artifacts' })
    expect(within(preview).getByText('weights.bin')).toBeTruthy()
    expect(within(preview).getByText((text) => text.includes('binary') && text.includes('application/octet-stream'))).toBeTruthy()
    expect(within(preview).getByText('Binary preview unavailable. Copy the path or open the linked artifact if a URL is provided.')).toBeTruthy()
    expect(within(preview).getByRole('button', { name: 'Copy weights.bin path' })).toBeTruthy()
    expect(within(preview).queryByRole('link', { name: 'Open' })).toBeNull()
  })

  it('opens only explicit artifact URLs while local paths remain copy targets', () => {
    render(
      <ToolResultBlock
        block={result({
          toolName: 'artifact_tool',
          content: JSON.stringify({
            artifacts: [
              { name: 'bundle.zip', path: '/tmp/aether/bundle.zip', download_url: 'https://example.com/bundle.zip', mime_type: 'application/zip', size_bytes: 1000 },
            ],
          }),
        })}
      />,
    )

    const preview = screen.getByRole('region', { name: 'Tool artifacts' })
    expect(within(preview).getByRole('link', { name: 'Open' }).getAttribute('href')).toBe('https://example.com/bundle.zip')
    expect(within(preview).getByRole('button', { name: 'Copy bundle.zip path' })).toBeTruthy()
    expect(within(preview).getByText((text) => text.includes('/tmp/aether/bundle.zip'))).toBeTruthy()
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

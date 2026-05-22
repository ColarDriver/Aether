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

  it('renders subagent results as an inline task summary', () => {
    render(
      <ToolResultBlock
        block={result({ toolName: 'spawn_agent', content: 'Mapped auth files', metadata: { model: 'gpt-5.4' } })}
        toolArguments={{ prompt: 'Explore auth flow' }}
      />,
    )

    expect(screen.getByRole('region', { name: 'Subagent result' })).toBeTruthy()
    expect(screen.getByText('Explore auth flow')).toBeTruthy()
    expect(screen.getByText('Model: gpt-5.4')).toBeTruthy()
    expect(screen.getByText('Mapped auth files')).toBeTruthy()
  })

  it('falls back to generic tool output for unknown tools', () => {
    render(<ToolResultBlock block={result({ toolName: 'unknown_tool', content: 'raw output' })} />)

    expect(screen.getByText('Tool output')).toBeTruthy()
    expect(screen.getByText('raw output')).toBeTruthy()
  })
})

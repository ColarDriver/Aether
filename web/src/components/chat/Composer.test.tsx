// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../../api/client'
import { Composer } from './Composer'

const slashCommands = [
  { name: '/help', description: 'Show help', category: 'local' },
  { name: '/plan', description: 'Plan mode', category: 'session' },
]

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('Composer', () => {
  it('applies slash completion instead of sending while popover is open', () => {
    const onSend = vi.fn()
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '/pl', selectionStart: 3 } })
    fireEvent.keyDown(textbox, { key: 'Enter' })

    expect(onSend).not.toHaveBeenCalled()
    expect(textbox.value).toBe('/plan ')
  })

  it('sends normal messages with Enter and preserves Shift+Enter', () => {
    const onSend = vi.fn()
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: 'hello', selectionStart: 5 } })
    fireEvent.keyDown(textbox, { key: 'Enter', shiftKey: true })
    expect(onSend).not.toHaveBeenCalled()

    fireEvent.keyDown(textbox, { key: 'Enter' })
    expect(onSend).toHaveBeenCalledWith('hello')
  })

  it('routes complete slash commands separately from agent prompts', () => {
    const onSend = vi.fn()
    const onSlashCommand = vi.fn()
    render(
      <Composer
        disabled={false}
        running={false}
        onCancel={() => undefined}
        onSend={onSend}
        onSlashCommand={onSlashCommand}
        slashCommands={slashCommands}
      />,
    )
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '/help', selectionStart: 5 } })
    fireEvent.keyDown(textbox, { key: 'Enter' })

    expect(onSlashCommand).toHaveBeenCalledWith('/help')
    expect(onSend).not.toHaveBeenCalled()
  })

  it('opens local inspector commands without sending to the agent', async () => {
    const onSend = vi.fn()
    const onSlashCommand = vi.fn()
    vi.spyOn(api, 'status').mockResolvedValue({ ok: true, name: 'Aether', version: '1.1.0', web: { enabled: true } })
    vi.spyOn(api, 'health').mockResolvedValue({
      status: 'ok',
      runtime: { python_version: '3.12', platform: 'linux', implementation: 'CPython' },
      services: [{ name: 'sessions', available: true, status: 'ok' }],
    })
    vi.spyOn(api, 'currentProvider').mockResolvedValue({
      family: 'openai',
      provider_name: 'codex',
      model: 'gpt-5.4',
      api_key_env_names: [],
      model_env_names: [],
      base_url_env_names: [],
      source: 'test',
    })
    render(
      <Composer
        disabled={false}
        running={false}
        onCancel={() => undefined}
        onSend={onSend}
        onSlashCommand={onSlashCommand}
        slashCommands={slashCommands}
        sessionId="session-123456"
        sessionSummary="Test session"
        messageCount={4}
        provider="codex"
        model="gpt-5.4"
      />,
    )
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '/status', selectionStart: 7 } })
    fireEvent.keyDown(textbox, { key: 'Enter' })

    expect(onSend).not.toHaveBeenCalled()
    expect(onSlashCommand).not.toHaveBeenCalled()
    expect(screen.getByLabelText('Status inspector')).toBeTruthy()
    await waitFor(() => expect(screen.getByText('Runtime')).toBeTruthy())
  })

  it('renders skill inspector data from the skills API', async () => {
    const onSend = vi.fn()
    vi.spyOn(api, 'skills').mockResolvedValue({
      skills: [
        {
          name: 'web-audit',
          description: 'Audit web UI parity',
          when_to_use: 'When checking web migration gaps',
          source: { source: 'local', path: '/workspace/Aether/skills/web-audit/SKILL.md' },
        },
      ],
    })
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '/skills', selectionStart: 7 } })
    fireEvent.keyDown(textbox, { key: 'Enter' })

    await waitFor(() => expect(screen.getByText('web-audit')).toBeTruthy())
    expect(onSend).not.toHaveBeenCalled()
  })

  it('shows MCP as an explicit unavailable local panel and closes with Escape', () => {
    const onSend = vi.fn()
    const onSlashCommand = vi.fn()
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} onSlashCommand={onSlashCommand} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '/mcp', selectionStart: 4 } })
    fireEvent.keyDown(textbox, { key: 'Enter' })

    expect(screen.getByText(/MCP management is not wired/)).toBeTruthy()
    expect(onSend).not.toHaveBeenCalled()
    expect(onSlashCommand).not.toHaveBeenCalled()

    fireEvent.keyDown(window, { key: 'Escape' })
    expect(screen.queryByLabelText('MCP inspector')).toBeNull()
  })

  it('includes local inspector commands in slash completion', () => {
    const onSend = vi.fn()
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '/stat', selectionStart: 5 } })

    expect(screen.getByRole('option', { name: /\/status/ })).toBeTruthy()
  })

  it('opens inspector panels from the composer control menu', () => {
    const onSend = vi.fn()
    const onSlashCommand = vi.fn()
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} onSlashCommand={onSlashCommand} slashCommands={slashCommands} />)

    fireEvent.click(screen.getByLabelText('Open composer menu'))
    expect(screen.getByRole('menu', { name: 'Composer menu' })).toBeTruthy()
    fireEvent.click(screen.getByRole('menuitem', { name: 'MCP' }))

    expect(screen.getByLabelText('MCP inspector')).toBeTruthy()
    expect(onSend).not.toHaveBeenCalled()
    expect(onSlashCommand).not.toHaveBeenCalled()
  })

  it('inserts slash and workspace triggers from the composer control menu', () => {
    const onSend = vi.fn()
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.click(screen.getByLabelText('Open composer menu'))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Slash command' }))
    expect(textbox.value).toBe('/')

    fireEvent.click(screen.getByLabelText('Open composer menu'))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Workspace reference' }))
    expect(textbox.value).toBe('/ @')
  })

  it('preserves drafts independently while switching sessions', async () => {
    const onSend = vi.fn()
    const { rerender } = render(
      <Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} sessionId="session-a" />
    )
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: 'draft for a', selectionStart: 11 } })
    rerender(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} sessionId="session-b" />)
    await waitFor(() => expect(textbox.value).toBe(''))

    fireEvent.change(textbox, { target: { value: 'draft for b', selectionStart: 11 } })
    rerender(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} sessionId="session-a" />)
    await waitFor(() => expect(textbox.value).toBe('draft for a'))

    rerender(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} sessionId="session-b" />)
    await waitFor(() => expect(textbox.value).toBe('draft for b'))
  })
  it('sends absolute paths as normal prompts', () => {
    const onSend = vi.fn()
    const onSlashCommand = vi.fn()
    render(
      <Composer
        disabled={false}
        running={false}
        onCancel={() => undefined}
        onSend={onSend}
        onSlashCommand={onSlashCommand}
        slashCommands={slashCommands}
      />,
    )
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '/workspace/Aether', selectionStart: 17 } })
    fireEvent.keyDown(textbox, { key: 'Enter' })

    expect(onSend).toHaveBeenCalledWith('/workspace/Aether')
    expect(onSlashCommand).not.toHaveBeenCalled()
  })

  it('previews selected files and sends them with the prompt', async () => {
    const onSend = vi.fn()
    const { container } = render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement
    const input = container.querySelector('input[type="file"]') as HTMLInputElement
    const file = new File(['notes'], 'notes.md', { type: 'text/markdown' })

    fireEvent.change(input, { target: { files: [file] } })

    await waitFor(() => expect(screen.getByText('notes.md')).toBeTruthy())
    fireEvent.change(textbox, { target: { value: 'summarize', selectionStart: 9 } })
    fireEvent.keyDown(textbox, { key: 'Enter' })

    expect(onSend).toHaveBeenCalledWith('summarize', [
      expect.objectContaining({ type: 'text', name: 'notes.md', mimeType: 'text/markdown' }),
    ])
  })

  it('accepts pasted files as attachments', async () => {
    const onSend = vi.fn()
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement
    const file = new File(['plain'], 'clip.txt', { type: 'text/plain' })

    fireEvent.paste(textbox, {
      clipboardData: {
        files: [file],
        items: [],
      },
    })

    await waitFor(() => expect(screen.getByText('clip.txt')).toBeTruthy())
  })

  it('turns selected workspace references into path tokens and attachments', async () => {
    const onSend = vi.fn()
    vi.spyOn(api, 'workspaceSearch').mockResolvedValue({
      root: '/workspace/Aether',
      query: 'app',
      entries: [{ kind: 'file', name: 'app.ts', path: 'src/app.ts' }],
    })
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '@app', selectionStart: 4 } })
    await waitFor(() => expect(screen.getByRole('option', { name: /app.ts/ })).toBeTruthy())
    fireEvent.keyDown(textbox, { key: 'Enter' })

    expect(textbox.value).toBe('@src/app.ts ')
    expect(screen.getByText('src/app.ts')).toBeTruthy()
    fireEvent.keyDown(textbox, { key: 'Enter' })
    expect(onSend).toHaveBeenCalledWith('@src/app.ts', [
      expect.objectContaining({ type: 'text', name: 'app.ts', path: 'src/app.ts' }),
    ])
  })

  it('shows workspace references as managed context chips and removes matching text tokens', async () => {
    const onSend = vi.fn()
    vi.spyOn(api, 'workspaceSearch').mockResolvedValue({
      root: '/workspace/Aether',
      query: 'app',
      entries: [{ kind: 'file', name: 'app.ts', path: 'src/app.ts' }],
    })
    render(<Composer disabled={false} running={false} onCancel={() => undefined} onSend={onSend} slashCommands={slashCommands} />)
    const textbox = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.change(textbox, { target: { value: '@app', selectionStart: 4 } })
    await waitFor(() => expect(screen.getByRole('option', { name: /app.ts/ })).toBeTruthy())
    fireEvent.keyDown(textbox, { key: 'Enter' })

    const context = screen.getByLabelText('Workspace context')
    expect(context).toBeTruthy()
    expect(within(context).getByText('Workspace context')).toBeTruthy()
    expect(within(context).getByText('1 ref')).toBeTruthy()
    expect(screen.getByText('app.ts')).toBeTruthy()
    expect(screen.getByText('src/app.ts')).toBeTruthy()

    fireEvent.click(screen.getByLabelText('Remove workspace reference app.ts'))
    await waitFor(() => expect(screen.queryByLabelText('Workspace context')).toBeNull())
    expect(textbox.value).toBe('')
  })

  it('renders command-surface metadata in the footer', () => {
    const onSend = vi.fn()
    render(
      <Composer
        disabled={false}
        running={false}
        onCancel={() => undefined}
        onSend={onSend}
        slashCommands={slashCommands}
        provider="codex"
        model="gpt-5.4"
        mode="plan"
        inputTokens={1200}
        outputTokens={800}
      />,
    )

    expect(screen.getByText('gpt-5.4')).toBeTruthy()
    expect(screen.getByText('Workspace')).toBeTruthy()
    expect(screen.getByText('plan')).toBeTruthy()
    expect(screen.getByLabelText(/2,000 active-run tokens/)).toBeTruthy()
  })
})

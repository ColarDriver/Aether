// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
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
})

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Composer } from './Composer'

const slashCommands = [
  { name: '/help', description: 'Show help', category: 'local' },
  { name: '/plan', description: 'Plan mode', category: 'session' },
]

afterEach(cleanup)

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
})

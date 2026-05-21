// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { SlashPopover } from './SlashPopover'

const commands = [
  { name: '/help', description: 'Show help', category: 'local' },
  { name: '/plan', description: 'Plan mode', category: 'session' },
]

afterEach(cleanup)

describe('SlashPopover', () => {
  it('renders matching commands and applies clicked selections', () => {
    const onApply = vi.fn()
    render(<SlashPopover commands={commands} value="/pl" cursorPosition={3} onApply={onApply} />)

    fireEvent.click(screen.getByRole('option', { name: /\/plan/ }))

    expect(onApply).toHaveBeenCalledWith('/plan ', 6)
  })

  it('does not render outside a slash trigger', () => {
    render(<SlashPopover commands={commands} value="hello" cursorPosition={5} onApply={() => undefined} />)

    expect(screen.queryByRole('listbox', { name: 'Slash commands' })).toBeNull()
  })
})

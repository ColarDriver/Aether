// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { CopyButton } from './CopyButton'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('CopyButton', () => {
  it('copies text through the browser clipboard and shows copied state', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText },
    })

    render(<CopyButton text="hello" label="Copy message" />)

    fireEvent.click(screen.getByRole('button', { name: 'Copy message' }))

    await waitFor(() => expect(writeText).toHaveBeenCalledWith('hello'))
    expect(screen.getByRole('button', { name: 'Copied' })).toBeTruthy()
  })

  it('disables empty copy actions', () => {
    render(<CopyButton text="" label="Copy empty" />)

    expect((screen.getByRole('button', { name: 'Copy empty' }) as HTMLButtonElement).disabled).toBe(true)
  })
})

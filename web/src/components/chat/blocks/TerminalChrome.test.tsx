// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { TerminalChrome } from './TerminalChrome'

afterEach(cleanup)

describe('TerminalChrome', () => {
  it('truncates long output and can expand it', () => {
    const output = Array.from({ length: 4 }, (_, index) => 'line ' + (index + 1)).join('\n')

    render(<TerminalChrome command="run long output" output={output} maxLines={2} />)

    expect(screen.getByText('run long output')).toBeTruthy()
    expect(screen.getByText(/line 1/)).toBeTruthy()
    expect(screen.queryByText(/line 4/)).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Show 2 more lines' }))
    expect(screen.getByText(/line 4/)).toBeTruthy()
  })
})

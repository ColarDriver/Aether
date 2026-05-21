// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { MarkdownRenderer } from './MarkdownRenderer'

afterEach(cleanup)

describe('MarkdownRenderer', () => {
  it('renders tables as semantic table elements', () => {
    render(<MarkdownRenderer text={'| Tool | Status |\n| --- | --- |\n| shell | blocked |'} />)

    expect(screen.getByRole('table')).toBeTruthy()
    expect(screen.getByRole('columnheader', { name: 'Tool' })).toBeTruthy()
    expect(screen.getByRole('cell', { name: 'blocked' })).toBeTruthy()
  })

  it('renders fenced code with language and syntax spans', () => {
    const fence = String.fromCharCode(96, 96, 96)
    render(<MarkdownRenderer text={fence + 'json\n{"ok": true, "count": 2}\n' + fence} />)

    expect(screen.getByText('json')).toBeTruthy()
    expect(document.querySelector('.syntax-string')?.textContent).toBe('"ok"')
    expect(document.querySelector('.syntax-boolean')?.textContent).toBe('true')
  })

  it('renders inline code and strong text without raw html', () => {
    const tick = String.fromCharCode(96)
    render(<MarkdownRenderer text={'Use ' + tick + 'agent.run' + tick + ' with **care**.'} />)

    expect(screen.getByText('agent.run').className).toContain('markdown-inline-code')
    expect(screen.getByText('care').tagName).toBe('STRONG')
  })

  it('renders partial streaming tables before the separator row arrives', () => {
    render(<MarkdownRenderer text={'| Tool | Status |\n| shell | running |'} />)

    expect(screen.getByRole('table')).toBeTruthy()
    expect(screen.getByRole('cell', { name: 'running' })).toBeTruthy()
  })

  it('renders blockquotes and safe links', () => {
    render(<MarkdownRenderer text={'> Read [docs](https://example.com).\n> Avoid [bad](javascript:alert(1)).'} />)

    expect(document.querySelector('blockquote')?.textContent).toContain('Read docs.')
    expect(screen.getByRole('link', { name: 'docs' }).getAttribute('href')).toBe('https://example.com')
    expect(screen.getByRole('link', { name: 'bad' }).getAttribute('href')).toBe('#')
  })
})

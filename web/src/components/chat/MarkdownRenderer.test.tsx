// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
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
    render(<MarkdownRenderer text={fence + 'typescript\nconst value = {"ok": true, "count": 2}\n// done\n' + fence} />)

    expect(screen.getByText('typescript')).toBeTruthy()
    expect(document.querySelector('.syntax-keyword')?.textContent).toBe('const')
    expect(document.querySelector('.syntax-string')?.textContent).toBe('"ok"')
    expect(document.querySelector('.syntax-boolean')?.textContent).toBe('true')
    expect(document.querySelector('.syntax-comment')?.textContent).toBe('// done')
  })

  it('renders inline code and strong text without raw html', () => {
    const tick = String.fromCharCode(96)
    render(<MarkdownRenderer text={'Use ' + tick + 'agent.run' + tick + ' with **care**.'} />)

    expect(screen.getByText('agent.run').className).toContain('markdown-inline-code')
    expect(screen.getByText('care').tagName).toBe('STRONG')
  })

  it('does not hang on a bare heading marker streamed before its content', () => {
    // Regression: a line like `## ` (heading marker, no content yet) is reported
    // as a block by startsBlock() but rejected by the heading handler, which used
    // to stall the parser's index and peg the main thread. Rendering must return.
    render(<MarkdownRenderer streaming text={'Intro paragraph.\n\n## '} />)
    expect(screen.getByText('Intro paragraph.')).toBeTruthy()
    expect(screen.getByText('##')).toBeTruthy()
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

  it('renders task lists, h4 headings, horizontal rules, and ordered paren lists', () => {
    render(<MarkdownRenderer text={'#### Checklist\n\n- [x] Done\n- [ ] Next\n\n---\n\n1) First\n2) Second'} />)

    expect(screen.getByRole('heading', { level: 4, name: 'Checklist' })).toBeTruthy()
    expect(screen.getAllByRole('checkbox')).toHaveLength(2)
    expect((screen.getAllByRole('checkbox')[0] as HTMLInputElement).checked).toBe(true)
    expect(document.querySelector('.markdown-hr')).toBeTruthy()
    expect(screen.getByText('Second').closest('ol')).toBeTruthy()
  })

  it('renders italic, strike, bare URLs, and paragraph line breaks', () => {
    render(<MarkdownRenderer text={'A *quiet* ~~old~~ link https://example.com\nnext line'} />)

    expect(screen.getByText('quiet').tagName).toBe('EM')
    expect(screen.getByText('old').tagName).toBe('DEL')
    expect(screen.getByRole('link', { name: 'https://example.com' }).getAttribute('href')).toBe('https://example.com')
    expect(document.querySelector('br')).toBeTruthy()
  })

  it('renders markdown images and opens the image preview dialog', () => {
    render(<MarkdownRenderer text={'Here is ![Diagram](https://example.com/diagram.png) and https://example.com/screenshot.webp'} />)

    expect(screen.getByRole('img', { name: 'Diagram' })).toBeTruthy()
    expect(screen.getByRole('img', { name: 'screenshot.webp' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /Diagram/ }))
    expect(screen.getByRole('dialog', { name: 'Diagram' })).toBeTruthy()
    expect(screen.getByText('1 / 2')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Next image' }))
    expect(screen.getByRole('dialog', { name: 'screenshot.webp' })).toBeTruthy()
  })


  it('renders inline and display math without treating prices as formulas', () => {
    render(<MarkdownRenderer text={'Inline $x^2 + y^2$ and slash math \\(a + b\\) and price $5 and $10.\n\n\\[\nE = mc^2\n\\]'} />)

    expect(document.querySelectorAll('.math-renderer-inline')).toHaveLength(2)
    expect(document.querySelector('.math-renderer-display')).toBeTruthy()
    expect(document.body.textContent).toContain('$5 and $10')
    expect(document.body.textContent).not.toContain('slash math (a + b)')
  })

  it('does not render unsafe markdown image sources as images', () => {
    render(<MarkdownRenderer text={'![bad](javascript:alert(1))'} />)

    expect(screen.queryByRole('img')).toBeNull()
    expect(document.querySelector('.markdown-image')).toBeNull()
  })
  it('keeps the streaming caret inside the last markdown block', () => {
    render(<MarkdownRenderer text={'- one\n- two'} streaming />)

    const caret = document.querySelector('.streaming-caret-inline')
    expect(caret).toBeTruthy()
    expect(caret?.closest('li')?.textContent).toContain('two')
  })
})

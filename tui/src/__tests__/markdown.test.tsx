import { describe, expect, it } from 'vitest'
import { render } from 'ink-testing-library'

import { Markdown } from '../lib/markdown.js'

describe('Markdown', () => {
  it('renders empty placeholder for empty text', () => {
    const { lastFrame, unmount } = render(<Markdown text="" />)
    expect(lastFrame()).toBe('...')
    unmount()
  })

  it('renders headings', () => {
    const { lastFrame, unmount } = render(<Markdown text="# Hello\n\nworld" />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('# Hello')
    expect(frame).toContain('world')
    unmount()
  })

  it('renders fenced code blocks with language tag', () => {
    const { lastFrame, unmount } = render(
      <Markdown text={'Some code:\n\n```js\nconst x = 1;\n```'} />
    )
    const frame = lastFrame() ?? ''
    expect(frame).toContain('Some code:')
    expect(frame).toContain('js')
    expect(frame).toContain('const x = 1')
    unmount()
  })

  it('renders bold and italic inline', () => {
    const { lastFrame, unmount } = render(
      <Markdown text={'Use **bold** and *italic* together.'} />
    )
    const frame = lastFrame() ?? ''
    expect(frame).toContain('bold')
    expect(frame).toContain('italic')
    unmount()
  })

  it('renders lists', () => {
    const { lastFrame, unmount } = render(<Markdown text={'- alpha\n- beta\n- gamma'} />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('alpha')
    expect(frame).toContain('beta')
    expect(frame).toContain('gamma')
    unmount()
  })

  it('renders inline code', () => {
    const { lastFrame, unmount } = render(<Markdown text={'use the `npm test` command'} />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('npm test')
    unmount()
  })
})

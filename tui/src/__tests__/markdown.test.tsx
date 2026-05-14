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
    expect(frame).toContain('Hello')
    expect(frame).not.toContain('# Hello')
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

  it('renders bold segments inside non-streaming list items', () => {
    const { lastFrame, unmount } = render(
      <Markdown text={'- **文件编辑 / diff 预览**\n- **shell 执行审批**'} />
    )
    const frame = lastFrame() ?? ''
    expect(frame).toContain('文件编辑 / diff 预览')
    expect(frame).toContain('shell 执行审批')
    expect(frame).not.toContain('**文件编辑 / diff 预览**')
    expect(frame).not.toContain('**shell 执行审批**')
    unmount()
  })

  it('renders nested list items with inline code in non-streaming markdown', () => {
    const { lastFrame, unmount } = render(
      <Markdown text={'- Python 3.12+\n- 依赖里有：\n  - `anthropic`\n  - `httpx`\n  - `prompt_toolkit`'} />
    )
    const frame = lastFrame() ?? ''
    expect(frame).toContain('Python 3.12+')
    expect(frame).toContain('依赖里有：')
    expect(frame).toContain('anthropic')
    expect(frame).toContain('httpx')
    expect(frame).toContain('prompt_toolkit')
    expect(frame).not.toContain('`anthropic`')
    expect(frame).not.toContain('`httpx`')
    expect(frame).not.toContain('`prompt_toolkit`')
    unmount()
  })

  it('renders inline code', () => {
    const { lastFrame, unmount } = render(<Markdown text={'use the `npm test` command'} />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('npm test')
    unmount()
  })

  it('renders markdown tables as table chrome instead of raw pipe syntax', () => {
    const { lastFrame, unmount } = render(
      <Markdown
        text={
          '| 模块 | 说明 |\n| --- | --- |\n| `agents/` | Agent 定义与逻辑 |\n| `gateway/` | Gateway 服务入口 |'
        }
      />
    )
    const frame = lastFrame() ?? ''
    expect(frame).toContain('┌')
    expect(frame).toContain('模块')
    expect(frame).toContain('agents/')
    expect(frame).not.toContain('| --- | --- |')
    unmount()
  })

  it('renders horizontal rules as a full divider instead of raw --- text', () => {
    const { lastFrame, unmount } = render(<Markdown text={'上文\n\n---\n\n下文'} />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('上文')
    expect(frame).toContain('下文')
    expect(frame).not.toContain('\n---\n')
    expect(frame).toMatch(/─{12,}/)
    unmount()
  })

  it('keeps markdown structure while streaming', () => {
    const { lastFrame, unmount } = render(
      <Markdown text={'# Hello\n\n- alpha\n- beta\n\nUse *italics*.'} streaming />
    )
    const frame = lastFrame() ?? ''
    expect(frame).toContain('Hello')
    expect(frame).toContain('alpha')
    expect(frame).toContain('beta')
    expect(frame).toContain('italics')
    unmount()
  })

  it('renders bold segments inside streaming list items', () => {
    const { lastFrame, unmount } = render(
      <Markdown text={'- **智能导航**: 跳转到定义'} streaming />
    )
    const frame = lastFrame() ?? ''
    expect(frame).toContain('智能导航')
    expect(frame).toContain('跳转到定义')
    unmount()
  })

  it('renders tables during streaming instead of leaving raw markdown pipes', () => {
    const { lastFrame, unmount } = render(
      <Markdown
        text={
          '| 模块 | 说明 |\n| --- | --- |\n| `agents/` | Agent 定义与逻辑 |\n| `gateway/` | Gateway 服务入口 |'
        }
        streaming
      />
    )
    const frame = lastFrame() ?? ''
    expect(frame).toContain('┌')
    expect(frame).toContain('模块')
    expect(frame).toContain('gateway/')
    expect(frame).not.toContain('| --- | --- |')
    unmount()
  })

  it('renders horizontal rules during streaming instead of raw --- text', () => {
    const { lastFrame, unmount } = render(<Markdown text={'上文\n\n---\n\n下文'} streaming />)
    const frame = lastFrame() ?? ''
    expect(frame).toContain('上文')
    expect(frame).toContain('下文')
    expect(frame).not.toContain('\n---\n')
    expect(frame).toMatch(/─{12,}/)
    unmount()
  })
})

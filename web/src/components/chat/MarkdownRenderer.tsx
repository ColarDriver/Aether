import type { ReactNode } from 'react'

type Props = {
  text: string
}

export function MarkdownRenderer({ text }: Props) {
  const blocks = splitMarkdownBlocks(text)
  return (
    <div className="markdown-renderer">
      {blocks.map((block, index) => renderBlock(block, index))}
    </div>
  )
}

function renderBlock(block: string, index: number) {
  if (block.startsWith('### ')) return <h3 key={index}>{renderInline(block.slice(4))}</h3>
  if (block.startsWith('## ')) return <h2 key={index}>{renderInline(block.slice(3))}</h2>
  if (block.startsWith('# ')) return <h1 key={index}>{renderInline(block.slice(2))}</h1>
  const fence = String.fromCharCode(96, 96, 96)
  if (block.startsWith(fence)) {
    const { language, code } = parseFence(block)
    return (
      <div className="markdown-code-wrap" key={index}>
        {language ? <div className="markdown-code-header">{language}</div> : null}
        <pre className="markdown-code"><code>{highlightCode(code, language)}</code></pre>
      </div>
    )
  }
  if (isMarkdownTable(block)) {
    return <MarkdownTable block={block} key={index} />
  }
  if (block.split('\n').every((line) => line.trim().startsWith('>'))) {
    return (
      <blockquote key={index}>
        {renderInline(block.split('\n').map((line) => line.replace(/^\s*>\s?/, '')).join('\n'))}
      </blockquote>
    )
  }
  if (block.includes('\n- ') || block.startsWith('- ')) {
    return (
      <ul key={index}>
        {block.split('\n').filter((line) => line.startsWith('- ')).map((line) => (
          <li key={line}>{renderInline(line.slice(2))}</li>
        ))}
      </ul>
    )
  }
  if (/^\d+\. /.test(block)) {
    return (
      <ol key={index}>
        {block.split('\n').filter((line) => /^\d+\. /.test(line)).map((line) => (
          <li key={line}>{renderInline(line.replace(/^\d+\. /, ''))}</li>
        ))}
      </ol>
    )
  }
  return <p key={index}>{renderInline(block)}</p>
}

function splitMarkdownBlocks(text: string): string[] {
  const lines = text.replace(/\r\n/g, '\n').split('\n')
  const blocks: string[] = []
  let current: string[] = []
  let inFence = false
  const fence = String.fromCharCode(96, 96, 96)

  for (const line of lines) {
    if (line.startsWith(fence)) {
      current.push(line)
      inFence = !inFence
      if (!inFence) {
        blocks.push(current.join('\n'))
        current = []
      }
      continue
    }
    if (!inFence && line.trim() === '') {
      if (current.length > 0) {
        blocks.push(current.join('\n'))
        current = []
      }
      continue
    }
    current.push(line)
  }

  if (current.length > 0) blocks.push(current.join('\n'))
  return blocks
}

function parseFence(block: string): { language: string; code: string } {
  const fence = String.fromCharCode(96, 96, 96)
  const lines = block.split('\n')
  const language = lines[0]?.slice(fence.length).trim() ?? ''
  const codeLines = lines.slice(1)
  if (codeLines[codeLines.length - 1]?.startsWith(fence)) codeLines.pop()
  return { language, code: codeLines.join('\n').trimEnd() }
}

function isMarkdownTable(block: string): boolean {
  const lines = block.split('\n').map((line) => line.trim())
  return (
    lines.length >= 2 &&
    lines[0].startsWith('|') &&
    (
      /^\|?[\s:-]+\|[\s|:-]+$/.test(lines[1] ?? '') ||
      lines.every((line) => line.startsWith('|') && line.includes('|'))
    )
  )
}

function MarkdownTable({ block }: { block: string }) {
  const lines = block.split('\n').map((line) => splitTableRow(line))
  const [headers, maybeSeparator, ...rest] = lines
  const hasSeparator = isSeparatorRow(maybeSeparator ?? [])
  const rows = hasSeparator ? rest : [maybeSeparator, ...rest].filter((row): row is string[] => Boolean(row))
  return (
    <div className="markdown-table-wrap">
      <table className="markdown-table">
        <thead>
          <tr>{headers.map((cell) => <th key={cell}>{renderInline(cell)}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.map((cell, cellIndex) => <td key={cellIndex}>{renderInline(cell)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function splitTableRow(line: string): string[] {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((cell) => cell.trim())
}

function isSeparatorRow(row: string[]): boolean {
  return row.length > 0 && row.every((cell) => /^:?-{2,}:?$/.test(cell) || cell === '')
}

function renderInline(text: string): ReactNode[] {
  const parts: ReactNode[] = []
  const tick = String.fromCharCode(96)
  const pattern = new RegExp('(\\[[^\\]]+\\]\\([^)]+\\)|' + tick + '[^' + tick + ']+' + tick + '|\\*\\*[^*]+\\*\\*)', 'g')
  let lastIndex = 0
  let match: RegExpExecArray | null
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) parts.push(text.slice(lastIndex, match.index))
    const value = match[0]
    if (value.startsWith(tick)) {
      parts.push(<code className="markdown-inline-code" key={parts.length}>{value.slice(1, -1)}</code>)
    } else if (value.startsWith('[')) {
      const link = parseLink(value)
      parts.push(
        <a href={safeHref(link.href)} key={parts.length} rel="noreferrer" target="_blank">
          {link.label}
        </a>,
      )
    } else {
      parts.push(<strong key={parts.length}>{value.slice(2, -2)}</strong>)
    }
    lastIndex = match.index + value.length
  }
  if (lastIndex < text.length) parts.push(text.slice(lastIndex))
  return parts
}

function parseLink(value: string): { label: string; href: string } {
  const match = value.match(/^\[([^\]]+)\]\(([^)]+)\)$/)
  return { label: match?.[1] ?? value, href: match?.[2] ?? '#' }
}

function safeHref(href: string): string {
  const trimmed = href.trim()
  if (/^(https?:|mailto:|#|\/(?!\/)|\.\/|\.\.\/)/i.test(trimmed)) return trimmed
  return '#'
}

function highlightCode(code: string, language: string): ReactNode {
  if (!['json', 'jsonc', 'js', 'javascript', 'ts', 'typescript'].includes(language.toLowerCase())) {
    return code
  }
  const pattern = /("(?:\\.|[^"\\])*"|\btrue\b|\bfalse\b|\bnull\b|-?\b\d+(?:\.\d+)?\b)/g
  const parts: ReactNode[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null
  while ((match = pattern.exec(code)) !== null) {
    if (match.index > lastIndex) parts.push(code.slice(lastIndex, match.index))
    const value = match[0]
    const tokenClass = value.startsWith('"')
      ? 'syntax-string'
      : value === 'true' || value === 'false'
        ? 'syntax-boolean'
        : value === 'null'
          ? 'syntax-null'
          : 'syntax-number'
    parts.push(<span className={tokenClass} key={parts.length}>{value}</span>)
    lastIndex = match.index + value.length
  }
  if (lastIndex < code.length) parts.push(code.slice(lastIndex))
  return parts
}

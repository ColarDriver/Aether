import type { ReactNode } from 'react'
import { CodeBlock } from './blocks/CodeBlock'

type Props = {
  text: string
  streaming?: boolean
}

type MarkdownBlock =
  | { type: 'code'; language: string; code: string }
  | { type: 'heading'; level: 1 | 2 | 3 | 4; content: string }
  | { type: 'hr' }
  | { type: 'table'; block: string }
  | { type: 'blockquote'; content: string }
  | { type: 'list'; ordered: boolean; items: ListItem[] }
  | { type: 'paragraph'; content: string }

type ListItem = {
  content: string
  checked?: boolean
}

export function MarkdownRenderer({ text, streaming = false }: Props) {
  const blocks = splitMarkdownBlocks(text)
  return (
    <div className="markdown-renderer">
      {blocks.map((block, index) => renderBlock(block, index, streaming && index === blocks.length - 1))}
      {blocks.length === 0 && streaming ? <span className="streaming-caret" /> : null}
    </div>
  )
}

function renderBlock(block: MarkdownBlock, index: number, streaming = false) {
  const caret = streaming ? <StreamingCaret /> : null
  if (block.type === 'heading') {
    const content = <>{renderInline(block.content)}{caret}</>
    if (block.level === 1) return <h1 key={index}>{content}</h1>
    if (block.level === 2) return <h2 key={index}>{content}</h2>
    if (block.level === 3) return <h3 key={index}>{content}</h3>
    return <h4 key={index}>{content}</h4>
  }
  if (block.type === 'code') return <CodeBlock code={block.code} key={index} language={block.language} />
  if (block.type === 'table') return <MarkdownTable block={block.block} key={index} />
  if (block.type === 'hr') return <hr className="markdown-hr" key={index} />
  if (block.type === 'blockquote') {
    return (
      <blockquote key={index}>
        {renderInline(block.content)}
        {caret}
      </blockquote>
    )
  }
  if (block.type === 'list') {
    const Tag = block.ordered ? 'ol' : 'ul'
    return (
      <Tag className={hasTasks(block.items) ? 'markdown-task-list' : undefined} key={index}>
        {block.items.map((item, itemIndex) => (
          <li className={item.checked != null ? 'markdown-task-item' : undefined} key={itemIndex}>
            {item.checked != null ? <input checked={item.checked} readOnly type="checkbox" /> : null}
            <span>{renderInline(item.content)}{streaming && itemIndex === block.items.length - 1 ? caret : null}</span>
          </li>
        ))}
      </Tag>
    )
  }
  return <p key={index}>{renderInline(block.content)}{caret}</p>
}

function StreamingCaret() {
  return <span aria-hidden="true" className="streaming-caret streaming-caret-inline" />
}

function splitMarkdownBlocks(text: string): MarkdownBlock[] {
  const lines = text.replace(/\r\n/g, '\n').split('\n')
  const fence = String.fromCharCode(96, 96, 96)
  const blocks: MarkdownBlock[] = []
  let index = 0

  while (index < lines.length) {
    const line = lines[index] ?? ''
    if (!line.trim()) {
      index += 1
      continue
    }
    if (line.startsWith(fence)) {
      const language = line.slice(fence.length).trim()
      const codeLines: string[] = []
      index += 1
      while (index < lines.length && !(lines[index] ?? '').startsWith(fence)) {
        codeLines.push(lines[index] ?? '')
        index += 1
      }
      if (index < lines.length) index += 1
      blocks.push({ type: 'code', language, code: codeLines.join('\n').trimEnd() })
      continue
    }

    const heading = line.match(/^(#{1,4})\s+(.+)$/)
    if (heading?.[1] && heading[2]) {
      blocks.push({ type: 'heading', level: heading[1].length as 1 | 2 | 3 | 4, content: heading[2] })
      index += 1
      continue
    }

    if (/^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
      blocks.push({ type: 'hr' })
      index += 1
      continue
    }

    if (line.trim().startsWith('|') && index + 1 < lines.length) {
      const tableLines: string[] = []
      while (index < lines.length && isTableLine(lines[index] ?? '')) {
        tableLines.push(lines[index] ?? '')
        index += 1
      }
      if (isMarkdownTable(tableLines.join('\n'))) {
        blocks.push({ type: 'table', block: tableLines.join('\n') })
      } else {
        blocks.push({ type: 'paragraph', content: tableLines.join('\n') })
      }
      continue
    }

    if (line.trim().startsWith('>')) {
      const quoteLines: string[] = []
      while (index < lines.length && (lines[index] ?? '').trim().startsWith('>')) {
        quoteLines.push((lines[index] ?? '').replace(/^\s*>\s?/, ''))
        index += 1
      }
      blocks.push({ type: 'blockquote', content: quoteLines.join('\n') })
      continue
    }

    const firstListItem = parseListItem(line)
    if (firstListItem) {
      const items: ListItem[] = []
      const ordered = firstListItem.ordered
      while (index < lines.length) {
        const next = parseListItem(lines[index] ?? '')
        if (!next || next.ordered !== ordered) break
        items.push({ content: next.content, checked: next.checked })
        index += 1
      }
      blocks.push({ type: 'list', ordered, items })
      continue
    }

    const paragraphLines: string[] = []
    while (index < lines.length && !startsBlock(lines[index] ?? '', lines[index + 1] ?? '')) {
      paragraphLines.push(lines[index] ?? '')
      index += 1
    }
    if (paragraphLines.length > 0) blocks.push({ type: 'paragraph', content: paragraphLines.join('\n') })
  }

  return blocks
}

function startsBlock(line: string, nextLine: string): boolean {
  if (!line.trim()) return true
  const fence = String.fromCharCode(96, 96, 96)
  return (
    line.startsWith(fence) ||
    /^(#{1,4})\s+/.test(line) ||
    /^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/.test(line) ||
    (line.trim().startsWith('|') && isTableLine(nextLine)) ||
    line.trim().startsWith('>') ||
    Boolean(parseListItem(line))
  )
}

function parseListItem(line: string): (ListItem & { ordered: boolean }) | null {
  const unordered = line.match(/^\s{0,3}[-*+]\s+(\[[ xX]\]\s+)?([\s\S]*)$/)
  if (unordered) {
    return {
      ordered: false,
      ...taskContent(unordered[1], unordered[2] ?? ''),
    }
  }
  const ordered = line.match(/^\s{0,3}\d+[.)]\s+(\[[ xX]\]\s+)?([\s\S]*)$/)
  if (ordered) {
    return {
      ordered: true,
      ...taskContent(ordered[1], ordered[2] ?? ''),
    }
  }
  return null
}

function taskContent(marker: string | undefined, content: string): ListItem {
  if (!marker) return { content }
  return { content, checked: /\[[xX]\]/.test(marker) }
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

function isTableLine(line: string): boolean {
  const trimmed = line.trim()
  return trimmed.startsWith('|') && trimmed.includes('|')
}

function hasTasks(items: ListItem[]): boolean {
  return items.some((item) => item.checked != null)
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
  const pattern = new RegExp(
    '(' +
      '\\[[^\\]]+\\]\\([^)]+\\)' +
      '|' + tick + '[^' + tick + ']+' + tick +
      '|\\*\\*[^*]+\\*\\*' +
      '|~~[^~]+~~' +
      '|(?<!\\*)\\*[^*]+\\*(?!\\*)' +
      '|\\bhttps?:\\/\\/[^\\s<>)\\]]+' +
      '|\\n' +
    ')',
    'g',
  )
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
    } else if (value.startsWith('~~')) {
      parts.push(<del key={parts.length}>{renderInline(value.slice(2, -2))}</del>)
    } else if (value.startsWith('*') && !value.startsWith('**')) {
      parts.push(<em key={parts.length}>{renderInline(value.slice(1, -1))}</em>)
    } else if (/^https?:\/\//i.test(value)) {
      parts.push(
        <a href={safeHref(value)} key={parts.length} rel="noreferrer" target="_blank">
          {value}
        </a>,
      )
    } else if (value === '\n') {
      parts.push(<br key={parts.length} />)
    } else {
      parts.push(<strong key={parts.length}>{renderInline(value.slice(2, -2))}</strong>)
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

import { Box, Text } from 'ink'
import { marked, type Token, type Tokens } from 'marked'
import stringWidth from 'string-width'
import { useMemo, type ReactElement, type ReactNode } from 'react'

import {
  parseMarkdownLite,
  type InlineSegment,
  type MarkdownBlock
} from './markdownLite.js'
import { theme } from './theme.js'

let highlightCache: ((code: string, language?: string) => string) | null = null
function getHighlight(): (code: string, language?: string) => string {
  if (highlightCache) {
    return highlightCache
  }
  try {
    // cli-highlight is a CommonJS module — wrap in a try/catch so a
    // missing/incompatible install never breaks the chat view. Falls back
    // to identity when unavailable.
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require('cli-highlight')
    const highlight = (mod.highlight ?? mod.default?.highlight) as
      | ((code: string, opts?: { language?: string }) => string)
      | undefined
    if (typeof highlight === 'function') {
      highlightCache = (code, language) =>
        highlight(code, language ? { language } : undefined)
      return highlightCache
    }
  } catch {
    // ignore — fall through to identity
  }
  highlightCache = (code) => code
  return highlightCache
}

const TOKEN_CACHE = new Map<string, Token[]>()
const TOKEN_CACHE_LIMIT = 64

function tokenise(text: string): Token[] {
  const cached = TOKEN_CACHE.get(text)
  if (cached) {
    return cached
  }
  const tokens = marked.lexer(text) as Token[]
  if (TOKEN_CACHE.size >= TOKEN_CACHE_LIMIT) {
    // Drop the oldest entry — Map preserves insertion order.
    const first = TOKEN_CACHE.keys().next().value
    if (first !== undefined) {
      TOKEN_CACHE.delete(first)
    }
  }
  TOKEN_CACHE.set(text, tokens)
  return tokens
}

export interface MarkdownProps {
  text: string
  streaming?: boolean
}

/**
 * Streaming-friendly markdown renderer for assistant messages. Uses
 * `marked.lexer` to get tokens and walks them into Ink primitives. Code
 * blocks pass through `cli-highlight` when the package is available so
 * syntax colouring matches what users see in their terminal Markdown viewer.
 *
 * The implementation is deliberately limited to the subset Aether needs:
 * paragraphs, headings, bold/italic/inline code/strikethrough, fenced code
 * blocks, blockquotes, lists. Tables fall back to a row-per-row plain text
 * rendering — Ink does not have a great table primitive.
 */
export function Markdown({ text, streaming = false }: MarkdownProps): ReactElement {
  const tokens = useMemo(
    () => (streaming ? [] : tokenise(text || '')),
    [streaming, text]
  )
  const liteBlocks = useMemo(
    () => (streaming ? parseMarkdownLite(text || '') : []),
    [streaming, text]
  )

  if (streaming) {
    if (liteBlocks.length === 0) {
      return <Text dimColor>...</Text>
    }
    return (
      <Box flexDirection="column">
        {liteBlocks.map((block, idx) => renderLiteBlock(block, idx))}
      </Box>
    )
  }

  if (tokens.length === 0) {
    return <Text dimColor>...</Text>
  }
  return (
    <Box flexDirection="column">
      {tokens.map((token, idx) => renderToken(token, idx))}
    </Box>
  )
}

function renderLiteBlock(block: MarkdownBlock, key: number): ReactElement {
  switch (block.kind) {
    case 'heading':
      return (
        <Box key={key} marginTop={key === 0 ? 0 : 1}>
          <Text bold underline={block.depth <= 2}>
            {renderLiteInline(block.segments)}
          </Text>
        </Box>
      )
    case 'paragraph':
      return (
        <Box key={key} marginTop={key === 0 ? 0 : 1}>
          <Text>{renderLiteInline(block.segments)}</Text>
        </Box>
      )
    case 'list':
      return (
        <Box key={key} flexDirection="column" marginLeft={2} marginTop={key === 0 ? 0 : 1}>
          {block.items.map((item, idx) => (
            <Box key={idx}>
              <Text dimColor>{block.ordered ? `${block.start + idx}.` : '•'} </Text>
              <Box flexDirection="column">
                <Text>{renderLiteInline(item)}</Text>
              </Box>
            </Box>
          ))}
        </Box>
      )
    case 'hr':
      return renderHorizontalRule(key)
    case 'table':
      return renderLiteTable(block, key)
    case 'code':
      return renderLiteCodeBlock(block, key)
  }
}

function renderLiteCodeBlock(
  block: Extract<MarkdownBlock, { kind: 'code' }>,
  key: number
): ReactElement {
  const highlighted = getHighlight()(block.text, block.language ?? undefined)
  return (
    <Box
      key={key}
      borderStyle="single"
      {...(theme.color('border') ? { borderColor: theme.color('border')! } : {})}
      paddingX={1}
      flexDirection="column"
      marginTop={key === 0 ? 0 : 1}
    >
      {block.language ? <Text dimColor>{block.language}</Text> : null}
      <Text>{highlighted}</Text>
    </Box>
  )
}

function renderLiteTable(
  block: Extract<MarkdownBlock, { kind: 'table' }>,
  key: number
): ReactElement {
  const headers = block.headers.map((segments, idx) => ({
    key: `h_${idx}`,
    display: flattenLiteInline(segments)
  }))
  const rows = block.rows.map((row, rowIdx) =>
    row.map((segments, cellIdx) => ({
      key: `r_${rowIdx}_${cellIdx}`,
      display: flattenLiteInline(segments)
    }))
  )
  return renderTableGrid(headers, rows, key)
}

function renderLiteInline(segments: InlineSegment[]): ReactNode {
  return segments.map((segment, index) => {
    switch (segment.kind) {
      case 'text':
        return segment.text
      case 'bold':
        return (
          <Text key={index} bold>
            {segment.text}
          </Text>
        )
      case 'italic':
        return (
          <Text key={index} italic>
            {segment.text}
          </Text>
        )
      case 'code':
        return (
          <Text
            key={index}
            {...(theme.color('dim') ? { color: theme.color('dim')! } : {})}
          >
            {segment.text}
          </Text>
        )
    }
  })
}

interface RenderOpts {
  // True when this token is being rendered inside a list_item. List items
  // already produce one visual row per child, so we suppress the inter-block
  // marginTop that paragraphs/lists/headings normally use — otherwise a
  // nested sub-list shows a blank line between the parent text and the
  // first sub-bullet (open-claude-code packs these tightly in `formatToken`).
  nested?: boolean
}

function renderToken(token: Token, key: number, opts: RenderOpts = {}): ReactElement | null {
  switch (token.type) {
    case 'heading':
      return renderHeading(token as Tokens.Heading, key, opts)
    case 'paragraph':
      return renderParagraph(token as Tokens.Paragraph, key, opts)
    case 'code':
      return renderCodeBlock(token as Tokens.Code, key, opts)
    case 'blockquote':
      return renderBlockquote(token as Tokens.Blockquote, key, opts)
    case 'list':
      return renderList(token as Tokens.List, key, opts)
    case 'hr':
      return renderHorizontalRule(key)
    case 'space':
      return null
    case 'text':
      return (
        <Text key={key}>
          {renderInlineNodes((token as Tokens.Text).tokens ?? [
            { type: 'text', text: (token as Tokens.Text).text } as Tokens.Text
          ])}
        </Text>
      )
    case 'table':
      return renderTable(token as Tokens.Table, key, opts)
    case 'html':
      return (
        <Text key={key} dimColor>
          {(token as Tokens.HTML).text}
        </Text>
      )
    default:
      return (
        <Text key={key}>{(token as { raw?: string }).raw ?? ''}</Text>
      )
  }
}

function blockMarginTop(key: number, opts: RenderOpts): 0 | 1 {
  if (opts.nested) {
    return 0
  }
  return key === 0 ? 0 : 1
}

function renderHeading(token: Tokens.Heading, key: number, opts: RenderOpts = {}): ReactElement {
  return (
    <Box key={key} marginTop={blockMarginTop(key, opts)}>
      <Text bold underline={token.depth <= 2}>
        {renderInlineNodes(token.tokens)}
      </Text>
    </Box>
  )
}

function renderParagraph(token: Tokens.Paragraph, key: number, opts: RenderOpts = {}): ReactElement {
  return (
    <Box key={key} marginTop={blockMarginTop(key, opts)}>
      <Text>{renderInlineNodes(token.tokens)}</Text>
    </Box>
  )
}

function renderCodeBlock(token: Tokens.Code, key: number, opts: RenderOpts = {}): ReactElement {
  const language = token.lang || ''
  const highlighted = getHighlight()(token.text, language)
  return (
    <Box
      key={key}
      borderStyle="single"
      {...(theme.color('border') ? { borderColor: theme.color('border')! } : {})}
      paddingX={1}
      flexDirection="column"
      marginTop={blockMarginTop(key, opts)}
    >
      {language ? (
        <Text dimColor>{language}</Text>
      ) : null}
      <Text>{highlighted}</Text>
    </Box>
  )
}

function renderBlockquote(token: Tokens.Blockquote, key: number, opts: RenderOpts = {}): ReactElement {
  return (
    <Box key={key} flexDirection="column" marginLeft={2} marginTop={blockMarginTop(key, opts)}>
      {(token.tokens ?? []).map((child, idx) => {
        const inner = renderToken(child, idx, { nested: true })
        return inner === null ? null : (
          <Box key={idx} flexDirection="row">
            <Text dimColor>│ </Text>
            <Box flexDirection="column">{inner}</Box>
          </Box>
        )
      })}
    </Box>
  )
}

function renderList(token: Tokens.List, key: number, opts: RenderOpts = {}): ReactElement {
  return (
    <Box key={key} flexDirection="column" marginLeft={2} marginTop={blockMarginTop(key, opts)}>
      {token.items.map((item, idx) => {
        const start = typeof token.start === 'number' ? token.start : 1
        const marker = token.ordered ? `${start + idx}.` : '•'
        return (
          <Box key={idx} alignItems="flex-start">
            <Text dimColor>{marker} </Text>
            <Box flexDirection="column">
              {(item.tokens ?? []).map((child, childIdx) => {
                const rendered = renderToken(child, childIdx, { nested: true })
                return rendered === null ? null : rendered
              })}
            </Box>
          </Box>
        )
      })}
    </Box>
  )
}

function renderTable(token: Tokens.Table, key: number, opts: RenderOpts = {}): ReactElement {
  const headers = token.header.map((cell, idx) => ({
    key: `h_${idx}`,
    display: inlineDisplay(cell.tokens)
  }))
  const rows = token.rows.map((row, rowIdx) =>
    row.map((cell, cellIdx) => ({
      key: `r_${rowIdx}_${cellIdx}`,
      display: inlineDisplay(cell.tokens)
    }))
  )
  return renderTableGrid(headers, rows, key, token.align ?? [], opts)
}

function renderInlineNodes(tokens: Token[] | undefined): ReactNode {
  if (!tokens || tokens.length === 0) {
    return null
  }
  const out: ReactNode[] = []
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i]
    if (!token) {
      continue
    }
    out.push(<InlineToken key={i} token={token} />)
  }
  return out
}

function InlineToken({ token }: { token: Token }): ReactElement {
  switch (token.type) {
    case 'text': {
      const textToken = token as Tokens.Text
      const nested = textToken.tokens ?? []
      if (nested.length > 0) {
        return <>{renderInlineNodes(nested)}</>
      }
      return <>{textToken.text}</>
    }
    case 'strong':
      return (
        <Text bold>{renderInlineNodes((token as Tokens.Strong).tokens)}</Text>
      )
    case 'em':
      return (
        <Text italic>{renderInlineNodes((token as Tokens.Em).tokens)}</Text>
      )
    case 'codespan':
      return (
        <Text {...(theme.color('dim') ? { color: theme.color('dim')! } : {})}>
          {(token as Tokens.Codespan).text}
        </Text>
      )
    case 'del':
      return (
        <Text strikethrough>
          {renderInlineNodes((token as Tokens.Del).tokens)}
        </Text>
      )
    case 'br':
      return <Text>{'\n'}</Text>
    case 'link':
      return (
        <Text {...(theme.color('info') ? { color: theme.color('info')! } : {})}>
          {(token as Tokens.Link).text || (token as Tokens.Link).href}
        </Text>
      )
    case 'image':
      return (
        <Text dimColor>
          [image: {(token as Tokens.Image).text || (token as Tokens.Image).href}]
        </Text>
      )
    case 'html':
      return (
        <Text dimColor>{(token as Tokens.HTML).text}</Text>
      )
    case 'escape':
      return <Text>{(token as Tokens.Escape).text}</Text>
    default:
      return <Text>{(token as { raw?: string }).raw ?? ''}</Text>
  }
}

interface TableRenderCell {
  key: string
  display: string
}

function renderTableGrid(
  headers: TableRenderCell[],
  rows: TableRenderCell[][],
  key: number,
  alignments: Array<'left' | 'center' | 'right' | null> = [],
  opts: RenderOpts = {}
): ReactElement {
  const layout = computeTableLayout(headers, rows, availableMarkdownWidth())
  const borderProps = theme.colorProps('dim')

  return (
    <Box key={key} flexDirection="column" marginTop={blockMarginTop(key, opts)}>
      <Text {...borderProps}>{tableBorder('┌', '┬', '┐', layout.widths)}</Text>
      <TableContentRow
        cells={headers}
        widths={layout.widths}
        hardWrap={layout.hardWrap}
        alignments={alignments}
        header
      />
      <Text {...borderProps}>{tableBorder('├', '┼', '┤', layout.widths)}</Text>
      {rows.map((row, idx) => (
        <TableContentRow
          key={idx}
          cells={row}
          widths={layout.widths}
          hardWrap={layout.hardWrap}
          alignments={alignments}
        />
      ))}
      <Text {...borderProps}>{tableBorder('└', '┴', '┘', layout.widths)}</Text>
    </Box>
  )
}

function TableContentRow({
  cells,
  widths,
  hardWrap,
  alignments,
  header = false
}: {
  cells: TableRenderCell[]
  widths: number[]
  hardWrap: boolean
  alignments: Array<'left' | 'center' | 'right' | null>
  header?: boolean
}): ReactElement {
  const borderProps = theme.colorProps('dim')
  const wrappedCells = cells.map((cell, idx) =>
    wrapPlainText(cell.display, Math.max(1, widths[idx] ?? 1), hardWrap)
  )
  const rowHeight = Math.max(1, ...wrappedCells.map((lines) => lines.length))

  return (
    <Box flexDirection="column">
      {Array.from({ length: rowHeight }, (_, lineIdx) => (
        <Text key={lineIdx}>
          <Text {...borderProps}>│ </Text>
          {cells.map((cell, idx) => {
            const width = widths[idx] ?? 1
            const line = wrappedCells[idx]?.[lineIdx] ?? ''
            const align = header ? 'center' : (alignments[idx] ?? 'left')
            const padded = padAligned(line, width, align)
            return (
              <Text key={cell.key + '_' + lineIdx}>
                <Text bold={header}>{padded}</Text>
                <Text {...borderProps}>
                  {idx === cells.length - 1 ? ' │' : ' │ '}
                </Text>
              </Text>
            )
          })}
        </Text>
      ))}
    </Box>
  )
}

function tableBorder(left: string, join: string, right: string, widths: number[]): string {
  return (
    left +
    widths.map((width) => '─'.repeat(Math.max(1, width + 2))).join(join) +
    right
  )
}

function flattenLiteInline(segments: InlineSegment[]): string {
  return segments.map((segment) => segment.text).join('')
}

function renderHorizontalRule(key: number): ReactElement {
  return (
    <Box key={key} marginTop={key === 0 ? 0 : 1}>
      <Text {...theme.colorProps('dim')}>{'─'.repeat(availableMarkdownWidth())}</Text>
    </Box>
  )
}

function inlineDisplay(tokens: Token[] | undefined): string {
  if (!tokens || tokens.length === 0) {
    return ''
  }
  let out = ''
  for (const token of tokens) {
    if (!token) {
      continue
    }
    switch (token.type) {
      case 'text':
      case 'escape':
        out += (token as Tokens.Text | Tokens.Escape).text
        break
      case 'strong':
      case 'em':
      case 'del':
        out += inlineDisplay((token as Tokens.Strong | Tokens.Em | Tokens.Del).tokens)
        break
      case 'codespan':
        out += (token as Tokens.Codespan).text
        break
      case 'link':
        out += (token as Tokens.Link).text || (token as Tokens.Link).href
        break
      case 'image':
        out += (token as Tokens.Image).text || (token as Tokens.Image).href
        break
      case 'html':
        out += (token as Tokens.HTML).text
        break
      case 'br':
        out += ' '
        break
      default:
        out += (token as { raw?: string }).raw ?? ''
        break
    }
  }
  return out.replace(/\s+/g, ' ').trim()
}

function availableMarkdownWidth(): number {
  const columns = process.stdout?.columns
  if (!Number.isFinite(columns) || !columns) {
    return 72
  }
  return Math.max(12, columns - 8)
}

function longestWordWidth(value: string): number {
  const words = value.split(/\s+/).filter(Boolean)
  if (words.length === 0) {
    return 3
  }
  return Math.max(...words.map((word) => stringWidth(word)), 3)
}

function computeTableLayout(
  headers: TableRenderCell[],
  rows: TableRenderCell[][],
  maxWidth: number
): { widths: number[]; hardWrap: boolean } {
  const columns = headers.length
  const minWidths = headers.map((header, idx) =>
    Math.max(
      longestWordWidth(header.display),
      ...rows.map((row) => longestWordWidth(row[idx]?.display ?? ''))
    )
  )
  const idealWidths = headers.map((header, idx) =>
    Math.max(
      3,
      stringWidth(header.display),
      ...rows.map((row) => stringWidth(row[idx]?.display ?? ''))
    )
  )
  const borderOverhead = 1 + columns * 3
  const available = Math.max(columns * 3, maxWidth - borderOverhead)
  const totalMin = minWidths.reduce((sum, width) => sum + width, 0)
  const totalIdeal = idealWidths.reduce((sum, width) => sum + width, 0)

  if (totalIdeal <= available) {
    return { widths: idealWidths, hardWrap: false }
  }

  if (totalMin <= available) {
    const extra = available - totalMin
    const overflow = idealWidths.map((ideal, idx) => ideal - minWidths[idx]!)
    const totalOverflow = overflow.reduce((sum, width) => sum + width, 0)
    const widths = minWidths.map((min, idx) => {
      if (totalOverflow <= 0) {
        return min
      }
      return min + Math.floor((overflow[idx]! / totalOverflow) * extra)
    })
    let remaining = available - widths.reduce((sum, width) => sum + width, 0)
    for (let idx = 0; remaining > 0 && idx < widths.length; idx += 1, remaining -= 1) {
      widths[idx] = (widths[idx] ?? 0) + 1
    }
    return { widths, hardWrap: false }
  }

  const scale = available / totalMin
  const widths = minWidths.map((width) => Math.max(3, Math.floor(width * scale)))
  let remaining = available - widths.reduce((sum, width) => sum + width, 0)
  for (let idx = 0; remaining > 0 && idx < widths.length; idx += 1, remaining -= 1) {
    widths[idx] = (widths[idx] ?? 0) + 1
  }
  return { widths, hardWrap: true }
}

function padAligned(
  text: string,
  width: number,
  align: 'left' | 'center' | 'right' | null
): string {
  const gap = Math.max(0, width - stringWidth(text))
  if (align === 'right') {
    return ' '.repeat(gap) + text
  }
  if (align === 'center') {
    const left = Math.floor(gap / 2)
    const right = gap - left
    return ' '.repeat(left) + text + ' '.repeat(right)
  }
  return text + ' '.repeat(gap)
}

function wrapPlainText(text: string, width: number, hardWrap: boolean): string[] {
  const source = text.trimEnd()
  if (!source) {
    return ['']
  }
  return source.split('\n').flatMap((line) => wrapPlainLine(line, width, hardWrap))
}

function wrapPlainLine(line: string, width: number, hardWrap: boolean): string[] {
  if (stringWidth(line) <= width) {
    return [line]
  }
  const lines: string[] = []
  let remaining = line
  while (remaining && stringWidth(remaining) > width) {
    const breakIndex = findWrapIndex(remaining, width, hardWrap)
    const head = remaining.slice(0, breakIndex).trimEnd()
    lines.push(head || remaining.slice(0, breakIndex))
    remaining = remaining.slice(breakIndex)
    if (!hardWrap) {
      remaining = remaining.replace(/^\s+/, '')
    }
  }
  if (remaining) {
    lines.push(remaining)
  }
  return lines.length > 0 ? lines : ['']
}

function findWrapIndex(text: string, width: number, hardWrap: boolean): number {
  let total = 0
  let lastWhitespace = -1
  for (let idx = 0; idx < text.length; idx++) {
    const char = text[idx] ?? ''
    total += stringWidth(char)
    if (/\s/.test(char)) {
      lastWhitespace = idx
    }
    if (total > width) {
      if (!hardWrap && lastWhitespace > 0) {
        return lastWhitespace + 1
      }
      return Math.max(1, idx)
    }
  }
  return text.length
}

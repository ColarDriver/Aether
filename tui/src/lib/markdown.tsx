import { Box, Text } from 'ink'
import { marked, type Token, type Tokens } from 'marked'
import { useMemo, type ReactElement, type ReactNode } from 'react'

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
export function Markdown({ text }: MarkdownProps): ReactElement {
  const tokens = useMemo(() => tokenise(text || ''), [text])
  if (tokens.length === 0) {
    return <Text dimColor>...</Text>
  }
  return (
    <Box flexDirection="column">
      {tokens.map((token, idx) => renderToken(token, idx))}
    </Box>
  )
}

function renderToken(token: Token, key: number): ReactElement | null {
  switch (token.type) {
    case 'heading':
      return renderHeading(token as Tokens.Heading, key)
    case 'paragraph':
      return renderParagraph(token as Tokens.Paragraph, key)
    case 'code':
      return renderCodeBlock(token as Tokens.Code, key)
    case 'blockquote':
      return renderBlockquote(token as Tokens.Blockquote, key)
    case 'list':
      return renderList(token as Tokens.List, key)
    case 'hr':
      return (
        <Box key={key}>
          <Text dimColor>──────────────</Text>
        </Box>
      )
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
      return renderTable(token as Tokens.Table, key)
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

function renderHeading(token: Tokens.Heading, key: number): ReactElement {
  const colorProps = theme.colorProps('brand')
  return (
    <Box key={key} marginTop={key === 0 ? 0 : 1}>
      <Text bold {...colorProps}>
        {'#'.repeat(token.depth)} {renderInlineNodes(token.tokens)}
      </Text>
    </Box>
  )
}

function renderParagraph(token: Tokens.Paragraph, key: number): ReactElement {
  return (
    <Text key={key}>{renderInlineNodes(token.tokens)}</Text>
  )
}

function renderCodeBlock(token: Tokens.Code, key: number): ReactElement {
  const language = token.lang || ''
  const highlighted = getHighlight()(token.text, language)
  return (
    <Box
      key={key}
      borderStyle="single"
      {...(theme.color('border') ? { borderColor: theme.color('border')! } : {})}
      paddingX={1}
      flexDirection="column"
      marginTop={key === 0 ? 0 : 1}
    >
      {language ? (
        <Text dimColor>{language}</Text>
      ) : null}
      <Text>{highlighted}</Text>
    </Box>
  )
}

function renderBlockquote(token: Tokens.Blockquote, key: number): ReactElement {
  return (
    <Box key={key} flexDirection="column" marginLeft={2}>
      {(token.tokens ?? []).map((child, idx) => {
        const inner = renderToken(child, idx)
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

function renderList(token: Tokens.List, key: number): ReactElement {
  return (
    <Box key={key} flexDirection="column" marginLeft={1}>
      {token.items.map((item, idx) => {
        const start = typeof token.start === 'number' ? token.start : 1
        const marker = token.ordered ? `${start + idx}.` : '•'
        return (
          <Box key={idx}>
            <Text dimColor>{marker} </Text>
            <Box flexDirection="column">
              <Text>{renderInlineNodes(item.tokens ?? [])}</Text>
            </Box>
          </Box>
        )
      })}
    </Box>
  )
}

function renderTable(token: Tokens.Table, key: number): ReactElement {
  const headers = token.header.map((cell) => cellText(cell.tokens))
  const rows = token.rows.map((row) => row.map((cell) => cellText(cell.tokens)))
  const widths = headers.map((header, idx) =>
    Math.max(header.length, ...rows.map((row) => row[idx]?.length ?? 0))
  )
  const formatRow = (cells: string[]): string =>
    cells.map((cell, idx) => cell.padEnd(widths[idx] ?? cell.length)).join('  ')
  return (
    <Box key={key} flexDirection="column" marginTop={key === 0 ? 0 : 1}>
      <Text bold>{formatRow(headers)}</Text>
      <Text dimColor>{widths.map((w) => '─'.repeat(w)).join('  ')}</Text>
      {rows.map((row, idx) => (
        <Text key={idx}>{formatRow(row)}</Text>
      ))}
    </Box>
  )
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
    case 'text':
      return <Text>{(token as Tokens.Text).text}</Text>
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
          `{(token as Tokens.Codespan).text}`
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

function cellText(tokens: Token[] | undefined): string {
  if (!tokens) {
    return ''
  }
  let out = ''
  for (const token of tokens) {
    if ('text' in token && typeof (token as { text?: unknown }).text === 'string') {
      out += (token as { text: string }).text
    } else if ('tokens' in token && (token as { tokens?: unknown }).tokens) {
      out += cellText((token as { tokens?: Token[] }).tokens)
    }
  }
  return out
}

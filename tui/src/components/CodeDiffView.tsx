import { Box, Text } from 'ink'
import { memo, useMemo, type ReactElement } from 'react'

import {
  DEFAULT_FOLD_THRESHOLD,
  type NumberedDiffLine,
  numberedDiffLines,
  parseUnifiedDiff
} from '../lib/diffRender.js'
import { theme } from '../lib/theme.js'

export interface CodeDiffViewProps {
  diff: string
  expanded?: boolean
  width?: number
  /**
   * Cap on payload rows shown when not expanded. Mirrors the unified-diff
   * fold semantics in DiffView — keeps a single ● Update row from
   * dominating the transcript on a 1000-line refactor.
   */
  foldThreshold?: number
}

/**
 * Claude-Code-style post-execution diff renderer.
 *
 * Compared with `DiffView` (used by the permission overlay):
 *
 * * No `@@` hunk headers — Claude Code shows continuous numbered lines
 *   and lets the line numbers themselves communicate structure.
 * * Each row has a 4-char right-aligned line-number gutter, then a
 *   single-char marker column (`+`, `−`, or blank), then the content
 *   with the unified-diff prefix character stripped.
 * * Add/delete rows get dark red/green backgrounds while the code text keeps
 *   lightweight syntax coloring, matching Claude/Codex-style transcript diffs.
 */
export function CodeDiffView({
  diff,
  expanded = true,
  width,
  foldThreshold = DEFAULT_FOLD_THRESHOLD
}: CodeDiffViewProps): ReactElement {
  const parsed = useMemo(() => parseUnifiedDiff(diff), [diff])
  const rows = useMemo(() => numberedDiffLines(parsed), [parsed])
  const visibleRows = useMemo(
    () => (expanded ? rows : rows.slice(0, foldThreshold)),
    [expanded, foldThreshold, rows]
  )
  const hiddenCount = rows.length - visibleRows.length
  const gutterWidth = Math.max(
    3,
    String(rows[rows.length - 1]?.lineNumber ?? 0).length + 1
  )
  const rowWidth = normaliseDiffWidth(width)
  const codeWidth = Math.max(1, rowWidth - gutterWidth - 2)
  return (
    <Box flexDirection="column" width={rowWidth}>
      {visibleRows.map((row, idx) => {
        const rowBg = backgroundFor(row.kind)
        const markerProps = markerPropsFor(row.kind)
        const lineLabel =
          row.lineNumber === null ? '' : String(row.lineNumber)
        return (
          <Box
            key={idx}
            width={rowWidth}
            overflow="hidden"
            {...(rowBg ? { backgroundColor: rowBg } : {})}
          >
            <Box width={gutterWidth} flexShrink={0}>
              <Text {...markerProps}>{lineLabel.padStart(gutterWidth - 1, ' ') + ' '}</Text>
            </Box>
            <Box width={2} flexShrink={0}>
              <Text {...markerProps}>{markerFor(row.kind)}</Text>
            </Box>
            <Box flexDirection="row" width={codeWidth} overflow="hidden">
              <HighlightedCode row={row} />
            </Box>
          </Box>
        )
      })}
      {hiddenCount > 0 ? (
        <Box marginTop={1}>
          <Text dimColor>
            ({hiddenCount} more line{hiddenCount === 1 ? '' : 's'} · press [E] to expand)
          </Text>
        </Box>
      ) : null}
    </Box>
  )
}

function normaliseDiffWidth(width: number | undefined): number {
  if (typeof width === 'number' && Number.isFinite(width)) {
    return Math.max(40, Math.floor(width))
  }
  const columns = process.stdout?.columns
  // Fill the app content width, not the whole terminal: App owns one column of
  // horizontal padding on each side. Keep a concrete width instead of `100%`;
  // Ink percentage widths inside nested boxes can overrun and create broken
  // background bands.
  return Math.max(40, (typeof columns === 'number' && columns > 0 ? columns : 100) - 2)
}

function markerPropsFor(kind: NumberedDiffLine['kind']): {
  color?: string
  dimColor?: boolean
} {
  if (!theme.isColorEnabled()) {
    return kind === 'context' || kind === 'no-newline' ? { dimColor: true } : {}
  }
  switch (kind) {
    case 'addition':
      return { color: '#22C55E' }
    case 'deletion':
      return { color: '#FB7185' }
    case 'no-newline':
      return { dimColor: true }
    default:
      return { dimColor: true }
  }
}

function backgroundFor(kind: NumberedDiffLine['kind']): string | undefined {
  if (!theme.isColorEnabled()) {
    return undefined
  }
  switch (kind) {
    case 'addition':
      return '#052E16'
    case 'deletion':
      return '#450A0A'
    default:
      return undefined
  }
}

function markerFor(kind: NumberedDiffLine['kind']): string {
  switch (kind) {
    case 'addition':
      return '+ '
    case 'deletion':
      return '- '
    case 'no-newline':
      return '  '
    default:
      return '  '
  }
}

const HighlightedCode = memo(function HighlightedCode({
  row
}: {
  row: NumberedDiffLine
}): ReactElement {
  const pieces = useMemo(() => tokenizeCode(row.text || ' '), [row.text])
  if (row.kind === 'no-newline') {
    return <Text dimColor>{row.text || ' '}</Text>
  }
  return (
    <Text wrap="truncate-end">
      {pieces.map((piece, index) => {
        const props = tokenProps(piece.kind, row.kind)
        return (
          <Text key={index} {...props}>
            {piece.text}
          </Text>
        )
      })}
    </Text>
  )
})

type TokenKind =
  | 'plain'
  | 'keyword'
  | 'string'
  | 'number'
  | 'comment'
  | 'function'
  | 'type'
  | 'builtin'
  | 'constant'
  | 'property'
  | 'operator'
  | 'punctuation'

interface CodeToken {
  kind: TokenKind
  text: string
}

const KEYWORDS = new Set([
  'and',
  'as',
  'async',
  'await',
  'break',
  'case',
  'catch',
  'class',
  'const',
  'continue',
  'def',
  'elif',
  'else',
  'except',
  'export',
  'false',
  'finally',
  'for',
  'from',
  'function',
  'if',
  'import',
  'in',
  'interface',
  'let',
  'new',
  'none',
  'null',
  'pass',
  'return',
  'switch',
  'true',
  'try',
  'type',
  'undefined',
  'while'
])

const CONSTANTS = new Set(['false', 'none', 'null', 'true', 'undefined'])

const BUILTINS = new Set([
  'bool',
  'dict',
  'float',
  'int',
  'input',
  'json',
  'len',
  'list',
  'max',
  'min',
  'print',
  'range',
  'set',
  'str',
  'sum',
  'tuple'
])

const TYPE_IDENTIFIERS = new Set([
  'Any',
  'Array',
  'Boolean',
  'Callable',
  'Dict',
  'Iterable',
  'List',
  'Mapping',
  'None',
  'Number',
  'Object',
  'Optional',
  'Path',
  'Record',
  'Set',
  'String',
  'Tuple',
  'Union',
  'bool',
  'dict',
  'float',
  'int',
  'list',
  'set',
  'str',
  'tuple'
])

function tokenizeCode(input: string): CodeToken[] {
  const tokens: CodeToken[] = []
  let index = 0
  while (index < input.length) {
    const char = input[index] ?? ''
    const next = input[index + 1] ?? ''
    if (/\s/.test(char)) {
      const start = index
      while (index < input.length && /\s/.test(input[index] ?? '')) {
        index += 1
      }
      tokens.push({ kind: 'plain', text: input.slice(start, index) })
      continue
    }
    if (char === '#' || (char === '/' && next === '/')) {
      tokens.push({ kind: 'comment', text: input.slice(index) })
      break
    }
    if (char === '"' || char === "'" || char === '`') {
      const end = readQuoted(input, index, char)
      tokens.push({ kind: 'string', text: input.slice(index, end) })
      index = end
      continue
    }
    if (/[0-9]/.test(char)) {
      const match = /^[0-9][0-9A-Fa-f_xX.bBoO]*/.exec(input.slice(index))
      const text = match?.[0] ?? char
      tokens.push({ kind: 'number', text })
      index += text.length
      continue
    }
    if (/[A-Za-z_]/.test(char)) {
      const match = /^[A-Za-z_][A-Za-z0-9_]*/.exec(input.slice(index))
      const text = match?.[0] ?? char
      const prev = previousNonWhitespace(input, index)
      tokens.push({ kind: identifierKind(input, index, text, prev, tokens), text })
      index += text.length
      continue
    }
    if (/[()[\]{}.,:;]/.test(char)) {
      tokens.push({ kind: 'punctuation', text: char })
      index += 1
      continue
    }
    tokens.push({ kind: 'operator', text: char })
    index += 1
  }
  return tokens
}

function identifierKind(
  input: string,
  index: number,
  text: string,
  previousChar: string,
  priorTokens: CodeToken[]
): TokenKind {
  const lower = text.toLowerCase()
  if (previousChar === '.') {
    return 'property'
  }
  if (CONSTANTS.has(lower)) {
    return 'constant'
  }
  if (KEYWORDS.has(lower)) {
    return 'keyword'
  }
  const previousToken = previousSignificantToken(priorTokens)
  if (previousToken?.kind === 'keyword') {
    const previous = previousToken.text.toLowerCase()
    if (previous === 'def' || previous === 'function') {
      return 'function'
    }
    if (previous === 'class' || previous === 'interface' || previous === 'type') {
      return 'type'
    }
  }
  if (TYPE_IDENTIFIERS.has(text) || /^[A-Z][A-Za-z0-9_]*$/.test(text)) {
    return 'type'
  }
  if (nextNonWhitespace(input, index + text.length) === '(') {
    return BUILTINS.has(lower) ? 'builtin' : 'function'
  }
  if (BUILTINS.has(lower)) {
    return 'builtin'
  }
  return 'plain'
}

function previousSignificantToken(tokens: CodeToken[]): CodeToken | null {
  for (let index = tokens.length - 1; index >= 0; index -= 1) {
    const token = tokens[index]
    if (!token || (token.kind === 'plain' && /^\s+$/.test(token.text))) {
      continue
    }
    return token
  }
  return null
}

function nextNonWhitespace(input: string, after: number): string {
  for (let index = after; index < input.length; index += 1) {
    const char = input[index] ?? ''
    if (!/\s/.test(char)) {
      return char
    }
  }
  return ''
}

function readQuoted(input: string, start: number, quote: string): number {
  const triple = input.slice(start, start + 3) === quote.repeat(3)
  let index = start + (triple ? 3 : 1)
  while (index < input.length) {
    if (!triple && input[index] === '\\') {
      index += 2
      continue
    }
    if (triple && input.slice(index, index + 3) === quote.repeat(3)) {
      return index + 3
    }
    if (!triple && input[index] === quote) {
      return index + 1
    }
    index += 1
  }
  return input.length
}

function previousNonWhitespace(input: string, before: number): string {
  for (let index = before - 1; index >= 0; index -= 1) {
    const char = input[index] ?? ''
    if (!/\s/.test(char)) {
      return char
    }
  }
  return ''
}

function tokenProps(
  kind: TokenKind,
  rowKind: NumberedDiffLine['kind']
): { color?: string; dimColor?: boolean } {
  if (!theme.isColorEnabled()) {
    return rowKind === 'context' ? { dimColor: true } : {}
  }
  if (rowKind === 'context' && kind === 'plain') {
    return { color: '#CBD5E1' }
  }
  switch (kind) {
    case 'keyword':
      return { color: '#F472B6' }
    case 'string':
      return { color: '#FACC15' }
    case 'number':
    case 'constant':
      return { color: '#A78BFA' }
    case 'comment':
      return { color: '#64748B' }
    case 'function':
      return { color: '#A3E635' }
    case 'type':
      return { color: '#86EFAC' }
    case 'builtin':
      return { color: '#22D3EE' }
    case 'property':
      return { color: '#7DD3FC' }
    case 'operator':
    case 'punctuation':
      return { color: '#D1D5DB' }
    default:
      return { color: rowKind === 'deletion' ? '#E5E7EB' : '#D1D5DB' }
  }
}

import { Box, Text } from 'ink'
import type { ReactElement } from 'react'

import {
  DEFAULT_FOLD_THRESHOLD,
  numberedDiffLines,
  parseUnifiedDiff
} from '../lib/diffRender.js'

export interface CodeDiffViewProps {
  diff: string
  expanded?: boolean
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
 * * Coloured by row kind: additions green, deletions red, context dim.
 */
export function CodeDiffView({
  diff,
  expanded = true,
  foldThreshold = DEFAULT_FOLD_THRESHOLD
}: CodeDiffViewProps): ReactElement {
  const parsed = parseUnifiedDiff(diff)
  const rows = numberedDiffLines(parsed)
  const visibleRows = expanded ? rows : rows.slice(0, foldThreshold)
  const hiddenCount = rows.length - visibleRows.length
  const gutterWidth = Math.max(
    3,
    String(rows[rows.length - 1]?.lineNumber ?? 0).length + 1
  )
  return (
    <Box flexDirection="column">
      {visibleRows.map((row, idx) => {
        const colorProps = colorPropsFor(row.kind)
        const marker = markerFor(row.kind)
        const lineLabel =
          row.lineNumber === null ? '' : String(row.lineNumber)
        return (
          <Box key={idx}>
            <Box width={gutterWidth} flexShrink={0}>
              <Text dimColor>{lineLabel.padStart(gutterWidth - 1, ' ') + ' '}</Text>
            </Box>
            <Box width={2} flexShrink={0}>
              <Text {...colorProps}>{marker}</Text>
            </Box>
            <Text {...colorProps}>{row.text || ' '}</Text>
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

function colorPropsFor(kind: 'addition' | 'deletion' | 'context' | 'no-newline'): {
  color?: string
  dimColor?: boolean
} {
  switch (kind) {
    case 'addition':
      return { color: 'green' }
    case 'deletion':
      return { color: 'red' }
    case 'no-newline':
      return { dimColor: true }
    default:
      return { dimColor: true }
  }
}

function markerFor(kind: 'addition' | 'deletion' | 'context' | 'no-newline'): string {
  switch (kind) {
    case 'addition':
      return '+ '
    case 'deletion':
      return '− '
    case 'no-newline':
      return '  '
    default:
      return '  '
  }
}

import { Box, Text } from 'ink'
import type { ReactElement } from 'react'

import {
  DEFAULT_FOLD_THRESHOLD,
  formatStats,
  parseUnifiedDiff,
  visibleLineIndices,
  type DiffLine
} from './diffRender.js'

const COLORS: Record<DiffLine['kind'], string | undefined> = {
  'file-header': 'cyan',
  'hunk-header': 'cyan',
  addition: 'green',
  deletion: 'red',
  context: undefined,
  'no-newline': undefined,
  meta: undefined
}

const DIM: Record<DiffLine['kind'], boolean> = {
  'file-header': true,
  'hunk-header': false,
  addition: false,
  deletion: false,
  context: false,
  'no-newline': true,
  meta: true
}

export interface DiffViewProps {
  diff: string
  expanded?: boolean
  foldThreshold?: number
  // When provided, replaces the default top header. Useful when the parent
  // already shows the file path / risk badge above the diff.
  showHeader?: boolean
}

/**
 * Coloured unified-diff renderer. Folds long diffs to `foldThreshold` payload
 * lines (structural lines like file/hunk headers always render). The parent
 * component owns the expand toggle key — pass `expanded` accordingly.
 */
export function DiffView(props: DiffViewProps): ReactElement {
  const parsed = parseUnifiedDiff(props.diff)
  const expanded = props.expanded ?? false
  const threshold = props.foldThreshold ?? DEFAULT_FOLD_THRESHOLD
  const { indices, hiddenCount } = visibleLineIndices(parsed.lines, {
    expanded,
    foldThreshold: threshold
  })

  return (
    <Box flexDirection="column">
      {props.showHeader !== false ? (
        <Box>
          <Text dimColor>{formatStats(parsed.stats)}</Text>
        </Box>
      ) : null}
      {indices.map((idx) => {
        const line = parsed.lines[idx]
        if (!line) {
          return null
        }
        const color = COLORS[line.kind]
        const dim = DIM[line.kind]
        const props: { color?: string; dimColor?: boolean } = {}
        if (color) {
          props.color = color
        }
        if (dim) {
          props.dimColor = true
        }
        return (
          <Text key={idx} {...props}>
            {line.text || ' '}
          </Text>
        )
      })}
      {hiddenCount > 0 ? (
        <Box marginTop={1}>
          <Text dimColor>
            ({hiddenCount} line{hiddenCount === 1 ? '' : 's'} hidden · press [E] to expand)
          </Text>
        </Box>
      ) : null}
    </Box>
  )
}

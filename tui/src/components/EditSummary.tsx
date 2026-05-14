import { Box, Text } from 'ink'
import { relative } from 'node:path'
import type { ReactElement } from 'react'

import { theme } from '../lib/theme.js'
import type { ToolCallSummary } from '../store/chatStore.js'

import { CodeDiffView } from './CodeDiffView.js'

export interface EditSummaryProps {
  summary: ToolCallSummary
  toolName: string
  expanded: boolean
  focused: boolean
}

/**
 * Claude-Code-style "● Update(path)" chat row that lands after a
 * file_edit / write_file is approved and executed. Mirrors the screenshots
 * in `tmp/code-style.png` and `tmp/code_style.png`:
 *
 *   ● Update(src/store/chatStore.ts)
 *     Added 12 lines, removed 5 lines
 *
 * The optional folded `CodeDiffView` body inlines a line-numbered unified
 * diff (no `@@` hunk headers, `+` / `−` markers, green/red coloring) when
 * the parent toggles expansion. For a no-op write (overwriting a file with
 * identical content), the subline collapses to `(no-op)` and the diff is
 * omitted — there is nothing to render.
 */
export function EditSummary({ summary, toolName, expanded }: EditSummaryProps): ReactElement {
  const brand = theme.colorProps('brand')
  const verb = headerVerb(toolName, summary)
  const path = displayPath(summary.path)
  const subline = summary.noOp ? '(no-op)' : buildCountsLine(summary)
  return (
    <Box flexDirection="column" marginTop={1}>
      <Box>
        <Text bold {...brand}>
          {theme.icon('assistant') || '●'}{' '}
        </Text>
        <Text bold>{verb}</Text>
        <Text>(</Text>
        <Text {...brand}>{path}</Text>
        <Text>)</Text>
      </Box>
      <Box marginLeft={2}>
        <Text dimColor>{subline}</Text>
      </Box>
      {expanded && summary.diff && !summary.noOp ? (
        <Box marginLeft={2} marginTop={1}>
          <CodeDiffView diff={summary.diff} expanded={true} />
        </Box>
      ) : null}
    </Box>
  )
}

/**
 * Choose the action verb shown next to the path. write_file with a
 * non-zero `linesRemoved` count is treated as an overwrite (Update);
 * with zero removed lines it is a fresh create. file_edit is always
 * an Update.
 */
function headerVerb(toolName: string, summary: ToolCallSummary): string {
  if (toolName === 'write_file' && summary.linesRemoved === 0) {
    return 'Create'
  }
  return 'Update'
}

function buildCountsLine(summary: ToolCallSummary): string {
  const parts: string[] = []
  if (summary.linesAdded > 0) {
    parts.push(`Added ${summary.linesAdded} line${summary.linesAdded === 1 ? '' : 's'}`)
  }
  if (summary.linesRemoved > 0) {
    parts.push(
      `removed ${summary.linesRemoved} line${summary.linesRemoved === 1 ? '' : 's'}`
    )
  }
  if (parts.length === 0) {
    parts.push('no changes')
  }
  if (summary.hunks && summary.hunks > 1) {
    parts.push(`across ${summary.hunks} hunks`)
  }
  return parts.join(', ')
}

/**
 * Render an absolute path as a project-relative one when the file is
 * inside the user's cwd — matches Claude Code's `Update(src/foo.ts)`
 * form. Falls back to the original absolute path if the file is outside
 * cwd or the relative-path computation throws (e.g. invalid path).
 */
function displayPath(absolutePath: string): string {
  try {
    const rel = relative(process.cwd(), absolutePath)
    if (rel && !rel.startsWith('..') && !rel.startsWith('/')) {
      return rel
    }
  } catch {
    // fall through to absolute path
  }
  return absolutePath
}

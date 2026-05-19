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
  width?: number
  // When set, the row is being shown alongside an open permission
  // modal. 'pending' forces the diff visible (user can't accidentally
  // collapse what they're about to approve) and tags the subline.
  // 'rejected' keeps the diff visible but marks the row as declined.
  previewStatus?: 'pending' | 'rejected'
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
export function EditSummary({
  summary,
  toolName,
  expanded,
  previewStatus,
  width
}: EditSummaryProps): ReactElement {
  const brand = theme.colorProps('brand')
  const verb = headerVerb(toolName, summary, previewStatus)
  const path = displayPath(summary.path)
  // Force the diff open during the permission prompt — the user is about
  // to approve this change and should always see the diff. After the
  // user decides, `previewStatus` is cleared (approved) or flipped to
  // `'rejected'` (still shown), so we keep showing the diff in both
  // cases. Post-execution rows fall back to the user-controlled
  // `expanded` toggle.
  const showDiff =
    summary.diff !== undefined &&
    !summary.noOp &&
    (previewStatus !== undefined || expanded)

  const subline = buildSubline(summary, previewStatus)
  const sublineProps =
    previewStatus === 'pending'
      ? theme.colorProps('accent')
      : previewStatus === 'rejected'
        ? { color: 'red' as const }
        : { dimColor: true }

  return (
    <Box flexDirection="column" marginTop={1} width="100%">
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
        <Text {...sublineProps}>{subline}</Text>
      </Box>
      {showDiff ? (
        <Box marginTop={1} width={width}>
          <CodeDiffView
            diff={summary.diff ?? ''}
            expanded={true}
            {...(width !== undefined ? { width } : {})}
          />
        </Box>
      ) : null}
    </Box>
  )
}

function buildSubline(
  summary: ToolCallSummary,
  previewStatus: 'pending' | 'rejected' | undefined
): string {
  if (summary.noOp) {
    return '(no-op)'
  }
  const counts = buildCountsLine(summary, previewStatus === 'pending')
  if (previewStatus === 'pending') {
    return `${counts} (pending approval)`
  }
  if (previewStatus === 'rejected') {
    return `${counts} (rejected)`
  }
  return counts
}

/**
 * Choose the action verb shown next to the path. write_file with a
 * non-zero `linesRemoved` count is treated as an overwrite (Update);
 * with zero removed lines it is a fresh create. file_edit is always
 * an Update. While the permission modal is open we switch to the
 * future tense ("Create" → "Create" stays; copy reads naturally with
 * the "(pending approval)" subline) but we leave the verb unchanged
 * to keep the row stable across the pending → applied transition.
 */
function headerVerb(
  toolName: string,
  summary: ToolCallSummary,
  _previewStatus: 'pending' | 'rejected' | undefined
): string {
  if (toolName === 'write_file' && summary.linesRemoved === 0) {
    return 'Create'
  }
  return 'Update'
}

function buildCountsLine(summary: ToolCallSummary, futureTense: boolean): string {
  const addVerb = futureTense ? 'Will add' : 'Added'
  const removeVerb = futureTense ? 'remove' : 'removed'
  const parts: string[] = []
  if (summary.linesAdded > 0) {
    parts.push(`${addVerb} ${summary.linesAdded} line${summary.linesAdded === 1 ? '' : 's'}`)
  }
  if (summary.linesRemoved > 0) {
    parts.push(
      `${removeVerb} ${summary.linesRemoved} line${summary.linesRemoved === 1 ? '' : 's'}`
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

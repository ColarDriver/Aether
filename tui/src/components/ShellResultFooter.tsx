import { Box, Text } from 'ink'
import type { ReactElement } from 'react'

import type { ToolResultMetadata } from '../gatewayTypes.js'
import { theme } from '../lib/theme.js'

export interface ShellResultFooterProps {
  metadata: ToolResultMetadata
}

/**
 * Compact "[exit 0 · 12ms]" / "[exit 1 · 39ms · stderr_lines=2]" footer
 * shown beneath the ToolCallPanel header for shell tool results.
 *
 * Returns null when the metadata lacks both exit_code and duration_ms so
 * we never render an empty bracket.
 */
export function ShellResultFooter({ metadata }: ShellResultFooterProps): ReactElement | null {
  const exitCode = typeof metadata.exit_code === 'number' ? metadata.exit_code : null
  const durationMs = typeof metadata.duration_ms === 'number' ? metadata.duration_ms : null
  if (exitCode === null && durationMs === null) {
    return null
  }
  const stderrLines = typeof metadata.stderr_lines === 'number' ? metadata.stderr_lines : 0
  const parts: string[] = []
  if (exitCode !== null) {
    parts.push(`exit ${exitCode}`)
  }
  if (durationMs !== null) {
    parts.push(formatDuration(durationMs))
  }
  if (stderrLines > 0) {
    parts.push(`stderr_lines=${stderrLines}`)
  }
  if (metadata.truncated === true) {
    parts.push('truncated')
  }
  if (metadata.timed_out === true) {
    parts.push('timed out')
  }
  if (metadata.interrupted === true) {
    parts.push('interrupted')
  }
  const isFailure = exitCode !== null && exitCode !== 0
  const colorKey: 'error' | 'dim' = isFailure ? 'error' : 'dim'
  return (
    <Box marginLeft={2}>
      <Text {...theme.colorProps(colorKey)}>[{parts.join(' · ')}]</Text>
    </Box>
  )
}

function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${ms}ms`
  }
  const seconds = ms / 1000
  if (seconds < 60) {
    return `${seconds.toFixed(seconds < 10 ? 2 : 1)}s`
  }
  const minutes = Math.floor(seconds / 60)
  const remaining = Math.round(seconds % 60)
  return `${minutes}m${remaining.toString().padStart(2, '0')}s`
}

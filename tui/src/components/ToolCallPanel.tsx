import { Box, Text } from 'ink'
import type { ReactElement } from 'react'

import { categoryFor, hintForCall } from '../lib/toolCategory.js'
import type { ChatItem } from '../store/chatStore.js'

import { ShellResultFooter } from './ShellResultFooter.js'

const COLLAPSED_PREVIEW_LIMIT = 100
const EXPANDED_RESULT_LIMIT = 1600
const EXPANDED_ARGS_LIMIT = 800
// Inline preview (collapsed state) for non-explore tools — bash output / web
// fetch body / etc. We show the head of the result so the user sees what came
// back without having to Tab+Enter to expand.
const INLINE_PREVIEW_LINES = 8
const INLINE_PREVIEW_LINE_WIDTH = 120

export interface ToolCallPanelProps {
  item: Extract<ChatItem, { kind: 'tool-call' }>
  expanded: boolean
  focused: boolean
}

export function ToolCallPanel({ item, expanded, focused }: ToolCallPanelProps): ReactElement {
  const hint = hintForCall(item.toolName, item.args) || item.argsPreview
  const collapsedHint = truncate(hint, COLLAPSED_PREVIEW_LIMIT)
  const isError = item.result?.isError ?? false
  const headerColor = isError ? 'red' : focused ? 'cyan' : 'yellow'
  const isShellCategory = categoryFor(item.toolName) === 'bash'
  const shellMetadata =
    isShellCategory && item.result?.metadata ? item.result.metadata : null

  if (!expanded) {
    return (
      <Box flexDirection="column" marginTop={item.iteration > 0 ? 0 : 1}>
        <Box>
          <Text color={headerColor}>▸ {item.toolName}</Text>
          {collapsedHint ? (
            <>
              <Text color="gray"> · </Text>
              <Text>{collapsedHint}</Text>
            </>
          ) : null}
          {item.durationMs !== null ? (
            <Text dimColor> · ⏱ {formatDuration(item.durationMs)}</Text>
          ) : (
            <Text dimColor> · running…</Text>
          )}
          {isError ? <Text color="red"> · error</Text> : null}
        </Box>
        {shellMetadata ? <ShellResultFooter metadata={shellMetadata} /> : null}
        {item.result ? <InlineResultPreview text={item.result.text} isError={isError} /> : null}
      </Box>
    )
  }

  return (
    <Box flexDirection="column" marginTop={1}>
      <Box>
        <Text color={headerColor}>▾ {item.toolName}</Text>
        {item.durationMs !== null ? (
          <Text dimColor> · ⏱ {formatDuration(item.durationMs)}</Text>
        ) : (
          <Text dimColor> · running…</Text>
        )}
        {isError ? <Text color="red"> · error</Text> : null}
      </Box>
      {shellMetadata ? <ShellResultFooter metadata={shellMetadata} /> : null}
      <Box marginLeft={2} flexDirection="column">
        <Text dimColor>arguments:</Text>
        <Text>{truncate(jsonPretty(item.args), EXPANDED_ARGS_LIMIT)}</Text>
        {item.result ? (
          <>
            <Text dimColor>result:</Text>
            {item.result.isError ? (
              <Text color="red">{truncate(item.result.text, EXPANDED_RESULT_LIMIT)}</Text>
            ) : (
              <Text>{truncate(item.result.text, EXPANDED_RESULT_LIMIT)}</Text>
            )}
          </>
        ) : (
          <Text dimColor>(awaiting result)</Text>
        )}
      </Box>
    </Box>
  )
}

/**
 * Multi-line indented preview shown beneath the collapsed header. Mirrors
 * Python's `render_tool_result` which always shows the first chunk of bash /
 * web output so the user sees what happened without having to expand.
 */
function InlineResultPreview({
  text,
  isError
}: {
  text: string
  isError: boolean
}): ReactElement | null {
  const lines = (text || '').split('\n').slice(0, INLINE_PREVIEW_LINES)
  if (lines.length === 0 || (lines.length === 1 && !lines[0])) {
    return null
  }
  const totalLines = (text || '').split('\n').length
  const truncated = totalLines > INLINE_PREVIEW_LINES
  const colorProps = isError ? { color: 'red' } : { dimColor: true }
  return (
    <Box flexDirection="column" marginLeft={2}>
      {lines.map((line, idx) => (
        <Text key={idx} {...colorProps}>
          {idx === 0 ? '⎿ ' : '  '}
          {truncate(line, INLINE_PREVIEW_LINE_WIDTH) || ' '}
        </Text>
      ))}
      {truncated ? (
        <Text dimColor>
          {'  '}… ({totalLines - INLINE_PREVIEW_LINES} more lines · Tab+Enter to expand)
        </Text>
      ) : null}
    </Box>
  )
}

function jsonPretty(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2) ?? ''
  } catch {
    return String(value)
  }
}

function truncate(value: string, limit: number): string {
  if (value.length <= limit) {
    return value
  }
  return `${value.slice(0, limit - 1)}…`
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

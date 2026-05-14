import { Box, Text } from 'ink'
import type { ReactElement } from 'react'

import { Markdown } from '../lib/markdown.js'
import { stripToolBlocks } from '../lib/phantomTool.js'
import { theme } from '../lib/theme.js'
import { isTurnFooterText } from '../lib/turnFooter.js'
import type { ChatItem } from '../store/chatStore.js'

import { ExploredTree } from './ExploredTree.js'
import { ToolCallPanel } from './ToolCallPanel.js'

export interface ChatMessageProps {
  item: ChatItem
  expanded: boolean
  focused: boolean
}

export function ChatMessage({ item, expanded, focused }: ChatMessageProps): ReactElement {
  if (item.kind === 'user') {
    // Mirrors Python `ui.py:render_input_echo` — `›` bullet in brand colour
    // followed by the message text on the same line. Multi-line messages
    // get a hanging indent so the bullet stays anchored.
    const userProps = theme.colorProps('brand')
    const lines = item.text.split('\n')
    const head = lines[0] ?? ''
    const rest = lines.slice(1)
    return (
      <Box flexDirection="column">
        <Text>
          <Text bold {...userProps}>
            {theme.icon('user')}{' '}
          </Text>
          {head}
        </Text>
        {rest.map((line, idx) => (
          <Text key={idx}>{'  ' + line}</Text>
        ))}
      </Box>
    )
  }

  if (item.kind === 'assistant') {
    // Mirrors Python `ui.py:_render_assistant` — `●` bullet in a fixed
    // left column followed by markdown body. Inline tool markup is stripped
    // before render so a half-streamed `<tool_call>` never leaks into prose.
    const cleaned = stripToolBlocks(item.text).trim()
    if (!cleaned) {
      return <></>
    }
    const assistantProps = theme.colorProps('assistant')
    return (
      <Box>
        <Box width={2} flexShrink={0}>
          <Text bold {...assistantProps}>
            {theme.icon('assistant') || '*'}
          </Text>
        </Box>
        <Box flexDirection="column" flexGrow={1}>
          <Markdown text={cleaned} streaming={item.streaming} />
        </Box>
      </Box>
    )
  }

  if (item.kind === 'tool-call') {
    // Calls inside an explore burst (read/search/list/write/edit) are
    // collapsed into the ExploredTree summary; we hide them in the
    // transcript so we do not double-render. Standalone calls (bash/web/etc.)
    // render as a per-call ToolCallPanel.
    if (item.coalesce) {
      return <></>
    }
    return <ToolCallPanel item={item} expanded={expanded} focused={focused} />
  }

  if (item.kind === 'tool-group') {
    return <ExploredTree group={item.group} />
  }

  if (item.kind === 'tool-result') {
    // Standalone tool results that did not get attached to their call (rare
    // race, e.g. result before call event) — render as a thin one-liner.
    return (
      <Text color={item.isError ? 'red' : 'gray'}>
        tool result {item.toolCallId}: {item.text}
      </Text>
    )
  }

  // Mirrors Python `ui.py:1958-1980` — each level prepends a glyph in its
  // own colour. The `●`-prefixed per-turn footers ("● done · …") supply
  // their own bullet so we never double-icon them.
  return renderNote(item)
}

function renderNote(
  item: Extract<ChatItem, { kind: 'note' }>
): ReactElement {
  const startsWithBullet =
    item.text.startsWith('●') ||
    item.text.startsWith('*') ||
    isTurnFooterText(item.text)
  const config = NOTE_LEVEL_CONFIG[item.level]
  const iconProps = theme.colorProps(config.iconColor)
  const textProps = theme.colorProps(config.textColor)
  if (startsWithBullet) {
    // Footers already carry their bullet — render the whole line in the
    // text-level colour so `● failed · 1 error · 33.5s` reads as red, etc.
    return <Text {...textProps}>{item.text}</Text>
  }
  return (
    <Text>
      <Text bold {...iconProps}>
        {theme.icon(config.icon)}{' '}
      </Text>
      <Text {...textProps}>{item.text}</Text>
    </Text>
  )
}

const NOTE_LEVEL_CONFIG: Record<
  'success' | 'info' | 'warn' | 'error',
  { icon: string; iconColor: 'success' | 'accent' | 'warning' | 'error'; textColor: 'success' | 'dim' | 'warning' | 'error' }
> = {
  // Python `success()` — green ✓ + green text.
  success: { icon: 'success', iconColor: 'success', textColor: 'success' },
  // Python `info()` — accent-coloured ℹ + dim body text.
  info: { icon: 'info', iconColor: 'accent', textColor: 'dim' },
  // Python `warn()` — yellow ⚠ + yellow body text.
  warn: { icon: 'warn', iconColor: 'warning', textColor: 'warning' },
  // Python `error()` — red ✗ + red body text.
  error: { icon: 'error', iconColor: 'error', textColor: 'error' }
}

import { Box, Text } from 'ink'
import type { ReactElement } from 'react'

import { Markdown } from '../lib/markdown.js'
import { stripToolBlocks } from '../lib/phantomTool.js'
import { theme } from '../lib/theme.js'
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
    const userProps = theme.colorProps('user')
    const lines = item.text.split('\n')
    const head = lines[0] ?? ''
    const rest = lines.slice(1)
    return (
      <Box flexDirection="column" marginTop={1}>
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
    // left column followed by markdown body. The streaming hint sits on
    // the same row as the bullet so the layout stays stable as deltas
    // arrive (no header line jumping in and out).
    const cleaned = stripToolBlocks(item.text)
    const assistantProps = theme.colorProps('assistant')
    const dim = theme.colorProps('dim')
    return (
      <Box marginTop={1}>
        <Box width={2} flexShrink={0}>
          <Text bold {...assistantProps}>
            {theme.icon('assistant') || '*'}
          </Text>
        </Box>
        <Box flexDirection="column" flexGrow={1}>
          {item.streaming ? (
            <Text {...dim}>streaming…</Text>
          ) : null}
          <Markdown text={cleaned} />
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

  const noteColorName =
    item.level === 'error' ? 'error' : item.level === 'warn' ? 'warning' : 'dim'
  const noteProps = theme.colorProps(noteColorName)
  return <Text {...noteProps}>{item.text}</Text>
}

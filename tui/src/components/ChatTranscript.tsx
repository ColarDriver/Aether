import { Box, Text, useInput } from 'ink'
import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState, type ReactElement } from 'react'

import { chatItems, type ChatItem } from '../store/chatStore.js'
import { focusActions, focusOwner } from '../store/focusStore.js'
import { overlayStack } from '../store/overlayStore.js'

import { ChatMessage } from './ChatMessage.js'

const MAX_VISIBLE = 80

/**
 * Tab/Shift+Tab cycle the focus between standalone tool-call items in the
 * transcript; Enter on a focused row toggles its expanded state.
 *
 * Focus arbitration is via `focusOwner`: the composer hands focus over when
 * the user presses Tab; this hook hands focus back to the composer when the
 * user Tabs past the last row (or has nothing focusable to begin with).
 */
export function ChatTranscript(): ReactElement {
  const items = useStore(chatItems)
  const overlays = useStore(overlayStack)
  const owner = useStore(focusOwner)
  const [focusedToolCallId, setFocusedToolCallId] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set())

  const visible = useMemo(() => items.slice(-MAX_VISIBLE), [items])

  const focusableTools = useMemo(
    () =>
      visible.filter(
        (item): item is Extract<ChatItem, { kind: 'tool-call' }> =>
          item.kind === 'tool-call' && !item.coalesce
      ),
    [visible]
  )

  // Keep focus pointing at a still-visible tool when the transcript shifts
  // (older items fall off the bottom-N window).
  useEffect(() => {
    if (
      focusedToolCallId &&
      !focusableTools.find((tool) => tool.toolCallId === focusedToolCallId)
    ) {
      const last = focusableTools[focusableTools.length - 1] ?? null
      setFocusedToolCallId(last?.toolCallId ?? null)
    }
  }, [focusableTools, focusedToolCallId])

  // If the composer-side handed focus to us but we have nothing focusable,
  // hand it back so the user keeps typing without a dead Tab.
  useEffect(() => {
    if (owner === 'transcript' && focusableTools.length === 0) {
      focusActions.set('composer')
    }
  }, [owner, focusableTools])

  useInput(
    (input, key) => {
      if (key.tab && !key.ctrl && !key.meta) {
        if (focusableTools.length === 0) {
          focusActions.set('composer')
          return
        }
        const direction = key.shift ? -1 : 1
        const index = focusedToolCallId
          ? focusableTools.findIndex((tool) => tool.toolCallId === focusedToolCallId)
          : -1
        // Cycle through tools; an extra step at each end hands focus back to
        // the composer so Tab on the last row escapes the focus loop cleanly.
        const total = focusableTools.length + 1 // +1 for the composer slot
        const startIndex = index === -1 ? (direction === 1 ? -1 : focusableTools.length) : index
        const nextIndex = (startIndex + direction + total) % total
        if (nextIndex === focusableTools.length) {
          setFocusedToolCallId(null)
          focusActions.set('composer')
          return
        }
        const nextTool = focusableTools[nextIndex]
        if (nextTool) {
          setFocusedToolCallId(nextTool.toolCallId)
        }
        return
      }
      if (key.escape) {
        setFocusedToolCallId(null)
        focusActions.set('composer')
        return
      }
      if (key.return && focusedToolCallId) {
        setExpanded((current) => {
          const next = new Set(current)
          if (next.has(focusedToolCallId)) {
            next.delete(focusedToolCallId)
          } else {
            next.add(focusedToolCallId)
          }
          return next
        })
        return
      }
    },
    { isActive: overlays.length === 0 && owner === 'transcript' }
  )

  if (visible.length === 0) {
    return (
      <Box marginTop={1}>
        <Text color="gray">Type a message or /help for commands.</Text>
      </Box>
    )
  }

  // No outer marginY — Python's prompt_toolkit transcript runs flush against
  // the banner above and the activity bar below; the per-item ChatMessage
  // marginTop already supplies visual breathing between turns.
  return (
    <Box flexDirection="column">
      {visible.map((item) => {
        const isToolCall = item.kind === 'tool-call'
        const focused = isToolCall && item.toolCallId === focusedToolCallId
        const isExpanded = isToolCall && expanded.has(item.toolCallId)
        return <ChatMessage key={item.id} item={item} focused={focused} expanded={isExpanded} />
      })}
    </Box>
  )
}

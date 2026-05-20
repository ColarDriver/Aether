import { Box, Text } from 'ink'
import { useStore } from '@nanostores/react'
import type { ReactElement } from 'react'

import { theme } from '../lib/theme.js'
import { useAnimationFrame } from '../lib/useAnimationFrame.js'
import { activityStatus } from '../store/activityStore.js'
import { reasoningState } from '../store/reasoningStore.js'

const FADE_AFTER_MS = 8000
const TICK_MS = 1000

/**
 * Italic gray excerpt of the latest reasoning delta. Hidden after
 * FADE_AFTER_MS of inactivity so it does not linger after a turn ends.
 *
 * The bar is intentionally one-line tall — long reasoning is truncated by
 * `reasoningStore.appendDelta` so the surrounding chrome never shifts.
 */
export function ReasoningLine(): ReactElement | null {
  const reasoning = useStore(reasoningState)
  const status = useStore(activityStatus)
  const [tickRef, now] = useAnimationFrame(reasoning.updatedAt ? TICK_MS : null)

  if (!reasoning.text || !reasoning.updatedAt) {
    return null
  }
  const stale = now - reasoning.updatedAt > FADE_AFTER_MS
  if (stale && status === 'idle') {
    return null
  }
  const flat = reasoning.text.replace(/[\r\n]+/g, ' ').trim()
  if (!flat) {
    return null
  }

  return (
    <Box ref={tickRef}>
      <Text dimColor>
        {'  '}
        {theme.icon('thinking') || '·'} thinking: {flat}
      </Text>
    </Box>
  )
}

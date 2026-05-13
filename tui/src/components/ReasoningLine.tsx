import { Box, Text } from 'ink'
import { useStore } from '@nanostores/react'
import { useEffect, useState, type ReactElement } from 'react'

import { activityState } from '../store/activityStore.js'
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
  const activity = useStore(activityState)
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    if (!reasoning.updatedAt) {
      return
    }
    const handle = setInterval(() => setNow(Date.now()), TICK_MS)
    return () => clearInterval(handle)
  }, [reasoning.updatedAt])

  if (!reasoning.text || !reasoning.updatedAt) {
    return null
  }
  const stale = now - reasoning.updatedAt > FADE_AFTER_MS
  if (stale && activity.status === 'idle') {
    return null
  }

  return (
    <Box marginTop={1}>
      <Text dimColor italic>
        {reasoning.text}
      </Text>
    </Box>
  )
}

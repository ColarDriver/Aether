import { useEffect, useState } from 'react'
import type { DOMElement } from 'ink'

import { nowMs, subscribe } from './animationClock.js'
import { useTerminalViewport } from './useTerminalViewport.js'

export function useAnimationFrame(
  intervalMs: number | null
): [ref: (element: DOMElement | null) => void, time: number] {
  const [viewportRef, viewport] = useTerminalViewport()
  const [time, setTime] = useState(() => nowMs())
  const active = intervalMs !== null && viewport.isVisible

  useEffect(() => {
    if (!active) {
      return
    }

    let lastFire = nowMs()
    return subscribe(() => {
      const current = nowMs()
      if (current - lastFire >= intervalMs) {
        lastFire = current
        setTime(current)
      }
    })
  }, [active, intervalMs])

  return [viewportRef, time]
}

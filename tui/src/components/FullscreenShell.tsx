import { Box, useStdout } from 'ink'
import { useInsertionEffect, type ReactElement, type ReactNode } from 'react'

const ENTER_ALT_SCREEN = '\x1b[?1049h'
const EXIT_ALT_SCREEN = '\x1b[?1049l'
const ENABLE_MOUSE_TRACKING = '\x1b[?1000h\x1b[?1002h\x1b[?1006h'
const DISABLE_MOUSE_TRACKING = '\x1b[?1000l\x1b[?1002l\x1b[?1006l'

export function FullscreenShell({
  rows,
  width,
  children
}: {
  rows: number
  width: number
  children: ReactNode
}): ReactElement {
  const { stdout } = useStdout()

  useInsertionEffect(() => {
    const stream = stdout as NodeJS.WriteStream | undefined
    if (!stream?.isTTY || process.env.AETHER_DISABLE_ALT_SCREEN === '1') {
      return
    }
    const mouseTracking = process.env.AETHER_DISABLE_MOUSE === '1' ? '' : ENABLE_MOUSE_TRACKING
    stream.write(`${ENTER_ALT_SCREEN}\x1b[2J\x1b[H${mouseTracking}`)
    return () => {
      stream.write(`${mouseTracking ? DISABLE_MOUSE_TRACKING : ''}${EXIT_ALT_SCREEN}`)
    }
  }, [stdout])

  return (
    <Box flexDirection="column" height={Math.max(1, rows)} width={Math.max(20, width)} flexShrink={0}>
      {children}
    </Box>
  )
}

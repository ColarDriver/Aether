import { Box, Text } from 'ink'
import { render } from 'ink-testing-library'
import { act } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { _resetClockForTests } from '../lib/animationClock.js'
import { useAnimationFrame } from '../lib/useAnimationFrame.js'

const viewport = vi.hoisted(() => ({
  isVisible: true,
  ref: vi.fn()
}))

vi.mock('../lib/useTerminalViewport.js', () => ({
  useTerminalViewport: () => [
    viewport.ref,
    { isVisible: viewport.isVisible }
  ]
}))

function Probe({ intervalMs }: { intervalMs: number | null }) {
  const [ref, time] = useAnimationFrame(intervalMs)
  return (
    <Box ref={ref}>
      <Text>{String(time)}</Text>
    </Box>
  )
}

describe('useAnimationFrame', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    viewport.isVisible = true
    viewport.ref.mockClear()
    _resetClockForTests()
  })

  afterEach(() => {
    _resetClockForTests()
    vi.useRealTimers()
  })

  it('ticks on the shared clock when visible', async () => {
    const { lastFrame, unmount } = render(<Probe intervalMs={100} />)
    expect(lastFrame()).toBe('0')

    await act(async () => {
      await vi.advanceTimersByTimeAsync(100)
    })

    expect(lastFrame()).toBe('100')
    unmount()
  })

  it('does not subscribe when interval is null', async () => {
    const { lastFrame, unmount } = render(<Probe intervalMs={null} />)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(200)
    })

    expect(lastFrame()).toBe('0')
    unmount()
  })

  it('pauses when the element is outside the terminal viewport', async () => {
    viewport.isVisible = false
    const { lastFrame, unmount } = render(<Probe intervalMs={100} />)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(200)
    })

    expect(lastFrame()).toBe('0')
    unmount()
  })
})

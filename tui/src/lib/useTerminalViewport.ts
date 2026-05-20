import { useCallback, useLayoutEffect, useRef } from 'react'
import { useStdout, type DOMElement } from 'ink'

interface ViewportEntry {
  isVisible: boolean
}

export function useTerminalViewport(): [
  ref: (element: DOMElement | null) => void,
  entry: ViewportEntry
] {
  const { stdout } = useStdout()
  const elementRef = useRef<DOMElement | null>(null)
  const entryRef = useRef<ViewportEntry>({ isVisible: true })

  const setElement = useCallback((element: DOMElement | null) => {
    elementRef.current = element
  }, [])

  useLayoutEffect(() => {
    const element = elementRef.current
    if (!element?.yogaNode) {
      return
    }

    const rows = readRows(stdout as NodeJS.WriteStream | undefined)
    const height = element.yogaNode.getComputedHeight()
    let absoluteTop = element.yogaNode.getComputedTop()
    let root = element.yogaNode
    let parent = element.parentNode

    while (parent) {
      if (parent.yogaNode) {
        absoluteTop += parent.yogaNode.getComputedTop()
        root = parent.yogaNode
      }
      const scrollTop = (parent as { scrollTop?: number }).scrollTop
      if (scrollTop) {
        absoluteTop -= scrollTop
      }
      parent = parent.parentNode
    }

    const screenHeight = root.getComputedHeight()
    const bottom = absoluteTop + height
    const cursorRestoreScroll = screenHeight > rows ? 1 : 0
    const viewportY = Math.max(0, screenHeight - rows) + cursorRestoreScroll
    const viewportBottom = viewportY + rows
    const visible = bottom > viewportY && absoluteTop < viewportBottom

    if (visible !== entryRef.current.isVisible) {
      entryRef.current = { isVisible: visible }
    }
  })

  return [setElement, entryRef.current]
}

function readRows(stdout: NodeJS.WriteStream | undefined): number {
  const rows = stdout?.rows
  return typeof rows === 'number' && rows > 0 ? rows : 24
}

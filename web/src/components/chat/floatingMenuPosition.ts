import type { CSSProperties } from 'react'

export function floatingMenuPosition(anchor: HTMLElement, preferredWidth: number, gap = 8, viewportPadding = 12): CSSProperties {
  const viewportWidth = window.innerWidth || preferredWidth + viewportPadding * 2
  const viewportHeight = window.innerHeight || 800
  const rect = anchor.getBoundingClientRect()
  const width = Math.min(preferredWidth, Math.max(180, viewportWidth - viewportPadding * 2))
  const maxLeft = Math.max(viewportPadding, viewportWidth - width - viewportPadding)
  const left = Math.min(Math.max(viewportPadding, rect.left), maxLeft)
  const bottom = Math.max(viewportPadding, viewportHeight - rect.top + gap)
  return { left, bottom, width }
}
